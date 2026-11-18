"""SQLite state of the web app: scans, queries and the user's verifications.

Embeddings and JPEGs are not stored here; they live in the numpy caches and the image cache.
Call `init` once with the database path before using the other functions.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)

FETCH_MODES = ("api", "scraper", "corpus")
"""How a scan's images were collected: Street View Static API, scraper, or exported corpus."""

SCAN_ACTIVE = ("fetching", "encoding", "tiling", "patching")
QUERY_ACTIVE = ("expanding", "stage1", "reranking")

SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    bbox_n REAL, bbox_s REAL, bbox_e REAL, bbox_w REAL,
    spacing_m INTEGER NOT NULL DEFAULT 50,
    embedding_tag TEXT NOT NULL,
    tile_tag TEXT NOT NULL,
    patch_region_tag TEXT,           -- set: rank with image + tile + patch-region scores
    region_filter TEXT,              -- only rank images of this region
    fetch_mode TEXT NOT NULL DEFAULT 'api',
    status TEXT NOT NULL,            -- pending|fetching|encoding|tiling|patching|ready|failed
    status_message TEXT,
    n_target INTEGER,
    n_fetched INTEGER NOT NULL DEFAULT 0,
    n_encoded INTEGER NOT NULL DEFAULT 0,
    n_tiled INTEGER NOT NULL DEFAULT 0,
    n_patched INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    ready_at TEXT,
    error TEXT
);

CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY,
    scan_id INTEGER NOT NULL REFERENCES scans(id),
    text TEXT NOT NULL,
    k INTEGER NOT NULL DEFAULT 50,
    vlm_model TEXT NOT NULL,
    min_date TEXT,
    status TEXT NOT NULL,            -- pending|expanding|stage1|reranking|ready|failed
    expansion_json TEXT,
    results_json TEXT,
    parent_query_id INTEGER REFERENCES queries(id),  -- set: relevance-feedback round on the parent
    created_at TEXT NOT NULL,
    ready_at TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS ix_queries_scan ON queries(scan_id);
CREATE INDEX IF NOT EXISTS ix_queries_text ON queries(text);

CREATE TABLE IF NOT EXISTS verifications (
    query_id INTEGER NOT NULL REFERENCES queries(id),
    image_id TEXT NOT NULL,
    verdict TEXT NOT NULL,           -- yes|no|unsure
    created_at TEXT NOT NULL,
    PRIMARY KEY (query_id, image_id)
);
"""

_db_path: Path | None = None


def now_iso() -> str:
    """Current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def init(path: Path) -> None:
    """Use the database at ``path``, creating it and its tables if needed."""
    global _db_path
    _db_path = path
    path.parent.mkdir(parents=True, exist_ok=True)
    with cursor() as cur:
        cur.executescript(SCHEMA)
    LOG.info("web database at %s", path)


def connect() -> sqlite3.Connection:
    """Open a connection in autocommit mode with dict-like rows.

    Raises:
        RuntimeError: If `init` was not called.
    """
    if _db_path is None:
        raise RuntimeError("call vectorview.web.db.init(path) first")
    conn = sqlite3.connect(_db_path, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:  # e.g. on network file systems; the default mode still works
        pass
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def cursor() -> Iterator[sqlite3.Cursor]:
    """Yield a cursor on a fresh connection, closed afterwards."""
    conn = connect()
    try:
        yield conn.cursor()
    finally:
        conn.close()


def _one(sql: str, args: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    with cursor() as cur:
        row = cur.execute(sql, args).fetchone()
    return dict(row) if row else None


def _update(table: str, row_id: int, values: dict[str, Any]) -> None:
    if not values:
        return
    sets = ", ".join(f"{col} = ?" for col in values)
    with cursor() as cur:
        cur.execute(f"UPDATE {table} SET {sets} WHERE id = ?", [*values.values(), row_id])


# ---------------- scans ----------------


def list_scans() -> list[dict[str, Any]]:
    """All scans, exported corpora first, then newest first."""
    with cursor() as cur:
        rows = cur.execute("SELECT * FROM scans ORDER BY fetch_mode = 'corpus' DESC, id DESC").fetchall()
    return [dict(r) for r in rows]


def get_scan(scan_id: int) -> dict[str, Any] | None:
    """The scan with this id, or ``None``."""
    return _one("SELECT * FROM scans WHERE id = ?", (scan_id,))


def create_scan(
    *,
    name: str,
    bbox: tuple[float, float, float, float] | None,
    embedding_tag: str,
    tile_tag: str,
    status: str = "pending",
    spacing_m: int = 50,
    fetch_mode: str = "api",
    region_filter: str | None = None,
    patch_region_tag: str | None = None,
) -> int:
    """Insert a scan and return its id.

    Args:
        name: Display name.
        bbox: ``(north, south, east, west)``, or ``None`` for an exported corpus.
        embedding_tag: Image cache to search.
        tile_tag: Tile cache to search.
        status: Initial status; ``"pending"`` scans are picked up by the worker.
        spacing_m: Distance between road candidates (API mode).
        fetch_mode: One of `FETCH_MODES`.
        region_filter: Only search images of this region.
        patch_region_tag: Patch-region cache, if any.

    Raises:
        ValueError: If ``fetch_mode`` is unknown.
    """
    if fetch_mode not in FETCH_MODES:
        raise ValueError(f"unknown fetch_mode {fetch_mode!r}")
    n, s, e, w = bbox if bbox else (None, None, None, None)
    created = now_iso()
    with cursor() as cur:
        cur.execute(
            """
            INSERT INTO scans (name, bbox_n, bbox_s, bbox_e, bbox_w, spacing_m, embedding_tag, tile_tag,
                               patch_region_tag, region_filter, fetch_mode, status, created_at, ready_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                n,
                s,
                e,
                w,
                spacing_m,
                embedding_tag,
                tile_tag,
                patch_region_tag,
                region_filter,
                fetch_mode,
                status,
                created,
                created if status == "ready" else None,
            ),
        )
        return int(cur.lastrowid or 0)


def update_scan(scan_id: int, *, ready: bool = False, **values: Any) -> None:
    """Set columns of a scan (``None`` values are skipped); ``ready=True`` also stamps ``ready_at``."""
    changes = {k: v for k, v in values.items() if v is not None}
    if ready:
        changes["ready_at"] = now_iso()
    _update("scans", scan_id, changes)


def next_pending_scan() -> dict[str, Any] | None:
    """The oldest pending scan, or ``None``."""
    return _one("SELECT * FROM scans WHERE status = 'pending' ORDER BY id LIMIT 1")


# ---------------- queries ----------------


def create_query(
    *, scan_id: int, text: str, k: int, vlm_model: str, min_date: str | None, parent_query_id: int | None = None
) -> int:
    """Insert a pending query and return its id."""
    with cursor() as cur:
        cur.execute(
            """
            INSERT INTO queries (scan_id, text, k, vlm_model, min_date, status, parent_query_id, created_at)
            VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)
            """,
            (scan_id, text, k, vlm_model, min_date, parent_query_id, now_iso()),
        )
        return int(cur.lastrowid or 0)


def get_query(query_id: int) -> dict[str, Any] | None:
    """The query with this id, or ``None``."""
    return _one("SELECT * FROM queries WHERE id = ?", (query_id,))


def list_queries(scan_id: int, limit: int = 20) -> list[dict[str, Any]]:
    """The latest queries of a scan."""
    with cursor() as cur:
        rows = cur.execute("SELECT * FROM queries WHERE scan_id = ? ORDER BY id DESC LIMIT ?", (scan_id, limit))
        return [dict(r) for r in rows.fetchall()]


def find_cached_expansion(text: str) -> dict[str, Any] | None:
    """The latest stored LLM expansion of exactly this query text, if any."""
    row = _one(
        "SELECT expansion_json FROM queries WHERE text = ? AND expansion_json IS NOT NULL ORDER BY id DESC LIMIT 1",
        (text,),
    )
    return json.loads(row["expansion_json"]) if row else None


def update_query(
    query_id: int,
    *,
    status: str | None = None,
    expansion: dict[str, Any] | None = None,
    results: list[dict[str, Any]] | None = None,
    error: str | None = None,
    ready: bool = False,
) -> None:
    """Update a query; ``expansion`` and ``results`` are stored as JSON."""
    changes: dict[str, Any] = {}
    if status is not None:
        changes["status"] = status
    if expansion is not None:
        changes["expansion_json"] = json.dumps(expansion)
    if results is not None:
        changes["results_json"] = json.dumps(results)
    if error is not None:
        changes["error"] = error
    if ready:
        changes["ready_at"] = now_iso()
    _update("queries", query_id, changes)


def next_pending_query() -> dict[str, Any] | None:
    """The oldest pending query, or ``None``."""
    return _one("SELECT * FROM queries WHERE status = 'pending' ORDER BY id LIMIT 1")


def reset_orphaned_jobs() -> tuple[int, int]:
    """Put queries and scans left mid-run by a stopped server back to ``pending``.

    Returns:
        ``(queries_reset, scans_reset)``.
    """
    counts = []
    with cursor() as cur:
        for table, active in (("queries", QUERY_ACTIVE), ("scans", SCAN_ACTIVE)):
            marks = ",".join("?" * len(active))
            cur.execute(f"UPDATE {table} SET status = 'pending', error = NULL WHERE status IN ({marks})", active)
            counts.append(cur.rowcount)
    return counts[0], counts[1]


# ---------------- verifications ----------------


def record_verification(query_id: int, image_id: str, verdict: str) -> None:
    """Store (or replace) the user's verdict on one result."""
    with cursor() as cur:
        cur.execute(
            """
            INSERT INTO verifications (query_id, image_id, verdict, created_at) VALUES (?, ?, ?, ?)
            ON CONFLICT(query_id, image_id) DO UPDATE SET verdict = excluded.verdict, created_at = excluded.created_at
            """,
            (query_id, image_id, verdict, now_iso()),
        )


def list_verifications(query_id: int) -> dict[str, str]:
    """``{image_id: verdict}`` for a query."""
    with cursor() as cur:
        rows = cur.execute("SELECT image_id, verdict FROM verifications WHERE query_id = ?", (query_id,)).fetchall()
    return {r["image_id"]: r["verdict"] for r in rows}


def verification_stats(query_id: int) -> dict[str, int]:
    """Number of ``yes``, ``no`` and ``unsure`` verdicts for a query."""
    out = {"yes": 0, "no": 0, "unsure": 0}
    for verdict in list_verifications(query_id).values():
        out[verdict] = out.get(verdict, 0) + 1
    return out
