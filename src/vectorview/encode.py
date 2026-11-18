"""Embed the corpus in Postgres with the SigLIP-2 encoder.

- `encode_images` fills ``images.embedding`` for images that have none.
- `encode_tiles` cuts each image into grid tiles and inserts their embeddings into ``tiles``.

Both are resumable: a rerun only processes rows that still need work.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from typing import Any

from PIL import Image
from tqdm import tqdm

from . import db
from .config import Config
from .encoder import Encoder, get_encoder
from .tile import Tile, cut_tiles

LOG = logging.getLogger(__name__)


def load_image(path: str) -> Image.Image | None:
    """Open an image as RGB, or return ``None`` (with a warning) if it cannot be read."""
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception as exc:  # unreadable files are skipped, not fatal
        LOG.warning("skipping %s: %s", path, exc)
        return None


def _region_clause(region: str | None, column: str) -> tuple[str, list[Any]]:
    return (f" AND {column} = %s", [region]) if region else ("", [])


def _pending_images(conn: db.Connection, region: str | None) -> list[tuple[str, str]]:
    clause, params = _region_clause(region, "region")
    with conn.cursor() as cur:
        cur.execute(
            "SELECT image_id, cache_path FROM images WHERE embedding IS NULL AND cache_path IS NOT NULL"
            f"{clause} ORDER BY image_id",
            params,
        )
        return [(r["image_id"], r["cache_path"]) for r in cur.fetchall()]


def _batches(rows: list[tuple[str, str]], size: int) -> Iterator[list[tuple[str, str]]]:
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def encode_images(
    cfg: Config, *, batch_size: int = 32, region: str | None = None, encoder: Encoder | None = None
) -> int:
    """Embed every stored image that has no embedding yet.

    Args:
        cfg: Configuration (database).
        batch_size: Images per encoder call.
        region: Only encode this region.
        encoder: Encoder to use; defaults to the shared SigLIP-2 encoder.

    Returns:
        Number of embeddings written.
    """
    from pgvector.psycopg import register_vector

    written = 0
    with db.connect(cfg) as conn:
        register_vector(conn)
        rows = _pending_images(conn, region)
        LOG.info("%d images to encode", len(rows))
        if not rows:
            return 0
        enc = encoder or get_encoder()
        with tqdm(total=len(rows), desc="encode images") as pbar:
            for batch in _batches(rows, batch_size):
                loaded = [(iid, img) for iid, path in batch if (img := load_image(path)) is not None]
                if loaded:
                    feats = enc.encode_images([img for _, img in loaded])
                    with conn.cursor() as cur:
                        for (image_id, _), vec in zip(loaded, feats, strict=True):
                            cur.execute(
                                "UPDATE images SET embedding = %s, encoded_at = NOW() WHERE image_id = %s",
                                (vec, image_id),
                            )
                    conn.commit()
                    written += len(loaded)
                pbar.update(len(batch))
    return written


def _pending_tile_images(
    conn: db.Connection, scales: Sequence[int], overlaps: Sequence[float], region: str | None
) -> list[tuple[str, str]]:
    """Images with fewer tile rows than the requested grids produce (recut entirely; inserts are idempotent)."""
    expected = sum(s * s for s in scales) * len(overlaps)
    clause, params = _region_clause(region, "i.region")
    with conn.cursor() as cur:
        cur.execute(
            "SELECT i.image_id AS image_id, i.cache_path AS cache_path FROM images i "
            "LEFT JOIN (SELECT image_id, COUNT(*) AS n FROM tiles "
            "           WHERE scale = ANY(%s) AND overlap = ANY(%s) GROUP BY image_id) t "
            "ON t.image_id = i.image_id "
            f"WHERE i.cache_path IS NOT NULL AND COALESCE(t.n, 0) < %s{clause} ORDER BY i.image_id",
            [list(scales), list(overlaps), expected, *params],
        )
        return [(r["image_id"], r["cache_path"]) for r in cur.fetchall()]


def encode_tiles(
    cfg: Config,
    *,
    scales: Sequence[int] = (2,),
    overlaps: Sequence[float] = (0.0,),
    batch_size: int = 32,
    region: str | None = None,
    encoder: Encoder | None = None,
) -> int:
    """Cut grid tiles for every ``(scale, overlap)`` pair and store their embeddings.

    Args:
        cfg: Configuration (database).
        scales: Grid sides, e.g. ``(2, 3)`` for 2x2 and 3x3 grids.
        overlaps: Overlap fractions in ``[0, 1)``.
        batch_size: Tiles per encoder call.
        region: Only tile this region.
        encoder: Encoder to use; defaults to the shared SigLIP-2 encoder.

    Returns:
        Number of tile rows inserted.
    """
    from pgvector.psycopg import register_vector

    with db.connect(cfg) as conn:
        register_vector(conn)
        rows = _pending_tile_images(conn, scales, overlaps, region)
        LOG.info("%d images to tile (scales=%s, overlaps=%s)", len(rows), list(scales), list(overlaps))
        if not rows:
            return 0
        enc = encoder or get_encoder()
        buf: list[tuple[str, str, Tile]] = []
        inserted = 0

        def flush() -> int:
            if not buf:
                return 0
            feats = enc.encode_images([t.image for _, _, t in buf])
            n = 0
            with conn.cursor() as cur:
                for (tile_id, image_id, t), vec in zip(buf, feats, strict=True):
                    cur.execute(
                        """
                        INSERT INTO tiles (tile_id, image_id, scale, overlap, x_px, y_px, w_px, h_px,
                                           embedding, encoded_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                        ON CONFLICT (tile_id) DO NOTHING
                        """,
                        (tile_id, image_id, t.scale, t.overlap, t.x_px, t.y_px, t.w_px, t.h_px, vec),
                    )
                    n += cur.rowcount
            conn.commit()
            buf.clear()
            return n

        for image_id, path in tqdm(rows, desc="encode tiles"):
            parent = load_image(path)
            if parent is None:
                continue
            for s in scales:
                for o in overlaps:
                    for t in cut_tiles(parent, scale=s, overlap=o):
                        buf.append((f"{image_id}_{t.suffix}", image_id, t))
                        if len(buf) >= batch_size:
                            inserted += flush()
        inserted += flush()
    return inserted
