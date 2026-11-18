"""Postgres (pgvector) storage for the ingested corpus. The schema is in ``db/init.sql``."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from typing import Any

import psycopg
from psycopg.rows import dict_row

from .config import Config

Connection = psycopg.Connection[dict[str, Any]]
"""A psycopg connection whose rows are dicts."""


@contextmanager
def connect(cfg: Config) -> Iterator[Connection]:
    """Open a connection; commit on success, roll back on error, always close."""
    conn = psycopg.connect(cfg.pg_dsn, row_factory=dict_row, autocommit=False)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def candidate_seen(conn: Connection, region: str, lat: float, lon: float) -> dict[str, Any] | None:
    """Return the ``candidates_seen`` row for ``(region, lat, lon)``, or ``None``."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT status, pano_id, road_bearing FROM candidates_seen WHERE region = %s AND lat = %s AND lon = %s",
            (region, lat, lon),
        )
        return cur.fetchone()


def mark_candidate(
    conn: Connection,
    region: str,
    lat: float,
    lon: float,
    road_bearing: float | None,
    status: str,
    pano_id: str | None,
) -> None:
    """Record the outcome of a candidate so a restarted ingest skips it."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO candidates_seen (region, lat, lon, road_bearing, status, pano_id)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (region, lat, lon) DO UPDATE SET
                road_bearing = EXCLUDED.road_bearing,
                status       = EXCLUDED.status,
                pano_id      = EXCLUDED.pano_id,
                queried_at   = NOW()
            """,
            (region, lat, lon, road_bearing, status, pano_id),
        )


def upsert_panorama(
    conn: Connection,
    *,
    pano_id: str,
    region: str,
    lat: float,
    lon: float,
    captured_at: date | None,
    road_bearing: float | None,
    key_used: str,
    copyright_str: str | None,
) -> None:
    """Insert a panorama; do nothing if it already exists."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO panoramas (pano_id, region, lat, lon, captured_at, road_bearing, key_used, copyright)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (pano_id) DO NOTHING
            """,
            (pano_id, region, lat, lon, captured_at, road_bearing, key_used, copyright_str),
        )


def upsert_image(
    conn: Connection,
    *,
    image_id: str,
    pano_id: str,
    heading: float,
    pitch: float,
    fov: float,
    width: int | None,
    height: int | None,
    cache_path: str | None,
    region: str,
) -> None:
    """Insert an image, or update its file path and size if it exists."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO images (image_id, pano_id, heading, pitch, fov, width, height, cache_path, region)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (image_id) DO UPDATE SET
                cache_path = EXCLUDED.cache_path,
                width      = EXCLUDED.width,
                height     = EXCLUDED.height
            """,
            (image_id, pano_id, heading, pitch, fov, width, height, cache_path, region),
        )


def region_state(conn: Connection, region: str) -> tuple[set[str], int]:
    """Return the panorama ids already stored for ``region`` and its number of downloaded images."""
    with conn.cursor() as cur:
        cur.execute("SELECT pano_id FROM panoramas WHERE region = %s", (region,))
        panos = {row["pano_id"] for row in cur.fetchall()}
        cur.execute("SELECT COUNT(*) AS n FROM images WHERE region = %s AND cache_path IS NOT NULL", (region,))
        row = cur.fetchone()
    return panos, int(row["n"]) if row else 0


def count_images_per_key(conn: Connection) -> dict[str, int]:
    """Downloaded images per API key slot, to compare with the quota ledger."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT p.key_used AS key_used, COUNT(*) AS n
            FROM images i JOIN panoramas p ON p.pano_id = i.pano_id
            WHERE i.cache_path IS NOT NULL
            GROUP BY p.key_used
            """
        )
        return {row["key_used"]: int(row["n"]) for row in cur.fetchall()}


def count_images_per_region(conn: Connection) -> dict[str, int]:
    """Downloaded images per region."""
    with conn.cursor() as cur:
        cur.execute("SELECT region, COUNT(*) AS n FROM images WHERE cache_path IS NOT NULL GROUP BY region")
        return {row["region"]: int(row["n"]) for row in cur.fetchall()}
