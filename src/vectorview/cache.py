"""Numpy embedding caches.

The web app ranks images with in-memory matrix products, so embeddings are kept as ``.npy``
files, one folder per *tag* under ``<data_dir>/embeddings/``:

    <tag>/embeddings.npy   (N, D) float32 image embeddings, or (N, K*K, D) float16 patch regions
    <tag>/image_ids.txt    N image ids, one per line, in row order
    <tag>/regions.txt      N region names (image caches only)
    <tag>/tile_ids.txt     N tile ids (tile caches only; image_ids.txt then holds each tile's parent)
    <tag>/manifest.csv     image metadata (see `vectorview.web.manifest`)

Caches are written by scans in the web app and by [`export_corpus`][vectorview.cache.export_corpus]
from the Postgres corpus.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .config import Config

LOG = logging.getLogger(__name__)


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def write_images(root: Path, tag: str, feats: np.ndarray, image_ids: list[str], regions: list[str]) -> Path:
    """Write an image-embedding cache and return its folder."""
    _check(len(feats) == len(image_ids) == len(regions), f"cache {tag}: row counts differ")
    d = root / tag
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "embeddings.npy", np.asarray(feats, dtype=np.float32))
    (d / "image_ids.txt").write_text("\n".join(image_ids))
    (d / "regions.txt").write_text("\n".join(regions))
    LOG.info("wrote %d image embeddings to %s", len(image_ids), d)
    return d


def load_images(root: Path, tag: str) -> tuple[np.ndarray, list[str], list[str]]:
    """Load an image cache as ``(feats, image_ids, regions)``.

    Raises:
        FileNotFoundError: If the cache does not exist.
        RuntimeError: If its files disagree on the number of rows.
    """
    d = root / tag
    feats = np.load(d / "embeddings.npy").astype(np.float32, copy=False)
    ids = _read_lines(d / "image_ids.txt")
    regions_path = d / "regions.txt"
    regions = _read_lines(regions_path) if regions_path.exists() else [""] * len(ids)
    _check(len(ids) == len(feats) == len(regions), f"cache {tag}: id/embedding mismatch ({len(ids)} vs {len(feats)})")
    return feats, ids, regions


def write_tiles(root: Path, tag: str, feats: np.ndarray, tile_ids: list[str], parent_ids: list[str]) -> Path:
    """Write a tile-embedding cache and return its folder."""
    _check(len(feats) == len(tile_ids) == len(parent_ids), f"tile cache {tag}: row counts differ")
    d = root / tag
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "embeddings.npy", np.asarray(feats, dtype=np.float32))
    (d / "tile_ids.txt").write_text("\n".join(tile_ids))
    (d / "image_ids.txt").write_text("\n".join(parent_ids))
    LOG.info("wrote %d tile embeddings to %s", len(tile_ids), d)
    return d


def load_tiles(root: Path, tag: str) -> tuple[np.ndarray, list[str], list[str]]:
    """Load a tile cache as ``(feats, tile_ids, parent_image_ids)``.

    Raises:
        FileNotFoundError: If the cache does not exist.
        RuntimeError: If its files disagree on the number of rows.
    """
    d = root / tag
    feats = np.load(d / "embeddings.npy").astype(np.float32, copy=False)
    tile_ids = _read_lines(d / "tile_ids.txt")
    parent_ids = _read_lines(d / "image_ids.txt")
    _check(len(tile_ids) == len(parent_ids) == len(feats), f"tile cache {tag}: row counts differ")
    return feats, tile_ids, parent_ids


def write_patch_regions(root: Path, tag: str, feats: np.ndarray, image_ids: list[str]) -> Path:
    """Write a patch-region cache, shape ``(N, K*K, D)``, and return its folder."""
    _check(feats.ndim == 3 and len(feats) == len(image_ids), f"patch cache {tag}: bad shape {feats.shape}")
    d = root / tag
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "embeddings.npy", np.asarray(feats, dtype=np.float16))
    (d / "image_ids.txt").write_text("\n".join(image_ids))
    return d


def load_patch_regions(root: Path, tag: str, image_ids: list[str], dtype: type = np.float32) -> np.ndarray:
    """Load a patch-region cache with rows aligned to ``image_ids``.

    Images missing from the cache get all-zero rows, which callers treat as "no region score".

    Returns:
        Shape ``(len(image_ids), K*K, D)``.
    """
    d = root / tag
    arr = np.load(d / "embeddings.npy")
    index = {iid: i for i, iid in enumerate(_read_lines(d / "image_ids.txt"))}
    out = np.zeros((len(image_ids), arr.shape[1], arr.shape[2]), dtype=dtype)
    for i, iid in enumerate(image_ids):
        j = index.get(iid)
        if j is not None:
            out[i] = arr[j]
    return out


def _vec(value: object) -> np.ndarray:
    """A pgvector value as a float32 array (pgvector returns ``Vector`` objects or arrays)."""
    to_numpy = getattr(value, "to_numpy", None)
    return np.asarray(to_numpy() if to_numpy else value, dtype=np.float32)


def _read_lines(path: Path) -> list[str]:
    text = path.read_text()
    return text.splitlines() if text else []


def export_corpus(cfg: Config, tag: str, *, region: str | None = None) -> tuple[int, int]:
    """Copy the Postgres corpus into numpy caches ``<tag>`` and ``<tag>_tiles`` plus a manifest.

    Args:
        cfg: Configuration (database connection and data directory).
        tag: Name of the image cache to write.
        region: Export only this region.

    Returns:
        ``(n_images, n_tiles)`` exported.

    Raises:
        RuntimeError: If no image has an embedding yet (run ``vectorview encode`` first).
    """
    from pgvector.psycopg import register_vector

    from . import db
    from .web import manifest

    where, params = ("AND i.region = %s", (region,)) if region else ("", ())
    with db.connect(cfg) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT i.image_id, i.region, i.embedding, i.heading, i.cache_path, "
                "       p.pano_id, p.lat, p.lon, p.captured_at "
                "FROM images i JOIN panoramas p ON p.pano_id = i.pano_id "
                f"WHERE i.embedding IS NOT NULL {where} ORDER BY i.image_id",
                params,
            )
            rows = cur.fetchall()
            _check(bool(rows), "no image embeddings in the database; run `vectorview encode` first")
            cur.execute(
                "SELECT t.tile_id, t.image_id, t.embedding FROM tiles t JOIN images i ON i.image_id = t.image_id "
                f"WHERE t.embedding IS NOT NULL AND i.embedding IS NOT NULL {where} ORDER BY t.tile_id",
                params,
            )
            tile_rows = cur.fetchall()

    root = cfg.embeddings_dir
    write_images(
        root,
        tag,
        np.stack([_vec(r["embedding"]) for r in rows]),
        [r["image_id"] for r in rows],
        [r["region"] for r in rows],
    )
    dim = _vec(rows[0]["embedding"]).shape[0]
    tile_feats = (
        np.stack([_vec(r["embedding"]) for r in tile_rows]) if tile_rows else np.empty((0, dim), dtype=np.float32)
    )
    write_tiles(root, f"{tag}_tiles", tile_feats, [r["tile_id"] for r in tile_rows], [r["image_id"] for r in tile_rows])
    manifest.write_manifest(
        manifest.manifest_path(root, tag),
        [
            manifest.ImageMeta(
                image_id=r["image_id"],
                pano_id=r["pano_id"],
                lat=float(r["lat"]),
                lon=float(r["lon"]),
                captured_at=r["captured_at"].isoformat() if r["captured_at"] else "",
                heading=float(r["heading"] or 0.0),
                cache_path=r["cache_path"] or "",
                region=r["region"],
            )
            for r in rows
        ],
    )
    return len(rows), len(tile_rows)
