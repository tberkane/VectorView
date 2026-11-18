"""Build a scan's search index: collect images, then embed images, tiles and patch regions.

Scans do not use Postgres. Their output is a set of numpy caches under
``<data_dir>/embeddings/``: ``scan_<id>`` (images and manifest), ``scan_<id>_tiles`` and
``scan_<id>_patches``. JPEGs go to the shared image cache.

Two collection modes share the embedding steps:

- ``api``: road points from OpenStreetMap, two views per panorama from the Street View Static
  API (needs ``GSV_API_KEY_*``; counted by the quota ledger);
- ``scraper``: every panorama in the box via `vectorview.scraper`, four crops per panorama at
  fixed compass headings (needs the ``scraper`` extra; no key).
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .. import cache, gsv, osm
from ..config import Config, load_config, require_gsv_keys
from ..encode import load_image
from ..encoder import get_encoder
from ..ingest import HEADING_OFFSETS, fetch_views
from ..regions import BBox
from ..tile import cut_tiles
from . import db
from .manifest import ImageMeta, manifest_path, write_manifest

LOG = logging.getLogger(__name__)

TILE_SCALES = (2,)
TILE_OVERLAPS = (0.0,)
ENCODE_BATCH = 32
PATCH_K = 8
PATCH_ENCODE_BATCH = 16

SCRAPER_HEADINGS = (0.0, 90.0, 180.0, 270.0)
SCRAPER_CROP_SIZE = 640
SCRAPER_FOV = 90.0
SCRAPER_PITCH = -10.0
SCRAPER_ZOOM = 3


def tags(scan_id: int) -> tuple[str, str, str]:
    """Cache tags of a scan: ``(images, tiles, patches)``."""
    return f"scan_{scan_id}", f"scan_{scan_id}_tiles", f"scan_{scan_id}_patches"


def _status(sid: int, status: str, msg: str, **counts: Any) -> None:
    db.update_scan(sid, status=status, status_message=msg, **counts)
    LOG.info("scan %d: %s (%s)", sid, status, msg)


def scan_bbox(scan: dict[str, Any]) -> BBox:
    """The scan's bounding box.

    Raises:
        RuntimeError: If the scan has no box.
    """
    if any(scan.get(k) is None for k in ("bbox_n", "bbox_s", "bbox_e", "bbox_w")):
        raise RuntimeError("this scan has no bounding box")
    return BBox(west=scan["bbox_w"], south=scan["bbox_s"], east=scan["bbox_e"], north=scan["bbox_n"])


# ---------------- collection ----------------


def ingest_api(scan: dict[str, Any], cfg: Config, client: gsv.GSVClient | None = None) -> list[ImageMeta]:
    """Collect two views per panorama along the roads of the scan's box with the Static API.

    Keys are used in slot order; when one reaches its cap the next one is used.

    Raises:
        RuntimeError: If no key is configured or no image could be fetched.
    """
    sid = int(scan["id"])
    require_gsv_keys(cfg)
    spacing = float(scan["spacing_m"])
    _status(sid, "fetching", "loading the road network")
    graph = osm.load_or_download_graph(f"scan_{sid}", scan_bbox(scan), cfg.osm_cache_dir)
    candidates = osm.sample_candidates(graph, spacing_m=spacing, dedup_m=spacing / 2)
    _status(sid, "fetching", f"{len(candidates)} road points", n_target=len(candidates) * len(HEADING_OFFSETS))

    client = client or gsv.open_client(cfg)
    slots = iter(client.slots)
    slot = next(slots)
    seen: set[str] = set()
    rows: list[ImageMeta] = []
    with client:
        for i, (lat, lon, road_bearing) in enumerate(candidates, 1):
            try:
                meta = client.metadata(lat, lon, slot)
            except gsv.GSVMetadataError as exc:
                LOG.warning("scan %d: metadata error at (%.5f, %.5f): %s", sid, lat, lon, exc)
                meta = None
            if meta is not None and meta["pano_id"] not in seen:
                seen.add(meta["pano_id"])
                while True:
                    try:
                        views = fetch_views(client, meta["pano_id"], road_bearing, slot, cfg.cache_dir)
                        break
                    except gsv.QuotaExhausted:
                        nxt = next(slots, None)
                        if nxt is None:
                            raise RuntimeError(f"every API key reached its cap after {len(rows)} images") from None
                        LOG.warning("scan %d: %s reached its cap; switching to %s", sid, slot, nxt)
                        slot = nxt
                captured = meta["date"].isoformat() if meta["date"] else ""
                rows.extend(
                    ImageMeta(
                        v.image_id,
                        meta["pano_id"],
                        meta["lat"],
                        meta["lon"],
                        captured,
                        v.heading,
                        str(v.path),
                        f"scan_{sid}",
                    )
                    for v in views
                )
            if i % 10 == 0 or i == len(candidates):
                msg = f"point {i}/{len(candidates)} · {len(rows)} images · {len(seen)} panoramas"
                _status(sid, "fetching", msg, n_fetched=len(rows))
    if not rows:
        raise RuntimeError("no image was fetched; the box may have no Street View coverage")
    return rows


def ingest_scraper(scan: dict[str, Any], cfg: Config) -> list[ImageMeta]:
    """Collect every panorama in the scan's box with `vectorview.scraper` and cut four crops each.

    The scraper's working directory (``<data_dir>/scrape/scan_<id>``) is kept, so a failed scan
    resumes where it stopped when it is run again.

    Raises:
        RuntimeError: If the scraper extra is missing or no crop was produced.
    """
    from shapely.geometry import box

    from ..scraper import GSVScraper

    sid = int(scan["id"])
    b = scan_bbox(scan)
    workdir = cfg.scrape_dir / f"scan_{sid}"
    scraper = GSVScraper(workdir, polygon=box(*b.as_tuple()), concurrency=8)

    _status(sid, "fetching", "discovering panoramas")
    scraper.discover()
    _status(sid, "fetching", "fetching panorama metadata")
    scraper.enrich()
    _status(sid, "fetching", "downloading panoramas")
    scraper.download(zoom=SCRAPER_ZOOM)
    _status(sid, "fetching", f"cutting {len(SCRAPER_HEADINGS)} crops per panorama")
    scraper.project(headings=SCRAPER_HEADINGS, fov=SCRAPER_FOV, pitch=SCRAPER_PITCH, size=SCRAPER_CROP_SIZE)

    rows = scraper_manifest(sid, workdir, cfg.cache_dir)
    if not rows:
        raise RuntimeError("the scraper produced no crops; the box may have no Street View coverage")
    _status(sid, "fetching", f"collected {len(rows)} crops", n_fetched=len(rows))
    return rows


def scraper_manifest(sid: int, workdir: Path, cache_root: Path) -> list[ImageMeta]:
    """List the crops of a scraper run and link (or copy) them into the shared image cache."""
    import sqlite3
    from contextlib import closing

    from ..scraper import crop_name

    with closing(sqlite3.connect(workdir / "state.db")) as conn:
        panos = conn.execute(
            "SELECT pano_id, lat, lon, date FROM panos WHERE downloaded_at IS NOT NULL AND image_path IS NOT NULL"
        ).fetchall()
    rows: list[ImageMeta] = []
    missing = 0
    for pano_id, lat, lon, captured in panos:
        for h in SCRAPER_HEADINGS:
            src = workdir / "crops" / crop_name(pano_id, h)
            if not src.exists():
                missing += 1
                continue
            image_id = gsv.make_image_id(pano_id, h)
            dest = gsv.cache_path_for(cache_root, image_id)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                try:
                    os.link(src, dest)
                except OSError:  # e.g. across file systems
                    shutil.copy2(src, dest)
            rows.append(
                ImageMeta(image_id, pano_id, float(lat), float(lon), str(captured or ""), h, str(dest), f"scan_{sid}")
            )
    if missing:
        LOG.warning("scan %d: %d crops missing on disk", sid, missing)
    return rows


# ---------------- embedding ----------------


def _batches(rows: list[ImageMeta], size: int) -> Iterator[list[tuple[ImageMeta, Image.Image]]]:
    for i in range(0, len(rows), size):
        yield [(r, img) for r in rows[i : i + size] if (img := load_image(r.cache_path)) is not None]


def encode_scan_images(sid: int, rows: list[ImageMeta], cfg: Config) -> list[ImageMeta]:
    """Embed the scan's images, write the image cache and manifest, and return the rows kept.

    Raises:
        RuntimeError: If no image could be read.
    """
    _status(sid, "encoding", "loading the encoder")
    enc = get_encoder()
    feats: list[np.ndarray] = []
    kept: list[ImageMeta] = []
    for batch in _batches(rows, ENCODE_BATCH):
        if batch:
            feats.append(enc.encode_images([img for _, img in batch]))
            kept.extend(r for r, _ in batch)
        _status(sid, "encoding", f"encoded {len(kept)}/{len(rows)}", n_encoded=len(kept))
    if not kept:
        raise RuntimeError("no image could be read")
    tag = tags(sid)[0]
    cache.write_images(
        cfg.embeddings_dir, tag, np.concatenate(feats), [r.image_id for r in kept], [r.region for r in kept]
    )
    write_manifest(manifest_path(cfg.embeddings_dir, tag), kept)
    return kept


def encode_scan_tiles(sid: int, rows: list[ImageMeta], cfg: Config) -> None:
    """Embed the grid tiles of the scan's images and write the tile cache."""
    enc = get_encoder()
    feats: list[np.ndarray] = []
    tile_ids: list[str] = []
    parents: list[str] = []
    pending: list[tuple[str, str, Image.Image]] = []

    def flush() -> None:
        if pending:
            feats.append(enc.encode_images([img for _, _, img in pending]))
            tile_ids.extend(t for t, _, _ in pending)
            parents.extend(p for _, p, _ in pending)
            pending.clear()

    for i, r in enumerate(rows, 1):
        parent = load_image(r.cache_path)
        if parent is not None:
            for s in TILE_SCALES:
                for o in TILE_OVERLAPS:
                    for t in cut_tiles(parent, scale=s, overlap=o):
                        pending.append((f"{r.image_id}_{t.suffix}", r.image_id, t.image))
                        if len(pending) >= ENCODE_BATCH:
                            flush()
        if i % 5 == 0 or i == len(rows):
            _status(sid, "tiling", f"tiled {i}/{len(rows)} images", n_tiled=i)
    flush()
    cache.write_tiles(cfg.embeddings_dir, tags(sid)[1], np.concatenate(feats), tile_ids, parents)


def encode_scan_patches(sid: int, rows: list[ImageMeta], cfg: Config) -> None:
    """Embed ``PATCH_K x PATCH_K`` patch regions of the scan's images and write the patch cache."""
    enc = get_encoder()
    feats: list[np.ndarray] = []
    ids: list[str] = []
    for batch in _batches(rows, PATCH_ENCODE_BATCH):
        if batch:
            feats.append(enc.encode_patch_regions([img for _, img in batch], k=PATCH_K))
            ids.extend(r.image_id for r, _ in batch)
        _status(sid, "patching", f"encoded regions of {len(ids)}/{len(rows)} images", n_patched=len(ids))
    cache.write_patch_regions(cfg.embeddings_dir, tags(sid)[2], np.concatenate(feats), ids)


def finalize(sid: int) -> None:
    """Point the scan at its caches and mark it ready."""
    image_tag, tile_tag, patch_tag = tags(sid)
    db.update_scan(
        sid,
        embedding_tag=image_tag,
        tile_tag=tile_tag,
        patch_region_tag=patch_tag,
        region_filter=f"scan_{sid}",
        status="ready",
        status_message="ready",
        ready=True,
    )


INGESTERS: dict[str, Callable[[dict[str, Any], Config], list[ImageMeta]]] = {
    "api": ingest_api,
    "scraper": ingest_scraper,
}


def build_scan(scan: dict[str, Any], cfg: Config | None = None) -> None:
    """Run every step of a scan synchronously."""
    cfg = cfg or load_config()
    sid = int(scan["id"])
    rows = INGESTERS[scan["fetch_mode"]](scan, cfg)
    rows = encode_scan_images(sid, rows, cfg)
    _status(sid, "tiling", "cutting tiles")
    encode_scan_tiles(sid, rows, cfg)
    _status(sid, "patching", "encoding patch regions")
    encode_scan_patches(sid, rows, cfg)
    finalize(sid)
    LOG.info("scan %d: ready", sid)


async def run_scan(scan: dict[str, Any]) -> None:
    """Run a pending scan in a worker thread; errors set ``status='failed'``."""
    sid = int(scan["id"])
    if scan.get("fetch_mode") not in INGESTERS:
        db.update_scan(sid, status="failed", error=f"unknown fetch_mode {scan.get('fetch_mode')!r}")
        return
    try:
        await asyncio.to_thread(build_scan, scan)
    except Exception as exc:  # reported to the UI
        LOG.exception("scan %d failed", sid)
        db.update_scan(sid, status="failed", error=str(exc))
