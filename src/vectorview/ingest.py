"""Collect Street View images along the roads of a region into the Postgres corpus.

For each road candidate from `vectorview.osm.sample_candidates`:

1. look up the nearest panorama (metadata requests are free);
2. skip panoramas already stored;
3. fetch two views, looking forward and backward along the road (one quota unit each);
4. write the JPEGs to the cache and the rows to Postgres.

Candidates are processed by a thread pool. Keys are used in slot order: when one reaches its
cap the ingest moves to the next. Every candidate's outcome is recorded in ``candidates_seen``,
so an interrupted ingest resumes where it stopped.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row
from tqdm import tqdm

from . import db, osm
from .config import Config
from .gsv import GSVClient, GSVImageError, GSVMetadataError, QuotaExhausted, cache_path_for, make_image_id
from .regions import BBox

LOG = logging.getLogger(__name__)

HEADING_OFFSETS = (0.0, 180.0)
"""Views per panorama, relative to the road bearing: forward and backward."""

IMAGE_SIZE_PX = 640


@dataclass(frozen=True)
class FetchedView:
    """One downloaded view of a panorama."""

    image_id: str
    heading: float
    path: Path


def fetch_views(
    client: GSVClient,
    pano_id: str,
    road_bearing: float,
    key_slot: str,
    cache_dir: Path,
    offsets: tuple[float, ...] = HEADING_OFFSETS,
) -> list[FetchedView]:
    """Download the views of one panorama into the cache, skipping views that fail.

    Raises:
        QuotaExhausted: If ``key_slot`` reaches its cap.
    """
    views: list[FetchedView] = []
    for offset in offsets:
        heading = (road_bearing + offset) % 360.0
        try:
            blob = client.image(pano_id, heading, key_slot)
        except GSVImageError as exc:
            LOG.warning("image error pano=%s heading=%.1f: %s", pano_id, heading, exc)
            continue
        image_id = make_image_id(pano_id, heading)
        path = cache_path_for(cache_dir, image_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        views.append(FetchedView(image_id, heading, path))
    return views


@dataclass
class IngestStats:
    """Counters of one ingest run."""

    region: str
    candidates_total: int = 0
    skipped_seen: int = 0
    no_coverage: int = 0
    duplicate_pano: int = 0
    panos_kept: int = 0
    images_kept: int = 0
    metadata_errors: int = 0
    image_errors: int = 0
    exhausted_slots: list[str] = field(default_factory=list)

    def add(self, delta: dict[str, int]) -> None:
        """Add the counts in ``delta``."""
        for k, v in delta.items():
            setattr(self, k, getattr(self, k) + v)


class _Worker:
    """Processes one candidate per call, with one database connection per thread."""

    def __init__(self, cfg: Config, region: str, client: GSVClient, seen_panos: set[str]) -> None:
        self.cfg = cfg
        self.region = region
        self.client = client
        self.seen = seen_panos
        self.lock = threading.Lock()
        self.local = threading.local()
        self.conns: list[db.Connection] = []

    def conn(self) -> db.Connection:
        c = getattr(self.local, "conn", None)
        if c is None or c.closed:
            c = psycopg.connect(self.cfg.pg_dsn, row_factory=dict_row, autocommit=False)
            self.local.conn = c
            with self.lock:
                self.conns.append(c)
        return c

    def close(self) -> None:
        for c in self.conns:
            c.close()

    def __call__(self, cand: osm.Candidate, key_slot: str) -> dict[str, int]:
        lat, lon, road_bearing = cand
        conn = self.conn()
        if db.candidate_seen(conn, self.region, lat, lon) is not None:
            return {"skipped_seen": 1}
        try:
            meta = self.client.metadata(lat, lon, key_slot)
        except GSVMetadataError as exc:
            LOG.warning("metadata error at (%.5f, %.5f): %s", lat, lon, exc)
            return {"metadata_errors": 1}
        if meta is None:
            db.mark_candidate(conn, self.region, lat, lon, road_bearing, "no_coverage", None)
            conn.commit()
            return {"no_coverage": 1}

        pano_id = meta["pano_id"]
        with self.lock:
            duplicate = pano_id in self.seen
            self.seen.add(pano_id)
        if duplicate:
            db.mark_candidate(conn, self.region, lat, lon, road_bearing, "duplicate_pano", pano_id)
            conn.commit()
            return {"duplicate_pano": 1}

        try:
            views = fetch_views(self.client, pano_id, road_bearing, key_slot, self.cfg.cache_dir)
        except QuotaExhausted:
            # Let the next key slot retry this candidate.
            with self.lock:
                self.seen.discard(pano_id)
            raise
        if not views:
            db.mark_candidate(conn, self.region, lat, lon, road_bearing, "image_fetch_failed", pano_id)
            conn.commit()
            return {"image_errors": 1}

        db.upsert_panorama(
            conn,
            pano_id=pano_id,
            region=self.region,
            lat=meta["lat"],
            lon=meta["lon"],
            captured_at=meta["date"],
            road_bearing=road_bearing,
            key_used=key_slot,
            copyright_str=meta["copyright"],
        )
        for v in views:
            db.upsert_image(
                conn,
                image_id=v.image_id,
                pano_id=pano_id,
                heading=v.heading,
                pitch=0.0,
                fov=90.0,
                width=IMAGE_SIZE_PX,
                height=IMAGE_SIZE_PX,
                cache_path=str(v.path),
                region=self.region,
            )
        db.mark_candidate(conn, self.region, lat, lon, road_bearing, "fetched", pano_id)
        conn.commit()
        return {"panos_kept": 1, "images_kept": len(views)}


def ingest_region(
    cfg: Config,
    name: str,
    bbox: BBox,
    client: GSVClient,
    *,
    max_images: int | None = None,
    spacing_m: float = 50.0,
    workers: int = 4,
) -> IngestStats:
    """Collect images along the roads of ``bbox`` into the corpus under region ``name``.

    Args:
        cfg: Configuration (database, cache directory).
        name: Region name; also names the cached road graph. Reuse it to resume.
        bbox: Area to cover.
        client: Street View client; its key slots are used in order.
        max_images: Stop once the region holds this many images (``None``: no limit).
        spacing_m: Distance between road candidates.
        workers: Concurrent candidates.

    Returns:
        Counters for the run; ``images_kept`` includes images stored by earlier runs.
    """
    graph = osm.load_or_download_graph(name, bbox, cfg.osm_cache_dir)
    candidates = osm.sample_candidates(graph, spacing_m=spacing_m, dedup_m=spacing_m / 2)
    with db.connect(cfg) as conn:
        seen_panos, images_already = db.region_state(conn, name)
    stats = IngestStats(region=name, candidates_total=len(candidates), images_kept=images_already)
    target = max_images if max_images is not None else float("inf")
    LOG.info("region=%s: %d candidates; resuming with %d images", name, len(candidates), images_already)

    worker = _Worker(cfg, name, client, seen_panos)
    pending: Iterator[osm.Candidate] = iter(candidates)
    retry: list[osm.Candidate] = []
    total = max_images if max_images is not None else len(candidates) * len(HEADING_OFFSETS)
    pbar = tqdm(total=total, initial=min(images_already, total), desc=f"images[{name}]")
    try:
        for slot in client.slots:
            if stats.images_kept >= target:
                break
            exhausted = _run_slot(worker, slot, pending, retry, stats, target, workers, pbar)
            if not exhausted:
                break  # candidates ran out or the target was reached
            stats.exhausted_slots.append(slot)
            LOG.info("region=%s: %s reached its cap; moving to the next key", name, slot)
    finally:
        pbar.close()
        worker.close()
    LOG.info("region=%s: %s", name, stats)
    return stats


def _run_slot(
    worker: _Worker,
    slot: str,
    pending: Iterator[osm.Candidate],
    retry: list[osm.Candidate],
    stats: IngestStats,
    target: float,
    workers: int,
    pbar: tqdm[Any],
) -> bool:
    """Process candidates with one key slot; return True if the slot hit its cap."""
    exhausted = False
    with ThreadPoolExecutor(max_workers=workers) as pool:
        in_flight: dict[Future[dict[str, int]], osm.Candidate] = {}

        def submit_next() -> None:
            if exhausted or stats.images_kept >= target:
                return
            cand = retry.pop() if retry else next(pending, None)
            if cand is not None:
                in_flight[pool.submit(worker, cand, slot)] = cand

        for _ in range(workers):
            submit_next()
        while in_flight:
            done, _ = wait(list(in_flight), return_when=FIRST_COMPLETED)
            for fut in done:
                cand = in_flight.pop(fut)
                try:
                    delta = fut.result()
                except QuotaExhausted:
                    exhausted = True
                    retry.append(cand)
                    continue
                except Exception as exc:  # a failed candidate must not stop the run
                    LOG.warning("candidate %s failed: %s", cand, exc)
                    continue
                stats.add(delta)
                pbar.update(delta.get("images_kept", 0))
                submit_next()
    return exhausted
