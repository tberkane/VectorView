"""Collect Street View panoramas with the `streetlevel` library (needs the ``scraper`` extra).

Unlike `vectorview.gsv`, this path does not use the Static API: it needs no API key and has no
quota ledger. It finds every panorama in an area rather than sampling points along roads.

Steps, each resumable through a SQLite state file in ``data_dir``:

1. `GSVScraper.discover` lists panoramas in the zoom-17 coverage tiles over the area;
2. `GSVScraper.enrich` fetches each panorama's metadata (capture date, country, heading);
3. `GSVScraper.download` saves equirectangular JPEGs;
4. `GSVScraper.project` cuts flat perspective crops at fixed compass headings;
5. `GSVScraper.export` writes the metadata to CSV.

Layout of ``data_dir``: ``state.db``, ``panos/<pano_id>.jpg``, ``crops/<pano_id>_h<heading>.jpg``.
"""

from __future__ import annotations

import asyncio
import csv
import logging
import math
import multiprocessing as mp
import os
import sqlite3
import sys
import time
import types
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path
from typing import Any

from tqdm import tqdm

LOG = logging.getLogger(__name__)

DEFAULT_CONCURRENCY = 16
DEFAULT_ZOOM = 3
TILE_ZOOM = 17
# The panorama tile endpoint rejects requests without a browser User-Agent.
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

SCHEMA = """
CREATE TABLE IF NOT EXISTS tiles (
    x INTEGER NOT NULL, y INTEGER NOT NULL, z INTEGER NOT NULL,
    fetched_at REAL, pano_count INTEGER,
    PRIMARY KEY (x, y, z)
);
CREATE TABLE IF NOT EXISTS panos (
    pano_id TEXT PRIMARY KEY,
    lat REAL NOT NULL, lon REAL NOT NULL,
    heading REAL, elevation REAL,
    date TEXT, country_code TEXT,
    enriched_at REAL, downloaded_at REAL,
    image_path TEXT, error TEXT
);
CREATE INDEX IF NOT EXISTS idx_panos_enriched ON panos(enriched_at);
CREATE INDEX IF NOT EXISTS idx_panos_downloaded ON panos(downloaded_at);
"""


def lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    """Slippy-map tile ``(x, y)`` containing ``(lon, lat)`` at zoom ``z``."""
    n = 2**z
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def yaw_for_heading(heading_deg: float, pano_heading_rad: float | None) -> float:
    """Yaw, in degrees within ``[-180, 180)``, that points a crop at compass ``heading_deg``.

    The centre column of an equirectangular panorama faces the panorama's own heading, which
    `streetlevel` reports in radians.
    """
    pano_deg = math.degrees(pano_heading_rad or 0.0)
    return ((heading_deg - pano_deg + 540.0) % 360.0) - 180.0


def crop_name(pano_id: str, heading: float) -> str:
    """File name of a perspective crop."""
    return f"{pano_id}_h{int(heading):03d}.jpg"


def install_pyexiv2_shim() -> bool:
    """Replace ``pyexiv2`` with a no-op module if its native library cannot load.

    `streetlevel` imports ``pyexiv2`` to write EXIF tags into downloaded panoramas. Its bundled
    library needs a recent glibc; on older systems the import fails with ``OSError``. VectorView
    keeps metadata in its own database, so the tags are not needed. Call before importing
    `streetlevel`.

    Returns:
        True if the shim was installed.
    """
    try:
        import pyexiv2  # noqa: F401
    except OSError:
        pass
    except ImportError:
        return False
    else:
        return False

    class _ImageData:
        def __init__(self, buf: bytes) -> None:
            self._buf = buf

        def __enter__(self) -> _ImageData:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def modify_exif(self, *args: object, **kwargs: object) -> None:
            pass

        def modify_xmp(self, *args: object, **kwargs: object) -> None:
            pass

        def get_bytes(self) -> bytes:
            return self._buf

    shim = types.ModuleType("pyexiv2")
    shim.ImageData = _ImageData  # type: ignore[attr-defined]
    sys.modules["pyexiv2"] = shim
    LOG.warning("pyexiv2 could not load its native library; panoramas are saved without EXIF tags")
    return True


def _streetview() -> Any:
    install_pyexiv2_shim()
    try:
        from streetlevel import streetview
    except ImportError as exc:
        raise RuntimeError(
            f'scraper mode needs the scraper extra ({exc.name} is missing): pip install "vectorview[scraper]"'
        ) from exc
    return streetview


def _session() -> Any:
    import aiohttp

    return aiohttp.ClientSession(headers={"User-Agent": USER_AGENT})


def project_one(args: tuple[str, str, float | None, list[float], float, float, int, str]) -> tuple[str, str | None]:
    """Cut the perspective crops of one panorama with py360convert.

    Top-level so that a multiprocessing pool can pickle it.

    Args:
        args: ``(pano_id, equirect_path, pano_heading_rad, headings, fov, pitch, size, out_dir)``.

    Returns:
        ``(pano_id, error)``; ``error`` is ``None`` on success.
    """
    pano_id, src, pano_heading, headings, fov, pitch, size, out_dir = args
    try:
        import numpy as np
        import py360convert
        from PIL import Image

        outs = [Path(out_dir) / crop_name(pano_id, h) for h in headings]
        if all(p.exists() for p in outs):
            return pano_id, None
        with Image.open(src) as im:
            equirect = np.asarray(im.convert("RGB"))
        for h, out in zip(headings, outs, strict=True):
            if out.exists():
                continue
            persp = py360convert.e2p(
                equirect,
                fov_deg=fov,
                u_deg=yaw_for_heading(h, pano_heading),
                v_deg=pitch,
                out_hw=(size, size),
                mode="bilinear",
            )
            Image.fromarray(persp).save(out, "JPEG", quality=92)
        return pano_id, None
    except Exception as exc:  # reported per panorama, not fatal
        return pano_id, repr(exc)


class GSVScraper:
    """Resumable panorama collection over an area.

    Args:
        data_dir: Directory for the state database, panoramas and crops.
        place: Place name geocoded with OpenStreetMap (e.g. ``"Cambridge, Massachusetts, USA"``).
        polygon: Area as a shapely geometry in WGS84; used instead of ``place`` when given.
            Only `discover` needs an area.
        concurrency: Concurrent HTTP requests.
        gpu: Project crops on a CUDA GPU when ``pyequilib`` and CUDA are available.
    """

    def __init__(
        self,
        data_dir: str | os.PathLike[str],
        *,
        place: str | None = None,
        polygon: Any = None,
        concurrency: int = DEFAULT_CONCURRENCY,
        gpu: bool = True,
    ) -> None:
        self.place = place
        self._polygon = polygon
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / "panos"
        self.crop_dir = self.data_dir / "crops"
        self.db_path = self.data_dir / "state.db"
        self.concurrency = concurrency
        self.gpu = gpu
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.executescript(SCHEMA)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def polygon(self) -> Any:
        """The area covered, geocoding ``place`` on first use.

        Raises:
            ValueError: If neither a place nor a polygon was given.
        """
        if self._polygon is None:
            if not self.place:
                raise ValueError("give a place name or a polygon")
            import osmnx as ox

            self._polygon = ox.geocode_to_gdf(self.place).geometry.iloc[0]
        return self._polygon

    def discover(self) -> int:
        """List the panoramas inside the area; return how many are known in total."""
        asyncio.run(self._discover_async())
        return self.stats()["panos_discovered"]

    def enrich(self) -> None:
        """Fetch metadata (date, country, heading) of each discovered panorama."""
        asyncio.run(self._enrich_async())

    def download(self, zoom: int = DEFAULT_ZOOM, limit: int | None = None) -> None:
        """Download equirectangular JPEGs; ``zoom`` 0-5 sets the resolution (3 is about 4096 px wide)."""
        asyncio.run(self._download_async(zoom=zoom, limit=limit))

    def downloaded(self, limit: int | None = None) -> list[tuple[str, str, float | None]]:
        """``(pano_id, image_path, heading_rad)`` of the downloaded panoramas, sorted by id."""
        sql = (
            "SELECT pano_id, image_path, heading FROM panos "
            "WHERE downloaded_at IS NOT NULL AND image_path IS NOT NULL ORDER BY pano_id"
        )
        with closing(self._connect()) as conn:
            rows = conn.execute(sql + (" LIMIT ?" if limit else ""), (limit,) if limit else ()).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def project(
        self,
        headings: Iterable[float] = (0, 90, 180, 270),
        fov: float = 90.0,
        pitch: float = -10.0,
        size: int = 512,
        workers: int | None = None,
        limit: int | None = None,
    ) -> None:
        """Cut flat perspective crops from the downloaded panoramas.

        Args:
            headings: Compass bearings of the crops (0 = north, 90 = east).
            fov: Horizontal field of view, in degrees.
            pitch: Vertical angle, in degrees (negative looks down).
            size: Crop side, in pixels.
            workers: CPU processes (default: CPU count minus one).
            limit: Only project this many panoramas.
        """
        self.crop_dir.mkdir(parents=True, exist_ok=True)
        hs = [float(h) for h in headings]
        rows = self.downloaded(limit)
        LOG.info("projecting %d panoramas x %d crops (fov=%s pitch=%s size=%d)", len(rows), len(hs), fov, pitch, size)
        if not rows:
            return
        if self.gpu:
            try:
                from ._gpu_project import project_gpu

                if project_gpu(rows, self.crop_dir, hs, fov, pitch, size):
                    return
            except Exception as exc:  # fall back to the CPU path
                LOG.warning("GPU projection failed (%s); using the CPU path", exc)
        work = [(pid, path, h, hs, fov, pitch, size, str(self.crop_dir)) for pid, path, h in rows]
        n_err = 0
        with mp.Pool(workers or max(1, (os.cpu_count() or 2) - 1)) as pool:
            for pid, err in tqdm(pool.imap_unordered(project_one, work, chunksize=8), total=len(work), desc="project"):
                if err is not None:
                    n_err += 1
                    LOG.warning("projection failed for %s: %s", pid, err)
        LOG.info("projection done: %d ok, %d failed", len(work) - n_err, n_err)

    def export(self, out_path: str | os.PathLike[str]) -> int:
        """Write the panorama metadata to a CSV file; return the number of rows."""
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, out.open("w", newline="") as f:
            rows = conn.execute(
                "SELECT pano_id, lat, lon, date, heading, elevation, country_code, image_path "
                "FROM panos ORDER BY date DESC NULLS LAST, pano_id"
            ).fetchall()
            w = csv.writer(f)
            w.writerow(["pano_id", "lat", "lon", "date", "heading", "elevation", "country_code", "image_path"])
            w.writerows(rows)
        LOG.info("wrote %d rows to %s", len(rows), out)
        return len(rows)

    def stats(self) -> dict[str, int]:
        """Progress counters of each step."""
        queries = {
            "tiles_fetched": "SELECT COUNT(*) FROM tiles",
            "panos_discovered": "SELECT COUNT(*) FROM panos",
            "panos_enriched": "SELECT COUNT(*) FROM panos WHERE enriched_at IS NOT NULL",
            "panos_with_date": "SELECT COUNT(*) FROM panos WHERE date IS NOT NULL",
            "panos_downloaded": "SELECT COUNT(*) FROM panos WHERE downloaded_at IS NOT NULL",
            "panos_errored": "SELECT COUNT(*) FROM panos WHERE error IS NOT NULL AND error <> ''",
        }
        with closing(self._connect()) as conn:
            return {k: int(conn.execute(q).fetchone()[0]) for k, q in queries.items()}

    async def _discover_async(self) -> None:
        from shapely.geometry import Point

        streetview = _streetview()
        polygon = self.polygon()
        minx, miny, maxx, maxy = polygon.bounds
        x_min, y_max = lonlat_to_tile(minx, miny, TILE_ZOOM)
        x_max, y_min = lonlat_to_tile(maxx, maxy, TILE_ZOOM)
        all_tiles = [(x, y) for x in range(x_min, x_max + 1) for y in range(y_min, y_max + 1)]
        with closing(self._connect()) as conn:
            done = {
                (r[0], r[1])
                for r in conn.execute("SELECT x, y FROM tiles WHERE z = ? AND fetched_at IS NOT NULL", (TILE_ZOOM,))
            }
            todo = [t for t in all_tiles if t not in done]
            LOG.info("coverage tiles: %d total, %d to fetch", len(all_tiles), len(todo))
            if not todo:
                return
            sem = asyncio.Semaphore(self.concurrency)

            async def fetch(session: Any, x: int, y: int) -> tuple[int, int, list[Any], str | None]:
                async with sem:
                    try:
                        return x, y, await streetview.get_coverage_tile_async(x, y, session=session), None
                    except Exception as exc:  # retried on the next run
                        return x, y, [], str(exc)

            async with _session() as session:
                tasks = [asyncio.create_task(fetch(session, x, y)) for x, y in todo]
                for i, coro in enumerate(tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="discover"), 1):
                    x, y, panos, err = await coro
                    if err:
                        LOG.warning("coverage tile %d/%d failed: %s", x, y, err)
                        continue
                    for p in panos:
                        if polygon.covers(Point(p.lon, p.lat)):
                            conn.execute(
                                "INSERT OR IGNORE INTO panos(pano_id, lat, lon, heading, elevation) VALUES (?,?,?,?,?)",
                                (p.id, p.lat, p.lon, p.heading, p.elevation),
                            )
                    conn.execute(
                        "INSERT OR REPLACE INTO tiles(x, y, z, fetched_at, pano_count) VALUES (?,?,?,?,?)",
                        (x, y, TILE_ZOOM, time.time(), len(panos)),
                    )
                    if i % 200 == 0:
                        conn.commit()
            conn.commit()

    @staticmethod
    async def _find(streetview: Any, session: Any, pid: str, lat: float, lon: float) -> Any:
        # Look up by id, falling back to the nearest panorama when the by-id lookup returns nothing.
        p = await streetview.find_panorama_by_id_async(pid, session=session)
        if p is None:
            p = await streetview.find_panorama_async(lat, lon, session=session)
        return p

    async def _enrich_async(self) -> None:
        streetview = _streetview()
        with closing(self._connect()) as conn:
            rows = conn.execute("SELECT pano_id, lat, lon FROM panos WHERE enriched_at IS NULL").fetchall()
            LOG.info("panoramas to enrich: %d", len(rows))
            if not rows:
                return
            sem = asyncio.Semaphore(self.concurrency)

            async def one(session: Any, pid: str, lat: float, lon: float) -> tuple[str, Any, str | None]:
                async with sem:
                    try:
                        p = await self._find(streetview, session, pid, lat, lon)
                        return pid, p, None if p else "not_found"
                    except Exception as exc:  # recorded per panorama
                        return pid, None, str(exc)

            async with _session() as session:
                tasks = [asyncio.create_task(one(session, *r)) for r in rows]
                for i, coro in enumerate(tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="enrich"), 1):
                    pid, p, err = await coro
                    if p is not None:
                        conn.execute(
                            "UPDATE panos SET date=?, country_code=?, elevation=?, heading=?, enriched_at=?, "
                            "error=NULL WHERE pano_id=?",
                            (str(p.date) if p.date else None, p.country_code, p.elevation, p.heading, time.time(), pid),
                        )
                    else:
                        conn.execute("UPDATE panos SET enriched_at=?, error=? WHERE pano_id=?", (time.time(), err, pid))
                    if i % 500 == 0:
                        conn.commit()
            conn.commit()

    async def _download_async(self, zoom: int, limit: int | None) -> None:
        streetview = _streetview()
        self.img_dir.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            # Retry earlier failures, except panoramas that could not be found.
            sql = (
                "SELECT pano_id, lat, lon FROM panos WHERE downloaded_at IS NULL "
                "AND (error IS NULL OR error NOT LIKE '%not_found%') ORDER BY pano_id"
            )
            rows = conn.execute(sql + (" LIMIT ?" if limit else ""), (limit,) if limit else ()).fetchall()
            LOG.info("panoramas to download: %d (zoom=%d)", len(rows), zoom)
            if not rows:
                return
            sem = asyncio.Semaphore(self.concurrency)

            async def one(session: Any, pid: str, lat: float, lon: float) -> tuple[str, str | None, str | None]:
                out = self.img_dir / f"{pid}.jpg"
                async with sem:
                    if out.exists():
                        return pid, str(out), None
                    try:
                        p = await self._find(streetview, session, pid, lat, lon)
                        if p is None:
                            return pid, None, "not_found"
                        await streetview.download_panorama_async(p, str(out), zoom=zoom, session=session)
                        return pid, str(out), None
                    except Exception as exc:  # retried on the next run
                        out.unlink(missing_ok=True)
                        return pid, None, str(exc)

            async with _session() as session:
                tasks = [asyncio.create_task(one(session, *r)) for r in rows]
                for i, coro in enumerate(tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="download"), 1):
                    pid, path, err = await coro
                    if path is not None:
                        conn.execute(
                            "UPDATE panos SET image_path=?, downloaded_at=?, error=NULL WHERE pano_id=?",
                            (path, time.time(), pid),
                        )
                    else:
                        conn.execute("UPDATE panos SET error=? WHERE pano_id=?", (err, pid))
                    if i % 100 == 0:
                        conn.commit()
            conn.commit()
