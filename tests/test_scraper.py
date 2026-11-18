"""The scraper, with streetlevel, aiohttp and py360convert replaced by fake modules."""

from __future__ import annotations

import math
import sys
import types
from contextlib import closing
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from PIL import Image
from shapely.geometry import box

from vectorview import scraper
from vectorview.scraper import GSVScraper, lonlat_to_tile, yaw_for_heading

from .conftest import Obj


class FakeStreetview:
    """Two panoramas per coverage tile; ids by tile; one of them is outside the area."""

    def __init__(self) -> None:
        self.downloaded: list[str] = []

    async def get_coverage_tile_async(self, x: int, y: int, session: Any) -> list[Any]:
        if (x + y) % 5 == 0:
            raise OSError("tile failed")
        return [
            Obj(id=f"p{x}_{y}", lat=0.0005, lon=0.0005, heading=math.pi / 2, elevation=1.0),
            Obj(id=f"far{x}_{y}", lat=5.0, lon=5.0, heading=0.0, elevation=1.0),
        ]

    async def find_panorama_by_id_async(self, pid: str, session: Any) -> Any:
        if pid.endswith("1"):
            return None  # by-id lookup returns nothing: fall back to coordinates
        if pid.endswith("2"):
            raise OSError("network")
        return Obj(id=pid, date=date(2022, 5, 1), country_code="US", elevation=2.0, heading=math.pi / 2)

    async def find_panorama_async(self, lat: float, lon: float, session: Any) -> Any:
        return Obj(id="x", date=None, country_code="US", elevation=2.0, heading=0.0)

    async def download_panorama_async(self, pano: Any, path: str, zoom: int, session: Any) -> None:
        if pano.id == "x" and len(self.downloaded) > 100:
            raise OSError("never")
        Image.new("RGB", (64, 32), (10, 20, 30)).save(path)
        self.downloaded.append(path)


class FakeSession:
    async def __aenter__(self) -> FakeSession:
        return self

    async def __aexit__(self, *a: object) -> None:
        return None


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> FakeStreetview:
    sv = FakeStreetview()
    pkg = types.ModuleType("streetlevel")
    pkg.streetview = sv  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "streetlevel", pkg)
    aio = types.ModuleType("aiohttp")
    aio.ClientSession = lambda **kw: FakeSession()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "aiohttp", aio)
    p360 = types.ModuleType("py360convert")
    p360.e2p = lambda img, fov_deg, u_deg, v_deg, out_hw, mode: np.zeros((*out_hw, 3), np.uint8)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "py360convert", p360)
    monkeypatch.setattr(scraper, "install_pyexiv2_shim", lambda: False)
    return sv


@given(st.floats(-180, 179.99), st.floats(-85, 85), st.integers(0, 20))
def test_tile_index_in_range(lon: float, lat: float, z: int) -> None:
    x, y = lonlat_to_tile(lon, lat, z)
    assert 0 <= x < 2**z and 0 <= y < 2**z


@given(st.floats(0, 360), st.floats(-10, 10))
def test_yaw_in_range(h: float, pano: float) -> None:
    yaw = yaw_for_heading(h, pano)
    assert -180.0 <= yaw < 180.0
    assert math.isclose((yaw + math.degrees(pano) - h) % 360.0, 0.0, abs_tol=1e-6) or math.isclose(
        (yaw + math.degrees(pano) - h) % 360.0, 360.0, abs_tol=1e-6
    )


def test_full_run(tmp_path: Path, fakes: FakeStreetview, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scraper.mp, "Pool", lambda n: InlinePool())  # worker processes would not see the fakes
    s = GSVScraper(tmp_path, polygon=box(0, 0, 0.01, 0.01), concurrency=4, gpu=False)
    n = s.discover()
    assert n > 0 and s.discover() == n  # second run fetches nothing new
    s.enrich()
    s.download(zoom=2)
    s.download(zoom=2)  # resumes: already downloaded panoramas are skipped
    rows = s.downloaded()
    assert rows and len(s.downloaded(limit=1)) == 1
    s.project(headings=(0, 90), fov=90, pitch=0, size=16, workers=1)
    crops = sorted((tmp_path / "crops").glob("*.jpg"))
    assert len(crops) == 2 * len(rows)
    assert s.export(tmp_path / "out" / "panos.csv") == s.stats()["panos_discovered"]
    assert s.stats()["panos_errored"] >= 1


def test_project_one_reports_errors(tmp_path: Path, fakes: FakeStreetview) -> None:
    pid, err = scraper.project_one(("p", str(tmp_path / "missing.jpg"), 0.0, [0.0], 90.0, 0.0, 8, str(tmp_path)))
    assert pid == "p" and err is not None
    Image.new("RGB", (8, 4)).save(tmp_path / "e.jpg")
    assert scraper.project_one(("p", str(tmp_path / "e.jpg"), None, [0.0], 90.0, 0.0, 8, str(tmp_path))) == ("p", None)
    assert scraper.project_one(("p", str(tmp_path / "e.jpg"), None, [0.0], 90.0, 0.0, 8, str(tmp_path))) == ("p", None)


def test_project_uses_the_gpu_path_when_it_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Any] = []
    gpu = types.ModuleType("vectorview._gpu_project")

    def fake_gpu(*a: Any) -> bool:
        calls.append(a)
        return True

    gpu.project_gpu = fake_gpu  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vectorview._gpu_project", gpu)
    s = GSVScraper(tmp_path, polygon=box(0, 0, 1, 1))
    s.project()  # nothing downloaded: returns early
    with closing(s._connect()) as conn:
        conn.execute("INSERT INTO panos(pano_id, lat, lon, downloaded_at, image_path) VALUES ('a', 0, 0, 1, 'x.jpg')")
        conn.commit()
    s.project()
    assert len(calls) == 1
    gpu.project_gpu = lambda *a: (_ for _ in ()).throw(RuntimeError("cuda"))  # type: ignore[attr-defined]
    monkeypatch.setattr(scraper.mp, "Pool", lambda n: FakePool())
    s.project()  # falls back to the CPU path


class InlinePool:
    def __enter__(self) -> InlinePool:
        return self

    def __exit__(self, *a: object) -> None:
        return None

    def imap_unordered(self, fn: Any, work: list[Any], chunksize: int) -> Any:
        return map(fn, work)


class FakePool:
    def __enter__(self) -> FakePool:
        return self

    def __exit__(self, *a: object) -> None:
        return None

    def imap_unordered(self, fn: Any, work: list[Any], chunksize: int) -> Any:
        return iter([(w[0], "failed") for w in work])


def test_polygon_from_place(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import osmnx as ox

    gdf = Obj(geometry=Obj(iloc=[box(0, 0, 1, 1)]))
    monkeypatch.setattr(ox, "geocode_to_gdf", lambda place: gdf)
    assert GSVScraper(tmp_path, place="Somewhere").polygon().bounds == (0, 0, 1, 1)
    with pytest.raises(ValueError):
        GSVScraper(tmp_path).polygon()


def test_missing_extra_is_explained(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scraper, "install_pyexiv2_shim", lambda: False)
    monkeypatch.setitem(sys.modules, "streetlevel", None)
    with pytest.raises(RuntimeError, match="scraper extra"):
        scraper._streetview()


def test_pyexiv2_shim(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real = builtins.__import__

    def failing(exc: type[Exception]) -> Any:
        def imp(name: str, *a: Any, **k: Any) -> Any:
            if name == "pyexiv2":
                raise exc("libexiv2.so: cannot open")
            return real(name, *a, **k)

        return imp

    monkeypatch.delitem(sys.modules, "pyexiv2", raising=False)
    monkeypatch.setattr(builtins, "__import__", failing(ImportError))
    assert scraper.install_pyexiv2_shim() is False
    monkeypatch.setattr(builtins, "__import__", failing(OSError))
    assert scraper.install_pyexiv2_shim() is True
    shim = sys.modules["pyexiv2"]
    with shim.ImageData(b"abc") as img:  # type: ignore[attr-defined]
        img.modify_exif({})
        img.modify_xmp({})
        assert img.get_bytes() == b"abc"
    monkeypatch.setattr(builtins, "__import__", real)
    assert scraper.install_pyexiv2_shim() is False  # the shim now imports fine
