"""Query pipeline, scan worker and worker loop, with every external service faked."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from vectorview.config import Config
from vectorview.gsv import QuotaExhausted
from vectorview.web import db, query_pipeline, scan_worker, worker
from vectorview.web.llm_expand import Expansion

from .conftest import FakeEncoder, jpeg_bytes, make_jpeg

EX = Expansion(["a tire"], ["a car"], ["by a curb"], "A loose tire.", "Mounted tires.")


class Judge:
    model = "fake"

    async def judge(self, image_bytes: bytes, prompt: str) -> dict[str, Any]:
        return {
            "decision": "yes",
            "confidence": len(image_bytes) % 100 / 100,
            "reason": "looks right",
            "bbox": [1, 2, 3, 4],
        }


@pytest.fixture
def services(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    async def expand(text: str) -> Expansion:
        calls.append(text)
        return EX

    monkeypatch.setattr(query_pipeline.llm_expand, "expand", expand)
    monkeypatch.setattr(query_pipeline.vlm_judge, "build_judge", lambda model: Judge())
    return calls


async def test_query_end_to_end_and_refine(corpus_scan: dict[str, Any], services: list[str]) -> None:
    sid = corpus_scan["id"]
    qid = db.create_query(scan_id=sid, text="loose tire", k=4, vlm_model="gemini-x", min_date=None)
    await query_pipeline.run_query(db.get_query(qid))  # type: ignore[arg-type]
    q = db.get_query(qid)
    assert q is not None and q["status"] == "ready", q
    results = json.loads(q["results_json"])
    assert len(results) == 4 and [r["rank"] for r in results] == [1, 2, 3, 4]
    assert {"pano_id", "lat", "lon", "bbox", "vlm_reason", "region"} <= set(results[0])
    assert services == ["loose tire"]

    # A repeated query reuses the stored expansion.
    q2 = db.create_query(scan_id=sid, text="loose tire", k=2, vlm_model="gemini-x", min_date="2022-06-01")
    await query_pipeline.run_query(db.get_query(q2))  # type: ignore[arg-type]
    assert services == ["loose tire"] and db.get_query(q2)["status"] == "ready"  # type: ignore[index]

    db.record_verification(qid, results[0]["image_id"], "yes")
    db.record_verification(qid, results[1]["image_id"], "no")
    child = db.create_query(
        scan_id=sid, text="loose tire", k=4, vlm_model="gemini-x", min_date=None, parent_query_id=qid
    )
    await query_pipeline.run_query(db.get_query(child))  # type: ignore[arg-type]
    assert db.get_query(child)["status"] == "ready"  # type: ignore[index]


async def test_query_failures_are_recorded(web: Config, services: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    sid = db.create_scan(name="s", bbox=None, embedding_tag="t", tile_tag="t", status="ready", fetch_mode="corpus")
    qid = db.create_query(scan_id=sid, text="x", k=1, vlm_model="m", min_date=None)
    monkeypatch.setattr(query_pipeline.db, "get_scan", lambda i: None)
    await query_pipeline.run_query(db.get_query(qid))  # type: ignore[arg-type]
    q = db.get_query(qid)
    assert q is not None and q["status"] == "failed" and "not found" in q["error"]


class FakeGSV:
    def __init__(self, caps: dict[str, int]) -> None:
        self.caps = caps
        self.used = dict.fromkeys(caps, 0)

    @property
    def slots(self) -> list[str]:
        return list(self.caps)

    def __enter__(self) -> FakeGSV:
        return self

    def __exit__(self, *a: object) -> None:
        return None

    def metadata(self, lat: float, lon: float, slot: str) -> dict[str, Any] | None:
        i = round(lat * 1000)
        if i == 3:
            from vectorview.gsv import GSVMetadataError

            raise GSVMetadataError("x")
        return {
            "pano_id": f"P{i // 2}",
            "lat": lat,
            "lon": lon,
            "date": date(2021, 1, 1) if i % 2 else None,
            "copyright": None,
        }

    def image(self, pano_id: str, heading: float, slot: str) -> bytes:
        if self.used[slot] >= self.caps[slot]:
            raise QuotaExhausted(slot)
        self.used[slot] += 1
        return jpeg_bytes(hash((pano_id, heading)) % 1000)


def api_scan(spacing: int = 50) -> dict[str, Any]:
    sid = db.create_scan(name="s", bbox=(0.01, 0.0, 0.01, 0.0), spacing_m=spacing, embedding_tag="", tile_tag="")
    scan = db.get_scan(sid)
    assert scan is not None
    return scan


@pytest.fixture
def roads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scan_worker.osm, "load_or_download_graph", lambda *a: "graph")
    monkeypatch.setattr(scan_worker.osm, "sample_candidates", lambda g, **kw: [(i / 1000, 0.0, 0.0) for i in range(12)])


def test_api_scan_builds_all_caches(web: Config, fake_encoder: FakeEncoder, roads: None) -> None:
    scan = api_scan()
    client = FakeGSV({"key_1": 3, "key_2": 100})
    rows = scan_worker.ingest_api(scan, web, client)  # type: ignore[arg-type]
    assert client.used["key_1"] == 3 and len(rows) == 2 * len({r.pano_id for r in rows})
    kept = scan_worker.encode_scan_images(scan["id"], rows, web)
    scan_worker.encode_scan_tiles(scan["id"], kept, web)
    scan_worker.encode_scan_patches(scan["id"], kept, web)
    scan_worker.finalize(scan["id"])
    done = db.get_scan(scan["id"])
    assert done is not None and done["status"] == "ready" and done["patch_region_tag"] == f"scan_{scan['id']}_patches"
    from vectorview.web import retrieval

    sc = retrieval.get_scan_cache(done, web.embeddings_dir)
    assert len(sc.image_ids) == len(rows) and sc.patch_feats is not None


def test_api_scan_errors(web: Config, fake_encoder: FakeEncoder, roads: None, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RuntimeError, match="every API key"):
        scan_worker.ingest_api(api_scan(), web, FakeGSV({"key_1": 1}))  # type: ignore[arg-type]
    monkeypatch.setattr(scan_worker.osm, "sample_candidates", lambda g, **kw: [])
    with pytest.raises(RuntimeError, match="no image"):
        scan_worker.ingest_api(api_scan(), web, FakeGSV({"key_1": 1}))  # type: ignore[arg-type]
    no_box = db.create_scan(name="x", bbox=None, embedding_tag="", tile_tag="")
    with pytest.raises(RuntimeError, match="bounding box"):
        scan_worker.scan_bbox(db.get_scan(no_box))  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="no image could be read"):
        scan_worker.encode_scan_images(1, [scan_worker.ImageMeta("a", "p", 0, 0, "", 0, "/missing.jpg", "r")], web)


def test_scraper_scan(web: Config, fake_encoder: FakeEncoder, monkeypatch: pytest.MonkeyPatch) -> None:
    steps: list[str] = []

    class FakeScraper:
        def __init__(self, workdir: Path, polygon: Any, concurrency: int) -> None:
            self.workdir = workdir
            workdir.mkdir(parents=True, exist_ok=True)
            assert polygon.bounds == (0.0, 0.0, 0.01, 0.01)

        def discover(self) -> None:
            steps.append("discover")
            with closing(sqlite3.connect(self.workdir / "state.db")) as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS panos "
                    "(pano_id TEXT, lat REAL, lon REAL, date TEXT, downloaded_at REAL, image_path TEXT)"
                )
                conn.executemany(
                    "INSERT OR IGNORE INTO panos VALUES (?, 1.0, 2.0, ?, 1, 'x')", [("A", "2020-05"), ("B", None)]
                )
                conn.commit()

        def enrich(self) -> None:
            steps.append("enrich")

        def download(self, zoom: int) -> None:
            steps.append("download")

        def project(self, **kw: Any) -> None:
            steps.append("project")
            for pid in ("A", "B"):
                for h in kw["headings"]:
                    if (pid, h) != ("B", 270.0):
                        make_jpeg(self.workdir / "crops" / f"{pid}_h{int(h):03d}.jpg", seed=int(h))

    import vectorview.scraper

    monkeypatch.setattr(vectorview.scraper, "GSVScraper", FakeScraper)
    scan = db.get_scan(
        db.create_scan(name="s", bbox=(0.01, 0.0, 0.01, 0.0), embedding_tag="", tile_tag="", fetch_mode="scraper")
    )
    rows = scan_worker.ingest_scraper(scan, web)  # type: ignore[arg-type]
    assert steps == ["discover", "enrich", "download", "project"]
    assert len(rows) == 7 and all(Path(r.cache_path).exists() for r in rows)
    assert {r.captured_at for r in rows} == {"2020-05", ""}
    monkeypatch.setattr(scan_worker.os, "link", _no_link)
    for r in rows:
        Path(r.cache_path).unlink()
    again = scan_worker.scraper_manifest(scan["id"], web.scrape_dir / f"scan_{scan['id']}", web.cache_dir)  # type: ignore[index]
    assert len(again) == 7 and all(Path(r.cache_path).exists() for r in again)  # copied when linking fails

    monkeypatch.setattr(scan_worker, "scraper_manifest", lambda *a: [])
    with pytest.raises(RuntimeError, match="no crops"):
        scan_worker.ingest_scraper(scan, web)  # type: ignore[arg-type]


async def test_run_scan_and_worker_loop(
    web: Config, fake_encoder: FakeEncoder, roads: None, monkeypatch: pytest.MonkeyPatch, services: list[str]
) -> None:
    monkeypatch.setattr(scan_worker.gsv, "open_client", lambda cfg: FakeGSV({"key_1": 100}))
    scan = api_scan()
    assert await worker.run_next() is True
    assert db.get_scan(scan["id"])["status"] == "ready"  # type: ignore[index]
    qid = db.create_query(scan_id=scan["id"], text="tire", k=3, vlm_model="g", min_date=None)
    assert await worker.run_next() is True
    assert db.get_query(qid)["status"] == "ready"  # type: ignore[index]
    assert await worker.run_next() is False

    bad = db.create_scan(name="b", bbox=None, embedding_tag="", tile_tag="")
    await scan_worker.run_scan(db.get_scan(bad))  # type: ignore[arg-type]
    assert db.get_scan(bad)["status"] == "failed"  # type: ignore[index]
    db.update_scan(bad, fetch_mode="ftp")
    await scan_worker.run_scan(db.get_scan(bad))  # type: ignore[arg-type]
    assert "unknown fetch_mode" in db.get_scan(bad)["error"]  # type: ignore[index]

    stop = asyncio.Event()
    seen = 0

    async def flaky() -> bool:
        nonlocal seen
        seen += 1
        if seen == 1:
            raise RuntimeError("unexpected")
        if seen == 2:
            return True
        stop.set()
        return False

    monkeypatch.setattr(worker, "run_next", flaky)
    await asyncio.wait_for(worker.worker_loop(stop, poll_interval=0.01), timeout=5)
    assert seen == 3


def _no_link(src: object, dst: object) -> None:
    raise OSError("cross-device link")
