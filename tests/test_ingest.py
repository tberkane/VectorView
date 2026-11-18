"""Ingest orchestration with Street View and Postgres replaced by fakes."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from vectorview import ingest
from vectorview.config import Config
from vectorview.gsv import GSVImageError, GSVMetadataError, QuotaExhausted
from vectorview.regions import BBox


class FakeConn:
    closed = False

    def commit(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class FakeCorpus:
    """In-memory stand-in for the functions of vectorview.db that ingest uses."""

    def __init__(self) -> None:
        self.seen: dict[tuple[str, float, float], str] = {}
        self.panos: dict[str, dict[str, Any]] = {}
        self.images: dict[str, dict[str, Any]] = {}

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        @contextmanager
        def connect(cfg: Config) -> Iterator[FakeConn]:
            yield FakeConn()

        db = ingest.db
        monkeypatch.setattr(db, "connect", connect)
        monkeypatch.setattr(ingest.psycopg, "connect", lambda *a, **k: FakeConn())
        monkeypatch.setattr(db, "candidate_seen", lambda c, r, la, lo: self.seen.get((r, la, lo)))
        monkeypatch.setattr(
            db, "mark_candidate", lambda c, r, la, lo, b, status, p: self.seen.__setitem__((r, la, lo), status)
        )
        monkeypatch.setattr(db, "upsert_panorama", lambda c, **kw: self.panos.setdefault(kw["pano_id"], kw))
        monkeypatch.setattr(db, "upsert_image", lambda c, **kw: self.images.__setitem__(kw["image_id"], kw))
        monkeypatch.setattr(
            db,
            "region_state",
            lambda c, region: ({p for p, v in self.panos.items() if v["region"] == region}, len(self.images)),
        )


class FakeClient:
    """Street View client: candidate i maps to panorama P<i // 2> (so pairs are duplicates)."""

    def __init__(
        self, caps: dict[str, int], fail_meta: frozenset[int] = frozenset(), fail_img: frozenset[str] = frozenset()
    ) -> None:  # type: ignore[assignment]
        self.caps = dict(caps)
        self.used = dict.fromkeys(caps, 0)
        self.fail_meta = fail_meta
        self.fail_img = fail_img

    @property
    def slots(self) -> list[str]:
        return list(self.caps)

    def metadata(self, lat: float, lon: float, slot: str) -> dict[str, Any] | None:
        i = int(round(lat * 1000))
        if i in self.fail_meta:
            raise GSVMetadataError("boom")
        if i % 7 == 6:
            return None
        return {"pano_id": f"P{i // 2}", "lat": lat, "lon": lon, "date": date(2020, 1, 1), "copyright": "c"}

    def image(self, pano_id: str, heading: float, slot: str) -> bytes:
        if self.used[slot] >= self.caps[slot]:
            raise QuotaExhausted(slot)
        self.used[slot] += 1
        if pano_id in self.fail_img:
            raise GSVImageError("bad")
        return b"jpeg"


def candidates(n: int) -> list[tuple[float, float, float]]:
    return [(i / 1000, 0.0, 10.0) for i in range(n)]


@pytest.fixture
def corpus(monkeypatch: pytest.MonkeyPatch, cfg: Config) -> FakeCorpus:
    c = FakeCorpus()
    c.install(monkeypatch)
    monkeypatch.setattr(ingest.osm, "load_or_download_graph", lambda name, bbox, d: "graph")
    monkeypatch.setattr(ingest.osm, "sample_candidates", lambda g, **kw: candidates(30))
    return c


BOX = BBox(0, 0, 1, 1)


def test_fetch_views_writes_files_and_skips_failures(tmp_path: Path) -> None:
    client = FakeClient({"k": 10}, fail_img=frozenset({"BAD"}))
    views = ingest.fetch_views(client, "P1", 350.0, "k", tmp_path)  # type: ignore[arg-type]
    assert [v.image_id for v in views] == ["P1_h350", "P1_h170"]
    assert all(v.path.read_bytes() == b"jpeg" for v in views)
    assert ingest.fetch_views(client, "BAD", 0.0, "k", tmp_path) == []  # type: ignore[arg-type]


def test_switches_keys_when_one_hits_its_cap(corpus: FakeCorpus, cfg: Config) -> None:
    client = FakeClient({"key_1": 3, "key_2": 100}, fail_meta=frozenset({4}))
    stats = ingest.ingest_region(cfg, "area", BOX, client, workers=2)  # type: ignore[arg-type]
    assert stats.exhausted_slots == ["key_1"]
    assert {p["key_used"] for p in corpus.panos.values()} == {"key_1", "key_2"}
    # Every panorama seen has both views; the one cut short by the cap was retried with key_2.
    assert stats.images_kept == len(corpus.images) == 2 * len(corpus.panos)
    assert stats.metadata_errors == 1 and stats.no_coverage > 0 and stats.duplicate_pano > 0
    assert client.used["key_1"] == 3


def test_resumes_and_respects_max_images(corpus: FakeCorpus, cfg: Config) -> None:
    client = FakeClient({"key_1": 1000})
    first = ingest.ingest_region(cfg, "area", BOX, client, max_images=6, workers=1)  # type: ignore[arg-type]
    assert first.images_kept == 6
    again = ingest.ingest_region(cfg, "area", BOX, client, max_images=6, workers=1)  # type: ignore[arg-type]
    assert again.images_kept == 6 and client.used["key_1"] == 6
    full = ingest.ingest_region(cfg, "area", BOX, client, workers=3)  # type: ignore[arg-type]
    assert full.skipped_seen > 0 and full.exhausted_slots == []


def test_failed_images_and_worker_errors_are_counted(
    corpus: FakeCorpus, cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakeClient({"key_1": 1000}, fail_img=frozenset({"P0"}))
    original = ingest._Worker.__call__

    def flaky(self: ingest._Worker, cand: tuple[float, float, float], slot: str) -> dict[str, int]:
        if cand[0] == 0.002:
            raise RuntimeError("unexpected")
        return original(self, cand, slot)

    monkeypatch.setattr(ingest._Worker, "__call__", flaky)
    stats = ingest.ingest_region(cfg, "area", BOX, client, workers=1)  # type: ignore[arg-type]
    assert stats.image_errors == 1
    assert "P0" not in corpus.panos


def test_all_keys_exhausted(corpus: FakeCorpus, cfg: Config) -> None:
    client = FakeClient({"key_1": 2, "key_2": 2})
    stats = ingest.ingest_region(cfg, "area", BOX, client, workers=1)  # type: ignore[arg-type]
    assert stats.exhausted_slots == ["key_1", "key_2"] and stats.images_kept == 4
