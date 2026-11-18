from __future__ import annotations

import json
import sys
from typing import Any

import pytest

from vectorview import __version__, cli
from vectorview.config import Config
from vectorview.ingest import IngestStats
from vectorview.web import db as web_db


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, Any]:
    code = cli.main(list(argv))
    out = capsys.readouterr().out.strip()
    return code, json.loads(out) if out else None


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert __version__ in capsys.readouterr().out


def test_bad_bbox(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["ingest", "--name", "x", "--bbox", "1,2,3"])
    assert "west,south,east,north" in capsys.readouterr().err


def test_ingest(cfg: Config, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_ingest(c: Config, name: str, bbox: Any, client: Any, **kw: Any) -> IngestStats:
        seen.update(name=name, bbox=bbox, slots=client.slots, **kw)
        return IngestStats(region=name, images_kept=3)

    monkeypatch.setattr("vectorview.ingest.ingest_region", fake_ingest)
    code, out = run(capsys, "ingest", "--name", "a", "--bbox", "0,0,1,1", "--max-images", "10")
    assert code == 0 and out["images_kept"] == 3
    assert seen["slots"] == ["key_1", "key_2"] and seen["max_images"] == 10 and str(seen["bbox"]) == "0.0,0.0,1.0,1.0"


def test_errors_exit_1(cfg: Config, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GSV_API_KEY_1")
    monkeypatch.delenv("GSV_API_KEY_2")
    code, out = run(capsys, "ingest", "--name", "a", "--bbox", "0,0,1,1")
    assert code == 1 and out is None


def test_encode(cfg: Config, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("vectorview.encode.encode_images", lambda c, **kw: 7)
    monkeypatch.setattr("vectorview.encode.encode_tiles", lambda c, **kw: 28 if kw["scales"] == (2, 3) else -1)
    assert run(capsys, "encode")[1] == {"written": 7}
    assert run(capsys, "encode", "--tiles", "--scales", "2,3", "--overlaps", "0,0.5")[1] == {"written": 28}


def test_export_registers_a_scan(
    cfg: Config, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("vectorview.cache.export_corpus", lambda c, tag, region=None: (5, 20))
    code, out = run(capsys, "export", "--tag", "corp")
    assert code == 0 and out["images"] == 5 and out["tiles"] == 20
    scan = web_db.get_scan(out["scan_id"])
    assert (
        scan is not None
        and scan["fetch_mode"] == "corpus"
        and scan["status"] == "ready"
        and scan["tile_tag"] == "corp_tiles"
    )
    assert "scan_id" not in run(capsys, "export", "--tag", "corp2", "--no-register")[1]


def test_serve(cfg: Config, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Any] = []
    monkeypatch.setattr("vectorview.web.app.serve", lambda *a, **k: calls.append((a, k)))
    assert run(capsys, "serve", "--port", "9000")[0] == 0
    assert calls == [(("127.0.0.1", 9000), {"reload": False, "log_level": "info"})]


def test_quota(cfg: Config, capsys: pytest.CaptureFixture[str]) -> None:
    from vectorview.gsv import QuotaLedger

    QuotaLedger(cfg.quota_state_path, cfg.gsv_caps).reserve("key_1", 2)
    assert run(capsys, "quota")[1] == {"key_1": {"used": 2, "cap": 4}, "key_2": {"used": 0, "cap": 100}}


class FakeScraper:
    calls: list[tuple[str, Any]] = []

    def __init__(self, data_dir: str, *, place: Any, polygon: Any, concurrency: int, gpu: bool) -> None:
        self.calls.append(("init", (place, polygon.bounds if polygon is not None else None, gpu)))

    def __getattr__(self, name: str) -> Any:
        return lambda *a, **k: self.calls.append((name, k))

    def stats(self) -> dict[str, int]:
        return {"panos_discovered": 1}


@pytest.mark.parametrize(
    ("argv", "step"),
    [
        (["--place", "Somewhere", "discover"], "discover"),
        (["--bbox", "0,0,1,1", "discover"], "discover"),
        (["enrich"], "enrich"),
        (["download", "--zoom", "2"], "download"),
        (["project", "--headings", "0,180", "--cpu"], "project"),
        (["export", "--out", "x.csv"], "export"),
        (["stats"], None),
    ],
)
def test_scrape(
    cfg: Config, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, argv: list[str], step: str | None
) -> None:
    monkeypatch.setattr("vectorview.scraper.GSVScraper", FakeScraper)
    FakeScraper.calls = []
    code, out = run(capsys, "scrape", "--data-dir", "d", *argv)
    assert code == 0 and out == {"panos_discovered": 1}
    names = [c[0] for c in FakeScraper.calls]
    assert names[0] == "init" and (step is None or step in names)
    if step == "project":
        assert FakeScraper.calls[0][1][2] is False and FakeScraper.calls[1][1]["headings"] == (0.0, 180.0)


def test_scrape_discover_needs_an_area(
    cfg: Config, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("vectorview.scraper.GSVScraper", FakeScraper)
    assert run(capsys, "scrape", "--data-dir", "d", "discover")[0] == 1


def test_module_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    import runpy

    monkeypatch.setattr(sys, "argv", ["vectorview", "--version"])
    with pytest.raises(SystemExit):
        runpy.run_module("vectorview", run_name="__main__")
