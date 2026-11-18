from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from vectorview.config import Config
from vectorview.web import app as app_mod
from vectorview.web import db, vlm_judge
from vectorview.web.llm_expand import Expansion

EX = Expansion(["a tire"], [], [], "A loose tire.", "")


@pytest.fixture
def client(web: Config) -> Iterator[TestClient]:
    with TestClient(app_mod.create_app(web, start_worker=False)) as c:
        yield c


def ready_query(scan_id: int, n: int = 3) -> int:
    qid = db.create_query(scan_id=scan_id, text="tire", k=n, vlm_model="gemini-x", min_date=None)
    results = [
        {
            "rank": i + 1,
            "image_id": f"pano{i}_h{90 * (i % 2):03d}",
            "lat": 42.0,
            "lon": -71.0,
            "heading": 0.0,
            "pano_id": f"pano{i}",
            "vlm_decision": "yes",
            "vlm_confidence": 0.5,
            "vlm_score": 0.5,
            "stage1_score": 0.1,
            "vlm_reason": "r",
            "bbox": [100, 100, 500, 500],
            "captured_at": "",
            "region": "north",
        }
        for i in range(n)
    ]
    db.update_query(qid, status="ready", expansion=EX.to_dict(), results=results, ready=True)
    return qid


def test_pages(client: TestClient, corpus_scan: dict[str, Any]) -> None:
    sid = corpus_scan["id"]
    assert "corp" in client.get("/").text
    assert client.get("/scans/new").status_code == 200
    assert client.get(f"/scans/{sid}").status_code == 200
    assert client.get(f"/scans/{sid}/status").status_code == 200
    assert client.get("/scans/999").status_code == 404
    assert client.get("/static/app.css").status_code == 200


def test_empty_index(client: TestClient) -> None:
    assert "No scans yet" in client.get("/").text


def test_scan_preview_and_create(client: TestClient) -> None:
    form = {"bbox_n": 42.37, "bbox_s": 42.36, "bbox_e": -71.05, "bbox_w": -71.06}
    api = client.post("/scans/preview", data=form).json()
    scr = client.post("/scans/preview", data={**form, "fetch_mode": "scraper"}).json()
    assert api["est_cost_usd"] > 0 and scr["est_cost_usd"] == 0 and scr["est_images"] > api["est_images"]
    assert client.post("/scans/preview", data={**form, "bbox_n": 0}).status_code == 400
    r = client.post("/scans", data={**form, "name": "here"}, follow_redirects=False)
    assert r.status_code == 303
    sid = int(r.headers["location"].rsplit("/", 1)[1])
    assert db.get_scan(sid)["status"] == "pending"  # type: ignore[index]
    assert "waiting for the worker" in client.get(f"/scans/{sid}").text
    assert client.post("/scans", data={**form, "name": " "}).status_code == 400
    assert client.post("/scans", data={**form, "name": "x", "fetch_mode": "ftp"}).status_code == 400
    assert client.post("/scans", data={**form, "name": "x", "spacing_m": 1}).status_code == 400


def test_query_flow(client: TestClient, corpus_scan: dict[str, Any]) -> None:
    sid = corpus_scan["id"]
    r = client.post(f"/scans/{sid}/queries", data={"text": " loose tire ", "k": 5}, follow_redirects=False)
    assert r.status_code == 303
    qid = int(r.headers["location"].split("q=")[1])
    assert db.get_query(qid)["text"] == "loose tire"  # type: ignore[index]
    assert client.get(f"/queries/{qid}/status").status_code == 200  # still pending: keep polling
    assert client.get(f"/scans/{sid}?q={qid}").status_code == 200
    assert client.post(f"/scans/{sid}/queries", data={"text": "   "}).status_code == 400
    assert client.post(f"/scans/{sid}/queries", data={"text": "x", "k": 0}).status_code == 400
    assert client.post("/scans/999/queries", data={"text": "x"}).status_code == 404
    pending = db.create_scan(name="p", bbox=None, embedding_tag="", tile_tag="")
    assert client.post(f"/scans/{pending}/queries", data={"text": "x"}).status_code == 400
    assert client.get(f"/scans/{pending}?q={qid}").status_code == 200  # query of another scan is ignored

    done = ready_query(sid)
    assert client.get(f"/queries/{done}/status").status_code == 286  # tells HTMX to stop polling
    page = client.get(f"/queries/{done}/results").text
    assert "pano0_h000" in page
    assert client.get("/queries/999/results").status_code == 404


def test_verify_and_refine(client: TestClient, corpus_scan: dict[str, Any], web: Config) -> None:
    qid = ready_query(corpus_scan["id"])
    assert client.post(f"/queries/{qid}/refine").status_code == 400  # nothing marked yes yet
    assert client.get(f"/queries/{qid}/verify").status_code == 200
    card = client.post(f"/queries/{qid}/verifications", data={"cursor": 0, "image_id": "pano0_h000", "verdict": "yes"})
    assert card.status_code == 200
    assert (
        client.post(
            f"/queries/{qid}/verifications", data={"cursor": 0, "image_id": "x", "verdict": "maybe"}
        ).status_code
        == 400
    )
    assert client.get(f"/queries/{qid}/verify").status_code == 200  # resumes at the first unverified result
    assert client.get(f"/queries/{qid}/verify/card", params={"cursor": 99}).status_code == 200  # past the end
    assert client.get(f"/queries/{qid}/verify/card", params={"cursor": -3}).status_code == 200

    prompt = vlm_judge.build_prompt("tire", EX.vlm_description, EX.vlm_exclusions)
    vlm_judge.append_cache(
        web.vlm_cache_path,
        [{"prompt_hash": vlm_judge.prompt_hash(prompt), "image_id": "pano0_h000", "model": "gemini-x"}],
    )
    cost = client.get(f"/queries/{qid}/refine/cost").json()
    assert cost["n_parent_cached"] == 1 and cost["worst_cost_usd"] >= cost["est_cost_usd"]
    r = client.post(f"/queries/{qid}/refine", follow_redirects=False)
    assert r.status_code == 303
    child = db.get_query(int(r.headers["location"].split("q=")[1]))
    assert child is not None and child["parent_query_id"] == qid

    fresh = db.create_query(scan_id=corpus_scan["id"], text="t", k=1, vlm_model="m", min_date=None)
    assert client.get(f"/queries/{fresh}/refine/cost").status_code == 400
    assert client.get(f"/queries/{fresh}/verify/card").status_code == 400
    assert client.get("/queries/999/refine/cost").status_code == 404


def test_images(client: TestClient, corpus_scan: dict[str, Any]) -> None:
    r = client.get("/cache/pano0_h000.jpg")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert client.get("/cache/nothere.jpg").status_code == 404
    assert client.get("/cache/..%5Csecret.jpg").status_code == 400


def test_lifespan_requeues_and_runs_the_worker(web: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    sid = db.create_scan(name="s", bbox=None, embedding_tag="", tile_tag="", status="encoding")
    started = []

    async def fake_loop(stop: Any) -> None:
        started.append(True)
        await stop.wait()

    monkeypatch.setattr(app_mod, "worker_loop", fake_loop)
    with TestClient(app_mod.create_app(web)) as c:
        assert c.get("/").status_code == 200
    assert started == [True] and db.get_scan(sid)["status"] == "pending"  # type: ignore[index]


def test_worker_that_ignores_stop_is_cancelled(web: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    async def stubborn(stop: Any) -> None:
        await asyncio.sleep(3600)

    monkeypatch.setattr(app_mod, "worker_loop", stubborn)
    monkeypatch.setattr(asyncio, "wait_for", _instant_timeout)
    with TestClient(app_mod.create_app(web)):
        pass


async def _instant_timeout(aw: Any, timeout: float) -> None:
    aw.close() if hasattr(aw, "close") else None
    raise TimeoutError


def test_default_config_and_serve(monkeypatch: pytest.MonkeyPatch, cfg: Config) -> None:
    assert app_mod.create_app().state.cfg.data_dir == cfg.data_dir
    calls: list[dict[str, Any]] = []
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: calls.append({"args": a, **k}))
    app_mod.serve(port=9999)
    assert calls[0]["port"] == 9999 and calls[0]["factory"] is True and calls[0]["host"] == "127.0.0.1"
    assert json.dumps(calls[0]["args"]) == '["vectorview.web.app:create_app"]'
