import json
import threading
from datetime import date
from pathlib import Path

import httpx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from vectorview.config import Config
from vectorview.gsv import (
    GSVClient,
    GSVImageError,
    GSVMetadataError,
    QuotaExhausted,
    QuotaLedger,
    cache_path_for,
    make_image_id,
    open_client,
    parse_date,
)


def test_ledger_reserves_up_to_the_cap(tmp_path: Path) -> None:
    ledger = QuotaLedger(tmp_path / "q.json", {"key_1": 3})
    assert ledger.caps == {"key_1": 3}
    assert [ledger.reserve("key_1") for _ in range(3)] == [1, 2, 3]
    with pytest.raises(QuotaExhausted):
        ledger.reserve("key_1")
    assert ledger.remaining("key_1") == 0
    assert json.loads((tmp_path / "q.json").read_text()) == {"key_1": 3}
    with pytest.raises(KeyError):
        ledger.reserve("key_9")


def test_ledger_survives_a_new_instance_and_a_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "q.json"
    QuotaLedger(path, {"key_1": 5}).reserve("key_1", 2)
    assert QuotaLedger(path, {"key_1": 5, "key_2": 1}).snapshot() == {"key_1": 2, "key_2": 0}
    path.write_text("{not json")
    assert QuotaLedger(path, {"key_1": 5}).snapshot() == {"key_1": 0}


@settings(max_examples=50, deadline=None)
@given(cap=st.integers(0, 20), requests=st.lists(st.integers(1, 5), max_size=20))
def test_ledger_never_exceeds_its_cap(tmp_path_factory: pytest.TempPathFactory, cap: int, requests: list[int]) -> None:
    ledger = QuotaLedger(tmp_path_factory.mktemp("q") / "q.json", {"k": cap})
    granted = 0
    for n in requests:
        try:
            ledger.reserve("k", n)
            granted += n
        except QuotaExhausted:
            pass
        assert ledger.snapshot()["k"] == granted <= cap


def test_ledger_is_safe_under_threads(tmp_path: Path) -> None:
    ledger = QuotaLedger(tmp_path / "q.json", {"k": 50})
    ok = []

    def grab() -> None:
        for _ in range(20):
            try:
                ledger.reserve("k")
                ok.append(1)
            except QuotaExhausted:
                pass

    threads = [threading.Thread(target=grab) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(ok) == 50 and ledger.snapshot()["k"] == 50


def client_with(handler, tmp_path: Path, cap: int = 10) -> GSVClient:  # type: ignore[no-untyped-def]
    ledger = QuotaLedger(tmp_path / "q.json", {"key_1": cap})
    return GSVClient({"key_1": "SECRET"}, ledger, transport=httpx.MockTransport(handler), sleep=lambda s: None)


def test_metadata_ok_and_no_coverage(tmp_path: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.params["key"] == "SECRET" and req.url.params["source"] == "outdoor"
        if req.url.params["location"].startswith("1.0"):
            return httpx.Response(200, json={"status": "ZERO_RESULTS"})
        body = {
            "status": "OK",
            "pano_id": "P1",
            "location": {"lat": 2.5, "lng": 3.5},
            "date": "2021-06",
            "copyright": "c",
        }
        return httpx.Response(200, json=body)

    with client_with(handler, tmp_path) as c:
        assert c.slots == ["key_1"]
        assert c.metadata(1.0, 1.0, "key_1") is None
        meta = c.metadata(2.0, 3.0, "key_1")
    assert meta == {"pano_id": "P1", "lat": 2.5, "lon": 3.5, "date": date(2021, 6, 1), "copyright": "c"}


@pytest.mark.parametrize(
    ("response", "outcome"),
    [
        (httpx.Response(400, text="bad"), None),
        (httpx.Response(403, text="denied"), GSVMetadataError),
        (httpx.Response(200, json={"status": "REQUEST_DENIED", "error_message": "no"}), GSVMetadataError),
    ],
)
def test_metadata_errors(tmp_path: Path, response: httpx.Response, outcome: type[Exception] | None) -> None:
    c = client_with(lambda req: response, tmp_path)
    if outcome is None:
        assert c.metadata(0, 0, "key_1") is None
    else:
        with pytest.raises(outcome):
            c.metadata(0, 0, "key_1")


def test_unknown_slot(tmp_path: Path) -> None:
    c = client_with(lambda req: httpx.Response(200), tmp_path)
    with pytest.raises(KeyError):
        c.metadata(0, 0, "key_2")


def test_image_reserves_before_the_request_and_retries(tmp_path: Path) -> None:
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req.url.params["heading"])
        if len(calls) == 1:
            return httpx.Response(503)
        if len(calls) == 2:
            raise httpx.ConnectTimeout("slow")
        return httpx.Response(200, content=b"JPEG", headers={"content-type": "image/jpeg"})

    sleeps: list[float] = []
    ledger = QuotaLedger(tmp_path / "q.json", {"key_1": 1})
    c = GSVClient({"key_1": "k"}, ledger, transport=httpx.MockTransport(handler), sleep=sleeps.append)
    assert c.image("P", 12.345, "key_1") == b"JPEG"
    assert calls == ["12.35"] * 3 and sleeps == [1.0, 2.0]
    with pytest.raises(QuotaExhausted):
        c.image("P", 0, "key_1")
    assert len(calls) == 3  # no request once the cap is reached


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(403, text="denied"),
        httpx.Response(204),
        httpx.Response(200, text="{}", headers={"content-type": "text/json"}),
    ],
)
def test_image_errors_still_count(tmp_path: Path, response: httpx.Response) -> None:
    c = client_with(lambda req: response, tmp_path)
    with pytest.raises(GSVImageError):
        c.image("P", 0, "key_1")
    assert c._ledger.snapshot()["key_1"] == 1


def test_gives_up_after_max_retries(tmp_path: Path) -> None:
    ledger = QuotaLedger(tmp_path / "q.json", {"key_1": 5})
    c = GSVClient(
        {"key_1": "k"},
        ledger,
        transport=httpx.MockTransport(lambda r: httpx.Response(500)),
        max_retries=2,
        sleep=lambda s: None,
    )
    with pytest.raises(RuntimeError, match="2 attempts"):
        c.metadata(0, 0, "key_1")


def test_helpers(tmp_path: Path, cfg: Config) -> None:
    assert parse_date("2020-01-02") == date(2020, 1, 2)
    assert parse_date(None) is None and parse_date("01/2020") is None
    assert make_image_id("abc", 359.6) == "abc_h000"
    assert make_image_id("abc", 90.2) == "abc_h090"
    assert cache_path_for(tmp_path, "abc_h090") == tmp_path / "ab" / "abc_h090.jpg"
    with open_client(cfg) as c:
        assert c.slots == ["key_1", "key_2"]


@given(st.text(st.characters(categories=["L", "N"]), min_size=2, max_size=20), st.floats(-720, 720))
def test_image_id_heading_is_in_range(pano: str, heading: float) -> None:
    h = int(make_image_id(pano, heading).rsplit("_h", 1)[1])
    assert 0 <= h < 360
