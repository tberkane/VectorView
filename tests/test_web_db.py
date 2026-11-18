import pytest

from vectorview.config import Config
from vectorview.web import db


def test_requires_init(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(db, "_db_path", None)
    with pytest.raises(RuntimeError, match="init"):
        db.connect()


def test_scan_lifecycle(web: Config) -> None:
    corpus = db.create_scan(
        name="c", bbox=None, embedding_tag="t", tile_tag="t_tiles", fetch_mode="corpus", status="ready"
    )
    sid = db.create_scan(name="s", bbox=(1.0, 0.0, 1.0, 0.0), embedding_tag="", tile_tag="")
    assert [s["id"] for s in db.list_scans()] == [corpus, sid]
    assert db.get_scan(corpus)["ready_at"] is not None  # type: ignore[index]
    assert db.next_pending_scan()["id"] == sid  # type: ignore[index]
    db.update_scan(sid, status="encoding", n_encoded=3, error=None)
    db.update_scan(sid)  # nothing to change
    assert db.get_scan(sid)["n_encoded"] == 3  # type: ignore[index]
    assert db.next_pending_scan() is None
    assert db.reset_orphaned_jobs() == (0, 1)
    db.update_scan(sid, status="ready", ready=True)
    assert db.get_scan(sid)["ready_at"]  # type: ignore[index]
    assert db.get_scan(999) is None
    with pytest.raises(ValueError):
        db.create_scan(name="x", bbox=None, embedding_tag="", tile_tag="", fetch_mode="ftp")


def test_queries_and_verifications(web: Config) -> None:
    sid = db.create_scan(name="s", bbox=None, embedding_tag="t", tile_tag="t", status="ready", fetch_mode="corpus")
    q1 = db.create_query(scan_id=sid, text="tire", k=5, vlm_model="m", min_date=None)
    q2 = db.create_query(scan_id=sid, text="tire", k=5, vlm_model="m", min_date="2020-01-01", parent_query_id=q1)
    assert db.next_pending_query()["id"] == q1  # type: ignore[index]
    assert db.find_cached_expansion("tire") is None
    db.update_query(q1, status="expanding", expansion={"paraphrases": ["a"]})
    db.update_query(q1)
    assert db.find_cached_expansion("tire") == {"paraphrases": ["a"]}
    assert db.reset_orphaned_jobs() == (1, 0)
    db.update_query(q1, status="ready", results=[{"image_id": "x"}], ready=True)
    db.update_query(q2, status="failed", error="boom")
    assert [q["id"] for q in db.list_queries(sid)] == [q2, q1]
    assert db.get_query(q2)["error"] == "boom"  # type: ignore[index]
    db.record_verification(q1, "a", "yes")
    db.record_verification(q1, "b", "no")
    db.record_verification(q1, "a", "unsure")
    assert db.list_verifications(q1) == {"a": "unsure", "b": "no"}
    assert db.verification_stats(q1) == {"yes": 0, "no": 1, "unsure": 1}
    assert db.now_iso().endswith("Z")
