"""Real SQL against Postgres + pgvector. Skipped unless VECTORVIEW_TEST_PG_DSN is set (CI sets it)."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import psycopg
import pytest

from vectorview import cache, db, encode, ingest
from vectorview.config import Config
from vectorview.encoder import EMBED_DIM
from vectorview.regions import BBox
from vectorview.web.manifest import manifest_path, read_manifest

from .conftest import FakeEncoder, make_jpeg
from .test_ingest import FakeClient, candidates

pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(not os.environ.get("VECTORVIEW_TEST_PG_DSN"), reason="VECTORVIEW_TEST_PG_DSN is not set"),
]

INIT_SQL = Path(__file__).parents[1] / "db" / "init.sql"


@pytest.fixture
def pg(cfg: Config) -> Iterator[Config]:
    with psycopg.connect(cfg.pg_dsn, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS tiles, images, panoramas, candidates_seen CASCADE")
        conn.execute(INIT_SQL.read_text())
        conn.execute(INIT_SQL.read_text())  # init.sql is idempotent
    yield cfg


def test_corpus_functions(pg: Config) -> None:
    with db.connect(pg) as conn:
        assert db.candidate_seen(conn, "r", 1.0, 2.0) is None
        db.mark_candidate(conn, "r", 1.0, 2.0, 90.0, "no_coverage", None)
        db.mark_candidate(conn, "r", 1.0, 2.0, 91.0, "fetched", "P")
        assert db.candidate_seen(conn, "r", 1.0, 2.0) == {"status": "fetched", "pano_id": "P", "road_bearing": 91.0}
        for _ in range(2):
            db.upsert_panorama(
                conn,
                pano_id="P",
                region="r",
                lat=1,
                lon=2,
                captured_at=None,
                road_bearing=1,
                key_used="key_1",
                copyright_str=None,
            )
        db.upsert_image(
            conn,
            image_id="P_h000",
            pano_id="P",
            heading=0,
            pitch=0,
            fov=90,
            width=1,
            height=1,
            cache_path=None,
            region="r",
        )
        db.upsert_image(
            conn,
            image_id="P_h000",
            pano_id="P",
            heading=0,
            pitch=0,
            fov=90,
            width=640,
            height=640,
            cache_path="/x.jpg",
            region="r",
        )
        assert db.region_state(conn, "r") == ({"P"}, 1)
        assert db.count_images_per_key(conn) == {"key_1": 1}
        assert db.count_images_per_region(conn) == {"r": 1}
    with pytest.raises(RuntimeError), db.connect(pg) as conn:
        db.mark_candidate(conn, "r", 5.0, 5.0, 0.0, "fetched", None)
        raise RuntimeError("rolled back")
    with db.connect(pg) as conn:
        assert db.candidate_seen(conn, "r", 5.0, 5.0) is None


def test_ingest_encode_export(pg: Config, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ingest.osm, "load_or_download_graph", lambda *a: "g")
    monkeypatch.setattr(ingest.osm, "sample_candidates", lambda g, **kw: candidates(20))
    client = FakeClient({"key_1": 5, "key_2": 100})
    monkeypatch.setattr(client, "image", _real_jpeg(client.image))
    stats = ingest.ingest_region(pg, "area", BBox(0, 0, 1, 1), client, workers=3)  # type: ignore[arg-type]
    assert stats.exhausted_slots == ["key_1"] and stats.images_kept > 0
    again = ingest.ingest_region(pg, "area", BBox(0, 0, 1, 1), client, workers=3)  # type: ignore[arg-type]
    assert again.images_kept == stats.images_kept and again.skipped_seen > 0

    enc = FakeEncoder(dim=EMBED_DIM)
    with db.connect(pg) as conn:
        conn.execute(
            "UPDATE images SET cache_path = '/missing.jpg' WHERE image_id = (SELECT MIN(image_id) FROM images)"
        )
    n_img = encode.encode_images(pg, batch_size=4, encoder=enc)
    assert n_img == stats.images_kept - 1 and encode.encode_images(pg, encoder=enc) == 0
    assert encode.encode_images(pg, region="elsewhere", encoder=enc) == 0
    n_tiles = encode.encode_tiles(pg, scales=(2,), overlaps=(0.0,), batch_size=5, encoder=enc, region="area")
    assert n_tiles == 4 * n_img and encode.encode_tiles(pg, encoder=enc) == 0
    assert encode.encode_tiles(pg, encoder=enc, region="elsewhere") == 0

    images, tiles = cache.export_corpus(pg, "corpus", region="area")
    assert (images, tiles) == (n_img, n_tiles)
    feats, ids, regions = cache.load_images(pg.embeddings_dir, "corpus")
    assert feats.shape == (n_img, EMBED_DIM) and set(regions) == {"area"}
    assert np.allclose(np.linalg.norm(feats, axis=1), 1.0, atol=1e-5)
    meta = read_manifest(manifest_path(pg.embeddings_dir, "corpus"))
    assert set(meta) == set(ids) and all(m.captured_at == "2020-01-01" for m in meta.values())
    assert cache.load_tiles(pg.embeddings_dir, "corpus_tiles")[0].shape == (n_tiles, EMBED_DIM)


def test_export_needs_embeddings(pg: Config) -> None:
    with pytest.raises(RuntimeError, match="vectorview encode"):
        cache.export_corpus(pg, "empty")


def test_export_without_tiles(pg: Config, tmp_path: Path) -> None:
    from pgvector.psycopg import register_vector

    with db.connect(pg) as conn:
        register_vector(conn)
        db.upsert_panorama(
            conn,
            pano_id="P",
            region="r",
            lat=1,
            lon=2,
            captured_at=None,
            road_bearing=None,
            key_used="k",
            copyright_str=None,
        )
        db.upsert_image(
            conn,
            image_id="P_h000",
            pano_id="P",
            heading=0,
            pitch=0,
            fov=90,
            width=1,
            height=1,
            cache_path=None,
            region="r",
        )
        conn.execute("UPDATE images SET embedding = %s", (np.ones(EMBED_DIM, np.float32),))
    assert cache.export_corpus(pg, "notiles") == (1, 0)
    assert cache.load_tiles(pg.embeddings_dir, "notiles_tiles")[0].shape == (0, EMBED_DIM)
    assert read_manifest(manifest_path(pg.embeddings_dir, "notiles"))["P_h000"].captured_at == ""


def _real_jpeg(image):  # type: ignore[no-untyped-def]
    import io

    from PIL import Image

    def wrapped(pano_id: str, heading: float, slot: str) -> bytes:
        image(pano_id, heading, slot)
        buf = io.BytesIO()
        Image.new("RGB", (32, 32), (len(pano_id) * 20 % 255, int(heading) % 255, 0)).save(buf, "JPEG")
        return buf.getvalue()

    return wrapped


_ = make_jpeg
