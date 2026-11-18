from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from vectorview import cache
from vectorview.config import Config
from vectorview.web import retrieval
from vectorview.web.llm_expand import Expansion
from vectorview.web.retrieval import (
    ScanCache,
    apply_relevance_feedback,
    build_query_vector,
    fused_scores,
    l2,
    stage1_rank,
)

from .conftest import FakeEncoder

EX = Expansion(["a tire", "old tire"], ["a car"], ["by a curb"], "desc", "excl")
vecs = arrays(np.float32, (4,), elements=st.floats(-1, 1, width=32)).filter(lambda v: float(np.linalg.norm(v)) > 1e-3)


def small_cache(n: int = 5, n_tiles: int = 2, patches: bool = False) -> ScanCache:
    rng = np.random.default_rng(0)
    feats = np.stack([l2(v) for v in rng.standard_normal((n, 4)).astype(np.float32)])
    tiles = (
        np.stack([l2(v) for v in rng.standard_normal((n * n_tiles, 4)).astype(np.float32)])
        if n_tiles
        else np.empty((0, 4), np.float32)
    )
    parent = np.repeat(np.arange(n), n_tiles)
    pf = rng.standard_normal((n, 3, 4)).astype(np.float32) if patches else None
    if pf is not None:
        pf[0] = 0  # image 0 has no patch regions
    return ScanCache(feats, [f"i{k}" for k in range(n)], tiles, parent, {}, pf)


@given(vecs)
def test_l2_is_unit(v: np.ndarray) -> None:
    assert np.isclose(np.linalg.norm(l2(v)), 1.0, atol=1e-5)
    assert np.array_equal(l2(np.zeros(3)), np.zeros(3))


def test_query_vector_is_unit_and_moves_away_from_negatives(fake_encoder: FakeEncoder) -> None:
    q = build_query_vector(fake_encoder, EX)
    q0 = build_query_vector(fake_encoder, EX, alpha_n=0.0)
    neg = fake_encoder.encode_texts(["a car"])[0]
    assert np.isclose(np.linalg.norm(q), 1.0) and np.isclose(np.linalg.norm(q0), 1.0)
    assert q @ neg < q0 @ neg
    no_neg = Expansion(["x"], [], [], "d", "e")
    assert np.allclose(build_query_vector(fake_encoder, no_neg), fake_encoder.encode_texts(["x"])[0])
    with pytest.raises(ValueError):
        build_query_vector(fake_encoder, Expansion([], ["n"], [], "d", "e"))


@given(st.lists(st.integers(0, 4), max_size=5), st.lists(st.integers(0, 4), max_size=8), vecs)
def test_relevance_feedback_stays_unit(yes: list[int], no: list[int], q: np.ndarray) -> None:
    sc = small_cache()
    out = apply_relevance_feedback(l2(q), sc, [f"i{i}" for i in yes] + ["unknown"], [f"i{i}" for i in no])
    assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-5)
    if not yes:
        assert np.array_equal(out, l2(q))


def test_relevance_feedback_pulls_toward_positives() -> None:
    sc = small_cache()
    q = l2(np.ones(4, np.float32))
    out = apply_relevance_feedback(q, sc, ["i1"], [])
    assert out @ sc.image_feats[1] > q @ sc.image_feats[1]
    assert np.array_equal(apply_relevance_feedback(q, sc, ["i1"], [], alpha=1.0), q)
    many_no = [f"i{k}" for k in (0, 2, 3, 4)] * 2
    gated = apply_relevance_feedback(q, sc, ["i1"], many_no[:4], min_neg=5)
    pushed = apply_relevance_feedback(q, sc, ["i1"], many_no, min_neg=5)
    assert not np.allclose(gated, pushed)


@given(vecs, st.booleans(), st.integers(0, 3))
def test_stage1_rank_is_sorted(q: np.ndarray, patches: bool, n_tiles: int) -> None:
    sc = small_cache(patches=patches, n_tiles=n_tiles)
    ranked = stage1_rank(l2(q), sc, top_k=3)
    scores = [s for _, s in ranked]
    assert len(ranked) == 3 and scores == sorted(scores, reverse=True)


def test_fusion_modes() -> None:
    sc = small_cache(n_tiles=0)
    q = l2(np.ones(4, np.float32))
    assert np.allclose(fused_scores(q, sc), sc.image_feats @ q)  # no tiles: image score only
    sc2 = small_cache(patches=True)
    s = fused_scores(q, sc2)
    img = sc2.image_feats @ q
    assert sc2.patch_feats is not None
    best_tile = (sc2.tile_feats @ q).reshape(5, 2).max(axis=1)
    expected0 = 0.55 * img[0] + 0.30 * best_tile[0] + 0.15 * img[0]  # missing patches fall back to the image score
    assert np.isclose(s[0], expected0)


def test_scan_cache_loading_filters_and_dates(
    corpus_scan: dict[str, Any], web: Config, fake_encoder: FakeEncoder
) -> None:
    sc = retrieval.get_scan_cache(corpus_scan, web.embeddings_dir)
    assert retrieval.get_scan_cache(corpus_scan, web.embeddings_dir) is sc
    assert len(sc.image_ids) == 6 and len(sc.tile_feats) == 20 and sc.patch_feats is None
    north = retrieval.load_scan_cache(
        web.embeddings_dir, embedding_tag="corp", tile_tag="corp_tiles", region_filter="north"
    )
    assert len(north.image_ids) == 4 and len(north.parent_idx) == 16 and north.parent_idx.max() == 3
    q = build_query_vector(fake_encoder, EX)
    dated = [iid for iid, _ in stage1_rank(q, sc, min_date="2023-01-01")]
    assert set(dated) == {"pano3_h090", "pano4_h000", "pano5_h090"}  # pano5 has no date and is kept
    sc.manifest.pop("pano4_h000")
    assert "pano4_h000" not in [iid for iid, _ in stage1_rank(q, sc, min_date="2000-01-01")]
    cache.write_patch_regions(web.embeddings_dir, "corp_patches", np.ones((1, 4, 16), np.float16), ["pano0_h000"])
    with_patches = retrieval.load_scan_cache(
        web.embeddings_dir, embedding_tag="corp", tile_tag="corp_tiles", patch_region_tag="corp_patches"
    )
    assert with_patches.patch_feats is not None and with_patches.patch_feats.shape == (6, 4, 16)
