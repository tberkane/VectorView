from pathlib import Path

import numpy as np
import pytest

from vectorview import cache


def test_image_cache_roundtrip(tmp_path: Path) -> None:
    feats = np.eye(3, 4, dtype=np.float32)
    cache.write_images(tmp_path, "t", feats, ["a", "b", "c"], ["r", "r", "s"])
    f, ids, regions = cache.load_images(tmp_path, "t")
    assert np.array_equal(f, feats) and ids == ["a", "b", "c"] and regions == ["r", "r", "s"]
    (tmp_path / "t" / "regions.txt").unlink()
    assert cache.load_images(tmp_path, "t")[2] == ["", "", ""]


def test_empty_cache(tmp_path: Path) -> None:
    cache.write_tiles(tmp_path, "e", np.empty((0, 4), np.float32), [], [])
    f, tids, parents = cache.load_tiles(tmp_path, "e")
    assert f.shape == (0, 4) and tids == [] and parents == []


def test_mismatched_rows_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        cache.write_images(tmp_path, "t", np.zeros((2, 4)), ["a"], ["r"])
    with pytest.raises(RuntimeError):
        cache.write_tiles(tmp_path, "t", np.zeros((2, 4)), ["a"], ["p", "q"])
    with pytest.raises(RuntimeError):
        cache.write_patch_regions(tmp_path, "t", np.zeros((2, 4)), ["a", "b"])
    cache.write_images(tmp_path, "t", np.zeros((2, 4)), ["a", "b"], ["r", "r"])
    (tmp_path / "t" / "image_ids.txt").write_text("a")
    with pytest.raises(RuntimeError, match="mismatch"):
        cache.load_images(tmp_path, "t")
    cache.write_tiles(tmp_path, "u", np.zeros((2, 4)), ["a1", "b1"], ["a", "b"])
    (tmp_path / "u" / "tile_ids.txt").write_text("a1")
    with pytest.raises(RuntimeError):
        cache.load_tiles(tmp_path, "u")


def test_patch_regions_align_and_zero_fill(tmp_path: Path) -> None:
    feats = np.arange(2 * 4 * 3, dtype=np.float16).reshape(2, 4, 3)
    cache.write_patch_regions(tmp_path, "p", feats, ["a", "b"])
    out = cache.load_patch_regions(tmp_path, "p", ["b", "missing", "a"])
    assert out.shape == (3, 4, 3) and out.dtype == np.float32
    assert np.array_equal(out[0], feats[1]) and np.array_equal(out[2], feats[0]) and not out[1].any()
