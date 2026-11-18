import pytest
from hypothesis import given
from hypothesis import strategies as st
from PIL import Image

from vectorview.tile import cut_tiles, tile_geometry


@given(st.integers(16, 2048), st.integers(1, 6), st.floats(0.0, 0.9))
def test_geometry_invariants(side: int, scale: int, overlap: float) -> None:
    size, starts = tile_geometry(side, scale, overlap)
    assert len(starts) == scale
    assert starts[0] == 0
    assert starts == sorted(starts)
    assert all(0 <= s and s + size <= side + 1 for s in starts)
    # Tiles cover the image up to rounding: the last tile ends within a pixel of the edge.
    assert side - (starts[-1] + size) <= 1


@pytest.mark.parametrize(
    ("scale", "overlap", "size", "starts"),
    [(1, 0.0, 640, [0]), (2, 0.0, 320, [0, 320]), (3, 0.5, 320, [0, 160, 320]), (4, 0.0, 160, [0, 160, 320, 480])],
)
def test_known_geometries(scale: int, overlap: float, size: int, starts: list[int]) -> None:
    assert tile_geometry(640, scale, overlap) == (size, starts)


@pytest.mark.parametrize(("scale", "overlap"), [(0, 0.0), (2, 1.0), (2, -0.1)])
def test_invalid_geometry(scale: int, overlap: float) -> None:
    with pytest.raises(ValueError):
        tile_geometry(100, scale, overlap)


@given(st.integers(1, 4), st.sampled_from([0.0, 0.25, 0.5]), st.integers(20, 120), st.integers(20, 120))
def test_cut_tiles(scale: int, overlap: float, w: int, h: int) -> None:
    tiles = cut_tiles(Image.new("L", (w, h)), scale, overlap)
    assert len(tiles) == scale * scale
    assert len({t.suffix for t in tiles}) == scale * scale
    for t in tiles:
        assert t.image.mode == "RGB"
        assert t.image.size == (t.w_px, t.h_px)
        assert t.x_px + t.w_px <= w + 1 and t.y_px + t.h_px <= h + 1
    assert tiles[-1].suffix == f"s{scale}_o{round(overlap * 100):03d}_x{scale - 1}_y{scale - 1}"
