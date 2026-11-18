import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from vectorview.regions import BBox


@st.composite
def bboxes(draw: st.DrawFn) -> BBox:
    w = draw(st.floats(-180, 179.9))
    e = draw(st.floats(w + 1e-6, 180).filter(lambda x: x > w))
    s = draw(st.floats(-90, 89.9))
    n = draw(st.floats(s + 1e-6, 90).filter(lambda x: x > s))
    return BBox(w, s, e, n)


@given(bboxes())
def test_parse_roundtrip(b: BBox) -> None:
    assert BBox.parse(str(b)) == b
    assert b.as_tuple() == (b.west, b.south, b.east, b.north)


@given(bboxes())
def test_area_is_non_negative_and_bounded(b: BBox) -> None:
    area = b.area_km2()
    assert area >= 0
    assert area <= (b.east - b.west) * 111.32 * (b.north - b.south) * 110.57 + 1e-6


def test_area_of_one_degree_at_equator() -> None:
    assert math.isclose(BBox(0, -0.5, 1, 0.5).area_km2(), 111.32 * 110.57, rel_tol=1e-4)


@pytest.mark.parametrize(
    "text",
    ["1,2,3", "a,b,c,d", "10,0,5,1", "0,10,1,5", "-181,0,0,1", "0,0,181,1", "0,-91,1,0", "0,0,1,91", "nan,0,1,1"],
)
def test_invalid_boxes(text: str) -> None:
    with pytest.raises(ValueError):
        BBox.parse(text)
