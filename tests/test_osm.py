import math
from pathlib import Path
from typing import Any

import networkx as nx
import pytest
from hypothesis import given
from hypothesis import strategies as st

from vectorview import osm
from vectorview.regions import BBox


def grid_graph(n: int = 3, step_deg: float = 0.001) -> nx.MultiDiGraph:
    """A small street grid near (lat 42.36, lon -71.06) in the format osmnx produces."""
    g = nx.MultiDiGraph(crs="EPSG:4326")
    for i in range(n):
        for j in range(n):
            g.add_node(i * n + j, x=-71.06 + j * step_deg, y=42.36 + i * step_deg)
    for i in range(n):
        for j in range(n):
            u = i * n + j
            if j + 1 < n:
                g.add_edge(u, u + 1, osmid=u, length=82.0)
            if i + 1 < n:
                g.add_edge(u, u + n, osmid=u, length=111.0)
    return g


lat = st.floats(-80, 80)
lon = st.floats(-179.9, 179.9)


@given(lat, lon, lat, lon)
def test_bearing_range(lat1: float, lon1: float, lat2: float, lon2: float) -> None:
    assert 0.0 <= osm.bearing(lat1, lon1, lat2, lon2) < 360.0


def test_bearing_cardinal_directions() -> None:
    assert osm.bearing(0, 0, 1, 0) == pytest.approx(0.0)
    assert osm.bearing(0, 0, 0, 1) == pytest.approx(90.0)
    assert osm.bearing(1, 0, 0, 0) == pytest.approx(180.0)
    assert osm.bearing(0, 1, 0, 0) == pytest.approx(270.0)


@given(lon, lat)
def test_utm_zone(lo: float, la: float) -> None:
    code = osm.utm_epsg(lo, la)
    zone = code % 100
    assert 1 <= zone <= 60
    assert code // 100 == (326 if la >= 0 else 327)
    assert abs(lo - (-183 + 6 * zone)) <= 3.0 + 1e-9


def test_sample_candidates_spacing_dedup_and_determinism() -> None:
    g = grid_graph()
    a = osm.sample_candidates(g, spacing_m=20, dedup_m=10)
    b = osm.sample_candidates(g, spacing_m=20, dedup_m=10)
    assert a == b and len(a) > 20
    for la, lo, br in a:
        assert 42.359 < la < 42.363 and -71.061 < lo < -71.057
        # Every edge of this grid runs east-west or north-south.
        assert min(abs(br - c) for c in (0, 90, 180, 270, 360)) < 1.0
    coarse = osm.sample_candidates(g, spacing_m=20, dedup_m=200)
    assert len(coarse) < len(a)
    assert sorted(osm.sample_candidates(g, spacing_m=20, dedup_m=10, seed=1)) == sorted(a)


def test_sample_candidates_empty_graph() -> None:
    g = nx.MultiDiGraph(crs="EPSG:4326")
    g.add_node(1, x=0.0, y=0.0)
    with pytest.raises(ValueError):  # osmnx refuses graphs without edges
        osm.sample_candidates(g)


def test_short_edges_are_skipped() -> None:
    g = nx.MultiDiGraph(crs="EPSG:4326")
    g.add_node(1, x=0.0, y=0.0)
    g.add_node(2, x=0.000001, y=0.0)
    g.add_edge(1, 2, osmid=1, length=0.1)
    assert osm.sample_candidates(g) == []


def test_graph_is_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import osmnx as ox

    calls: list[Any] = []

    def fake_download(**kw: Any) -> nx.MultiDiGraph:
        calls.append(kw)
        return grid_graph()

    monkeypatch.setattr(ox, "graph_from_bbox", fake_download)
    bbox = BBox(-71.061, 42.359, -71.057, 42.363)
    g1 = osm.load_or_download_graph("area", bbox, tmp_path)
    g2 = osm.load_or_download_graph("area", bbox, tmp_path)
    assert len(calls) == 1 and calls[0]["bbox"] == bbox.as_tuple()
    assert (tmp_path / "area.graphml").exists()
    assert len(g1.edges) == len(g2.edges)
    assert math.isclose(len(osm.sample_candidates(g2)), len(osm.sample_candidates(g1)))


def test_edges_shorter_than_the_spacing_get_both_ends() -> None:
    cands = osm.sample_candidates(grid_graph(n=2), spacing_m=500, dedup_m=1)
    # Each of the 4 edges yields its start and its end; the end looks back along the edge.
    assert len(cands) == 4
    assert all(min(abs(b - c) for c in (0, 90, 180, 270, 360)) < 1.0 for _, _, b in cands)
