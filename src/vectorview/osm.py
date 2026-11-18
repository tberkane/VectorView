"""Sample points along the OpenStreetMap road network.

Each candidate is a ``(lat, lon, bearing)`` triple spaced along a drivable road. The bearing of
the road at that point becomes the Street View heading, so images look along the road.

1. Download the drivable graph with osmnx (cached on disk as GraphML).
2. Project the edges to the local UTM zone so distances are in metres.
3. Place a point every ``spacing_m`` metres along each edge and compute the road bearing there.
4. Drop near-duplicates (intersections share endpoints) with a grid of ``dedup_m`` cells.
"""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from typing import Any

import shapely
from pyproj import Transformer

from .regions import BBox

LOG = logging.getLogger(__name__)

Candidate = tuple[float, float, float]
"""``(lat, lon, road_bearing_deg)``."""


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the WGS84 UTM zone containing ``(lon, lat)``."""
    zone = min(int((lon + 180) / 6) + 1, 60)
    return (32600 if lat >= 0 else 32700) + zone


def load_or_download_graph(name: str, bbox: BBox, cache_dir: Path) -> Any:
    """Return the drivable road network inside ``bbox`` as an osmnx ``MultiDiGraph``.

    The graph is cached as ``<cache_dir>/<name>.graphml``; later calls with the same name read
    the cache and make no network request.
    """
    import osmnx as ox

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{name}.graphml"
    if cache_path.exists():
        LOG.info("loading cached road graph %s", cache_path)
        return ox.load_graphml(cache_path)
    LOG.info("downloading the drivable road network for %s (bbox %s)", name, bbox)
    graph = ox.graph_from_bbox(bbox=bbox.as_tuple(), network_type="drive", simplify=True)
    ox.save_graphml(graph, cache_path)
    LOG.info("saved %s: %d nodes, %d edges", cache_path, len(graph.nodes), len(graph.edges))
    return graph


def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from point 1 to point 2, in degrees clockwise from north, in ``[0, 360)``."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def sample_candidates(
    graph: Any,
    *,
    spacing_m: float = 50.0,
    dedup_m: float = 25.0,
    seed: int = 1234,
) -> list[Candidate]:
    """Place a candidate every ``spacing_m`` metres along every edge of ``graph``.

    Args:
        graph: osmnx road graph, e.g. from `load_or_download_graph`.
        spacing_m: Distance between candidates along an edge.
        dedup_m: Side of the grid cells used to drop near-duplicates; the first candidate in a cell wins.
        seed: Shuffle seed. The order is deterministic, so an interrupted ingest resumes in the same order.

    Returns:
        Deduplicated, shuffled ``(lat, lon, bearing)`` candidates.
    """
    import osmnx as ox

    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True, fill_edge_geometry=True)
    if edges.empty:
        return []
    centroid = edges.geometry.union_all().centroid
    utm = utm_epsg(centroid.x, centroid.y)
    fwd = Transformer.from_crs("EPSG:4326", f"EPSG:{utm}", always_xy=True)
    back = Transformer.from_crs(f"EPSG:{utm}", "EPSG:4326", always_xy=True)

    candidates: list[Candidate] = []
    for geom in edges.geometry:
        if geom is None or geom.is_empty:
            continue
        projected = shapely.transform(geom, fwd.transform, interleaved=False)
        length_m = projected.length
        if length_m < 1.0:
            continue
        for i in range(max(1, int(length_m // spacing_m)) + 1):
            d = min(i * spacing_m, length_m)
            pt = projected.interpolate(d)
            lon, lat = back.transform(pt.x, pt.y)
            # Bearing: look 10 m further along the edge, or 10 m back at its end.
            d_next = min(d + 10.0, length_m)
            if d_next == d:
                d_prev = max(0.0, d - 10.0)
                p = projected.interpolate(d_prev)
                prev_lon, prev_lat = back.transform(p.x, p.y)
                b = bearing(prev_lat, prev_lon, lat, lon)
            else:
                p = projected.interpolate(d_next)
                next_lon, next_lat = back.transform(p.x, p.y)
                b = bearing(lat, lon, next_lat, next_lon)
            candidates.append((lat, lon, b))

    return _dedup_and_shuffle(candidates, fwd, dedup_m=dedup_m, seed=seed)


def _dedup_and_shuffle(candidates: list[Candidate], fwd: Transformer, *, dedup_m: float, seed: int) -> list[Candidate]:
    cells: dict[tuple[int, int], Candidate] = {}
    for lat, lon, b in candidates:
        x, y = fwd.transform(lon, lat)
        cells.setdefault((int(x // dedup_m), int(y // dedup_m)), (lat, lon, b))
    deduped = list(cells.values())
    random.Random(seed).shuffle(deduped)
    LOG.info("sampled %d candidates (%d before dedup)", len(deduped), len(candidates))
    return deduped
