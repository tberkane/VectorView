"""Geographic bounding boxes."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BBox:
    """An axis-aligned WGS84 bounding box.

    Attributes:
        west: Minimum longitude, in degrees.
        south: Minimum latitude, in degrees.
        east: Maximum longitude, in degrees.
        north: Maximum latitude, in degrees.

    Raises:
        ValueError: If a coordinate is out of range or the box is empty. Boxes that cross the
            antimeridian are not supported.
    """

    west: float
    south: float
    east: float
    north: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.west, self.south, self.east, self.north)):
            raise ValueError(f"bbox coordinates must be finite, got {self!r}")
        if not -180.0 <= self.west < self.east <= 180.0:
            raise ValueError(f"need -180 <= west < east <= 180, got west={self.west} east={self.east}")
        if not -90.0 <= self.south < self.north <= 90.0:
            raise ValueError(f"need -90 <= south < north <= 90, got south={self.south} north={self.north}")

    @classmethod
    def parse(cls, text: str) -> BBox:
        """Parse ``"west,south,east,north"``.

        Raises:
            ValueError: If the text does not hold four numbers or the box is invalid.
        """
        parts = [p.strip() for p in text.split(",")]
        if len(parts) != 4:
            raise ValueError(f"expected 'west,south,east,north', got {text!r}")
        west, south, east, north = (float(p) for p in parts)
        return cls(west, south, east, north)

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return ``(west, south, east, north)``, the order osmnx expects."""
        return (self.west, self.south, self.east, self.north)

    def __str__(self) -> str:
        return f"{self.west},{self.south},{self.east},{self.north}"

    def area_km2(self) -> float:
        """Approximate area in km² (equirectangular approximation at the mid latitude)."""
        lat_mid = (self.north + self.south) / 2.0
        dx_km = (self.east - self.west) * 111.32 * math.cos(math.radians(lat_mid))
        dy_km = (self.north - self.south) * 110.57
        return max(0.0, dx_km * dy_km)
