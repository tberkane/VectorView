"""Image metadata per embedding cache (``<tag>/manifest.csv``), so queries never touch Postgres."""

from __future__ import annotations

import csv
from dataclasses import astuple, dataclass
from pathlib import Path

COLUMNS = ["image_id", "pano_id", "lat", "lon", "captured_at", "heading", "cache_path", "region"]
"""CSV header, in the field order of `ImageMeta`."""


@dataclass(frozen=True)
class ImageMeta:
    """Metadata of one image.

    Attributes:
        image_id: ``"<pano_id>_h<heading>"``.
        pano_id: Street View panorama id.
        lat: Latitude of the panorama.
        lon: Longitude of the panorama.
        captured_at: Capture date as ``"YYYY-MM-DD"``, or ``""`` if unknown.
        heading: Compass heading of the view, in degrees.
        cache_path: Path of the JPEG.
        region: Region or scan the image belongs to.
    """

    image_id: str
    pano_id: str
    lat: float
    lon: float
    captured_at: str
    heading: float
    cache_path: str
    region: str


def manifest_path(embeddings_dir: Path, tag: str) -> Path:
    """Location of the manifest of cache ``tag``."""
    return embeddings_dir / tag / "manifest.csv"


def write_manifest(path: Path, rows: list[ImageMeta]) -> None:
    """Write ``rows`` as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(astuple(r) for r in rows)


def read_manifest(path: Path) -> dict[str, ImageMeta]:
    """Read a manifest as ``{image_id: ImageMeta}``.

    Raises:
        FileNotFoundError: If the manifest does not exist.
    """
    with path.open(newline="") as f:
        rows = [
            ImageMeta(
                image_id=r["image_id"],
                pano_id=r["pano_id"],
                lat=float(r["lat"]),
                lon=float(r["lon"]),
                captured_at=r["captured_at"],
                heading=float(r["heading"]),
                cache_path=r["cache_path"],
                region=r["region"],
            )
            for r in csv.DictReader(f)
        ]
    return {r.image_id: r for r in rows}
