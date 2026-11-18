"""Overlapping grid tiles cut from an image.

For an image side ``W``, an ``N x N`` grid and an overlap fraction ``o`` in ``[0, 1)``:

    T = W / (1 + (N - 1) * (1 - o))     # tile side, in pixels
    S = T * (1 - o)                     # stride between neighbouring tiles
    starts: 0, S, 2S, ..., (N - 1) * S

At ``W = 640``: ``(N=2, o=0)`` gives four 320-px tiles; ``(N=3, o=0.5)`` gives nine 320-px tiles
with stride 160. Width and height are handled independently for non-square images.

A tile's id is its parent image id plus the suffix ``s{N}_o{overlap_pct:03d}_x{i}_y{j}``,
for example ``"<image_id>_s2_o000_x0_y1"``.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class Tile:
    """One tile cut from a parent image.

    Attributes:
        suffix: Tile id suffix; the full tile id is ``f"{parent_id}_{suffix}"``.
        scale: Grid side ``N``.
        overlap: Overlap fraction in ``[0, 1)``.
        x_px: Left edge in the parent, in pixels.
        y_px: Top edge in the parent, in pixels.
        w_px: Tile width, in pixels.
        h_px: Tile height, in pixels.
        image: The cropped RGB image.
    """

    suffix: str
    scale: int
    overlap: float
    x_px: int
    y_px: int
    w_px: int
    h_px: int
    image: Image.Image


def tile_geometry(side_px: int, scale: int, overlap: float) -> tuple[int, list[int]]:
    """Return ``(tile_size, starts)`` along one axis.

    Sizes and starts are rounded to whole pixels; the last start is clamped so the last tile
    ends inside the image.

    Raises:
        ValueError: If ``scale < 1`` or ``overlap`` is outside ``[0, 1)``.
    """
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    if scale == 1:
        return side_px, [0]
    raw_t = side_px / (1.0 + (scale - 1) * (1.0 - overlap))
    raw_s = raw_t * (1.0 - overlap)
    tile_size = round(raw_t)
    starts = [round(i * raw_s) for i in range(scale)]
    starts[-1] = min(starts[-1], max(0, side_px - tile_size))
    return tile_size, starts


def cut_tiles(image: Image.Image, scale: int, overlap: float = 0.0) -> list[Tile]:
    """Cut ``image`` into a ``scale x scale`` grid of tiles with the given overlap.

    Args:
        image: Parent image; it is converted to RGB.
        scale: Grid side ``N``.
        overlap: Overlap between neighbouring tiles, as a fraction of the tile side.

    Returns:
        ``scale * scale`` tiles in row-major order (``y`` outer, ``x`` inner).
    """
    img = image.convert("RGB")
    width, height = img.size
    tw, x_starts = tile_geometry(width, scale, overlap)
    th, y_starts = tile_geometry(height, scale, overlap)
    overlap_pct = round(overlap * 100)
    tiles: list[Tile] = []
    for yi, y in enumerate(y_starts):
        for xi, x in enumerate(x_starts):
            tiles.append(
                Tile(
                    suffix=f"s{scale}_o{overlap_pct:03d}_x{xi}_y{yi}",
                    scale=scale,
                    overlap=overlap,
                    x_px=x,
                    y_px=y,
                    w_px=tw,
                    h_px=th,
                    image=img.crop((x, y, x + tw, y + th)),
                )
            )
    return tiles
