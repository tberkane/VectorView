"""Perspective crops on a CUDA GPU with ``pyequilib`` (used by `vectorview.scraper`)."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .scraper import crop_name, yaw_for_heading

LOG = logging.getLogger(__name__)


def project_gpu(
    rows: list[tuple[str, str, float | None]],
    crop_dir: Path,
    headings: list[float],
    fov: float,
    pitch: float,
    size: int,
) -> bool:
    """Cut the crops of each ``(pano_id, equirect_path, heading_rad)`` row in ``rows``.

    All headings of a panorama are projected in one batched call. Panoramas are processed one
    at a time because their equirectangular sizes differ.

    Returns:
        False if torch, CUDA or ``pyequilib`` is unavailable (nothing is done), True otherwise.
    """
    try:
        import torch
        from equilib.equi2pers.base import equi2pers
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False

    device = torch.device("cuda")
    work = [
        (pid, src, h, [crop_dir / crop_name(pid, hd) for hd in headings])
        for pid, src, h in rows
        if not all((crop_dir / crop_name(pid, hd)).exists() for hd in headings)
    ]
    LOG.info("GPU projection: %d panoramas", len(work))
    pitch_rad = math.radians(pitch)
    n_err = 0
    for i, (_pid, src, pano_h, outs) in enumerate(work, 1):
        try:
            with Image.open(src) as im:
                arr = np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
        except Exception as exc:  # skip unreadable panoramas
            LOG.warning("unreadable panorama %s: %s", src, exc)
            n_err += 1
            continue
        equi = torch.from_numpy(arr).permute(2, 0, 1).contiguous().unsqueeze(0).to(device)
        # equilib wants one equirect per rotation; repeat (not expand) for contiguous strides.
        batch = equi.repeat(len(headings), 1, 1, 1)
        rots = [{"roll": 0.0, "pitch": pitch_rad, "yaw": math.radians(yaw_for_heading(h, pano_h))} for h in headings]
        # z_down=True with positive yaw = right and positive pitch = down matches py360convert.e2p.
        crops = equi2pers(equi=batch, rots=rots, height=size, width=size, fov_x=fov, z_down=True, mode="bilinear")
        crops_np = crops.cpu().numpy()
        for j, out in enumerate(outs):
            Image.fromarray(np.transpose(crops_np[j], (1, 2, 0))).save(out, "JPEG", quality=92)
        if i % 64 == 0:
            LOG.info("GPU projection: %d/%d panoramas", i, len(work))
    LOG.info("GPU projection done: %d ok, %d failed", len(work) - n_err, n_err)
    return True
