"""SigLIP-2 encoder (needs the ``encoding`` extra)."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from .encoder import EMBED_DIM, MODEL_ID

LOG = logging.getLogger(__name__)


class SigLIP2:
    """SigLIP-2 image and text encoder.

    Runs in float16 on CUDA when available, otherwise in float32 on CPU.

    Args:
        model_id: Hugging Face model id. Its vision width must equal `EMBED_DIM`.
        device: Torch device; picked automatically when ``None``.

    Raises:
        RuntimeError: If the model's embedding size does not match `EMBED_DIM`.
    """

    def __init__(self, model_id: str = MODEL_ID, device: str | None = None) -> None:
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        LOG.info("loading %s onto %s (%s)", model_id, self.device, self.dtype)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, dtype=self.dtype).to(self.device).eval()
        actual = self.model.config.vision_config.hidden_size
        if actual != EMBED_DIM:
            raise RuntimeError(f"{model_id} has embedding size {actual}, but the schema uses {EMBED_DIM}")

    def _pixels(self, images: list[Image.Image]) -> dict[str, torch.Tensor]:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        return inputs

    def encode_images(self, images: Iterable[Image.Image]) -> np.ndarray:
        """L2-normalised image embeddings, shape ``(N, EMBED_DIM)``, float32."""
        batch = list(images)
        if not batch:
            return np.empty((0, EMBED_DIM), dtype=np.float32)
        with torch.inference_mode():
            out = self.model.get_image_features(**self._pixels(batch))
            feats = out.pooler_output if hasattr(out, "pooler_output") else out
            feats = torch.nn.functional.normalize(feats, dim=-1)
            return feats.float().cpu().numpy()

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """L2-normalised text embeddings, shape ``(N, EMBED_DIM)``, float32."""
        if not texts:
            return np.empty((0, EMBED_DIM), dtype=np.float32)
        with torch.inference_mode():
            inputs = self.processor(text=texts, padding="max_length", return_tensors="pt").to(self.device)
            out = self.model.get_text_features(**inputs)
            feats = out.pooler_output if hasattr(out, "pooler_output") else out
            feats = torch.nn.functional.normalize(feats, dim=-1)
            return feats.float().cpu().numpy()

    def encode_patch_regions(self, images: Iterable[Image.Image], k: int = 8) -> np.ndarray:
        """Embed ``k x k`` regions of each image.

        The model's attention-pooling head is applied to each ``k x k`` block of the patch grid,
        so the region embeddings live in the same space as text embeddings.

        Returns:
            Shape ``(N, k*k, EMBED_DIM)``, float16, L2-normalised along the last axis.

        Raises:
            ValueError: If ``k`` does not divide the patch grid side.
        """
        batch = list(images)
        if not batch:
            return np.empty((0, k * k, EMBED_DIM), dtype=np.float16)
        with torch.inference_mode():
            out = self.model.vision_model(
                pixel_values=self._pixels(batch)["pixel_values"], output_hidden_states=False, return_dict=True
            )
            h = out.last_hidden_state  # (B, P, D), after the final layer norm
            b, p, d = h.shape
            grid = math.isqrt(p)
            if grid * grid != p:
                raise RuntimeError(f"{self.model_id} has a non-square patch grid ({p} patches)")
            if grid % k:
                raise ValueError(f"k={k} does not divide the {grid}x{grid} patch grid")
            bs = grid // k
            g = h.view(b, k, bs, k, bs, d).permute(0, 1, 3, 2, 4, 5).contiguous().view(b * k * k, bs * bs, d)
            pooled = self.model.vision_model.head(g).view(b, k * k, d)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            return pooled.half().cpu().numpy()
