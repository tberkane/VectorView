"""The image/text encoder interface and a shared, lazily loaded instance.

The real encoder, [`SigLIP2`][vectorview.siglip.SigLIP2], needs the ``encoding`` extra
(torch and transformers). Everything else in VectorView talks to the `Encoder` protocol,
so it can be imported, and tested, without torch.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterable
from typing import Protocol

import numpy as np
from PIL import Image

LOG = logging.getLogger(__name__)

MODEL_ID = "google/siglip2-so400m-patch16-512"
"""Default model. Changing it means re-encoding every image and tile."""

EMBED_DIM = 1152
"""Embedding size of `MODEL_ID`; also the ``VECTOR`` size in ``db/init.sql``."""


class Encoder(Protocol):
    """Maps images and texts into one joint embedding space."""

    def encode_images(self, images: Iterable[Image.Image]) -> np.ndarray:
        """Return L2-normalised image embeddings, shape ``(N, D)``, float32."""
        ...

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised text embeddings, shape ``(N, D)``, float32."""
        ...

    def encode_patch_regions(self, images: Iterable[Image.Image], k: int = 8) -> np.ndarray:
        """Return L2-normalised embeddings of ``k x k`` image regions, shape ``(N, k*k, D)``, float16."""
        ...


_lock = threading.Lock()
_shared: Encoder | None = None


def load_encoder(model_id: str = MODEL_ID) -> Encoder:
    """Load a new [`SigLIP2`][vectorview.siglip.SigLIP2] encoder.

    Raises:
        RuntimeError: If the ``encoding`` extra is not installed.
    """
    try:
        from .siglip import SigLIP2
    except ImportError as exc:
        raise RuntimeError(
            f"the encoder needs torch and transformers ({exc.name} is missing); "
            'install the torch build for your platform, then `pip install "vectorview[encoding]"`'
        ) from exc
    return SigLIP2(model_id=model_id)


def get_encoder() -> Encoder:
    """Return the process-wide encoder, loading it on first use (thread-safe)."""
    global _shared
    with _lock:
        if _shared is None:
            LOG.info("loading the encoder (%s); this takes a while on CPU", MODEL_ID)
            t0 = time.perf_counter()
            _shared = load_encoder()
            LOG.info("encoder loaded in %.1fs", time.perf_counter() - t0)
        return _shared


def set_encoder(encoder: Encoder | None) -> None:
    """Replace (or, with ``None``, reset) the process-wide encoder."""
    global _shared
    with _lock:
        _shared = encoder
