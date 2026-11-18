"""Stage-1 ranking of a scan's images against a text query.

The query vector is built from the LLM expansion (`build_query_vector`), optionally refined with
the user's verdicts (`apply_relevance_feedback`), and compared with every image by
`stage1_rank`. Each image's score combines its whole-image similarity with the best similarity
among its tiles and, when available, among its patch regions; a small object can then rank high
even if it fills a small part of the image.

A scan's embeddings are loaded once per process and kept in memory (`get_scan_cache`).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .. import cache
from ..encoder import Encoder
from .llm_expand import Expansion
from .manifest import ImageMeta, manifest_path, read_manifest

LOG = logging.getLogger(__name__)

DEFAULT_ALPHA_T = 0.70
"""Two-way fusion: weight of the whole-image score (the rest goes to the best tile)."""
DEFAULT_ALPHA_N = 0.20
"""Weight of the negative-concept vector subtracted from the query."""
DEFAULT_W_IMG = 0.55
DEFAULT_W_TILE = 0.30
DEFAULT_W_PATCH = 0.15
"""Three-way fusion weights (image, best tile, best patch region), used when a scan has patch regions."""

DEFAULT_RF_ALPHA = 0.70
"""Relevance feedback: weight of the text query against the mean of the images marked yes."""
DEFAULT_RF_GAMMA = 0.40
"""Relevance feedback: weight of the mean of the images marked no, subtracted from the query."""
DEFAULT_RF_MIN_NEG = 5
"""Relevance feedback: images marked no needed before they are subtracted."""


@dataclass
class ScanCache:
    """In-memory embeddings of one scan.

    Attributes:
        image_feats: ``(N, D)`` L2-normalised image embeddings.
        image_ids: The ``N`` image ids, in row order.
        tile_feats: ``(N_t, D)`` L2-normalised tile embeddings.
        parent_idx: ``(N_t,)`` row in ``image_feats`` of each tile's parent image.
        manifest: Metadata of every image in the cache.
        patch_feats: ``(N, K*K, D)`` patch-region embeddings, or ``None``.
    """

    image_feats: np.ndarray
    image_ids: list[str]
    tile_feats: np.ndarray
    parent_idx: np.ndarray
    manifest: dict[str, ImageMeta]
    patch_feats: np.ndarray | None = None


def l2(v: np.ndarray) -> np.ndarray:
    """``v`` divided by its L2 norm (unchanged if the norm is 0)."""
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def load_scan_cache(
    root: Path,
    *,
    embedding_tag: str,
    tile_tag: str,
    region_filter: str | None = None,
    patch_region_tag: str | None = None,
) -> ScanCache:
    """Load a scan's image, tile and (optional) patch caches, keeping only ``region_filter`` images."""
    t0 = time.perf_counter()
    feats_all, ids_all, regions_all = cache.load_images(root, embedding_tag)
    keep = np.array([region_filter is None or r == region_filter for r in regions_all], dtype=bool)
    image_ids = [iid for iid, k in zip(ids_all, keep, strict=True) if k]
    pos = {iid: i for i, iid in enumerate(image_ids)}

    tile_feats_all, _, tile_parents = cache.load_tiles(root, tile_tag)
    tile_keep = np.array([p in pos for p in tile_parents], dtype=bool)
    parent_idx = np.array([pos[p] for p in tile_parents if p in pos], dtype=np.int64)

    patch_feats = None
    if patch_region_tag:
        patch_feats = cache.load_patch_regions(root, patch_region_tag, image_ids)

    LOG.info(
        "loaded scan cache %s in %.1fs: %d images, %d tiles%s",
        embedding_tag,
        time.perf_counter() - t0,
        len(image_ids),
        int(tile_keep.sum()),
        ", with patch regions" if patch_feats is not None else "",
    )
    return ScanCache(
        image_feats=feats_all[keep],
        image_ids=image_ids,
        tile_feats=tile_feats_all[tile_keep],
        parent_idx=parent_idx,
        manifest=read_manifest(manifest_path(root, embedding_tag)),
        patch_feats=patch_feats,
    )


_cache_lock = threading.Lock()
_scan_caches: dict[int, ScanCache] = {}


def get_scan_cache(scan: dict[str, Any], root: Path) -> ScanCache:
    """Return the scan's `ScanCache`, loading it on first use (memoised per scan id)."""
    sid = int(scan["id"])
    with _cache_lock:
        if sid not in _scan_caches:
            _scan_caches[sid] = load_scan_cache(
                root,
                embedding_tag=scan["embedding_tag"],
                tile_tag=scan["tile_tag"],
                region_filter=scan.get("region_filter"),
                patch_region_tag=scan.get("patch_region_tag"),
            )
        return _scan_caches[sid]


def clear_scan_caches() -> None:
    """Forget every loaded scan cache."""
    with _cache_lock:
        _scan_caches.clear()


def build_query_vector(encoder: Encoder, expansion: Expansion, *, alpha_n: float = DEFAULT_ALPHA_N) -> np.ndarray:
    """Combine the expansion's texts into one unit query vector.

    ``q_pos = L2(mean(paraphrases + visual_hints))``, ``q_neg = L2(mean(negatives))`` and
    ``q = L2(q_pos - alpha_n * q_neg)`` (just ``q_pos`` without negatives).

    Raises:
        ValueError: If the expansion has no paraphrases or visual hints.
    """
    pos_texts = [*expansion.paraphrases, *expansion.visual_hints]
    if not pos_texts:
        raise ValueError("expansion has no positive texts to encode")
    feats = encoder.encode_texts(pos_texts + list(expansion.negatives))
    q_pos = l2(feats[: len(pos_texts)].mean(axis=0))
    neg = feats[len(pos_texts) :]
    q = l2(q_pos - alpha_n * l2(neg.mean(axis=0))) if len(neg) and alpha_n > 0 else q_pos
    return q.astype(np.float32, copy=False)


def apply_relevance_feedback(
    q_vec: np.ndarray,
    scan_cache: ScanCache,
    yes_ids: list[str],
    no_ids: list[str],
    *,
    alpha: float = DEFAULT_RF_ALPHA,
    gamma: float = DEFAULT_RF_GAMMA,
    min_neg: int = DEFAULT_RF_MIN_NEG,
) -> np.ndarray:
    """Rocchio-style refinement with the user's verdicts.

    ``q' = L2(alpha * q + (1 - alpha) * mean_yes)``, then ``q'' = L2(q' - gamma * mean_no)`` when at
    least ``min_neg`` images were marked no (means are L2-normalised). Returns ``q_vec`` unchanged
    if no marked-yes image is in the cache.
    """
    idx = {iid: i for i, iid in enumerate(scan_cache.image_ids)}
    pos = [idx[i] for i in yes_ids if i in idx]
    if not pos:
        return q_vec
    proto_pos = l2(scan_cache.image_feats[pos].mean(axis=0))
    q = q_vec if alpha >= 1.0 else l2(alpha * q_vec + (1.0 - alpha) * proto_pos)
    neg = [idx[i] for i in no_ids if i in idx]
    if gamma > 0.0 and len(neg) >= min_neg:
        q = l2(q - gamma * l2(scan_cache.image_feats[neg].mean(axis=0)))
    return q.astype(np.float32, copy=False)


def fused_scores(
    q: np.ndarray,
    sc: ScanCache,
    *,
    alpha_t: float = DEFAULT_ALPHA_T,
    w_img: float = DEFAULT_W_IMG,
    w_tile: float = DEFAULT_W_TILE,
    w_patch: float = DEFAULT_W_PATCH,
) -> np.ndarray:
    """Score every image of ``sc`` against the unit query ``q``.

    With patch regions: ``w_img * image + w_tile * best_tile + w_patch * best_patch``.
    Without: ``alpha_t * image + (1 - alpha_t) * best_tile``. An image with no tiles (or no
    patch regions) uses its image score in their place.
    """
    img = sc.image_feats @ q
    max_tile = img.copy()
    if sc.tile_feats.size:
        best = np.full(len(img), -np.inf, dtype=np.float32)
        np.maximum.at(best, sc.parent_idx, sc.tile_feats @ q)
        max_tile = np.where(np.isfinite(best), best, img)
    if sc.patch_feats is not None:
        regions = sc.patch_feats @ q
        has_patch = np.abs(sc.patch_feats).sum(axis=(1, 2)) > 0
        max_patch = np.where(has_patch, regions.max(axis=1), img)
        return w_img * img + w_tile * max_tile + w_patch * max_patch
    return alpha_t * img + (1.0 - alpha_t) * max_tile


def stage1_rank(
    q: np.ndarray, sc: ScanCache, *, min_date: str | None = None, top_k: int = 200, **weights: float
) -> list[tuple[str, float]]:
    """Return the ``top_k`` ``(image_id, score)`` pairs, best first.

    Args:
        q: Unit query vector.
        sc: The scan's embeddings.
        min_date: Drop images captured before this ISO date; images without a date are kept.
        top_k: Number of results.
        **weights: Overrides for `fused_scores` (``alpha_t``, ``w_img``, ``w_tile``, ``w_patch``).
    """
    scores = fused_scores(q, sc, **weights)
    rows = np.arange(len(scores))
    if min_date:
        keep = []
        for i, iid in enumerate(sc.image_ids):
            meta = sc.manifest.get(iid)
            if meta is None:
                continue
            if not meta.captured_at or meta.captured_at >= min_date:
                keep.append(i)
        rows = np.asarray(keep, dtype=np.int64)
    order = rows[np.argsort(-scores[rows], kind="stable")[:top_k]]
    return [(sc.image_ids[i], float(scores[i])) for i in order]
