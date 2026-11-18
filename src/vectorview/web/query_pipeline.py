"""Run one free-text query: LLM expansion, SigLIP-2 text encoding, stage-1 ranking, VLM rerank.

The query row's ``status`` moves through ``expanding``, ``stage1``, ``reranking`` and ``ready``
(or ``failed``), which the UI polls. A query with a ``parent_query_id`` is a relevance-feedback
round: it reuses the parent's expansion and refines the query vector with the parent's verdicts.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from ..config import load_config
from ..encoder import get_encoder
from . import db, llm_expand, retrieval, vlm_judge

LOG = logging.getLogger(__name__)

DEFAULT_STAGE1_TOPK = 200


async def _expansion(query: dict[str, Any]) -> llm_expand.Expansion:
    parent_id = query.get("parent_query_id")
    parent = db.get_query(int(parent_id)) if parent_id else None
    if parent and parent.get("expansion_json"):
        return llm_expand.Expansion.from_dict(json.loads(parent["expansion_json"]))
    cached = db.find_cached_expansion(query["text"])
    if cached:
        return llm_expand.Expansion.from_dict(cached)
    return await llm_expand.expand(query["text"])


async def run_query(query: dict[str, Any]) -> None:
    """Execute one pending query row and store its results; errors set ``status='failed'``."""
    qid = int(query["id"])
    k = int(query["k"])
    cfg = load_config()
    LOG.info("query %d: %r (scan %s, K=%d, %s)", qid, query["text"], query["scan_id"], k, query["vlm_model"])
    try:
        scan = db.get_scan(int(query["scan_id"]))
        if not scan:
            raise RuntimeError(f"scan {query['scan_id']} not found")

        db.update_query(qid, status="expanding")
        expansion = await _expansion(query)
        db.update_query(qid, expansion=expansion.to_dict())

        db.update_query(qid, status="stage1")
        sc = await asyncio.to_thread(retrieval.get_scan_cache, scan, cfg.embeddings_dir)
        encoder = await asyncio.to_thread(get_encoder)
        q_vec = await asyncio.to_thread(retrieval.build_query_vector, encoder, expansion)
        if query.get("parent_query_id"):
            verdicts = db.list_verifications(int(query["parent_query_id"]))
            yes = [i for i, v in verdicts.items() if v == "yes"]
            no = [i for i, v in verdicts.items() if v == "no"]
            q_vec = retrieval.apply_relevance_feedback(q_vec, sc, yes, no)
        ranked = retrieval.stage1_rank(q_vec, sc, min_date=query["min_date"], top_k=max(DEFAULT_STAGE1_TOPK, k))

        candidates = [
            (iid, meta.cache_path, score)
            for iid, score in ranked[:k]
            if (meta := sc.manifest.get(iid)) is not None and meta.cache_path
        ]
        db.update_query(qid, status="reranking")
        reranked = await vlm_judge.rerank(
            query=query["text"],
            description=expansion.vlm_description,
            exclusions=expansion.vlm_exclusions,
            candidates=candidates,
            cache_path=cfg.vlm_cache_path,
            model=query["vlm_model"],
        )

        results: list[dict[str, Any]] = []
        for rank, r in enumerate(reranked, start=1):
            meta = sc.manifest[r["image_id"]]
            verdict = r["vlm_verdict"]
            results.append(
                {
                    "rank": rank,
                    "image_id": r["image_id"],
                    "stage1_score": round(r["stage1_score"], 4),
                    "vlm_score": round(r["vlm_score"], 4),
                    "vlm_decision": verdict.get("decision", "no"),
                    "vlm_confidence": vlm_judge.confidence(verdict),
                    "vlm_reason": verdict.get("reason", ""),
                    "bbox": r["bbox"],  # [y_min, x_min, y_max, x_max] in 0-1000, or None
                    "pano_id": meta.pano_id,
                    "lat": meta.lat,
                    "lon": meta.lon,
                    "captured_at": meta.captured_at,
                    "heading": meta.heading,
                    "region": meta.region,
                }
            )
        db.update_query(qid, status="ready", results=results, ready=True)
        LOG.info("query %d: %d results", qid, len(results))
    except Exception as exc:  # reported to the UI
        LOG.exception("query %d failed", qid)
        db.update_query(qid, status="failed", error=str(exc))
