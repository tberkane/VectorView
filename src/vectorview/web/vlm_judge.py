"""Rerank candidates with a vision-language model (VLM) judge.

Each candidate image is shown to the VLM with the query; it answers ``yes``, ``no`` or
``unsure`` with a confidence and, for ``yes``/``unsure``, a bounding box. Two backends:

- `GeminiJudge` (default): JSON output constrained by a schema;
- `OpenAIJudge`: JSON mode.

Verdicts are cached in a JSON-lines file keyed by ``(prompt hash, image id, model)``, so a
changed prompt or another model never reuses stale verdicts.
"""

from __future__ import annotations

import asyncio
import base64
import datetime as dt
import hashlib
import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

LOG = logging.getLogger(__name__)

DEFAULT_VLM_MODEL = "gemini-3.1-flash-lite"
DEFAULT_MAX_IMAGE_SIDE = 640
DEFAULT_CONCURRENCY = 8

COST_PER_CALL_USD = {
    "gemini-3.1-flash-lite": 0.00015,
    "gemini-2.5-flash-lite": 0.00015,
    "gpt-5-mini": 0.001,
}
"""Rough cost of one judge call, shown in the UI; check your provider's current prices."""
DEFAULT_COST_USD = 0.001

Candidate = tuple[str, str, float]
"""``(image_id, image_path, stage1_score)``."""

PROMPT_TEMPLATE = """You are judging a single street-level photograph for a visual search.

User is looking for: {query}

Description of what counts as YES: {description}

Do NOT call yes when: {exclusions}

Reply with strict JSON, no prose. Schema:
{{
  "decision": "yes" | "no" | "unsure",
  "confidence": <0.0-1.0>,
  "reason": "<one short sentence under 20 words>",
  "bbox": [y_min, x_min, y_max, x_max]
}}

- "yes" if the photograph clearly shows what the user is looking for.
- "no" if it doesn't, or only vaguely suggests it.
- "unsure" only if the photograph is too low-quality, occluded, or ambiguous.
- confidence: how confident you are (0.0 = guess, 1.0 = certain).
- bbox: only when decision is "yes" or "unsure". Use a 0-1000 normalized
  coordinate system (top-left origin), giving [y_min, x_min, y_max, x_max]
  for the smallest rectangle that contains the matching object/region.
  Omit the bbox field entirely when decision is "no"."""


def build_prompt(query: str, description: str, exclusions: str) -> str:
    """Fill `PROMPT_TEMPLATE`, with placeholders when the description or exclusions are empty."""
    return PROMPT_TEMPLATE.format(
        query=query.strip(),
        description=description.strip() or "(no extra description provided)",
        exclusions=exclusions.strip() or "(no exclusions specified)",
    )


def prompt_hash(prompt: str) -> str:
    """Short hash of a prompt, used in cache keys."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def encode_image(path: Path, max_side: int = DEFAULT_MAX_IMAGE_SIDE) -> bytes:
    """Read an image, shrink it so its longer side is at most ``max_side``, and return JPEG bytes."""
    with Image.open(path) as im:
        img = im.convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def parse_bbox(raw: Any) -> list[float] | None:
    """Validate a ``[y_min, x_min, y_max, x_max]`` box in 0-1000 coordinates.

    Values are clamped to ``[0, 1000]``. Returns ``None`` if the box is missing, malformed or empty.
    """
    if not isinstance(raw, list | tuple) or len(raw) != 4:
        return None
    try:
        vals = [min(1000.0, max(0.0, float(v))) for v in raw]
    except (TypeError, ValueError):
        return None
    ymin, xmin, ymax, xmax = vals
    if ymax <= ymin or xmax <= xmin:
        return None
    return vals


def confidence(verdict: dict[str, Any]) -> float:
    """The verdict's confidence clamped to ``[0, 1]`` (0 if missing or not a number)."""
    try:
        conf = float(verdict.get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, conf)) if conf == conf else 0.0


def vlm_score(verdict: dict[str, Any]) -> float:
    """Score in ``[0, 1]``: confidence for ``yes``, half of it for ``unsure``, 0 for ``no``."""
    conf = confidence(verdict)
    decision = verdict.get("decision", "no")
    if decision == "yes":
        return conf
    if decision == "unsure":
        return 0.5 * conf
    return 0.0


def load_cache(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Read the verdict cache; malformed lines are skipped."""
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
                out[(row["prompt_hash"], row["image_id"], row.get("model", ""))] = row
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return out


def append_cache(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append verdicts to the cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


class Judge(Protocol):
    """A VLM backend."""

    model: str

    async def judge(self, image_bytes: bytes, prompt: str) -> dict[str, Any]:
        """Return the parsed JSON verdict for one JPEG image."""
        ...


def _parse_json(text: str, backend: str) -> dict[str, Any]:
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{backend} reply is not JSON: {text[:200]!r}") from exc
    if not isinstance(doc, dict):
        raise RuntimeError(f"{backend} reply is not a JSON object: {text[:200]!r}")
    return doc


class GeminiJudge:
    """Gemini judge (``google-genai``) with schema-constrained JSON output.

    Args:
        model: Gemini model name.
        client: A ``google.genai.Client``; created from ``GEMINI_API_KEY`` if omitted.

    Raises:
        RuntimeError: If the ``llm`` extra or the API key is missing.
    """

    RESPONSE_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["yes", "no", "unsure"]},
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
            "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
        },
        "required": ["decision", "confidence"],
    }

    def __init__(self, model: str = DEFAULT_VLM_MODEL, client: Any = None) -> None:
        try:
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError('the Gemini judge needs the llm extra: pip install "vectorview[llm]"') from exc
        self._types = types
        if client is None:
            from google import genai

            key = os.environ.get("GEMINI_API_KEY")
            if not key:
                raise RuntimeError("GEMINI_API_KEY is not set")
            client = genai.Client(api_key=key)
        self.client = client
        self.model = model

    async def judge(self, image_bytes: bytes, prompt: str) -> dict[str, Any]:
        """Return the parsed JSON verdict for one JPEG image."""
        types = self._types
        resp = await self.client.aio.models.generate_content(
            model=self.model,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"), prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_schema=self.RESPONSE_SCHEMA, temperature=0.0
            ),
        )
        return _parse_json(resp.text or "", "Gemini")


class OpenAIJudge:
    """OpenAI judge (JSON mode). Its boxes tend to be less precise than Gemini's.

    Args:
        model: OpenAI model name.
        client: An ``openai.AsyncOpenAI``-compatible client; created from ``OPENAI_API_KEY`` if omitted.

    Raises:
        RuntimeError: If the ``llm`` extra or the API key is missing.
    """

    def __init__(self, model: str = "gpt-5-mini", client: Any = None) -> None:
        if client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise RuntimeError('the OpenAI judge needs the llm extra: pip install "vectorview[llm]"') from exc
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not set")
            client = AsyncOpenAI()
        self.client = client
        self.model = model

    async def judge(self, image_bytes: bytes, prompt: str) -> dict[str, Any]:
        """Return the parsed JSON verdict for one JPEG image."""
        b64 = base64.b64encode(image_bytes).decode("ascii")
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}},
                    ],
                }
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=2000,
        )
        return _parse_json(resp.choices[0].message.content or "", "OpenAI")


def build_judge(model: str) -> Judge:
    """Create the judge for a model name (``gemini-*`` or ``gpt-*``).

    Raises:
        ValueError: For any other model name.
    """
    if model.startswith("gemini"):
        return GeminiJudge(model=model)
    if model.startswith("gpt"):
        return OpenAIJudge(model=model)
    raise ValueError(f"unknown VLM model: {model!r}")


def _result(image_id: str, path: str, s1: float, verdict: dict[str, Any], cache_hit: bool) -> dict[str, Any]:
    return {
        "image_id": image_id,
        "stage1_score": s1,
        "vlm_verdict": verdict,
        "vlm_score": vlm_score(verdict),
        "bbox": parse_bbox(verdict.get("bbox")),
        "cache_path": path,
        "cache_hit": cache_hit,
    }


async def rerank(
    *,
    query: str,
    description: str,
    exclusions: str,
    candidates: list[Candidate],
    cache_path: Path,
    model: str = DEFAULT_VLM_MODEL,
    judge: Judge | None = None,
    max_image_side: int = DEFAULT_MAX_IMAGE_SIDE,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list[dict[str, Any]]:
    """Judge each candidate and sort by ``(vlm_score, stage1_score)``, highest first.

    A failed call counts as ``no`` with confidence 0 (its reason records the error) and is not cached.

    Args:
        query: The user's text.
        description: What counts as a match (from the expansion).
        exclusions: Look-alikes to reject (from the expansion).
        candidates: ``(image_id, image_path, stage1_score)`` to judge.
        cache_path: Verdict cache file.
        model: VLM model name.
        judge: Backend to use; built from ``model`` if omitted (only when some verdict is not cached).
        max_image_side: Images are shrunk to this size before upload.
        concurrency: Concurrent VLM calls.

    Returns:
        One dict per candidate with ``image_id``, ``stage1_score``, ``vlm_verdict``, ``vlm_score``,
        ``bbox`` (or ``None``), ``cache_path`` and ``cache_hit``.
    """
    prompt = build_prompt(query, description, exclusions)
    p_hash = prompt_hash(prompt)
    cache = load_cache(cache_path)
    judged: dict[str, dict[str, Any]] = {}
    misses: list[Candidate] = []
    for image_id, path, s1 in candidates:
        hit = cache.get((p_hash, image_id, model))
        if hit is None:
            misses.append((image_id, path, s1))
            continue
        verdict = {k: hit[k] for k in ("decision", "confidence", "reason", "bbox") if hit.get(k) is not None}
        judged[image_id] = _result(image_id, path, s1, verdict, cache_hit=True)
    LOG.info("rerank (%s): %d candidates, %d cached, %d calls", model, len(candidates), len(judged), len(misses))

    if misses:
        backend = judge or build_judge(model)
        sem = asyncio.Semaphore(concurrency)
        new_rows: list[dict[str, Any]] = []

        async def one(image_id: str, path: str, s1: float) -> None:
            async with sem:
                try:
                    verdict = await backend.judge(encode_image(Path(path), max_image_side), prompt)
                except Exception as exc:  # one failed call must not fail the query
                    LOG.warning("VLM judge failed for %s: %s", image_id, exc)
                    verdict = {"decision": "no", "confidence": 0.0, "reason": f"judge error: {exc}"}
                    judged[image_id] = _result(image_id, path, s1, verdict, cache_hit=False)
                    return  # errors are not cached, so the next query retries
            res = _result(image_id, path, s1, verdict, cache_hit=False)
            judged[image_id] = res
            new_rows.append(
                {
                    "prompt_hash": p_hash,
                    "image_id": image_id,
                    "decision": verdict.get("decision"),
                    "confidence": verdict.get("confidence"),
                    "reason": verdict.get("reason"),
                    "bbox": res["bbox"],
                    "model": model,
                    "ts": dt.datetime.now(dt.UTC).isoformat(),
                }
            )

        await asyncio.gather(*(one(*c) for c in misses))
        append_cache(cache_path, new_rows)

    out = [judged[iid] for iid, _, _ in candidates]
    out.sort(key=lambda r: (r["vlm_score"], r["stage1_score"]), reverse=True)
    return out
