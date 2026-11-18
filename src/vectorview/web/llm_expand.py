"""Expand a free-text query with an LLM (OpenAI).

One call turns the user's text into:

- ``paraphrases``: short rewordings, averaged into the text query embedding;
- ``negatives``: look-alike concepts, subtracted from the query embedding;
- ``visual_hints``: scene context, also averaged into the query embedding;
- ``vlm_description`` and ``vlm_exclusions``: what the VLM judge should accept and reject.

The web app stores each expansion and reuses it for repeated identical queries.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any

LOG = logging.getLogger(__name__)

EXPANSION_MODEL = "gpt-5.4-mini"
EXPANSION_COST_USD = 0.005
"""Rough cost of one expansion call, shown in the UI; check your provider's current prices."""

MAX_ITEMS = 8

SYSTEM_PROMPT = """You help build visual search queries for street-level photographs (Google Street View).
The user types a free-text description of something they want to find in city imagery.
Your job is to expand that into structured fields used by a CLIP-style retrieval model and a downstream VLM verifier.
Each field has specific structural constraints (described below) because of how the downstream pipeline consumes them — follow them precisely."""

USER_PROMPT_TEMPLATE = """User query: "{query}"

Produce JSON with exactly these fields:

- "paraphrases": list of 3 to 5 SHORT (3-8 words each) surface-form rephrasings of the same
    visual concept. Vary wording (synonyms, syntactic order); do NOT vary the meaning.
    These are averaged into a single CLIP text embedding, so brevity + lexical diversity
    matter more than completeness.
    Good: ["a discarded tire", "an abandoned roadside tire", "an unmounted tire in the grass"]
    Bad:  ["a detached rubber car tire lying separately on the asphalt near a parked vehicle"]

- "negatives": list of 3 to 5 visually-similar-but-wrong things commonly confused with the query
    in street-view photographs.
    CRITICAL CONSTRAINT: these are used as NEGATION embeddings (subtracted from the positive
    query vector: q' = L2norm(q_pos - 0.20 * q_neg)). If a negative shares the same MAIN NOUN
    as the user's query, we end up subtracting the positive concept too and the retrieval gets
    worse, not better. Every negative MUST use a DIFFERENT main noun than the user query.
    Good (note how the main noun differs from the query's main noun):
      query "dumpster"    (main noun: dumpster) → "a wheeled residential bin"   (main noun: bin)
      query "loose tire"  (main noun: tire)     → "a vehicle parked at the curb" (main noun: vehicle)
      query "storm drain" (main noun: drain)    → "a metal manhole cover"        (main noun: cover)
      query "standing water" (main noun: water) → "wet pavement in shadow"       (main noun: pavement)
    Bad — DO NOT do this (head-noun reused; subtracts positive signal):
      query "tire"        → "a tire mounted on a car"
      query "dumpster"    → "a dumpster being lifted by a truck"
      query "storm drain" → "a clogged storm drain"

- "visual_hints": list of 3 to 5 SHORT (4-10 words) scene-context phrases describing what tends
    to appear AROUND the queried concept in real street-view photos — the setting, surrounding
    objects, typical placement. Helps the CLIP encoder anchor on scene context, not just the
    focal object. Phrasings should NOT just rename the object; they should describe context.
    Good: ["next to a curb after rain", "behind a restaurant in an alley", "near a sidewalk drain inlet"]
    Bad:  ["a tire", "a large metal bin"]  (these rename the object — that belongs in paraphrases)

- "vlm_description": one or two sentences for a VLM verification prompt of the form
    "Does this image clearly show: <vlm_description>". ENUMERATE the specific positive sub-cases
    the user likely means (different appearances, contexts, materials, conditions). Write
    declaratively — do not start with "This image shows...".
    Good for "loose tire": "A tire or tires not attached to a vehicle. Includes discarded tires
    on lawns or streets, stacked tire piles, tire-shop outdoor inventory, and tires used as
    planters or decoration."
    Bad: "This image clearly shows a loose tire: a detached rubber tire lying separately outdoors."
    (the bad version is generic; the good one enumerates concrete sub-cases.)

- "vlm_exclusions": one or two sentences naming SPECIFIC boundary-case objects that look similar
    but should be called NO. Be concrete — name the lookalike object or scenario.
    Good for "loose tire": "Excludes tires currently mounted on a moving or parked car, truck,
    bicycle, or motorcycle. Excludes wheel rims without rubber and tire-shaped marks on pavement."
    Bad: "Do not call yes if the tire is part of something else." (too vague to act on.)

Reply ONLY with valid JSON. No prose, no markdown."""


@dataclass
class Expansion:
    """Structured expansion of one query (see the module docstring)."""

    paraphrases: list[str]
    negatives: list[str]
    visual_hints: list[str]
    vlm_description: str
    vlm_exclusions: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Expansion:
        """Build from an LLM reply or a stored dict, keeping at most `MAX_ITEMS` per list."""
        return cls(
            paraphrases=[str(x) for x in d.get("paraphrases", [])][:MAX_ITEMS],
            negatives=[str(x) for x in d.get("negatives", [])][:MAX_ITEMS],
            visual_hints=[str(x) for x in d.get("visual_hints", [])][:MAX_ITEMS],
            vlm_description=str(d.get("vlm_description", "")).strip(),
            vlm_exclusions=str(d.get("vlm_exclusions", "")).strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Plain dict, as stored in the database."""
        return asdict(self)

    def validate(self) -> None:
        """Raise ``ValueError`` if there are no paraphrases or no VLM description."""
        if not self.paraphrases:
            raise ValueError("expansion has no paraphrases")
        if not self.vlm_description:
            raise ValueError("expansion has no vlm_description")


def _client() -> Any:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError('query expansion needs the llm extra: pip install "vectorview[llm]"') from exc
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    return AsyncOpenAI()


async def expand(query: str, *, model: str = EXPANSION_MODEL, client: Any = None) -> Expansion:
    """Ask the LLM for an `Expansion` of ``query``. Always calls the API; caching is the caller's job.

    Args:
        query: The user's text.
        model: OpenAI chat model.
        client: An ``openai.AsyncOpenAI``-compatible client; created from ``OPENAI_API_KEY`` if omitted.

    Raises:
        RuntimeError: If the ``llm`` extra or the API key is missing, or the reply is not JSON.
        ValueError: If the reply lacks paraphrases or a VLM description.
    """
    client = client or _client()
    LOG.info("expanding query with %s: %r", model, query)
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(query=query)},
        ],
        response_format={"type": "json_object"},
        max_completion_tokens=2000,
    )
    text = resp.choices[0].message.content or ""
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"the expansion reply is not JSON: {text[:200]!r}") from exc
    ex = Expansion.from_dict(doc)
    ex.validate()
    LOG.info(
        "expansion: %d paraphrases, %d negatives, %d hints",
        len(ex.paraphrases),
        len(ex.negatives),
        len(ex.visual_hints),
    )
    return ex
