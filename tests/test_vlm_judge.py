import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from PIL import Image

from vectorview.web import vlm_judge
from vectorview.web.vlm_judge import GeminiJudge, OpenAIJudge, build_judge, parse_bbox, rerank, vlm_score

from .conftest import Obj, make_jpeg


@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=4, max_size=4)
)
def test_parse_bbox_invariants(raw: list[float]) -> None:
    box = parse_bbox(raw)
    if box is not None:
        assert all(0 <= v <= 1000 for v in box)
        assert box[2] > box[0] and box[3] > box[1]


@pytest.mark.parametrize("raw", [None, [1, 2, 3], "abcd", [1, 2, "x", 4], [5, 5, 5, 5], [500, 0, 100, 10]])
def test_parse_bbox_rejects(raw: Any) -> None:
    assert parse_bbox(raw) is None


@given(st.sampled_from(["yes", "no", "unsure", "maybe"]), st.one_of(st.floats(), st.text(max_size=3), st.none()))
def test_vlm_score_in_unit_interval(decision: str, conf: Any) -> None:
    assert 0.0 <= vlm_score({"decision": decision, "confidence": conf}) <= 1.0


def test_vlm_score_values() -> None:
    assert vlm_score({"decision": "yes", "confidence": 0.8}) == 0.8
    assert vlm_score({"decision": "unsure", "confidence": 0.8}) == 0.4
    assert vlm_score({"decision": "no", "confidence": 0.8}) == 0.0
    assert vlm_score({}) == 0.0


def test_encode_image_shrinks(tmp_path: Path) -> None:
    import io

    p = make_jpeg(tmp_path / "a.jpg", size=(200, 100))
    out = Image.open(io.BytesIO(vlm_judge.encode_image(p, max_side=50)))
    assert out.size == (50, 25)
    assert Image.open(io.BytesIO(vlm_judge.encode_image(p, max_side=500))).size == (200, 100)


class ScriptedJudge:
    model = "gemini-test"

    def __init__(self) -> None:
        self.calls = 0

    async def judge(self, image_bytes: bytes, prompt: str) -> dict[str, Any]:
        self.calls += 1
        assert "User is looking for: tire" in prompt
        if self.calls == 2:
            raise RuntimeError("rate limited")
        return {"decision": "yes", "confidence": 0.1 * self.calls, "reason": "r", "bbox": [0, 0, 500, 500]}


async def test_rerank_caches_verdicts_but_not_errors(tmp_path: Path) -> None:
    cands = [(f"i{n}", str(make_jpeg(tmp_path / f"{n}.jpg", seed=n)), 1.0 - n / 10) for n in range(3)]
    cache_path = tmp_path / "vlm.jsonl"
    judge = ScriptedJudge()
    out = await rerank(
        query="tire", description="", exclusions="", candidates=cands, cache_path=cache_path, judge=judge, concurrency=1
    )
    assert judge.calls == 3 and len(out) == 3
    assert [r["vlm_score"] for r in out] == sorted((r["vlm_score"] for r in out), reverse=True)
    errored = [r for r in out if r["vlm_verdict"]["reason"].startswith("judge error")]
    assert len(errored) == 1 and errored[0]["vlm_score"] == 0.0
    assert len(cache_path.read_text().splitlines()) == 2
    cache_path.write_text(cache_path.read_text() + "garbage\n")

    again = await rerank(
        query="tire", description="", exclusions="", candidates=cands, cache_path=cache_path, judge=judge
    )
    assert judge.calls == 4  # only the failed one is judged again
    hits = {r["image_id"]: r["cache_hit"] for r in again}
    assert sum(hits.values()) == 2
    assert all(r["bbox"] == [0, 0, 500, 500] for r in again if r["cache_hit"])


async def test_rerank_without_misses_builds_no_judge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vlm_judge, "build_judge", lambda m: pytest.fail("no judge needed"))
    assert await rerank(query="q", description="d", exclusions="e", candidates=[], cache_path=tmp_path / "c") == []


def test_build_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vlm_judge, "GeminiJudge", lambda model: ("gemini", model))
    monkeypatch.setattr(vlm_judge, "OpenAIJudge", lambda model: ("openai", model))
    assert build_judge("gemini-x") == ("gemini", "gemini-x")
    assert build_judge("gpt-x") == ("openai", "gpt-x")
    with pytest.raises(ValueError):
        build_judge("llava")


@pytest.fixture
def fake_genai(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    gtypes = types.ModuleType("google.genai.types")
    gtypes.Part = Obj(from_bytes=lambda data, mime_type: ("part", mime_type))  # type: ignore[attr-defined]
    gtypes.GenerateContentConfig = lambda **kw: kw  # type: ignore[attr-defined]
    genai = types.ModuleType("google.genai")
    genai.types = gtypes  # type: ignore[attr-defined]
    genai.Client = lambda api_key: Obj(api_key=api_key)  # type: ignore[attr-defined]
    google = types.ModuleType("google")
    google.genai = genai  # type: ignore[attr-defined]
    for name, mod in {"google": google, "google.genai": genai, "google.genai.types": gtypes}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return seen


async def test_gemini_judge(fake_genai: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    async def generate(**kw: Any) -> Any:
        fake_genai.update(kw)
        return Obj(text=json.dumps({"decision": "no", "confidence": 0.9}))

    client = Obj(aio=Obj(models=Obj(generate_content=generate)))
    j = GeminiJudge("gemini-test", client=client)
    assert await j.judge(b"img", "prompt") == {"decision": "no", "confidence": 0.9}
    assert fake_genai["config"]["response_schema"] == GeminiJudge.RESPONSE_SCHEMA
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiJudge()
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert GeminiJudge().client.api_key == "k"

    async def bad(**kw: Any) -> Any:
        return Obj(text="[1]")

    j.client = Obj(aio=Obj(models=Obj(generate_content=bad)))
    with pytest.raises(RuntimeError, match="not a JSON object"):
        await j.judge(b"img", "p")


def test_gemini_needs_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "google.genai", None)
    with pytest.raises(RuntimeError, match="llm extra"):
        GeminiJudge()


async def test_openai_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[Any] = []

    async def create(**kw: Any) -> Any:
        sent.append(kw)
        return Obj(choices=[Obj(message=Obj(content="not json"))])

    j = OpenAIJudge(client=Obj(chat=Obj(completions=Obj(create=create))))
    with pytest.raises(RuntimeError, match="not JSON"):
        await j.judge(b"img", "p")
    assert sent[0]["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    fake = types.ModuleType("openai")
    fake.AsyncOpenAI = lambda: "client"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIJudge()
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert OpenAIJudge().client == "client"
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(RuntimeError, match="llm extra"):
        OpenAIJudge()
