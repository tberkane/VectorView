import json
from typing import Any

import pytest

from vectorview.web import llm_expand
from vectorview.web.llm_expand import Expansion, expand

from .conftest import Obj

GOOD = {
    "paraphrases": ["a tire", "an old tire"],
    "negatives": ["a car"],
    "visual_hints": ["by a curb"],
    "vlm_description": "A loose tire. ",
    "vlm_exclusions": "Mounted tires.",
}


class FakeOpenAI:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []
        self.chat = Obj(completions=Obj(create=self.create))

    async def create(self, **kw: Any) -> Any:
        self.calls.append(kw)
        return Obj(choices=[Obj(message=Obj(content=self.content))])


async def test_expand_parses_and_validates() -> None:
    client = FakeOpenAI(json.dumps(GOOD))
    ex = await expand("loose tire", client=client)
    assert ex.paraphrases == ["a tire", "an old tire"] and ex.vlm_description == "A loose tire."
    assert '"loose tire"' in client.calls[0]["messages"][1]["content"]
    assert client.calls[0]["response_format"] == {"type": "json_object"}
    assert Expansion.from_dict(ex.to_dict()) == ex


async def test_expand_errors() -> None:
    with pytest.raises(RuntimeError, match="not JSON"):
        await expand("x", client=FakeOpenAI("nope"))
    with pytest.raises(ValueError, match="paraphrases"):
        await expand("x", client=FakeOpenAI(json.dumps({**GOOD, "paraphrases": []})))
    with pytest.raises(ValueError, match="vlm_description"):
        await expand("x", client=FakeOpenAI(json.dumps({**GOOD, "vlm_description": ""})))


def test_from_dict_truncates() -> None:
    ex = Expansion.from_dict({"paraphrases": [str(i) for i in range(20)]})
    assert len(ex.paraphrases) == llm_expand.MAX_ITEMS and ex.negatives == []


def test_client_needs_key(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    fake = types.ModuleType("openai")
    fake.AsyncOpenAI = lambda: "client"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        llm_expand._client()
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert llm_expand._client() == "client"


def test_client_needs_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(RuntimeError, match="llm extra"):
        llm_expand._client()
