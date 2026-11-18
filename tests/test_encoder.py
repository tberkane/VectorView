import builtins
from typing import Any

import pytest

from vectorview import encoder
from vectorview.encoder import get_encoder, load_encoder, set_encoder

from .conftest import FakeEncoder


def test_load_encoder_explains_missing_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def no_torch(name: str, *args: Any, **kw: Any) -> Any:
        if name == "torch" or name.endswith("siglip"):
            raise ImportError("No module named 'torch'", name="torch")
        return real_import(name, *args, **kw)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    with pytest.raises(RuntimeError, match="encoding"):
        load_encoder()


def test_shared_encoder_is_loaded_once(monkeypatch: pytest.MonkeyPatch) -> None:
    made: list[FakeEncoder] = []

    def fake_load(model_id: str = encoder.MODEL_ID) -> FakeEncoder:
        made.append(FakeEncoder())
        return made[-1]

    monkeypatch.setattr(encoder, "load_encoder", fake_load)
    set_encoder(None)
    try:
        assert get_encoder() is get_encoder() is made[0]
        assert len(made) == 1
    finally:
        set_encoder(None)
