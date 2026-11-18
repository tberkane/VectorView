"""Shared fixtures. Nothing here needs API keys, a GPU, model weights or network access."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from vectorview import encoder as encoder_mod
from vectorview.config import Config, load_config
from vectorview.web import db as web_db

DIM = 16


def _unit(seed: bytes, dim: int = DIM) -> np.ndarray:
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(seed).digest()[:8], "little"))
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class FakeEncoder:
    """Deterministic stand-in for SigLIP-2: embeddings derived from pixel bytes or text."""

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim
        self.image_calls = 0
        self.text_calls = 0

    def encode_images(self, images: Iterable[Image.Image]) -> np.ndarray:
        self.image_calls += 1
        imgs = list(images)
        return np.stack([_unit(im.tobytes(), self.dim) for im in imgs]) if imgs else np.empty((0, self.dim), np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        self.text_calls += 1
        return np.stack([_unit(t.encode(), self.dim) for t in texts]) if texts else np.empty((0, self.dim), np.float32)

    def encode_patch_regions(self, images: Iterable[Image.Image], k: int = 8) -> np.ndarray:
        imgs = list(images)
        out = np.stack([[_unit(im.tobytes() + bytes([i]), self.dim) for i in range(k * k)] for im in imgs])
        return out.astype(np.float16)


@pytest.fixture(autouse=True)
def _fresh_scan_caches() -> Iterator[None]:
    from vectorview.web import retrieval

    retrieval.clear_scan_caches()
    yield
    retrieval.clear_scan_caches()


@pytest.fixture
def fake_encoder() -> Iterator[FakeEncoder]:
    enc = FakeEncoder()
    encoder_mod.set_encoder(enc)
    yield enc
    encoder_mod.set_encoder(None)


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    """Configuration rooted in a temporary directory, with two fake API keys."""
    dsn = os.environ.get("VECTORVIEW_TEST_PG_DSN")
    for name in list(os.environ):
        if name.startswith(("GSV_", "PG_", "VECTORVIEW_")) or name in (
            "DATABASE_URL",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
        ):
            monkeypatch.delenv(name)
    env = {
        "VECTORVIEW_DATA_DIR": str(tmp_path / "data"),
        "GSV_API_KEY_1": "key-one",
        "GSV_IMAGE_CAP_1": "4",
        "GSV_API_KEY_2": "key-two",
        "GSV_IMAGE_CAP_2": "100",
    }
    if dsn:
        env["DATABASE_URL"] = dsn
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return load_config()


@pytest.fixture
def web(cfg: Config) -> Config:
    """A fresh web database."""
    web_db.init(cfg.web_db_path)
    return cfg


def make_jpeg(path: Path, seed: int = 0, size: tuple[int, int] = (64, 48)) -> Path:
    """Write a small random JPEG."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, "JPEG")
    return path


def jpeg_bytes(seed: int = 0) -> bytes:
    import io

    buf = io.BytesIO()
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(buf, "JPEG")
    return buf.getvalue()


class Obj:
    """Attribute bag for fake SDK responses."""

    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


@pytest.fixture
def corpus_scan(web: Config, fake_encoder: FakeEncoder) -> dict[str, Any]:
    """A ready scan over 6 cached JPEGs with image, tile and manifest caches."""
    from vectorview import cache
    from vectorview.gsv import cache_path_for
    from vectorview.tile import cut_tiles
    from vectorview.web import retrieval
    from vectorview.web.manifest import ImageMeta, manifest_path, write_manifest

    retrieval.clear_scan_caches()
    rows = []
    imgs = []
    for i in range(6):
        iid = f"pano{i}_h{90 * (i % 2):03d}"
        path = make_jpeg(cache_path_for(web.cache_dir, iid), seed=i)
        region = "north" if i < 4 else "south"
        captured = "" if i == 5 else f"202{i}-01-01"
        rows.append(ImageMeta(iid, f"pano{i}", 42.0 + i / 100, -71.0, captured, 90.0 * (i % 2), str(path), region))
        imgs.append(Image.open(path).convert("RGB"))
    root = web.embeddings_dir
    cache.write_images(
        root, "corp", fake_encoder.encode_images(imgs), [r.image_id for r in rows], [r.region for r in rows]
    )
    write_manifest(manifest_path(root, "corp"), rows)
    tiles = [
        (f"{r.image_id}_{t.suffix}", r.image_id, t.image)
        for r, im in zip(rows[:5], imgs[:5], strict=True)
        for t in cut_tiles(im, 2)
    ]
    cache.write_tiles(
        root,
        "corp_tiles",
        fake_encoder.encode_images([t[2] for t in tiles]),
        [t[0] for t in tiles],
        [t[1] for t in tiles],
    )
    sid = web_db.create_scan(
        name="corp", bbox=None, embedding_tag="corp", tile_tag="corp_tiles", fetch_mode="corpus", status="ready"
    )
    scan = web_db.get_scan(sid)
    assert scan is not None
    return scan
