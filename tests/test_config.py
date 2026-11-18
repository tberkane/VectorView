from pathlib import Path

import pytest

from vectorview.config import DEFAULT_IMAGE_CAP, Config, load_config, require_gsv_keys


def test_keys_and_caps_are_read_in_numeric_order() -> None:
    env = {
        "GSV_API_KEY_10": "c",
        "GSV_API_KEY_2": " b ",
        "GSV_API_KEY_1": "a",
        "GSV_IMAGE_CAP_2": "7",
        "GSV_API_KEY_3": "",
    }
    cfg = load_config(env)
    assert list(cfg.gsv_keys) == ["key_1", "key_2", "key_10"]
    assert cfg.gsv_keys["key_2"] == "b"
    assert cfg.gsv_caps == {"key_1": DEFAULT_IMAGE_CAP, "key_2": 7, "key_10": DEFAULT_IMAGE_CAP}


def test_defaults_and_paths(tmp_path: Path) -> None:
    cfg = load_config({"VECTORVIEW_DATA_DIR": str(tmp_path)})
    assert cfg.gsv_keys == {}
    assert cfg.pg_dsn == "postgresql://vectorview:vectorview@127.0.0.1:5432/vectorview"
    assert cfg.cache_dir == tmp_path / "cache"
    assert cfg.osm_cache_dir.parent == cfg.embeddings_dir.parent == cfg.scrape_dir.parent == tmp_path
    assert cfg.quota_state_path.name == "quota_state.json"
    assert cfg.web_db_path.name == "web.db"
    assert cfg.vlm_cache_path.name == "vlm_cache.jsonl"


def test_database_url_overrides_parts() -> None:
    cfg = load_config({"DATABASE_URL": "postgresql:///x", "PG_HOST": "ignored"})
    assert cfg.pg_dsn == "postgresql:///x"


def test_secrets_are_not_in_repr() -> None:
    cfg = Config(data_dir=Path("."), gsv_keys={"key_1": "SECRET1"}, pg_password="SECRET2", openai_api_key="SECRET3")
    assert "SECRET" not in repr(cfg)


def test_dotenv_is_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GSV_API_KEY_1", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("GSV_API_KEY_1=from-dotenv\n")
    assert load_config(dotenv=True).gsv_keys == {"key_1": "from-dotenv"}
    monkeypatch.delenv("GSV_API_KEY_1")


def test_require_gsv_keys() -> None:
    with pytest.raises(RuntimeError, match="GSV_API_KEY_1"):
        require_gsv_keys(Config(data_dir=Path(".")))
    require_gsv_keys(Config(data_dir=Path("."), gsv_keys={"key_1": "k"}))
