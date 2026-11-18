"""Runtime configuration, read from environment variables (and optionally a ``.env`` file)."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_IMAGE_CAP = 1_000
"""Images a Street View API key may fetch when ``GSV_IMAGE_CAP_<n>`` is not set."""

_KEY_RE = re.compile(r"^GSV_API_KEY_(\d+)$")


@dataclass(frozen=True)
class Config:
    """Settings shared by the CLI and the web app.

    Attributes:
        data_dir: Root for every file VectorView writes (images, embeddings, state).
        gsv_keys: Street View Static API keys by slot name (``"key_1"``, ``"key_2"``, ...).
        gsv_caps: Hard cap on image fetches per slot, enforced by `vectorview.gsv.QuotaLedger`.
        pg_host: Postgres host.
        pg_port: Postgres port.
        pg_db: Postgres database name.
        pg_user: Postgres user.
        pg_password: Postgres password.
        openai_api_key: Key for LLM query expansion and the OpenAI judge.
        gemini_api_key: Key for the Gemini judge.
        database_url: Full Postgres connection string; overrides the ``pg_*`` fields when set.
    """

    data_dir: Path
    gsv_keys: dict[str, str] = field(default_factory=dict, repr=False)
    gsv_caps: dict[str, int] = field(default_factory=dict)
    pg_host: str = "127.0.0.1"
    pg_port: int = 5432
    pg_db: str = "vectorview"
    pg_user: str = "vectorview"
    pg_password: str = field(default="vectorview", repr=False)
    openai_api_key: str = field(default="", repr=False)
    gemini_api_key: str = field(default="", repr=False)
    database_url: str = field(default="", repr=False)

    @property
    def pg_dsn(self) -> str:
        """Connection string for psycopg (``database_url`` if set)."""
        if self.database_url:
            return self.database_url
        return f"postgresql://{self.pg_user}:{self.pg_password}@{self.pg_host}:{self.pg_port}/{self.pg_db}"

    @property
    def cache_dir(self) -> Path:
        """Downloaded JPEGs, sharded by the first two characters of the image id."""
        return self.data_dir / "cache"

    @property
    def osm_cache_dir(self) -> Path:
        """Cached OpenStreetMap road graphs (GraphML)."""
        return self.data_dir / "osm"

    @property
    def embeddings_dir(self) -> Path:
        """Numpy embedding caches, one folder per tag."""
        return self.data_dir / "embeddings"

    @property
    def scrape_dir(self) -> Path:
        """Working directories of scraper-mode scans."""
        return self.data_dir / "scrape"

    @property
    def quota_state_path(self) -> Path:
        """JSON file holding the per-key image counts."""
        return self.data_dir / "quota_state.json"

    @property
    def web_db_path(self) -> Path:
        """SQLite database of the web app (scans, queries, verifications)."""
        return self.data_dir / "web.db"

    @property
    def vlm_cache_path(self) -> Path:
        """JSON-lines cache of VLM verdicts."""
        return self.data_dir / "vlm_cache.jsonl"


def load_config(env: Mapping[str, str] | None = None, *, dotenv: bool = False) -> Config:
    """Build a `Config` from environment variables.

    Street View keys are read from ``GSV_API_KEY_1``, ``GSV_API_KEY_2``, ... and their caps from
    ``GSV_IMAGE_CAP_<n>`` (default `DEFAULT_IMAGE_CAP`). Files go under ``VECTORVIEW_DATA_DIR``
    (default ``./data``). Postgres is reached through ``DATABASE_URL`` or ``PG_HOST``, ``PG_PORT``,
    ``PG_DB``, ``PG_USER`` and ``PG_PASSWORD``.

    Args:
        env: Variables to read; defaults to ``os.environ``.
        dotenv: Load ``.env`` from the working directory first, without overriding variables already set.

    Returns:
        The configuration.
    """
    if dotenv:
        from dotenv import find_dotenv, load_dotenv

        load_dotenv(find_dotenv(usecwd=True))
    if env is None:
        env = os.environ

    keys: dict[str, str] = {}
    caps: dict[str, int] = {}
    numbered = sorted((int(m.group(1)), name) for name in env if (m := _KEY_RE.match(name)))
    for n, name in numbered:
        value = env[name].strip()
        if value:
            keys[f"key_{n}"] = value
            caps[f"key_{n}"] = int(env.get(f"GSV_IMAGE_CAP_{n}", DEFAULT_IMAGE_CAP))

    return Config(
        data_dir=Path(env.get("VECTORVIEW_DATA_DIR", "./data")).resolve(),
        gsv_keys=keys,
        gsv_caps=caps,
        pg_host=env.get("PG_HOST", "127.0.0.1"),
        pg_port=int(env.get("PG_PORT", "5432")),
        pg_db=env.get("PG_DB", "vectorview"),
        pg_user=env.get("PG_USER", "vectorview"),
        pg_password=env.get("PG_PASSWORD", "vectorview"),
        openai_api_key=env.get("OPENAI_API_KEY", "").strip(),
        gemini_api_key=env.get("GEMINI_API_KEY", "").strip(),
        database_url=env.get("DATABASE_URL", "").strip(),
    )


def require_gsv_keys(cfg: Config) -> None:
    """Raise ``RuntimeError`` unless at least one Street View API key is configured."""
    if not cfg.gsv_keys:
        raise RuntimeError("no Street View API key configured; set GSV_API_KEY_1 (and optionally _2, _3, ...)")
