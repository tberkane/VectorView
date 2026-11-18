# Contributing

Thanks for your interest in VectorView. Issues and pull requests are welcome.

## Development setup

```bash
git clone https://github.com/tberkane/VectorView && cd VectorView
python -m venv .venv && source .venv/bin/activate     # Python ≥ 3.11
pip install -e ".[dev]"
```

The tests do not need torch; install the `encoding` extra only to run the real encoder.

## Checks (the same ones CI runs)

```bash
ruff check . && ruff format --check .   # lint and formatting
mypy                                     # type checks
pytest                                   # tests with branch coverage
mkdocs build --strict                    # documentation
```

Tests marked `postgres` run against a real Postgres server with pgvector when
`VECTORVIEW_TEST_PG_DSN` is set (CI does this on Ubuntu), and are skipped otherwise:

```bash
VECTORVIEW_TEST_PG_DSN=postgresql://user:pass@localhost:5432/vectorview_test pytest
```

The test database's VectorView tables are dropped and recreated.

## Guidelines

- **Tests run offline.** They must not need API keys, a GPU, model weights or network access: use
  the fakes in `tests/conftest.py` (encoder) and `httpx.MockTransport` or fake modules for HTTP,
  `streetlevel`, OpenAI and Gemini.
- Code that needs torch, model weights or a CUDA GPU (`siglip.py`, `_gpu_project.py`) is excluded
  from coverage; keep it thin.
- **Never commit** API keys, `.env` files, quota state, or Street View images.
- **Public API** changes need type hints, docstrings and a CHANGELOG entry under "Unreleased".
- Every pull request is reviewed before merging.

## Reporting security issues

Please follow [SECURITY.md](SECURITY.md) instead of opening a public issue.
