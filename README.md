# VectorView

**Text search over street-level imagery.**

[![CI](https://github.com/tberkane/VectorView/actions/workflows/ci.yml/badge.svg)](https://github.com/tberkane/VectorView/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-tberkane.github.io%2FVectorView-blue)](https://tberkane.github.io/VectorView/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

VectorView finds images of a visual feature, described in plain words, among the Street View images
of an area: "a discarded tire on a lawn", "standing water in a roadside ditch", "an overflowing
dumpster". It was built to help map possible mosquito breeding sites, but works for any feature
visible from the street.

It:

1. **collects images** of an area you draw on a map, either along the roads with the Google Street
   View Static API (with a hard per-key image cap) or for every panorama in the area with the
   [`streetlevel`](https://github.com/sk-zk/streetlevel) library;
2. **embeds** each image, its grid tiles and its patch regions with
   [SigLIP-2](https://huggingface.co/google/siglip2-so400m-patch16-512), so small objects can be found;
3. **expands your query** with an LLM into paraphrases, scene context and look-alikes to subtract;
4. **ranks** the images and **reranks** the top ones with a vision-language model (Gemini or OpenAI),
   which answers yes/no with a reason and a bounding box;
5. lets you **verify** results in a local web app and **refine** the query from your yes/no
   verdicts (relevance feedback).

Collections can also be built from the command line into Postgres with pgvector, then exported to
the web app.

## Install

```bash
# 1. Install the PyTorch build for your machine: https://pytorch.org/get-started/locally/
# 2. Install VectorView with the extras you need:
pip install "vectorview[encoding,llm,web] @ git+https://github.com/tberkane/VectorView@v0.1.0"
# add "scraper" to the extras for the streetlevel source
```

Python 3.11 or later, on Linux or macOS. Do not `pip install vectorview` from PyPI: that name belongs to
an unrelated package.

| Extra | For |
|---|---|
| `encoding` | the SigLIP-2 encoder (torch, transformers); a GPU helps but is not required |
| `llm` | query expansion (OpenAI) and the VLM judge (Gemini or OpenAI) |
| `web` | the web app (FastAPI) |
| `scraper` | the `streetlevel` source |

Keys go in environment variables or a `.env` file in the working directory (see
[`.env.example`](.env.example)): `OPENAI_API_KEY`, `GEMINI_API_KEY`, and `GSV_API_KEY_1` (plus
`GSV_IMAGE_CAP_1`, default 1,000 images) for the Static API.

## Quickstart

```bash
vectorview serve        # http://127.0.0.1:8765, run where your .env is
```

In the browser: **New scan**, draw a rectangle, choose the image source and submit. The scan
collects and embeds the images, which takes a while: the page shows its progress and an estimate
of the image count and API cost before you start. Then type a query, review the results on the map
and grid, mark them yes/no, and **Refine**.

Command-line pipeline (needs Postgres with pgvector; in a clone of this repository,
`db/start_pg.sh` starts a local server and creates the tables from `db/init.sql`):

```bash
vectorview ingest --name cambridge --bbox -71.12,42.36,-71.10,42.38 --max-images 500
vectorview encode && vectorview encode --tiles
vectorview export --tag cambridge   # adds the collection to the web app
```

See the [documentation](https://tberkane.github.io/VectorView/) for how ranking works, the CLI and
the Python API.

## Limitations

- Results are only as good as SigLIP-2 and the VLM judge: expect misses and false positives, and
  verify what matters. No accuracy figures are claimed here.
- The web app is single-user and has no authentication; it binds to 127.0.0.1 by default.
- Street View coverage, capture dates and image quality vary by place.
- Ranking is a brute-force matrix product over embeddings held in memory: about 23 KB per image
  with 2×2 tiles, plus about 150 KB per image for the patch regions of web-app scans.
- Each query costs LLM and VLM calls, and the Static API charges per image; the UI shows rough
  estimates only.

## Development

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check . && mypy && pytest
```

Tests need no API keys, GPU, model weights or network. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation and license

If you use VectorView, please cite it with [CITATION.cff](CITATION.cff). Apache License 2.0; see
[LICENSE](LICENSE) and [NOTICE](NOTICE).
