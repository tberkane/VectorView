# Quickstart

## 1. Configure

VectorView reads settings from environment variables, or from a `.env` file in the directory you
run it from:

```bash
OPENAI_API_KEY=...          # query expansion (and the OpenAI judge)
GEMINI_API_KEY=...          # the default VLM judge
GSV_API_KEY_1=...           # Street View Static API (API source only)
GSV_IMAGE_CAP_1=1000        # hard cap on images fetched with that key
VECTORVIEW_DATA_DIR=./data  # where images, embeddings and state are written
```

More keys can be added as `GSV_API_KEY_2`, `GSV_API_KEY_3`, ...; they are used in order, moving to
the next when one reaches its cap. Metadata requests are free and are not counted.

## 2. Search an area in the web app

```bash
vectorview serve            # http://127.0.0.1:8765
```

1. **New scan**: draw a rectangle on the map, pick the source and the road-point spacing. The
   preview estimates the number of images, the API cost and the time.
2. The scan collects the images, then embeds them, their tiles and their patch regions. The page
   shows each step's progress; the first scan also downloads the SigLIP-2 weights (several GB) from Hugging Face.
3. Type a query, set *K* (how many top images the VLM judges) and optionally a minimum capture date.
4. Review the results on the map and grid. **Verify** walks through them one by one (yes / no /
   unsure); **Refine** reruns the query using your verdicts.

## 3. Or build a collection from the command line

This path stores the corpus in Postgres with pgvector. In a clone of the repository,
`db/start_pg.sh` starts a local server (it needs `pg_ctl` and the pgvector extension, e.g. from
conda-forge or Homebrew) and applies `db/init.sql`.

```bash
vectorview ingest --name cambridge --bbox -71.12,42.36,-71.10,42.38 --max-images 500
vectorview encode            # whole-image embeddings
vectorview encode --tiles    # 2x2 tile embeddings
vectorview export --tag cambridge
vectorview serve             # the collection appears in the scan list
```

`ingest` and `encode` are resumable: rerun them with the same arguments after an interruption.
`vectorview quota` prints how many images each key has fetched.

## 4. From Python

```python
from vectorview.config import load_config
from vectorview.gsv import open_client
from vectorview.ingest import ingest_region
from vectorview.regions import BBox

cfg = load_config(dotenv=True)
with open_client(cfg) as client:
    stats = ingest_region(cfg, "cambridge", BBox.parse("-71.12,42.36,-71.10,42.38"), client, max_images=100)
print(stats.images_kept)
```

The [API reference](api.md) lists the public functions.
