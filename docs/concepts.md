# How it works

## Collecting images

**Static API source** (`vectorview ingest`, or *Street View API* in the web app). VectorView downloads
the drivable road network of the area from OpenStreetMap and places a point every 50 m (by default)
along each road. For each point it asks the Street View metadata endpoint for the nearest outdoor
panorama, skips panoramas it already has, and fetches two 640×640 views looking forward and backward
along the road.

Every image request is counted in a JSON ledger (`quota_state.json`) under a file lock *before* it is
sent, and never refunded, so concurrent processes and crashes cannot push a key past its cap.

**Scraper source** (`vectorview scrape`, or *Scraper* in the web app). The
[`streetlevel`](https://github.com/sk-zk/streetlevel) library lists every panorama in the area's
coverage tiles, downloads the full equirectangular panoramas, and VectorView cuts four perspective
crops per panorama at fixed compass headings (on a CUDA GPU with `pyequilib` when available, else
with `py360convert`). This source does not use the Static API: it needs no key and has no quota
ledger. It finds panoramas off the road network too.

## Embedding

Images are embedded with SigLIP-2 (`google/siglip2-so400m-patch16-512`, 1152 dimensions), which
places images and text in one space. Each image gets:

- one **image** embedding;
- one embedding per **tile** of a 2×2 grid;
- in web-app scans, one embedding per **patch region**: SigLIP-2's attention-pooling head applied to
  each block of an 8×8 grid over the model's patch features, which keeps those embeddings comparable
  with text.

Tiles and patch regions let a small object score high even when it fills a small part of the image.

## Ranking a query

1. **Expansion.** An LLM rewrites the query into short paraphrases, scene-context phrases, look-alike
   concepts ("negatives") and a description of what the judge should accept and reject. Expansions
   are stored and reused for identical queries.
2. **Query vector.** `q = L2(mean(paraphrases + hints) − 0.2 · mean(negatives))`.
3. **Stage 1.** Each image scores `0.55 · image + 0.30 · best tile + 0.15 · best patch region`
   (`0.7 · image + 0.3 · best tile` without patch regions); the top 200 are kept, optionally only
   those captured after a date.
4. **Rerank.** The top *K* images go to a VLM judge (Gemini by default, or OpenAI) with the query and
   the expansion's description. It answers `yes`, `no` or `unsure` with a confidence, a reason and a
   bounding box. Results are sorted by the judge's score, then the stage-1 score. Verdicts are cached
   by prompt, image and model.
5. **Refine.** After you mark results, the query vector moves toward the images marked yes and, once
   at least five are marked no, away from those (Rocchio feedback); stage 1 and the rerank run again,
   reusing cached verdicts.

The weights above are fixed defaults in [`vectorview.web.retrieval`](api.md#vectorview.web.retrieval),
the same for every query; `stage1_rank` accepts others.

## Where things are stored

| What | Where |
|---|---|
| JPEGs | `<data_dir>/cache/<2 chars>/<image_id>.jpg` |
| Embedding caches and manifests | `<data_dir>/embeddings/<tag>/` |
| Web app state (scans, queries, verdicts) | `<data_dir>/web.db` (SQLite) |
| VLM verdict cache | `<data_dir>/vlm_cache.jsonl` |
| Quota ledger | `<data_dir>/quota_state.json` |
| Command-line corpus | Postgres (`db/init.sql`) |

## Limitations

- **Accuracy.** Results depend on SigLIP-2 and on the VLM judge. Expect misses (the target is in an
  image ranked below *K*) and false positives (look-alikes the judge accepts). VectorView reports
  no accuracy figures; verify the results you rely on.
- **Coverage.** Street View coverage, capture dates and image quality vary by place; areas without
  coverage return nothing.
- **Cost.** The Static API charges per image, and every query makes one LLM call (unless cached)
  plus up to *K* VLM calls. The web app shows rough estimates; check current prices.
- **Scale.** Ranking is a brute-force matrix product over embeddings held in memory: about 23 KB per
  image with 2×2 tiles, plus about 150 KB per image for patch regions.
- **Single user.** The web app has no authentication and binds to 127.0.0.1 by default; its worker
  runs one job at a time.
- **Platforms.** Linux and macOS (the quota ledger uses POSIX file locks).
