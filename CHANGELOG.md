# Changelog

All notable changes to VectorView are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-11-18

First public release.

### Added

- Image collection along OpenStreetMap roads with the Street View Static API, with a file-locked
  hard cap per API key and resumable ingest into Postgres with pgvector (`vectorview ingest`).
- Panorama collection with the `streetlevel` library and perspective crops on CPU or CUDA GPU
  (`vectorview scrape`).
- SigLIP-2 embeddings of images and grid tiles (`vectorview encode`), plus patch regions in
  web-app scans.
- Export of the Postgres corpus to in-memory search caches (`vectorview export`).
- LLM query expansion (OpenAI), stage-1 ranking that fuses image, tile and patch-region scores, and
  VLM reranking with Gemini or OpenAI, with cached verdicts and bounding boxes.
- Local web app (`vectorview serve`): draw an area, build a scan, query, verify results and refine
  them with relevance feedback.
- Documentation site, tests with mocked services, and CI.

[Unreleased]: https://github.com/tberkane/VectorView/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tberkane/VectorView/releases/tag/v0.1.0
