# API reference

## Configuration

::: vectorview.config

::: vectorview.regions

## Collecting images

::: vectorview.gsv

::: vectorview.osm

::: vectorview.ingest
    options:
      members: [ingest_region, fetch_views, FetchedView, IngestStats, HEADING_OFFSETS]

::: vectorview.scraper
    options:
      members: [GSVScraper, install_pyexiv2_shim, yaw_for_heading, lonlat_to_tile, project_one]

## Embedding

::: vectorview.encoder

::: vectorview.siglip

::: vectorview.tile

::: vectorview.encode

::: vectorview.cache

::: vectorview.db

## Searching

::: vectorview.web.llm_expand
    options:
      members: [Expansion, expand, EXPANSION_MODEL]

::: vectorview.web.retrieval

::: vectorview.web.vlm_judge
    options:
      members: [rerank, GeminiJudge, OpenAIJudge, Judge, build_judge, build_prompt, vlm_score, confidence, parse_bbox]

::: vectorview.web.query_pipeline

## Web app

::: vectorview.web.app
    options:
      members: [create_app, serve]

::: vectorview.web.scan_worker
    options:
      members: [build_scan, run_scan, ingest_api, ingest_scraper]

::: vectorview.web.manifest

::: vectorview.web.db
