# CLI

All commands read the environment and `.env` (see the [quickstart](quickstart.md)), log to stderr
and print a JSON summary to stdout. Run `vectorview <command> --help` for every option.

| Command | Does |
|---|---|
| `vectorview ingest --name N --bbox W,S,E,N [--max-images M] [--spacing 50] [--workers 4]` | Collect Static API images along the roads of a box into Postgres (resumable). |
| `vectorview encode [--region R] [--batch-size 32]` | Embed stored images that have no embedding yet. |
| `vectorview encode --tiles [--scales 2] [--overlaps 0]` | Embed grid tiles of the stored images. |
| `vectorview export --tag T [--region R] [--no-register]` | Copy the corpus into numpy caches `T` and `T_tiles` and add it to the web app. |
| `vectorview serve [--host 127.0.0.1] [--port 8765]` | Start the web app. |
| `vectorview scrape --data-dir D [--place P \| --bbox W,S,E,N] STEP` | Run one scraper step: `discover`, `enrich`, `download`, `project`, `export` or `stats`. |
| `vectorview quota` | Print images fetched and the cap of each key. |

Exit codes: 0 on success, 1 on a configuration or runtime error (the message is logged), 2 on bad
arguments.

::: vectorview.cli
    options:
      members: [main, build_parser]
      show_root_heading: false
