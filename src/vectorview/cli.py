"""Command-line interface: ``vectorview <command>``.

Commands:

- ``ingest``: collect Street View images along the roads of a box into Postgres;
- ``encode``: embed the stored images (and, with ``--tiles``, their grid tiles);
- ``export``: copy the Postgres corpus into numpy caches and add it to the web app;
- ``serve``: start the web app;
- ``scrape``: run one step of the `streetlevel` scraper over a place or box;
- ``quota``: show how many images each API key has fetched.

Settings come from environment variables and ``.env`` (see `vectorview.config`).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence

from . import __version__
from .config import Config, load_config
from .regions import BBox


def _bbox(text: str) -> BBox:
    try:
        return BBox.parse(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x.strip())


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in text.split(",") if x.strip())


def build_parser() -> argparse.ArgumentParser:
    """The argument parser of ``vectorview``."""
    p = argparse.ArgumentParser(prog="vectorview", description="Text search over street-level imagery.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--log-level", default="INFO", help="logging level (default INFO)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("ingest", help="collect Street View images into Postgres")
    s.add_argument("--name", required=True, help="region name; reuse it to resume")
    s.add_argument("--bbox", required=True, type=_bbox, help="west,south,east,north in degrees")
    s.add_argument("--max-images", type=int, default=None, help="stop once the region holds this many images")
    s.add_argument("--spacing", type=float, default=50.0, help="metres between road points (default 50)")
    s.add_argument("--workers", type=int, default=4, help="concurrent requests (default 4)")

    s = sub.add_parser("encode", help="embed stored images or tiles")
    s.add_argument("--tiles", action="store_true", help="embed grid tiles instead of whole images")
    s.add_argument("--scales", type=_ints, default=(2,), help="grid sides for --tiles (default 2)")
    s.add_argument("--overlaps", type=_floats, default=(0.0,), help="tile overlap fractions (default 0)")
    s.add_argument("--batch-size", type=int, default=32)
    s.add_argument("--region", default=None, help="only this region")

    s = sub.add_parser("export", help="copy the corpus into numpy caches and add it to the web app")
    s.add_argument("--tag", required=True, help="name of the cache to write")
    s.add_argument("--region", default=None, help="only this region")
    s.add_argument("--no-register", action="store_true", help="do not add the corpus to the web app")

    s = sub.add_parser("serve", help="start the web app")
    s.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1, this machine only)")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--reload", action="store_true", help="reload on code changes (development)")

    s = sub.add_parser("scrape", help="collect panoramas with streetlevel (scraper extra)")
    area = s.add_mutually_exclusive_group()
    area.add_argument("--place", help="place name geocoded with OpenStreetMap")
    area.add_argument("--bbox", type=_bbox, help="west,south,east,north in degrees")
    s.add_argument("--data-dir", required=True, help="working directory (state database, images)")
    s.add_argument("--concurrency", type=int, default=16)
    steps = s.add_subparsers(dest="step", required=True)
    steps.add_parser("discover", help="list panoramas in the area (needs --place or --bbox)")
    steps.add_parser("enrich", help="fetch panorama metadata")
    d = steps.add_parser("download", help="download equirectangular panoramas")
    d.add_argument("--zoom", type=int, default=3, help="0-5 (default 3)")
    d.add_argument("--limit", type=int, default=None)
    pr = steps.add_parser("project", help="cut perspective crops")
    pr.add_argument("--headings", type=_floats, default=(0.0, 90.0, 180.0, 270.0), help="compass bearings")
    pr.add_argument("--fov", type=float, default=90.0)
    pr.add_argument("--pitch", type=float, default=-10.0)
    pr.add_argument("--size", type=int, default=512)
    pr.add_argument("--workers", type=int, default=None)
    pr.add_argument("--limit", type=int, default=None)
    pr.add_argument("--cpu", action="store_true", help="do not try the GPU path")
    e = steps.add_parser("export", help="write panorama metadata to CSV")
    e.add_argument("--out", default="panoramas.csv")
    steps.add_parser("stats", help="print progress counters as JSON")

    sub.add_parser("quota", help="print images fetched per API key as JSON")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    cfg = load_config(dotenv=True)
    try:
        return _run(args, cfg)
    except RuntimeError as exc:
        logging.getLogger("vectorview").error("%s", exc)
        return 1


def _run(args: argparse.Namespace, cfg: Config) -> int:
    if args.command == "ingest":
        from .config import require_gsv_keys
        from .gsv import open_client
        from .ingest import ingest_region

        require_gsv_keys(cfg)
        with open_client(cfg) as client:
            stats = ingest_region(
                cfg,
                args.name,
                args.bbox,
                client,
                max_images=args.max_images,
                spacing_m=args.spacing,
                workers=args.workers,
            )
        print(json.dumps(stats.__dict__))
    elif args.command == "encode":
        from .encode import encode_images, encode_tiles

        if args.tiles:
            n = encode_tiles(
                cfg, scales=args.scales, overlaps=args.overlaps, batch_size=args.batch_size, region=args.region
            )
        else:
            n = encode_images(cfg, batch_size=args.batch_size, region=args.region)
        print(json.dumps({"written": n}))
    elif args.command == "export":
        from .cache import export_corpus

        n_img, n_tiles = export_corpus(cfg, args.tag, region=args.region)
        out: dict[str, object] = {"tag": args.tag, "images": n_img, "tiles": n_tiles}
        if not args.no_register:
            from .web import db as web_db

            web_db.init(cfg.web_db_path)
            out["scan_id"] = web_db.create_scan(
                name=args.tag,
                bbox=None,
                embedding_tag=args.tag,
                tile_tag=f"{args.tag}_tiles",
                fetch_mode="corpus",
                status="ready",
            )
        print(json.dumps(out))
    elif args.command == "serve":
        from .web.app import serve

        serve(args.host, args.port, reload=args.reload, log_level=args.log_level.lower())
    elif args.command == "scrape":
        return _scrape(args)
    elif args.command == "quota":
        from .gsv import QuotaLedger

        snap = QuotaLedger(cfg.quota_state_path, cfg.gsv_caps).snapshot()
        print(json.dumps({slot: {"used": snap.get(slot, 0), "cap": cap} for slot, cap in cfg.gsv_caps.items()}))
    return 0


def _scrape(args: argparse.Namespace) -> int:
    from .scraper import GSVScraper

    polygon = None
    if args.bbox is not None:
        from shapely.geometry import box

        polygon = box(*args.bbox.as_tuple())
    s = GSVScraper(
        args.data_dir,
        place=args.place,
        polygon=polygon,
        concurrency=args.concurrency,
        gpu=not getattr(args, "cpu", False),
    )
    if args.step == "discover":
        if not (args.place or polygon is not None):
            raise RuntimeError("discover needs --place or --bbox")
        s.discover()
    elif args.step == "enrich":
        s.enrich()
    elif args.step == "download":
        s.download(zoom=args.zoom, limit=args.limit)
    elif args.step == "project":
        s.project(
            headings=args.headings,
            fov=args.fov,
            pitch=args.pitch,
            size=args.size,
            workers=args.workers,
            limit=args.limit,
        )
    elif args.step == "export":
        s.export(args.out)
    print(json.dumps(s.stats()))
    return 0
