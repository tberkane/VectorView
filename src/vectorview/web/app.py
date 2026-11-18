"""FastAPI app: pages for scans, queries and verification, and the background worker.

Start it with ``vectorview serve``. It binds to 127.0.0.1 by default and has no
authentication: it is meant for one user on their own machine.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..config import Config, load_config
from ..gsv import cache_path_for
from ..ingest import HEADING_OFFSETS
from ..regions import BBox
from . import db, llm_expand, vlm_judge
from .worker import worker_loop

LOG = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent
TEMPLATES = Jinja2Templates(directory=str(WEB_DIR / "templates"))

API_COST_PER_IMAGE_USD = 0.007
"""Rough Street View Static API price per image, shown in the scan preview; check current prices."""
PANOS_PER_KM2_AT_50M = 20
"""Rough panorama density used by the scan preview; the real count depends on the road network."""
API_IMAGES_PER_S = 9.0
SCRAPER_IMAGES_PER_S = 3.0
REFINE_EXPECTED_OVERLAP = 0.65
"""Assumed share of a refined query's top K already judged for its parent (for the cost estimate)."""


def _scraper_installed() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("streetlevel", "py360convert", "aiohttp"))


def create_app(cfg: Config | None = None, *, start_worker: bool = True) -> FastAPI:
    """Build the app.

    Args:
        cfg: Configuration; read from the environment (and ``.env``) if omitted.
        start_worker: Run the background worker while the app is up.
    """
    cfg = cfg or load_config(dotenv=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        db.init(cfg.web_db_path)
        nq, ns = db.reset_orphaned_jobs()
        if nq or ns:
            LOG.warning("requeued %d queries and %d scans left unfinished by a previous run", nq, ns)
        if not start_worker:
            yield
            return
        stop = asyncio.Event()
        task = asyncio.create_task(worker_loop(stop))
        try:
            yield
        finally:
            stop.set()
            try:
                await asyncio.wait_for(task, timeout=5)
            except TimeoutError:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    app = FastAPI(title="VectorView", lifespan=lifespan)
    app.state.cfg = cfg
    app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")

    def get_scan_or_404(scan_id: int) -> dict[str, Any]:
        scan = db.get_scan(scan_id)
        if not scan:
            raise HTTPException(404, f"scan {scan_id} not found")
        return scan

    def get_query_or_404(query_id: int) -> dict[str, Any]:
        q = db.get_query(query_id)
        if not q:
            raise HTTPException(404, f"query {query_id} not found")
        return q

    def ready_results(query_id: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        q = get_query_or_404(query_id)
        if q["status"] != "ready":
            raise HTTPException(400, f"query {query_id} is not ready ({q['status']})")
        return q, json.loads(q["results_json"]) if q["results_json"] else []

    # ---------------- scans ----------------

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(request, "index.html", {"scans": db.list_scans()})

    @app.get("/scans/new", response_class=HTMLResponse)
    def scan_new(request: Request) -> HTMLResponse:
        ctx = {"has_gsv_key": bool(cfg.gsv_keys), "has_scraper": _scraper_installed()}
        return TEMPLATES.TemplateResponse(request, "scan_new.html", ctx)

    def form_bbox(n: float, s: float, e: float, w: float) -> BBox:
        try:
            return BBox(west=w, south=s, east=e, north=n)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/scans/preview")
    def scan_preview(
        bbox_n: float = Form(...),
        bbox_s: float = Form(...),
        bbox_e: float = Form(...),
        bbox_w: float = Form(...),
        spacing_m: int = Form(50),
        fetch_mode: str = Form("api"),
    ) -> dict[str, Any]:
        """Rough size, cost and time of a scan, before any request is made."""
        area = form_bbox(bbox_n, bbox_s, bbox_e, bbox_w).area_km2()
        est_panos = round(area * PANOS_PER_KM2_AT_50M * max(1.0, 50.0 / max(1, spacing_m)))
        if fetch_mode == "scraper":
            # The scraper finds panoramas off the road network too, and cuts four crops per panorama.
            est_panos = round(est_panos * 1.5)
            est_images, est_cost, rate = est_panos * 4, 0.0, SCRAPER_IMAGES_PER_S
            warn = "Scraper mode finds every panorama in the box and cuts 4 crops per panorama."
        else:
            est_images = est_panos * len(HEADING_OFFSETS)
            est_cost, rate = est_images * API_COST_PER_IMAGE_USD, API_IMAGES_PER_S
            warn = "Rough estimate: the real count depends on the road network."
        return {
            "area_km2": round(area, 2),
            "est_panos": est_panos,
            "est_images": est_images,
            "est_cost_usd": round(est_cost, 2),
            "est_minutes": round(est_images / rate / 60.0, 1),
            "fetch_mode": fetch_mode,
            "warn": warn,
        }

    @app.post("/scans")
    def create_scan(
        name: str = Form(...),
        bbox_n: float = Form(...),
        bbox_s: float = Form(...),
        bbox_e: float = Form(...),
        bbox_w: float = Form(...),
        spacing_m: int = Form(50),
        fetch_mode: str = Form("api"),
    ) -> RedirectResponse:
        if not name.strip():
            raise HTTPException(400, "name is required")
        if fetch_mode not in ("api", "scraper"):
            raise HTTPException(400, f"unknown fetch_mode {fetch_mode!r}")
        if spacing_m < 5:
            raise HTTPException(400, "spacing_m must be at least 5")
        form_bbox(bbox_n, bbox_s, bbox_e, bbox_w)
        sid = db.create_scan(
            name=name.strip(),
            bbox=(bbox_n, bbox_s, bbox_e, bbox_w),
            spacing_m=spacing_m,
            embedding_tag="",
            tile_tag="",
            fetch_mode=fetch_mode,
        )
        db.update_scan(sid, status_message="waiting for the worker")
        return RedirectResponse(f"/scans/{sid}", status_code=303)

    @app.get("/scans/{scan_id}/status", response_class=HTMLResponse)
    def scan_status(request: Request, scan_id: int) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(request, "_scan_status.html", {"scan": get_scan_or_404(scan_id)})

    @app.get("/scans/{scan_id}", response_class=HTMLResponse)
    def scan_detail(request: Request, scan_id: int, q: int | None = None) -> HTMLResponse:
        scan = get_scan_or_404(scan_id)
        active = db.get_query(q) if q else None
        if active and active["scan_id"] != scan_id:
            active = None
        ctx = {"scan": scan, "queries": db.list_queries(scan_id), "active_query": active}
        return TEMPLATES.TemplateResponse(request, "scan_detail.html", ctx)

    # ---------------- queries ----------------

    @app.post("/scans/{scan_id}/queries")
    def submit_query(
        scan_id: int,
        text: str = Form(...),
        k: int = Form(50),
        vlm_model: str = Form(vlm_judge.DEFAULT_VLM_MODEL),
        min_date: str = Form(""),
    ) -> RedirectResponse:
        scan = get_scan_or_404(scan_id)
        if scan["status"] != "ready":
            raise HTTPException(400, f"scan {scan_id} is not ready ({scan['status']})")
        if not text.strip():
            raise HTTPException(400, "query text is empty")
        if not 1 <= k <= 500:
            raise HTTPException(400, "k must be between 1 and 500")
        qid = db.create_query(
            scan_id=scan_id, text=text.strip(), k=k, vlm_model=vlm_model, min_date=min_date.strip() or None
        )
        return RedirectResponse(f"/scans/{scan_id}?q={qid}", status_code=303)

    @app.post("/queries/{query_id}/refine")
    def submit_refine(query_id: int) -> RedirectResponse:
        """Start a relevance-feedback round; needs at least one result marked yes."""
        parent, _ = ready_results(query_id)
        if "yes" not in db.list_verifications(query_id).values():
            raise HTTPException(400, "mark at least one result as a match before refining")
        child = db.create_query(
            scan_id=int(parent["scan_id"]),
            text=parent["text"],
            k=int(parent["k"]),
            vlm_model=parent["vlm_model"],
            min_date=parent["min_date"],
            parent_query_id=query_id,
        )
        return RedirectResponse(f"/scans/{parent['scan_id']}?q={child}", status_code=303)

    @app.get("/queries/{query_id}/refine/cost")
    def refine_cost(query_id: int) -> dict[str, Any]:
        """Rough cost of a refine: verdicts already cached for the parent are free."""
        parent = get_query_or_404(query_id)
        if not parent.get("expansion_json"):
            raise HTTPException(400, f"query {query_id} has no expansion yet")
        expansion = llm_expand.Expansion.from_dict(json.loads(parent["expansion_json"]))
        p_hash = vlm_judge.prompt_hash(
            vlm_judge.build_prompt(parent["text"], expansion.vlm_description, expansion.vlm_exclusions)
        )
        k, model = int(parent["k"]), parent["vlm_model"]
        per_call = vlm_judge.COST_PER_CALL_USD.get(model, vlm_judge.DEFAULT_COST_USD)
        results = json.loads(parent["results_json"]) if parent.get("results_json") else []
        parent_ids = [r["image_id"] for r in results[:k]]
        verdicts = vlm_judge.load_cache(cfg.vlm_cache_path)
        est_misses = round(k * (1.0 - REFINE_EXPECTED_OVERLAP))
        return {
            "k": k,
            "vlm_model": model,
            "per_call_usd": per_call,
            "expected_overlap": REFINE_EXPECTED_OVERLAP,
            "est_misses": est_misses,
            "est_cost_usd": round(est_misses * per_call, 4),
            "worst_cost_usd": round(k * per_call, 4),
            "n_parent_cached": sum((p_hash, iid, model) in verdicts for iid in parent_ids),
            "n_parent_top_k": len(parent_ids),
        }

    @app.get("/queries/{query_id}/status", response_class=HTMLResponse)
    def query_status(request: Request, query_id: int) -> HTMLResponse:
        q = get_query_or_404(query_id)
        # HTTP 286 tells HTMX to stop polling once the query is finished.
        code = 286 if q["status"] in ("ready", "failed") else 200
        return TEMPLATES.TemplateResponse(request, "_status.html", {"q": q}, status_code=code)

    @app.get("/queries/{query_id}/results", response_class=HTMLResponse)
    def query_results(request: Request, query_id: int) -> HTMLResponse:
        q = get_query_or_404(query_id)
        results = json.loads(q["results_json"]) if q["results_json"] else []
        ctx = {"q": q, "results": results, "verifications": db.list_verifications(query_id)}
        return TEMPLATES.TemplateResponse(request, "_results_grid.html", ctx)

    # ---------------- verification ----------------

    def verify_context(query_id: int, cursor: int) -> dict[str, Any]:
        q, results = ready_results(query_id)
        verdicts = db.list_verifications(query_id)
        stats = db.verification_stats(query_id)
        n_verified = sum(stats.values())
        cursor = max(0, cursor)
        current = results[cursor] if cursor < len(results) else None
        return {
            "q": q,
            "results": results,
            "cursor": cursor,
            "total": len(results),
            "current": current,
            "existing_verdict": verdicts.get(current["image_id"]) if current else None,
            "n_verified": n_verified,
            "n_yes": stats["yes"],
            "n_no": stats["no"],
            "n_unsure": stats["unsure"],
            "precision": stats["yes"] / n_verified if n_verified else 0.0,
            "done": current is None,
            "prev_cursor": max(0, cursor - 1),
            "next_cursor": cursor + 1,
        }

    @app.get("/queries/{query_id}/verify", response_class=HTMLResponse)
    def verify_page(request: Request, query_id: int, cursor: int | None = None) -> HTMLResponse:
        if cursor is None:  # resume at the first result without a verdict
            _, results = ready_results(query_id)
            verdicts = db.list_verifications(query_id)
            cursor = next((i for i, r in enumerate(results) if r["image_id"] not in verdicts), 0)
        return TEMPLATES.TemplateResponse(request, "verify.html", verify_context(query_id, cursor))

    @app.get("/queries/{query_id}/verify/card", response_class=HTMLResponse)
    def verify_card(request: Request, query_id: int, cursor: int = 0) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(request, "_verify_card.html", verify_context(query_id, cursor))

    @app.post("/queries/{query_id}/verifications", response_class=HTMLResponse)
    def record_verification(
        request: Request,
        query_id: int,
        cursor: int = Form(...),
        image_id: str = Form(...),
        verdict: str = Form(...),
    ) -> HTMLResponse:
        if verdict not in ("yes", "no", "unsure"):
            raise HTTPException(400, f"bad verdict {verdict!r}")
        get_query_or_404(query_id)
        db.record_verification(query_id, image_id, verdict)
        return TEMPLATES.TemplateResponse(request, "_verify_card.html", verify_context(query_id, cursor + 1))

    # ---------------- images ----------------

    @app.get("/cache/{image_id}.jpg")
    def serve_image(image_id: str) -> FileResponse:
        if not image_id or "/" in image_id or "\\" in image_id or ".." in image_id:
            raise HTTPException(400, "bad image id")
        path = cache_path_for(cfg.cache_dir, image_id)
        if not path.is_file():
            raise HTTPException(404, f"image {image_id} is not cached")
        return FileResponse(path, media_type="image/jpeg")

    return app


def serve(host: str = "127.0.0.1", port: int = 8765, *, reload: bool = False, log_level: str = "info") -> None:
    """Run the app with uvicorn."""
    import uvicorn

    uvicorn.run("vectorview.web.app:create_app", factory=True, host=host, port=port, reload=reload, log_level=log_level)
