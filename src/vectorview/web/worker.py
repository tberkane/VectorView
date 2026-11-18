"""Background worker started by the web app.

One asyncio task polls SQLite for pending work and runs it serially, which keeps GPU memory
and VLM rate limits predictable. Queries take priority over scans (seconds versus minutes).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

from . import db
from .query_pipeline import run_query
from .scan_worker import run_scan

LOG = logging.getLogger(__name__)

POLL_INTERVAL_S = 0.5


async def run_next() -> bool:
    """Run the next pending query, else the next pending scan; return False if there was none."""
    q = db.next_pending_query()
    if q is not None:
        await run_query(q)
        return True
    s = db.next_pending_scan()
    if s is not None:
        await run_scan(s)
        return True
    return False


async def worker_loop(stop_event: asyncio.Event, poll_interval: float = POLL_INTERVAL_S) -> None:
    """Process pending work until ``stop_event`` is set."""
    LOG.info("worker started")
    while not stop_event.is_set():
        try:
            if await run_next():
                continue
        except Exception:  # run_query/run_scan record their own errors; this is a last resort
            LOG.exception("worker: unhandled error")
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=poll_interval)
    LOG.info("worker stopped")
