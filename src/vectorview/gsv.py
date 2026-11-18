"""Street View Static API client with a hard per-key image cap.

- `GSVClient.metadata` looks up the nearest panorama. Metadata requests are free and are not
  counted.
- `GSVClient.image` fetches one perspective image and counts one unit against the key's cap.

`QuotaLedger` keeps the per-key counts in a JSON file under an exclusive ``fcntl`` lock, so
concurrent threads and processes cannot overspend. A unit is reserved *before* each image
request and is not refunded if the request fails: a crash between the request and the write
can never let a key exceed its cap.

API documentation: https://developers.google.com/maps/documentation/streetview
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, TypedDict

import httpx

from .config import Config

LOG = logging.getLogger(__name__)

METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
IMAGE_URL = "https://maps.googleapis.com/maps/api/streetview"

DEFAULT_IMAGE_SIZE = "640x640"
DEFAULT_FOV = 90.0
DEFAULT_PITCH = 0.0
_RETRY_STATUSES = (429, 500, 502, 503, 504)


class QuotaExhausted(Exception):
    """Reserving one more image would exceed the key's cap."""


class GSVMetadataError(Exception):
    """The metadata endpoint returned an error that retrying will not fix."""


class GSVImageError(Exception):
    """The image endpoint returned an error that retrying will not fix."""


class PanoMetadata(TypedDict):
    """Metadata of one panorama."""

    pano_id: str
    lat: float
    lon: float
    date: date | None
    copyright: str | None


class QuotaLedger:
    """File-backed per-key image counter with atomic, locked increments.

    The state file maps slot names to counts, e.g. ``{"key_1": 12, "key_2": 0}``; missing slots
    count as 0. Delete or edit the file to reset the counts.

    Args:
        state_path: JSON state file; created if missing.
        caps: Maximum count per slot.
    """

    def __init__(self, state_path: Path, caps: Mapping[str, int]) -> None:
        self._path = state_path
        self._caps = dict(caps)
        if not state_path.exists():
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps({k: 0 for k in self._caps}))

    @property
    def caps(self) -> dict[str, int]:
        """Cap per slot."""
        return dict(self._caps)

    @contextmanager
    def _locked(self) -> Iterator[dict[str, int]]:
        """Yield the parsed state under an exclusive lock; changes are fsynced before unlocking."""
        fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            with os.fdopen(fd, "r+", closefd=False) as f:
                raw = f.read() or "{}"
                try:
                    state = json.loads(raw)
                except json.JSONDecodeError:
                    LOG.warning("quota state file %s is corrupt; counting from zero", self._path)
                    state = {}
                for k in self._caps:
                    state.setdefault(k, 0)
                yield state
                f.seek(0)
                f.truncate()
                f.write(json.dumps(state, indent=2, sort_keys=True))
                f.flush()
                os.fsync(f.fileno())
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def snapshot(self) -> dict[str, int]:
        """Current count per slot."""
        with self._locked() as state:
            return dict(state)

    def remaining(self, key_slot: str) -> int:
        """Images ``key_slot`` may still fetch."""
        with self._locked() as state:
            return max(0, self._caps[key_slot] - state.get(key_slot, 0))

    def reserve(self, key_slot: str, n: int = 1) -> int:
        """Atomically add ``n`` to the count of ``key_slot``.

        Returns:
            The count after the increment.

        Raises:
            KeyError: If ``key_slot`` has no cap.
            QuotaExhausted: If the new count would exceed the cap (nothing is reserved).
        """
        if key_slot not in self._caps:
            raise KeyError(f"unknown key slot {key_slot!r}; expected one of {list(self._caps)}")
        with self._locked() as state:
            current = state.get(key_slot, 0)
            if current + n > self._caps[key_slot]:
                raise QuotaExhausted(f"{key_slot}: cap={self._caps[key_slot]} current={current} requested={n}")
            state[key_slot] = current + n
            return current + n


class GSVClient:
    """Street View Static API client.

    Transient failures (timeouts, 429 and 5xx) are retried with exponential backoff.

    Args:
        keys: API keys by slot name.
        ledger: Counter that enforces the per-key caps.
        timeout: Request timeout, in seconds.
        max_retries: Attempts per request.
        transport: Optional httpx transport (used in tests).
        sleep: Function used to wait between retries.
    """

    def __init__(
        self,
        keys: Mapping[str, str],
        ledger: QuotaLedger,
        *,
        timeout: float = 30.0,
        max_retries: int = 6,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._keys = dict(keys)
        self._ledger = ledger
        self._client = httpx.Client(timeout=timeout, transport=transport)
        self._max_retries = max_retries
        self._sleep = sleep

    @property
    def slots(self) -> list[str]:
        """Configured key slots, in order."""
        return list(self._keys)

    def close(self) -> None:
        """Close the HTTP connection pool."""
        self._client.close()

    def __enter__(self) -> GSVClient:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()

    def _key(self, key_slot: str) -> str:
        if key_slot not in self._keys:
            raise KeyError(f"unknown key slot {key_slot!r}")
        return self._keys[key_slot]

    def _get(self, url: str, params: dict[str, Any], *, allow_4xx: bool = False) -> httpx.Response:
        backoff = 1.0
        last: str = ""
        for attempt in range(self._max_retries):
            try:
                resp = self._client.get(url, params=params)
            except httpx.TransportError as exc:
                last = repr(exc)
                LOG.warning("transient error (attempt %d): %s", attempt + 1, exc)
            else:
                if resp.status_code not in _RETRY_STATUSES:
                    if 400 <= resp.status_code < 500 and not allow_4xx:
                        raise GSVImageError(f"client error {resp.status_code}: {resp.text[:200]}")
                    return resp
                last = f"status {resp.status_code}"
                LOG.warning("status %d (attempt %d); retrying in %.0fs", resp.status_code, attempt + 1, backoff)
            self._sleep(backoff)
            backoff = min(backoff * 2, 30.0)
        raise RuntimeError(f"giving up after {self._max_retries} attempts ({last})")

    def metadata(self, lat: float, lon: float, key_slot: str, *, radius_m: int = 50) -> PanoMetadata | None:
        """Find the nearest outdoor panorama within ``radius_m`` of ``(lat, lon)``.

        Does not count against the quota.

        Returns:
            The panorama's metadata, or ``None`` when there is no coverage.

        Raises:
            GSVMetadataError: On errors such as ``REQUEST_DENIED`` or ``OVER_QUERY_LIMIT``.
        """
        params = {
            "location": f"{lat:.6f},{lon:.6f}",
            "radius": radius_m,
            "source": "outdoor",
            "key": self._key(key_slot),
        }
        resp = self._get(METADATA_URL, params, allow_4xx=True)
        if resp.status_code == 400:
            LOG.warning("metadata 400 for (%.6f, %.6f): %s", lat, lon, resp.text[:120])
            return None
        if resp.status_code != 200:
            raise GSVMetadataError(f"unexpected status {resp.status_code}: {resp.text[:200]}")
        payload = resp.json()
        status = payload.get("status")
        if status in ("ZERO_RESULTS", "NOT_FOUND"):
            return None
        if status != "OK":
            raise GSVMetadataError(f"metadata status={status}: {payload.get('error_message', '')}")
        loc = payload.get("location") or {}
        return PanoMetadata(
            pano_id=payload["pano_id"],
            lat=float(loc.get("lat", lat)),
            lon=float(loc.get("lng", lon)),
            date=parse_date(payload.get("date")),
            copyright=payload.get("copyright"),
        )

    def image(
        self,
        pano_id: str,
        heading: float,
        key_slot: str,
        *,
        fov: float = DEFAULT_FOV,
        pitch: float = DEFAULT_PITCH,
        size: str = DEFAULT_IMAGE_SIZE,
    ) -> bytes:
        """Fetch one perspective image of a panorama as JPEG bytes.

        One unit is reserved against ``key_slot`` before the request and is not refunded on failure.

        Raises:
            QuotaExhausted: If the key's cap is reached (no request is made).
            GSVImageError: If the API returns an error or a non-image response.
        """
        key = self._key(key_slot)
        self._ledger.reserve(key_slot, n=1)
        params = {
            "pano": pano_id,
            "heading": f"{heading:.2f}",
            "pitch": f"{pitch:.2f}",
            "fov": f"{fov:.2f}",
            "size": size,
            "source": "outdoor",
            "key": key,
        }
        resp = self._get(IMAGE_URL, params)
        if resp.status_code != 200:
            raise GSVImageError(f"image status {resp.status_code}: {resp.text[:200]}")
        ctype = resp.headers.get("content-type", "")
        if "image" not in ctype.lower():
            raise GSVImageError(f"unexpected content-type {ctype!r}: {resp.text[:200]}")
        return resp.content


def parse_date(raw: str | None) -> date | None:
    """Parse a metadata date (``"YYYY-MM"`` or ``"YYYY-MM-DD"``); ``None`` if absent or malformed."""
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    LOG.warning("could not parse Street View date %r", raw)
    return None


def make_image_id(pano_id: str, heading: float) -> str:
    """Stable id of one view: ``"<pano_id>_h<heading>"``, heading rounded to whole degrees in ``[0, 360)``."""
    return f"{pano_id}_h{round(heading) % 360:03d}"


def cache_path_for(cache_root: Path, image_id: str) -> Path:
    """Path of an image in the cache, sharded by the first two characters of its id."""
    return cache_root / image_id[:2] / f"{image_id}.jpg"


def open_client(cfg: Config) -> GSVClient:
    """Create a client with the configured keys and a ledger at ``cfg.quota_state_path``."""
    return GSVClient(cfg.gsv_keys, QuotaLedger(cfg.quota_state_path, cfg.gsv_caps))
