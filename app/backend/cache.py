"""In-memory cache for the current-season forecast, refreshed both on
demand and automatically on a background schedule.

Building a forecast isn't free - it fetches live data over the network,
refits the Poisson/Dixon-Coles model on ~12,700 historical matches, and
runs 50,000 Monte Carlo simulations, all told a few seconds. Refitting on
every single API request would make the API sluggish and would hammer the
live data source (openfootball/football.json on GitHub) far more than
necessary.

This is deliberately a simple, single-process cache: a module-level
singleton behind a lock. Two things keep it fresh:

1. `POST /simulation/refresh` (see app/backend/main.py) - refresh on
   demand, immediately.
2. `background_refresh_loop()` - a periodic automatic refresh, run as a
   background asyncio task for as long as the server process stays up
   (wired into the FastAPI app via its lifespan context manager). Without
   this, a long-running server would keep serving whatever forecast it
   built on its very first request forever, never picking up new match
   results played after that - which is exactly the gap that prompted
   adding this.

Both together are still an appropriate scope for a portfolio deployment;
a production service serving real traffic would want a proper shared
cache (e.g. Redis) and a task scheduler that survives process restarts,
which is out of scope here - this only refreshes for as long as one
`uvicorn` process keeps running.
"""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timezone

from src.live_forecast import LiveForecast, build_current_forecast

DEFAULT_REFRESH_INTERVAL_HOURS = 3.0

_lock = threading.Lock()
_cached_forecast: LiveForecast | None = None
_last_refresh_error: str | None = None
_last_refresh_attempt_at: datetime | None = None


def get_forecast(force_refresh: bool = False) -> LiveForecast:
    global _cached_forecast
    with _lock:
        if _cached_forecast is None or force_refresh:
            _cached_forecast = build_current_forecast()
        return _cached_forecast


def cache_status() -> dict:
    """Observability for GET /meta: when the cache last actually served a
    forecast, and whether the most recent background refresh attempt (if
    any) succeeded - so a caller can tell "still showing an old forecast
    because refreshes have been failing" apart from "just hasn't refreshed
    yet"."""
    with _lock:
        return {
            "cached_forecast_generated_at": (
                _cached_forecast.generated_at.isoformat() if _cached_forecast else None
            ),
            "last_background_refresh_attempt_at": (
                _last_refresh_attempt_at.isoformat() if _last_refresh_attempt_at else None
            ),
            "last_background_refresh_error": _last_refresh_error,
        }


def refresh_once() -> bool:
    """One refresh attempt, with error handling - the part that's actually
    worth unit testing in isolation from the infinite sleep loop around
    it. Returns True on success. A failed attempt (e.g. a transient
    network error reaching GitHub) is logged and swallowed rather than
    raised: one bad refresh should leave the previous good forecast being
    served, not crash the background loop and silently stop all future
    refresh attempts for the rest of the process's life."""
    global _last_refresh_error, _last_refresh_attempt_at
    _last_refresh_attempt_at = datetime.now(timezone.utc)
    try:
        get_forecast(force_refresh=True)
        _last_refresh_error = None
        print(f"[cache] background refresh succeeded at {_last_refresh_attempt_at.isoformat()}")
        return True
    except Exception as e:  # noqa: BLE001 - deliberately broad: any failure must not kill the loop
        _last_refresh_error = str(e)
        print(f"[cache] background refresh FAILED at {_last_refresh_attempt_at.isoformat()}: {e}")
        return False


async def background_refresh_loop(interval_hours: float = DEFAULT_REFRESH_INTERVAL_HOURS) -> None:
    """Runs forever (until cancelled), calling refresh_once() every
    `interval_hours`. Does NOT refresh immediately on startup - the first
    real request already triggers a build via get_forecast()'s own
    lazy-init, so an immediate duplicate build here would just waste the
    first few seconds of server startup for no benefit."""
    while True:
        await asyncio.sleep(interval_hours * 3600)
        await asyncio.to_thread(refresh_once)
