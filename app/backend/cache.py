"""In-memory cache for the current-season forecast.

Building a forecast isn't free - it fetches live data over the network,
refits the Poisson/Dixon-Coles model on ~12,700 historical matches, and
runs 50,000 Monte Carlo simulations, all told a few seconds. Refitting on
every single API request would make the API sluggish and would hammer the
live data source (openfootball/football.json on GitHub) far more than
necessary, since the underlying data only actually changes when a match is
played (at most a few times a day).

This is deliberately a simple, single-process cache: a module-level
singleton behind a lock, with no TTL-based auto-expiry - it only refreshes
when explicitly asked to (POST /simulation/refresh). That's an appropriate
scope for a portfolio-scale deployment; a production service serving real
traffic would want a proper shared cache (e.g. Redis) and a scheduled
background refresh job, which is out of scope here.
"""

from __future__ import annotations

import threading

from src.live_forecast import LiveForecast, build_current_forecast

_lock = threading.Lock()
_cached_forecast: LiveForecast | None = None


def get_forecast(force_refresh: bool = False) -> LiveForecast:
    global _cached_forecast
    with _lock:
        if _cached_forecast is None or force_refresh:
            _cached_forecast = build_current_forecast()
        return _cached_forecast
