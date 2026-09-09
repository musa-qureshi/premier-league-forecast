"""FastAPI backend - a thin HTTP layer over src/.

Contains NO modeling or simulation logic of its own: every endpoint calls
into src/live_forecast.py's build_current_forecast() (cached - see
cache.py) or the fitted Poisson/Dixon-Coles model it returns, and formats
the result as JSON. If you're looking for how a probability or a
simulation is actually computed, it isn't here - see src/models/,
src/simulation/, and src/live_forecast.py, all of which are exercised by
their own tests independent of this API layer.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.backend.cache import DEFAULT_REFRESH_INTERVAL_HOURS, background_refresh_loop, cache_status, get_forecast
from app.backend.schemas import (
    ForecastMeta,
    MatchPrediction,
    MatchweekStandings,
    PositionDistribution,
    RemainingFixture,
    Scoreline,
    TeamForecast,
    TeamStanding,
)
from src.models.poisson_model import top_scorelines
from src.simulation.summary import position_distribution

# How often the cache refreshes itself in the background, in hours -
# configurable via an env var so a real deployment can tune it (e.g.
# faster during a live matchday, slower otherwise) without a code change.
# See app/backend/cache.py for the full reasoning on why this exists.
REFRESH_INTERVAL_HOURS = float(
    os.environ.get("FORECAST_REFRESH_INTERVAL_HOURS", DEFAULT_REFRESH_INTERVAL_HOURS)
)


async def _warm_cache_on_startup() -> None:
    """Fire-and-forget: builds the forecast once as soon as the process
    boots, rather than leaving whoever sends the very first request to eat
    the build cost (a live fetch + refit + 50,000 simulations, a few real
    seconds). Barely mattered on a dev machine that's usually already
    warm; matters a lot on a real deployment (Phase 14) where a visitor
    hitting a just-booted container is a normal case, not an edge case.

    Deliberately NOT awaited in lifespan() before yield - the app must
    start accepting connections (health checks included) immediately, not
    block on a live network fetch. If this fails (or simply hasn't
    finished yet), get_forecast()'s own lazy-init in every endpoint
    already covers it: the next request just builds it there instead,
    exactly as it always has.

    Calls the module-level `get_forecast` name (not cache.refresh_once())
    on purpose, so tests/test_api.py's `monkeypatch.setattr(main_module,
    "get_forecast", ...)` covers this path too - no second, untested route
    to a real network call.
    """
    try:
        await asyncio.to_thread(get_forecast, force_refresh=True)
    except Exception as e:  # noqa: BLE001 - startup warmup must never crash the app
        print(f"[main] startup cache warmup failed (a normal request will retry): {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(_warm_cache_on_startup())
    task = asyncio.create_task(background_refresh_loop(REFRESH_INTERVAL_HOURS))
    print(f"[main] background refresh loop started, every {REFRESH_INTERVAL_HOURS}h")
    yield
    task.cancel()


app = FastAPI(
    title="Premier League Probabilistic Forecast API",
    description=(
        "Probabilistic match and season forecasts for the Premier League, "
        "built from historical data (1993-94 to present), Elo/Poisson/"
        "logistic-regression/XGBoost models, and Monte Carlo season "
        "simulation. Every probability here is a simulated estimate, not a "
        "prediction of what will happen - see the project README for the "
        "full methodology, backtesting results, and known limitations."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Frontend (Phase 12) runs on a separate origin during development (Vite's
# default dev server port). Wide open here deliberately - this API has no
# authentication, no user accounts, and serves only public football data
# (its one write-ish endpoint, /simulation/refresh, just re-triggers a
# recomputation from public data, not a state-changing action on anyone's
# behalf) - a permissive CORS policy doesn't carry the risk it would for a
# service handling secrets or user-specific state.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _forecast_meta(forecast) -> dict:
    status = cache_status()
    return {
        "season": forecast.season,
        "generated_at": forecast.generated_at.isoformat(),
        "n_simulations": forecast.n_simulations,
        "n_played": forecast.n_played,
        "n_remaining": forecast.n_remaining,
        "refresh_interval_hours": REFRESH_INTERVAL_HOURS,
        "last_background_refresh_attempt_at": status["last_background_refresh_attempt_at"],
        "last_background_refresh_error": status["last_background_refresh_error"],
    }


def _require_known_team(team: str, forecast) -> None:
    if team not in forecast.standings.index:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown team '{team}'. Known teams: {sorted(forecast.standings.index)}",
        )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/meta", response_model=ForecastMeta)
def meta() -> dict:
    return _forecast_meta(get_forecast())


@app.get("/standings", response_model=list[TeamStanding])
def standings() -> list[dict]:
    forecast = get_forecast()
    df = forecast.standings.reset_index().rename(columns={"index": "team"})
    return df.to_dict(orient="records")


@app.get("/standings/history", response_model=list[MatchweekStandings])
def standings_history() -> list[dict]:
    """The table exactly as it stood after every matchweek that's fully
    completed so far this season - recomputed fresh on every cache
    refresh from the live source's full season-to-date match list, not a
    saved/persisted snapshot (see src/features/league_table.py::
    standings_by_matchweek for why that's deliberate). Empty list for any
    season with no matchweeks complete yet (or, in principle, a data
    source with no round information)."""
    forecast = get_forecast()
    return [
        {
            "matchweek": entry["matchweek"],
            "standings": entry["standings"].reset_index().rename(columns={"index": "team"}).to_dict(
                orient="records"
            ),
        }
        for entry in forecast.matchweek_standings
    ]


@app.get("/forecast", response_model=list[TeamForecast])
def forecast_all() -> list[dict]:
    forecast = get_forecast()
    df = forecast.summary.reset_index().rename(columns={"index": "team"})
    return df.to_dict(orient="records")


@app.get("/teams/{team}", response_model=TeamForecast)
def team_forecast(team: str) -> dict:
    forecast = get_forecast()
    _require_known_team(team, forecast)
    row = forecast.summary.loc[team]
    return {"team": team, **row.to_dict()}


@app.get("/teams/{team}/position-distribution", response_model=PositionDistribution)
def team_position_distribution(team: str) -> dict:
    forecast = get_forecast()
    _require_known_team(team, forecast)
    dist = position_distribution(forecast.positions)
    row = dist.loc[team]
    return {"team": team, "distribution": {str(pos): prob for pos, prob in row.items()}}


@app.get("/matches/upcoming", response_model=list[RemainingFixture])
def upcoming_matches() -> list[dict]:
    forecast = get_forecast()
    df = forecast.remaining_fixtures.rename(
        columns={"HomeTeam": "home_team", "AwayTeam": "away_team"}
    )
    return df.to_dict(orient="records")


@app.get("/matches/predict", response_model=MatchPrediction)
def predict_match(
    home: str = Query(..., description="Home team name, e.g. 'Arsenal'"),
    away: str = Query(..., description="Away team name, e.g. 'Liverpool'"),
) -> dict:
    forecast = get_forecast()
    _require_known_team(home, forecast)
    _require_known_team(away, forecast)
    if home == away:
        raise HTTPException(status_code=400, detail="home and away must be different teams")

    model = forecast.model
    lam_h, lam_a = model.predict_expected_goals(home, away)
    grid = model.predict_score_grid(home, away)
    proba = model.predict_proba(pd.DataFrame({"HomeTeam": [home], "AwayTeam": [away]})).iloc[0]

    scorelines = [
        Scoreline(home_goals=h, away_goals=a, probability=p)
        for h, a, p in top_scorelines(grid, k=5)
    ]

    return {
        "home_team": home,
        "away_team": away,
        "expected_home_goals": lam_h,
        "expected_away_goals": lam_a,
        "home_win_probability": float(proba["H"]),
        "draw_probability": float(proba["D"]),
        "away_win_probability": float(proba["A"]),
        "most_likely_scorelines": scorelines,
    }


@app.post("/simulation/refresh", response_model=ForecastMeta)
def refresh_simulation() -> dict:
    """Forces a full refresh: re-fetches the live fixture file, refits the
    model, and re-runs the Monte Carlo simulation - the only endpoint that
    does real work rather than reading the cache. Every other endpoint
    reads whatever this (or the automatic first-request build) last
    produced."""
    return _forecast_meta(get_forecast(force_refresh=True))
