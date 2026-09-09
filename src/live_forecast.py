"""Builds the current-season live forecast: live fixture data -> current
table state + remaining fixtures -> fitted Poisson/Dixon-Coles model ->
Monte Carlo simulation -> summary statistics.

This is the single reusable pipeline behind both
`experiments/run_phase9_live_forecast.py` (CLI/reporting) and the FastAPI
backend (Phase 11, `app/backend/`) - neither duplicates this logic, per
the project's "no modeling logic in the backend" architecture principle.
Adding a new consumer (a notebook, a scheduled job, a different frontend)
means calling `build_current_forecast()`, not reimplementing the pipeline.

Deliberately reuses existing, already-tested modules rather than building
parallel logic: src/data/validate.py's load_openfootball_matches/
canonicalize_teams for parsing and cleaning, src/features/league_table.py's
LeagueTableTracker for the current table state, src/models/poisson_model.py
for match probabilities, and src/simulation/ for the simulation itself -
the same code paths already tested against historical data in Phases 2-8.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.data.ingest import download_json, load_config
from src.data.validate import canonicalize_teams, load_openfootball_matches, load_team_name_map
from src.features.league_table import LeagueTableTracker, standings_by_matchweek
from src.models.poisson_model import DixonColesModel
from src.simulation.engine import CurrentTableState, Fixture, simulate_final_tables
from src.simulation.summary import summarize_simulation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_N_SIMULATIONS = 50_000

# Deliberately NOT DixonColesModel's tuned class default (xi=0.5,
# promoted_penalty=0.2). Phase 10's tuning only ever validated WHOLE-
# SEASON-AHEAD predictions; applied here (predicting the rest of a season
# from only a handful of matchdays played) it overreacts badly to small
# early-season samples - confirmed in practice, not hypothetically: with
# xi=0.5, Hull City (newly promoted, 1 match played, a single 2-0 win over
# Manchester United) came out as the simulated #2 title favorite at 30.7%.
# These are the same values Phase 8's validation (also a mid-season
# cutoff) used and confirmed sensible against a season with a known real
# outcome. See src/models/poisson_model.py's docstring for the full story,
# and README.md "Hyperparameter tuning (Phase 10)" for the write-up.
LIVE_FORECAST_POISSON_CONFIG = {"use_correlation": True, "xi": 0.3, "promoted_penalty": 0.4}


@dataclass
class LiveForecast:
    season: str
    generated_at: pd.Timestamp  # UTC-aware - see the two construction sites below for why
    n_simulations: int
    n_played: int
    n_remaining: int
    standings: pd.DataFrame            # current table, one row per team, sorted by points desc
    remaining_fixtures: pd.DataFrame   # HomeTeam/AwayTeam/expected goals for unplayed matches
    positions: pd.DataFrame            # (n_simulations, n_teams) simulated final positions
    points: pd.DataFrame               # (n_simulations, n_teams) simulated final points
    summary: pd.DataFrame              # per-team headline stats (title probability, etc.)
    model: DixonColesModel
    # [] for simulate_historical_cutoff() - matches.parquet has no `Round`
    # column (only the live openfootball source does), so there's nothing
    # to compute matchweek history from. Populated for build_current_forecast()
    # - see standings_by_matchweek()'s docstring for why this is recomputed
    # from scratch every refresh rather than persisted between refreshes.
    matchweek_standings: list[dict] = field(default_factory=list)


def _forecast_from_matches(
    train_matches: pd.DataFrame,
    played_this_season: pd.DataFrame,
    remaining_this_season: pd.DataFrame,
    season: str,
    n_simulations: int,
    seed: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, DixonColesModel]:
    """The shared core behind both build_current_forecast() (live) and
    simulate_historical_cutoff() (backtesting - see
    src/simulation/calibration_backtest.py): fit the model, build the
    current table, predict remaining fixtures, simulate, summarize.
    Neither caller duplicates this logic - the only difference between
    them is WHERE `train_matches`/`played_this_season`/
    `remaining_this_season` come from (a live JSON download vs. a
    historical cutoff date), not how the forecast itself is computed.

    Returns (standings, remaining_fixtures, positions, points, summary, model).
    """
    tracker = LeagueTableTracker()
    for _, row in played_this_season.sort_values("Date").iterrows():
        tracker.snapshot(row["HomeTeam"], row["AwayTeam"], season)
        tracker.apply_result(row["HomeTeam"], row["AwayTeam"], row["FTHG"], row["FTAG"])
    standings_dict = tracker.current_standings()
    teams = sorted(standings_dict.keys())
    current = CurrentTableState(
        teams=teams,
        points={t: standings_dict[t]["points"] for t in teams},
        goals_for={t: standings_dict[t]["goals_for"] for t in teams},
        goals_against={t: standings_dict[t]["goals_against"] for t in teams},
    )
    standings = pd.DataFrame(standings_dict).T
    standings.index.name = "team"
    standings = standings.sort_values(["points", "goal_difference"], ascending=False)

    model = DixonColesModel(**LIVE_FORECAST_POISSON_CONFIG).fit(train_matches)

    fixtures = []
    fixture_rows = []
    for _, row in remaining_this_season.sort_values("Date").iterrows():
        lam_h, lam_a = model.predict_expected_goals(row["HomeTeam"], row["AwayTeam"])
        fixtures.append(Fixture(row["HomeTeam"], row["AwayTeam"], lam_h, lam_a))
        fixture_rows.append({
            "HomeTeam": row["HomeTeam"], "AwayTeam": row["AwayTeam"],
            "expected_home_goals": lam_h, "expected_away_goals": lam_a,
        })
    remaining_fixtures = pd.DataFrame(fixture_rows)

    positions, points = simulate_final_tables(current, fixtures, n_simulations=n_simulations, seed=seed)
    summary = summarize_simulation(positions, points)

    return standings, remaining_fixtures, positions, points, summary, model


def _build_training_set(historical: pd.DataFrame, played: pd.DataFrame) -> pd.DataFrame:
    """Historical matches plus this season's matches played so far,
    chronologically sorted - the exact training set `DixonColesModel`
    should fit on. Pulled out as its own function specifically so
    `_matchweek_forecasts()` (below) can build a DIFFERENT training set
    per matchweek (only played-as-of-that-matchweek, not everything played
    to date) without duplicating this concat/column-alignment logic."""
    return pd.concat(
        [historical, played[historical.columns.intersection(played.columns)]], ignore_index=True
    ).sort_values("Date").reset_index(drop=True)


MATCHWEEK_FORECAST_N_SIMULATIONS = 10_000


def _matchweek_forecasts(
    historical: pd.DataFrame,
    all_current: pd.DataFrame,
    season: str,
    seed: int | None,
    cached: dict[int, dict] | None = None,
) -> list[dict]:
    """For every matchweek that's fully complete so far this season,
    reruns the ENTIRE forecast pipeline (refit the model, re-simulate)
    using only data that would genuinely have been available right after
    that matchweek finished - not just the final table
    (standings_by_matchweek() already covers that), but the model's own
    title/top-4/relegation probabilities AS THEY WOULD HAVE LOOKED at that
    point in the season. This is what makes browsing matchweek history
    show "what did the model think back then", not just "what was the
    score", matching how the live forecast page frames every number as a
    simulated estimate rather than a fact.

    Leakage guard, same shape as simulate_historical_cutoff()'s: for
    matchweek N, `played` is capped at Date <= that matchweek's
    standings_by_matchweek()-reported `cutoff_date`, and `remaining` is
    everything after it - even fixtures that have SINCE been played and
    have a real score by now must be treated as future/unknown for this
    matchweek's own forecast, exactly the way they'd have been unknown at
    the time. Reuses _forecast_from_matches() - the same core
    build_current_forecast() and simulate_historical_cutoff() both call -
    so this isn't a parallel, separately-maintained forecasting path.

    `cached`, keyed by matchweek number, lets a caller (cache.py, across
    background refreshes) skip recomputing a matchweek whose forecast it
    already has: unlike the "current" forecast, a completed matchweek's
    "as of then" forecast can never change once computed - it depends
    only on data through a fixed past point, not on anything played
    since. Without this, EVERY refresh would re-simulate EVERY complete
    matchweek from scratch, an unnecessary cost that only grows as a
    season progresses (confirmed by direct benchmarking: even at a
    reduced 10,000 simulations, 3 early-season matchweeks - each with
    ~350+ remaining fixtures still to simulate - took ~6s combined; a
    mid-season refresh recomputing every one of 15-20 already-known
    matchweeks for no new information would take real, wasted time on
    every single refresh instead of just the one genuinely new matchweek).

    Runs at MATCHWEEK_FORECAST_N_SIMULATIONS (10,000), not the live
    forecast's usual 50,000 - a deliberate, separate reduction from the
    "current" forecast's simulation count: this is a supplementary,
    once-per-matchweek historical view, not the headline number, and
    10,000 simulations (the same scale Phase 8/13's validation and
    calibration backtests already relied on) is still a large, stable
    Monte Carlo sample.
    """
    cached = cached or {}
    snapshots = standings_by_matchweek(all_current, season)
    results = []
    for entry in snapshots:
        mw = entry["matchweek"]
        if mw in cached:
            results.append(cached[mw])
            continue

        cutoff = entry["cutoff_date"]
        played = all_current[(all_current["Date"] <= cutoff) & all_current["FTHG"].notna()].copy()
        remaining = all_current[all_current["Date"] > cutoff].copy().sort_values("Date")
        played["FTHG"] = played["FTHG"].astype(int)
        played["FTAG"] = played["FTAG"].astype(int)
        played["Season"] = season

        train = _build_training_set(historical, played)
        _, _, _, _, summary, _ = _forecast_from_matches(
            train, played, remaining, season, MATCHWEEK_FORECAST_N_SIMULATIONS, seed
        )

        results.append({"matchweek": mw, "standings": entry["standings"], "summary": summary})
    return results


def build_current_forecast(
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    seed: int | None = 0,
    previous_matchweek_forecasts: list[dict] | None = None,
) -> LiveForecast:
    """`previous_matchweek_forecasts`: the prior LiveForecast's own
    `matchweek_standings` (if any), threaded through by app/backend/
    cache.py across refreshes purely as a cache - see
    _matchweek_forecasts()'s docstring for why a completed matchweek's
    forecast never needs to be recomputed once it exists."""
    config = load_config()
    live = config["live_source"]
    current_season = live["current_season"]
    url = live["url_template"].format(season=current_season)
    dest = PROJECT_ROOT / "data" / "raw" / f"openfootball_{current_season.replace('-', '_')}_live"

    # force=True: unlike the frozen historical sources, this file changes
    # every time a match is played, so re-fetch every run rather than
    # trusting whatever was downloaded last time.
    live_file = download_json(url, dest, filename="matches.json", force=True)

    team_map = load_team_name_map(PROJECT_ROOT / config["team_name_map_path"])

    all_current = load_openfootball_matches([live_file], completed_only=False)
    all_current = canonicalize_teams(all_current, team_map)
    all_current["Date"] = pd.to_datetime(all_current["Date"])

    played = all_current.dropna(subset=["FTHG", "FTAG"]).copy()
    # Explicit chronological sort rather than trusting the source file's row
    # order - it happens to already be date-ordered today, but "remaining
    # fixtures, in order" (what the API and frontend both assume - e.g. a
    # team's "next N fixtures") shouldn't silently depend on that holding.
    remaining = all_current[all_current["FTHG"].isna()].copy().sort_values("Date")

    played["FTHG"] = played["FTHG"].astype(int)
    played["FTAG"] = played["FTAG"].astype(int)
    played["Season"] = current_season

    historical = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "matches.parquet")
    train = _build_training_set(historical, played)

    standings, remaining_fixtures, positions, points, summary, model = _forecast_from_matches(
        train, played, remaining, current_season, n_simulations, seed
    )

    # Recomputed from scratch on every refresh, not persisted between
    # refreshes - see standings_by_matchweek()'s docstring for why that's
    # deliberate. `all_current` (not `played`) is passed because
    # completeness-checking a matchweek needs to know about its
    # not-yet-played fixtures too, not just the ones already done.
    cached_matchweeks = {e["matchweek"]: e for e in (previous_matchweek_forecasts or [])}
    matchweek_standings = _matchweek_forecasts(historical, all_current, current_season, seed, cached_matchweeks)

    return LiveForecast(
        season=current_season,
        # tz="UTC", not a naive pd.Timestamp.now(): a naive timestamp
        # serializes with no timezone marker at all (app/backend/main.py's
        # .isoformat()), and a browser's `new Date(...)` interprets a
        # marker-less ISO string as ITS OWN local time, not UTC - silently
        # skewing the frontend's "updated N ago" display by exactly the
        # visitor's UTC offset (confirmed in production: a real deploy
        # displayed a permanently-stuck "updated 3 hours ago" for a
        # visitor 3 hours off UTC, even seconds after a fresh forecast).
        # Matches cache.py's already-correct datetime.now(timezone.utc).
        generated_at=pd.Timestamp.now(tz="UTC"),
        n_simulations=n_simulations,
        n_played=len(played),
        n_remaining=len(remaining),
        standings=standings,
        remaining_fixtures=remaining_fixtures,
        positions=positions,
        points=points,
        summary=summary,
        model=model,
        matchweek_standings=matchweek_standings,
    )


def simulate_historical_cutoff(
    matches: pd.DataFrame,
    season: str,
    cutoff_date: pd.Timestamp,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    seed: int | None = 0,
) -> LiveForecast:
    """The same forecast pipeline as build_current_forecast(), replayed
    against a historical season frozen at `cutoff_date` instead of a live
    download - the basis of the Phase 13 simulation-calibration backtest
    (src/simulation/calibration_backtest.py). Matches in `season` with
    Date < cutoff_date count as "played" (used to build the current table
    and folded into training); Date >= cutoff_date count as "remaining"
    (predicted, not trained on).

    Leakage guard, the one thing this function has to get exactly right:
    the training set is historical seasons STRICTLY BEFORE `season`, never
    "every other season" - `matches` (typically the full matches.parquet)
    contains seasons chronologically AFTER `season` too, and including
    those in training would let a backtest "predicting" e.g. 2010-11 learn
    from data that, at that point in history, doesn't exist yet. See
    tests/test_live_forecast.py for a regression test on exactly this.
    """
    season_order = sorted(matches["Season"].unique())
    if season not in season_order:
        raise ValueError(f"Season {season!r} not found in the given matches.")
    prior_seasons = season_order[:season_order.index(season)]
    prior_matches = matches[matches["Season"].isin(prior_seasons)]

    season_matches = matches[matches["Season"] == season]
    played = season_matches[season_matches["Date"] < cutoff_date].copy()
    remaining = season_matches[season_matches["Date"] >= cutoff_date].copy().sort_values("Date")

    train = pd.concat([prior_matches, played], ignore_index=True).sort_values("Date").reset_index(drop=True)

    standings, remaining_fixtures, positions, points, summary, model = _forecast_from_matches(
        train, played, remaining, season, n_simulations, seed
    )

    return LiveForecast(
        season=season,
        generated_at=pd.Timestamp.now(tz="UTC"),  # see build_current_forecast()'s comment on why tz-aware
        n_simulations=n_simulations,
        n_played=len(played),
        n_remaining=len(remaining),
        standings=standings,
        remaining_fixtures=remaining_fixtures,
        positions=positions,
        points=points,
        summary=summary,
        model=model,
    )
