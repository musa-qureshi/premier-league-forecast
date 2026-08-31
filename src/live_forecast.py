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

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.ingest import download_json, load_config
from src.data.validate import canonicalize_teams, load_openfootball_matches, load_team_name_map
from src.features.league_table import LeagueTableTracker
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
    generated_at: pd.Timestamp
    n_simulations: int
    n_played: int
    n_remaining: int
    standings: pd.DataFrame            # current table, one row per team, sorted by points desc
    remaining_fixtures: pd.DataFrame   # HomeTeam/AwayTeam/expected goals for unplayed matches
    positions: pd.DataFrame            # (n_simulations, n_teams) simulated final positions
    points: pd.DataFrame               # (n_simulations, n_teams) simulated final points
    summary: pd.DataFrame              # per-team headline stats (title probability, etc.)
    model: DixonColesModel


def build_current_forecast(
    n_simulations: int = DEFAULT_N_SIMULATIONS, seed: int | None = 0
) -> LiveForecast:
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

    tracker = LeagueTableTracker()
    for _, row in played.sort_values("Date").iterrows():
        tracker.snapshot(row["HomeTeam"], row["AwayTeam"], current_season)
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

    historical = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "matches.parquet")
    train = pd.concat(
        [historical, played[historical.columns.intersection(played.columns)]], ignore_index=True
    ).sort_values("Date").reset_index(drop=True)

    model = DixonColesModel(**LIVE_FORECAST_POISSON_CONFIG).fit(train)

    fixtures = []
    fixture_rows = []
    for _, row in remaining.iterrows():
        lam_h, lam_a = model.predict_expected_goals(row["HomeTeam"], row["AwayTeam"])
        fixtures.append(Fixture(row["HomeTeam"], row["AwayTeam"], lam_h, lam_a))
        fixture_rows.append({
            "HomeTeam": row["HomeTeam"], "AwayTeam": row["AwayTeam"],
            "expected_home_goals": lam_h, "expected_away_goals": lam_a,
        })
    remaining_fixtures = pd.DataFrame(fixture_rows)

    positions, points = simulate_final_tables(current, fixtures, n_simulations=n_simulations, seed=seed)
    summary = summarize_simulation(positions, points)

    return LiveForecast(
        season=current_season,
        generated_at=pd.Timestamp.now(),
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
