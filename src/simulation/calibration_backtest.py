"""Systematic simulation-level calibration backtest.

Phases 3-7 validated the MATCH-level model's calibration (does a 70%
predicted home-win probability actually happen ~70% of the time). Phase 8
validated the SIMULATOR qualitatively, but against exactly one historical
instance (the 2020-21 season, frozen at a January cutoff). Neither answers
the real question a simulator's output needs to answer honestly: across
many different forecasts, does "this team has a 65% title probability"
actually correspond to that team winning the title roughly 65% of the
time? A simulator can be built entirely correctly (Phase 8's unit tests
already cover that: point assignment, valid tables, reproducibility) and
still be systematically over- or under-confident if the underlying match
probabilities it's fed are - which is exactly the kind of thing calibration
checks catch that a single validated example can't.

Methodology: for every eligible season (the same `min_train_seasons`
convention used everywhere else in this project - skip the first 10
seasons as pure training-only) and several fractions of the way through
that season's fixture list (25%, 50%, 75% by default - an early-, mid-,
and late-season snapshot), simulate the rest of that season from that
cutoff (src/live_forecast.py::simulate_historical_cutoff - the exact same
pipeline the live forecast uses, just replayed against a historical
cutoff instead of "right now"), then compare the simulated title/top-4/
relegation probabilities against what ACTUALLY happened in that season
(known, since it's historical).

Aggregated across every (season, cutoff, team) instance, this becomes
exactly the same shape src/evaluation/calibration.py's calibration_curve()
and expected_calibration_error() were built for - a predicted probability
paired with a 0/1 actual outcome - reused here completely unchanged for a
validation task those functions were never specifically written for. That
reuse is only possible because Phase 7 built them generically (any
predicted-probability/actual-outcome pair) rather than hardcoded to
match-level W/D/L outcomes.
"""

from __future__ import annotations

import pandas as pd

from src.features.league_table import LeagueTableTracker
from src.live_forecast import simulate_historical_cutoff
from src.simulation.summary import DEFAULT_COMPETITION_CONFIG

DEFAULT_MIN_TRAIN_SEASONS = 10
DEFAULT_CUTOFF_FRACTIONS = (0.25, 0.5, 0.75)


def actual_final_outcomes(matches: pd.DataFrame, season: str, competition_config: dict | None = None) -> pd.DataFrame:
    """The real final table for one season, as 0/1 title/top-4/relegation
    outcomes per team - computed from the complete historical record. This
    is the ground truth every simulated forecast in this backtest gets
    checked against. Uses the exact same tracker (and the same points ->
    goal-difference -> goals-scored tie-break, no head-to-head) as every
    other historical-table computation in this project, for consistency."""
    config = competition_config or DEFAULT_COMPETITION_CONFIG

    season_matches = matches[matches["Season"] == season].sort_values("Date")
    tracker = LeagueTableTracker()
    for _, row in season_matches.iterrows():
        tracker.snapshot(row["HomeTeam"], row["AwayTeam"], season)
        tracker.apply_result(row["HomeTeam"], row["AwayTeam"], row["FTHG"], row["FTAG"])
    standings = tracker.current_standings()

    ranked = sorted(
        standings.items(),
        key=lambda kv: (-kv[1]["points"], -kv[1]["goal_difference"], -kv[1]["goals_for"]),
    )
    n_teams = len(ranked)
    n_champions_league = config["champions_league"]
    n_european_total = config["champions_league"] + config["europa_league"] + config["conference_league"]
    n_relegated = config["relegation"]

    rows = []
    for position, (team, _) in enumerate(ranked, start=1):
        rows.append({
            "team": team,
            "actual_position": position,
            "actual_title": position == 1,
            "actual_top4": position <= n_champions_league,
            "actual_european": position <= n_european_total,
            "actual_relegated": position > (n_teams - n_relegated),
        })
    return pd.DataFrame(rows).set_index("team")


def cutoff_date_for_fraction(matches: pd.DataFrame, season: str, fraction: float) -> pd.Timestamp:
    """The date of the match `fraction` of the way through that season's
    fixture list, chronologically - e.g. fraction=0.5 lands around the
    season's halfway point regardless of exactly how many matches that
    season had (462 for the 22-team era, 380 since, or a truncated count
    for an incomplete source season). Matches with Date < this are
    "played"; Date >= this are "remaining"."""
    season_dates = matches.loc[matches["Season"] == season, "Date"].sort_values().reset_index(drop=True)
    idx = min(int(len(season_dates) * fraction), len(season_dates) - 1)
    return season_dates.iloc[idx]


def run_calibration_backtest(
    matches: pd.DataFrame,
    min_train_seasons: int = DEFAULT_MIN_TRAIN_SEASONS,
    cutoff_fractions: tuple[float, ...] = DEFAULT_CUTOFF_FRACTIONS,
    n_simulations: int = 20_000,
    seed: int | None = 0,
) -> pd.DataFrame:
    """Runs the full backtest: for every eligible season and cutoff
    fraction, simulates the rest of that season and compares against the
    real outcome. Returns one row per (season, cutoff_fraction, team) with
    predicted probabilities and actual 0/1 outcomes - the raw material fed
    into calibration_curve()/expected_calibration_error() in
    experiments/run_phase13_simulation_calibration.py.
    """
    seasons = sorted(matches["Season"].unique())
    eligible_seasons = seasons[min_train_seasons:]

    rows = []
    for season in eligible_seasons:
        actuals = actual_final_outcomes(matches, season)
        for fraction in cutoff_fractions:
            cutoff = cutoff_date_for_fraction(matches, season, fraction)
            forecast = simulate_historical_cutoff(
                matches, season, cutoff, n_simulations=n_simulations, seed=seed
            )
            for team, summary_row in forecast.summary.iterrows():
                if team not in actuals.index:
                    continue  # a team with no remaining fixtures data (shouldn't happen; guarded anyway)
                actual = actuals.loc[team]
                rows.append({
                    "season": season,
                    "cutoff_fraction": fraction,
                    "team": team,
                    "predicted_title": summary_row["title_probability"],
                    "actual_title": int(actual["actual_title"]),
                    "predicted_top4": summary_row["champions_league_probability"],
                    "actual_top4": int(actual["actual_top4"]),
                    "predicted_relegation": summary_row["relegation_probability"],
                    "actual_relegation": int(actual["actual_relegated"]),
                })
    return pd.DataFrame(rows)
