"""Turns raw simulation output (a positions-per-simulation, points-per-
simulation table from src/simulation/engine.py) into the actual numbers
the project plan asks for: title probability, top-4 probability,
relegation probability, expected position, position distribution,
expected points, points distribution.

European qualification cutoffs are configurable rather than hardcoded
(project plan requirement) - the real UEFA qualification rules are
genuinely complex and change year to year (how many Champions League spots
a country gets depends on a rolling coefficient; Europa/Conference League
spots also depend on domestic cup winners, which this project doesn't
model). DEFAULT_COMPETITION_CONFIG below is a simplified, clearly-labeled
approximation (fixed position cutoffs only), not a claim of exact UEFA
rule accuracy.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_COMPETITION_CONFIG = {
    "champions_league": 4,   # positions 1-4
    "europa_league": 1,      # position 5
    "conference_league": 1,  # position 6
    "relegation": 3,         # bottom 3 positions
}


def title_probability(positions: pd.DataFrame) -> pd.Series:
    return (positions == 1).mean().rename("title_probability")


def top_n_probability(positions: pd.DataFrame, n: int) -> pd.Series:
    return (positions <= n).mean().rename(f"top_{n}_probability")


def relegation_probability(positions: pd.DataFrame, n_relegated: int) -> pd.Series:
    n_teams = positions.shape[1]
    cutoff = n_teams - n_relegated
    return (positions > cutoff).mean().rename("relegation_probability")


def expected_position(positions: pd.DataFrame) -> pd.Series:
    return positions.mean().rename("expected_position")


def position_distribution(positions: pd.DataFrame) -> pd.DataFrame:
    """team x position matrix of P(team finishes in that exact position)."""
    n_teams = positions.shape[1]
    dist = pd.DataFrame(
        {pos: (positions == pos).mean() for pos in range(1, n_teams + 1)}
    )
    dist.columns.name = "position"
    return dist


def points_distribution(points: pd.DataFrame, percentiles: tuple[float, ...] = (5, 50, 95)) -> pd.DataFrame:
    rows = {
        "mean": points.mean(),
        "median": points.median(),
        **{f"p{int(p)}": points.quantile(p / 100) for p in percentiles},
    }
    return pd.DataFrame(rows)


def summarize_simulation(
    positions: pd.DataFrame,
    points: pd.DataFrame,
    competition_config: dict = DEFAULT_COMPETITION_CONFIG,
) -> pd.DataFrame:
    """One row per team with every headline simulation statistic - the
    table the project plan's dashboard/API (Phases 9-11) will read from
    directly."""
    n_champions_league = competition_config["champions_league"]
    n_european_total = (
        competition_config["champions_league"]
        + competition_config["europa_league"]
        + competition_config["conference_league"]
    )
    n_relegated = competition_config["relegation"]

    summary = pd.DataFrame({
        "title_probability": title_probability(positions),
        "champions_league_probability": top_n_probability(positions, n_champions_league),
        "european_qualification_probability": top_n_probability(positions, n_european_total),
        "relegation_probability": relegation_probability(positions, n_relegated),
        "expected_position": expected_position(positions),
        "expected_points": points.mean(),
        "median_points": points.median(),
        "points_p05": points.quantile(0.05),
        "points_p95": points.quantile(0.95),
    })
    return summary.sort_values("expected_position")
