"""Assembles the full leakage-safe feature matrix from the cleaned match
dataset: Elo ratings, rolling-form stats, league-table state, and their
derived relative features.

This is the single entry point Phase 3+ models should read from - no model
should ever need to know how any individual feature was computed, and no
model should ever read data/processed/matches.parquet directly and
recompute features ad hoc (that's exactly how a leakage bug creeps in
silently: two slightly different implementations of "current form" in two
different notebooks).

Note on "home advantage" as an explicit feature (mentioned in the project
plan's feature list): in this match-level row shape (one row per fixture,
home_*/away_* columns side by side), a literal is_home column would be
constant (always 1) and carry no information - home-field advantage is
already structurally encoded by the asymmetric home_*/away_* features and,
for Elo specifically, by the home_advantage constant inside the rating
model itself (see src/features/elo.py). A meaningful is_home indicator only
makes sense in a long/team-perspective row shape (one row per team per
match) - not needed for the W/D/L-per-fixture models this project targets.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.features.elo import compute_elo_features
from src.features.league_table import compute_league_table_features
from src.features.rolling import attach_rolling_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_features(matches: pd.DataFrame, elo_config: dict | None = None) -> pd.DataFrame:
    """Runs the full feature pipeline in the only order that's leakage-safe
    and internally consistent: Elo first (needs only Date/HomeTeam/AwayTeam/
    goals/Season), then rolling form (same requirements), then league-table
    state (also needs the same base columns - order relative to Elo/rolling
    doesn't matter for it, since it only reads from the original match
    columns, not from Elo/rolling's output)."""
    if not matches["Date"].is_monotonic_increasing:
        raise ValueError("build_features requires matches sorted chronologically.")

    result = compute_elo_features(matches, config=elo_config)
    result = attach_rolling_features(result)
    result = compute_league_table_features(result)
    return result


def main() -> None:
    matches = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "matches.parquet")
    with open(PROJECT_ROOT / "configs" / "elo.yaml") as f:
        elo_config = yaml.safe_load(f)

    features = build_features(matches, elo_config=elo_config)

    out_path = PROJECT_ROOT / "data" / "processed" / "features.parquet"
    features.to_parquet(out_path, index=False)
    print(f"[build_dataset] wrote {len(features)} rows x {len(features.columns)} "
          f"columns -> {out_path}")

    n_nan_elo = features["elo_diff"].isna().sum()
    n_nan_rolling5 = features["home_rolling_ppg_5"].isna().sum()
    print(f"[build_dataset] rows with NaN elo_diff: {n_nan_elo} (expect 0 - every "
          f"team gets an Elo rating from its very first match)")
    print(f"[build_dataset] rows with NaN home_rolling_ppg_5: {n_nan_rolling5} "
          f"(expect > 0 - a team's first-ever match has no prior form)")


if __name__ == "__main__":
    main()
