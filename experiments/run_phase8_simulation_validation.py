"""Phase 8 validation: does the whole pipeline (train model on prior data
-> current table state -> remaining fixtures -> simulate 20,000 seasons)
actually produce a sensible forecast?

Uses a real, FULLY COMPLETED historical season (2020-21 - unlike the
truncated 2021-22, we know exactly how it really ended) frozen at a
January 1 cutoff: everything before that date is treated as "already
played" (both for fitting the model and for the current table state),
everything after is treated as "remaining fixtures" to simulate - exactly
mirroring how Phase 9's live current-season forecast will eventually work,
just validated retrospectively against a season whose real outcome is
already known, rather than an in-progress one whose outcome isn't.
"""

from pathlib import Path

import pandas as pd

from src.features.league_table import LeagueTableTracker
from src.models.poisson_model import DixonColesModel
from src.simulation.engine import CurrentTableState, Fixture, simulate_final_tables
from src.simulation.summary import summarize_simulation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_SEASON = "2020-21"
CUTOFF_DATE = pd.Timestamp("2021-01-01")
N_SIMULATIONS = 20_000


def main() -> None:
    matches = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "matches.parquet")

    # Everything up to the cutoff (prior seasons + this season so far) is
    # fair game for fitting - exactly what would genuinely be available
    # "as of" that date. Nothing after the cutoff is used for fitting.
    train = matches[matches["Date"] < CUTOFF_DATE]
    print(f"[validation] training on {len(train)} matches through {CUTOFF_DATE.date()}")
    model = DixonColesModel(use_correlation=True, xi=0.3).fit(train)

    season_matches = matches[matches["Season"] == VALIDATION_SEASON].sort_values("Date")
    played_so_far = season_matches[season_matches["Date"] < CUTOFF_DATE]
    remaining = season_matches[season_matches["Date"] >= CUTOFF_DATE]
    print(f"[validation] {VALIDATION_SEASON}: {len(played_so_far)} played before cutoff, "
          f"{len(remaining)} remaining fixtures to simulate")

    # Current table state as of the cutoff, using the exact same
    # leakage-safe tracker Phase 2 uses for historical features.
    tracker = LeagueTableTracker()
    for _, row in played_so_far.iterrows():
        tracker.snapshot(row["HomeTeam"], row["AwayTeam"], row["Season"])
        tracker.apply_result(row["HomeTeam"], row["AwayTeam"], row["FTHG"], row["FTAG"])

    standings = tracker.current_standings()
    teams = sorted(standings.keys())
    current = CurrentTableState(
        teams=teams,
        points={t: standings[t]["points"] for t in teams},
        goals_for={t: standings[t]["goals_for"] for t in teams},
        goals_against={t: standings[t]["goals_against"] for t in teams},
    )

    fixtures = []
    for _, row in remaining.iterrows():
        lam_h, lam_a = model.predict_expected_goals(row["HomeTeam"], row["AwayTeam"])
        fixtures.append(Fixture(row["HomeTeam"], row["AwayTeam"], lam_h, lam_a))

    positions, points = simulate_final_tables(current, fixtures, n_simulations=N_SIMULATIONS, seed=0)
    summary = summarize_simulation(positions, points)

    # What actually happened - the real final table for this season.
    real_tracker = LeagueTableTracker()
    for _, row in season_matches.iterrows():
        real_tracker.snapshot(row["HomeTeam"], row["AwayTeam"], row["Season"])
        real_tracker.apply_result(row["HomeTeam"], row["AwayTeam"], row["FTHG"], row["FTAG"])
    real_standings = real_tracker.current_standings()
    actual_points = pd.Series({t: real_standings[t]["points"] for t in teams}, name="actual_final_points")
    actual_rank = actual_points.rank(ascending=False, method="min").astype(int).rename("actual_final_position")

    comparison = summary.join(actual_points).join(actual_rank)
    comparison = comparison.sort_values("actual_final_position")

    print(f"\n=== Simulated forecast (as of {CUTOFF_DATE.date()}) vs. actual {VALIDATION_SEASON} result ===")
    cols = ["title_probability", "champions_league_probability", "relegation_probability",
            "expected_position", "actual_final_position", "expected_points", "actual_final_points"]
    print(comparison[cols].to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
