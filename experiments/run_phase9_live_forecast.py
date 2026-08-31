"""Phase 9: the actual current-season forecast - the payoff of everything
built so far. Downloads the live 2026-27 season file (played matches +
remaining fixture list) from openfootball/football.json, fits the Poisson/
Dixon-Coles model on the full historical dataset plus this season's
matches so far, simulates the remaining season 50,000 times, and reports
title/Champions-League/relegation probabilities and expected points for
every team.

Deliberately reuses existing modules rather than duplicating logic:
src/data/validate.py's load_openfootball_matches/canonicalize_teams for
parsing and cleaning, src/features/league_table.py's LeagueTableTracker for
the current table state, src/models/poisson_model.py for match
probabilities, and src/simulation/ for the simulation itself - the same
code paths already tested against historical data in Phases 2-8, not a
parallel implementation built just for this script.
"""

from pathlib import Path

import pandas as pd

from src.data.ingest import download_json, load_config
from src.data.validate import canonicalize_teams, load_openfootball_matches, load_team_name_map
from src.features.league_table import LeagueTableTracker
from src.models.poisson_model import DixonColesModel
from src.simulation.engine import CurrentTableState, Fixture, simulate_final_tables
from src.simulation.summary import summarize_simulation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
N_SIMULATIONS = 50_000


def main() -> None:
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
    remaining = all_current[all_current["FTHG"].isna()].copy()
    print(f"[live_forecast] {current_season}: {len(played)} played, "
          f"{len(remaining)} remaining fixtures")

    played["FTHG"] = played["FTHG"].astype(int)
    played["FTAG"] = played["FTAG"].astype(int)
    played["Season"] = current_season

    # Current table state, using the exact same tracker historical
    # features are built with (Phase 2) - not a separate implementation.
    tracker = LeagueTableTracker()
    for _, row in played.sort_values("Date").iterrows():
        tracker.snapshot(row["HomeTeam"], row["AwayTeam"], current_season)
        tracker.apply_result(row["HomeTeam"], row["AwayTeam"], row["FTHG"], row["FTAG"])
    standings = tracker.current_standings()
    teams = sorted(standings.keys())
    current = CurrentTableState(
        teams=teams,
        points={t: standings[t]["points"] for t in teams},
        goals_for={t: standings[t]["goals_for"] for t in teams},
        goals_against={t: standings[t]["goals_against"] for t in teams},
    )

    # Train on the full historical dataset PLUS this season's matches so
    # far - exactly what's genuinely available as of today, same approach
    # validated retrospectively in Phase 8 against the real 2020-21 season.
    historical = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "matches.parquet")
    train = pd.concat([historical, played[historical.columns.intersection(played.columns)]],
                       ignore_index=True)
    train = train.sort_values("Date").reset_index(drop=True)
    print(f"[live_forecast] training on {len(train)} matches through {train['Date'].max().date()}")

    # Deliberately NOT the tuned class default (xi=0.5, promoted_penalty=0.2,
    # see src/models/poisson_model.py) - explicit xi=0.3/promoted_penalty=0.4
    # instead. Phase 10's tuning validated hyperparameters against WHOLE-
    # SEASON-AHEAD predictions only (the expanding-window backtest trains on
    # complete prior seasons and predicts a complete season at once - it
    # never evaluates "predict the rest of a season after only 1-2
    # matchdays have been played", which is exactly this script's actual
    # scenario). Verified this matters, not just in theory: with the tuned
    # xi=0.5, Hull City - newly promoted, 1 match played, a single 2-0 win
    # over Manchester United - came out with a simulated 30.7% title
    # probability (2nd favorite, ahead of Arsenal), because the faster
    # recency decay overweights that one small-sample result and their
    # fitted defense parameter briefly looked stronger than most of the
    # league's. The more conservative xi=0.3/promoted_penalty=0.4 - the
    # same values Phase 8's validation (also a mid-season cutoff) used and
    # confirmed sensible against a season with a known real outcome - avoids
    # that failure mode. A dedicated mid-season-cutoff tuning pass (sweeping
    # xi validated specifically against partial-season predictions, the way
    # Phase 8 checks the simulator but Phase 10 never checked the
    # hyperparameters) is real future work this discovery motivates, not
    # yet built.
    model = DixonColesModel(use_correlation=True, xi=0.3, promoted_penalty=0.4).fit(train)

    fixtures = []
    for _, row in remaining.iterrows():
        lam_h, lam_a = model.predict_expected_goals(row["HomeTeam"], row["AwayTeam"])
        fixtures.append(Fixture(row["HomeTeam"], row["AwayTeam"], lam_h, lam_a))

    positions, points = simulate_final_tables(current, fixtures, n_simulations=N_SIMULATIONS, seed=0)
    summary = summarize_simulation(positions, points)

    print(f"\n=== {current_season} forecast ({N_SIMULATIONS:,} simulated seasons) ===")
    cols = ["title_probability", "champions_league_probability",
            "relegation_probability", "expected_position", "expected_points"]
    print(summary[cols].to_string(float_format=lambda x: f"{x:.3f}"))

    out_path = PROJECT_ROOT / "experiments" / "live_forecast_2026_27.csv"
    summary.to_csv(out_path)
    print(f"\n[live_forecast] wrote -> {out_path}")


if __name__ == "__main__":
    main()
