"""Phase 13: does the SIMULATOR'S output mean what it says?

Phases 3-7 already checked whether the match-level model's probabilities
are calibrated (a 70% predicted home win happens ~70% of the time, across
thousands of individual matches). Phase 8 sanity-checked the full
live-forecast pipeline - fit model, build current table, simulate 20,000
seasons - against exactly one historical instance (2020-21, frozen at a
January cutoff) and it looked sensible. But "looked sensible once" isn't
the same claim as "is calibrated": a simulator built entirely correctly
(point rules, valid tables, reproducibility - all unit-tested) can still
be systematically over- or under-confident once thousands of match-level
probabilities get compounded into a season-long title/top-4/relegation
probability, in a way one example can't reveal either direction of.

This script runs src/simulation/calibration_backtest.py's backtest across
every eligible season (the standard min_train_seasons=10 convention used
everywhere else in this project - skip the first 10 seasons as pure
training data) and three cutoffs per season (25%/50%/75% of the way
through - an early-, mid-, and late-season snapshot), each one simulating
the rest of that season from src/live_forecast.py's exact live pipeline
(simulate_historical_cutoff, not a parallel implementation) and comparing
the simulated title/top-4/relegation probabilities against what actually
happened. Pooling every (season, cutoff, team) instance turns this into
the same predicted-probability/actual-outcome shape Phase 7's
calibration_curve()/expected_calibration_error() were built for - reused
here completely unchanged, which is only possible because Phase 7 built
them generically rather than hardcoded to match-level W/D/L outcomes.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.calibration import calibration_curve, expected_calibration_error
from src.simulation.calibration_backtest import run_calibration_backtest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIN_TRAIN_SEASONS = 10
CUTOFF_FRACTIONS = (0.25, 0.5, 0.75)
# 20,000 rather than live_forecast's default 50,000 - the same reduced
# sample size Phase 7/8's backtests use, since this runs the simulator
# ~69 times (23 eligible seasons x 3 cutoffs) rather than once; the extra
# Monte Carlo noise at 20k is negligible next to the bin widths a
# calibration curve looks at.
N_SIMULATIONS = 20_000

OUTCOMES = {
    "title": ("predicted_title", "actual_title"),
    "top4": ("predicted_top4", "actual_top4"),
    "relegation": ("predicted_relegation", "actual_relegation"),
}


def main() -> None:
    matches = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "matches.parquet")
    seasons = sorted(matches["Season"].unique())
    eligible = seasons[MIN_TRAIN_SEASONS:]
    print(f"[calibration-backtest] {len(eligible)} eligible seasons "
          f"({eligible[0]} to {eligible[-1]}) x {len(CUTOFF_FRACTIONS)} cutoffs "
          f"= {len(eligible) * len(CUTOFF_FRACTIONS)} simulated instances")

    results = run_calibration_backtest(
        matches,
        min_train_seasons=MIN_TRAIN_SEASONS,
        cutoff_fractions=CUTOFF_FRACTIONS,
        n_simulations=N_SIMULATIONS,
        seed=0,
    )
    results_out = PROJECT_ROOT / "experiments" / "simulation_calibration_results.csv"
    results.to_csv(results_out, index=False)
    print(f"[calibration-backtest] wrote {len(results)} raw (season, cutoff, team) rows -> {results_out}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

    ece_rows = []
    for label, (pred_col, actual_col) in OUTCOMES.items():
        # Title and relegation are rare-event outcomes (~1 in 20 teams
        # each season) - 10 equal-width bins over [0, 1] leave the upper
        # bins thin, but calibration_curve() already drops empty bins
        # rather than fabricating a point for them, so this just means a
        # coarser (not wrong) picture in the high-probability range.
        n_bins = 10
        curve = calibration_curve(results[actual_col], results[pred_col], n_bins=n_bins)
        ece = expected_calibration_error(results[actual_col], results[pred_col], n_bins=n_bins)
        ece_rows.append({"outcome": label, "ece": ece, "n": len(results)})

        print(f"\n=== {label} ===")
        print(curve.to_string(index=False))
        print(f"ECE: {ece:.4f}")

        ax.plot(curve["mean_predicted"], curve["observed_frequency"], marker="o", label=label)

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Simulator calibration: {len(eligible)} seasons x {len(CUTOFF_FRACTIONS)} cutoffs")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()

    plot_out = PROJECT_ROOT / "experiments" / "simulation_calibration.png"
    fig.savefig(plot_out, dpi=150)
    print(f"\n[calibration-backtest] wrote reliability diagram -> {plot_out}")

    ece_df = pd.DataFrame(ece_rows).set_index("outcome")
    ece_out = PROJECT_ROOT / "experiments" / "simulation_calibration_ece.csv"
    ece_df.to_csv(ece_out)
    print(f"[calibration-backtest] wrote ECE summary -> {ece_out}\n")
    print(ece_df.to_string())


if __name__ == "__main__":
    main()
