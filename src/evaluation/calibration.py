"""Calibration analysis: are the model's predicted probabilities honest?

Why this matters specifically for this project, beyond just being another
metric: log loss/Brier/RPS reward good calibration ON AVERAGE, but a model
can post a solid aggregate log loss while still being subtly miscalibrated
in exactly the range that matters most once its probabilities get fed into
the Monte Carlo simulator (Phase 8). If a model says "70% chance of a home
win" across a group of matches, but the true rate in situations like those
is actually 55%, every simulated season built from that model quietly
overstates how often that kind of team wins - not by much in any single
match, but compounded across a 38-game season and then across thousands of
simulated seasons, a small systematic miscalibration can turn into a
materially wrong title/relegation probability. Log loss doesn't surface
this shape of error directly the way a calibration curve does; a model can
have good log loss and still be worth checking here before trusting it to
drive the simulator.

Reliability diagram / calibration curve: pick ONE outcome at a time (e.g.
"home win"), bin the model's predicted probability of that outcome into
buckets (deciles, by default), and for each bucket compare the model's
average predicted probability there against the fraction of matches in
that bucket where the outcome actually happened. A perfectly calibrated
model's points fall on the diagonal (predicted == observed); points above
the diagonal mean the model is underconfident in that range, points below
mean overconfident.

Expected Calibration Error (ECE) summarizes a reliability diagram into one
number: the bin-size-weighted average absolute gap between predicted and
observed probability across bins.

    ECE = sum_b (n_b / N) * |predicted_avg_b - observed_freq_b|
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calibration_curve(y_true, predicted_prob, n_bins: int = 10) -> pd.DataFrame:
    """For ONE outcome class at a time: `y_true` is a 0/1 (or bool)
    indicator of whether that outcome actually happened for each match,
    `predicted_prob` is the model's predicted probability of that same
    outcome for the same matches. Bins by predicted probability into
    `n_bins` equal-width bins over [0, 1] and returns one row per
    non-empty bin: the bin's mean predicted probability, observed
    frequency, and match count.
    """
    df = pd.DataFrame({
        "y": np.asarray(y_true, dtype=float),
        "p": np.asarray(predicted_prob, dtype=float),
    })
    bin_edges = np.linspace(0, 1, n_bins + 1)
    df["bin"] = pd.cut(df["p"], bins=bin_edges, include_lowest=True)

    grouped = (
        df.groupby("bin", observed=True)
        .agg(mean_predicted=("p", "mean"), observed_frequency=("y", "mean"), count=("y", "size"))
        .reset_index()
    )
    return grouped


def expected_calibration_error(y_true, predicted_prob, n_bins: int = 10) -> float:
    curve = calibration_curve(y_true, predicted_prob, n_bins)
    n = curve["count"].sum()
    weighted_gap = (curve["count"] / n) * (curve["mean_predicted"] - curve["observed_frequency"]).abs()
    return float(weighted_gap.sum())
