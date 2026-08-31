"""Probabilistic forecast-quality metrics for the W/D/L outcome.

Why not just accuracy? Accuracy only asks "was the single most likely
outcome correct" - it throws away everything about *how confident* the
model was, which is exactly the information this project's whole premise
(probabilistic forecasting, then Monte Carlo simulation) depends on. A
model that says "52% home win" and a model that says "95% home win" get
identical accuracy credit for a home win either way, but feeding the first
model's uncertainty-honest 52% into 100,000 simulated seasons produces a
very different (and much more useful) spread of outcomes than feeding in
the second model's overconfident 95%. The metrics below all grade the
*probabilities themselves*, not just whether the top pick was right.

CLASS_ORDER fixes a single consistent column ordering used by every metric
in this module: Away, Draw, Home. For log loss/Brier the ordering is
arbitrary (both are symmetric across classes), but it matters for RPS,
which treats the outcome as ordinal - see rps() below - so one shared
ordering is used everywhere to avoid ever mismatching a probability column
against the wrong outcome.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CLASS_ORDER = ["A", "D", "H"]


def _to_probability_matrix(proba: pd.DataFrame) -> np.ndarray:
    """Extracts columns in CLASS_ORDER as a plain (N, 3) array, and checks
    they're a valid probability distribution - this check exists because
    Monte Carlo simulation later assumes it can sample directly from these
    numbers, so a bug that produces probabilities which don't sum to 1
    needs to fail loudly here, not silently bias every simulated season."""
    arr = proba[CLASS_ORDER].to_numpy(dtype=float)
    row_sums = arr.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        bad = np.where(~np.isclose(row_sums, 1.0, atol=1e-6))[0]
        raise ValueError(
            f"Probabilities must sum to 1 per row; {len(bad)} row(s) don't "
            f"(e.g. row {bad[0]} sums to {row_sums[bad[0]]:.6f})."
        )
    if (arr < -1e-9).any():
        raise ValueError("Probabilities must be non-negative.")
    return np.clip(arr, 0.0, 1.0)


def _to_one_hot(y_true: pd.Series | np.ndarray) -> np.ndarray:
    y_true = pd.Series(y_true).to_numpy()
    invalid = set(y_true) - set(CLASS_ORDER)
    if invalid:
        raise ValueError(f"y_true contains labels outside {CLASS_ORDER}: {invalid}")
    return np.array([[1.0 if y == c else 0.0 for c in CLASS_ORDER] for y in y_true])


def log_loss(y_true: pd.Series, proba: pd.DataFrame, eps: float = 1e-15) -> float:
    """Mean negative log-probability the model assigned to the outcome that
    actually happened: -mean(log(P(actual outcome))).

    Intuition: log loss rewards being confidently right and punishes being
    confidently wrong much more harshly than being unsure. Predicting 99%
    for an outcome that doesn't happen costs far more than predicting 60%
    for one that doesn't - log(0.01) vs log(0.40) - so a model can't game
    log loss by just being maximally confident about its best guess; it's
    only rewarded for confidence that's actually justified. This
    asymmetry is exactly why log loss (unlike accuracy) is sensitive to
    calibration, which is what the Monte Carlo simulator needs.

    Mathematically, for N matches with true one-hot outcome o_ic and
    predicted probability p_ic over classes c:

        LogLoss = -(1/N) * sum_i sum_c  o_ic * log(p_ic)

    Since o_ic is 1 only for the actual outcome, this reduces to averaging
    -log(p) at just the actual-outcome column each match.
    """
    p = _to_probability_matrix(proba)
    o = _to_one_hot(y_true)
    p_clipped = np.clip(p, eps, 1 - eps)
    per_match = -(o * np.log(p_clipped)).sum(axis=1)
    return float(per_match.mean())


def brier_score(y_true: pd.Series, proba: pd.DataFrame) -> float:
    """Mean squared error between the predicted probability vector and the
    one-hot actual outcome, summed over the 3 classes and averaged over
    matches:

        Brier = (1/N) * sum_i sum_c (p_ic - o_ic)^2

    Intuition: treat each class's predicted probability as a separate
    "will this happen: yes/no" forecast and score it like a squared error
    against the true 0/1 outcome. Unlike log loss, Brier score is bounded
    (0 = perfect, 2 = worst possible for 3 classes under this sum-over-
    classes convention) and doesn't blow up toward infinity for a
    confident-and-wrong prediction - it penalizes overconfidence, but
    gently compared to log loss's asymptotic penalty.
    """
    p = _to_probability_matrix(proba)
    o = _to_one_hot(y_true)
    per_match = ((p - o) ** 2).sum(axis=1)
    return float(per_match.mean())


def rps(y_true: pd.Series, proba: pd.DataFrame) -> float:
    """Ranked Probability Score - Brier score's ordinal-aware sibling, and
    the standard metric in the football-forecasting research literature
    (e.g. Constantinou & Fenton) specifically because W/D/L is an ORDINAL
    outcome: a Draw is "between" an Away win and a Home win in a way plain
    multi-class Brier score doesn't know about.

    Intuition: instead of comparing predicted-vs-actual probability class
    by class, RPS compares predicted-vs-actual CUMULATIVE probability
    along the natural Away -> Draw -> Home ordering. That means a
    confidently-wrong prediction that's still "close" (predicting a big
    Home win when it was actually a Draw) is penalized less than one
    that's "far" (predicting a big Home win when it was actually a heavy
    Away win) - exactly the distinction plain Brier score is blind to,
    since it would score both misses identically.

    For r ordered classes:

        RPS = 1/(r-1) * sum_{i=1}^{r-1} (CP_i - CE_i)^2

    where CP_i and CE_i are the cumulative predicted and actual
    probabilities through class i (in CLASS_ORDER). Bounded 0 (perfect) to
    1 (worst possible).
    """
    p = _to_probability_matrix(proba)
    o = _to_one_hot(y_true)
    cum_p = np.cumsum(p, axis=1)[:, :-1]  # drop the final cumulative column (always 1.0 - 1.0 = 0)
    cum_o = np.cumsum(o, axis=1)[:, :-1]
    r_minus_1 = p.shape[1] - 1
    per_match = ((cum_p - cum_o) ** 2).sum(axis=1) / r_minus_1
    return float(per_match.mean())


def accuracy(y_true: pd.Series, proba: pd.DataFrame) -> float:
    """Fraction of matches where the highest-probability class was the
    actual outcome. Included for completeness/interpretability, but
    deliberately NOT the metric this project selects models by - see the
    module docstring and README "Model evaluation" section for why."""
    p = _to_probability_matrix(proba)
    y_true = pd.Series(y_true).to_numpy()
    pred_idx = p.argmax(axis=1)
    pred_class = np.array(CLASS_ORDER)[pred_idx]
    return float((pred_class == y_true).mean())


def evaluate_all(y_true: pd.Series, proba: pd.DataFrame) -> dict:
    """Convenience wrapper computing every metric at once, for backtest
    reporting."""
    return {
        "log_loss": log_loss(y_true, proba),
        "brier_score": brier_score(y_true, proba),
        "rps": rps(y_true, proba),
        "accuracy": accuracy(y_true, proba),
        "n_matches": len(y_true),
    }
