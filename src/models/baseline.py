"""Model 0: naive baselines - the performance floor every other model in
this project must beat to justify its own existence.

FrequencyBaseline is the real probabilistic baseline: it learns the
unconditional H/D/A frequency from the training window (e.g. historically
around 46% home win / 25% draw / 29% away win) and predicts that same fixed
distribution for every match, regardless of which teams are playing. This
is the standard "climatology" baseline used in forecasting generally
(weather forecasting calls the equivalent "always predict the historical
average") - it has genuinely zero skill at distinguishing matches from each
other, but it IS a valid, well-calibrated probability distribution, so
log loss/Brier/RPS can be computed on it meaningfully.

`always_predict_home_win_accuracy` is deliberately NOT implemented as a
probabilistic model - see its docstring for why.
"""

from __future__ import annotations

import pandas as pd

from src.evaluation.metrics import CLASS_ORDER


class FrequencyBaseline:
    """Predicts the training set's unconditional H/D/A frequency for every
    match. Has no features, no per-match differentiation - it exists to
    give every metric a concrete number below which a model has learned
    nothing at all."""

    def __init__(self) -> None:
        self.frequencies_: dict[str, float] | None = None

    def fit(self, train: pd.DataFrame) -> "FrequencyBaseline":
        counts = train["FTR"].value_counts(normalize=True)
        self.frequencies_ = {c: float(counts.get(c, 0.0)) for c in CLASS_ORDER}
        return self

    def predict_proba(self, test: pd.DataFrame) -> pd.DataFrame:
        if self.frequencies_ is None:
            raise RuntimeError("Call fit() before predict_proba().")
        row = {c: self.frequencies_[c] for c in CLASS_ORDER}
        return pd.DataFrame([row] * len(test), index=test.index)


def always_predict_home_win_accuracy(y_true: pd.Series) -> float:
    """The accuracy (NOT a probabilistic forecast) of a decision rule that
    always guesses "home win", regardless of the teams.

    This is deliberately kept separate from FrequencyBaseline and excluded
    from the log loss/Brier/RPS comparison table entirely, because it
    isn't really a probabilistic model at all - "predict home win with
    100% confidence, always" would produce infinite log loss the moment it's
    wrong even once, which is a meaningless number to report. What IS worth
    reporting is its accuracy, specifically because that number is
    deceptively close to far more sophisticated models' accuracy (home win
    is the single most common outcome in football, ~46% of matches
    historically) - which is the concrete illustration of why this project
    does not select models by accuracy alone (see README "Model
    evaluation")."""
    return float((pd.Series(y_true) == "H").mean())
