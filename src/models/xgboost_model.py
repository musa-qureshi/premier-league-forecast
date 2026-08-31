"""Model 4: gradient boosting (XGBoost) over a broad engineered feature set.

Why this model needs to exist, specifically: Phases 3-5 built three
fairly different modeling approaches - Elo (a single calibrated rating
gap), Poisson/Dixon-Coles (per-team attack/defense strength), and logistic
regression (11 curated features) - and all three landed within about 0.006
log loss of each other (see README "Model comparison"), with Elo very
slightly ahead every time. That closeness raises a genuinely open,
testable question: is it because these signals have little more to give
ANY model, or because every model tried so far is structurally LINEAR
(Elo's calibration, logistic regression, and Poisson's attack+defense are
all additive in their respective link functions) and there's real signal
sitting in interactions none of them can represent - e.g. recent form
possibly mattering more specifically when Elo ratings are close, or a rest
advantage mattering more for an away fixture than a home one? Gradient-
boosted trees can capture exactly that kind of non-linearity and feature
interaction automatically, via how tree splits combine. This model's
purpose is to actually test that hypothesis against the backtest, not to
assume "more flexible model = automatically better" - see the README
result for whether it holds up.

Feature set is deliberately broader than Model 2's: logistic regression
had to curate features to avoid feeding a linear model severely collinear
inputs (all three rolling-window sizes measure closely related things).
Tree ensembles don't have that problem - each split considers one feature
at a time, and boosting is naturally robust to redundant inputs - so this
model uses all three rolling windows (3/5/10) plus everything else
available. XGBoost also has native missing-value handling (each split
learns which direction a missing value should default to), so unlike
Model 2's fixed-zero imputation, NaN rolling-form values for a team's very
first match are passed straight through, not imputed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.evaluation.metrics import CLASS_ORDER

# CLASS_ORDER = ["A", "D", "H"] - this mapping is what lets XGBoost's
# integer-encoded predictions be read back out in that same column order.
_LABEL_TO_INT = {c: i for i, c in enumerate(CLASS_ORDER)}

FEATURE_COLUMNS = [
    "elo_home_pre", "elo_away_pre", "elo_diff",
    "home_rolling_gf_3", "home_rolling_ga_3", "home_rolling_ppg_3",
    "home_rolling_win_rate_3", "home_rolling_draw_rate_3",
    "home_rolling_gf_5", "home_rolling_ga_5", "home_rolling_ppg_5",
    "home_rolling_win_rate_5", "home_rolling_draw_rate_5",
    "home_rolling_gf_10", "home_rolling_ga_10", "home_rolling_ppg_10",
    "home_rolling_win_rate_10", "home_rolling_draw_rate_10",
    "away_rolling_gf_3", "away_rolling_ga_3", "away_rolling_ppg_3",
    "away_rolling_win_rate_3", "away_rolling_draw_rate_3",
    "away_rolling_gf_5", "away_rolling_ga_5", "away_rolling_ppg_5",
    "away_rolling_win_rate_5", "away_rolling_draw_rate_5",
    "away_rolling_gf_10", "away_rolling_ga_10", "away_rolling_ppg_10",
    "away_rolling_win_rate_10", "away_rolling_draw_rate_10",
    "home_rest_days", "away_rest_days", "rest_days_diff",
    "attack_vs_defence_3", "away_attack_vs_home_defence_3", "ppg_diff_3",
    "attack_vs_defence_5", "away_attack_vs_home_defence_5", "ppg_diff_5",
    "attack_vs_defence_10", "away_attack_vs_home_defence_10", "ppg_diff_10",
    "home_table_points", "home_table_played", "home_table_goal_diff", "home_table_position",
    "away_table_points", "away_table_played", "away_table_goal_diff", "away_table_position",
]


def _prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURE_COLUMNS]  # NaNs deliberately left in - see module docstring


class XGBoostOutcomeModel:
    def __init__(self, n_estimators: int = 200, max_depth: int = 3, learning_rate: float = 0.05) -> None:
        """Defaults chosen to be conservative (shallow trees, moderate
        learning rate) given this dataset's size (~7,000-10,000 training
        rows per fold) relative to the feature count (~50) - not yet
        tuned via a dedicated hyperparameter search, which is future work
        rather than something hardcoded as "correct" here."""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self._clf: XGBClassifier | None = None

    def fit(self, train: pd.DataFrame) -> "XGBoostOutcomeModel":
        X = _prepare_X(train)
        y = train["FTR"].map(_LABEL_TO_INT).to_numpy()
        self._clf = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective="multi:softprob",
            num_class=len(CLASS_ORDER),
            eval_metric="mlogloss",
            tree_method="hist",
            importance_type="gain",
        )
        self._clf.fit(X, y)
        return self

    def predict_proba(self, test: pd.DataFrame) -> pd.DataFrame:
        if self._clf is None:
            raise RuntimeError("Call fit() before predict_proba().")
        X = _prepare_X(test)
        proba = self._clf.predict_proba(X)  # columns already in 0,1,2 == A,D,H order
        return pd.DataFrame(proba, columns=CLASS_ORDER, index=test.index)

    def feature_importances(self) -> pd.Series:
        """Gain-based feature importance (average improvement in the
        splitting criterion contributed by each feature) - see README
        "Explainability" for why this, rather than plain split-count
        importance, is reported."""
        if self._clf is None:
            raise RuntimeError("Call fit() before feature_importances().")
        return pd.Series(
            self._clf.feature_importances_, index=FEATURE_COLUMNS
        ).sort_values(ascending=False)
