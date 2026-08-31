"""Model 2: multinomial logistic regression over a curated, multi-signal
feature set - Elo, recent form, matchup-specific attack/defense
comparisons, rest, and league-table context.

Why keep this separate from Model 1 (Elo, src/models/elo_model.py), when
both are "just" multinomial logistic regression under the hood? Model 1
answers "how much of the story does Elo alone tell" with a single input
feature. Model 2 answers a different question: does adding the richer
signal set the project plan calls for (recent form, attacking/defensive
matchup, rest, table position) actually improve on Elo alone, once
properly regularized and compared on the same backtest? That's an
empirical question this model's whole purpose is to let the comparison
table answer - not something to assume in either direction.

Feature selection is deliberately curated, not "every column in
features.parquet". Two reasons: including all three rolling-window sizes
(3/5/10) as separate raw features would hand a LINEAR model a set of
highly collinear inputs (a team's 3-match and 10-match form are strongly
correlated with each other), which mostly just makes the fitted
coefficients unstable and hard to interpret without adding real signal;
window=5 is used as a representative middle ground rather than all three.
Second, several columns in features.parquet exist for other models or as
intermediate bookkeeping (elo_home_pre/elo_away_pre alongside elo_diff,
prev_match_date, table `played` counts) and would be redundant or
uninformative here. FEATURE_COLUMNS below is the actual, explicit
contract of what this model is allowed to see - and doubles as a leakage
safeguard: anything not listed here (including every post-match column
like FTHG/HS/HY/etc.) is structurally impossible for this model to use.

Missing values: rolling-form-derived features are NaN for a team's very
first match in the dataset (no prior matches exist yet - see
src/features/rolling.py), and rest_days_diff is NaN whenever either side
has no previous match on record. Both are filled with a fixed 0 rather
than a fit-time statistic (e.g. the training mean) - simpler, fully
deterministic (identical treatment at fit and predict time, no risk of a
subtly different imputation value leaking between backtest folds), and a
defensible neutral value for features that are specifically DIFFERENCES
(home minus away): 0 reads as "no information to differentiate the two
sides yet", which is honestly what a team's competitive debut represents.
This affects under 0.5% of rows in this dataset.
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import CLASS_ORDER

FEATURE_COLUMNS = [
    "elo_diff",
    "ppg_diff_5",
    "attack_vs_defence_5",
    "away_attack_vs_home_defence_5",
    "rest_days_diff",
    "home_table_position",
    "away_table_position",
    "home_table_points",
    "away_table_points",
    "home_table_goal_diff",
    "away_table_goal_diff",
]


def _prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURE_COLUMNS].fillna(0.0)


class LogisticOutcomeModel:
    def __init__(self, C: float = 1.0) -> None:
        """`C` is sklearn's inverse-regularization-strength - lower C means
        stronger L2 regularization. Left at sklearn's sensible default
        rather than tuned here; regularization-strength tuning belongs in
        a dedicated hyperparameter search, not hardcoded into the model
        class."""
        self.C = C
        self._pipeline: Pipeline | None = None

    def fit(self, train: pd.DataFrame) -> "LogisticOutcomeModel":
        X = _prepare_X(train)
        y = train["FTR"].to_numpy()
        # StandardScaler matters here, not just as a nicety: L2
        # regularization penalizes coefficient magnitude, and this feature
        # set mixes wildly different scales (elo_diff in the hundreds,
        # ppg_diff_5 in single digits, table_position in tens) - without
        # scaling, regularization would unfairly suppress features that
        # merely happen to have small raw units, not features that are
        # genuinely uninformative.
        self._pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=self.C, max_iter=1000)),
        ])
        self._pipeline.fit(X, y)
        return self

    def predict_proba(self, test: pd.DataFrame) -> pd.DataFrame:
        if self._pipeline is None:
            raise RuntimeError("Call fit() before predict_proba().")
        X = _prepare_X(test)
        proba = self._pipeline.predict_proba(X)
        clf = self._pipeline.named_steps["clf"]
        assert list(clf.classes_) == CLASS_ORDER, (
            f"Expected sklearn class order {CLASS_ORDER}, got {list(clf.classes_)}"
        )
        return pd.DataFrame(proba, columns=CLASS_ORDER, index=test.index)

    def coefficients(self) -> pd.DataFrame:
        """Standardized coefficients per class, for interpretability - see
        README "Explainability" section. Because features were
        standardized before fitting, coefficients are directly comparable
        to each other in magnitude (unlike raw-scale coefficients, where a
        big number might just mean "small units", not "important")."""
        if self._pipeline is None:
            raise RuntimeError("Call fit() before coefficients().")
        clf = self._pipeline.named_steps["clf"]
        return pd.DataFrame(clf.coef_, index=clf.classes_, columns=FEATURE_COLUMNS)
