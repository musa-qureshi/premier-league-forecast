"""Unit tests for src/models/xgboost_model.py."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import CLASS_ORDER
from src.models.xgboost_model import FEATURE_COLUMNS, XGBoostOutcomeModel


def _synthetic_training_data(n_per_bucket: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(n_per_bucket):
        base = {c: rng.normal(0, 1) for c in FEATURE_COLUMNS}
        home_strong = {**base, "elo_diff": 300 + rng.normal(0, 20), "FTR": "H"}
        rows.append(home_strong)

        base2 = {c: rng.normal(0, 1) for c in FEATURE_COLUMNS}
        away_strong = {**base2, "elo_diff": -300 + rng.normal(0, 20), "FTR": "A"}
        rows.append(away_strong)

        base3 = {c: rng.normal(0, 1) for c in FEATURE_COLUMNS}
        even = {**base3, "elo_diff": rng.normal(0, 20), "FTR": rng.choice(["H", "D", "A"])}
        rows.append(even)
    return pd.DataFrame(rows)


class TestXGBoostOutcomeModel:
    def test_predicted_probabilities_are_valid(self):
        model = XGBoostOutcomeModel(n_estimators=20).fit(_synthetic_training_data())
        test = _synthetic_training_data(n_per_bucket=5)
        proba = model.predict_proba(test)
        assert list(proba.columns) == CLASS_ORDER
        assert (proba.sum(axis=1) - 1.0).abs().max() < 1e-4
        assert (proba.to_numpy() >= 0).all()

    def test_handles_nan_features_without_crashing(self):
        # XGBoost has native missing-value handling, unlike the logistic
        # model, which needed explicit imputation - this test confirms
        # NaN rows really do pass straight through rather than erroring.
        train = _synthetic_training_data()
        model = XGBoostOutcomeModel(n_estimators=20).fit(train)
        test = pd.DataFrame({c: [np.nan] for c in FEATURE_COLUMNS})
        proba = model.predict_proba(test)
        assert (proba.sum(axis=1) - 1.0).abs().max() < 1e-4

    def test_large_positive_elo_diff_favors_home_win(self):
        model = XGBoostOutcomeModel(n_estimators=50).fit(_synthetic_training_data())
        test = pd.DataFrame({c: [0.0] for c in FEATURE_COLUMNS})
        test["elo_diff"] = 400
        proba = model.predict_proba(test)
        assert proba["H"].iloc[0] > proba["A"].iloc[0]

    def test_raises_if_predict_called_before_fit(self):
        with pytest.raises(RuntimeError):
            XGBoostOutcomeModel().predict_proba(pd.DataFrame({c: [0.0] for c in FEATURE_COLUMNS}))

    def test_raises_if_feature_importances_called_before_fit(self):
        with pytest.raises(RuntimeError):
            XGBoostOutcomeModel().feature_importances()

    def test_feature_importances_shape_and_validity(self):
        model = XGBoostOutcomeModel(n_estimators=20).fit(_synthetic_training_data())
        importances = model.feature_importances()
        assert set(importances.index) == set(FEATURE_COLUMNS)
        assert (importances >= 0).all()
        # sorted descending
        assert list(importances) == sorted(importances, reverse=True)
