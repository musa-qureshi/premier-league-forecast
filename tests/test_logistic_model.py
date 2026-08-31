"""Unit tests for src/models/logistic_model.py."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import CLASS_ORDER
from src.models.logistic_model import FEATURE_COLUMNS, LogisticOutcomeModel, _prepare_X


def _synthetic_training_data(n_per_bucket: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(n_per_bucket):
        base = {c: rng.normal(0, 1) for c in FEATURE_COLUMNS}
        base_strong_home = {**base, "elo_diff": 300 + rng.normal(0, 20)}
        base_strong_home["FTR"] = "H"
        rows.append(base_strong_home)

        base_strong_away = {c: rng.normal(0, 1) for c in FEATURE_COLUMNS}
        base_strong_away["elo_diff"] = -300 + rng.normal(0, 20)
        base_strong_away["FTR"] = "A"
        rows.append(base_strong_away)

        base_even = {c: rng.normal(0, 1) for c in FEATURE_COLUMNS}
        base_even["elo_diff"] = rng.normal(0, 20)
        base_even["FTR"] = rng.choice(["H", "D", "A"])
        rows.append(base_even)
    return pd.DataFrame(rows)


class TestPrepareX:
    def test_fills_missing_values_with_zero(self):
        df = pd.DataFrame({c: [1.0] for c in FEATURE_COLUMNS})
        df.loc[0, "ppg_diff_5"] = np.nan
        X = _prepare_X(df)
        assert X["ppg_diff_5"].iloc[0] == 0.0

    def test_selects_only_declared_feature_columns(self):
        df = pd.DataFrame({c: [1.0] for c in FEATURE_COLUMNS})
        df["FTHG"] = [3]  # a result column that must never leak into X
        df["FTR"] = ["H"]
        X = _prepare_X(df)
        assert list(X.columns) == FEATURE_COLUMNS
        assert "FTHG" not in X.columns
        assert "FTR" not in X.columns


class TestLogisticOutcomeModel:
    def test_predicted_probabilities_are_valid(self):
        model = LogisticOutcomeModel().fit(_synthetic_training_data())
        test = _synthetic_training_data(n_per_bucket=5)
        proba = model.predict_proba(test)
        assert list(proba.columns) == CLASS_ORDER
        assert (proba.sum(axis=1) - 1.0).abs().max() < 1e-6
        assert (proba.to_numpy() >= 0).all()

    def test_handles_nan_features_without_crashing(self):
        train = _synthetic_training_data()
        model = LogisticOutcomeModel().fit(train)
        test = pd.DataFrame({c: [np.nan] for c in FEATURE_COLUMNS})
        proba = model.predict_proba(test)
        assert (proba.sum(axis=1) - 1.0).abs().max() < 1e-6

    def test_large_positive_elo_diff_favors_home_win(self):
        model = LogisticOutcomeModel().fit(_synthetic_training_data())
        test = pd.DataFrame({c: [0.0] for c in FEATURE_COLUMNS})
        test["elo_diff"] = 400
        proba = model.predict_proba(test)
        assert proba["H"].iloc[0] > proba["A"].iloc[0]

    def test_raises_if_predict_called_before_fit(self):
        with pytest.raises(RuntimeError):
            LogisticOutcomeModel().predict_proba(pd.DataFrame({c: [0.0] for c in FEATURE_COLUMNS}))

    def test_raises_if_coefficients_called_before_fit(self):
        with pytest.raises(RuntimeError):
            LogisticOutcomeModel().coefficients()

    def test_coefficients_shape(self):
        model = LogisticOutcomeModel().fit(_synthetic_training_data())
        coefs = model.coefficients()
        assert list(coefs.columns) == FEATURE_COLUMNS
        assert set(coefs.index) == set(CLASS_ORDER)
