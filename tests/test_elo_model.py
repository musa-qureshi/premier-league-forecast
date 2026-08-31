"""Unit tests for src/models/elo_model.py."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import CLASS_ORDER
from src.models.elo_model import EloOutcomeModel


def _synthetic_training_data(n_per_bucket: int = 30) -> pd.DataFrame:
    """A clear, learnable relationship: large positive elo_diff -> mostly
    home wins, large negative -> mostly away wins, near zero -> a genuine
    mix including draws - enough signal for a 1-feature multinomial logit
    to recover a sensible decision boundary, and enough draws present that
    the fitted model doesn't collapse to a 2-class problem."""
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(n_per_bucket):
        rows.append({"elo_diff": 300 + rng.normal(0, 20), "FTR": "H"})
        rows.append({"elo_diff": -300 + rng.normal(0, 20), "FTR": "A"})
        rows.append({"elo_diff": rng.normal(0, 20), "FTR": rng.choice(["H", "D", "A"])})
    return pd.DataFrame(rows)


class TestEloOutcomeModel:
    def test_predicted_probabilities_are_valid(self):
        model = EloOutcomeModel().fit(_synthetic_training_data())
        test = pd.DataFrame({"elo_diff": [100, -100, 0]})
        proba = model.predict_proba(test)
        assert list(proba.columns) == CLASS_ORDER
        assert (proba.sum(axis=1) - 1.0).abs().max() < 1e-9
        assert (proba.to_numpy() >= 0).all()

    def test_large_positive_elo_diff_favors_home_win(self):
        model = EloOutcomeModel().fit(_synthetic_training_data())
        proba = model.predict_proba(pd.DataFrame({"elo_diff": [400]}))
        assert proba["H"].iloc[0] > proba["A"].iloc[0]
        assert proba["H"].iloc[0] > 0.5

    def test_large_negative_elo_diff_favors_away_win(self):
        model = EloOutcomeModel().fit(_synthetic_training_data())
        proba = model.predict_proba(pd.DataFrame({"elo_diff": [-400]}))
        assert proba["A"].iloc[0] > proba["H"].iloc[0]
        assert proba["A"].iloc[0] > 0.5

    def test_raises_if_predict_called_before_fit(self):
        with pytest.raises(RuntimeError):
            EloOutcomeModel().predict_proba(pd.DataFrame({"elo_diff": [0]}))
