"""Unit tests for src/models/baseline.py."""

import pandas as pd
import pytest

from src.models.baseline import FrequencyBaseline, always_predict_home_win_accuracy


class TestFrequencyBaseline:
    def test_learns_training_frequencies(self):
        train = pd.DataFrame({"FTR": ["H", "H", "H", "D", "A"]})
        model = FrequencyBaseline().fit(train)
        assert model.frequencies_["H"] == pytest.approx(0.6)
        assert model.frequencies_["D"] == pytest.approx(0.2)
        assert model.frequencies_["A"] == pytest.approx(0.2)

    def test_predicts_same_distribution_for_every_row(self):
        train = pd.DataFrame({"FTR": ["H", "D", "A"]})
        model = FrequencyBaseline().fit(train)
        test = pd.DataFrame({"HomeTeam": ["Arsenal", "Chelsea"]})
        proba = model.predict_proba(test)
        assert len(proba) == 2
        assert (proba.iloc[0] == proba.iloc[1]).all()

    def test_predicted_probabilities_sum_to_one(self):
        train = pd.DataFrame({"FTR": ["H", "H", "D", "A"]})
        model = FrequencyBaseline().fit(train)
        proba = model.predict_proba(pd.DataFrame({"x": [1, 2, 3]}))
        assert (proba.sum(axis=1) - 1.0).abs().max() < 1e-9

    def test_missing_class_in_training_gets_zero_probability(self):
        train = pd.DataFrame({"FTR": ["H", "H", "H"]})  # no draws or away wins at all
        model = FrequencyBaseline().fit(train)
        assert model.frequencies_["D"] == 0.0
        assert model.frequencies_["A"] == 0.0

    def test_raises_if_predict_called_before_fit(self):
        with pytest.raises(RuntimeError):
            FrequencyBaseline().predict_proba(pd.DataFrame({"x": [1]}))


class TestAlwaysPredictHomeWinAccuracy:
    def test_computes_fraction_of_home_wins(self):
        y = pd.Series(["H", "H", "D", "A"])
        assert always_predict_home_win_accuracy(y) == pytest.approx(0.5)
