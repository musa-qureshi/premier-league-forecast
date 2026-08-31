"""Unit tests for src/evaluation/metrics.py."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import accuracy, brier_score, evaluate_all, log_loss, rps


def _proba(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)[["A", "D", "H"]]


class TestProbabilityValidation:
    def test_rejects_rows_not_summing_to_one(self):
        y = pd.Series(["H"])
        p = _proba([{"A": 0.1, "D": 0.1, "H": 0.5}])  # sums to 0.7
        with pytest.raises(ValueError, match="sum to 1"):
            log_loss(y, p)

    def test_rejects_negative_probabilities(self):
        y = pd.Series(["H"])
        p = pd.DataFrame([{"A": -0.1, "D": 0.1, "H": 1.0}])
        with pytest.raises(ValueError, match="non-negative"):
            log_loss(y, p)

    def test_rejects_unknown_label(self):
        y = pd.Series(["X"])
        p = _proba([{"A": 0.3, "D": 0.3, "H": 0.4}])
        with pytest.raises(ValueError, match="outside"):
            log_loss(y, p)


class TestLogLoss:
    def test_perfect_confident_prediction_near_zero(self):
        y = pd.Series(["H", "A", "D"])
        p = _proba([
            {"A": 0.0, "D": 0.0, "H": 1.0},
            {"A": 1.0, "D": 0.0, "H": 0.0},
            {"A": 0.0, "D": 1.0, "H": 0.0},
        ])
        assert log_loss(y, p) == pytest.approx(0.0, abs=1e-9)

    def test_uniform_prediction_equals_log_of_num_classes(self):
        y = pd.Series(["H", "A", "D"])
        p = _proba([{"A": 1 / 3, "D": 1 / 3, "H": 1 / 3}] * 3)
        assert log_loss(y, p) == pytest.approx(np.log(3), abs=1e-6)

    def test_confidently_wrong_is_penalized_heavily(self):
        y = pd.Series(["A"])
        confident_wrong = _proba([{"A": 0.01, "D": 0.01, "H": 0.98}])
        unsure = _proba([{"A": 0.2, "D": 0.3, "H": 0.5}])
        assert log_loss(y, confident_wrong) > log_loss(y, unsure)


class TestBrierScore:
    def test_perfect_confident_prediction_is_zero(self):
        y = pd.Series(["H"])
        p = _proba([{"A": 0.0, "D": 0.0, "H": 1.0}])
        assert brier_score(y, p) == pytest.approx(0.0)

    def test_uniform_prediction_matches_known_value(self):
        # (1/3 - 1)^2 + (1/3)^2 + (1/3)^2 = 4/9 + 1/9 + 1/9 = 6/9
        y = pd.Series(["H"])
        p = _proba([{"A": 1 / 3, "D": 1 / 3, "H": 1 / 3}])
        assert brier_score(y, p) == pytest.approx(6 / 9)

    def test_bounded_between_zero_and_two(self):
        y = pd.Series(["A"])
        p = _proba([{"A": 0.0, "D": 0.0, "H": 1.0}])  # worst possible
        assert brier_score(y, p) == pytest.approx(2.0)


class TestRPS:
    def test_perfect_prediction_is_zero(self):
        y = pd.Series(["D"])
        p = _proba([{"A": 0.0, "D": 1.0, "H": 0.0}])
        assert rps(y, p) == pytest.approx(0.0)

    def test_ordinal_sensitivity_close_miss_penalized_less_than_far_miss(self):
        # Actual outcome is an Away win (an extreme on the A-D-H axis). A
        # model confidently predicting Draw (one step away) should be
        # penalized less by RPS than one confidently predicting Home (two
        # steps away, the opposite extreme) - this is exactly the ordinal
        # awareness plain Brier score lacks. (Note: this asymmetry only
        # shows up when the actual outcome is at an extreme - if the actual
        # outcome were the middle class (Draw), the two extreme wrong
        # predictions are symmetric in cumulative-probability space and
        # RPS scores them identically, same as Brier.)
        y = pd.Series(["A"])
        predicts_draw = _proba([{"A": 0.0, "D": 1.0, "H": 0.0}])
        predicts_home = _proba([{"A": 0.0, "D": 0.0, "H": 1.0}])
        assert rps(y, predicts_draw) < rps(y, predicts_home)

    def test_brier_score_is_blind_to_this_same_distinction(self):
        # Confirms the claim above by contrast: plain Brier score treats
        # "wrong" as wrong regardless of how far off the class is, since it
        # has no notion of class order - both misses cost the same.
        y = pd.Series(["A"])
        predicts_draw = _proba([{"A": 0.0, "D": 1.0, "H": 0.0}])
        predicts_home = _proba([{"A": 0.0, "D": 0.0, "H": 1.0}])
        assert brier_score(y, predicts_draw) == pytest.approx(brier_score(y, predicts_home))

    def test_bounded_between_zero_and_one(self):
        y = pd.Series(["A"])
        p = _proba([{"A": 0.0, "D": 0.0, "H": 1.0}])  # worst possible, opposite extreme
        assert rps(y, p) == pytest.approx(1.0)


class TestAccuracy:
    def test_counts_top_class_matches(self):
        y = pd.Series(["H", "D", "A"])
        p = _proba([
            {"A": 0.1, "D": 0.1, "H": 0.8},  # correct (H)
            {"A": 0.1, "D": 0.1, "H": 0.8},  # wrong (predicts H, actual D)
            {"A": 0.7, "D": 0.2, "H": 0.1},  # correct (A)
        ])
        assert accuracy(y, p) == pytest.approx(2 / 3)


class TestEvaluateAll:
    def test_returns_all_expected_keys(self):
        y = pd.Series(["H", "D"])
        p = _proba([{"A": 0.2, "D": 0.3, "H": 0.5}, {"A": 0.3, "D": 0.4, "H": 0.3}])
        result = evaluate_all(y, p)
        assert set(result.keys()) == {"log_loss", "brier_score", "rps", "accuracy", "n_matches"}
        assert result["n_matches"] == 2
