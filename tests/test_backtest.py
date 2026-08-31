"""Unit tests for src/evaluation/backtest.py."""

import pandas as pd
import pytest

from src.evaluation.backtest import expanding_window_backtest, expanding_window_predictions, summarize
from src.models.baseline import FrequencyBaseline


def _synthetic_features(n_seasons: int = 5, matches_per_season: int = 10) -> pd.DataFrame:
    rows = []
    for s in range(n_seasons):
        season = f"20{10+s}-{11+s}"
        for m in range(matches_per_season):
            outcome = ["H", "D", "A"][m % 3]
            rows.append({"Season": season, "FTR": outcome})
    return pd.DataFrame(rows)


class TestExpandingWindowBacktest:
    def test_one_row_per_held_out_season(self):
        features = _synthetic_features(n_seasons=5)
        result = expanding_window_backtest(features, FrequencyBaseline, min_train_seasons=2)
        # 5 seasons total, first 2 used purely as initial training -> 3 held-out folds
        assert len(result) == 3

    def test_training_window_expands_each_fold(self):
        features = _synthetic_features(n_seasons=5)
        result = expanding_window_backtest(features, FrequencyBaseline, min_train_seasons=2)
        assert list(result["n_train_seasons"]) == [2, 3, 4]

    def test_held_out_seasons_are_the_later_ones_in_order(self):
        features = _synthetic_features(n_seasons=5)
        result = expanding_window_backtest(features, FrequencyBaseline, min_train_seasons=2)
        seasons = sorted(features["Season"].unique())
        assert list(result["season"]) == seasons[2:]

    def test_raises_when_not_enough_seasons(self):
        features = _synthetic_features(n_seasons=3)
        with pytest.raises(ValueError, match="Not enough seasons"):
            expanding_window_backtest(features, FrequencyBaseline, min_train_seasons=5)

    def test_metrics_columns_present(self):
        features = _synthetic_features(n_seasons=4)
        result = expanding_window_backtest(features, FrequencyBaseline, min_train_seasons=2)
        for col in ("log_loss", "brier_score", "rps", "accuracy", "n_matches"):
            assert col in result.columns


class TestExpandingWindowPredictions:
    def test_returns_one_row_per_held_out_match(self):
        features = _synthetic_features(n_seasons=5, matches_per_season=10)
        preds = expanding_window_predictions(features, FrequencyBaseline, min_train_seasons=2)
        # 3 held-out seasons x 10 matches each = 30 rows
        assert len(preds) == 30

    def test_contains_probability_columns_and_actual_outcome(self):
        features = _synthetic_features(n_seasons=4, matches_per_season=10)
        preds = expanding_window_predictions(features, FrequencyBaseline, min_train_seasons=2)
        assert {"A", "D", "H", "FTR", "Season"} <= set(preds.columns)
        assert (preds[["A", "D", "H"]].sum(axis=1) - 1.0).abs().max() < 1e-9

    def test_uses_same_fold_boundaries_as_backtest(self):
        features = _synthetic_features(n_seasons=5, matches_per_season=10)
        backtest = expanding_window_backtest(features, FrequencyBaseline, min_train_seasons=2)
        preds = expanding_window_predictions(features, FrequencyBaseline, min_train_seasons=2)
        assert set(preds["Season"].unique()) == set(backtest["season"].unique())


class TestSummarize:
    def test_weighted_average_matches_hand_computation(self):
        results = pd.DataFrame({
            "log_loss": [1.0, 2.0],
            "brier_score": [0.5, 0.7],
            "rps": [0.2, 0.3],
            "accuracy": [0.4, 0.6],
            "n_matches": [10, 30],  # second season weighted 3x more heavily
        })
        summary = summarize(results)
        # weighted mean of log_loss = (1.0*10 + 2.0*30) / 40 = 70/40 = 1.75
        assert summary["log_loss"] == pytest.approx(1.75)
        assert summary["n_seasons"] == 2
        assert summary["n_matches"] == 40
