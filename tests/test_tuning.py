"""Unit tests for src/evaluation/tuning.py.

The most important property to verify is leakage-safety at the
hyperparameter level: tuning must never see the final test block's
seasons, and the final test backtest must never re-touch the validation
block used for tuning.
"""

import pandas as pd
import pytest

from src.evaluation.tuning import (
    final_test_backtest,
    season_blocks,
    tune_elo_features,
    tune_model,
)
from src.models.baseline import FrequencyBaseline


def _synthetic_features(n_seasons: int = 12, matches_per_season: int = 20) -> pd.DataFrame:
    rows = []
    for s in range(n_seasons):
        season = f"20{10+s:02d}-{11+s:02d}"
        for m in range(matches_per_season):
            outcome = ["H", "D", "A"][m % 3]
            rows.append({"Season": season, "FTR": outcome})
    return pd.DataFrame(rows)


class TestSeasonBlocks:
    def test_splits_into_three_contiguous_blocks(self):
        features = _synthetic_features(n_seasons=12)
        initial, validation, test = season_blocks(features, initial_train_seasons=5, n_validation_seasons=3)
        assert len(initial) == 5
        assert len(validation) == 3
        assert len(test) == 4  # 12 - 5 - 3
        all_seasons = sorted(features["Season"].unique())
        assert initial + validation + test == all_seasons

    def test_blocks_are_non_overlapping(self):
        features = _synthetic_features(n_seasons=12)
        initial, validation, test = season_blocks(features, initial_train_seasons=5, n_validation_seasons=3)
        assert set(initial) & set(validation) == set()
        assert set(validation) & set(test) == set()
        assert set(initial) & set(test) == set()


class TestBestParamsDtype:
    """Regression test: extracting a single row from a DataFrame whose
    columns have mixed dtypes (e.g. int n_estimators alongside float
    learning_rate) via .iloc[0] silently upcasts every value in that row
    to a common dtype, usually float64. Harmless for Elo/Poisson/logistic
    regression's float-only hyperparameters, but XGBoost's n_estimators/
    max_depth strictly require a Python int and crash deep inside its
    training loop on numpy.float64(400.0) - a real failure only found by
    running tuning against XGBoost's actual grid, not a synthetic
    all-float test case."""

    def test_int_hyperparameter_survives_as_a_python_int(self):
        features = _synthetic_features(n_seasons=12)

        class RecordingModel:
            def fit(self, train):
                self._freq = FrequencyBaseline().fit(train)
                return self

            def predict_proba(self, test):
                return self._freq.predict_proba(test)

        candidates = [
            {"n_estimators": 100, "learning_rate": 0.1},   # int alongside float, like XGBoost's real grid
            {"n_estimators": 400, "learning_rate": 0.01},
        ]
        best, _ = tune_model(
            features, candidates, model_factory_builder=lambda params: RecordingModel,
            initial_train_seasons=5, n_validation_seasons=3,
        )
        assert isinstance(best["n_estimators"], int)
        assert not isinstance(best["n_estimators"], bool)  # bool is an int subclass in Python - guard explicitly


class TestTuneModel:
    def test_tuning_never_sees_final_test_block_seasons(self):
        features = _synthetic_features(n_seasons=12)
        _, _, test_seasons = season_blocks(features, initial_train_seasons=5, n_validation_seasons=3)

        seen_seasons = set()

        class RecordingModel:
            def fit(self, train):
                seen_seasons.update(train["Season"].unique())
                self._freq = FrequencyBaseline().fit(train)
                return self

            def predict_proba(self, test):
                seen_seasons.update(test["Season"].unique())
                return self._freq.predict_proba(test)

        tune_model(
            features,
            candidates=[{"dummy": 1}],
            model_factory_builder=lambda params: RecordingModel,
            initial_train_seasons=5,
            n_validation_seasons=3,
        )
        assert seen_seasons.isdisjoint(test_seasons)

    def test_picks_the_lower_log_loss_candidate(self):
        features = _synthetic_features(n_seasons=12)

        class GoodModel:
            """Predicts the true frequency - should win on log loss."""
            def fit(self, train):
                self._freq = FrequencyBaseline().fit(train)
                return self

            def predict_proba(self, test):
                return self._freq.predict_proba(test)

        class BadModel:
            """Deliberately overconfident in the wrong direction."""
            def fit(self, train):
                return self

            def predict_proba(self, test):
                return pd.DataFrame(
                    [{"A": 0.98, "D": 0.01, "H": 0.01}] * len(test), index=test.index
                )

        def factory_builder(params):
            return GoodModel if params["variant"] == "good" else BadModel

        best, results = tune_model(
            features,
            candidates=[{"variant": "good"}, {"variant": "bad"}],
            model_factory_builder=factory_builder,
            initial_train_seasons=5,
            n_validation_seasons=3,
        )
        assert best["variant"] == "good"
        assert len(results) == 2


class TestFinalTestBacktest:
    def test_only_scores_seasons_after_validation_block(self):
        features = _synthetic_features(n_seasons=12)
        _, _, expected_test_seasons = season_blocks(features, initial_train_seasons=5, n_validation_seasons=3)

        result = final_test_backtest(
            features, FrequencyBaseline, initial_train_seasons=5, n_validation_seasons=3
        )
        assert list(result["season"]) == expected_test_seasons


class TestTuneEloFeatures:
    def _matches(self, n_seasons: int = 12, matches_per_season: int = 12) -> pd.DataFrame:
        # Randomized (seeded) goals so H/D/A all appear within any given
        # training window - a small fixed cyclic pattern can accidentally
        # produce zero draws in a short window, which sklearn's
        # LogisticRegression (and EloOutcomeModel's own class-order
        # assertion) correctly refuses to treat as a real 3-class problem.
        import random
        gen = random.Random(0)
        rows = []
        base_date = pd.Timestamp("2010-08-01")
        teams = ["Alpha", "Bravo", "Charlie", "Delta"]
        for s in range(n_seasons):
            season = f"20{10+s:02d}-{11+s:02d}"
            for m in range(matches_per_season):
                home, away = teams[m % 4], teams[(m + 1) % 4]
                fthg, ftag = gen.randint(0, 3), gen.randint(0, 3)
                ftr = "H" if fthg > ftag else ("A" if fthg < ftag else "D")
                rows.append({
                    "Date": base_date + pd.Timedelta(days=365 * s + m * 7),
                    "HomeTeam": home, "AwayTeam": away,
                    "FTHG": fthg, "FTAG": ftag, "FTR": ftr,
                    "Season": season,
                })
        return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)

    def test_regenerates_features_per_candidate_and_picks_one(self):
        matches = self._matches()
        candidates = [
            {"initial_rating": 1500.0, "k_factor": 10.0, "home_advantage": 50.0,
             "use_goal_difference_multiplier": True, "promoted_team_penalty": 150.0, "season_shrinkage": 0.33},
            {"initial_rating": 1500.0, "k_factor": 30.0, "home_advantage": 100.0,
             "use_goal_difference_multiplier": True, "promoted_team_penalty": 150.0, "season_shrinkage": 0.33},
        ]

        from src.models.elo_model import EloOutcomeModel

        best, results = tune_elo_features(
            matches, candidates, model_factory_builder=lambda params: EloOutcomeModel,
            initial_train_seasons=5, n_validation_seasons=3,
        )
        assert best["k_factor"] in (10.0, 30.0)
        assert len(results) == 2
        assert "log_loss" in results.columns
