"""Unit tests for src/features/rolling.py, focused on the leakage-safety
properties: rolling stats must reflect only strictly prior matches, and
each side of a fixture must receive its OWN history, never the opponent's."""

import numpy as np
import pandas as pd
import pytest

from src.features.rolling import attach_rolling_features, compute_rolling_features


def _matches(rows: list[tuple]) -> pd.DataFrame:
    """rows: (date, home, away, fthg, ftag)."""
    df = pd.DataFrame(rows, columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df["Date"] = pd.to_datetime(df["Date"])
    df["Season"] = "2023-24"
    df["match_id"] = range(len(df))
    return df


class TestComputeRollingFeatures:
    def test_first_match_of_a_team_has_nan_rolling_features(self):
        matches = _matches([
            ("2023-08-12", "Arsenal", "Chelsea", 2, 0),
        ])
        long = compute_rolling_features(matches, windows=(3,))
        assert long["rolling_gf_3"].isna().all()
        assert long["rolling_ga_3"].isna().all()
        assert long["rolling_ppg_3"].isna().all()

    def test_rolling_mean_excludes_current_match(self):
        # Arsenal: match1 scores 2, match2 scores 4. Going into match2,
        # rolling_gf must be 2 (only match1) - NOT (2+4)/2=3, which would
        # mean match2 leaked its own result into its own feature.
        matches = _matches([
            ("2023-08-12", "Arsenal", "Chelsea", 2, 0),
            ("2023-08-19", "Arsenal", "Everton", 4, 1),
        ])
        long = compute_rolling_features(matches, windows=(3,))
        arsenal = long[long["team"] == "Arsenal"].sort_values("Date")
        assert pd.isna(arsenal["rolling_gf_3"].iloc[0])
        assert arsenal["rolling_gf_3"].iloc[1] == pytest.approx(2.0)

    def test_rolling_window_caps_at_window_size(self):
        # 4 prior matches with goals_for = [1,2,3,4]; a window of 3 going
        # into the 5th match must average only the most recent 3 (2,3,4),
        # not all 4.
        rows = [
            ("2023-08-05", "Arsenal", "X1", 1, 0),
            ("2023-08-12", "Arsenal", "X2", 2, 0),
            ("2023-08-19", "Arsenal", "X3", 3, 0),
            ("2023-08-26", "Arsenal", "X4", 4, 0),
            ("2023-09-02", "Arsenal", "X5", 0, 0),
        ]
        long = compute_rolling_features(_matches(rows), windows=(3,))
        arsenal = long[long["team"] == "Arsenal"].sort_values("Date").reset_index(drop=True)
        assert arsenal["rolling_gf_3"].iloc[4] == pytest.approx((2 + 3 + 4) / 3)

    def test_rest_days_computed_from_gap_between_matches(self):
        rows = [
            ("2023-08-12", "Arsenal", "Chelsea", 1, 0),
            ("2023-08-19", "Arsenal", "Everton", 1, 0),
        ]
        long = compute_rolling_features(_matches(rows), windows=(3,))
        arsenal = long[long["team"] == "Arsenal"].sort_values("Date").reset_index(drop=True)
        assert np.isnan(arsenal["rest_days"].iloc[0])
        assert arsenal["rest_days"].iloc[1] == 7


class TestAttachRollingFeatures:
    def test_home_and_away_features_come_from_the_correct_team(self):
        rows = [
            ("2023-08-05", "Arsenal", "Watford", 3, 0),   # Arsenal builds form
            ("2023-08-12", "Chelsea", "Fulham", 0, 0),     # Chelsea builds form
            ("2023-08-19", "Arsenal", "Chelsea", 1, 1),    # the match under test
        ]
        result = attach_rolling_features(_matches(rows), windows=(3,))
        target = result.iloc[2]
        # Arsenal (home in the target match) scored 3 in their only prior match.
        assert target["home_rolling_gf_3"] == pytest.approx(3.0)
        # Chelsea (away) scored 0 in their only prior match.
        assert target["away_rolling_gf_3"] == pytest.approx(0.0)

    def test_relative_features_are_internally_consistent(self):
        rows = [
            ("2023-08-05", "Arsenal", "Watford", 3, 0),
            ("2023-08-12", "Chelsea", "Fulham", 1, 1),
            ("2023-08-19", "Arsenal", "Chelsea", 1, 1),
        ]
        result = attach_rolling_features(_matches(rows), windows=(3,))
        target = result.iloc[2]
        expected = target["home_rolling_gf_3"] - target["away_rolling_ga_3"]
        assert target["attack_vs_defence_3"] == pytest.approx(expected)

    def test_does_not_mutate_input(self):
        rows = [("2023-08-12", "Arsenal", "Chelsea", 1, 0)]
        original = _matches(rows)
        cols_before = list(original.columns)
        attach_rolling_features(original, windows=(3,))
        assert list(original.columns) == cols_before
