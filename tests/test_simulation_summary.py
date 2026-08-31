"""Unit tests for src/simulation/summary.py."""

import pandas as pd
import pytest

from src.simulation.summary import (
    DEFAULT_COMPETITION_CONFIG,
    expected_position,
    position_distribution,
    relegation_probability,
    summarize_simulation,
    title_probability,
    top_n_probability,
)


def _toy_positions() -> pd.DataFrame:
    # 4 simulations, 3 teams. Alpha always wins the title; Charlie always
    # finishes bottom; Bravo is always 2nd.
    return pd.DataFrame({
        "Alpha": [1, 1, 1, 1],
        "Bravo": [2, 2, 2, 2],
        "Charlie": [3, 3, 3, 3],
    })


def _mixed_positions() -> pd.DataFrame:
    # Alpha wins the title in 3 of 4 sims, finishes 2nd once.
    return pd.DataFrame({
        "Alpha": [1, 1, 1, 2],
        "Bravo": [2, 2, 2, 1],
        "Charlie": [3, 3, 3, 3],
    })


class TestTitleProbability:
    def test_certain_winner_gets_probability_one(self):
        result = title_probability(_toy_positions())
        assert result["Alpha"] == 1.0
        assert result["Bravo"] == 0.0

    def test_partial_winner_gets_fractional_probability(self):
        result = title_probability(_mixed_positions())
        assert result["Alpha"] == pytest.approx(0.75)
        assert result["Bravo"] == pytest.approx(0.25)


class TestTopNProbability:
    def test_top_two_includes_both_leaders(self):
        result = top_n_probability(_toy_positions(), n=2)
        assert result["Alpha"] == 1.0
        assert result["Bravo"] == 1.0
        assert result["Charlie"] == 0.0


class TestRelegationProbability:
    def test_bottom_team_always_relegated(self):
        result = relegation_probability(_toy_positions(), n_relegated=1)
        assert result["Charlie"] == 1.0
        assert result["Alpha"] == 0.0
        assert result["Bravo"] == 0.0


class TestExpectedPosition:
    def test_matches_mean_of_positions(self):
        result = expected_position(_mixed_positions())
        assert result["Alpha"] == pytest.approx((1 + 1 + 1 + 2) / 4)


class TestPositionDistribution:
    def test_rows_sum_to_one(self):
        dist = position_distribution(_mixed_positions())
        assert (dist.sum(axis=1) - 1.0).abs().max() < 1e-9

    def test_matches_known_values(self):
        dist = position_distribution(_mixed_positions())
        assert dist.loc["Alpha", 1] == pytest.approx(0.75)
        assert dist.loc["Alpha", 2] == pytest.approx(0.25)
        assert dist.loc["Alpha", 3] == pytest.approx(0.0)


class TestSummarizeSimulation:
    def _points(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Alpha": [90, 88, 91, 85],
            "Bravo": [80, 82, 79, 81],
            "Charlie": [30, 28, 25, 32],
        })

    def test_returns_one_row_per_team_sorted_by_expected_position(self):
        summary = summarize_simulation(_mixed_positions(), self._points())
        assert list(summary.index) == ["Alpha", "Bravo", "Charlie"]  # best expected position first

    def test_expected_columns_present(self):
        summary = summarize_simulation(_mixed_positions(), self._points())
        expected_cols = {
            "title_probability", "champions_league_probability",
            "european_qualification_probability", "relegation_probability",
            "expected_position", "expected_points", "median_points",
            "points_p05", "points_p95",
        }
        assert expected_cols <= set(summary.columns)

    def test_custom_competition_config_changes_cutoffs(self):
        small_league_config = {
            "champions_league": 1, "europa_league": 0, "conference_league": 0, "relegation": 1,
        }
        summary = summarize_simulation(_toy_positions(), pd.DataFrame({
            "Alpha": [90] * 4, "Bravo": [80] * 4, "Charlie": [30] * 4,
        }), competition_config=small_league_config)
        assert summary.loc["Alpha", "champions_league_probability"] == 1.0
        assert summary.loc["Bravo", "champions_league_probability"] == 0.0
        assert summary.loc["Charlie", "relegation_probability"] == 1.0

    def test_default_config_has_expected_totals(self):
        assert DEFAULT_COMPETITION_CONFIG["champions_league"] == 4
        assert DEFAULT_COMPETITION_CONFIG["relegation"] == 3
