"""Unit tests for src/features/elo.py.

Covers the pure rating-math functions in isolation, then the stateful
EloRatingSystem's sequencing/leakage-safety behavior, per the project's
Step 18 testing requirements: initial ratings, rating update, home
advantage, draw handling.
"""

import pandas as pd
import pytest

from src.features.elo import (
    EloRatingSystem,
    compute_elo_features,
    expected_score,
    goal_difference_multiplier,
    match_result_score,
    update_ratings,
)


class TestExpectedScore:
    def test_equal_ratings_no_home_advantage_is_fifty_fifty(self):
        assert expected_score(1500, 1500, home_advantage=0) == pytest.approx(0.5)

    def test_home_advantage_favors_home_side(self):
        assert expected_score(1500, 1500, home_advantage=100) > 0.5

    def test_higher_rating_is_favored(self):
        assert expected_score(1600, 1500, home_advantage=0) > 0.5

    def test_symmetric_without_home_advantage(self):
        e_a = expected_score(1600, 1400, home_advantage=0)
        e_b = expected_score(1400, 1600, home_advantage=0)
        assert e_a + e_b == pytest.approx(1.0)

    def test_result_bounded_between_zero_and_one(self):
        assert 0.0 < expected_score(3000, 100, home_advantage=0) < 1.0
        assert 0.0 < expected_score(100, 3000, home_advantage=0) < 1.0


class TestMatchResultScore:
    def test_home_win(self):
        assert match_result_score(2, 0) == 1.0

    def test_away_win(self):
        assert match_result_score(0, 2) == 0.0

    def test_draw(self):
        assert match_result_score(1, 1) == 0.5


class TestGoalDifferenceMultiplier:
    def test_one_goal_margin_is_baseline(self):
        assert goal_difference_multiplier(0) == 1.0
        assert goal_difference_multiplier(1) == 1.0
        assert goal_difference_multiplier(-1) == 1.0

    def test_two_goal_margin(self):
        assert goal_difference_multiplier(2) == 1.5

    def test_larger_margins_increase_but_sublinearly(self):
        m3 = goal_difference_multiplier(3)
        m6 = goal_difference_multiplier(6)
        assert m3 > 1.5
        assert m6 > m3
        # doubling the margin (3 -> 6) must not double the multiplier -
        # extra goals beyond a comfortable win are weaker signal, not
        # proportionally stronger.
        assert m6 < 2 * m3

    def test_symmetric_in_sign(self):
        assert goal_difference_multiplier(3) == goal_difference_multiplier(-3)


class TestUpdateRatings:
    def test_home_win_increases_home_rating(self):
        new_home, new_away = update_ratings(1500, 1500, 2, 0, k_factor=20, home_advantage=0)
        assert new_home > 1500
        assert new_away < 1500

    def test_away_win_decreases_home_rating(self):
        new_home, new_away = update_ratings(1500, 1500, 0, 2, k_factor=20, home_advantage=0)
        assert new_home < 1500
        assert new_away > 1500

    def test_draw_between_equal_teams_leaves_ratings_unchanged(self):
        new_home, new_away = update_ratings(1500, 1500, 1, 1, k_factor=20, home_advantage=0)
        assert new_home == pytest.approx(1500)
        assert new_away == pytest.approx(1500)

    def test_underdog_draw_gains_rating(self):
        # A draw against a much stronger team beats expectations for the
        # underdog, so their rating should rise even though the result
        # wasn't a win.
        new_home, new_away = update_ratings(1400, 1700, 1, 1, k_factor=20, home_advantage=0)
        assert new_home > 1400
        assert new_away < 1700

    def test_zero_sum(self):
        new_home, new_away = update_ratings(1500, 1450, 3, 1, k_factor=20, home_advantage=50)
        assert (new_home - 1500) == pytest.approx(-(new_away - 1450))

    def test_bigger_margin_moves_rating_further(self):
        _, away_small_margin = update_ratings(1500, 1500, 1, 0, k_factor=20, home_advantage=0)
        _, away_big_margin = update_ratings(1500, 1500, 4, 0, k_factor=20, home_advantage=0)
        assert away_big_margin < away_small_margin  # bigger loss, rating drops further

    def test_goal_difference_multiplier_can_be_disabled(self):
        _, away_1 = update_ratings(1500, 1500, 1, 0, k_factor=20, home_advantage=0,
                                     use_goal_difference_multiplier=False)
        _, away_4 = update_ratings(1500, 1500, 4, 0, k_factor=20, home_advantage=0,
                                     use_goal_difference_multiplier=False)
        assert away_1 == pytest.approx(away_4)  # margin shouldn't matter when disabled


class TestEloRatingSystem:
    def test_first_team_ever_gets_initial_rating(self):
        elo = EloRatingSystem(initial_rating=1500)
        r_home, r_away = elo.process_match("Arsenal", "Chelsea", 1, 1, season="2023-24")
        assert r_home == 1500
        assert r_away == 1500

    def test_pre_match_rating_excludes_current_match_result(self):
        # The rating returned for a match must be what the team's rating
        # WAS before this match, not after - otherwise the "prediction"
        # for a match would already encode its own result. This is the
        # core leakage-safety property of the whole module.
        elo = EloRatingSystem(initial_rating=1500)
        r_home_1, r_away_1 = elo.process_match("Arsenal", "Chelsea", 3, 0, season="2023-24")
        assert r_home_1 == 1500  # unaffected by the 3-0 result about to happen
        r_home_2, _ = elo.process_match("Arsenal", "Chelsea", 0, 0, season="2023-24")
        assert r_home_2 > 1500  # now reflects the earlier win

    def test_newly_promoted_team_starts_below_league_mean(self):
        elo = EloRatingSystem(initial_rating=1500, promoted_team_penalty=150)
        elo.process_match("Arsenal", "Chelsea", 1, 1, season="2023-24")  # both at 1500
        r_new, _ = elo.process_match("Luton", "Arsenal", 0, 0, season="2023-24")
        assert r_new == pytest.approx(1500 - 150)

    def test_season_shrinkage_regresses_toward_mean(self):
        elo = EloRatingSystem(initial_rating=1500, season_shrinkage=0.5, k_factor=20)
        # Arsenal wins big to move well above 1500, Chelsea correspondingly below.
        elo.process_match("Arsenal", "Chelsea", 5, 0, season="2023-24")
        pre_shrink_arsenal = elo.ratings["Arsenal"]
        assert pre_shrink_arsenal > 1500

        # Crossing into a new season should shrink Arsenal back toward the mean.
        r_arsenal_new_season, _ = elo.process_match("Arsenal", "Chelsea", 1, 1, season="2024-25")
        assert 1500 < r_arsenal_new_season < pre_shrink_arsenal

    def test_current_ratings_snapshot(self):
        elo = EloRatingSystem()
        elo.process_match("Arsenal", "Chelsea", 2, 1, season="2023-24")
        snapshot = elo.current_ratings()
        assert set(snapshot.keys()) == {"Arsenal", "Chelsea"}
        assert snapshot["Arsenal"] > snapshot["Chelsea"]


class TestComputeEloFeatures:
    def _toy_matches(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Date": pd.to_datetime(["2023-08-12", "2023-08-19", "2023-08-26"]),
            "HomeTeam": ["Arsenal", "Chelsea", "Arsenal"],
            "AwayTeam": ["Chelsea", "Arsenal", "Chelsea"],
            "FTHG": [2, 1, 0],
            "FTAG": [0, 1, 0],
            "Season": ["2023-24", "2023-24", "2023-24"],
        })

    def test_adds_expected_columns(self):
        result = compute_elo_features(self._toy_matches())
        assert {"elo_home_pre", "elo_away_pre", "elo_diff"} <= set(result.columns)

    def test_elo_diff_is_consistent(self):
        result = compute_elo_features(self._toy_matches())
        expected_diff = result["elo_home_pre"] - result["elo_away_pre"]
        pd.testing.assert_series_equal(result["elo_diff"], expected_diff, check_names=False)

    def test_first_match_uses_initial_rating(self):
        result = compute_elo_features(self._toy_matches(), config={"initial_rating": 1500})
        assert result.iloc[0]["elo_home_pre"] == 1500
        assert result.iloc[0]["elo_away_pre"] == 1500

    def test_later_match_reflects_earlier_result(self):
        result = compute_elo_features(self._toy_matches())
        # Arsenal won match 1 (2-0), so their rating going into match 3
        # (as home team again) should be above the initial rating.
        assert result.iloc[2]["elo_home_pre"] > 1500

    def test_does_not_mutate_input(self):
        original = self._toy_matches()
        original_columns = list(original.columns)
        compute_elo_features(original)
        assert list(original.columns) == original_columns

    def test_out_of_order_input_raises(self):
        df = self._toy_matches().iloc[::-1].reset_index(drop=True)
        with pytest.raises(ValueError, match="chronologically"):
            compute_elo_features(df)
