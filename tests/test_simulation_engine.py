"""Unit tests for src/simulation/engine.py.

Covers the project's explicit simulation testing requirements: points
assigned correctly (win=3, draw=1, loss=0), every simulation produces a
valid table, and the simulator's internal bookkeeping is self-consistent
(no fixture silently double-counted or dropped).
"""

import numpy as np
import pandas as pd
import pytest

from src.simulation.engine import (
    CurrentTableState,
    Fixture,
    scorelines_to_points,
    simulate_final_tables,
    simulate_scorelines,
)


class TestScorelinesToPoints:
    def test_home_win_awards_three_and_zero(self):
        home_pts, away_pts = scorelines_to_points(np.array([2]), np.array([0]))
        assert home_pts[0] == 3
        assert away_pts[0] == 0

    def test_away_win_awards_zero_and_three(self):
        home_pts, away_pts = scorelines_to_points(np.array([0]), np.array([2]))
        assert home_pts[0] == 0
        assert away_pts[0] == 3

    def test_draw_awards_one_and_one(self):
        home_pts, away_pts = scorelines_to_points(np.array([1]), np.array([1]))
        assert home_pts[0] == 1
        assert away_pts[0] == 1

    def test_scoreless_draw_awards_one_and_one(self):
        home_pts, away_pts = scorelines_to_points(np.array([0]), np.array([0]))
        assert home_pts[0] == 1
        assert away_pts[0] == 1

    def test_vectorized_over_many_matches(self):
        home_goals = np.array([2, 0, 1, 3])
        away_goals = np.array([0, 2, 1, 3])
        home_pts, away_pts = scorelines_to_points(home_goals, away_goals)
        assert list(home_pts) == [3, 0, 1, 1]
        assert list(away_pts) == [0, 3, 1, 1]

    def test_every_match_distributes_two_or_three_points_total(self):
        # Each team gets 3/1/0 points for a win/draw/loss (standard
        # football scoring - a draw awards 1 point to EACH side, never 2
        # to either one). Summed across BOTH sides of one match, that
        # means the combined total is 3+0=3 for a decisive result or
        # 1+1=2 for a draw - never anything else. This combined-total
        # invariant is what the season-level conservation test below
        # relies on; it is not a claim about either team's individual
        # points award.
        rng = np.random.default_rng(0)
        home_goals = rng.integers(0, 6, size=1000)
        away_goals = rng.integers(0, 6, size=1000)
        home_pts, away_pts = scorelines_to_points(home_goals, away_goals)
        totals = home_pts + away_pts
        assert set(np.unique(totals)) <= {2, 3}


class TestSimulateScorelines:
    def test_output_shape(self):
        fixtures = [Fixture("A", "B", 1.5, 1.2), Fixture("B", "C", 1.0, 1.0)]
        home_goals, away_goals = simulate_scorelines(fixtures, n_simulations=100)
        assert home_goals.shape == (100, 2)
        assert away_goals.shape == (100, 2)

    def test_goals_are_non_negative(self):
        fixtures = [Fixture("A", "B", 1.5, 1.2)]
        home_goals, away_goals = simulate_scorelines(fixtures, n_simulations=500)
        assert (home_goals >= 0).all()
        assert (away_goals >= 0).all()

    def test_higher_lambda_produces_higher_mean_goals(self):
        fixtures = [Fixture("A", "B", 3.0, 0.3)]
        home_goals, away_goals = simulate_scorelines(fixtures, n_simulations=20000, rng=np.random.default_rng(0))
        assert home_goals.mean() > away_goals.mean()
        assert home_goals.mean() == pytest.approx(3.0, abs=0.15)
        assert away_goals.mean() == pytest.approx(0.3, abs=0.05)


class TestSimulateFinalTables:
    def _current(self) -> CurrentTableState:
        return CurrentTableState(
            teams=["Alpha", "Bravo", "Charlie", "Delta"],
            points={"Alpha": 20, "Bravo": 15, "Charlie": 10, "Delta": 5},
            goals_for={"Alpha": 25, "Bravo": 20, "Charlie": 15, "Delta": 10},
            goals_against={"Alpha": 10, "Bravo": 15, "Charlie": 20, "Delta": 25},
        )

    def _fixtures(self) -> list[Fixture]:
        return [
            Fixture("Alpha", "Delta", 2.0, 0.8),
            Fixture("Bravo", "Charlie", 1.5, 1.2),
            Fixture("Charlie", "Alpha", 0.8, 2.2),
            Fixture("Delta", "Bravo", 1.0, 1.6),
        ]

    def test_positions_are_a_valid_permutation_every_simulation(self):
        positions, _ = simulate_final_tables(self._current(), self._fixtures(), n_simulations=200, seed=0)
        n_teams = positions.shape[1]
        for _, row in positions.iterrows():
            assert sorted(row.tolist()) == list(range(1, n_teams + 1))

    def test_points_conservation_invariant(self):
        # Every match's two teams combine for either 3 points (decisive
        # result: 3+0) or 2 (a draw: 1 to each side) - never any other
        # combined amount, and never lost or double-counted - so total
        # points across all teams
        # in any single simulation must fall between current_total +
        # 2*n_fixtures (every remaining match a draw) and current_total +
        # 3*n_fixtures (every remaining match decisive), for every single
        # simulation, regardless of which scorelines were actually
        # sampled. This is a strong, deterministic correctness check that
        # would catch a fixture being silently dropped or double-applied.
        current = self._current()
        fixtures = self._fixtures()
        _, points = simulate_final_tables(current, fixtures, n_simulations=300, seed=1)

        current_total = sum(current.points.values())
        lower_bound = current_total + 2 * len(fixtures)
        upper_bound = current_total + 3 * len(fixtures)
        row_totals = points.sum(axis=1)
        assert (row_totals >= lower_bound).all()
        assert (row_totals <= upper_bound).all()

    def test_empty_fixtures_reproduces_current_table_exactly(self):
        current = self._current()
        positions, points = simulate_final_tables(current, [], n_simulations=50, seed=0)

        expected_points_row = [current.points[t] for t in current.teams]
        assert (points == expected_points_row).all().all()
        # Alpha (20 pts) should rank 1st, Delta (5 pts) last, every sim.
        assert (positions["Alpha"] == 1).all()
        assert (positions["Delta"] == 4).all()

    def test_reproducible_with_same_seed(self):
        current, fixtures = self._current(), self._fixtures()
        positions_1, points_1 = simulate_final_tables(current, fixtures, n_simulations=100, seed=42)
        positions_2, points_2 = simulate_final_tables(current, fixtures, n_simulations=100, seed=42)
        pd.testing.assert_frame_equal(positions_1, positions_2)
        pd.testing.assert_frame_equal(points_1, points_2)

    def test_team_leading_on_points_ranks_better_on_average(self):
        # Alpha starts 15 points clear of Delta with only 4 fixtures left
        # (max swing 12 points) - Alpha's average finishing position must
        # be better (numerically lower) than Delta's.
        current, fixtures = self._current(), self._fixtures()
        positions, _ = simulate_final_tables(current, fixtures, n_simulations=2000, seed=0)
        assert positions["Alpha"].mean() < positions["Delta"].mean()

    def test_output_shape_matches_teams_and_simulations(self):
        current, fixtures = self._current(), self._fixtures()
        positions, points = simulate_final_tables(current, fixtures, n_simulations=77, seed=0)
        assert positions.shape == (77, 4)
        assert points.shape == (77, 4)
        assert list(positions.columns) == current.teams
