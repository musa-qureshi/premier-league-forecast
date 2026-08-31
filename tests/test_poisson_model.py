"""Unit tests for src/models/poisson_model.py.

The most important test here is TestParameterRecovery: it simulates match
data from KNOWN attack/defense/home-advantage parameters via an explicit
Poisson process, fits the model to that simulated data, and checks the
fitted model's predicted expected goals are close to the true values used
to generate it. Individual attack_i/defense_i parameters aren't directly
comparable to the ground truth (they're only identified up to the additive
gauge-shift explained in the module docstring), but the combination that
actually determines predictions - lambda_home/lambda_away for a given
matchup - is identified and is what this test checks.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.poisson_model import DixonColesModel, dixon_coles_tau, top_scorelines


class TestDixonColesTau:
    def test_rho_zero_gives_tau_one_everywhere(self):
        for x, y in [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)]:
            assert dixon_coles_tau(x, y, lambda_home=1.5, lambda_away=1.2, rho=0.0) == pytest.approx(1.0)

    def test_special_case_formulas(self):
        lh, la, rho = 1.5, 1.2, 0.1
        assert dixon_coles_tau(0, 0, lh, la, rho) == pytest.approx(1 - lh * la * rho)
        assert dixon_coles_tau(0, 1, lh, la, rho) == pytest.approx(1 + lh * rho)
        assert dixon_coles_tau(1, 0, lh, la, rho) == pytest.approx(1 + la * rho)
        assert dixon_coles_tau(1, 1, lh, la, rho) == pytest.approx(1 - rho)

    def test_scores_above_one_are_unaffected(self):
        for x, y in [(2, 0), (0, 2), (2, 2), (3, 5)]:
            assert dixon_coles_tau(x, y, lambda_home=1.5, lambda_away=1.2, rho=0.15) == pytest.approx(1.0)


def _simulate_matches(
    teams: list[str],
    true_attack: dict[str, float],
    true_defense: dict[str, float],
    true_home_adv: float,
    n_rounds: int,
    seed: int = 0,
) -> pd.DataFrame:
    """Every team plays every other team home and away, `n_rounds` times,
    with goals drawn from the exact Poisson process the model assumes."""
    rng = np.random.default_rng(seed)
    rows = []
    date = pd.Timestamp("2020-08-01")
    for _ in range(n_rounds):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                lam_h = np.exp(true_attack[home] + true_defense[away] + true_home_adv)
                lam_a = np.exp(true_attack[away] + true_defense[home])
                rows.append({
                    "Date": date,
                    "HomeTeam": home,
                    "AwayTeam": away,
                    "FTHG": rng.poisson(lam_h),
                    "FTAG": rng.poisson(lam_a),
                })
        date += pd.Timedelta(days=7)
    return pd.DataFrame(rows)


class TestParameterRecovery:
    TEAMS = ["Alpha", "Bravo", "Charlie", "Delta"]
    TRUE_ATTACK = {"Alpha": 0.35, "Bravo": 0.10, "Charlie": -0.10, "Delta": -0.35}
    TRUE_DEFENSE = {"Alpha": -0.25, "Bravo": 0.00, "Charlie": 0.05, "Delta": 0.30}
    TRUE_HOME_ADV = 0.25

    def test_recovers_true_expected_goals(self):
        train = _simulate_matches(
            self.TEAMS, self.TRUE_ATTACK, self.TRUE_DEFENSE, self.TRUE_HOME_ADV, n_rounds=40
        )
        # xi=0 (no recency decay) so every simulated round is weighted
        # equally, matching how the data was generated.
        model = DixonColesModel(use_correlation=False, xi=0.0).fit(train)

        for home in self.TEAMS:
            for away in self.TEAMS:
                if home == away:
                    continue
                true_lam_h = np.exp(self.TRUE_ATTACK[home] + self.TRUE_DEFENSE[away] + self.TRUE_HOME_ADV)
                true_lam_a = np.exp(self.TRUE_ATTACK[away] + self.TRUE_DEFENSE[home])
                pred_lam_h, pred_lam_a = model.predict_expected_goals(home, away)
                assert pred_lam_h == pytest.approx(true_lam_h, rel=0.2)
                assert pred_lam_a == pytest.approx(true_lam_a, rel=0.2)

    def test_recovers_true_home_advantage(self):
        train = _simulate_matches(
            self.TEAMS, self.TRUE_ATTACK, self.TRUE_DEFENSE, self.TRUE_HOME_ADV, n_rounds=40
        )
        model = DixonColesModel(use_correlation=False, xi=0.0).fit(train)
        assert model.home_advantage_ == pytest.approx(self.TRUE_HOME_ADV, abs=0.1)


class TestPromotedTeamFallback:
    def test_unseen_team_gets_below_average_attack_and_above_average_defense(self):
        train = _simulate_matches(
            TestParameterRecovery.TEAMS,
            TestParameterRecovery.TRUE_ATTACK,
            TestParameterRecovery.TRUE_DEFENSE,
            TestParameterRecovery.TRUE_HOME_ADV,
            n_rounds=20,
        )
        model = DixonColesModel(use_correlation=False, xi=0.0).fit(train)
        attack, defense = model._team_params("Newly Promoted FC")
        assert attack < model.mean_attack_
        assert defense > model.mean_defense_

    def test_unseen_team_does_not_raise(self):
        train = _simulate_matches(
            TestParameterRecovery.TEAMS,
            TestParameterRecovery.TRUE_ATTACK,
            TestParameterRecovery.TRUE_DEFENSE,
            TestParameterRecovery.TRUE_HOME_ADV,
            n_rounds=20,
        )
        model = DixonColesModel(use_correlation=False, xi=0.0).fit(train)
        lam_h, lam_a = model.predict_expected_goals("Newly Promoted FC", "Alpha")
        assert lam_h > 0
        assert lam_a > 0


class TestScoreGridAndProba:
    def _fitted_model(self, **kwargs) -> DixonColesModel:
        train = _simulate_matches(
            TestParameterRecovery.TEAMS,
            TestParameterRecovery.TRUE_ATTACK,
            TestParameterRecovery.TRUE_DEFENSE,
            TestParameterRecovery.TRUE_HOME_ADV,
            n_rounds=15,
        )
        return DixonColesModel(xi=0.0, **kwargs).fit(train)

    def test_score_grid_sums_to_one(self):
        model = self._fitted_model()
        grid = model.predict_score_grid("Alpha", "Delta")
        assert grid.sum() == pytest.approx(1.0)
        assert (grid >= 0).all()

    def test_predict_proba_valid_distribution(self):
        model = self._fitted_model()
        test = pd.DataFrame({"HomeTeam": ["Alpha", "Delta"], "AwayTeam": ["Delta", "Alpha"]})
        proba = model.predict_proba(test)
        assert list(proba.columns) == ["A", "D", "H"]
        assert (proba.sum(axis=1) - 1.0).abs().max() < 1e-6
        assert (proba.to_numpy() >= 0).all()

    def test_stronger_home_team_favored(self):
        # Alpha (strong attack, strong defense) at home against Delta
        # (weak attack, weak defense) should be predicted to win more
        # often than not.
        model = self._fitted_model()
        proba = model.predict_proba(pd.DataFrame({"HomeTeam": ["Alpha"], "AwayTeam": ["Delta"]}))
        assert proba["H"].iloc[0] > proba["A"].iloc[0]

    def test_correlation_disabled_matches_plain_independent_poisson(self):
        # With use_correlation=False, rho must never be optimized (stays 0),
        # and the score grid must be an exact outer product of the two
        # independent Poisson pmfs - no tau adjustment applied anywhere.
        model = self._fitted_model(use_correlation=False)
        assert model.rho_ == 0.0

        from scipy.stats import poisson as sp_poisson
        lam_h, lam_a = model.predict_expected_goals("Alpha", "Bravo")
        goals = np.arange(model.max_goals + 1)
        expected_grid = np.outer(sp_poisson.pmf(goals, lam_h), sp_poisson.pmf(goals, lam_a))
        expected_grid /= expected_grid.sum()

        actual_grid = model.predict_score_grid("Alpha", "Bravo")
        assert np.allclose(actual_grid, expected_grid, atol=1e-9)


class TestTopScorelines:
    def test_returns_k_results_sorted_by_probability_descending(self):
        grid = np.zeros((4, 4))
        grid[1, 0] = 0.3
        grid[1, 1] = 0.25
        grid[0, 0] = 0.2
        grid[2, 1] = 0.1
        grid[0, 1] = 0.05
        grid /= grid.sum()

        result = top_scorelines(grid, k=3)
        assert len(result) == 3
        assert result[0][:2] == (1, 0)
        assert result[1][:2] == (1, 1)
        assert result[2][:2] == (0, 0)
        probs = [r[2] for r in result]
        assert probs == sorted(probs, reverse=True)

    def test_probabilities_match_grid_values(self):
        grid = np.zeros((3, 3))
        grid[2, 1] = 0.6
        grid[0, 0] = 0.4
        result = top_scorelines(grid, k=2)
        assert result[0] == (2, 1, 0.6)
        assert result[1] == (0, 0, 0.4)

    def test_k_larger_than_grid_size_returns_whole_grid(self):
        grid = np.array([[0.5, 0.5]])
        result = top_scorelines(grid, k=10)
        assert len(result) == 2
