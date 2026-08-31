"""Model 3: Poisson goal model (Maher, 1982), with an optional Dixon-Coles
(1997) low-score correlation adjustment.

Why model goals instead of the W/D/L outcome directly? A W/D/L classifier
throws away information: a 3-1 and a 1-0 are both "home win" to it, even
though they say different things about how dominant the home team was.
Modeling the actual goal-scoring PROCESS instead - as two counts, home
goals and away goals - keeps that information, and as a side effect gives
this project something a plain classifier can't: an actual expected-goals
number per match, and a full scoreline probability grid (needed later for
"most likely scorelines" in match predictions, and useful groundwork for
sampling realistic scorelines in the Monte Carlo simulator).

The model (Maher, 1982): goals are Poisson-distributed counts, and each
team gets two strength parameters - an attack rating and a defense rating
- combined additively in log-space:

    home_goals ~ Poisson(lambda_home),   lambda_home = exp(attack_home + defense_away + home_adv)
    away_goals ~ Poisson(lambda_away),   lambda_away = exp(attack_away + defense_home)

Intuition: a team's expected goals in a given match depends on how good
THEIR attack is and how bad the OPPONENT's defense is (plus, for the home
side, a fixed home-advantage bonus) - exactly the "attacking strength
relative to opponent" feature idea from the project plan, but built into
the generative model itself rather than engineered as an input feature.
Higher `defense_i` means team i concedes MORE (a worse defense), by this
module's sign convention.

Why Poisson specifically? Goal counts are non-negative integers with no
fixed upper bound and (empirically, for a single team's goals in a single
match) a distribution close to Poisson's - the same distributional
assumption used to model any count of independent-ish events in a fixed
window (goals in 90 minutes, arrivals per hour, etc).

Fitting is a maximum-likelihood problem, not a sequential update like Elo:
every team's attack/defense parameters must be estimated jointly from the
whole training window at once, via numerical optimization (scipy.optimize)
of the Poisson (optionally Dixon-Coles-adjusted) log-likelihood. Two
consequences follow directly from that:

1. Identifiability: adding a constant c to every team's attack rating and
   subtracting c from every team's defense rating leaves every lambda
   unchanged (attack_i + defense_j is invariant to that joint shift) - so
   the optimizer has a flat, unidentified direction unless one parameter
   is pinned down. This model fixes the alphabetically-first team's attack
   rating to 0 as a reference point (a standard normalization choice - it
   doesn't privilege that team's actual strength, it's a pure gauge fix,
   analogous to how you can only ever ask about ELO ratings *relative* to
   each other, never in some absolute unit).
2. Recency: unlike Elo, which naturally forgets the past through its
   sequential K-factor updates, batch MLE over a huge expanding training
   window would weight a match from 1994 exactly as heavily as one from
   last month, even though squads have completely turned over since. This
   model applies an optional exponential recency weight to each training
   match, `exp(-xi * days_since_match / 365)`, following the same idea
   Dixon & Coles used in their original paper (there, tuned as a "half-
   life" on match importance) - xi=0 recovers uniform weighting.

The Dixon-Coles correlation adjustment: independent Poisson slightly
overstates the probability of a 0-0 or 1-1 draw and understates 1-0/0-1
results, because real matches aren't quite independent at very low scores
- both teams tend to play a bit more cautiously/reactively when the score
is still level and low. Dixon & Coles (1997) correct for this with a small
multiplicative factor applied only to the four low-scoring cells:

    P(x, y) = tau(x, y; lambda_home, lambda_away, rho) * Poisson(x; lambda_home) * Poisson(y; lambda_away)

with tau == 1 everywhere except (0,0), (0,1), (1,0), (1,1) - see
dixon_coles_tau() below for the exact formula. rho=0 makes tau == 1
everywhere, recovering the plain independent-Poisson model exactly - so
this module implements both models as one class with a use_correlation
toggle, rather than two near-duplicate implementations.

Teams with no rating history in the training window (promoted from the
Championship, or simply not yet encountered) get a fallback rating below
the training window's mean attack / above its mean defense - the same
"weaker than the incumbent average" idea used for new teams in
src/features/elo.py, kept as a separate, not-yet-empirically-tuned
constant here rather than assumed to be identical to Elo's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from src.evaluation.metrics import CLASS_ORDER

MAX_GOALS = 10  # truncate the scoreline grid; P(either team scores > 10) is negligible in football


def dixon_coles_tau(x: int, y: int, lambda_home: float, lambda_away: float, rho: float) -> float:
    """The Dixon-Coles low-score correlation correction factor. Affects
    only the four cells where both teams score 0 or 1; every other
    scoreline is untouched (tau == 1)."""
    if x == 0 and y == 0:
        return 1 - (lambda_home * lambda_away * rho)
    if x == 0 and y == 1:
        return 1 + (lambda_home * rho)
    if x == 1 and y == 0:
        return 1 + (lambda_away * rho)
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


class DixonColesModel:
    def __init__(
        self,
        use_correlation: bool = True,
        xi: float = 0.3,
        promoted_penalty: float = 0.4,
        max_goals: int = MAX_GOALS,
    ) -> None:
        self.use_correlation = use_correlation
        self.xi = xi
        self.promoted_penalty = promoted_penalty
        self.max_goals = max_goals

        self.teams_: list[str] = []
        self.attack_: dict[str, float] = {}
        self.defense_: dict[str, float] = {}
        self.home_advantage_: float = 0.0
        self.rho_: float = 0.0
        self.mean_attack_: float = 0.0
        self.mean_defense_: float = 0.0

    def fit(self, train: pd.DataFrame) -> "DixonColesModel":
        """Two-stage fit, for speed, not just convenience:

        Stage 1 fits attack/defense/home_advantage by maximizing the plain
        independent-Poisson log-likelihood, using an ANALYTICAL gradient
        (derived below) rather than letting scipy fall back to numerical
        finite-difference gradients. That distinction matters a lot in
        practice: with ~50 Premier League teams there are ~100 free
        parameters, and a numerical gradient needs on the order of 100
        extra log-likelihood evaluations per optimizer step - measured on
        this project's full dataset, that made a single fit take
        60-100+ seconds. The analytical gradient below costs one pass over
        the data, the same as the log-likelihood itself, and reduced that
        to well under a second - the difference between "runs 19 times
        during a backtest without a second thought" and "impractical to
        iterate on".

        Stage 2, only if use_correlation=True, fits rho - and ONLY rho -
        by maximizing the Dixon-Coles-adjusted log-likelihood with
        attack/defense/home_advantage held fixed at their Stage 1 values.
        This is a deliberate simplification versus jointly re-optimizing
        every parameter together: rho is a single scalar with a small,
        well-behaved effect on the likelihood (it only touches the four
        low-scoring cells), so a 1-D search here is both fast (no gradient
        even needed) and, in practice, very close to the fully joint MLE -
        a standard two-stage estimation strategy for this kind of model.
        """
        teams = sorted(set(train["HomeTeam"]) | set(train["AwayTeam"]))
        n = len(teams)
        team_idx = {t: i for i, t in enumerate(teams)}

        if self.xi > 0:
            latest = train["Date"].max()
            days_ago = (latest - train["Date"]).dt.days.to_numpy()
            weights = np.exp(-self.xi * days_ago / 365.0)
        else:
            weights = np.ones(len(train))

        home_idx = train["HomeTeam"].map(team_idx).to_numpy()
        away_idx = train["AwayTeam"].map(team_idx).to_numpy()
        fthg = train["FTHG"].to_numpy()
        ftag = train["FTAG"].to_numpy()

        # Parameter vector layout: [attack_1..attack_{n-1}, defense_0..defense_{n-1}, home_adv]
        # attack_0 (the alphabetically-first team) is fixed at 0 - see
        # module docstring "Identifiability".
        n_attack_free = n - 1

        def unpack(params: np.ndarray):
            attack = np.concatenate([[0.0], params[:n_attack_free]])
            defense = params[n_attack_free:n_attack_free + n]
            home_adv = params[n_attack_free + n]
            return attack, defense, home_adv

        def neg_log_lik_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
            attack, defense, home_adv = unpack(params)
            lam_home = np.exp(attack[home_idx] + defense[away_idx] + home_adv)
            lam_away = np.exp(attack[away_idx] + defense[home_idx])

            ll = weights * (poisson.logpmf(fthg, lam_home) + poisson.logpmf(ftag, lam_away))
            neg_ll = -float(ll.sum())

            # d(log-likelihood)/d(linear predictor) for a Poisson GLM is
            # the classic (observed - expected) residual: (y - lambda).
            # Each per-match residual is scattered into the gradient of
            # whichever team's attack/defense parameter it flows through -
            # np.add.at accumulates that scatter in one vectorized pass.
            resid_home = weights * (fthg - lam_home)
            resid_away = weights * (ftag - lam_away)

            grad_attack = np.zeros(n)
            grad_defense = np.zeros(n)
            np.add.at(grad_attack, home_idx, resid_home)
            np.add.at(grad_attack, away_idx, resid_away)
            np.add.at(grad_defense, away_idx, resid_home)
            np.add.at(grad_defense, home_idx, resid_away)
            grad_home_adv = resid_home.sum()

            # Negate (minimizing, not maximizing) and drop attack[0]'s
            # gradient - it's fixed, not a free parameter.
            grad = -np.concatenate([grad_attack[1:], grad_defense, [grad_home_adv]])
            return neg_ll, grad

        n_params = n_attack_free + n + 1
        x0 = np.zeros(n_params)
        result = minimize(neg_log_lik_and_grad, x0, jac=True, method="L-BFGS-B")

        attack, defense, home_adv = unpack(result.x)
        self.teams_ = teams
        self.attack_ = dict(zip(teams, attack))
        self.defense_ = dict(zip(teams, defense))
        self.home_advantage_ = float(home_adv)
        self.mean_attack_ = float(attack.mean())
        self.mean_defense_ = float(defense.mean())

        if self.use_correlation:
            lam_home = np.exp(attack[home_idx] + defense[away_idx] + home_adv)
            lam_away = np.exp(attack[away_idx] + defense[home_idx])

            def neg_log_lik_rho_only(rho: float) -> float:
                tau = np.array([
                    dixon_coles_tau(x, y, lh, la, rho)
                    for x, y, lh, la in zip(fthg, ftag, lam_home, lam_away)
                ])
                tau = np.clip(tau, 1e-10, None)
                return -float((weights * np.log(tau)).sum())

            rho_result = minimize(
                lambda r: neg_log_lik_rho_only(r[0]), x0=[0.0], method="L-BFGS-B",
                bounds=[(-0.5, 0.5)],
            )
            self.rho_ = float(rho_result.x[0])
        else:
            self.rho_ = 0.0

        return self

    def _team_params(self, team: str) -> tuple[float, float]:
        if team in self.attack_:
            return self.attack_[team], self.defense_[team]
        return (
            self.mean_attack_ - self.promoted_penalty,
            self.mean_defense_ + self.promoted_penalty,
        )

    def predict_expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Returns (lambda_home, lambda_away) - the expected-goals numbers
        referenced throughout the project plan's match-prediction output."""
        attack_h, defense_h = self._team_params(home_team)
        attack_a, defense_a = self._team_params(away_team)
        lam_home = float(np.exp(attack_h + defense_a + self.home_advantage_))
        lam_away = float(np.exp(attack_a + defense_h))
        return lam_home, lam_away

    def predict_score_grid(self, home_team: str, away_team: str) -> np.ndarray:
        """Returns a (max_goals+1, max_goals+1) array where cell [i, j] is
        P(home scores i, away scores j), Dixon-Coles-adjusted if enabled.
        Used both for deriving W/D/L probabilities and (Phase 10+) for
        reporting "most likely scorelines"."""
        lam_home, lam_away = self.predict_expected_goals(home_team, away_team)
        goals = np.arange(self.max_goals + 1)
        p_home = poisson.pmf(goals, lam_home)
        p_away = poisson.pmf(goals, lam_away)
        grid = np.outer(p_home, p_away)

        if self.use_correlation:
            for x in range(2):
                for y in range(2):
                    grid[x, y] *= dixon_coles_tau(x, y, lam_home, lam_away, self.rho_)

        grid = grid / grid.sum()  # renormalize: truncating at max_goals and the tau
                                    # adjustment both nudge the grid slightly off exactly 1
        return grid

    def predict_proba(self, test: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for row in test.itertuples(index=False):
            grid = self.predict_score_grid(row.HomeTeam, row.AwayTeam)
            goals = np.arange(self.max_goals + 1)
            home_wins = np.greater.outer(goals, goals)
            draws = np.equal.outer(goals, goals)
            away_wins = np.less.outer(goals, goals)
            rows.append({
                "A": float(grid[away_wins].sum()),
                "D": float(grid[draws].sum()),
                "H": float(grid[home_wins].sum()),
            })
        return pd.DataFrame(rows, columns=CLASS_ORDER, index=test.index)
