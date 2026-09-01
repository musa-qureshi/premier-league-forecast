"""Vectorized Monte Carlo season simulator.

The idea, in plain terms: given every team's current league-table state
and the expected goals for every fixture still left to play, simulate an
entire remaining season thousands of times, each time letting every match
land on a randomly sampled scoreline (weighted by how likely that
scoreline actually is), then tally up how the final table looked in each
one of those simulated worlds. Run that 10,000-100,000 times and the
fraction of simulated worlds where, say, Arsenal finished 1st IS Arsenal's
simulated title probability - Monte Carlo simulation is nothing more than
"turn a probability distribution into a huge pile of concrete samples,
then just count".

Why not simulate one match at a time in a Python loop, 100,000 times?
That's the naive approach, and it would work, but it would be very slow:
for ~100 remaining fixtures x 100,000 simulations that's 10 million
individual random draws, each wrapped in Python-level loop overhead. This
module instead draws every simulation's every fixture's scoreline in a
handful of single, vectorized NumPy calls - the random sampling, the
points/goal-difference bookkeeping, and the final-table ranking are each
one array operation across the whole (n_simulations x n_fixtures) or
(n_simulations x n_teams) grid at once, not a nested loop.

Scoreline sampling: home_goals and away_goals are sampled INDEPENDENTLY
from their own marginal Poisson(lambda) distributions, rather than jointly
from the Dixon-Coles-adjusted score grid (src/models/poisson_model.py's
predict_score_grid). This is a deliberate, quantified simplification, not
an oversight: independent sampling vectorizes trivially (two
`rng.poisson(...)` calls covering every simulation and fixture at once),
while jointly sampling from the full adjusted grid at 100,000 simulations
x ~100 fixtures x ~121 possible scorelines would need well over a gigabyte
of intermediate probability mass just to set up the sampling - for a
correlation effect Phase 4's backtest already measured as barely moving
aggregate log loss (0.9891 independent vs. 0.9890 Dixon-Coles-adjusted,
a ~0.006% relative difference). Trading that already-tiny effect for a
much simpler, much faster simulator is a reasonable exchange, and is
recorded here precisely so it isn't later mistaken for a bug.

Tie-breaking: final standings are ranked by points, then goal difference,
then goals scored - the real Premier League order, MINUS head-to-head
record (excluded as a deliberate simplification: head-to-head isn't
cleanly vectorizable across many independent simulations, and it almost
never decides the title/top-4/relegation outcomes this simulator actually
reports - see src/features/league_table.py, which documents and uses the
same simplification for the historical-data tie-break).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CurrentTableState:
    """Each team's accumulated points/goals from matches ALREADY played
    this season - the fixed starting point every simulation adds its own
    simulated remaining fixtures on top of."""
    teams: list[str]
    points: dict[str, int] = field(default_factory=dict)
    goals_for: dict[str, int] = field(default_factory=dict)
    goals_against: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for t in self.teams:
            self.points.setdefault(t, 0)
            self.goals_for.setdefault(t, 0)
            self.goals_against.setdefault(t, 0)


@dataclass
class Fixture:
    home_team: str
    away_team: str
    lambda_home: float
    lambda_away: float


def simulate_scorelines(
    fixtures: list[Fixture], n_simulations: int, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (home_goals, away_goals), each shape (n_simulations,
    n_fixtures) - every fixture's expected goals sampled independently for
    every simulated season, in two vectorized calls.

    Cast down to int16 immediately: `rng.poisson()` has no dtype
    parameter and always hands back int64, but a single match's goal
    count is never remotely close to int16's ~32,000 headroom - carrying
    int64 through the rest of the pipeline for a value this small was
    pure waste. At realistic production scale (50,000 simulations x ~370
    remaining fixtures, an early-season live forecast), these two arrays
    alone were measured at ~296MB combined as int64 - the single largest
    contributor to a peak RSS that OOM-killed the deployed backend on
    Render's free 512MB tier (see README "Live deployment" for the full
    incident writeup). int16 cuts that to ~74MB for the same arrays, with
    zero change to any simulated value - this is a storage-width fix, not
    a behavior change."""
    rng = rng or np.random.default_rng()
    lambda_home = np.array([f.lambda_home for f in fixtures])
    lambda_away = np.array([f.lambda_away for f in fixtures])
    n_fixtures = len(fixtures)

    home_goals = rng.poisson(lam=lambda_home[None, :], size=(n_simulations, n_fixtures)).astype(np.int16)
    away_goals = rng.poisson(lam=lambda_away[None, :], size=(n_simulations, n_fixtures)).astype(np.int16)
    return home_goals, away_goals


def scorelines_to_points(home_goals: np.ndarray, away_goals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standard football points: 3 for a win, 1 each for a draw, 0 for a
    loss. Pulled out as its own pure function specifically so point
    assignment can be unit-tested directly against known scorelines,
    independent of any randomness.

    The 3/1/0 literals are explicitly np.int16 so np.where's output stays
    int16 rather than silently upcasting to int64 (NumPy's default int
    type) the moment a wider-typed literal enters the expression - the
    same memory reasoning as simulate_scorelines() above applies here."""
    three, one, zero = np.int16(3), np.int16(1), np.int16(0)
    home_pts = np.where(home_goals > away_goals, three, np.where(home_goals == away_goals, one, zero))
    away_pts = np.where(away_goals > home_goals, three, np.where(home_goals == away_goals, one, zero))
    return home_pts, away_pts


def simulate_final_tables(
    current: CurrentTableState,
    fixtures: list[Fixture],
    n_simulations: int,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs the full simulation and returns (positions, points), each a
    (n_simulations x n_teams) DataFrame - positions has 1 = champions in
    each cell, points has that simulation's final points total. This is
    the core output every summary statistic in src/simulation/summary.py
    is computed from.

    If `fixtures` is empty, every simulation trivially reproduces the
    current table exactly (no randomness left to resolve) - the natural
    and correct behavior for a season that's already finished.
    """
    rng = np.random.default_rng(seed)
    teams = current.teams
    team_idx = {t: i for i, t in enumerate(teams)}

    home_goals, away_goals = simulate_scorelines(fixtures, n_simulations, rng)
    home_pts, away_pts = scorelines_to_points(home_goals, away_goals)

    # Explicit dtype=int64 matters even (especially) when fixtures is empty:
    # np.array([]) defaults to float64, which numpy's advanced indexing
    # rejects outright - see test_empty_fixtures_reproduces_current_table.
    home_team_idx = np.array([team_idx[f.home_team] for f in fixtures], dtype=np.int64)
    away_team_idx = np.array([team_idx[f.away_team] for f in fixtures], dtype=np.int64)

    total_points = np.tile(
        np.array([current.points[t] for t in teams], dtype=np.int64), (n_simulations, 1)
    )
    total_gf = np.tile(
        np.array([current.goals_for[t] for t in teams], dtype=np.int64), (n_simulations, 1)
    )
    total_ga = np.tile(
        np.array([current.goals_against[t] for t in teams], dtype=np.int64), (n_simulations, 1)
    )

    sim_rows = np.arange(n_simulations)[:, None]  # broadcasts against the fixture axis below

    np.add.at(total_points, (sim_rows, home_team_idx[None, :]), home_pts)
    np.add.at(total_points, (sim_rows, away_team_idx[None, :]), away_pts)
    np.add.at(total_gf, (sim_rows, home_team_idx[None, :]), home_goals)
    np.add.at(total_gf, (sim_rows, away_team_idx[None, :]), away_goals)
    np.add.at(total_ga, (sim_rows, home_team_idx[None, :]), away_goals)
    np.add.at(total_ga, (sim_rows, away_team_idx[None, :]), home_goals)

    goal_diff = total_gf - total_ga

    # Combine (points, goal_diff, goals_for) into one sortable integer key
    # per (simulation, team) cell, so the whole league's final ranking for
    # every simulation can be computed with a single vectorized argsort
    # instead of a per-row/per-simulation Python loop. The multipliers are
    # generous safety margins above any value realistically reachable in a
    # single Premier League season (max points ~114, |goal difference|
    # comfortably under 500, goals for comfortably under 1000).
    sort_key = total_points * 2_000_000 + (goal_diff + 1000) * 1000 + total_gf

    order = np.argsort(-sort_key, axis=1)          # best-to-worst team index per simulation
    positions = np.argsort(order, axis=1) + 1        # invert the permutation -> each team's rank

    return pd.DataFrame(positions, columns=teams), pd.DataFrame(total_points, columns=teams)
