"""Sequential Elo rating system for team strength.

Why Elo at all? A league table's points total only reflects the *past* - it
says nothing about whether a team's underlying quality has improved or
declined recently, doesn't distinguish a 3-0 win from a 1-0 win, and treats
early- and late-season form identically. Elo instead maintains one running
number per team that moves up after eachwin (more after a surprising or
big one) and down after each loss, so it reacts to *current* strength rather
than only to *cumulative* results - which is exactly the kind of signal a
match-outcome model needs. It's also cheap to compute sequentially and easy
to reason about, unlike a model that would need the whole season's data to
retrospectively rank teams.

The intuition: every match is a bet. Before kickoff, the rating gap between
two teams implies a probability of who "should" win (`expected_score`).
After the match, whichever side did better than that probability implied
gains rating; whichever did worse loses rating - by an amount controlled by
K (how reactive the system is) and, optionally, scaled by how big the
margin of victory was.

The mathematical formulation (identical in spirit to chess Elo):

    E_home = 1 / (1 + 10^(-(R_home + H - R_away) / 400))

`E_home` is the probability the model currently assigns to a home win/draw-
weighted-as-half outcome; `H` is a fixed home-advantage bonus added only for
this calculation (it never becomes part of either team's actual rating).
After the result is known:

    R_home_new = R_home + K * M * (S_home - E_home)
    R_away_new = R_away - K * M * (S_home - E_home)

`S_home` is 1 for a home win, 0.5 for a draw, 0 for an away win. `M` is an
optional goal-difference multiplier (bigger wins move the rating more).
The two updates are equal and opposite - Elo is a zero-sum system, ratings
only ever get redistributed among teams, never created or destroyed. That
symmetry is what keeps the system's overall scale stable over decades of
matches rather than drifting.

Limitations worth being explicit about: Elo only "knows" wins/draws/losses
(plus goal margin, if enabled) - it has no concept of *how* a team is
winning (playing style, injuries, fixture congestion), and it reacts to
results with a lag (a team's true quality can change faster than K lets the
rating catch up). It's a strong, cheap baseline signal, not a complete
description of team strength - which is exactly why it becomes one feature
among several rather than the final answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


def expected_score(rating_a: float, rating_b: float, home_advantage: float = 0.0) -> float:
    """Probability that side A performs at least as well as a draw against
    side B, given a home_advantage bonus (in Elo points) added to A's side
    only. Symmetric in the sense that expected_score(a, b) == 1 -
    expected_score(b, a) when home_advantage=0."""
    return 1.0 / (1.0 + 10 ** (-((rating_a + home_advantage) - rating_b) / 400.0))


def match_result_score(home_goals: int, away_goals: int) -> float:
    """1.0 for a home win, 0.5 for a draw, 0.0 for an away win - the actual
    outcome, on the same 0-1 scale as expected_score, so the two can be
    compared directly."""
    if home_goals > away_goals:
        return 1.0
    if home_goals < away_goals:
        return 0.0
    return 0.5


def goal_difference_multiplier(goal_diff: int) -> float:
    """Scales K by margin of victory, using the widely-used World Football
    Elo Ratings formula. A 4-0 win is stronger evidence of a quality gap
    than a 1-0 win, so it should move ratings further - but the relationship
    is sub-linear (a 6-0 win shouldn't move ratings 6x as much as a 1-0 win;
    most of that margin is likely variance, not signal, once the game is
    already well out of reach)."""
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0


def update_ratings(
    rating_home: float,
    rating_away: float,
    home_goals: int,
    away_goals: int,
    k_factor: float = 20.0,
    home_advantage: float = 100.0,
    use_goal_difference_multiplier: bool = True,
) -> tuple[float, float]:
    """Returns (new_rating_home, new_rating_away) after one match. Zero-sum:
    whatever the home team gains, the away team loses, and vice versa."""
    expected_home = expected_score(rating_home, rating_away, home_advantage)
    actual_home = match_result_score(home_goals, away_goals)
    multiplier = (
        goal_difference_multiplier(home_goals - away_goals)
        if use_goal_difference_multiplier
        else 1.0
    )
    delta = k_factor * multiplier * (actual_home - expected_home)
    return rating_home + delta, rating_away - delta


@dataclass
class EloRatingSystem:
    """Stateful, sequential Elo tracker for a whole league across seasons.

    Call `process_match` once per match **in chronological order** - the
    whole point of Elo as a leakage-safe feature is that the rating
    returned for a given match reflects only matches strictly before it.
    """

    initial_rating: float = 1500.0
    k_factor: float = 20.0
    home_advantage: float = 100.0
    use_goal_difference_multiplier: bool = True
    promoted_team_penalty: float = 150.0
    season_shrinkage: float = 0.33

    ratings: dict[str, float] = field(default_factory=dict)
    _current_season: str | None = field(default=None, repr=False)

    def _get_or_init_rating(self, team: str, had_ratings_before_this_match: bool,
                             mean_before_this_match: float | None) -> float:
        """`had_ratings_before_this_match` / `mean_before_this_match` must be
        computed once per match, from the state *before either side of this
        match is looked up* - not recomputed per-team. Otherwise, in the
        very first match of the whole dataset, looking up the home team
        first would insert it into `self.ratings`, making the dict look
        non-empty by the time the away team is looked up a moment later,
        and wrongly giving the away team a promoted-team penalty in what
        should be the league's inaugural, fully-symmetric match."""
        if team not in self.ratings:
            if not had_ratings_before_this_match:
                # First team(s) ever seen in the dataset - no basis to rate
                # them any differently from the anchor value.
                self.ratings[team] = self.initial_rating
            else:
                # A team with no rating history in this data: either newly
                # promoted from the Championship, or (for a handful of
                # clubs) simply not yet encountered this early in the
                # dataset. Start below the league mean *as it stood before
                # this match*, reflecting that promoted teams are typically
                # weaker than the incumbent top-flight average - refined
                # empirically in Phase 3, not asserted as fact here.
                self.ratings[team] = mean_before_this_match - self.promoted_team_penalty
        return self.ratings[team]

    def _maybe_apply_season_shrinkage(self, season: str) -> None:
        """At a season boundary, regress every currently-rated team's
        rating partway back toward the league mean. This models squad
        turnover over the summer break (transfers, managerial changes)
        making last season's rating a less confident estimate of this
        season's true strength - and is also the mechanism by which a team
        that gets relegated and later promoted back carries forward a
        (partially regressed) rating instead of an arbitrary hard reset."""
        season_changed = self._current_season is not None and season != self._current_season
        if season_changed and self.ratings:
            mean_rating = sum(self.ratings.values()) / len(self.ratings)
            for team in self.ratings:
                self.ratings[team] = mean_rating + (1 - self.season_shrinkage) * (
                    self.ratings[team] - mean_rating
                )
        self._current_season = season

    def process_match(
        self, home_team: str, away_team: str, home_goals: int, away_goals: int, season: str
    ) -> tuple[float, float]:
        """Processes one match and updates both teams' ratings. Returns the
        **pre-match** ratings (rating_home_before, rating_away_before) -
        this is what should be recorded as the feature for this match, since
        the post-match rating already incorporates information (the result)
        that a predictive model is not allowed to see."""
        self._maybe_apply_season_shrinkage(season)

        had_ratings_before_this_match = bool(self.ratings)
        mean_before_this_match = (
            sum(self.ratings.values()) / len(self.ratings)
            if had_ratings_before_this_match
            else None
        )
        rating_home_before = self._get_or_init_rating(
            home_team, had_ratings_before_this_match, mean_before_this_match
        )
        rating_away_before = self._get_or_init_rating(
            away_team, had_ratings_before_this_match, mean_before_this_match
        )

        new_home, new_away = update_ratings(
            rating_home_before,
            rating_away_before,
            home_goals,
            away_goals,
            self.k_factor,
            self.home_advantage,
            self.use_goal_difference_multiplier,
        )
        self.ratings[home_team] = new_home
        self.ratings[away_team] = new_away

        return rating_home_before, rating_away_before

    def current_ratings(self) -> dict[str, float]:
        """A snapshot of every team's rating as of the last processed match
        - used for Phase 8 (current-season forecasting): the starting point
        for simulating remaining fixtures."""
        return dict(self.ratings)


def compute_elo_features(matches: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Adds elo_home_pre, elo_away_pre, and elo_diff columns to `matches`.

    `matches` must already be sorted chronologically (this is a hard
    leakage-safety requirement, not a style preference - see
    tests/test_elo.py::test_out_of_order_input_raises). Returns a new
    DataFrame; does not mutate the input.
    """
    if not matches["Date"].is_monotonic_increasing:
        raise ValueError(
            "compute_elo_features requires matches sorted chronologically - "
            "Elo ratings are only leakage-safe if processed in playing order."
        )

    config = config or {}
    elo = EloRatingSystem(**config)

    elo_home_pre = []
    elo_away_pre = []
    for row in matches.itertuples(index=False):
        r_home, r_away = elo.process_match(
            row.HomeTeam, row.AwayTeam, row.FTHG, row.FTAG, row.Season
        )
        elo_home_pre.append(r_home)
        elo_away_pre.append(r_away)

    result = matches.copy()
    result["elo_home_pre"] = elo_home_pre
    result["elo_away_pre"] = elo_away_pre
    result["elo_diff"] = result["elo_home_pre"] - result["elo_away_pre"]
    return result
