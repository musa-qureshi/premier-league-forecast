"""Sequential, leakage-safe league-table state: for each match, what were
each team's points/goal difference/games played/table position BEFORE this
match, within the current season (tables reset every season)?

Implemented as a small stateful tracker processed strictly in chronological
order, the same style as src/features/elo.py - deliberately NOT a
pandas groupby/cumsum vectorized trick. Points/goals-for/goals-against
alone could be done that way (each team's own cumulative history has no
cross-team ambiguity). Table POSITION can't: it requires ranking every team
in the league against every other team as of the same moment, and this
dataset only records a match's date, not its kickoff time. A sequential
pass processed one calendar date at a time - snapshotting every match on
that date using the state as it stood BEFORE that date, then applying every
result from that date together - makes "state as of right now" unambiguous
by construction, regardless of which order same-date matches happen to
appear in the source data. See test_league_table.py::
test_same_date_matches_see_identical_snapshot for why this matters: a
naive row-by-row update would let one same-day match's result leak into
another same-day match's position feature, depending on arbitrary row
order - a subtle leakage bug this project explicitly wants to avoid.

At this dataset's scale (~11k matches total), the performance cost of a
Python-level loop is irrelevant - this runs once as a preprocessing step,
not inside the Monte Carlo simulator, which is where vectorization actually
matters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class _TeamState:
    played: int = 0
    points: int = 0
    goals_for: int = 0
    goals_against: int = 0

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against


class LeagueTableTracker:
    def __init__(self) -> None:
        self._season: str | None = None
        self._table: dict[str, _TeamState] = {}

    def _maybe_reset_for_new_season(self, season: str) -> None:
        if season != self._season:
            self._table = {}
            self._season = season

    def _register(self, team: str) -> None:
        if team not in self._table:
            self._table[team] = _TeamState()

    def _position(self, team: str) -> int:
        """1 = top of the table. Ties broken by points, then goal
        difference, then goals scored - the real Premier League tie-break
        order, excluding head-to-head record (a deliberate simplification,
        also used by the Monte Carlo simulator - see project plan for why:
        head-to-head isn't cleanly vectorizable and almost never decides
        the title/top-4/relegation outcomes the simulator actually reports)."""
        ranked = sorted(
            self._table.items(),
            key=lambda kv: (-kv[1].points, -kv[1].goal_difference, -kv[1].goals_for),
        )
        for i, (name, _) in enumerate(ranked, start=1):
            if name == team:
                return i
        raise KeyError(f"{team} not registered in the current table")

    def snapshot(self, home_team: str, away_team: str, season: str) -> dict:
        """Read-only: registers both teams if new, and returns their
        pre-match points/played/goal-difference/position. Does NOT apply
        this match's result - call apply_result separately, after every
        match on the same date has been snapshotted."""
        self._maybe_apply_season_reset_and_register(home_team, away_team, season)
        home = self._table[home_team]
        away = self._table[away_team]
        return {
            "home_table_points": home.points,
            "home_table_played": home.played,
            "home_table_goal_diff": home.goal_difference,
            "home_table_position": self._position(home_team),
            "away_table_points": away.points,
            "away_table_played": away.played,
            "away_table_goal_diff": away.goal_difference,
            "away_table_position": self._position(away_team),
        }

    def _maybe_apply_season_reset_and_register(self, home_team: str, away_team: str, season: str) -> None:
        self._maybe_reset_for_new_season(season)
        self._register(home_team)
        self._register(away_team)

    def apply_result(self, home_team: str, away_team: str, home_goals: int, away_goals: int) -> None:
        """Mutates state to reflect a completed match. Must be called after
        snapshot() has already been taken for this match (and, for
        same-date matches, after every match on that date has been
        snapshotted first)."""
        home = self._table[home_team]
        away = self._table[away_team]

        home.played += 1
        away.played += 1
        home.goals_for += home_goals
        home.goals_against += away_goals
        away.goals_for += away_goals
        away.goals_against += home_goals

        if home_goals > away_goals:
            home.points += 3
        elif home_goals < away_goals:
            away.points += 3
        else:
            home.points += 1
            away.points += 1

    def current_standings(self) -> dict[str, dict]:
        """A public snapshot of every registered team's current state
        (points/played/goals_for/goals_against/goal_difference) - the
        same role EloRatingSystem.current_ratings() plays for Elo. Used to
        seed the Monte Carlo simulator's starting table state (Phase 8)
        and, eventually, the live current-season forecast (Phase 9) -
        callers should use this rather than reaching into the tracker's
        internal _table representation directly."""
        return {
            team: {
                "points": state.points,
                "played": state.played,
                "goals_for": state.goals_for,
                "goals_against": state.goals_against,
                "goal_difference": state.goal_difference,
            }
            for team, state in self._table.items()
        }


def standings_by_matchweek(matches: pd.DataFrame, season: str) -> list[dict]:
    """Reconstructs the table exactly as it stood after every COMPLETED
    matchweek this season - the "history" the live forecast's matchweek
    browser reads from. `matches` should be the season's full schedule
    (played and not-yet-played), each row carrying the integer `Round`
    column src/data/validate.py::load_openfootball_matches populates from
    the source's "Matchday N" label.

    Deliberately NOT a persisted/saved snapshot: the live data source
    always returns the complete season-to-date match list on every fetch,
    so every matchweek's final state is always fully recoverable by
    replaying from scratch - recomputing here costs nothing meaningful
    (this is the same lightweight sequential tracker used everywhere else
    in this project, not the Monte Carlo simulator) and can never lose
    history to e.g. a container restart wiping local disk.

    A matchweek only appears in the result once every one of its fixtures
    has been played - "the table after matchweek N" isn't well-defined
    while N is still in progress. Returns `[]` if `matches` has no `Round`
    column at all (true for any season sourced purely from the Kaggle
    mirrors, which don't carry this field - see load_openfootball_matches).
    """
    if "Round" not in matches.columns:
        return []

    fixtures_per_round = matches.groupby("Round").size().to_dict()
    played = matches.dropna(subset=["FTHG", "FTAG"]).sort_values("Date")

    tracker = LeagueTableTracker()
    played_per_round: dict[float, int] = {}
    already_snapshotted: set[float] = set()
    snapshots: list[dict] = []

    # Date-batched, same as compute_league_table_features: every match on
    # a shared date must see (and contribute to) an identical table state,
    # regardless of row order.
    for _, day_matches in played.groupby("Date", sort=True):
        for _, row in day_matches.iterrows():
            tracker.snapshot(row["HomeTeam"], row["AwayTeam"], season)
        for _, row in day_matches.iterrows():
            tracker.apply_result(row["HomeTeam"], row["AwayTeam"], row["FTHG"], row["FTAG"])
            r = row["Round"]
            if pd.isna(r):
                continue
            played_per_round[r] = played_per_round.get(r, 0) + 1

        # After applying this whole date's results, snapshot any round
        # that JUST became fully complete - checked here rather than only
        # once at the end, since a round's fixtures can span several
        # dates and this needs the state as it stood right when the last
        # of them was played, not the season's final state.
        for r, count in played_per_round.items():
            if r not in already_snapshotted and count == fixtures_per_round.get(r):
                already_snapshotted.add(r)
                standings = tracker.current_standings()
                df = pd.DataFrame(standings).T
                df.index.name = "team"
                df = df.sort_values(["points", "goal_difference"], ascending=False)
                snapshots.append({"matchweek": int(r), "standings": df})

    snapshots.sort(key=lambda s: s["matchweek"])
    return snapshots


def compute_league_table_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Adds home_table_*/away_table_* columns to `matches`. `matches` must
    already be sorted chronologically by Date."""
    if not matches["Date"].is_monotonic_increasing:
        raise ValueError(
            "compute_league_table_features requires matches sorted "
            "chronologically - table state is only leakage-safe if "
            "processed in playing order."
        )

    tracker = LeagueTableTracker()
    snapshots: dict[int, dict] = {}

    # Group by Date (not by row) so every match on the same calendar date
    # sees an identical pre-match snapshot - see module docstring.
    for _, day_matches in matches.groupby("Date", sort=True):
        for idx, row in day_matches.iterrows():
            snapshots[idx] = tracker.snapshot(row["HomeTeam"], row["AwayTeam"], row["Season"])
        for idx, row in day_matches.iterrows():
            tracker.apply_result(row["HomeTeam"], row["AwayTeam"], row["FTHG"], row["FTAG"])

    feat = pd.DataFrame.from_dict(snapshots, orient="index").sort_index()
    return pd.concat([matches, feat], axis=1)
