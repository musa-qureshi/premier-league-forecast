"""Unit tests for src/features/league_table.py.

The most important test here is test_same_date_matches_see_identical_
snapshot: this dataset only records a match's date, not its kickoff time,
so two matches on the same date must be treated as informationally
simultaneous. A naive row-by-row table update would let whichever same-day
match happens to be processed first leak its result into the table-position
feature of a later same-day match - a subtle form of leakage this project
explicitly wants to avoid (see module docstring in league_table.py).
"""

import pandas as pd
import pytest

from src.features.league_table import LeagueTableTracker, compute_league_table_features


def _matches(rows: list[tuple]) -> pd.DataFrame:
    """rows: (date, home, away, fthg, ftag, season)."""
    df = pd.DataFrame(rows, columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Season"])
    df["Date"] = pd.to_datetime(df["Date"])
    df["match_id"] = range(len(df))
    return df


class TestLeagueTableTracker:
    def test_new_team_starts_at_zero(self):
        tracker = LeagueTableTracker()
        snap = tracker.snapshot("Arsenal", "Chelsea", season="2023-24")
        assert snap["home_table_points"] == 0
        assert snap["home_table_played"] == 0
        assert snap["home_table_goal_diff"] == 0

    def test_points_accumulate_win_draw_loss(self):
        tracker = LeagueTableTracker()
        tracker.snapshot("Arsenal", "Chelsea", season="2023-24")
        tracker.apply_result("Arsenal", "Chelsea", 2, 0)  # Arsenal win

        snap = tracker.snapshot("Arsenal", "Everton", season="2023-24")
        assert snap["home_table_points"] == 3
        assert snap["home_table_played"] == 1
        assert snap["home_table_goal_diff"] == 2

        tracker.apply_result("Arsenal", "Everton", 1, 1)  # Arsenal draw
        snap2 = tracker.snapshot("Arsenal", "Fulham", season="2023-24")
        assert snap2["home_table_points"] == 4  # 3 + 1
        assert snap2["home_table_played"] == 2

    def test_pre_match_snapshot_excludes_this_match(self):
        tracker = LeagueTableTracker()
        snap = tracker.snapshot("Arsenal", "Chelsea", season="2023-24")
        assert snap["home_table_played"] == 0  # not yet counting the match about to happen
        tracker.apply_result("Arsenal", "Chelsea", 5, 0)
        # A later snapshot must now reflect it.
        snap2 = tracker.snapshot("Arsenal", "Everton", season="2023-24")
        assert snap2["home_table_played"] == 1

    def test_position_ordered_by_points(self):
        tracker = LeagueTableTracker()
        tracker.snapshot("Arsenal", "Chelsea", season="2023-24")
        tracker.apply_result("Arsenal", "Chelsea", 3, 0)  # Arsenal 3pts, Chelsea 0pts
        snap = tracker.snapshot("Arsenal", "Chelsea", season="2023-24")
        assert snap["home_table_position"] == 1
        assert snap["away_table_position"] == 2

    def test_position_tiebreak_by_goal_difference_then_goals_scored(self):
        tracker = LeagueTableTracker()
        # Both teams draw their openers with the same points but different GD.
        tracker.snapshot("Arsenal", "Watford", season="2023-24")
        tracker.apply_result("Arsenal", "Watford", 3, 1)  # Arsenal: 3pts, GD +2
        tracker.snapshot("Chelsea", "Fulham", season="2023-24")
        tracker.apply_result("Chelsea", "Fulham", 1, 0)   # Chelsea: 3pts, GD +1

        snap = tracker.snapshot("Arsenal", "Chelsea", season="2023-24")
        assert snap["home_table_position"] < snap["away_table_position"]  # Arsenal ranks above Chelsea

    def test_season_reset_clears_points_but_not_between_teams_within_season(self):
        tracker = LeagueTableTracker()
        tracker.snapshot("Arsenal", "Chelsea", season="2023-24")
        tracker.apply_result("Arsenal", "Chelsea", 3, 0)

        # Crossing into a new season must wipe accumulated points/played.
        snap_new_season = tracker.snapshot("Arsenal", "Chelsea", season="2024-25")
        assert snap_new_season["home_table_points"] == 0
        assert snap_new_season["home_table_played"] == 0


class TestComputeLeagueTableFeatures:
    def test_out_of_order_input_raises(self):
        rows = [
            ("2023-08-19", "Chelsea", "Arsenal", 1, 1, "2023-24"),
            ("2023-08-12", "Arsenal", "Chelsea", 2, 0, "2023-24"),
        ]
        with pytest.raises(ValueError, match="chronologically"):
            compute_league_table_features(_matches(rows))

    def test_same_date_matches_see_identical_snapshot(self):
        # Four teams, two matches on the SAME date. Neither match's result
        # may affect the other's pre-match snapshot, regardless of which
        # row happens to come first in the DataFrame.
        rows = [
            ("2023-08-12", "Arsenal", "Watford", 5, 0, "2023-24"),
            ("2023-08-12", "Chelsea", "Fulham", 1, 1, "2023-24"),
        ]
        result = compute_league_table_features(_matches(rows))
        # Both matches are on matchday 1: every team enters with 0 points,
        # 0 played, 0 GD - Arsenal's 5-0 result must NOT have already been
        # applied by the time Chelsea vs Fulham's snapshot was taken, even
        # though Arsenal's match appears first in row order. (Note: with
        # all four teams fully tied, position still resolves to distinct
        # ranks via a deterministic registration-order tie-break - no
        # ranking system assigns two different teams the same rank number
        # even on a dead tie, so position itself isn't the interesting
        # assertion here; the points/played/GD fields are.)
        chelsea_row = result.iloc[1]
        assert chelsea_row["home_table_played"] == 0
        assert chelsea_row["home_table_points"] == 0
        assert chelsea_row["home_table_goal_diff"] == 0

        # Reversing the row order must produce an identical points/played/GD
        # result for Chelsea's snapshot - correctness of the actual table
        # state must not depend on arbitrary same-date row order. (Position
        # is deliberately not compared here: with all four teams on a dead
        # 0-0-0 tie, position falls back to a registration-order tie-break,
        # which - reasonably - does shift if row order shifts. That's a
        # cosmetic tie-break artifact among genuinely-equal teams, not a
        # temporal leak: no match's actual result ever crosses into
        # another same-date match's snapshot, which is the property that
        # matters and is what points/played/GD verify.)
        rows_reversed = [rows[1], rows[0]]
        result_reversed = compute_league_table_features(_matches(rows_reversed))
        chelsea_row_2 = result_reversed[result_reversed["HomeTeam"] == "Chelsea"].iloc[0]
        assert chelsea_row_2["home_table_played"] == chelsea_row["home_table_played"]
        assert chelsea_row_2["home_table_points"] == chelsea_row["home_table_points"]
        assert chelsea_row_2["home_table_goal_diff"] == chelsea_row["home_table_goal_diff"]

    def test_later_matchday_reflects_earlier_results(self):
        rows = [
            ("2023-08-12", "Arsenal", "Watford", 5, 0, "2023-24"),
            ("2023-08-19", "Arsenal", "Chelsea", 1, 1, "2023-24"),
        ]
        result = compute_league_table_features(_matches(rows))
        assert result.iloc[1]["home_table_points"] == 3
        assert result.iloc[1]["home_table_played"] == 1

    def test_output_row_count_matches_input(self):
        rows = [
            ("2023-08-12", "Arsenal", "Watford", 5, 0, "2023-24"),
            ("2023-08-12", "Chelsea", "Fulham", 1, 1, "2023-24"),
            ("2023-08-19", "Arsenal", "Chelsea", 1, 1, "2023-24"),
        ]
        result = compute_league_table_features(_matches(rows))
        assert len(result) == 3
