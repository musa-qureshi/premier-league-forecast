"""Unit tests for src/data/validate.py.

These deliberately use small synthetic DataFrames rather than the real
downloaded dataset - the point of keeping validate.py's transformations as
pure functions is that we can test the *logic* (season boundaries, name
canonicalization, result derivation) without needing network access or a
Kaggle token in CI.
"""

import pandas as pd
import pytest

from src.data.validate import (
    assign_season,
    canonicalize_teams,
    check_result_consistency,
    derive_result,
    parse_dates,
    sanity_check,
)


class TestAssignSeason:
    def test_january_belongs_to_season_starting_previous_year(self):
        dates = pd.Series(pd.to_datetime(["2024-01-15"]))
        assert assign_season(dates).iloc[0] == "2023-24"

    def test_september_belongs_to_season_starting_same_year(self):
        dates = pd.Series(pd.to_datetime(["2024-09-01"]))
        assert assign_season(dates).iloc[0] == "2024-25"

    def test_august_cutover_boundary(self):
        # August 1st onward counts as the start of the new season year.
        july = pd.Series(pd.to_datetime(["2024-07-31"]))
        august = pd.Series(pd.to_datetime(["2024-08-01"]))
        assert assign_season(july).iloc[0] == "2023-24"
        assert assign_season(august).iloc[0] == "2024-25"

    def test_covid_delayed_season_stragglers_stay_in_prior_season(self):
        # Regression test: the COVID-disrupted 2019-20 season finished in
        # late July 2020. A July cutover (instead of August) misfiled these
        # matches into 2020-21, inflating that season to 446 matches / 23
        # teams instead of the correct 380/20 - caught by sanity_check
        # against the real downloaded dataset.
        late_july_2020 = pd.Series(pd.to_datetime(["2020-07-26"]))
        assert assign_season(late_july_2020).iloc[0] == "2019-20"

    def test_century_boundary_formats_correctly(self):
        dates = pd.Series(pd.to_datetime(["1999-10-01"]))
        assert assign_season(dates).iloc[0] == "1999-00"


class TestParseDates:
    def test_iso_string_with_day_le_12_is_not_day_month_swapped(self):
        # Regression test: pandas' automatic format-guessing (with or
        # without format="mixed") resolves day/month ambiguity in ISO
        # strings by scanning the array for *some* row unambiguous enough
        # to lock in a format, then applying that to every row - so a batch
        # containing only ambiguous rows (every date component <= 12, as
        # here) silently misparses "1993-09-01T00:00:00Z" (1 Sep) as 9 Jan.
        # This inflated every season's implied team count when it first hit
        # the pipeline, and it matters beyond that one bug: correctness
        # must not depend on which other rows happen to share the batch -
        # exactly the shape of a small incremental live-data update
        # (Phase 7). parse_dates() must get this right regardless of batch
        # size/composition by detecting ISO input explicitly rather than
        # guessing.
        df = pd.DataFrame({
            "Date": ["1993-09-01T00:00:00Z"],  # every component here is <= 12
            "HomeTeam": ["Arsenal"], "AwayTeam": ["Coventry"],
            "FTHG": [0], "FTAG": [3], "FTR": ["A"],
        })
        result = parse_dates(df)
        assert result["Date"].iloc[0] == pd.Timestamp("1993-09-01")

    def test_ddmmyyyy_string_still_parsed_dayfirst(self):
        # A raw football-data.co.uk-style DD/MM/YYYY string must still be
        # interpreted day-first, not US-style month-first.
        df = pd.DataFrame({
            "Date": ["01/09/1993"],
            "HomeTeam": ["Arsenal"], "AwayTeam": ["Coventry"],
            "FTHG": [0], "FTAG": [3], "FTR": ["A"],
        })
        result = parse_dates(df)
        assert result["Date"].iloc[0] == pd.Timestamp("1993-09-01")


class TestCanonicalizeTeams:
    def test_known_alias_is_mapped(self):
        df = pd.DataFrame({"HomeTeam": ["Man Utd"], "AwayTeam": ["Spurs"]})
        team_map = {"Man Utd": "Manchester United", "Spurs": "Tottenham Hotspur"}
        result = canonicalize_teams(df, team_map)
        assert result["HomeTeam"].iloc[0] == "Manchester United"
        assert result["AwayTeam"].iloc[0] == "Tottenham Hotspur"

    def test_unmapped_name_is_left_unchanged(self):
        df = pd.DataFrame({"HomeTeam": ["Arsenal"], "AwayTeam": ["Chelsea"]})
        result = canonicalize_teams(df, {"Man Utd": "Manchester United"})
        assert result["HomeTeam"].iloc[0] == "Arsenal"
        assert result["AwayTeam"].iloc[0] == "Chelsea"

    def test_whitespace_is_stripped_before_mapping(self):
        df = pd.DataFrame({"HomeTeam": [" Man Utd "], "AwayTeam": ["Arsenal"]})
        result = canonicalize_teams(df, {"Man Utd": "Manchester United"})
        assert result["HomeTeam"].iloc[0] == "Manchester United"


class TestDeriveResult:
    def test_home_win(self):
        result = derive_result(pd.Series([2]), pd.Series([1]))
        assert result.iloc[0] == "H"

    def test_away_win(self):
        result = derive_result(pd.Series([0]), pd.Series([3]))
        assert result.iloc[0] == "A"

    def test_draw(self):
        result = derive_result(pd.Series([1]), pd.Series([1]))
        assert result.iloc[0] == "D"

    def test_scoreless_draw(self):
        result = derive_result(pd.Series([0]), pd.Series([0]))
        assert result.iloc[0] == "D"


class TestCheckResultConsistency:
    def test_no_disagreement(self):
        ftr = pd.Series(["H", "D", "A"])
        derived = pd.Series(["H", "D", "A"])
        assert check_result_consistency(ftr, derived) == 0

    def test_counts_disagreements(self):
        ftr = pd.Series(["H", "D", "A"])
        derived = pd.Series(["H", "H", "A"])  # one mismatch
        assert check_result_consistency(ftr, derived) == 1


class TestSanityCheck:
    def _valid_matches(self) -> pd.DataFrame:
        # A minimal internally-consistent 2-team round-robin: each of the
        # 2 teams plays the other home and away -> n*(n-1) = 2 matches.
        return pd.DataFrame({
            "Date": pd.to_datetime(["2023-08-12", "2023-12-26"]),
            "HomeTeam": ["Arsenal", "Chelsea"],
            "AwayTeam": ["Chelsea", "Arsenal"],
            "FTHG": [2, 1],
            "FTAG": [1, 1],
            "FTR": ["H", "D"],
            "Season": ["2023-24", "2023-24"],
            "match_id": [0, 1],
        })

    def test_valid_data_passes(self):
        sanity_check(self._valid_matches())  # should not raise

    def test_unsorted_dates_raise(self):
        df = self._valid_matches().iloc[::-1].reset_index(drop=True)
        with pytest.raises(AssertionError, match="chronologically sorted"):
            sanity_check(df)

    def test_invalid_ftr_raises(self):
        df = self._valid_matches()
        df.loc[0, "FTR"] = "X"
        with pytest.raises(AssertionError, match="H/D/A"):
            sanity_check(df)

    def test_negative_goals_raise(self):
        df = self._valid_matches()
        df.loc[0, "FTHG"] = -1
        with pytest.raises(AssertionError, match="negative"):
            sanity_check(df)

    def test_duplicate_match_id_raises(self):
        df = self._valid_matches()
        df["match_id"] = [0, 0]
        with pytest.raises(AssertionError, match="unique"):
            sanity_check(df)
