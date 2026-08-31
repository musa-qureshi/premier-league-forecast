"""Unit tests for src/data/validate.py.

These deliberately use small synthetic DataFrames rather than the real
downloaded dataset - the point of keeping validate.py's transformations as
pure functions is that we can test the *logic* (season boundaries, name
canonicalization, result derivation) without needing network access or a
Kaggle token in CI.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.validate import (
    assign_season,
    canonicalize_teams,
    check_result_consistency,
    derive_result,
    find_near_duplicate_team_names,
    load_openfootball_matches,
    merge_sources,
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


class TestLoadOpenfootballMatches:
    """Regression tests for a real data-quality quirk found in the live
    2025-26 openfootball/football.json file: "score" is shaped
    inconsistently between matches - usually {"ft": [...], "ht": [...]},
    but sometimes a bare [home, away] list with no half-time breakdown at
    all. Both must be handled without crashing or silently misreading a
    goal count."""

    def _write_json(self, tmp_path: Path, matches: list[dict]) -> Path:
        f = tmp_path / "matches.json"
        f.write_text(json.dumps({"name": "Test", "matches": matches}), encoding="utf-8")
        return f

    def test_dict_shaped_score_with_half_time(self, tmp_path):
        f = self._write_json(tmp_path, [
            {"date": "2025-08-15", "team1": "Liverpool FC", "team2": "AFC Bournemouth",
             "score": {"ft": [4, 2], "ht": [1, 0]}},
        ])
        result = load_openfootball_matches([f])
        assert result.iloc[0][["FTHG", "FTAG", "HTHG", "HTAG"]].tolist() == [4, 2, 1, 0]

    def test_bare_list_shaped_score_with_no_half_time(self, tmp_path):
        f = self._write_json(tmp_path, [
            {"date": "2025-08-16", "team1": "Aston Villa FC", "team2": "Newcastle United FC",
             "score": [0, 0]},
        ])
        result = load_openfootball_matches([f])
        assert result.iloc[0][["FTHG", "FTAG"]].tolist() == [0, 0]
        assert pd.isna(result.iloc[0]["HTHG"])
        assert pd.isna(result.iloc[0]["HTAG"])

    def test_unplayed_match_has_no_score_key(self, tmp_path):
        f = self._write_json(tmp_path, [
            {"date": "2026-08-28", "team1": "Crystal Palace FC", "team2": "Manchester City FC"},
        ])
        completed = load_openfootball_matches([f], completed_only=True)
        assert len(completed) == 0

        all_matches = load_openfootball_matches([f], completed_only=False)
        assert len(all_matches) == 1
        assert pd.isna(all_matches.iloc[0]["FTHG"])

    def test_mixed_shapes_in_one_file_all_parse(self, tmp_path):
        f = self._write_json(tmp_path, [
            {"date": "2025-08-15", "team1": "A", "team2": "B", "score": {"ft": [1, 0], "ht": [0, 0]}},
            {"date": "2025-08-16", "team1": "C", "team2": "D", "score": [2, 2]},
            {"date": "2025-08-17", "team1": "E", "team2": "F"},
        ])
        completed = load_openfootball_matches([f], completed_only=True)
        assert len(completed) == 2  # the unplayed E vs F match is excluded


class TestMergeSources:
    """Regression tests for a real bug this merge logic had and was fixed
    for: an earlier version picked "whichever source is later in the
    list, for every season it covers" - which silently dropped 45 real
    matches each for two specific seasons where the later-listed source
    turned out to be incomplete while the earlier one was not. The fix
    picks per-season, by actual match count, not by source recency."""

    def _season_frame(self, season: str, n_matches: int) -> pd.DataFrame:
        return pd.DataFrame({
            "Date": pd.date_range("2020-08-01", periods=n_matches, freq="7D"),
            "HomeTeam": [f"Team{i}" for i in range(n_matches)],
            "AwayTeam": [f"Opp{i}" for i in range(n_matches)],
            "Season": [season] * n_matches,
        })

    def test_more_complete_earlier_source_wins_over_incomplete_later_source(self):
        # This is exactly the real-world scenario found in practice: the
        # later (generally-preferred) source is missing matches for this
        # specific season; the earlier source has it complete.
        earlier_source = self._season_frame("2003-04", n_matches=380)
        later_source = self._season_frame("2003-04", n_matches=335)
        merged = merge_sources([earlier_source, later_source])
        assert len(merged) == 380

    def test_later_source_wins_when_more_complete(self):
        earlier_source = self._season_frame("2021-22", n_matches=309)  # truncated
        later_source = self._season_frame("2021-22", n_matches=380)    # complete
        merged = merge_sources([earlier_source, later_source])
        assert len(merged) == 380

    def test_later_source_breaks_an_exact_tie(self):
        earlier_source = self._season_frame("2010-11", n_matches=380)
        later_source = self._season_frame("2010-11", n_matches=380)
        later_source["HomeTeam"] = "MARKER_" + later_source["HomeTeam"]
        merged = merge_sources([earlier_source, later_source])
        assert merged["HomeTeam"].str.startswith("MARKER_").all()

    def test_non_overlapping_seasons_all_kept(self):
        source_a = self._season_frame("1998-99", n_matches=380)
        source_b = self._season_frame("2010-11", n_matches=380)
        merged = merge_sources([source_a, source_b])
        assert len(merged) == 760
        assert set(merged["Season"].unique()) == {"1998-99", "2010-11"}


class TestFindNearDuplicateTeamNames:
    """Regression tests for a real bug: adding a new data source
    introduced "Leicester City FC" and "Southampton FC" as team names
    distinct from the already-canonical "Leicester City" and
    "Southampton", silently splitting each club's history in two until
    this check caught it."""

    def test_flags_fc_suffix_variant(self):
        found = find_near_duplicate_team_names(["Southampton", "Southampton FC", "Arsenal"])
        assert ("Southampton FC", "Southampton") in found

    def test_flags_afc_suffix_variant(self):
        found = find_near_duplicate_team_names(["Bournemouth", "Bournemouth AFC"])
        assert ("Bournemouth AFC", "Bournemouth") in found

    def test_no_false_positive_when_base_name_absent(self):
        # "AFC Bournemouth" is this project's actual canonical name (the
        # "AFC" is a genuine part of the club's name, not a suffix
        # variant) - there is no bare "Bournemouth" entry to flag against.
        found = find_near_duplicate_team_names(["AFC Bournemouth", "Arsenal", "Chelsea"])
        assert found == []

    def test_no_false_positive_for_unrelated_teams(self):
        found = find_near_duplicate_team_names(["Arsenal", "Chelsea", "Liverpool"])
        assert found == []
