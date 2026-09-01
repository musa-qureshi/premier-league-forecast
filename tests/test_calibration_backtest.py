"""Unit tests for src/simulation/calibration_backtest.py.

actual_final_outcomes() and cutoff_date_for_fraction() are pure functions
over an in-memory DataFrame, so they get direct tests against synthetic
data with a known, deterministic final table. run_calibration_backtest()
is the orchestration function tying those together with
simulate_historical_cutoff() (already covered by
tests/test_live_forecast.py's leakage regression test) - it gets one small
end-to-end test with tiny n_simulations/min_train_seasons, checked for
shape and for getting the known winner/loser right, not re-litigating the
leakage guard itself.
"""

from __future__ import annotations

import pandas as pd

from src.simulation.calibration_backtest import (
    actual_final_outcomes,
    cutoff_date_for_fraction,
    run_calibration_backtest,
)

# Strongest to weakest - the higher-ranked (lower-index) team always wins
# 2-0, regardless of home/away, so the final table is fully deterministic
# and known ahead of time: exactly what a test for "did we rank/label the
# real table correctly" needs.
TEAMS = ["Alpha", "Bravo", "Charlie", "Delta"]


def _dominance_season_matches(season: str, start_date: pd.Timestamp) -> pd.DataFrame:
    rows = []
    date = start_date
    for home in TEAMS:
        for away in TEAMS:
            if home == away:
                continue
            home_rank, away_rank = TEAMS.index(home), TEAMS.index(away)
            if home_rank < away_rank:
                fthg, ftag, ftr = 2, 0, "H"
            else:
                fthg, ftag, ftr = 0, 2, "A"
            rows.append({
                "Date": date, "Season": season,
                "HomeTeam": home, "AwayTeam": away,
                "FTHG": fthg, "FTAG": ftag, "FTR": ftr,
            })
            date += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


def _multi_season_dominance_matches() -> pd.DataFrame:
    seasons = {
        "2000-01": pd.Timestamp("2000-08-01"),
        "2001-02": pd.Timestamp("2001-08-01"),
        "2002-03": pd.Timestamp("2002-08-01"),
    }
    frames = [_dominance_season_matches(season, start) for season, start in seasons.items()]
    return pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)


# A 4-team competition config matching TEAMS' size, so "top4"/"relegation"
# thresholds mean something on a league this small: 1 title, 1 more
# European spot, 1 relegated.
SMALL_CONFIG = {"champions_league": 1, "europa_league": 1, "conference_league": 0, "relegation": 1}


class TestActualFinalOutcomes:
    def test_ranks_and_labels_the_known_table_correctly(self):
        matches = _dominance_season_matches("2000-01", pd.Timestamp("2000-08-01"))
        outcomes = actual_final_outcomes(matches, "2000-01", competition_config=SMALL_CONFIG)

        # Alpha beats everyone, Delta loses to everyone - fully deterministic.
        assert outcomes.loc["Alpha", "actual_position"] == 1
        assert outcomes.loc["Delta", "actual_position"] == 4
        assert outcomes.loc["Alpha", "actual_title"] == True  # noqa: E712
        assert outcomes.loc["Bravo", "actual_title"] == False  # noqa: E712

        # champions_league=1 + europa_league=1 -> top 2 are "actual_european".
        assert outcomes.loc["Alpha", "actual_european"] == True  # noqa: E712
        assert outcomes.loc["Bravo", "actual_european"] == True  # noqa: E712
        assert outcomes.loc["Charlie", "actual_european"] == False  # noqa: E712

        # relegation=1 -> only the bottom team (Delta) is relegated.
        assert outcomes.loc["Delta", "actual_relegated"] == True  # noqa: E712
        assert outcomes.loc["Charlie", "actual_relegated"] == False  # noqa: E712


class TestCutoffDateForFraction:
    def test_clamps_to_the_first_and_last_match_date(self):
        dates = pd.date_range("2000-08-01", periods=10, freq="D")
        matches = pd.DataFrame({"Season": "2000-01", "Date": dates})

        assert cutoff_date_for_fraction(matches, "2000-01", 0.0) == dates[0]
        # fraction=1.0 would index past the end without clamping - must
        # land on the LAST date, not raise or roll over.
        assert cutoff_date_for_fraction(matches, "2000-01", 1.0) == dates[-1]

    def test_lands_near_the_requested_fraction(self):
        dates = pd.date_range("2000-08-01", periods=10, freq="D")
        matches = pd.DataFrame({"Season": "2000-01", "Date": dates})

        assert cutoff_date_for_fraction(matches, "2000-01", 0.5) == dates[5]


class TestRunCalibrationBacktest:
    def test_end_to_end_shape_and_known_outcome(self):
        matches = _multi_season_dominance_matches()

        results = run_calibration_backtest(
            matches,
            min_train_seasons=2,  # only "2002-03" is eligible (3 seasons total)
            cutoff_fractions=(0.5,),
            n_simulations=200,
            seed=0,
        )

        # One row per (season, cutoff_fraction, team): 1 season * 1 fraction * 4 teams.
        assert len(results) == 4
        assert set(results["season"].unique()) == {"2002-03"}
        assert set(results["team"]) == set(TEAMS)

        for col in ("predicted_title", "predicted_top4", "predicted_relegation"):
            assert results[col].between(0, 1).all()

        # Alpha has won every match in every prior season and (by the cutoff)
        # every match so far this season too - the model should have picked
        # up on that dominance strongly enough to make it the clear title
        # favourite, and (since relegation here uses run_calibration_backtest's
        # real, competition-size-agnostic DEFAULT_COMPETITION_CONFIG - bottom
        # 3 of what is only a 4-team synthetic league - virtually everyone
        # BUT Alpha is a near-certain "relegation" case) the only team that
        # should reliably escape relegation.
        by_team = results.set_index("team")
        assert by_team["predicted_title"].idxmax() == "Alpha"
        assert by_team.loc["Alpha", "predicted_relegation"] < 0.05
        assert by_team.loc["Delta", "predicted_relegation"] > 0.9

        # And the ground truth those predictions get judged against for
        # this eligible season should show the very same (known) result.
        assert by_team.loc["Alpha", "actual_title"] == 1
        assert by_team.loc["Alpha", "actual_relegation"] == 0
        assert by_team.loc["Delta", "actual_relegation"] == 1
