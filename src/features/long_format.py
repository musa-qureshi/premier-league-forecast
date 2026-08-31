"""Converts the match-level dataset (one row per fixture, home/away columns
side by side) into a team-level long format (one row per team per fixture,
from that team's own perspective).

This is the natural shape for any "how has this team been doing lately"
feature - rolling form, cumulative season stats, rest days - since those
are properties of a single team's own match sequence, not of a fixture.
Both src/features/rolling.py and (conceptually) the league-table tracker
build on this same representation, so it lives in one shared place rather
than being duplicated.
"""

from __future__ import annotations

import pandas as pd


def to_team_perspective(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, match) - every match appears twice, once from
    each side's perspective. Sorted by team then Date, so a groupby("team")
    on the result processes each team's matches in chronological order."""
    home = pd.DataFrame({
        "match_id": matches["match_id"],
        "Season": matches["Season"],
        "Date": matches["Date"],
        "team": matches["HomeTeam"],
        "opponent": matches["AwayTeam"],
        "is_home": True,
        "goals_for": matches["FTHG"],
        "goals_against": matches["FTAG"],
    })
    away = pd.DataFrame({
        "match_id": matches["match_id"],
        "Season": matches["Season"],
        "Date": matches["Date"],
        "team": matches["AwayTeam"],
        "opponent": matches["HomeTeam"],
        "is_home": False,
        "goals_for": matches["FTAG"],
        "goals_against": matches["FTHG"],
    })
    long = pd.concat([home, away], ignore_index=True)

    long["points"] = 0
    long.loc[long["goals_for"] > long["goals_against"], "points"] = 3
    long.loc[long["goals_for"] == long["goals_against"], "points"] = 1
    long["win"] = (long["goals_for"] > long["goals_against"]).astype(int)
    long["draw"] = (long["goals_for"] == long["goals_against"]).astype(int)
    long["loss"] = (long["goals_for"] < long["goals_against"]).astype(int)

    return long.sort_values(["team", "Date"], kind="stable").reset_index(drop=True)
