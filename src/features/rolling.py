"""Rolling-window form features and rest-days, computed per team from their
own match sequence.

Why rolling form on top of Elo? Elo is a slowly-moving, whole-history
signal by design - a big K would make it react faster, but only at the cost
of turning it into noise. Recent form (points per game, goals for/against
over the last 3/5/10 matches) is a deliberately shorter, faster-reacting
signal that can pick up things Elo is slow to reflect: a new manager, an
injury crisis, a run of tough fixtures. Whether these actually add
predictive power beyond Elo alone is an empirical question for Phase 3+
(feature importance / ablation), not assumed here.

Leakage-safety approach: for each stat, compute the rolling aggregate over
the CURRENT row's own last N matches (which necessarily includes today's
match), then shift the result down by one row within each team's group. The
shift is what makes it leakage-safe - it turns "average of my last N
matches including today" into "average of my last N matches before today".
A team's very first match in the dataset has no prior matches at all, so
every rolling feature is NaN there by construction - left as NaN rather
than filled with a guessed default, since imputing it silently would hide
that the model genuinely has no information yet for that case.
"""

from __future__ import annotations

import pandas as pd

from src.features.long_format import to_team_perspective

ROLLING_WINDOWS = (3, 5, 10)

# Columns produced per team-match row that are inputs/bookkeeping, not
# features to attach to the match-level output.
_NON_FEATURE_COLUMNS = {
    "match_id", "Season", "Date", "team", "opponent", "is_home",
    "goals_for", "goals_against", "points", "win", "draw", "loss",
    "prev_match_date",
}


def _shifted_rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Rolling mean over the last `window` matches INCLUDING the current
    row, then shifted down by one - see module docstring for why the shift
    is what makes this leakage-safe rather than a style choice."""
    return series.rolling(window, min_periods=1).mean().shift(1)


def compute_rolling_features(matches: pd.DataFrame, windows: tuple[int, ...] = ROLLING_WINDOWS) -> pd.DataFrame:
    """Returns the long (team-perspective) format with rolling-form and
    rest-day columns added. Exposed separately from attach_rolling_features
    so tests can inspect the long format directly."""
    long = to_team_perspective(matches)
    grouped = long.groupby("team", group_keys=False)

    for w in windows:
        long[f"rolling_gf_{w}"] = grouped["goals_for"].transform(lambda s, w=w: _shifted_rolling_mean(s, w))
        long[f"rolling_ga_{w}"] = grouped["goals_against"].transform(lambda s, w=w: _shifted_rolling_mean(s, w))
        long[f"rolling_ppg_{w}"] = grouped["points"].transform(lambda s, w=w: _shifted_rolling_mean(s, w))
        long[f"rolling_win_rate_{w}"] = grouped["win"].transform(lambda s, w=w: _shifted_rolling_mean(s, w))
        long[f"rolling_draw_rate_{w}"] = grouped["draw"].transform(lambda s, w=w: _shifted_rolling_mean(s, w))

    # Rest days: days since this team's previous match in this dataset.
    # Caveat (documented in README): this dataset is Premier-League-only,
    # so this understates true rest whenever a team also played a midweek
    # cup/European fixture not present here - directionally informative,
    # not exact.
    long["prev_match_date"] = grouped["Date"].transform(lambda s: s.shift(1))
    long["rest_days"] = (long["Date"] - long["prev_match_date"]).dt.days

    return long


def attach_rolling_features(matches: pd.DataFrame, windows: tuple[int, ...] = ROLLING_WINDOWS) -> pd.DataFrame:
    """Merges home_*/away_* rolling-form columns onto the match-level
    dataset, keyed on match_id so each side's own history is attached to
    the correct column prefix."""
    long = compute_rolling_features(matches, windows)
    feature_cols = [c for c in long.columns if c not in _NON_FEATURE_COLUMNS]

    home_feats = long.loc[long["is_home"], ["match_id"] + feature_cols].add_prefix("home_")
    home_feats = home_feats.rename(columns={"home_match_id": "match_id"})
    away_feats = long.loc[~long["is_home"], ["match_id"] + feature_cols].add_prefix("away_")
    away_feats = away_feats.rename(columns={"away_match_id": "match_id"})

    result = matches.merge(home_feats, on="match_id", how="left")
    result = result.merge(away_feats, on="match_id", how="left")

    # Relative ("matchup") features: how a team's typical attacking output
    # compares to the opponent's typical defensive concession, and vice
    # versa - a more direct signal for this specific fixture than either
    # team's rolling stats in isolation.
    for w in windows:
        result[f"attack_vs_defence_{w}"] = result[f"home_rolling_gf_{w}"] - result[f"away_rolling_ga_{w}"]
        result[f"away_attack_vs_home_defence_{w}"] = result[f"away_rolling_gf_{w}"] - result[f"home_rolling_ga_{w}"]
        result[f"ppg_diff_{w}"] = result[f"home_rolling_ppg_{w}"] - result[f"away_rolling_ppg_{w}"]

    result["rest_days_diff"] = result["home_rest_days"] - result["away_rest_days"]

    return result
