"""Unit tests for src/live_forecast.py.

build_current_forecast() itself is an integration/orchestration function -
it fetches live data over the network and reads the full historical
dataset from disk, so it's deliberately not unit-tested here (consistent
with how this project treats other orchestration scripts, e.g.
experiments/run_phase8_simulation_validation.py and run_phase10_tuning.py:
verified by actually running them, not by pytest). One config constant is
still tested directly: the one most likely to cause real damage if
silently "simplified" back to the tuned class default without
understanding why it isn't - see the module's own comment and README
"Hyperparameter tuning (Phase 10)" for the full Hull City story this
guards against.

simulate_historical_cutoff(), unlike build_current_forecast(), operates
entirely on an in-memory `matches` DataFrame passed in as an argument -
no network, no disk reads - so it IS fully unit-testable, and its most
important property (the training set for a historical backtest instance
must never include seasons chronologically after the one being predicted)
gets a direct regression test below.
"""

import pandas as pd
import pytest

import src.live_forecast as live_forecast_module
from src.live_forecast import LIVE_FORECAST_POISSON_CONFIG, simulate_historical_cutoff


class TestLiveForecastPoissonConfig:
    def test_uses_conservative_values_not_the_tuned_class_default(self):
        # DixonColesModel's tuned class default is xi=0.5/promoted_penalty=0.2
        # (see src/models/poisson_model.py) - good for whole-season-ahead
        # backtesting, but confirmed to overreact to small early-season
        # samples when used for partial-season live prediction. This must
        # stay pinned to the more conservative values Phase 8's validation
        # already confirmed sensible for a mid-season cutoff.
        assert LIVE_FORECAST_POISSON_CONFIG["xi"] == 0.3
        assert LIVE_FORECAST_POISSON_CONFIG["promoted_penalty"] == 0.4
        assert LIVE_FORECAST_POISSON_CONFIG["use_correlation"] is True


def _synthetic_multi_season_matches() -> pd.DataFrame:
    teams = ["Alpha", "Bravo", "Charlie", "Delta"]
    season_starts = {
        "2000-01": pd.Timestamp("2000-08-01"),
        "2001-02": pd.Timestamp("2001-08-01"),
        "2002-03": pd.Timestamp("2002-08-01"),
    }
    rows = []
    for season, start in season_starts.items():
        date = start
        for _ in range(10):  # 10 "matchdays", each a full round of home/away fixtures
            for home in teams:
                for away in teams:
                    if home == away:
                        continue
                    rows.append({
                        "Date": date, "Season": season,
                        "HomeTeam": home, "AwayTeam": away,
                        "FTHG": 1, "FTAG": 1, "FTR": "D",
                    })
            date += pd.Timedelta(days=7)
    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)


class _SpyModel:
    """Stands in for DixonColesModel, recording exactly what training data
    it was fit on - the leakage regression test's whole mechanism."""
    captured_train_seasons: list[set[str]] = []

    def __init__(self, **kwargs):
        pass

    def fit(self, train_matches):
        _SpyModel.captured_train_seasons.append(set(train_matches["Season"].unique()))
        return self

    def predict_expected_goals(self, home, away):
        return 1.0, 1.0


class TestSimulateHistoricalCutoff:
    def test_training_data_excludes_seasons_after_the_target(self, monkeypatch):
        _SpyModel.captured_train_seasons = []
        monkeypatch.setattr(live_forecast_module, "DixonColesModel", _SpyModel)

        matches = _synthetic_multi_season_matches()
        cutoff = pd.Timestamp("2001-09-15")  # partway through 2001-02
        simulate_historical_cutoff(matches, season="2001-02", cutoff_date=cutoff, n_simulations=50)

        assert len(_SpyModel.captured_train_seasons) == 1
        trained_on = _SpyModel.captured_train_seasons[0]
        assert "2002-03" not in trained_on, "leaked a season chronologically AFTER the target into training"
        assert "2000-01" in trained_on
        assert "2001-02" in trained_on  # this season's played-so-far matches ARE legitimately included

    def test_played_and_remaining_split_on_cutoff_date(self):
        matches = _synthetic_multi_season_matches()
        cutoff = pd.Timestamp("2001-09-15")
        forecast = simulate_historical_cutoff(matches, season="2001-02", cutoff_date=cutoff, n_simulations=50)

        season_matches = matches[matches["Season"] == "2001-02"]
        expected_played = (season_matches["Date"] < cutoff).sum()
        expected_remaining = (season_matches["Date"] >= cutoff).sum()
        assert forecast.n_played == expected_played
        assert forecast.n_remaining == expected_remaining

    def test_raises_for_unknown_season(self):
        matches = _synthetic_multi_season_matches()
        with pytest.raises(ValueError, match="not found"):
            simulate_historical_cutoff(matches, season="2099-00", cutoff_date=pd.Timestamp("2099-01-01"))

    def test_generated_at_is_timezone_aware(self):
        # Regression test for a real production bug: a naive
        # pd.Timestamp.now() serializes with no timezone marker at all,
        # and a browser's `new Date(...)` interprets a marker-less ISO
        # string as ITS OWN local time rather than UTC - silently skewing
        # the frontend's "updated N ago" display by exactly the visitor's
        # UTC offset (a real deploy showed a permanently-stuck "updated 3
        # hours ago" for a visitor 3 hours off UTC, even seconds after a
        # fresh forecast). See src/live_forecast.py's construction sites
        # for the full story.
        matches = _synthetic_multi_season_matches()
        forecast = simulate_historical_cutoff(
            matches, season="2001-02", cutoff_date=pd.Timestamp("2001-09-15"), n_simulations=50
        )
        assert forecast.generated_at.tzinfo is not None
