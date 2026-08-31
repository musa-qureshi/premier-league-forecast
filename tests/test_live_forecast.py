"""Unit tests for src/live_forecast.py.

build_current_forecast() itself is an integration/orchestration function -
it fetches live data over the network and reads the full historical
dataset from disk, so it's deliberately not unit-tested here (consistent
with how this project treats other orchestration scripts, e.g.
experiments/run_phase8_simulation_validation.py and run_phase10_tuning.py:
verified by actually running them, not by pytest). What IS tested here is
the one config constant most likely to cause real damage if silently
"simplified" back to the tuned class default without understanding why it
isn't - see the module's own comment and README "Hyperparameter tuning
(Phase 10)" for the full Hull City story this guards against.
"""

from src.live_forecast import LIVE_FORECAST_POISSON_CONFIG


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
