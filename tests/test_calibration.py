"""Unit tests for src/evaluation/calibration.py."""

import numpy as np
import pytest

from src.evaluation.calibration import calibration_curve, expected_calibration_error


class TestCalibrationCurve:
    def test_returns_expected_columns(self):
        y = [1, 0, 1, 0]
        p = [0.9, 0.1, 0.8, 0.2]
        curve = calibration_curve(y, p, n_bins=10)
        assert {"bin", "mean_predicted", "observed_frequency", "count"} <= set(curve.columns)

    def test_empty_bins_are_dropped(self):
        # All predictions cluster inside a single bin, (0.9, 1.0] -
        # every other one of the 10 bins should have no rows and not
        # appear in the output at all.
        y = [1, 1, 0, 1]
        p = [0.91, 0.93, 0.95, 0.99]
        curve = calibration_curve(y, p, n_bins=10)
        assert len(curve) == 1

    def test_bin_counts_sum_to_total_matches(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 1, 200)
        y = (rng.uniform(0, 1, 200) < p).astype(int)
        curve = calibration_curve(y, p, n_bins=10)
        assert curve["count"].sum() == 200


class TestExpectedCalibrationError:
    def test_perfectly_calibrated_predictions_have_near_zero_ece(self):
        # Construct predictions where, within each bin, the actual outcome
        # rate exactly matches the predicted probability by design.
        rng = np.random.default_rng(0)
        n = 20000
        p = rng.uniform(0, 1, n)
        y = (rng.uniform(0, 1, n) < p).astype(int)  # outcome occurs with probability exactly p
        ece = expected_calibration_error(y, p, n_bins=10)
        assert ece < 0.02  # small residual sampling noise, not zero, but close

    def test_systematically_overconfident_predictions_have_high_ece(self):
        # Model always says 90% confident, but the true rate is only 50%.
        n = 5000  # large n keeps sampling noise well under the tolerance below
        p = [0.9] * n
        rng = np.random.default_rng(0)
        y = (rng.uniform(0, 1, n) < 0.5).astype(int)
        ece = expected_calibration_error(y, p, n_bins=10)
        assert ece == pytest.approx(0.4, abs=0.03)

    def test_zero_for_a_single_perfect_bin(self):
        y = [1, 1, 1, 1]
        p = [1.0, 1.0, 1.0, 1.0]
        assert expected_calibration_error(y, p) == pytest.approx(0.0)
