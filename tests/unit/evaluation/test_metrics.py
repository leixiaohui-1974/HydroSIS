"""Unit tests for evaluation metrics."""
import pytest
import math
from hydrosis.evaluation.metrics import (
    rmse,
    mae,
    percent_bias,
    nash_sutcliffe_efficiency,
    log_nash_sutcliffe_efficiency,
    kling_gupta_efficiency,
    pearson_correlation,
    available_metrics,
    DEFAULT_METRICS,
    DEFAULT_ORIENTATION,
)


class TestRMSE:
    """Test root-mean-square error metric."""

    def test_perfect_match(self):
        """Test RMSE with perfect match."""
        simulated = [1.0, 2.0, 3.0, 4.0, 5.0]
        observed = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert rmse(simulated, observed) == 0.0

    def test_constant_error(self):
        """Test RMSE with constant error."""
        simulated = [2.0, 3.0, 4.0, 5.0, 6.0]
        observed = [1.0, 2.0, 3.0, 4.0, 5.0]
        # Error is consistently 1.0
        assert rmse(simulated, observed) == 1.0

    def test_known_value(self):
        """Test RMSE with known calculated value."""
        simulated = [2.0, 4.0, 6.0]
        observed = [1.0, 3.0, 5.0]
        # Squared errors: [1, 1, 1], mean=1, sqrt=1
        assert rmse(simulated, observed) == 1.0

    def test_larger_differences(self):
        """Test RMSE with larger differences."""
        simulated = [10.0, 20.0, 30.0]
        observed = [0.0, 0.0, 0.0]
        # Squared errors: [100, 400, 900], mean=1400/3
        expected = math.sqrt(1400.0 / 3.0)
        assert abs(rmse(simulated, observed) - expected) < 1e-9

    def test_empty_series(self):
        """Test RMSE with empty series."""
        assert rmse([], []) == 0.0

    def test_single_value(self):
        """Test RMSE with single value."""
        assert rmse([5.0], [3.0]) == 2.0

    def test_negative_values(self):
        """Test RMSE with negative values."""
        simulated = [-1.0, -2.0, -3.0]
        observed = [-2.0, -3.0, -4.0]
        assert rmse(simulated, observed) == 1.0

    def test_mixed_positive_negative(self):
        """Test RMSE with mixed positive and negative values."""
        simulated = [1.0, -1.0, 2.0, -2.0]
        observed = [2.0, -2.0, 3.0, -3.0]
        assert rmse(simulated, observed) == 1.0

    def test_length_mismatch_raises(self):
        """Test that length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="same length"):
            rmse([1.0, 2.0, 3.0], [1.0, 2.0])


class TestMAE:
    """Test mean absolute error metric."""

    def test_perfect_match(self):
        """Test MAE with perfect match."""
        simulated = [1.0, 2.0, 3.0, 4.0, 5.0]
        observed = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert mae(simulated, observed) == 0.0

    def test_constant_error(self):
        """Test MAE with constant error."""
        simulated = [2.0, 3.0, 4.0, 5.0, 6.0]
        observed = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert mae(simulated, observed) == 1.0

    def test_known_value(self):
        """Test MAE with known calculated value."""
        simulated = [5.0, 10.0, 15.0]
        observed = [3.0, 8.0, 14.0]
        # Absolute errors: [2, 2, 1], mean = 5/3
        expected = 5.0 / 3.0
        assert abs(mae(simulated, observed) - expected) < 1e-9

    def test_symmetric_errors(self):
        """Test MAE with symmetric positive and negative errors."""
        simulated = [2.0, 1.0, 3.0, 0.0]
        observed = [1.0, 2.0, 2.0, 1.0]
        # Absolute errors: [1, 1, 1, 1]
        assert mae(simulated, observed) == 1.0

    def test_empty_series(self):
        """Test MAE with empty series."""
        assert mae([], []) == 0.0

    def test_single_value(self):
        """Test MAE with single value."""
        assert mae([5.0], [2.0]) == 3.0

    def test_negative_values(self):
        """Test MAE with negative values."""
        simulated = [-5.0, -10.0, -15.0]
        observed = [-3.0, -8.0, -14.0]
        expected = 5.0 / 3.0
        assert abs(mae(simulated, observed) - expected) < 1e-9

    def test_length_mismatch_raises(self):
        """Test that length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="same length"):
            mae([1.0, 2.0], [1.0, 2.0, 3.0])


class TestPercentBias:
    """Test percent bias metric."""

    def test_perfect_match(self):
        """Test percent bias with perfect match."""
        simulated = [1.0, 2.0, 3.0, 4.0, 5.0]
        observed = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert percent_bias(simulated, observed) == 0.0

    def test_overestimation(self):
        """Test percent bias with overestimation."""
        simulated = [11.0, 12.0, 13.0, 14.0, 15.0]  # sum = 65
        observed = [10.0, 10.0, 10.0, 10.0, 10.0]  # sum = 50
        # bias = 100 * (65 - 50) / 50 = 30%
        assert percent_bias(simulated, observed) == 30.0

    def test_underestimation(self):
        """Test percent bias with underestimation."""
        simulated = [9.0, 8.0, 7.0, 6.0, 5.0]  # sum = 35
        observed = [10.0, 10.0, 10.0, 10.0, 10.0]  # sum = 50
        # bias = 100 * (35 - 50) / 50 = -30%
        assert percent_bias(simulated, observed) == -30.0

    def test_double_overestimation(self):
        """Test percent bias with 100% overestimation."""
        simulated = [20.0, 20.0]  # sum = 40
        observed = [10.0, 10.0]  # sum = 20
        # bias = 100 * (40 - 20) / 20 = 100%
        assert percent_bias(simulated, observed) == 100.0

    def test_zero_observed_zero_simulated(self):
        """Test percent bias with both zero."""
        simulated = [0.0, 0.0, 0.0]
        observed = [0.0, 0.0, 0.0]
        assert percent_bias(simulated, observed) == 0.0

    def test_zero_observed_positive_simulated(self):
        """Test percent bias with zero observed and positive simulated."""
        simulated = [1.0, 2.0, 3.0]
        observed = [0.0, 0.0, 0.0]
        assert percent_bias(simulated, observed) == float('inf')

    def test_zero_observed_negative_simulated(self):
        """Test percent bias with zero observed and negative simulated."""
        simulated = [-1.0, -2.0, -3.0]
        observed = [0.0, 0.0, 0.0]
        assert percent_bias(simulated, observed) == float('-inf')

    def test_negative_values(self):
        """Test percent bias with negative values."""
        simulated = [-20.0, -30.0]  # sum = -50
        observed = [-10.0, -15.0]  # sum = -25
        # bias = 100 * (-50 - (-25)) / -25 = 100 * (-25) / -25 = 100%
        assert percent_bias(simulated, observed) == 100.0

    def test_length_mismatch_raises(self):
        """Test that length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="same length"):
            percent_bias([1.0], [1.0, 2.0])


class TestNashSutcliffeEfficiency:
    """Test Nash-Sutcliffe efficiency metric."""

    def test_perfect_match(self):
        """Test NSE with perfect match."""
        simulated = [1.0, 2.0, 3.0, 4.0, 5.0]
        observed = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert nash_sutcliffe_efficiency(simulated, observed) == 1.0

    def test_mean_prediction(self):
        """Test NSE when simulated equals mean of observed (NSE = 0)."""
        observed = [1.0, 2.0, 3.0, 4.0, 5.0]  # mean = 3.0
        simulated = [3.0, 3.0, 3.0, 3.0, 3.0]
        nse = nash_sutcliffe_efficiency(simulated, observed)
        assert abs(nse - 0.0) < 1e-9

    def test_known_value_positive(self):
        """Test NSE with known positive value."""
        observed = [10.0, 20.0, 30.0, 40.0, 50.0]  # mean = 30
        simulated = [12.0, 18.0, 32.0, 38.0, 50.0]
        # Numerator: (10-12)^2 + (20-18)^2 + (30-32)^2 + (40-38)^2 + (50-50)^2 = 4+4+4+4+0 = 16
        # Denominator: (10-30)^2 + (20-30)^2 + (30-30)^2 + (40-30)^2 + (50-30)^2 = 400+100+0+100+400 = 1000
        # NSE = 1 - 16/1000 = 0.984
        expected = 1.0 - 16.0 / 1000.0
        assert abs(nash_sutcliffe_efficiency(simulated, observed) - expected) < 1e-9

    def test_poor_performance(self):
        """Test NSE with poor performance (negative NSE)."""
        observed = [10.0, 20.0, 30.0, 40.0, 50.0]  # mean = 30
        # Very poor simulation
        simulated = [50.0, 10.0, 50.0, 10.0, 50.0]
        nse = nash_sutcliffe_efficiency(simulated, observed)
        # NSE should be negative for poor performance
        assert nse < 0.0

    def test_constant_observed(self):
        """Test NSE with constant observed values."""
        observed = [5.0, 5.0, 5.0, 5.0, 5.0]
        simulated = [5.0, 5.0, 5.0, 5.0, 5.0]
        # Denominator is zero, should return 1.0
        assert nash_sutcliffe_efficiency(simulated, observed) == 1.0

    def test_constant_observed_different_simulated(self):
        """Test NSE with constant observed and different simulated."""
        observed = [5.0, 5.0, 5.0, 5.0, 5.0]
        simulated = [3.0, 4.0, 6.0, 7.0, 5.0]
        # Denominator is zero, should return 1.0
        assert nash_sutcliffe_efficiency(simulated, observed) == 1.0

    def test_empty_series(self):
        """Test NSE with empty series."""
        assert nash_sutcliffe_efficiency([], []) == 1.0

    def test_single_value_match(self):
        """Test NSE with single matching value."""
        assert nash_sutcliffe_efficiency([5.0], [5.0]) == 1.0

    def test_single_value_mismatch(self):
        """Test NSE with single mismatched value."""
        # Denominator is zero for single value, should return 1.0
        assert nash_sutcliffe_efficiency([5.0], [3.0]) == 1.0

    def test_length_mismatch_raises(self):
        """Test that length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="same length"):
            nash_sutcliffe_efficiency([1.0, 2.0], [1.0])


class TestLogNashSutcliffeEfficiency:
    """Test log Nash-Sutcliffe efficiency metric."""

    def test_perfect_match(self):
        """Test log NSE with perfect match."""
        simulated = [1.0, 2.0, 3.0, 4.0, 5.0]
        observed = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert log_nash_sutcliffe_efficiency(simulated, observed) == 1.0

    def test_known_value(self):
        """Test log NSE with known value."""
        observed = [10.0, 20.0, 30.0]
        simulated = [10.0, 20.0, 30.0]
        assert log_nash_sutcliffe_efficiency(simulated, observed) == 1.0

    def test_low_flow_emphasis(self):
        """Test that log NSE emphasizes low flows."""
        observed = [1.0, 10.0, 100.0]
        # Simulated has same error magnitude (1.0) but at different scales
        sim_low_error = [2.0, 11.0, 101.0]  # Error at low flow
        sim_high_error = [1.0, 10.0, 101.0]  # Error at high flow

        log_nse_low = log_nash_sutcliffe_efficiency(sim_low_error, observed)
        log_nse_high = log_nash_sutcliffe_efficiency(sim_high_error, observed)

        # Error at low flow should have larger impact in log space
        assert log_nse_low < log_nse_high

    def test_with_zeros(self):
        """Test log NSE with zero values (uses epsilon)."""
        simulated = [0.0, 1.0, 2.0, 3.0]
        observed = [0.0, 1.0, 2.0, 3.0]
        # Should handle zeros with epsilon
        result = log_nash_sutcliffe_efficiency(simulated, observed)
        assert result == 1.0

    def test_custom_epsilon(self):
        """Test log NSE with custom epsilon."""
        simulated = [0.0, 0.0, 0.0]
        observed = [0.0, 0.0, 0.0]
        result = log_nash_sutcliffe_efficiency(simulated, observed, epsilon=1e-3)
        assert result == 1.0

    def test_empty_series(self):
        """Test log NSE with empty series."""
        assert log_nash_sutcliffe_efficiency([], []) == 1.0

    def test_length_mismatch_raises(self):
        """Test that length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="same length"):
            log_nash_sutcliffe_efficiency([1.0, 2.0, 3.0], [1.0, 2.0])


class TestKlingGuptaEfficiency:
    """Test Kling-Gupta efficiency metric."""

    def test_perfect_match(self):
        """Test KGE with perfect match."""
        simulated = [1.0, 2.0, 3.0, 4.0, 5.0]
        observed = [1.0, 2.0, 3.0, 4.0, 5.0]
        # r=1, alpha=1, beta=1, KGE = 1 - sqrt(0) = 1
        assert abs(kling_gupta_efficiency(simulated, observed) - 1.0) < 1e-9

    def test_known_value(self):
        """Test KGE with known calculated value."""
        observed = [1.0, 2.0, 3.0, 4.0, 5.0]
        simulated = [1.0, 2.0, 3.0, 4.0, 5.0]
        kge = kling_gupta_efficiency(simulated, observed)
        assert abs(kge - 1.0) < 1e-9

    def test_constant_bias(self):
        """Test KGE with constant bias (affects beta)."""
        observed = [1.0, 2.0, 3.0, 4.0, 5.0]  # mean = 3
        simulated = [2.0, 3.0, 4.0, 5.0, 6.0]  # mean = 4
        # beta = 4/3 ≠ 1, so KGE < 1
        kge = kling_gupta_efficiency(simulated, observed)
        assert kge < 1.0

    def test_variability_difference(self):
        """Test KGE with different variability (affects alpha)."""
        observed = [1.0, 2.0, 3.0, 4.0, 5.0]
        simulated = [2.5, 2.5, 3.0, 3.5, 3.5]  # Less variability
        # alpha ≠ 1, so KGE < 1
        kge = kling_gupta_efficiency(simulated, observed)
        assert kge < 1.0

    def test_poor_correlation(self):
        """Test KGE with poor correlation."""
        observed = [1.0, 2.0, 3.0, 4.0, 5.0]
        simulated = [5.0, 4.0, 3.0, 2.0, 1.0]  # Negative correlation
        kge = kling_gupta_efficiency(simulated, observed)
        # Should have low or negative KGE
        assert kge < 0.5

    def test_zero_variance_observed(self):
        """Test KGE with zero variance in observed."""
        observed = [3.0, 3.0, 3.0, 3.0, 3.0]
        simulated = [3.0, 3.0, 3.0, 3.0, 3.0]
        kge = kling_gupta_efficiency(simulated, observed)
        assert kge == 1.0

    def test_zero_variance_observed_different_simulated(self):
        """Test KGE with zero variance in observed, different simulated."""
        observed = [3.0, 3.0, 3.0, 3.0, 3.0]
        simulated = [1.0, 2.0, 3.0, 4.0, 5.0]
        kge = kling_gupta_efficiency(simulated, observed)
        # alpha = 0 (since obs std = 0), this affects KGE
        assert kge < 1.0

    def test_empty_series(self):
        """Test KGE with empty series."""
        assert kling_gupta_efficiency([], []) == 1.0

    def test_single_value(self):
        """Test KGE with single value."""
        # Single value: r=1, alpha=1, beta=1 (both std=0)
        assert kling_gupta_efficiency([5.0], [5.0]) == 1.0

    def test_length_mismatch_raises(self):
        """Test that length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="same length"):
            kling_gupta_efficiency([1.0], [1.0, 2.0])


class TestPearsonCorrelation:
    """Test Pearson correlation coefficient."""

    def test_perfect_positive_correlation(self):
        """Test correlation with perfect positive correlation."""
        simulated = [1.0, 2.0, 3.0, 4.0, 5.0]
        observed = [2.0, 4.0, 6.0, 8.0, 10.0]  # Perfectly linear
        assert abs(pearson_correlation(simulated, observed) - 1.0) < 1e-9

    def test_perfect_negative_correlation(self):
        """Test correlation with perfect negative correlation."""
        simulated = [1.0, 2.0, 3.0, 4.0, 5.0]
        observed = [5.0, 4.0, 3.0, 2.0, 1.0]
        assert abs(pearson_correlation(simulated, observed) - (-1.0)) < 1e-9

    def test_no_correlation(self):
        """Test correlation with no correlation."""
        simulated = [1.0, 2.0, 1.0, 2.0, 1.0]
        observed = [3.0, 3.0, 3.0, 3.0, 3.0]  # Constant
        # Constant has zero variance
        result = pearson_correlation(simulated, observed)
        assert result == 0.0

    def test_identical_series(self):
        """Test correlation with identical series."""
        simulated = [1.0, 2.0, 3.0, 4.0, 5.0]
        observed = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert pearson_correlation(simulated, observed) == 1.0

    def test_zero_variance_both(self):
        """Test correlation when both series have zero variance."""
        simulated = [5.0, 5.0, 5.0, 5.0]
        observed = [3.0, 3.0, 3.0, 3.0]
        # Both constant, same variance (zero), should return 1.0
        assert pearson_correlation(simulated, observed) == 1.0

    def test_zero_variance_one(self):
        """Test correlation when one series has zero variance."""
        simulated = [1.0, 2.0, 3.0, 4.0, 5.0]
        observed = [3.0, 3.0, 3.0, 3.0, 3.0]
        # One constant, should return 0.0
        assert pearson_correlation(simulated, observed) == 0.0

    def test_known_value(self):
        """Test correlation with known calculated value."""
        simulated = [1.0, 2.0, 3.0]
        observed = [1.0, 3.0, 2.0]
        # Manual calculation would give specific value
        corr = pearson_correlation(simulated, observed)
        assert -1.0 <= corr <= 1.0

    def test_empty_series(self):
        """Test correlation with empty series."""
        assert pearson_correlation([], []) == 1.0

    def test_single_value(self):
        """Test correlation with single value."""
        assert pearson_correlation([5.0], [3.0]) == 1.0

    def test_length_mismatch_raises(self):
        """Test that length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="same length"):
            pearson_correlation([1.0, 2.0], [1.0, 2.0, 3.0])


class TestAvailableMetrics:
    """Test available metrics function."""

    def test_returns_iterable(self):
        """Test that available_metrics returns an iterable."""
        metrics = available_metrics()
        assert hasattr(metrics, '__iter__')

    def test_contains_expected_metrics(self):
        """Test that all expected metrics are available."""
        metrics = list(available_metrics())
        expected = ["rmse", "mae", "pbias", "nse", "log_nse", "kge", "correlation"]
        for metric in expected:
            assert metric in metrics

    def test_matches_default_metrics(self):
        """Test that available_metrics matches DEFAULT_METRICS keys."""
        metrics = set(available_metrics())
        default_keys = set(DEFAULT_METRICS.keys())
        assert metrics == default_keys


class TestDefaultMetrics:
    """Test DEFAULT_METRICS dictionary."""

    def test_all_metrics_callable(self):
        """Test that all metrics in DEFAULT_METRICS are callable."""
        for name, func in DEFAULT_METRICS.items():
            assert callable(func), f"{name} should be callable"

    def test_all_metrics_work(self):
        """Test that all metrics can be called successfully."""
        simulated = [1.0, 2.0, 3.0, 4.0, 5.0]
        observed = [1.0, 2.0, 3.0, 4.0, 5.0]

        for name, func in DEFAULT_METRICS.items():
            result = func(simulated, observed)
            assert isinstance(result, (int, float)), f"{name} should return a number"

    def test_metrics_dictionary_completeness(self):
        """Test that DEFAULT_METRICS contains all expected metrics."""
        expected_metrics = {
            "rmse": rmse,
            "mae": mae,
            "pbias": percent_bias,
            "nse": nash_sutcliffe_efficiency,
            "log_nse": log_nash_sutcliffe_efficiency,
            "kge": kling_gupta_efficiency,
            "correlation": pearson_correlation,
        }
        assert DEFAULT_METRICS == expected_metrics


class TestDefaultOrientation:
    """Test DEFAULT_ORIENTATION dictionary."""

    def test_all_orientations_valid(self):
        """Test that all orientations are valid."""
        valid_orientations = {"min", "max", "minabs"}
        for name, orientation in DEFAULT_ORIENTATION.items():
            assert orientation in valid_orientations, f"{name}: {orientation} not valid"

    def test_error_metrics_minimize(self):
        """Test that error metrics should be minimized."""
        assert DEFAULT_ORIENTATION["rmse"] == "min"
        assert DEFAULT_ORIENTATION["mae"] == "min"

    def test_efficiency_metrics_maximize(self):
        """Test that efficiency metrics should be maximized."""
        assert DEFAULT_ORIENTATION["nse"] == "max"
        assert DEFAULT_ORIENTATION["log_nse"] == "max"
        assert DEFAULT_ORIENTATION["kge"] == "max"
        assert DEFAULT_ORIENTATION["correlation"] == "max"

    def test_bias_minabs(self):
        """Test that bias should minimize absolute value."""
        assert DEFAULT_ORIENTATION["pbias"] == "minabs"

    def test_orientation_completeness(self):
        """Test that all metrics have orientation defined."""
        for metric_name in DEFAULT_METRICS.keys():
            assert metric_name in DEFAULT_ORIENTATION, f"{metric_name} missing orientation"


class TestMetricsIntegration:
    """Integration tests for metrics with realistic data."""

    def test_realistic_hydrograph_comparison(self):
        """Test metrics with realistic hydrograph data."""
        # Simulate a rising and falling hydrograph
        observed = [10.0, 15.0, 25.0, 40.0, 50.0, 45.0, 30.0, 20.0, 12.0, 10.0]
        # Good simulation with slight overestimation
        simulated = [11.0, 16.0, 27.0, 42.0, 52.0, 46.0, 31.0, 21.0, 13.0, 11.0]

        # All metrics should indicate good performance
        assert nash_sutcliffe_efficiency(simulated, observed) > 0.9
        assert kling_gupta_efficiency(simulated, observed) > 0.9
        assert pearson_correlation(simulated, observed) > 0.95
        assert rmse(simulated, observed) < 2.0
        assert mae(simulated, observed) < 1.5
        # Slight positive bias
        assert 0 < percent_bias(simulated, observed) < 10.0

    def test_poor_simulation(self):
        """Test metrics with poor simulation."""
        observed = [10.0, 20.0, 30.0, 40.0, 50.0]
        # Very poor simulation (reversed pattern)
        simulated = [50.0, 40.0, 30.0, 20.0, 10.0]

        # Metrics should indicate poor performance
        assert nash_sutcliffe_efficiency(simulated, observed) < 0.5
        assert kling_gupta_efficiency(simulated, observed) < 0.5
        # Negative correlation
        assert pearson_correlation(simulated, observed) < 0.0

    def test_baseflow_simulation(self):
        """Test metrics emphasizing low flows."""
        # Observed with low baseflow and peak
        observed = [1.0, 1.0, 1.0, 20.0, 2.0, 1.0, 1.0]
        # Good peak, poor baseflow
        sim_poor_baseflow = [0.5, 0.5, 0.5, 20.0, 2.0, 0.5, 0.5]
        # Good baseflow, poor peak
        sim_poor_peak = [1.0, 1.0, 1.0, 15.0, 2.0, 1.0, 1.0]

        # Log NSE should penalize poor baseflow more
        log_nse_poor_base = log_nash_sutcliffe_efficiency(sim_poor_baseflow, observed)
        log_nse_poor_peak = log_nash_sutcliffe_efficiency(sim_poor_peak, observed)

        # Poor baseflow should have lower log NSE
        assert log_nse_poor_base < log_nse_poor_peak

    def test_volume_conservation(self):
        """Test that percent bias reflects volume conservation."""
        observed = [10.0, 20.0, 30.0, 20.0, 10.0]
        obs_sum = sum(observed)  # 90

        # Perfect volume conservation
        simulated_perfect = [8.0, 22.0, 28.0, 22.0, 10.0]  # sum = 90
        assert abs(percent_bias(simulated_perfect, observed)) < 1e-9

        # 20% overestimation
        simulated_over = [12.0, 24.0, 36.0, 24.0, 12.0]  # sum = 108
        assert abs(percent_bias(simulated_over, observed) - 20.0) < 1e-9

        # 20% underestimation
        simulated_under = [8.0, 16.0, 24.0, 16.0, 8.0]  # sum = 72
        assert abs(percent_bias(simulated_under, observed) - (-20.0)) < 1e-9
