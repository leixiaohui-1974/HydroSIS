"""Unit tests for timeseries validation."""
import pytest
import numpy as np
import pandas as pd
from hydrosis.validation.timeseries import (
    TimeSeriesCriteria,
    validate_time_series,
    validate_multiple_series,
)


class TestTimeSeriesCriteria:
    """Test TimeSeriesCriteria dataclass."""

    def test_default_initialization(self):
        """Test default criteria initialization."""
        criteria = TimeSeriesCriteria()
        assert criteria.max_missing_ratio == 0.1
        assert criteria.max_consecutive_missing == 24
        assert criteria.min_value == 0.0
        assert criteria.max_value == 1e6
        assert criteria.max_hourly_change_ratio == 0.5
        assert criteria.max_daily_change_ratio == 2.0
        assert criteria.allow_negative_trend is True

    def test_custom_initialization(self):
        """Test custom criteria initialization."""
        criteria = TimeSeriesCriteria(
            max_missing_ratio=0.05,
            max_consecutive_missing=12,
            min_value=-10.0,
            max_value=1000.0,
        )
        assert criteria.max_missing_ratio == 0.05
        assert criteria.max_consecutive_missing == 12
        assert criteria.min_value == -10.0
        assert criteria.max_value == 1000.0

    def test_from_dict(self):
        """Test creating criteria from dictionary."""
        data = {
            "max_missing_ratio": 0.15,
            "max_consecutive_missing": 48,
            "min_value": 0.0,
            "max_value": 5000.0,
        }
        criteria = TimeSeriesCriteria.from_dict(data)
        assert criteria.max_missing_ratio == 0.15
        assert criteria.max_consecutive_missing == 48
        assert criteria.min_value == 0.0
        assert criteria.max_value == 5000.0

    def test_from_dict_partial(self):
        """Test from_dict with partial data."""
        data = {"max_missing_ratio": 0.2}
        criteria = TimeSeriesCriteria.from_dict(data)
        assert criteria.max_missing_ratio == 0.2
        # Other values should be defaults
        assert criteria.max_consecutive_missing == 24


class TestValidateTimeSeries:
    """Test validate_time_series function."""

    def test_perfect_series(self):
        """Test validation of perfect series."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = validate_time_series(series)

        assert result.is_valid
        assert result.metrics['missing_count'] == 0
        assert result.metrics['missing_ratio'] == 0.0
        assert result.metrics['min_value'] == 1.0
        assert result.metrics['max_value'] == 5.0
        assert len(result.errors) == 0

    def test_series_with_no_missing(self):
        """Test series with no missing values."""
        series = pd.Series(range(100))
        result = validate_time_series(series)

        assert result.metrics['missing_count'] == 0
        assert result.metrics['missing_ratio'] == 0.0
        assert result.is_valid

    def test_series_with_few_missing(self):
        """Test series with acceptable missing values."""
        data = [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        series = pd.Series(data)
        criteria = TimeSeriesCriteria(max_missing_ratio=0.2)

        result = validate_time_series(series, criteria)

        assert result.metrics['missing_count'] == 1
        assert result.metrics['missing_ratio'] == 0.1
        assert result.is_valid

    def test_series_with_too_many_missing(self):
        """Test series with too many missing values."""
        data = [1.0, np.nan, np.nan, 4.0, np.nan, np.nan, 7.0, 8.0, 9.0, 10.0]
        series = pd.Series(data)
        criteria = TimeSeriesCriteria(max_missing_ratio=0.2)

        result = validate_time_series(series, criteria)

        assert result.metrics['missing_count'] == 4
        assert result.metrics['missing_ratio'] == 0.4
        assert not result.is_valid
        assert any("缺失率过高" in err for err in result.errors)

    def test_consecutive_missing_acceptable(self):
        """Test series with acceptable consecutive missing."""
        data = [1.0, 2.0, np.nan, np.nan, np.nan, 6.0, 7.0, 8.0, 9.0, 10.0]
        series = pd.Series(data)
        criteria = TimeSeriesCriteria(
            max_missing_ratio=0.5,
            max_consecutive_missing=5
        )

        result = validate_time_series(series, criteria)

        # Should pass - 3 consecutive is less than limit of 5
        assert result.is_valid

    def test_consecutive_missing_too_many(self):
        """Test series with too many consecutive missing."""
        data = [1.0, 2.0] + [np.nan] * 10 + [13.0, 14.0, 15.0]
        series = pd.Series(data)
        criteria = TimeSeriesCriteria(
            max_missing_ratio=0.8,
            max_consecutive_missing=5
        )

        result = validate_time_series(series, criteria)

        # Should have warning about consecutive missing
        assert any("连续缺失" in str(w) for w in result.warnings)

    def test_negative_values_error(self):
        """Test that negative values trigger error."""
        series = pd.Series([1.0, 2.0, -5.0, 4.0, 5.0])
        criteria = TimeSeriesCriteria(min_value=0.0)

        result = validate_time_series(series, criteria)

        assert not result.is_valid
        assert any("负值" in err for err in result.errors)
        assert result.metrics['min_value'] == -5.0

    def test_negative_values_allowed(self):
        """Test that negative values are allowed with proper criteria."""
        series = pd.Series([1.0, 2.0, -5.0, 4.0, 5.0])
        criteria = TimeSeriesCriteria(min_value=-10.0)

        result = validate_time_series(series, criteria)

        assert result.is_valid
        assert result.metrics['min_value'] == -5.0

    def test_very_high_values_warning(self):
        """Test that very high values trigger warning."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 10000.0])
        criteria = TimeSeriesCriteria(max_value=1000.0)

        result = validate_time_series(series, criteria)

        assert any("异常高值" in w for w in result.warnings)
        assert result.metrics['max_value'] == 10000.0

    def test_large_hourly_change_warning(self):
        """Test large hourly changes trigger warning."""
        # Jump from 10 to 100 (90% of max change)
        series = pd.Series([10.0, 100.0, 20.0, 30.0, 40.0])
        criteria = TimeSeriesCriteria(max_hourly_change_ratio=0.5)

        result = validate_time_series(series, criteria)

        # Should have warning about large change rate
        assert any("变化率" in str(w) for w in result.warnings)

    def test_acceptable_hourly_change(self):
        """Test acceptable hourly changes pass."""
        series = pd.Series([10.0, 15.0, 20.0, 25.0, 30.0])
        criteria = TimeSeriesCriteria(max_hourly_change_ratio=0.5)

        result = validate_time_series(series, criteria)

        # No warnings about change rate
        assert not any("变化率" in str(w) for w in result.warnings)

    def test_negative_trend_warning(self):
        """Test negative trend triggers warning when not allowed."""
        # Decreasing series
        series = pd.Series([100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0, 5.0])
        criteria = TimeSeriesCriteria(allow_negative_trend=False)

        result = validate_time_series(series, criteria)

        assert any("负趋势" in w for w in result.warnings)
        assert result.metrics['trend_slope'] < 0

    def test_negative_trend_allowed(self):
        """Test negative trend is allowed when configured."""
        series = pd.Series([100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0, 5.0])
        criteria = TimeSeriesCriteria(allow_negative_trend=True)

        result = validate_time_series(series, criteria)

        # Should not have warning about negative trend
        assert not any("负趋势" in w for w in result.warnings)

    def test_positive_trend(self):
        """Test positive trend."""
        series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0])
        result = validate_time_series(series)

        assert result.metrics['trend_slope'] > 0
        # Positive trend should not trigger warnings
        assert len(result.warnings) == 0

    def test_statistics_calculation(self):
        """Test that statistics are calculated correctly."""
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        series = pd.Series(data)
        result = validate_time_series(series)

        assert result.metrics['min_value'] == 10.0
        assert result.metrics['max_value'] == 50.0
        assert result.metrics['mean_value'] == 30.0
        assert abs(result.metrics['std_value'] - np.std(data, ddof=1)) < 0.01

    def test_empty_series(self):
        """Test validation of empty series."""
        series = pd.Series([])
        result = validate_time_series(series)

        # Empty series has NaN missing ratio, but is considered valid by default
        # since it has 0 missing count
        assert result.metrics['missing_count'] == 0

    def test_all_nan_series(self):
        """Test validation of all NaN series."""
        series = pd.Series([np.nan, np.nan, np.nan, np.nan])
        result = validate_time_series(series)

        assert result.metrics['missing_ratio'] == 1.0
        assert not result.is_valid

    def test_series_name_in_result(self):
        """Test that series name appears in result."""
        series = pd.Series([1.0, 2.0, 3.0])
        result = validate_time_series(series, series_name="Temperature")

        assert "Temperature" in result.step_name

    def test_custom_step_name(self):
        """Test custom step name."""
        series = pd.Series([1.0, 2.0, 3.0])
        result = validate_time_series(
            series,
            series_name="Discharge",
            step_name="Custom Step"
        )

        assert "Custom Step" in result.step_name
        assert "Discharge" in result.step_name


class TestValidateMultipleSeries:
    """Test validate_multiple_series function."""

    def test_single_column_dataframe(self):
        """Test validation of single column DataFrame."""
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        results = validate_multiple_series(df)

        assert len(results) == 1
        assert 'A' in results
        assert results['A'].is_valid

    def test_multiple_columns_dataframe(self):
        """Test validation of multiple column DataFrame."""
        df = pd.DataFrame({
            'col1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'col2': [10.0, 20.0, 30.0, 40.0, 50.0],
            'col3': [0.1, 0.2, 0.3, 0.4, 0.5],
        })
        results = validate_multiple_series(df)

        assert len(results) == 3
        assert 'col1' in results
        assert 'col2' in results
        assert 'col3' in results
        assert all(r.is_valid for r in results.values())

    def test_mixed_quality_series(self):
        """Test DataFrame with mixed quality series."""
        df = pd.DataFrame({
            'good': [1.0, 2.0, 3.0, 4.0, 5.0],
            'missing': [1.0, np.nan, np.nan, np.nan, 5.0],
            'negative': [1.0, 2.0, -10.0, 4.0, 5.0],
        })
        criteria = TimeSeriesCriteria(max_missing_ratio=0.2)
        results = validate_multiple_series(df, criteria)

        assert results['good'].is_valid
        assert not results['missing'].is_valid  # Too many missing
        assert not results['negative'].is_valid  # Negative values

    def test_custom_criteria(self):
        """Test multiple series with custom criteria."""
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0, 4.0, 5.0],
            'B': [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        criteria = TimeSeriesCriteria(max_value=25.0)
        results = validate_multiple_series(df, criteria)

        assert results['A'].is_valid
        # B has values > 25, should have warnings
        assert len(results['B'].warnings) > 0

    def test_empty_dataframe(self):
        """Test validation of empty DataFrame."""
        df = pd.DataFrame()
        results = validate_multiple_series(df)

        assert len(results) == 0

    def test_results_are_independent(self):
        """Test that results for each series are independent."""
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0],
            'B': [10.0, 20.0, 30.0],
        })
        results = validate_multiple_series(df)

        # Modifying one result should not affect the other
        results['A'].metrics['test'] = 'modified'
        assert 'test' not in results['B'].metrics


class TestTimeSeriesIntegration:
    """Integration tests for timeseries validation."""

    def test_realistic_discharge_series(self):
        """Test validation of realistic discharge series."""
        # Simulate discharge hydrograph
        dates = pd.date_range('2020-01-01', periods=100, freq='H')
        discharge = np.concatenate([
            np.linspace(10, 100, 20),  # Rising limb
            np.linspace(100, 50, 30),  # Falling limb
            np.linspace(50, 20, 50),   # Baseflow recession
        ])
        series = pd.Series(discharge, index=dates)

        result = validate_time_series(series, series_name="Discharge")

        assert result.is_valid
        assert result.metrics['min_value'] >= 0
        assert result.metrics['missing_count'] == 0

    def test_precipitation_with_zeros(self):
        """Test validation of precipitation series with many zeros."""
        # Precipitation has many zeros (dry periods)
        precip = [0.0] * 50 + [5.0, 10.0, 15.0, 20.0, 10.0] + [0.0] * 45
        series = pd.Series(precip)

        result = validate_time_series(series, series_name="Precipitation")

        assert result.is_valid
        assert result.metrics['min_value'] == 0.0
        assert result.metrics['max_value'] == 20.0

    def test_temperature_series(self):
        """Test validation of temperature series (allows negatives)."""
        # Temperature can be negative
        temp = [-5.0, -3.0, 0.0, 5.0, 10.0, 15.0, 20.0, 18.0, 12.0, 5.0, -2.0]
        series = pd.Series(temp)
        criteria = TimeSeriesCriteria(min_value=-50.0, max_value=50.0)

        result = validate_time_series(series, criteria, series_name="Temperature")

        assert result.is_valid
        assert result.metrics['min_value'] == -5.0
        assert result.metrics['max_value'] == 20.0

    def test_data_quality_assessment(self):
        """Test comprehensive data quality assessment."""
        # Create series with various issues
        data = [10.0, 12.0, np.nan, 15.0, 20.0, np.nan, np.nan, 25.0, 30.0, 28.0]
        series = pd.Series(data)
        criteria = TimeSeriesCriteria(
            max_missing_ratio=0.4,
            max_consecutive_missing=1,  # Lower limit to trigger warning
            max_hourly_change_ratio=0.3
        )

        result = validate_time_series(series, criteria)

        # Should have warnings but still be valid
        assert result.is_valid
        assert result.metrics['missing_count'] == 3
        assert len(result.warnings) > 0  # Consecutive missing warning

    def test_multi_station_validation(self):
        """Test validation of multiple station data."""
        df = pd.DataFrame({
            'Station_A': np.random.uniform(10, 100, 50),
            'Station_B': np.random.uniform(5, 50, 50),
            'Station_C': np.random.uniform(20, 200, 50),
        })

        results = validate_multiple_series(df)

        assert len(results) == 3
        for station, result in results.items():
            assert result.is_valid
            assert 'min_value' in result.metrics
            assert 'max_value' in result.metrics
            assert 'mean_value' in result.metrics

    def test_long_term_series(self):
        """Test validation of long-term series."""
        # 1 year of hourly data
        n_hours = 365 * 24
        series = pd.Series(np.random.uniform(0, 100, n_hours))

        result = validate_time_series(series)

        assert result.is_valid
        assert result.metrics['missing_count'] == 0
        assert 'trend_slope' in result.metrics

    def test_series_with_outliers(self):
        """Test detection of outliers via large changes."""
        data = [10.0, 12.0, 11.0, 13.0, 500.0, 14.0, 12.0, 11.0]  # 500 is outlier
        series = pd.Series(data)
        criteria = TimeSeriesCriteria(max_hourly_change_ratio=0.5)

        result = validate_time_series(series, criteria)

        # Large jump should trigger warning
        assert len(result.warnings) > 0
