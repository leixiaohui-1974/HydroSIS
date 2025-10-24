"""Unit tests for water balance analysis."""
import pytest
import math
from hydrosis.evaluation.water_balance import (
    WaterBalanceResult,
    calculate_water_balance,
    compare_water_balance,
    precip_mmh_to_m3s,
    runoff_m3s_to_mm,
)


class TestWaterBalanceResult:
    """Test WaterBalanceResult dataclass."""

    def test_dataclass_creation(self):
        """Test creating a WaterBalanceResult."""
        result = WaterBalanceResult(
            total_precipitation_mm=100.0,
            basin_area_km2=50.0,
            timestep_hours=1.0,
            total_runoff_mm=60.0,
            total_runoff_volume_m3=3000000.0,
            runoff_coefficient=0.6,
            storage_change_mm=40.0,
            is_balanced=True,
            balance_quality="good",
            warnings=[],
        )
        assert result.total_precipitation_mm == 100.0
        assert result.basin_area_km2 == 50.0
        assert result.runoff_coefficient == 0.6
        assert result.is_balanced is True
        assert result.balance_quality == "good"

    def test_str_representation(self):
        """Test string representation of WaterBalanceResult."""
        result = WaterBalanceResult(
            total_precipitation_mm=100.0,
            basin_area_km2=50.0,
            timestep_hours=1.0,
            total_runoff_mm=60.0,
            total_runoff_volume_m3=3000000.0,
            runoff_coefficient=0.6,
            storage_change_mm=40.0,
            is_balanced=True,
            balance_quality="good",
            warnings=[],
        )
        str_repr = str(result)
        assert "Water Balance Analysis Result" in str_repr
        assert "100.00 mm" in str_repr
        assert "0.6000" in str_repr
        assert "good" in str_repr
        assert "No warnings" in str_repr

    def test_str_with_warnings(self):
        """Test string representation with warnings."""
        result = WaterBalanceResult(
            total_precipitation_mm=100.0,
            basin_area_km2=50.0,
            timestep_hours=1.0,
            total_runoff_mm=95.0,
            total_runoff_volume_m3=4750000.0,
            runoff_coefficient=0.95,
            storage_change_mm=5.0,
            is_balanced=True,
            balance_quality="poor",
            warnings=["Very high runoff coefficient"],
        )
        str_repr = str(result)
        assert "Warnings:" in str_repr
        assert "Very high runoff coefficient" in str_repr


class TestCalculateWaterBalance:
    """Test calculate_water_balance function."""

    def test_basic_calculation(self):
        """Test basic water balance calculation."""
        # Simple case: 100 km², 10mm precip, 50% runoff
        precip = [10.0]  # mm
        area = 100.0  # km²

        # Expected: 5mm runoff = 500,000 m³
        # For 1-hour timestep: Q = 500000 m³ / 3600 s = 138.89 m³/s
        runoff = [138.89]  # m³/s

        result = calculate_water_balance(precip, runoff, area, timestep_hours=1.0)

        assert result.total_precipitation_mm == 10.0
        assert result.basin_area_km2 == 100.0
        assert abs(result.total_runoff_mm - 5.0) < 0.1
        assert abs(result.runoff_coefficient - 0.5) < 0.01

    def test_perfect_runoff(self):
        """Test with 100% runoff (no infiltration/storage)."""
        precip = [20.0, 30.0, 10.0]  # Total: 60 mm
        area = 100.0
        timestep = 1.0

        # 60mm over 100 km² = 6,000,000 m³
        # Over 3 hours: 6,000,000 / (3*3600) = 555.56 m³/s average
        # Distribute proportionally
        total_m3 = 60.0 * area * 1000  # 6,000,000
        runoff = [
            20.0 * area * 1000 / 3600,  # 555.56
            30.0 * area * 1000 / 3600,  # 833.33
            10.0 * area * 1000 / 3600,  # 277.78
        ]

        result = calculate_water_balance(precip, runoff, area, timestep)

        assert abs(result.total_precipitation_mm - 60.0) < 0.1
        assert abs(result.total_runoff_mm - 60.0) < 0.1
        assert abs(result.runoff_coefficient - 1.0) < 0.01
        assert abs(result.storage_change_mm) < 0.1

    def test_zero_precipitation(self):
        """Test with zero precipitation."""
        precip = [0.0, 0.0, 0.0]
        runoff = [0.0, 0.0, 0.0]
        area = 100.0

        result = calculate_water_balance(precip, runoff, area)

        assert result.total_precipitation_mm == 0.0
        assert result.total_runoff_mm == 0.0
        assert result.runoff_coefficient == 0.0

    def test_runoff_coefficient_range_good(self):
        """Test runoff coefficient in good range (0.3-0.7)."""
        precip = [100.0]
        area = 100.0
        # 50% runoff: 5,000,000 m³ / 3600 s = 1388.89 m³/s
        runoff = [1388.89]

        result = calculate_water_balance(precip, runoff, area)

        assert result.balance_quality == "good"
        assert len(result.warnings) == 0

    def test_runoff_coefficient_acceptable(self):
        """Test runoff coefficient in acceptable range."""
        precip = [100.0]
        area = 100.0
        # 25% runoff
        runoff = [694.44]

        result = calculate_water_balance(precip, runoff, area)

        assert result.balance_quality == "acceptable"

    def test_runoff_coefficient_too_high(self):
        """Test very high runoff coefficient (>0.9)."""
        precip = [100.0]
        area = 100.0
        # 95% runoff
        runoff = [2638.89]

        result = calculate_water_balance(precip, runoff, area)

        assert result.balance_quality == "poor"
        assert any("high runoff coefficient" in w.lower() for w in result.warnings)

    def test_runoff_coefficient_exceeds_one(self):
        """Test runoff coefficient > 1.0 (error condition)."""
        precip = [100.0]
        area = 100.0
        # 120% runoff (impossible - indicates error)
        runoff = [3333.33]

        result = calculate_water_balance(precip, runoff, area)

        assert result.balance_quality == "critical"
        assert result.is_balanced is False
        assert any("coefficient > 1.0" in w.lower() for w in result.warnings)

    def test_runoff_coefficient_too_low(self):
        """Test very low runoff coefficient (<0.1)."""
        precip = [100.0]
        area = 100.0
        # 5% runoff
        runoff = [138.89]

        result = calculate_water_balance(precip, runoff, area)

        assert result.balance_quality == "poor"
        assert any("low runoff coefficient" in w.lower() for w in result.warnings)

    def test_large_storage_change(self):
        """Test large storage change warning."""
        precip = [100.0]
        area = 100.0
        # 10% runoff, 90% stored (large change)
        runoff = [277.78]

        result = calculate_water_balance(precip, runoff, area)

        assert any("storage change" in w.lower() for w in result.warnings)

    def test_different_timestep(self):
        """Test with different timestep."""
        precip = [10.0, 10.0]  # Total: 20mm over 2 timesteps
        area = 100.0
        timestep = 2.0  # 2-hour timesteps

        # Total precip volume: 20*2 = 40mm equivalent
        # If 50% runoff: 20mm = 2,000,000 m³
        # Over 2 timesteps of 2 hours each = 4 hours = 14400s
        # Q = 2,000,000 / 14400 = 138.89 m³/s average
        runoff = [138.89, 138.89]

        result = calculate_water_balance(precip, runoff, area, timestep)

        assert result.timestep_hours == 2.0
        # Total precip: 10+10 = 20mm * 2hr = 40mm
        assert abs(result.total_precipitation_mm - 40.0) < 0.1

    def test_length_mismatch_raises(self):
        """Test that mismatched lengths raise error."""
        precip = [10.0, 20.0, 30.0]
        runoff = [100.0, 200.0]
        area = 100.0

        with pytest.raises(ValueError, match="same length"):
            calculate_water_balance(precip, runoff, area)

    def test_negative_area_raises(self):
        """Test that negative area raises error."""
        precip = [10.0]
        runoff = [100.0]

        with pytest.raises(ValueError, match="positive"):
            calculate_water_balance(precip, runoff, basin_area_km2=-50.0)

    def test_zero_area_raises(self):
        """Test that zero area raises error."""
        precip = [10.0]
        runoff = [100.0]

        with pytest.raises(ValueError, match="positive"):
            calculate_water_balance(precip, runoff, basin_area_km2=0.0)

    def test_realistic_scenario(self):
        """Test with realistic hydrograph scenario."""
        # 5-day storm event, hourly timesteps
        precip = [2.0, 5.0, 10.0, 15.0, 8.0, 3.0, 1.0, 0.5] + [0.0] * 16  # mm/hr
        # Runoff response with lag and attenuation
        runoff = [10.0, 30.0, 80.0, 150.0, 100.0, 60.0, 30.0, 15.0] + [5.0] * 16  # m³/s
        area = 50.0  # km²

        result = calculate_water_balance(precip, runoff, area, timestep_hours=1.0)

        # Should have reasonable runoff coefficient
        assert 0.1 < result.runoff_coefficient < 0.9
        assert result.total_precipitation_mm > 0
        assert result.total_runoff_mm > 0


class TestCompareWaterBalance:
    """Test compare_water_balance function."""

    def test_empty_results(self):
        """Test comparison with no results."""
        report = compare_water_balance({})
        assert "No results to compare" in report

    def test_single_model(self):
        """Test comparison with single model."""
        result = WaterBalanceResult(
            total_precipitation_mm=100.0,
            basin_area_km2=100.0,
            timestep_hours=1.0,
            total_runoff_mm=50.0,
            total_runoff_volume_m3=5000000.0,
            runoff_coefficient=0.5,
            storage_change_mm=50.0,
            is_balanced=True,
            balance_quality="good",
            warnings=[],
        )

        report = compare_water_balance({"Model1": result})

        assert "Water Balance Comparison" in report
        assert "Model1" in report
        assert "50.00" in report
        assert "0.5000" in report

    def test_two_models_good_agreement(self):
        """Test comparison with two models in good agreement."""
        result1 = WaterBalanceResult(
            total_precipitation_mm=100.0,
            basin_area_km2=100.0,
            timestep_hours=1.0,
            total_runoff_mm=50.0,
            total_runoff_volume_m3=5000000.0,
            runoff_coefficient=0.50,
            storage_change_mm=50.0,
            is_balanced=True,
            balance_quality="good",
            warnings=[],
        )

        result2 = WaterBalanceResult(
            total_precipitation_mm=100.0,
            basin_area_km2=100.0,
            timestep_hours=1.0,
            total_runoff_mm=52.0,
            total_runoff_volume_m3=5200000.0,
            runoff_coefficient=0.52,
            storage_change_mm=48.0,
            is_balanced=True,
            balance_quality="good",
            warnings=[],
        )

        report = compare_water_balance({"HBV": result1, "VIC": result2})

        assert "HBV" in report
        assert "VIC" in report
        assert "Comparison Analysis" in report
        assert "good agreement" in report
        assert "Δ < 0.1" in report

    def test_two_models_moderate_difference(self):
        """Test comparison with moderate differences."""
        result1 = WaterBalanceResult(
            total_precipitation_mm=100.0,
            basin_area_km2=100.0,
            timestep_hours=1.0,
            total_runoff_mm=40.0,
            total_runoff_volume_m3=4000000.0,
            runoff_coefficient=0.40,
            storage_change_mm=60.0,
            is_balanced=True,
            balance_quality="acceptable",
            warnings=[],
        )

        result2 = WaterBalanceResult(
            total_precipitation_mm=100.0,
            basin_area_km2=100.0,
            timestep_hours=1.0,
            total_runoff_mm=55.0,
            total_runoff_volume_m3=5500000.0,
            runoff_coefficient=0.55,
            storage_change_mm=45.0,
            is_balanced=True,
            balance_quality="good",
            warnings=[],
        )

        report = compare_water_balance({"Model A": result1, "Model B": result2})

        assert "moderate differences" in report
        assert "0.1 < Δ < 0.2" in report

    def test_two_models_large_difference(self):
        """Test comparison with large differences."""
        result1 = WaterBalanceResult(
            total_precipitation_mm=100.0,
            basin_area_km2=100.0,
            timestep_hours=1.0,
            total_runoff_mm=30.0,
            total_runoff_volume_m3=3000000.0,
            runoff_coefficient=0.30,
            storage_change_mm=70.0,
            is_balanced=True,
            balance_quality="acceptable",
            warnings=[],
        )

        result2 = WaterBalanceResult(
            total_precipitation_mm=100.0,
            basin_area_km2=100.0,
            timestep_hours=1.0,
            total_runoff_mm=70.0,
            total_runoff_volume_m3=7000000.0,
            runoff_coefficient=0.70,
            storage_change_mm=30.0,
            is_balanced=True,
            balance_quality="good",
            warnings=[],
        )

        report = compare_water_balance({"Low": result1, "High": result2})

        assert "large differences" in report
        assert "Δ > 0.2" in report
        assert "Check model configurations" in report

    def test_multiple_models(self):
        """Test comparison with multiple models."""
        results = {}
        for i, coeff in enumerate([0.35, 0.42, 0.48, 0.55]):
            results[f"Model{i+1}"] = WaterBalanceResult(
                total_precipitation_mm=100.0,
                basin_area_km2=100.0,
                timestep_hours=1.0,
                total_runoff_mm=coeff * 100,
                total_runoff_volume_m3=coeff * 100 * 100000,
                runoff_coefficient=coeff,
                storage_change_mm=(1-coeff) * 100,
                is_balanced=True,
                balance_quality="good",
                warnings=[],
            )

        report = compare_water_balance(results)

        assert "Model1" in report
        assert "Model4" in report
        assert "Highest runoff coefficient" in report
        assert "Lowest runoff coefficient" in report
        assert "Coefficient range" in report


class TestPrecipMMHToM3S:
    """Test precip_mmh_to_m3s conversion."""

    def test_single_value_conversion(self):
        """Test conversion of single value."""
        # 10 mm/h over 100 km²
        # = 10 * 100 * 1000 / 3600 m³/s
        # = 277.78 m³/s
        result = precip_mmh_to_m3s(10.0, 100.0)
        expected = 10.0 * 100.0 * 1000.0 / 3600.0
        assert abs(result - expected) < 0.01

    def test_zero_precipitation(self):
        """Test conversion of zero precipitation."""
        result = precip_mmh_to_m3s(0.0, 100.0)
        assert result == 0.0

    def test_sequence_conversion(self):
        """Test conversion of sequence."""
        precip = [5.0, 10.0, 15.0]
        area = 50.0
        result = precip_mmh_to_m3s(precip, area)

        assert len(result) == 3
        for i, p in enumerate(precip):
            expected = p * area * 1000.0 / 3600.0
            assert abs(result[i] - expected) < 0.01

    def test_small_area(self):
        """Test with small area."""
        result = precip_mmh_to_m3s(10.0, 1.0)
        expected = 10.0 * 1000.0 / 3600.0  # 2.78 m³/s
        assert abs(result - expected) < 0.01

    def test_large_area(self):
        """Test with large area."""
        result = precip_mmh_to_m3s(10.0, 1000.0)
        expected = 10.0 * 1000.0 * 1000.0 / 3600.0  # 2777.78 m³/s
        assert abs(result - expected) < 0.1

    def test_known_value(self):
        """Test with known conversion value."""
        # 1 mm/h over 1 km² = 0.2778 m³/s
        result = precip_mmh_to_m3s(1.0, 1.0)
        assert abs(result - 1000.0/3600.0) < 0.001


class TestRunoffM3SToMM:
    """Test runoff_m3s_to_mm conversion."""

    def test_single_value_conversion(self):
        """Test conversion of single value."""
        # 277.78 m³/s over 100 km² for 1 hour
        # = 277.78 * 3600 / (100 * 1000) mm
        # = 10.0 mm
        result = runoff_m3s_to_mm(277.78, 100.0, timestep_hours=1.0)
        expected = 277.78 * 3600.0 / (100.0 * 1000.0)
        assert abs(result - expected) < 0.01

    def test_zero_runoff(self):
        """Test conversion of zero runoff."""
        result = runoff_m3s_to_mm(0.0, 100.0)
        assert result == 0.0

    def test_sequence_conversion(self):
        """Test conversion of sequence."""
        runoff = [100.0, 200.0, 300.0]
        area = 50.0
        timestep = 1.0
        result = runoff_m3s_to_mm(runoff, area, timestep)

        assert len(result) == 3
        for i, q in enumerate(runoff):
            expected = q * 3600.0 * timestep / (area * 1000.0)
            assert abs(result[i] - expected) < 0.01

    def test_different_timestep(self):
        """Test with different timestep."""
        # 100 m³/s for 2 hours over 50 km²
        result = runoff_m3s_to_mm(100.0, 50.0, timestep_hours=2.0)
        expected = 100.0 * 3600.0 * 2.0 / (50.0 * 1000.0)
        assert abs(result - expected) < 0.01

    def test_small_area(self):
        """Test with small area."""
        result = runoff_m3s_to_mm(10.0, 1.0, 1.0)
        expected = 10.0 * 3600.0 / 1000.0  # 36.0 mm
        assert abs(result - expected) < 0.01

    def test_large_area(self):
        """Test with large area."""
        result = runoff_m3s_to_mm(1000.0, 1000.0, 1.0)
        expected = 1000.0 * 3600.0 / (1000.0 * 1000.0)  # 3.6 mm
        assert abs(result - expected) < 0.01

    def test_known_value(self):
        """Test with known conversion value."""
        # 0.2778 m³/s over 1 km² for 1 hour = 1.0 mm
        result = runoff_m3s_to_mm(1000.0/3600.0, 1.0, 1.0)
        assert abs(result - 1.0) < 0.01


class TestUnitConversionRoundTrip:
    """Test round-trip unit conversions."""

    def test_precip_to_runoff_roundtrip(self):
        """Test that converting precip to m³/s and back gives original value."""
        precip_mmh = 10.0
        area = 100.0
        timestep = 1.0

        # Convert to m³/s
        flow = precip_mmh_to_m3s(precip_mmh, area)

        # Convert back to mm
        depth = runoff_m3s_to_mm(flow, area, timestep)

        # Should equal original (for 1-hour timestep, mmh = mm)
        assert abs(depth - precip_mmh) < 0.01

    def test_sequence_roundtrip(self):
        """Test round-trip with sequence."""
        precip = [5.0, 10.0, 15.0, 20.0]
        area = 50.0
        timestep = 1.0

        # Convert to m³/s and back
        flow = precip_mmh_to_m3s(precip, area)
        depth = runoff_m3s_to_mm(flow, area, timestep)

        for i in range(len(precip)):
            assert abs(depth[i] - precip[i]) < 0.01


class TestWaterBalanceIntegration:
    """Integration tests for water balance analysis."""

    def test_complete_workflow(self):
        """Test complete water balance workflow."""
        # Setup
        precip_mm = [10.0, 20.0, 15.0, 5.0]  # mm
        area = 100.0  # km²
        timestep = 1.0  # hours

        # Convert precip to m³/s (assuming it all becomes runoff)
        runoff_m3s = precip_mmh_to_m3s(precip_mm, area)

        # Calculate water balance
        result = calculate_water_balance(precip_mm, runoff_m3s, area, timestep)

        # Verify
        assert abs(result.runoff_coefficient - 1.0) < 0.01
        assert abs(result.storage_change_mm) < 0.1
        assert "high runoff" in str(result.warnings).lower()

    def test_multi_model_comparison(self):
        """Test comparing multiple models."""
        precip = [50.0]
        area = 100.0

        # Three models with different runoff coefficients (close together)
        models = {
            "HBV": 0.50,
            "VIC": 0.48,
            "XAJ": 0.52,
        }

        results = {}
        for name, coeff in models.items():
            runoff_mm = 50.0 * coeff
            runoff_m3s = [runoff_mm * area * 1000 / 3600]
            results[name] = calculate_water_balance(precip, runoff_m3s, area)

        # Compare
        report = compare_water_balance(results)

        assert "HBV" in report
        assert "VIC" in report
        assert "XAJ" in report
        assert "good agreement" in report

    def test_storm_event_analysis(self):
        """Test analysis of a complete storm event."""
        # Rainfall hyetograph (mm)
        precip = [0, 2, 5, 10, 15, 12, 8, 4, 2, 1] + [0] * 10
        area = 75.0

        # Simulated runoff with lag and attenuation
        # Peak runoff occurs after peak rainfall
        runoff_m3s = [0, 10, 40, 100, 180, 200, 150, 100, 60, 30, 15, 8, 5, 3, 2, 1] + [0] * 4

        # Pad or trim to match length
        if len(runoff_m3s) < len(precip):
            runoff_m3s.extend([0] * (len(precip) - len(runoff_m3s)))
        else:
            runoff_m3s = runoff_m3s[:len(precip)]

        result = calculate_water_balance(precip, runoff_m3s, area)

        # Verify reasonable results
        assert result.total_precipitation_mm > 0
        assert result.total_runoff_mm > 0
        assert 0 < result.runoff_coefficient < 1
        assert result.balance_quality in ["good", "acceptable", "poor"]
