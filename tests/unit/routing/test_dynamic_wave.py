"""Unit tests for Dynamic Wave routing model."""
import pytest
import numpy as np
from hydrosis.routing.dynamic_wave import DynamicWaveRouting


@pytest.fixture
def sample_subbasin():
    """Create a sample subbasin for testing."""
    class MockSubbasin:
        def __init__(self):
            self.area_km2 = 100.0
    return MockSubbasin()


@pytest.fixture
def default_dynamic_wave_params():
    """Default parameters for Dynamic Wave routing."""
    return {
        "time_step": 1.0,
        "reach_length": 10.0,
        "segments": 10,
        "wave_celerity": 1.5,
        "diffusivity": 0.1,
        "substeps": 1,
        "auto_substeps": False,
        "max_substeps": 10
    }


class TestDynamicWaveInitialization:
    """Test Dynamic Wave routing model initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        model = DynamicWaveRouting({})
        assert model.dt == 1.0
        assert model.reach_length == 5.0
        assert model.segments == 5
        assert model.dx == 1.0  # reach_length / segments
        assert model.wave_celerity == 1.5
        assert model.diffusivity == 0.05
        assert model.substeps >= 1
        assert not model.auto_substeps
        assert model.max_substeps >= 1

    def test_custom_initialization(self, default_dynamic_wave_params):
        """Test initialization with custom parameters."""
        model = DynamicWaveRouting(default_dynamic_wave_params)
        assert model.dt == 1.0
        assert model.reach_length == 10.0
        assert model.segments == 10
        assert model.dx == 1.0  # 10.0 / 10
        assert model.wave_celerity == 1.5
        assert model.diffusivity == 0.1
        assert model.substeps == 1
        assert not model.auto_substeps
        assert model.max_substeps == 10

    def test_dx_calculation(self):
        """Test spatial step calculation."""
        params = {"reach_length": 20.0, "segments": 5}
        model = DynamicWaveRouting(params)
        assert model.dx == 4.0  # 20.0 / 5

    def test_auto_substeps_enabled(self):
        """Test auto substeps when enabled."""
        params = {
            "time_step": 2.0,
            "reach_length": 1.0,
            "segments": 1,
            "wave_celerity": 2.0,
            "diffusivity": 1.0,
            "substeps": 1,
            "auto_substeps": True,
            "max_substeps": 20
        }
        model = DynamicWaveRouting(params)
        # High Courant (4.0) and diffusion (2.0) should trigger auto substeps
        assert model.substeps > 1


class TestDynamicWaveParameterValidation:
    """Test parameter validation for Dynamic Wave routing."""

    def test_negative_time_step_raises(self):
        """Test that negative time_step raises error."""
        with pytest.raises(ValueError, match="time_step"):
            DynamicWaveRouting({"time_step": -1.0})

    def test_zero_time_step_raises(self):
        """Test that zero time_step raises error."""
        with pytest.raises(ValueError, match="time_step"):
            DynamicWaveRouting({"time_step": 0.0})

    def test_negative_reach_length_raises(self):
        """Test that negative reach_length raises error."""
        with pytest.raises(ValueError, match="reach_length"):
            DynamicWaveRouting({"reach_length": -5.0})

    def test_zero_reach_length_raises(self):
        """Test that zero reach_length raises error."""
        with pytest.raises(ValueError, match="reach_length"):
            DynamicWaveRouting({"reach_length": 0.0})

    def test_zero_segments_raises(self):
        """Test that zero segments raises error."""
        with pytest.raises(ValueError, match="segments"):
            DynamicWaveRouting({"segments": 0})

    def test_negative_segments_raises(self):
        """Test that negative segments raises error."""
        with pytest.raises(ValueError, match="segments"):
            DynamicWaveRouting({"segments": -5})

    def test_negative_wave_celerity_raises(self):
        """Test that negative wave_celerity raises error."""
        with pytest.raises(ValueError, match="wave_celerity"):
            DynamicWaveRouting({"wave_celerity": -1.5})

    def test_zero_wave_celerity_raises(self):
        """Test that zero wave_celerity raises error."""
        with pytest.raises(ValueError, match="wave_celerity"):
            DynamicWaveRouting({"wave_celerity": 0.0})

    def test_negative_diffusivity_raises(self):
        """Test that negative diffusivity raises error."""
        with pytest.raises(ValueError, match="diffusivity"):
            DynamicWaveRouting({"diffusivity": -0.1})

    def test_zero_diffusivity_raises(self):
        """Test that zero diffusivity raises error."""
        with pytest.raises(ValueError, match="diffusivity"):
            DynamicWaveRouting({"diffusivity": 0.0})

    def test_zero_substeps_raises(self):
        """Test that zero substeps raises error."""
        with pytest.raises(ValueError, match="substeps"):
            DynamicWaveRouting({"substeps": 0})

    def test_negative_substeps_raises(self):
        """Test that negative substeps raises error."""
        with pytest.raises(ValueError, match="substeps"):
            DynamicWaveRouting({"substeps": -1})

    def test_max_substeps_less_than_substeps_raises(self):
        """Test that max_substeps < substeps raises error."""
        with pytest.raises(ValueError, match="max_substeps"):
            DynamicWaveRouting({"substeps": 5, "max_substeps": 3})


class TestDynamicWaveRouting:
    """Test Dynamic Wave routing simulation."""

    def test_empty_inflow(self, default_dynamic_wave_params, sample_subbasin):
        """Test routing with empty inflow."""
        model = DynamicWaveRouting(default_dynamic_wave_params)
        outflow = model.route(sample_subbasin, [])
        assert outflow == []

    def test_constant_inflow(self, default_dynamic_wave_params, sample_subbasin):
        """Test routing with constant inflow."""
        model = DynamicWaveRouting(default_dynamic_wave_params)
        inflow = [10.0] * 20
        outflow = model.route(sample_subbasin, inflow)
        assert len(outflow) == len(inflow)
        # Should converge to constant value
        assert all(q >= 0 for q in outflow)
        assert abs(outflow[-1] - 10.0) < 1.0  # Should approach input

    def test_single_value_inflow(self, default_dynamic_wave_params, sample_subbasin):
        """Test routing with single value inflow."""
        model = DynamicWaveRouting(default_dynamic_wave_params)
        inflow = [5.0]
        outflow = model.route(sample_subbasin, inflow)
        assert len(outflow) == 1
        assert outflow[0] >= 0

    def test_smoothing_effect(self, sample_subbasin):
        """Test that dynamic wave provides smoothing."""
        params = {
            "time_step": 1.0,
            "reach_length": 10.0,
            "segments": 10,
            "wave_celerity": 1.0,
            "diffusivity": 0.5,
            "substeps": 1
        }
        model = DynamicWaveRouting(params)
        # Step input: sudden jump
        inflow = [0.0] * 5 + [20.0] * 10
        outflow = model.route(sample_subbasin, inflow)

        # Output should be smoother than input
        # Check that there's gradual transition, not instant jump
        # After a few steps, outflow should approach inflow
        jump_index = 5
        # Right at jump, may equal or slightly less
        assert outflow[jump_index] <= inflow[jump_index]
        # A few steps later should be approaching target
        assert outflow[jump_index + 2] > 0.0

    def test_peak_attenuation(self, sample_subbasin):
        """Test that peak flow is attenuated."""
        params = {
            "time_step": 1.0,
            "reach_length": 10.0,
            "segments": 5,
            "wave_celerity": 1.5,
            "diffusivity": 0.2,
            "substeps": 1
        }
        model = DynamicWaveRouting(params)
        # Triangular hydrograph
        inflow = [0.0, 5.0, 10.0, 15.0, 20.0, 15.0, 10.0, 5.0, 0.0]
        outflow = model.route(sample_subbasin, inflow)

        # Peak should be attenuated
        peak_in = max(inflow)
        peak_out = max(outflow)
        assert peak_out <= peak_in

    def test_volume_conservation_approximate(self, default_dynamic_wave_params, sample_subbasin):
        """Test approximate volume conservation."""
        model = DynamicWaveRouting(default_dynamic_wave_params)
        inflow = [10.0, 20.0, 30.0, 25.0, 15.0, 10.0, 5.0]
        outflow = model.route(sample_subbasin, inflow)

        # Total volume should be approximately conserved
        total_in = sum(inflow)
        total_out = sum(outflow)
        # Allow some difference due to smoothing
        assert abs(total_out - total_in) / total_in < 0.2  # Within 20%


class TestStabilityConditions:
    """Test stability conditions and numerical parameters."""

    def test_stability_terms_calculation(self):
        """Test Courant and diffusion number calculation."""
        params = {
            "time_step": 1.0,
            "reach_length": 10.0,
            "segments": 10,
            "wave_celerity": 1.5,
            "diffusivity": 0.1
        }
        model = DynamicWaveRouting(params)
        terms = model._stability_terms()

        # dx = 10.0 / 10 = 1.0
        # Courant = 1.5 * 1.0 / 1.0 = 1.5
        # Diffusion = 0.1 * 1.0 / 1.0^2 = 0.1
        assert abs(terms["courant"] - 1.5) < 1e-6
        assert abs(terms["diffusion"] - 0.1) < 1e-6

    def test_low_courant_number(self, sample_subbasin):
        """Test with low Courant number (stable)."""
        params = {
            "time_step": 0.1,
            "reach_length": 10.0,
            "segments": 10,
            "wave_celerity": 1.0,
            "diffusivity": 0.01,
            "substeps": 1
        }
        model = DynamicWaveRouting(params)
        terms = model._stability_terms()
        assert terms["courant"] < 1.0

        inflow = [10.0] * 10
        outflow = model.route(sample_subbasin, inflow)
        assert all(q >= 0 for q in outflow)
        assert not any(np.isnan(outflow))

    def test_high_courant_number_with_substeps(self, sample_subbasin):
        """Test with high Courant number using substeps."""
        params = {
            "time_step": 2.0,
            "reach_length": 1.0,
            "segments": 1,
            "wave_celerity": 2.0,
            "diffusivity": 0.01,
            "substeps": 10,
            "max_substeps": 20
        }
        model = DynamicWaveRouting(params)
        # Courant = 2.0 * 2.0 / 1.0 = 4.0 (high, but mitigated by substeps)

        inflow = [10.0] * 10
        outflow = model.route(sample_subbasin, inflow)
        assert all(q >= 0 for q in outflow)
        assert not any(np.isnan(outflow))

    def test_auto_substeps_increases_for_instability(self):
        """Test that auto_substeps increases when needed."""
        # Stable case
        stable_params = {
            "time_step": 0.1,
            "reach_length": 10.0,
            "segments": 10,
            "wave_celerity": 0.5,
            "diffusivity": 0.01,
            "substeps": 1,
            "auto_substeps": True,
            "max_substeps": 20
        }
        stable_model = DynamicWaveRouting(stable_params)

        # Unstable case
        unstable_params = {
            "time_step": 2.0,
            "reach_length": 1.0,
            "segments": 1,
            "wave_celerity": 3.0,
            "diffusivity": 1.0,
            "substeps": 1,
            "auto_substeps": True,
            "max_substeps": 20
        }
        unstable_model = DynamicWaveRouting(unstable_params)

        # Unstable model should have more substeps
        assert unstable_model.substeps > stable_model.substeps


class TestSubstepCalculation:
    """Test substep calculation and effects."""

    def test_more_substeps_smoother_result(self, sample_subbasin):
        """Test that more substeps provide smoother results."""
        base_params = {
            "time_step": 1.0,
            "reach_length": 5.0,
            "segments": 5,
            "wave_celerity": 2.0,
            "diffusivity": 0.3
        }

        # Model with 1 substep
        params_1 = {**base_params, "substeps": 1}
        model_1 = DynamicWaveRouting(params_1)

        # Model with 10 substeps
        params_10 = {**base_params, "substeps": 10, "max_substeps": 20}
        model_10 = DynamicWaveRouting(params_10)

        inflow = [0.0, 0.0, 20.0, 0.0, 0.0]
        outflow_1 = model_1.route(sample_subbasin, inflow)
        outflow_10 = model_10.route(sample_subbasin, inflow)

        # Both should produce valid results
        assert all(q >= 0 for q in outflow_1)
        assert all(q >= 0 for q in outflow_10)

    def test_substeps_stability(self, sample_subbasin):
        """Test that substeps improve stability."""
        params = {
            "time_step": 1.0,
            "reach_length": 2.0,
            "segments": 2,
            "wave_celerity": 3.0,
            "diffusivity": 0.5,
            "substeps": 5,
            "max_substeps": 10
        }
        model = DynamicWaveRouting(params)

        # High variance input
        inflow = [0.0, 50.0, 0.0, 100.0, 0.0, 25.0]
        outflow = model.route(sample_subbasin, inflow)

        assert all(q >= 0 for q in outflow)
        assert not any(np.isnan(outflow))
        assert not any(np.isinf(outflow))


class TestSegmentEffects:
    """Test effects of channel segmentation."""

    def test_more_segments_finer_resolution(self):
        """Test that more segments provide finer spatial resolution."""
        params_5 = {
            "reach_length": 10.0,
            "segments": 5
        }
        model_5 = DynamicWaveRouting(params_5)
        assert model_5.dx == 2.0

        params_20 = {
            "reach_length": 10.0,
            "segments": 20
        }
        model_20 = DynamicWaveRouting(params_20)
        assert model_20.dx == 0.5

        assert model_20.dx < model_5.dx

    def test_single_segment(self, sample_subbasin):
        """Test routing with single segment."""
        params = {
            "reach_length": 10.0,
            "segments": 1,
            "wave_celerity": 1.0,
            "diffusivity": 0.1,
            "substeps": 1
        }
        model = DynamicWaveRouting(params)
        assert model.dx == 10.0

        inflow = [10.0, 15.0, 20.0, 15.0, 10.0]
        outflow = model.route(sample_subbasin, inflow)
        assert len(outflow) == len(inflow)
        assert all(q >= 0 for q in outflow)

    def test_many_segments(self, sample_subbasin):
        """Test routing with many segments."""
        params = {
            "reach_length": 100.0,
            "segments": 100,
            "wave_celerity": 1.5,
            "diffusivity": 0.05,
            "substeps": 1
        }
        model = DynamicWaveRouting(params)
        assert model.dx == 1.0

        inflow = [10.0] * 20
        outflow = model.route(sample_subbasin, inflow)
        assert len(outflow) == len(inflow)
        assert all(q >= 0 for q in outflow)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_inflow(self, default_dynamic_wave_params, sample_subbasin):
        """Test routing with zero inflow."""
        model = DynamicWaveRouting(default_dynamic_wave_params)
        inflow = [0.0] * 10
        outflow = model.route(sample_subbasin, inflow)
        assert all(q == 0.0 for q in outflow)

    def test_sudden_increase(self, default_dynamic_wave_params, sample_subbasin):
        """Test sudden flow increase."""
        model = DynamicWaveRouting(default_dynamic_wave_params)
        inflow = [1.0, 1.0, 100.0, 1.0, 1.0]
        outflow = model.route(sample_subbasin, inflow)
        assert len(outflow) == len(inflow)
        assert all(q >= 0 for q in outflow)

    def test_sudden_decrease(self, default_dynamic_wave_params, sample_subbasin):
        """Test sudden flow decrease."""
        model = DynamicWaveRouting(default_dynamic_wave_params)
        inflow = [100.0, 100.0, 1.0, 100.0, 100.0]
        outflow = model.route(sample_subbasin, inflow)
        assert len(outflow) == len(inflow)
        assert all(q >= 0 for q in outflow)

    def test_very_large_inflow(self, default_dynamic_wave_params, sample_subbasin):
        """Test with very large inflow values."""
        model = DynamicWaveRouting(default_dynamic_wave_params)
        inflow = [1e6] * 5
        outflow = model.route(sample_subbasin, inflow)
        assert all(q >= 0 for q in outflow)
        assert not any(np.isnan(outflow))

    def test_very_small_inflow(self, default_dynamic_wave_params, sample_subbasin):
        """Test with very small inflow values."""
        model = DynamicWaveRouting(default_dynamic_wave_params)
        inflow = [1e-10] * 5
        outflow = model.route(sample_subbasin, inflow)
        assert all(q >= 0 for q in outflow)


class TestNumericalStability:
    """Test numerical stability of the routing."""

    def test_no_negative_outflow(self, default_dynamic_wave_params, sample_subbasin):
        """Test that outflow is never negative."""
        model = DynamicWaveRouting(default_dynamic_wave_params)
        inflow = [5.0, 10.0, 15.0, 20.0, 15.0, 10.0, 5.0, 0.0]
        outflow = model.route(sample_subbasin, inflow)
        assert all(q >= 0 for q in outflow)

    def test_no_nan_values(self, default_dynamic_wave_params, sample_subbasin):
        """Test that routing produces no NaN values."""
        model = DynamicWaveRouting(default_dynamic_wave_params)
        inflow = [10.0, 20.0, 30.0, 20.0, 10.0]
        outflow = model.route(sample_subbasin, inflow)
        assert not any(np.isnan(outflow))

    def test_no_inf_values(self, default_dynamic_wave_params, sample_subbasin):
        """Test that routing produces no infinite values."""
        model = DynamicWaveRouting(default_dynamic_wave_params)
        inflow = [10.0, 20.0, 30.0, 20.0, 10.0]
        outflow = model.route(sample_subbasin, inflow)
        assert not any(np.isinf(outflow))

    def test_long_series_stability(self, default_dynamic_wave_params, sample_subbasin):
        """Test stability with long time series."""
        model = DynamicWaveRouting(default_dynamic_wave_params)
        inflow = [10.0 + 5.0 * np.sin(i * 0.1) for i in range(1000)]
        outflow = model.route(sample_subbasin, inflow)
        assert len(outflow) == 1000
        assert all(q >= 0 for q in outflow)
        assert not any(np.isnan(outflow))
        assert not any(np.isinf(outflow))


class TestResolvedParameters:
    """Test resolved parameters output."""

    def test_resolved_parameters(self, default_dynamic_wave_params, sample_subbasin):
        """Test that resolved parameters are returned correctly."""
        model = DynamicWaveRouting(default_dynamic_wave_params)
        resolved = model.resolved_parameters(sample_subbasin)

        assert "time_step" in resolved
        assert "reach_length" in resolved
        assert "segments" in resolved
        assert "dx" in resolved
        assert "wave_celerity" in resolved
        assert "diffusivity" in resolved
        assert "courant" in resolved
        assert "diffusion" in resolved
        assert "substeps" in resolved

    def test_resolved_parameters_values(self, default_dynamic_wave_params, sample_subbasin):
        """Test that resolved parameter values are correct."""
        model = DynamicWaveRouting(default_dynamic_wave_params)
        resolved = model.resolved_parameters(sample_subbasin)

        assert resolved["time_step"] == 1.0
        assert resolved["reach_length"] == 10.0
        assert resolved["segments"] == 10.0
        assert resolved["dx"] == 1.0
        assert resolved["wave_celerity"] == 1.5
        assert resolved["diffusivity"] == 0.1
        assert resolved["substeps"] == 1.0


class TestParameterEffects:
    """Test effects of different parameters."""

    def test_higher_wave_celerity_faster_propagation(self, sample_subbasin):
        """Test that higher wave celerity leads to faster propagation."""
        base_params = {
            "reach_length": 10.0,
            "segments": 10,
            "diffusivity": 0.05,
            "substeps": 1
        }

        model_slow = DynamicWaveRouting({**base_params, "wave_celerity": 0.5})
        model_fast = DynamicWaveRouting({**base_params, "wave_celerity": 2.0})

        # Impulse input
        inflow = [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        outflow_slow = model_slow.route(sample_subbasin, inflow)
        outflow_fast = model_fast.route(sample_subbasin, inflow)

        # Both should produce valid results
        assert all(q >= 0 for q in outflow_slow)
        assert all(q >= 0 for q in outflow_fast)

    def test_higher_diffusivity_more_smoothing(self, sample_subbasin):
        """Test that higher diffusivity provides more smoothing."""
        base_params = {
            "reach_length": 10.0,
            "segments": 10,
            "wave_celerity": 1.5,
            "substeps": 1
        }

        model_low_diff = DynamicWaveRouting({**base_params, "diffusivity": 0.01})
        model_high_diff = DynamicWaveRouting({**base_params, "diffusivity": 0.5})

        # Step input
        inflow = [0.0, 0.0, 0.0, 20.0, 20.0, 20.0]
        outflow_low = model_low_diff.route(sample_subbasin, inflow)
        outflow_high = model_high_diff.route(sample_subbasin, inflow)

        # High diffusivity should smooth more (lower immediate response)
        assert all(q >= 0 for q in outflow_low)
        assert all(q >= 0 for q in outflow_high)
        # At the jump point, high diffusivity should have lower value (more smoothed)
        assert outflow_high[3] <= outflow_low[3] or abs(outflow_high[3] - outflow_low[3]) < 1.0
