"""Comprehensive test to verify parameter validation for all models."""

from hydrosis.runoff.scs_curve_number import SCSCurveNumber
from hydrosis.runoff.xinanjiang import XinAnJiangRunoff
from hydrosis.runoff.hbv import HBVRunoff
from hydrosis.runoff.vic import VICRunoff
from hydrosis.runoff.hymod import HYMODRunoff
from hydrosis.runoff.linear_reservoir import LinearReservoirRunoff
from hydrosis.runoff.distributed_green_ampt import DistributedGreenAmpt
from hydrosis.runoff.wetspa import WETSPARunoff
from hydrosis.routing.muskingum import MuskingumRouting
from hydrosis.routing.lag import LagRouting
from hydrosis.routing.dynamic_wave import DynamicWaveRouting
from hydrosis.validation import ParameterValidationError

print("Testing Comprehensive Parameter Validation...")
print("=" * 80)

test_count = 0
pass_count = 0
fail_count = 0

def test_valid(name, model_class, params):
    """Test that valid parameters are accepted."""
    global test_count, pass_count, fail_count
    test_count += 1
    try:
        model = model_class(params)
        print(f"   ✅ PASS: {name} - Valid parameters accepted")
        pass_count += 1
        return True
    except ParameterValidationError as e:
        print(f"   ❌ FAIL: {name} - {e}")
        fail_count += 1
        return False

def test_invalid(name, model_class, params, expected_param):
    """Test that invalid parameters are rejected."""
    global test_count, pass_count, fail_count
    test_count += 1
    try:
        model = model_class(params)
        print(f"   ❌ FAIL: {name} - Invalid parameters accepted (should have failed)")
        fail_count += 1
        return False
    except ParameterValidationError as e:
        if expected_param in str(e):
            print(f"   ✅ PASS: {name} - Correctly rejected: {e}")
            pass_count += 1
            return True
        else:
            print(f"   ❌ FAIL: {name} - Wrong parameter rejected: {e}")
            fail_count += 1
            return False

# =============================================================================
# RUNOFF MODELS
# =============================================================================

print("\n" + "=" * 80)
print("RUNOFF MODELS")
print("=" * 80)

# SCS Curve Number
print("\n1. SCS Curve Number")
test_valid("SCS valid", SCSCurveNumber, {"curve_number": 75, "initial_abstraction_ratio": 0.2})
test_invalid("SCS invalid CN", SCSCurveNumber, {"curve_number": 150}, "curve_number")

# XinAnJiang
print("\n2. XinAnJiang")
test_valid("XAJ valid", XinAnJiangRunoff, {"wm": 150, "b": 0.3, "imp": 0.05, "recession": 0.6})
test_invalid("XAJ invalid wm", XinAnJiangRunoff, {"wm": -10}, "wm")

# HBV
print("\n3. HBV")
test_valid("HBV valid", HBVRunoff, {
    "degree_day_factor": 3.0,
    "field_capacity": 100.0,
    "beta": 1.0,
    "k0": 0.15,
    "k1": 0.05,
    "k2": 0.01
})
test_invalid("HBV invalid k0", HBVRunoff, {"k0": 1.5}, "k0")
test_invalid("HBV invalid field_capacity", HBVRunoff, {"field_capacity": -10}, "field_capacity")

# VIC
print("\n4. VIC")
test_valid("VIC valid", VICRunoff, {
    "infiltration_shape": 0.3,
    "max_soil_moisture": 150.0,
    "baseflow_coefficient": 0.005,
    "recession": 0.95
})
test_invalid("VIC invalid infiltration_shape", VICRunoff, {"infiltration_shape": -0.5}, "infiltration_shape")
test_invalid("VIC invalid recession", VICRunoff, {"recession": 1.5}, "recession")

# HYMOD
print("\n5. HYMOD")
test_valid("HYMOD valid", HYMODRunoff, {
    "max_storage": 100.0,
    "beta": 1.0,
    "quickflow_ratio": 0.7,
    "quick_k": 0.5,
    "slow_k": 0.05,
    "num_quick_reservoirs": 3
})
test_invalid("HYMOD invalid max_storage", HYMODRunoff, {"max_storage": -50}, "max_storage")
test_invalid("HYMOD invalid num_quick_reservoirs", HYMODRunoff, {"num_quick_reservoirs": 0}, "num_quick_reservoirs")

# Linear Reservoir
print("\n6. Linear Reservoir")
test_valid("LinearRes valid", LinearReservoirRunoff, {"recession": 0.9, "conversion": 1.0})
test_invalid("LinearRes invalid recession", LinearReservoirRunoff, {"recession": 1.5}, "recession")

# Distributed Green-Ampt
print("\n7. Distributed Green-Ampt")
test_valid("GreenAmpt valid", DistributedGreenAmpt, {
    "saturated_conductivity": 10.0,
    "wetting_front_suction": 100.0,
    "initial_moisture": 0.2,
    "saturated_moisture": 0.4,
    "porosity": 0.45,
    "zones": 5
})
test_invalid("GreenAmpt invalid zones", DistributedGreenAmpt, {"zones": 0}, "zones")
test_invalid("GreenAmpt moisture conflict", DistributedGreenAmpt, {
    "initial_moisture": 0.5,
    "saturated_moisture": 0.4
}, "initial_moisture")

# WETSPA
print("\n8. WETSPA")
test_valid("WETSPA valid", WETSPARunoff, {
    "soil_storage_max": 200.0,
    "infiltration_coefficient": 0.6,
    "surface_runoff_coefficient": 0.4,
    "percolation_coefficient": 0.05,
    "baseflow_constant": 0.04
})
test_invalid("WETSPA invalid soil_storage_max", WETSPARunoff, {"soil_storage_max": -100}, "soil_storage_max")
test_invalid("WETSPA invalid infiltration", WETSPARunoff, {"infiltration_coefficient": 1.5}, "infiltration_coefficient")

# =============================================================================
# ROUTING MODELS
# =============================================================================

print("\n" + "=" * 80)
print("ROUTING MODELS")
print("=" * 80)

# Muskingum
print("\n9. Muskingum")
test_valid("Muskingum valid", MuskingumRouting, {
    "travel_time": 12.0,
    "weighting_factor": 0.2,
    "time_step": 1.0
})
test_invalid("Muskingum invalid weighting_factor", MuskingumRouting, {"weighting_factor": 0.8}, "weighting_factor")

# Lag
print("\n10. Lag")
test_valid("Lag valid", LagRouting, {"lag_steps": 2})
test_invalid("Lag invalid lag_steps", LagRouting, {"lag_steps": -1}, "lag_steps")

# Dynamic Wave
print("\n11. Dynamic Wave")
test_valid("DynamicWave valid", DynamicWaveRouting, {
    "time_step": 1.0,
    "reach_length": 5.0,
    "segments": 5,
    "wave_celerity": 1.5,
    "diffusivity": 0.05,
    "substeps": 1
})
test_invalid("DynamicWave invalid time_step", DynamicWaveRouting, {"time_step": -1.0}, "time_step")
test_invalid("DynamicWave invalid segments", DynamicWaveRouting, {"segments": 0}, "segments")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("VALIDATION TEST SUMMARY")
print("=" * 80)
print(f"Total tests: {test_count}")
print(f"Passed: {pass_count} ({100*pass_count/test_count:.1f}%)")
print(f"Failed: {fail_count} ({100*fail_count/test_count:.1f}%)")
print("=" * 80)

if fail_count == 0:
    print("✅ ALL TESTS PASSED!")
else:
    print(f"❌ {fail_count} TEST(S) FAILED")

print()
