"""Quick test to verify parameter validation works correctly."""

from hydrosis.runoff.scs_curve_number import SCSCurveNumber
from hydrosis.runoff.xinanjiang import XinAnJiangRunoff
from hydrosis.routing.muskingum import MuskingumRouting
from hydrosis.routing.lag import LagRouting
from hydrosis.validation import ParameterValidationError

print("Testing Parameter Validation...")
print("=" * 60)

# Test 1: Valid SCS parameters - should work
print("\n1. Testing SCS with VALID parameters...")
try:
    model = SCSCurveNumber({"curve_number": 75, "initial_abstraction_ratio": 0.2})
    print("   ✅ PASS: Valid parameters accepted")
except ParameterValidationError as e:
    print(f"   ❌ FAIL: {e}")

# Test 2: Invalid SCS curve number - should fail
print("\n2. Testing SCS with INVALID curve number (150)...")
try:
    model = SCSCurveNumber({"curve_number": 150})
    print("   ❌ FAIL: Invalid parameters accepted (should have failed)")
except ParameterValidationError as e:
    print(f"   ✅ PASS: Correctly rejected - {e}")

# Test 3: Valid XinAnJiang parameters - should work
print("\n3. Testing XinAnJiang with VALID parameters...")
try:
    model = XinAnJiangRunoff({"wm": 150, "b": 0.3, "imp": 0.05, "recession": 0.6})
    print("   ✅ PASS: Valid parameters accepted")
except ParameterValidationError as e:
    print(f"   ❌ FAIL: {e}")

# Test 4: Invalid XinAnJiang wm - should fail
print("\n4. Testing XinAnJiang with INVALID wm (-10)...")
try:
    model = XinAnJiangRunoff({"wm": -10})
    print("   ❌ FAIL: Invalid parameters accepted (should have failed)")
except ParameterValidationError as e:
    print(f"   ✅ PASS: Correctly rejected - {e}")

# Test 5: Valid Muskingum parameters - should work
print("\n5. Testing Muskingum with VALID parameters...")
try:
    model = MuskingumRouting({"travel_time": 12, "weighting_factor": 0.2, "time_step": 1.0})
    print("   ✅ PASS: Valid parameters accepted")
except ParameterValidationError as e:
    print(f"   ❌ FAIL: {e}")

# Test 6: Invalid Muskingum weighting factor - should fail
print("\n6. Testing Muskingum with INVALID weighting_factor (0.8)...")
try:
    model = MuskingumRouting({"weighting_factor": 0.8})
    print("   ❌ FAIL: Invalid parameters accepted (should have failed)")
except ParameterValidationError as e:
    print(f"   ✅ PASS: Correctly rejected - {e}")

# Test 7: Valid Lag parameters - should work
print("\n7. Testing Lag with VALID parameters...")
try:
    model = LagRouting({"lag_steps": 2})
    print("   ✅ PASS: Valid parameters accepted")
except ParameterValidationError as e:
    print(f"   ❌ FAIL: {e}")

# Test 8: Invalid Lag parameters - should fail
print("\n8. Testing Lag with INVALID lag_steps (-1)...")
try:
    model = LagRouting({"lag_steps": -1})
    print("   ❌ FAIL: Invalid parameters accepted (should have failed)")
except ParameterValidationError as e:
    print(f"   ✅ PASS: Correctly rejected - {e}")

print("\n" + "=" * 60)
print("Parameter Validation Tests Complete!")
print("=" * 60)
