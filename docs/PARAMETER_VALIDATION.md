# Parameter Validation in HydroSIS

**Status**: ✅ Complete
**Date**: 2025-10-22
**Task**: Stage 1, Task 1.3 - Add Parameter Validation

## Overview

This document describes the parameter validation framework added to HydroSIS to ensure all hydrological model parameters satisfy physical constraints before simulation begins. This feature prevents runtime errors, improves model reliability, and provides clear, actionable error messages to users.

## Motivation

Hydrological models require parameters that represent physical quantities with inherent constraints:
- Probabilities must be in [0, 1]
- Storage capacities must be positive
- Curve numbers have specific valid ranges
- Time steps must be positive
- Initial conditions must be non-negative

Without validation, invalid parameters can cause:
- Runtime errors during simulation
- Numerical instability
- Physically meaningless results
- Difficult-to-debug failures

## Implementation

### Core Validation Module

**File**: `hydrosis/validation.py` (191 lines)

The validation module provides a comprehensive set of validation functions:

#### Exception Class

```python
class ParameterValidationError(ValueError):
    """Raised when a model parameter fails validation."""

    def __init__(self, parameter_name: str, value: Any, message: str):
        self.parameter_name = parameter_name
        self.value = value
        super().__init__(f"Parameter '{parameter_name}' = {value}: {message}")
```

#### Validation Functions

1. **`validate_positive(name, value, strict=True)`**
   - Validates that a parameter is positive (> 0) or non-negative (≥ 0)
   - Used for: storage capacities, time steps, physical coefficients

2. **`validate_range(name, value, min_value, max_value, min_inclusive, max_inclusive)`**
   - Validates that a parameter is within a specified range
   - Supports inclusive/exclusive bounds
   - Used for: bounded parameters with specific physical limits

3. **`validate_probability(name, value)`**
   - Validates that a parameter is in [0, 1]
   - Used for: fractions, ratios, recession coefficients

4. **`validate_integer(name, value, min_value, max_value)`**
   - Validates that a parameter is an integer within a range
   - Used for: counts, steps, segments

5. **`validate_curve_number(name, value)`**
   - Validates SCS Curve Number (0, 100]
   - Special case: CN must be > 0 (not = 0) as CN=0 means infinite abstraction

6. **`validate_manning_n(name, value)`**
   - Validates Manning's roughness coefficient
   - Range: [0.001, 1.0] with warning if > 0.2

### Base Class Integration

**Files**: `hydrosis/runoff/base.py`, `hydrosis/routing/base.py`

All model base classes call `validate_parameters()` during initialization:

```python
class RunoffModel:
    def __init__(self, parameters: Mapping[str, float]):
        self.parameters = dict(parameters)
        self.validate_parameters()  # Automatic validation

    def validate_parameters(self) -> None:
        """Validate model parameters against physical constraints."""
        pass  # Default: no validation (backward compatibility)
```

This design:
- Ensures validation happens automatically when models are instantiated
- Maintains backward compatibility (default implementation does nothing)
- Allows each model to define its own validation rules

## Model-Specific Validation

### Runoff Models

#### 1. SCS Curve Number (`scs_curve_number.py`)

**Parameters validated**:
- `curve_number`: (0, 100] - must be > 0 and ≤ 100
- `initial_abstraction_ratio`: [0, 1] - fraction of potential retention

**Physical constraints**:
- CN = 0 would mean infinite abstraction (physically impossible)
- CN = 100 represents completely impervious surface
- Initial abstraction ratio is a fraction of maximum retention

#### 2. XinAnJiang (`xinanjiang.py`)

**Parameters validated**:
- `wm`: > 0 - tension water capacity (mm)
- `b`: ≥ 0 - storage distribution curve exponent
- `imp`: [0, 1] - impervious area fraction
- `recession`: [0, 1] - groundwater recession coefficient
- `initial_tension_water`: ≥ 0
- `initial_groundwater`: ≥ 0

**Physical constraints**:
- Water storage capacities must be positive
- Fractions must be in [0, 1]
- Initial conditions must be non-negative

#### 3. HBV (`hbv.py`)

**Parameters validated**:
- `degree_day_factor`: ≥ 0 - snowmelt rate (mm/°C/day)
- `field_capacity`: > 0 - soil water capacity (mm)
- `beta`: > 0 - non-linear recharge exponent
- `k0`, `k1`, `k2`: [0, 1] - recession coefficients
- `percolation`: ≥ 0 - percolation rate (mm/day)
- All initial storages: ≥ 0

**Physical constraints**:
- Recession coefficients represent fractions of storage released per time step
- Storage capacities must be positive
- Initial conditions must be physically realistic

#### 4. VIC (`vic.py`)

**Parameters validated**:
- `infiltration_shape`: > 0 - ARNO/VIC infiltration curve shape parameter
- `max_soil_moisture`: > 0 - maximum soil moisture capacity (mm)
- `baseflow_coefficient`: ≥ 0 - baseflow generation rate
- `recession`: [0, 1] - deep layer recession coefficient
- All initial storages: ≥ 0

**Physical constraints**:
- Shape parameter must be positive for the infiltration curve to be well-defined
- Storage capacities must be positive

#### 5. HYMOD (`hymod.py`)

**Parameters validated**:
- `max_storage`: > 0 - maximum soil moisture capacity (mm)
- `beta`: ≥ 0 - degree of spatial variability in soil moisture capacity
- `quickflow_ratio`: [0, 1] - fraction of effective rainfall to quickflow
- `quick_k`, `slow_k`: [0, 1] - recession coefficients
- `num_quick_reservoirs`: ≥ 1 - number of quick reservoirs in series
- `initial_soil_storage`: [0, max_storage]
- `initial_quick_storage`, `initial_slow_storage`: ≥ 0

**Physical constraints**:
- Quickflow ratio partitions effective rainfall
- Must have at least one quick reservoir
- Initial soil storage cannot exceed capacity

#### 6. Linear Reservoir (`linear_reservoir.py`)

**Parameters validated**:
- `recession`: [0, 1] - recession coefficient
- `conversion`: ≥ 0 - precipitation to storage conversion factor
- `initial_storage`: ≥ 0

**Physical constraints**:
- Recession coefficient represents fraction released per time step
- All physical quantities must be non-negative

#### 7. Distributed Green-Ampt (`distributed_green_ampt.py`)

**Parameters validated**:
- `saturated_conductivity`: > 0 - hydraulic conductivity (mm/hr)
- `wetting_front_suction`: > 0 - suction head (mm)
- `initial_moisture`: [0, 1] - initial soil moisture content
- `saturated_moisture`: [0, 1] - saturated soil moisture content
- `porosity`: [0, 1] - soil porosity
- `zones`: ≥ 1 - number of spatial zones

**Cross-parameter constraints**:
- `initial_moisture < saturated_moisture ≤ porosity`

**Physical constraints**:
- Hydraulic properties must be positive
- Moisture contents are volumetric fractions
- Logical ordering: initial < saturated < porosity

#### 8. WETSPA (`wetspa.py`)

**Parameters validated**:
- `soil_storage_max`: > 0 - maximum soil storage (mm)
- `infiltration_coefficient`: [0, 1] - fraction of precipitation that infiltrates
- `surface_runoff_coefficient`: [0, 1] - fraction of excess that becomes runoff
- `percolation_coefficient`: [0, 1] - fraction of soil moisture that percolates
- `baseflow_constant`: [0, 1] - baseflow recession coefficient
- `initial_soil_moisture`: [0, soil_storage_max]
- `initial_groundwater`: ≥ 0

**Physical constraints**:
- All coefficients represent fractions or rates
- Initial soil moisture cannot exceed capacity

### Routing Models

#### 9. Muskingum (`muskingum.py`)

**Parameters validated**:
- `travel_time` (K): > 0 - wave travel time through reach (hours)
- `weighting_factor` (X): [0, 0.5] - spatial weighting factor
- `time_step` (dt): > 0 - computational time step (hours)

**Stability constraint**:
- `dt ≤ 2*K*(1-X)` - Muskingum stability condition

**Physical constraints**:
- X ∈ [0, 0.5] ensures numerical stability
- X = 0: complete attenuation (reservoir routing)
- X = 0.5: no attenuation (pure translation)

#### 10. Lag (`lag.py`)

**Parameters validated**:
- `lag_steps`: ≥ 0 - number of time steps to delay flow

**Physical constraints**:
- Integer number of time steps
- Zero lag means no translation

#### 11. Dynamic Wave (`dynamic_wave.py`)

**Parameters validated**:
- `time_step`: > 0 - computational time step
- `reach_length`: > 0 - length of reach (km)
- `segments`: ≥ 1 - number of computational segments
- `wave_celerity`: > 0 - flood wave speed (m/s)
- `diffusivity`: > 0 - hydraulic diffusivity (m²/s)
- `substeps`: ≥ 1 - number of sub-steps per time step
- `max_substeps`: ≥ substeps

**Physical constraints**:
- All time and space scales must be positive
- At least one computational segment required
- Substeps improve numerical stability

## Testing

### Basic Test (`test_validation.py`)

Tests fundamental validation for 4 models (2 runoff, 2 routing):
- 8 test cases (4 valid, 4 invalid)
- Verifies validation accepts valid parameters and rejects invalid ones

### Comprehensive Test (`test_validation_comprehensive.py`)

Tests all models with validation:
- 28 test cases covering 11 models
- Tests both valid and invalid parameter sets
- Tests cross-parameter constraints (e.g., Green-Ampt moisture ordering)

**Results**: ✅ All 28 tests passed (100%)

## Usage

### For Model Users

Parameter validation is automatic. If you provide invalid parameters, you'll get a clear error message:

```python
from hydrosis.runoff.scs_curve_number import SCSCurveNumber

# This will raise ParameterValidationError
try:
    model = SCSCurveNumber({"curve_number": 150})
except ParameterValidationError as e:
    print(e)
    # Output: Parameter 'curve_number' = 150.0: must be <= 100.0
```

### For Model Developers

To add validation to a new model:

1. Import validation functions:
```python
from ..validation import validate_positive, validate_probability
```

2. Override `validate_parameters()`:
```python
def validate_parameters(self) -> None:
    """Validate model parameters.

    Validates:
        - param1: Description and constraints
        - param2: Description and constraints
    """
    param1 = float(self.parameters.get("param1", default_value))
    validate_positive("param1", param1, strict=True)

    param2 = float(self.parameters.get("param2", default_value))
    validate_probability("param2", param2)
```

3. Document validation in the docstring

## Benefits

1. **Early Error Detection**: Invalid parameters are caught at initialization, not during simulation
2. **Clear Error Messages**: Users see exactly which parameter is invalid and why
3. **Physical Realism**: Ensures all parameters represent physically meaningful values
4. **Numerical Stability**: Catches parameter combinations that would cause numerical issues
5. **Better Debugging**: Validation errors are much easier to debug than runtime failures
6. **Documentation**: Validation serves as documentation of parameter constraints
7. **Cross-Parameter Checks**: Can validate relationships between parameters (e.g., initial < saturated < porosity)

## Files Modified

### New Files
- `hydrosis/validation.py` - Core validation module (191 lines)
- `test_validation.py` - Basic validation tests (79 lines)
- `test_validation_comprehensive.py` - Comprehensive tests (184 lines)
- `docs/PARAMETER_VALIDATION.md` - This documentation

### Modified Files - Runoff Models
- `hydrosis/runoff/base.py` - Added validation hook to base class
- `hydrosis/runoff/scs_curve_number.py` - Added validation
- `hydrosis/runoff/xinanjiang.py` - Added validation
- `hydrosis/runoff/hbv.py` - Added validation
- `hydrosis/runoff/vic.py` - Added validation
- `hydrosis/runoff/hymod.py` - Added validation
- `hydrosis/runoff/linear_reservoir.py` - Added validation
- `hydrosis/runoff/distributed_green_ampt.py` - Added validation
- `hydrosis/runoff/wetspa.py` - Added validation

### Modified Files - Routing Models
- `hydrosis/routing/base.py` - Added validation hook to base class
- `hydrosis/routing/muskingum.py` - Added validation
- `hydrosis/routing/lag.py` - Added validation
- `hydrosis/routing/dynamic_wave.py` - Added validation

### Bug Fixes
- `hydrosis/delineation/pour_points.py` - Fixed missing imports (`field`, `Dict`, `Tuple`, `math`, `numpy`, `rasterio`)

## Statistics

- **Total validation functions**: 6
- **Models with validation**: 11 (8 runoff, 3 routing)
- **Test cases**: 28 (all passing)
- **Lines of validation code**: ~450 lines
- **Documentation**: 450+ lines

## Future Enhancements

Possible improvements for future work:

1. **Warning System**: Add warnings for unusual but valid parameter values
2. **Parameter Correlation Checks**: Validate relationships between parameters across different model components
3. **Unit System Validation**: Ensure parameter units are consistent
4. **Range Recommendations**: Provide typical parameter ranges in validation messages
5. **Configuration File Validation**: Validate entire configuration files before simulation begins
6. **Performance Profiling**: Ensure validation overhead is negligible

## Conclusion

The parameter validation framework significantly improves HydroSIS reliability and user experience by:
- Catching errors early and providing clear, actionable feedback
- Ensuring physical realism of all model parameters
- Preventing numerical instability from invalid parameter combinations
- Serving as executable documentation of parameter constraints

All validation is thoroughly tested and documented, ready for production use.

---

**Task 1.3 Status**: ✅ **COMPLETE**
