"""Validation framework for HydroSIS workflows.

This module provides a comprehensive validation system for hydrological
modeling workflows, including spatial data validation, time series validation,
and hydrologic process validation.

All validation criteria are configurable through YAML files, following the
HydroSIS development guidelines of avoiding hardcoded values.
"""
from __future__ import annotations

from .base import (
    ValidationCriteria,
    ValidationResult,
    BaseValidator,
)
from .hydrologic import (
    validate_water_balance,
    validate_runoff_coefficient,
    validate_mass_conservation,
)

# Import parameter validation utilities from legacy validation module
# These are kept for backward compatibility with existing code
from ..validation_legacy import (
    ParameterValidationError,
    validate_positive,
    validate_range,
    validate_probability,
    validate_integer,
    validate_curve_number,
    validate_manning_n,
)

__all__ = [
    # Base classes
    "ValidationCriteria",
    "ValidationResult",
    "BaseValidator",
    # Hydrologic validation
    "validate_water_balance",
    "validate_runoff_coefficient",
    "validate_mass_conservation",
    # Parameter validation (legacy)
    "ParameterValidationError",
    "validate_positive",
    "validate_range",
    "validate_probability",
    "validate_integer",
    "validate_curve_number",
    "validate_manning_n",
]
