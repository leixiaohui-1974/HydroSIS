"""Parameter validation utilities for HydroSIS models.

Provides common validation functions to ensure model parameters satisfy
physical constraints and prevent runtime errors.
"""
from __future__ import annotations

from typing import Any, Optional


class ParameterValidationError(ValueError):
    """Raised when a model parameter fails validation."""

    def __init__(self, parameter_name: str, value: Any, message: str):
        self.parameter_name = parameter_name
        self.value = value
        super().__init__(f"Parameter '{parameter_name}' = {value}: {message}")


def validate_positive(
    name: str, value: float, strict: bool = True, allow_zero: bool = False
) -> None:
    """Validate that a parameter is positive (or non-negative).

    Args:
        name: Parameter name for error messages
        value: Parameter value to validate
        strict: If True, require value > 0; if False, allow value >= 0
        allow_zero: Deprecated, use strict=False instead

    Raises:
        ParameterValidationError: If validation fails
    """
    if allow_zero:
        strict = False

    if strict:
        if value <= 0:
            raise ParameterValidationError(
                name, value, "must be strictly positive (> 0)"
            )
    else:
        if value < 0:
            raise ParameterValidationError(
                name, value, "must be non-negative (>= 0)"
            )


def validate_range(
    name: str,
    value: float,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    min_inclusive: bool = True,
    max_inclusive: bool = True
) -> None:
    """Validate that a parameter is within a specified range.

    Args:
        name: Parameter name for error messages
        value: Parameter value to validate
        min_value: Minimum allowed value (None = no minimum)
        max_value: Maximum allowed value (None = no maximum)
        min_inclusive: If True, allow value == min_value
        max_inclusive: If True, allow value == max_value

    Raises:
        ParameterValidationError: If validation fails
    """
    if min_value is not None:
        if min_inclusive:
            if value < min_value:
                raise ParameterValidationError(
                    name, value, f"must be >= {min_value}"
                )
        else:
            if value <= min_value:
                raise ParameterValidationError(
                    name, value, f"must be > {min_value}"
                )

    if max_value is not None:
        if max_inclusive:
            if value > max_value:
                raise ParameterValidationError(
                    name, value, f"must be <= {max_value}"
                )
        else:
            if value >= max_value:
                raise ParameterValidationError(
                    name, value, f"must be < {max_value}"
                )


def validate_probability(name: str, value: float) -> None:
    """Validate that a parameter is a valid probability [0, 1].

    Args:
        name: Parameter name for error messages
        value: Parameter value to validate

    Raises:
        ParameterValidationError: If value is not in [0, 1]
    """
    validate_range(name, value, 0.0, 1.0, min_inclusive=True, max_inclusive=True)


def validate_integer(
    name: str,
    value: Any,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None
) -> None:
    """Validate that a parameter is an integer within a range.

    Args:
        name: Parameter name for error messages
        value: Parameter value to validate
        min_value: Minimum allowed value (None = no minimum)
        max_value: Maximum allowed value (None = no maximum)

    Raises:
        ParameterValidationError: If validation fails
    """
    if not isinstance(value, int):
        raise ParameterValidationError(
            name, value, f"must be an integer, got {type(value).__name__}"
        )

    if min_value is not None and value < min_value:
        raise ParameterValidationError(
            name, value, f"must be >= {min_value}"
        )

    if max_value is not None and value > max_value:
        raise ParameterValidationError(
            name, value, f"must be <= {max_value}"
        )


def validate_curve_number(name: str, value: float) -> None:
    """Validate SCS Curve Number (must be in valid range 0-100).

    Args:
        name: Parameter name for error messages
        value: Curve number value to validate

    Raises:
        ParameterValidationError: If value is not in (0, 100]

    Note:
        Curve number must be > 0 (not == 0) and <= 100.
        CN = 0 would mean infinite abstraction.
    """
    validate_range(
        name, value,
        min_value=0.0, max_value=100.0,
        min_inclusive=False,  # CN must be > 0, not >= 0
        max_inclusive=True    # CN can be 100 (impervious)
    )


def validate_manning_n(name: str, value: float) -> None:
    """Validate Manning's roughness coefficient.

    Args:
        name: Parameter name for error messages
        value: Manning's n value to validate

    Raises:
        ParameterValidationError: If value is not in reasonable range

    Note:
        Typical values: 0.01 (smooth) to 0.15 (rough vegetation).
        We allow 0.001 to 1.0 to accommodate unusual cases.
    """
    validate_range(
        name, value,
        min_value=0.001, max_value=1.0,
        min_inclusive=True, max_inclusive=True
    )

    # Warn if value is unusual (but don't fail)
    if value > 0.2:
        import warnings
        warnings.warn(
            f"Manning's n = {value} is unusually high. "
            f"Typical range is 0.01-0.15.",
            UserWarning
        )
