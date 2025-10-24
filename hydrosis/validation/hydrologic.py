"""Hydrologic validation functions.

This module provides validation functions for hydrological processes,
including water balance, runoff coefficients, and mass conservation.

All validation criteria are configurable through the ValidationCriteria
system, avoiding hardcoded thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Union

import numpy as np

from .base import ValidationCriteria, ValidationResult


@dataclass
class HydrologicCriteria(ValidationCriteria):
    """Validation criteria for hydrologic processes.

    Attributes
    ----------
    runoff_coefficient_min : float
        Minimum acceptable runoff coefficient (default: 0.0)
    runoff_coefficient_max : float
        Maximum acceptable runoff coefficient (default: 1.0)
    runoff_coefficient_warning_low : float
        Warn if coefficient below this (default: 0.05)
    runoff_coefficient_warning_high : float
        Warn if coefficient above this (default: 0.9)
    water_balance_max_error : float
        Maximum acceptable water balance error ratio (default: 0.01)
    mass_conservation_tolerance : float
        Tolerance for mass conservation check (default: 0.001)
    """
    runoff_coefficient_min: float = 0.0
    runoff_coefficient_max: float = 1.0
    runoff_coefficient_warning_low: float = 0.05
    runoff_coefficient_warning_high: float = 0.9
    water_balance_max_error: float = 0.01
    mass_conservation_tolerance: float = 0.001

    @classmethod
    def from_dict(cls, data: Dict) -> HydrologicCriteria:
        """Create HydrologicCriteria from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing criteria configuration

        Returns
        -------
        HydrologicCriteria
            Initialized criteria object
        """
        base = super().from_dict(data)
        return cls(
            name=base.name,
            description=base.description,
            strict_mode=base.strict_mode,
            runoff_coefficient_min=data.get("runoff_coefficient_min", 0.0),
            runoff_coefficient_max=data.get("runoff_coefficient_max", 1.0),
            runoff_coefficient_warning_low=data.get("runoff_coefficient_warning_low", 0.05),
            runoff_coefficient_warning_high=data.get("runoff_coefficient_warning_high", 0.9),
            water_balance_max_error=data.get("water_balance_max_error", 0.01),
            mass_conservation_tolerance=data.get("mass_conservation_tolerance", 0.001),
        )


def validate_runoff_coefficient(
    runoff_coefficients: Union[Dict[str, float], Sequence[float]],
    zone_ids: Optional[Sequence[str]] = None,
    criteria: Optional[HydrologicCriteria] = None,
    step_name: str = "径流系数验证"
) -> ValidationResult:
    """Validate runoff coefficients.

    Checks if runoff coefficients are within acceptable physical ranges.
    Runoff coefficient must be between 0 and 1, as runoff cannot exceed
    precipitation.

    Parameters
    ----------
    runoff_coefficients : dict or sequence of float
        Runoff coefficients to validate. Can be:
        - Dict mapping zone IDs to coefficients
        - Sequence of coefficient values
    zone_ids : sequence of str, optional
        Zone identifiers (if runoff_coefficients is a sequence)
    criteria : HydrologicCriteria, optional
        Validation criteria. If None, default criteria are used.
    step_name : str, optional
        Name of the validation step

    Returns
    -------
    ValidationResult
        Validation results with errors and warnings

    Examples
    --------
    >>> coeffs = {"zone1": 0.45, "zone2": 0.67, "zone3": 1.2}
    >>> result = validate_runoff_coefficient(coeffs)
    >>> print(result.summary())
    """
    if criteria is None:
        criteria = HydrologicCriteria()

    result = ValidationResult(step_name=step_name)

    # Convert to dict if sequence
    if isinstance(runoff_coefficients, (list, tuple, np.ndarray)):
        if zone_ids is None:
            zone_ids = [f"zone_{i}" for i in range(len(runoff_coefficients))]
        runoff_coefficients = dict(zip(zone_ids, runoff_coefficients))

    # Validate each coefficient
    for zone_id, coeff in runoff_coefficients.items():
        # Add metric
        result.add_metric(f"{zone_id}_runoff_coefficient", coeff)

        # Check hard limits (physical constraints)
        if coeff < criteria.runoff_coefficient_min:
            result.add_error(
                f"{zone_id}: 径流系数 ({coeff:.4f}) 小于最小值 "
                f"({criteria.runoff_coefficient_min})"
            )
        elif coeff > criteria.runoff_coefficient_max:
            result.add_error(
                f"{zone_id}: 径流系数 ({coeff:.4f}) 大于最大值 "
                f"({criteria.runoff_coefficient_max})，径流不能超过降雨"
            )

        # Check warning thresholds
        if criteria.runoff_coefficient_min <= coeff < criteria.runoff_coefficient_warning_low:
            result.add_warning(
                f"{zone_id}: 径流系数过低 ({coeff:.4f})，"
                f"可能存在数据问题或下渗极强"
            )
        elif coeff > criteria.runoff_coefficient_warning_high:
            result.add_warning(
                f"{zone_id}: 径流系数过高 ({coeff:.4f})，"
                f"接近不透水表面"
            )

    # Add summary metrics
    coeffs_values = list(runoff_coefficients.values())
    result.add_metric("mean_runoff_coefficient", np.mean(coeffs_values))
    result.add_metric("max_runoff_coefficient", np.max(coeffs_values))
    result.add_metric("min_runoff_coefficient", np.min(coeffs_values))

    return result


def validate_water_balance(
    precipitation: Union[float, np.ndarray],
    runoff: Union[float, np.ndarray],
    evapotranspiration: Optional[Union[float, np.ndarray]] = None,
    storage_change: Optional[Union[float, np.ndarray]] = None,
    criteria: Optional[HydrologicCriteria] = None,
    step_name: str = "水量平衡验证"
) -> ValidationResult:
    """Validate water balance.

    Checks the water balance equation:
    Precipitation = Runoff + Evapotranspiration + ΔStorage + Error

    Parameters
    ----------
    precipitation : float or ndarray
        Total precipitation (mm)
    runoff : float or ndarray
        Total runoff (mm)
    evapotranspiration : float or ndarray, optional
        Total evapotranspiration (mm)
    storage_change : float or ndarray, optional
        Change in storage (mm)
    criteria : HydrologicCriteria, optional
        Validation criteria
    step_name : str, optional
        Name of the validation step

    Returns
    -------
    ValidationResult
        Validation results

    Examples
    --------
    >>> result = validate_water_balance(
    ...     precipitation=100.0,
    ...     runoff=45.0,
    ...     evapotranspiration=50.0,
    ...     storage_change=4.0
    ... )
    """
    if criteria is None:
        criteria = HydrologicCriteria()

    result = ValidationResult(step_name=step_name)

    # Convert to arrays
    precip = np.atleast_1d(precipitation)
    runoff_arr = np.atleast_1d(runoff)
    et = np.atleast_1d(evapotranspiration) if evapotranspiration is not None else np.zeros_like(precip)
    storage = np.atleast_1d(storage_change) if storage_change is not None else np.zeros_like(precip)

    # Calculate water balance
    outputs = runoff_arr + et + storage
    balance_error = precip - outputs
    relative_error = np.abs(balance_error) / (precip + 1e-10)  # Avoid division by zero

    # Add metrics
    result.add_metric("total_precipitation_mm", float(np.sum(precip)))
    result.add_metric("total_runoff_mm", float(np.sum(runoff_arr)))
    result.add_metric("total_et_mm", float(np.sum(et)))
    result.add_metric("total_storage_change_mm", float(np.sum(storage)))
    result.add_metric("balance_error_mm", float(np.sum(balance_error)))
    result.add_metric("relative_error", float(np.mean(relative_error)))

    # Validate
    max_relative_error = float(np.max(relative_error))
    if max_relative_error > criteria.water_balance_max_error:
        result.add_error(
            f"水量平衡误差过大: {max_relative_error:.4f} "
            f"(阈值: {criteria.water_balance_max_error})"
        )

    # Check if runoff exceeds precipitation
    if np.any(runoff_arr > precip + 0.01):  # Allow 0.01mm numerical tolerance
        excess_locations = np.where(runoff_arr > precip)[0]
        result.add_error(
            f"径流量超过降雨量（违反水量平衡）在 {len(excess_locations)} 个位置"
        )

    return result


def validate_mass_conservation(
    inflow: Union[float, np.ndarray],
    outflow: Union[float, np.ndarray],
    storage_change: Optional[Union[float, np.ndarray]] = None,
    criteria: Optional[HydrologicCriteria] = None,
    step_name: str = "质量守恒验证"
) -> ValidationResult:
    """Validate mass conservation for routing.

    Checks: Inflow = Outflow + ΔStorage

    Parameters
    ----------
    inflow : float or ndarray
        Total inflow (m³ or m³/s·hr)
    outflow : float or ndarray
        Total outflow (m³ or m³/s·hr)
    storage_change : float or ndarray, optional
        Change in channel storage (m³)
    criteria : HydrologicCriteria, optional
        Validation criteria
    step_name : str, optional
        Name of the validation step

    Returns
    -------
    ValidationResult
        Validation results

    Examples
    --------
    >>> result = validate_mass_conservation(
    ...     inflow=1000.0,
    ...     outflow=980.0,
    ...     storage_change=20.0
    ... )
    """
    if criteria is None:
        criteria = HydrologicCriteria()

    result = ValidationResult(step_name=step_name)

    # Convert to arrays
    inflow_arr = np.atleast_1d(inflow)
    outflow_arr = np.atleast_1d(outflow)
    storage = np.atleast_1d(storage_change) if storage_change is not None else np.zeros_like(inflow_arr)

    # Calculate mass balance
    mass_error = inflow_arr - outflow_arr - storage
    relative_error = np.abs(mass_error) / (inflow_arr + 1e-10)

    # Add metrics
    result.add_metric("total_inflow", float(np.sum(inflow_arr)))
    result.add_metric("total_outflow", float(np.sum(outflow_arr)))
    result.add_metric("total_storage_change", float(np.sum(storage)))
    result.add_metric("mass_error", float(np.sum(mass_error)))
    result.add_metric("relative_mass_error", float(np.mean(relative_error)))

    # Validate
    max_relative_error = float(np.max(relative_error))
    if max_relative_error > criteria.mass_conservation_tolerance:
        result.add_error(
            f"质量守恒误差过大: {max_relative_error:.4f} "
            f"(阈值: {criteria.mass_conservation_tolerance})"
        )

    return result
