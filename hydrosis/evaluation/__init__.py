"""Evaluation and comparison utilities for HydroSIS simulations."""

from .comparison import ModelComparator, ModelScore, SimulationEvaluator
from .metrics import (
    DEFAULT_METRICS,
    DEFAULT_ORIENTATION,
    mae,
    nash_sutcliffe_efficiency,
    percent_bias,
    rmse,
)
from .water_balance import (
    WaterBalanceResult,
    calculate_water_balance,
    compare_water_balance,
    precip_mmh_to_m3s,
    runoff_m3s_to_mm,
)

__all__ = [
    "ModelComparator",
    "ModelScore",
    "SimulationEvaluator",
    "DEFAULT_METRICS",
    "DEFAULT_ORIENTATION",
    "mae",
    "nash_sutcliffe_efficiency",
    "percent_bias",
    "rmse",
    "WaterBalanceResult",
    "calculate_water_balance",
    "compare_water_balance",
    "precip_mmh_to_m3s",
    "runoff_m3s_to_mm",
]
