"""Simplified Muskingum routing implementation."""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

from .base import RoutingModel, RoutingModelConfig
from ..validation import validate_positive, validate_range, ParameterValidationError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..model import Subbasin


class MuskingumRouting(RoutingModel):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.k = float(self.parameters.get("travel_time", 12.0))
        self.x = float(self.parameters.get("weighting_factor", 0.2))
        self.dt = float(self.parameters.get("time_step", 1.0))

    def validate_parameters(self) -> None:
        """Validate Muskingum routing parameters.

        Validates:
            - travel_time (k): Must be > 0
            - weighting_factor (x): Must be in [0, 0.5] for stability
            - time_step (dt): Must be > 0
            - Stability condition: dt <= 2*k*(1-x) for numerical stability
        """
        k = float(self.parameters.get("travel_time", 12.0))
        validate_positive("travel_time", k, strict=True)

        x = float(self.parameters.get("weighting_factor", 0.2))
        validate_range("weighting_factor", x, 0.0, 0.5, min_inclusive=True, max_inclusive=True)

        dt = float(self.parameters.get("time_step", 1.0))
        validate_positive("time_step", dt, strict=True)

        # Check Muskingum stability condition
        max_dt = 2.0 * k * (1.0 - x)
        if dt > max_dt:
            raise ParameterValidationError(
                "time_step",
                dt,
                f"violates Muskingum stability condition: dt must be <= 2*k*(1-x) = {max_dt:.2f}"
            )

    def route(self, subbasin: "Subbasin", inflow: List[float]) -> List[float]:
        c0 = (-self.k * self.x + 0.5 * self.dt) / (self.k - self.k * self.x + 0.5 * self.dt)
        c1 = (self.k * self.x + 0.5 * self.dt) / (self.k - self.k * self.x + 0.5 * self.dt)
        c2 = (self.k - self.k * self.x - 0.5 * self.dt) / (self.k - self.k * self.x + 0.5 * self.dt)

        outflow: List[float] = []
        prev_in = inflow[0] if inflow else 0.0
        prev_out = inflow[0] if inflow else 0.0
        for current_in in inflow:
            current_out = c0 * current_in + c1 * prev_in + c2 * prev_out
            outflow.append(max(current_out, 0.0))
            prev_in = current_in
            prev_out = current_out
        return outflow

    def resolved_parameters(self, subbasin: "Subbasin") -> Dict[str, float]:
        return {
            "travel_time": self.k,
            "weighting_factor": self.x,
            "time_step": self.dt,
        }


RoutingModelConfig.register("muskingum", MuskingumRouting)
