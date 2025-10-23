"""Conceptual HBV style runoff implementation."""
from __future__ import annotations

from typing import TYPE_CHECKING, List

from .base import RunoffModel, RunoffModelConfig
from ..validation import validate_positive, validate_probability

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..model import Subbasin


class HBVRunoff(RunoffModel):
    """A minimalist HBV style bucket model.

    The model keeps snow, soil and groundwater storages and provides a
    degree-day melt option so configurations that rely on HBV parameters
    can be ported with limited effort.
    """

    def __init__(self, parameters):
        super().__init__(parameters)
        self.degree_day_factor = float(self.parameters.get("degree_day_factor", 3.0))
        self.snow_threshold = float(self.parameters.get("snow_threshold", 0.0))
        self.field_capacity = float(self.parameters.get("field_capacity", 100.0))
        self.beta = max(1e-6, float(self.parameters.get("beta", 1.0)))
        self.k0 = float(self.parameters.get("k0", 0.15))
        self.k1 = float(self.parameters.get("k1", 0.05))
        self.k2 = float(self.parameters.get("k2", 0.01))
        self.percolation = float(self.parameters.get("percolation", 2.0))
        self.snow = float(self.parameters.get("initial_snow", 0.0))
        self.soil = float(self.parameters.get("initial_soil", 40.0))
        self.upper = float(self.parameters.get("initial_upper", 5.0))
        self.lower = float(self.parameters.get("initial_lower", 20.0))

    def validate_parameters(self) -> None:
        """Validate HBV model parameters.

        Validates:
            - degree_day_factor: Must be >= 0
            - field_capacity: Must be > 0
            - beta: Must be > 0
            - k0, k1, k2: Recession coefficients, must be in [0, 1]
            - percolation: Must be >= 0
            - initial_snow, initial_soil, initial_upper, initial_lower: Must be >= 0
        """
        degree_day = float(self.parameters.get("degree_day_factor", 3.0))
        validate_positive("degree_day_factor", degree_day, strict=False)

        fc = float(self.parameters.get("field_capacity", 100.0))
        validate_positive("field_capacity", fc, strict=True)

        beta = float(self.parameters.get("beta", 1.0))
        validate_positive("beta", beta, strict=True)

        k0 = float(self.parameters.get("k0", 0.15))
        validate_probability("k0", k0)

        k1 = float(self.parameters.get("k1", 0.05))
        validate_probability("k1", k1)

        k2 = float(self.parameters.get("k2", 0.01))
        validate_probability("k2", k2)

        perc = float(self.parameters.get("percolation", 2.0))
        validate_positive("percolation", perc, strict=False)

        init_snow = float(self.parameters.get("initial_snow", 0.0))
        validate_positive("initial_snow", init_snow, strict=False)

        init_soil = float(self.parameters.get("initial_soil", 40.0))
        validate_positive("initial_soil", init_soil, strict=False)

        init_upper = float(self.parameters.get("initial_upper", 5.0))
        validate_positive("initial_upper", init_upper, strict=False)

        init_lower = float(self.parameters.get("initial_lower", 20.0))
        validate_positive("initial_lower", init_lower, strict=False)

    def simulate(self, subbasin: "Subbasin", precipitation: List[float]) -> List[float]:
        flows: List[float] = []
        for p in precipitation:
            rainfall = max(0.0, p - self.snow_threshold)
            snowfall = max(0.0, p - rainfall)
            self.snow += snowfall

            melt = self.degree_day_factor * max(0.0, rainfall - self.snow_threshold)
            melt = min(melt, self.snow)
            self.snow -= melt

            effective_precip = rainfall + melt
            soil_deficit = max(0.0, self.field_capacity - self.soil)
            recharge = effective_precip * ((self.soil / self.field_capacity) ** self.beta)
            recharge = min(recharge, soil_deficit)
            self.soil += effective_precip - recharge

            quickflow = self.k0 * self.upper
            # Limit percolation to available water in upper reservoir
            actual_percolation = min(self.percolation, max(0.0, self.upper + recharge - quickflow))
            self.upper += recharge - quickflow - actual_percolation
            self.upper = max(0.0, self.upper)  # Ensure non-negative
            self.lower += actual_percolation - self.k2 * self.lower
            self.lower = max(0.0, self.lower)  # Ensure non-negative
            baseflow = self.k1 * self.upper + self.k2 * self.lower

            flows.append((quickflow + baseflow) * subbasin.area_km2)
        return flows


RunoffModelConfig.register("hbv", HBVRunoff)


__all__ = ["HBVRunoff"]
