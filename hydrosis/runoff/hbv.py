"""Conceptual HBV style runoff implementation."""
from __future__ import annotations

from typing import TYPE_CHECKING, List

from .base import RunoffModel, RunoffModelConfig

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
            self.upper += recharge - quickflow - self.percolation
            self.lower += self.percolation - self.k2 * self.lower
            baseflow = self.k1 * self.upper + self.k2 * self.lower

            flows.append((quickflow + baseflow) * subbasin.area_km2)
        return flows


RunoffModelConfig.register("hbv", HBVRunoff)


__all__ = ["HBVRunoff"]
