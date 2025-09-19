"""Implementation of the SCS Curve Number runoff method."""
from __future__ import annotations

from typing import List

from .base import RunoffModel, RunoffModelConfig
from ..model import Subbasin


class SCSCurveNumber(RunoffModel):
    """Compute direct runoff using the SCS Curve Number method."""

    def __init__(self, parameters):
        super().__init__(parameters)
        self.cn = float(self.parameters.get("curve_number", 75))
        self.initial_abstraction_ratio = float(self.parameters.get("initial_abstraction_ratio", 0.2))

    def simulate(self, subbasin: Subbasin, precipitation: List[float]) -> List[float]:
        s = max(0.0, (1000.0 / self.cn - 10.0) * 25.4)
        ia = self.initial_abstraction_ratio * s
        runoff: List[float] = []
        for p in precipitation:
            if p <= ia:
                runoff.append(0.0)
            else:
                q = (p - ia) ** 2 / (p - ia + s)
                runoff.append(q * subbasin.area_km2)
        return runoff


RunoffModelConfig.register("scs_curve_number", SCSCurveNumber)
