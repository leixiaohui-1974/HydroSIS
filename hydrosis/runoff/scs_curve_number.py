"""Implementation of the SCS Curve Number runoff method."""
from __future__ import annotations

from typing import TYPE_CHECKING, List

from .base import RunoffModel, RunoffModelConfig
from ..validation import validate_curve_number, validate_probability

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..model import Subbasin


class SCSCurveNumber(RunoffModel):
    """Compute direct runoff using the SCS Curve Number method."""

    def __init__(self, parameters):
        super().__init__(parameters)
        self.cn = float(self.parameters.get("curve_number", 75))
        self.initial_abstraction_ratio = float(self.parameters.get("initial_abstraction_ratio", 0.2))

    def validate_parameters(self) -> None:
        """Validate SCS Curve Number model parameters.

        Validates:
            - curve_number: Must be in range (0, 100]
            - initial_abstraction_ratio: Must be in range [0, 1]
        """
        cn = float(self.parameters.get("curve_number", 75))
        validate_curve_number("curve_number", cn)

        ia_ratio = float(self.parameters.get("initial_abstraction_ratio", 0.2))
        validate_probability("initial_abstraction_ratio", ia_ratio)

    def simulate(self, subbasin: "Subbasin", precipitation: List[float]) -> List[float]:
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
