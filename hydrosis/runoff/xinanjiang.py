"""XinAnJiang runoff generation model implementation."""
from __future__ import annotations

from typing import TYPE_CHECKING, List

from .base import RunoffModel, RunoffModelConfig
from ..validation import validate_positive, validate_probability, validate_range

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..model import Subbasin


class XinAnJiangRunoff(RunoffModel):
    """Simplified XinAnJiang runoff formulation.

    The implementation captures the core XinAnJiang concepts of a finite
    watershed tension water capacity, an impervious fraction, and a linear
    groundwater recession. The equations are simplified to operate only on
    precipitation inputs while maintaining the characteristic non-linear soil
    moisture accounting behaviour expected by configuration users.
    """

    def __init__(self, parameters):
        super().__init__(parameters)
        self.wm = max(1e-6, float(self.parameters.get("wm", 150.0)))
        self.b = max(0.0, float(self.parameters.get("b", 0.3)))
        self.imp = min(1.0, max(0.0, float(self.parameters.get("imp", 0.05))))
        self.k = min(1.0, max(0.0, float(self.parameters.get("recession", 0.6))))
        initial_storage = float(
            self.parameters.get("initial_tension_water", 0.5 * self.wm)
        )
        self.tension_water = min(self.wm, max(0.0, initial_storage))
        self.groundwater = float(self.parameters.get("initial_groundwater", 0.0))

    def validate_parameters(self) -> None:
        """Validate XinAnJiang model parameters.

        Validates:
            - wm: Tension water capacity, must be > 0
            - b: Storage distribution curve exponent, must be >= 0
            - imp: Impervious area fraction, must be in [0, 1]
            - recession: Groundwater recession coefficient, must be in [0, 1]
            - initial_tension_water: Must be >= 0
            - initial_groundwater: Must be >= 0
        """
        wm = float(self.parameters.get("wm", 150.0))
        validate_positive("wm", wm, strict=True)

        b = float(self.parameters.get("b", 0.3))
        validate_positive("b", b, strict=False)  # Can be 0

        imp = float(self.parameters.get("imp", 0.05))
        validate_probability("imp", imp)

        k = float(self.parameters.get("recession", 0.6))
        validate_probability("recession", k)

        initial_tw = float(self.parameters.get("initial_tension_water", 0.5 * wm))
        validate_positive("initial_tension_water", initial_tw, strict=False)

        initial_gw = float(self.parameters.get("initial_groundwater", 0.0))
        validate_positive("initial_groundwater", initial_gw, strict=False)

    def _infiltration_capacity(self) -> float:
        storage_ratio = min(1.0, max(0.0, self.tension_water / self.wm))
        # Non-linear capacity curve following the classic XinAnJiang storage
        # distribution assumption.
        return self.wm * (1.0 - (1.0 - storage_ratio) ** (1.0 / (1.0 + self.b)))

    def simulate(self, subbasin: "Subbasin", precipitation: List[float]) -> List[float]:
        flows: List[float] = []
        for p in precipitation:
            effective_rain = max(0.0, p * (1.0 - self.imp))
            capacity = self._infiltration_capacity()
            infiltration = min(effective_rain, max(0.0, capacity - self.tension_water))
            excess = max(0.0, effective_rain - infiltration)

            self.tension_water = min(self.wm, self.tension_water + infiltration)

            # Update groundwater storage with a fraction of the tension water.
            recharge = max(0.0, self.tension_water * (1.0 - self.k))
            self.tension_water -= recharge
            self.groundwater += recharge

            baseflow = self.k * self.groundwater
            self.groundwater -= baseflow

            total_runoff = excess + baseflow
            flows.append(total_runoff * subbasin.area_km2)
        return flows


RunoffModelConfig.register("xin_an_jiang", XinAnJiangRunoff)
