"""XinAnJiang runoff generation model implementation."""
from __future__ import annotations

from typing import TYPE_CHECKING, List

from .base import RunoffModel, RunoffModelConfig

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
