"""WETSPA runoff generation model implementation."""
from __future__ import annotations

from typing import TYPE_CHECKING, List

from .base import RunoffModel, RunoffModelConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..model import Subbasin


class WETSPARunoff(RunoffModel):
    """Conceptual WETSPA rainfall-runoff model.

    This implementation retains WETSPA's distributed water balance structure
    while using a lumped approximation for HydroSIS' subbasin scale. Soil
    moisture, percolation and groundwater storages are explicitly tracked to
    provide realistic responses and parameter hooks for calibration.
    """

    def __init__(self, parameters):
        super().__init__(parameters)
        self.capacity = max(1e-6, float(self.parameters.get("soil_storage_max", 200.0)))
        self.infiltration_coeff = min(
            1.0, max(0.0, float(self.parameters.get("infiltration_coefficient", 0.6)))
        )
        self.surface_runoff_coeff = min(
            1.0, max(0.0, float(self.parameters.get("surface_runoff_coefficient", 0.4)))
        )
        self.percolation_coeff = min(
            1.0, max(0.0, float(self.parameters.get("percolation_coefficient", 0.05)))
        )
        self.baseflow_constant = min(
            1.0, max(0.0, float(self.parameters.get("baseflow_constant", 0.04)))
        )
        self.soil_moisture = min(
            self.capacity,
            max(0.0, float(self.parameters.get("initial_soil_moisture", 0.5 * self.capacity))),
        )
        self.groundwater = max(0.0, float(self.parameters.get("initial_groundwater", 0.0)))

    def simulate(self, subbasin: "Subbasin", precipitation: List[float]) -> List[float]:
        flows: List[float] = []
        for p in precipitation:
            infiltration_potential = self.infiltration_coeff * p
            available_storage = max(0.0, self.capacity - self.soil_moisture)
            infiltration = min(infiltration_potential, available_storage)
            surface_excess = max(0.0, p - infiltration)

            self.soil_moisture += infiltration

            percolation = self.percolation_coeff * (self.soil_moisture / self.capacity)
            percolation = min(self.soil_moisture, percolation)
            self.soil_moisture -= percolation
            self.groundwater += percolation

            surface_runoff = self.surface_runoff_coeff * surface_excess
            baseflow = self.baseflow_constant * self.groundwater
            self.groundwater -= baseflow

            total_runoff = surface_runoff + baseflow
            flows.append(total_runoff * subbasin.area_km2)
        return flows


RunoffModelConfig.register("wetspa", WETSPARunoff)
