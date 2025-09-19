"""HYMOD runoff generation model implementation."""
from __future__ import annotations

from typing import TYPE_CHECKING, List

from .base import RunoffModel, RunoffModelConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..model import Subbasin


class HYMODRunoff(RunoffModel):
    """Lumped HYMOD rainfall-runoff model with quick and slow reservoirs."""

    def __init__(self, parameters):
        super().__init__(parameters)
        self.smax = max(1e-6, float(self.parameters.get("max_storage", 100.0)))
        self.beta = max(0.0, float(self.parameters.get("beta", 1.0)))
        self.quickflow_ratio = min(
            1.0, max(0.0, float(self.parameters.get("quickflow_ratio", 0.7)))
        )
        self.k_quick = min(1.0, max(0.0, float(self.parameters.get("quick_k", 0.5))))
        self.k_slow = min(1.0, max(0.0, float(self.parameters.get("slow_k", 0.05))))
        self.num_quick = max(1, int(self.parameters.get("num_quick_reservoirs", 3)))
        self.soil_storage = min(
            self.smax,
            max(0.0, float(self.parameters.get("initial_soil_storage", 0.5 * self.smax))),
        )
        initial_quick = float(self.parameters.get("initial_quick_storage", 0.0))
        self.quick_states = [initial_quick for _ in range(self.num_quick)]
        self.slow_state = float(self.parameters.get("initial_slow_storage", 0.0))

    def _effective_rain(self, rainfall: float) -> float:
        if rainfall <= 0.0:
            return 0.0
        capacity_ratio = max(0.0, 1.0 - self.soil_storage / self.smax)
        # Soil moisture accounting following the HyMOD non-linear store.
        infiltration = self.smax * (1.0 - capacity_ratio ** (1.0 / (1.0 + self.beta)))
        infiltration = max(0.0, infiltration - self.soil_storage)
        infiltration = min(rainfall, infiltration)
        self.soil_storage = min(self.smax, self.soil_storage + infiltration)
        return max(0.0, rainfall - infiltration)

    def _route_quickflow(self, inflow: float) -> float:
        routed = inflow
        for i in range(self.num_quick):
            storage = self.quick_states[i] + routed
            outflow = self.k_quick * storage
            self.quick_states[i] = storage - outflow
            routed = outflow
        return routed

    def simulate(self, subbasin: "Subbasin", precipitation: List[float]) -> List[float]:
        flows: List[float] = []
        for p in precipitation:
            effective_rain = self._effective_rain(p)
            quick_input = effective_rain * self.quickflow_ratio
            slow_input = effective_rain - quick_input

            quick_flow = self._route_quickflow(quick_input)

            slow_storage = self.slow_state + slow_input
            slow_flow = self.k_slow * slow_storage
            self.slow_state = slow_storage - slow_flow

            total_runoff = quick_flow + slow_flow
            flows.append(total_runoff * subbasin.area_km2)
        return flows


RunoffModelConfig.register("hymod", HYMODRunoff)
