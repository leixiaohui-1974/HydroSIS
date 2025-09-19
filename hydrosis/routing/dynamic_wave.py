"""Simplified dynamic wave routing implementation."""
from __future__ import annotations

from typing import TYPE_CHECKING, List

from .base import RoutingModel, RoutingModelConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..model import Subbasin


class DynamicWaveRouting(RoutingModel):
    """A kinematic/dynamic wave hybrid solver using an explicit scheme."""

    def __init__(self, parameters):
        super().__init__(parameters)
        self.dt = float(self.parameters.get("time_step", 1.0))
        self.dx = float(self.parameters.get("reach_length", 5.0))
        self.wave_celerity = float(self.parameters.get("wave_celerity", 1.5))
        self.diffusivity = max(1e-6, float(self.parameters.get("diffusivity", 0.05)))

    def route(self, subbasin: "Subbasin", inflow: List[float]) -> List[float]:
        if not inflow:
            return []

        courant = self.wave_celerity * self.dt / max(self.dx, 1e-6)
        diffusion = self.diffusivity * self.dt / (self.dx ** 2)
        outflow: List[float] = []
        prev = inflow[0]
        for q in inflow:
            routed = prev + courant * (q - prev) + diffusion * (q - 2 * prev + q)
            outflow.append(max(routed, 0.0))
            prev = outflow[-1]
        return outflow


RoutingModelConfig.register("dynamic_wave", DynamicWaveRouting)


__all__ = ["DynamicWaveRouting"]
