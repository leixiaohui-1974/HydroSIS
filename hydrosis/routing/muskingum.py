"""Simplified Muskingum routing implementation."""
from __future__ import annotations

from typing import List

from .base import RoutingModel, RoutingModelConfig
from ..model import Subbasin


class MuskingumRouting(RoutingModel):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.k = float(self.parameters.get("travel_time", 12.0))
        self.x = float(self.parameters.get("weighting_factor", 0.2))
        self.dt = float(self.parameters.get("time_step", 1.0))

    def route(self, subbasin: Subbasin, inflow: List[float]) -> List[float]:
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


RoutingModelConfig.register("muskingum", MuskingumRouting)
