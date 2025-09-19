"""Lag-based translation routing."""
from __future__ import annotations

from typing import TYPE_CHECKING, List

from .base import RoutingModel, RoutingModelConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..model import Subbasin


class LagRouting(RoutingModel):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.lag_steps = int(self.parameters.get("lag_steps", 1))

    def route(self, subbasin: "Subbasin", inflow: List[float]) -> List[float]:
        padding = [0.0] * self.lag_steps
        return padding + inflow[:-self.lag_steps] if self.lag_steps < len(inflow) else padding


RoutingModelConfig.register("lag", LagRouting)
