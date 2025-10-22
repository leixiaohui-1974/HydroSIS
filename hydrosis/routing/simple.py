"""A simple, no-op routing model for demonstration and testing."""
from __future__ import annotations

from typing import TYPE_CHECKING, List

from .base import RoutingModel, RoutingModelConfig

if TYPE_CHECKING:
    from ..model import Subbasin


class SimpleRouting(RoutingModel):
    """A simple routing model that passes inflow through as outflow."""

    def route(self, subbasin: "Subbasin", inflow: List[float]) -> List[float]:
        """Simply returns the inflow as outflow."""
        return inflow


RoutingModelConfig.register("simple", SimpleRouting)