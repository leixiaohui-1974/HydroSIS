"""A simple, no-op runoff model for demonstration and testing."""
from __future__ import annotations

from typing import TYPE_CHECKING, List

from .base import RunoffModel, RunoffModelConfig

if TYPE_CHECKING:
    from ..model import Subbasin


class SimpleRunoff(RunoffModel):
    """A simple runoff model that passes precipitation through as runoff."""

    def simulate(
        self, subbasin: "Subbasin", precipitation: List[float], initial_storage: float | None = None
    ) -> tuple[List[float], float]:
        """Simply returns the precipitation as runoff."""
        return precipitation, 0.0


RunoffModelConfig.register("simple", SimpleRunoff)