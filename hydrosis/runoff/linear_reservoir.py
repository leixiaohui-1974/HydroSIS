"""A simple linear reservoir runoff production model."""
from __future__ import annotations

from typing import TYPE_CHECKING, List

from .base import RunoffModel, RunoffModelConfig
from ..validation import validate_positive, validate_probability

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..model import Subbasin


class LinearReservoirRunoff(RunoffModel):
    """Conceptual runoff model using a single linear reservoir."""

    def __init__(self, parameters):
        super().__init__(parameters)
        self.recession = float(self.parameters.get("recession", 0.9))
        self.conversion = float(self.parameters.get("conversion", 1.0))
        self.state = float(self.parameters.get("initial_storage", 0.0))

    def validate_parameters(self) -> None:
        """Validate Linear Reservoir model parameters.

        Validates:
            - recession: Must be in [0, 1]
            - conversion: Must be >= 0
            - initial_storage: Must be >= 0
        """
        recession = float(self.parameters.get("recession", 0.9))
        validate_probability("recession", recession)

        conversion = float(self.parameters.get("conversion", 1.0))
        validate_positive("conversion", conversion, strict=False)

        init_storage = float(self.parameters.get("initial_storage", 0.0))
        validate_positive("initial_storage", init_storage, strict=False)

    def simulate(self, subbasin: "Subbasin", precipitation: List[float]) -> List[float]:
        flows: List[float] = []
        for p in precipitation:
            self.state = self.state * self.recession + p * self.conversion
            direct_runoff = (1 - self.recession) * self.state
            flows.append(direct_runoff * subbasin.area_km2)
        return flows


RunoffModelConfig.register("linear_reservoir", LinearReservoirRunoff)
