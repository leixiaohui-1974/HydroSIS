"""Variable Infiltration Capacity (VIC) inspired runoff model."""
from __future__ import annotations

from typing import TYPE_CHECKING, List

from .base import RunoffModel, RunoffModelConfig
from ..validation import validate_positive, validate_probability

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..model import Subbasin


class VICRunoff(RunoffModel):
    """Simplified VIC style runoff generator.

    The model keeps track of three soil moisture storages (surface, root
    zone and deep zone) and applies an ARNO style variable infiltration
    curve to partition precipitation between fast runoff and infiltration.
    The implementation is intentionally compact while still exposing key
    VIC parameters so users can calibrate the behaviour through the
    configuration file.
    """

    def __init__(self, parameters):
        super().__init__(parameters)
        self.infiltration_shape = max(1e-6, float(self.parameters.get("infiltration_shape", 0.3)))
        self.max_soil_moisture = max(1e-6, float(self.parameters.get("max_soil_moisture", 150.0)))
        self.baseflow_coefficient = max(0.0, float(self.parameters.get("baseflow_coefficient", 0.005)))
        self.recession = min(1.0, max(0.0, float(self.parameters.get("recession", 0.95))))
        self.surface_storage = float(self.parameters.get("initial_surface", 5.0))
        self.root_storage = float(self.parameters.get("initial_root", 50.0))
        self.deep_storage = float(self.parameters.get("initial_deep", 20.0))

    def validate_parameters(self) -> None:
        """Validate VIC model parameters.

        Validates:
            - infiltration_shape: Must be > 0
            - max_soil_moisture: Must be > 0
            - baseflow_coefficient: Must be >= 0
            - recession: Must be in [0, 1]
            - initial_surface, initial_root, initial_deep: Must be >= 0
        """
        inf_shape = float(self.parameters.get("infiltration_shape", 0.3))
        validate_positive("infiltration_shape", inf_shape, strict=True)

        max_sm = float(self.parameters.get("max_soil_moisture", 150.0))
        validate_positive("max_soil_moisture", max_sm, strict=True)

        bf_coeff = float(self.parameters.get("baseflow_coefficient", 0.005))
        validate_positive("baseflow_coefficient", bf_coeff, strict=False)

        recession = float(self.parameters.get("recession", 0.95))
        validate_probability("recession", recession)

        init_surf = float(self.parameters.get("initial_surface", 5.0))
        validate_positive("initial_surface", init_surf, strict=False)

        init_root = float(self.parameters.get("initial_root", 50.0))
        validate_positive("initial_root", init_root, strict=False)

        init_deep = float(self.parameters.get("initial_deep", 20.0))
        validate_positive("initial_deep", init_deep, strict=False)

    def simulate(self, subbasin: "Subbasin", precipitation: List[float]) -> List[float]:
        runoff: List[float] = []
        for p in precipitation:
            available = p + self.surface_storage
            capacity = self.max_soil_moisture - self.root_storage
            frac = min(1.0, max(0.0, (available / self.max_soil_moisture) ** (1.0 / self.infiltration_shape)))
            infiltration = frac * min(capacity, available)
            quick_runoff = max(0.0, available - infiltration)

            self.surface_storage = max(0.0, self.surface_storage + p - quick_runoff)
            self.root_storage = min(self.max_soil_moisture, self.root_storage + infiltration)

            percolation = max(0.0, 0.1 * self.root_storage)
            self.root_storage -= percolation
            self.deep_storage += percolation

            baseflow = self.baseflow_coefficient * self.deep_storage
            self.deep_storage = max(0.0, self.deep_storage * self.recession)

            total = (quick_runoff + baseflow) * subbasin.area_km2
            runoff.append(total)
        return runoff


RunoffModelConfig.register("vic", VICRunoff)


__all__ = ["VICRunoff"]
