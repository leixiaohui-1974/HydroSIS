"""Conceptual HBV style runoff implementation."""
from __future__ import annotations

from typing import TYPE_CHECKING, List

from .base import RunoffModel, RunoffModelConfig
from ..validation import validate_positive, validate_probability

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..model import Subbasin


class HBVRunoff(RunoffModel):
    """A minimalist HBV style bucket model.

    The model keeps snow, soil and groundwater storages and provides a
    degree-day melt option so configurations that rely on HBV parameters
    can be ported with limited effort.
    """

    def __init__(self, parameters):
        # Store parameters first (needed for attribute initialization)
        self.parameters = dict(parameters)

        # Support both uppercase (workflow convention) and lowercase parameter names
        # Degree-day melt parameters
        self.degree_day_factor = float(
            self.parameters.get("degree_day_factor") or
            self.parameters.get("CFMAX") or
            self.parameters.get("cfmax") or 3.0
        )
        self.snow_threshold = float(
            self.parameters.get("snow_threshold") or
            self.parameters.get("TT") or
            self.parameters.get("tt") or 0.0
        )

        # Soil moisture parameters
        self.field_capacity = float(
            self.parameters.get("field_capacity") or
            self.parameters.get("FC") or
            self.parameters.get("fc") or 100.0
        )
        self.beta = max(1e-6, float(
            self.parameters.get("beta") or
            self.parameters.get("BETA") or 1.0
        ))

        # Recession coefficients
        self.k0 = float(
            self.parameters.get("k0") or
            self.parameters.get("K0") or 0.15
        )
        self.k1 = float(
            self.parameters.get("k1") or
            self.parameters.get("K1") or 0.05
        )
        self.k2 = float(
            self.parameters.get("k2") or
            self.parameters.get("K2") or 0.01
        )

        # Percolation rate
        self.percolation = float(
            self.parameters.get("percolation") or
            self.parameters.get("PERC") or
            self.parameters.get("perc") or 2.0
        )

        # Initial states
        self.snow = float(self.parameters.get("initial_snow", 0.0))
        # Set initial soil to 50% of field capacity if not specified
        init_soil = self.parameters.get("initial_soil")
        if init_soil is None or init_soil == 0:
            self.soil = self.field_capacity * 0.5
        else:
            self.soil = float(init_soil)
        self.upper = float(self.parameters.get("initial_upper", 5.0))
        self.lower = float(self.parameters.get("initial_lower", 20.0))

        # Now call parent init which will call validate_parameters()
        # Don't call super().__init__() since we already set self.parameters
        # and we'll call validate manually
        self.validate_parameters()

    def validate_parameters(self) -> None:
        """Validate HBV model parameters.

        Validates:
            - degree_day_factor/CFMAX: Must be >= 0
            - field_capacity/FC: Must be > 0
            - beta/BETA: Must be > 0
            - k0/K0, k1/K1, k2/K2: Recession coefficients, must be in [0, 1]
            - percolation/PERC: Must be >= 0
            - initial states: Must be >= 0

        Note: Supports both uppercase (workflow) and lowercase parameter names.
        """
        # Validate using the actual values that were loaded (after name resolution)
        validate_positive("degree_day_factor", self.degree_day_factor, strict=False)
        validate_positive("field_capacity", self.field_capacity, strict=True)
        validate_positive("beta", self.beta, strict=True)
        validate_probability("k0", self.k0)
        validate_probability("k1", self.k1)
        validate_probability("k2", self.k2)
        validate_positive("percolation", self.percolation, strict=False)
        validate_positive("initial_snow", self.snow, strict=False)
        validate_positive("initial_soil", self.soil, strict=False)
        validate_positive("initial_upper", self.upper, strict=False)
        validate_positive("initial_lower", self.lower, strict=False)

    def simulate(self, subbasin: "Subbasin", precipitation: List[float]) -> List[float]:
        flows: List[float] = []
        for p in precipitation:
            rainfall = max(0.0, p - self.snow_threshold)
            snowfall = max(0.0, p - rainfall)
            self.snow += snowfall

            melt = self.degree_day_factor * max(0.0, rainfall - self.snow_threshold)
            melt = min(melt, self.snow)
            self.snow -= melt

            effective_precip = rainfall + melt
            soil_deficit = max(0.0, self.field_capacity - self.soil)
            recharge = effective_precip * ((self.soil / self.field_capacity) ** self.beta)
            recharge = min(recharge, soil_deficit)
            self.soil += effective_precip - recharge

            quickflow = self.k0 * self.upper
            # Limit percolation to available water in upper reservoir
            actual_percolation = min(self.percolation, max(0.0, self.upper + recharge - quickflow))
            self.upper += recharge - quickflow - actual_percolation
            self.upper = max(0.0, self.upper)  # Ensure non-negative
            self.lower += actual_percolation - self.k2 * self.lower
            self.lower = max(0.0, self.lower)  # Ensure non-negative
            baseflow = self.k1 * self.upper + self.k2 * self.lower

            flows.append((quickflow + baseflow) * subbasin.area_km2)
        return flows


RunoffModelConfig.register("hbv", HBVRunoff)


__all__ = ["HBVRunoff"]
