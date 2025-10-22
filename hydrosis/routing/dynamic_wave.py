"""Simplified dynamic wave routing implementation."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List

from .base import RoutingModel, RoutingModelConfig
from ..validation import validate_positive, validate_integer

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..model import Subbasin


class DynamicWaveRouting(RoutingModel):
    """A kinematic/dynamic wave hybrid solver using an explicit scheme."""

    def __init__(self, parameters):
        super().__init__(parameters)
        self.dt = float(self.parameters.get("time_step", 1.0))
        self.reach_length = max(1e-6, float(self.parameters.get("reach_length", 5.0)))
        self.segments = max(1, int(self.parameters.get("segments", 5)))
        self.dx = self.reach_length / self.segments
        self.wave_celerity = float(self.parameters.get("wave_celerity", 1.5))
        self.diffusivity = max(1e-6, float(self.parameters.get("diffusivity", 0.05)))
        self._base_substeps = max(1, int(self.parameters.get("substeps", 1)))
        self.auto_substeps = bool(self.parameters.get("auto_substeps", False))
        self.max_substeps = max(self._base_substeps, int(self.parameters.get("max_substeps", 1)))
        self.substeps = self._determine_substeps()

    def validate_parameters(self) -> None:
        """Validate Dynamic Wave routing parameters.

        Validates:
            - time_step: Must be > 0
            - reach_length: Must be > 0
            - segments: Must be >= 1
            - wave_celerity: Must be > 0
            - diffusivity: Must be > 0
            - substeps: Must be >= 1
            - max_substeps: Must be >= substeps
        """
        dt = float(self.parameters.get("time_step", 1.0))
        validate_positive("time_step", dt, strict=True)

        reach_len = float(self.parameters.get("reach_length", 5.0))
        validate_positive("reach_length", reach_len, strict=True)

        segments = int(self.parameters.get("segments", 5))
        validate_integer("segments", segments, min_value=1)

        wave_cel = float(self.parameters.get("wave_celerity", 1.5))
        validate_positive("wave_celerity", wave_cel, strict=True)

        diffusivity = float(self.parameters.get("diffusivity", 0.05))
        validate_positive("diffusivity", diffusivity, strict=True)

        substeps = int(self.parameters.get("substeps", 1))
        validate_integer("substeps", substeps, min_value=1)

        max_substeps = int(self.parameters.get("max_substeps", 1))
        validate_integer("max_substeps", max_substeps, min_value=substeps)

    def _stability_terms(self, step_dt: float | None = None) -> Dict[str, float]:
        dt = float(step_dt or self.dt)
        courant = self.wave_celerity * dt / max(self.dx, 1e-12)
        diffusion = self.diffusivity * dt / max(self.dx**2, 1e-12)
        return {"courant": courant, "diffusion": diffusion}

    def _determine_substeps(self) -> int:
        if not self.auto_substeps:
            return self._base_substeps
        terms = self._stability_terms()
        courant = terms["courant"]
        diffusion = terms["diffusion"]
        if courant <= 1.0 and diffusion <= 0.5:
            return self._base_substeps
        required = max(courant, diffusion / 0.5)
        return min(self.max_substeps, max(self._base_substeps, int(math.ceil(required))))

    def route(self, subbasin: "Subbasin", inflow: List[float]) -> List[float]:
        if not inflow:
            return []

        substeps = max(1, self.substeps)
        sub_dt = self.dt / substeps
        terms = self._stability_terms(sub_dt)
        smoothing = max(0.0, min(1.0, terms["courant"] + 2.0 * terms["diffusion"]))

        outflow: List[float] = []
        prev = inflow[0]
        for q in inflow:
            state = prev
            for _ in range(substeps):
                state = state + smoothing * (q - state)
                state = max(state, 0.0)
            outflow.append(state)
            prev = state
        return outflow

    def resolved_parameters(self, subbasin: "Subbasin") -> Dict[str, float]:
        terms = self._stability_terms()
        return {
            "time_step": self.dt,
            "reach_length": self.reach_length,
            "segments": float(self.segments),
            "dx": self.dx,
            "wave_celerity": self.wave_celerity,
            "diffusivity": self.diffusivity,
            "courant": terms["courant"],
            "diffusion": terms["diffusion"],
            "substeps": float(self.substeps),
        }


RoutingModelConfig.register("dynamic_wave", DynamicWaveRouting)


__all__ = ["DynamicWaveRouting"]
