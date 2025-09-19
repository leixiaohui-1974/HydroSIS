"""Core HydroSIS distributed hydrological model orchestrating runoff and routing."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional

from .parameters.zone import ParameterZone, ParameterZoneBuilder
from .runoff.base import RunoffModel
from .routing.base import RoutingModel


@dataclass
class Subbasin:
    """Representation of a delineated subbasin unit."""

    id: str
    area_km2: float
    downstream: Optional[str]
    parameters: Dict[str, float] = field(default_factory=dict)

    def update_parameters(self, updates: Mapping[str, float]) -> None:
        """Apply parameter updates to the subbasin."""

        self.parameters.update(updates)


class HydroSISModel:
    """Main interface for running HydroSIS simulations."""

    def __init__(
        self,
        subbasins: Iterable[Subbasin],
        parameter_zones: Iterable[ParameterZone],
        runoff_models: Mapping[str, RunoffModel],
        routing_models: Mapping[str, RoutingModel],
    ) -> None:
        self.subbasins = {sub.id: sub for sub in subbasins}
        self.parameter_zones = {zone.id: zone for zone in parameter_zones}
        self.runoff_models = dict(runoff_models)
        self.routing_models = dict(routing_models)

        self._assign_zone_parameters()

    def _assign_zone_parameters(self) -> None:
        for zone in self.parameter_zones.values():
            for sub_id in zone.controlled_subbasins:
                if sub_id in self.subbasins:
                    self.subbasins[sub_id].update_parameters(zone.parameters)

    @classmethod
    def from_config(cls, config: "ModelConfig") -> "HydroSISModel":
        from .config import ModelConfig  # local import to avoid cycle

        delineated = config.delineation.to_subbasins()
        zones = ParameterZoneBuilder.from_config(config.parameter_zones, delineated)

        runoff_models = {
            run_cfg.id: run_cfg.build()
            for run_cfg in config.runoff_models
        }
        routing_models = {
            route_cfg.id: route_cfg.build()
            for route_cfg in config.routing_models
        }

        return cls(delineated, zones, runoff_models, routing_models)

    def run(self, forcing: Mapping[str, List[float]]) -> Dict[str, List[float]]:
        """Run the distributed hydrological simulation."""

        runoff_results: Dict[str, List[float]] = {}
        for sub_id, subbasin in self.subbasins.items():
            model_key = subbasin.parameters.get("runoff_model")
            if model_key is None:
                raise ValueError(f"Subbasin {sub_id} missing runoff_model parameter")

            runoff_model = self.runoff_models[model_key]
            forcings = forcing.get(sub_id, [])
            runoff_results[sub_id] = runoff_model.simulate(subbasin, forcings)

        routed: Dict[str, List[float]] = {}
        for sub_id, flows in runoff_results.items():
            subbasin = self.subbasins[sub_id]
            model_key = subbasin.parameters.get("routing_model")
            if model_key is None:
                raise ValueError(f"Subbasin {sub_id} missing routing_model parameter")
            routing_model = self.routing_models[model_key]
            routed[sub_id] = routing_model.route(subbasin, flows)

        return routed


__all__ = ["HydroSISModel", "Subbasin"]
