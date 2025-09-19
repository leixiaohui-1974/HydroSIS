"""Flow routing abstractions for HydroSIS."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Type

from ..model import Subbasin


class RoutingModel:
    """Base class for flow routing algorithms."""

    def __init__(self, parameters: Mapping[str, float]):
        self.parameters = dict(parameters)

    def route(self, subbasin: Subbasin, inflow: List[float]) -> List[float]:
        raise NotImplementedError


@dataclass
class RoutingModelConfig:
    """Configuration for constructing routing models."""

    id: str
    model_type: str
    parameters: Dict[str, float]

    REGISTRY: Dict[str, Type[RoutingModel]] = {}

    @classmethod
    def register(cls, key: str, model: Type[RoutingModel]) -> None:
        cls.REGISTRY[key] = model

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "RoutingModelConfig":
        return cls(
            id=data["id"],
            model_type=data["model_type"],
            parameters=dict(data.get("parameters", {})),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "model_type": self.model_type,
            "parameters": dict(self.parameters),
        }

    def build(self) -> RoutingModel:
        try:
            model_cls = self.REGISTRY[self.model_type]
        except KeyError as exc:
            raise KeyError(f"Routing model {self.model_type} not registered") from exc
        return model_cls(self.parameters)
