"""Runoff generation model abstractions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Dict, List, Mapping, Type

if TYPE_CHECKING:  # pragma: no cover - for typing only
    from ..model import Subbasin


class RunoffModel:
    """Base class for runoff generation algorithms."""

    def __init__(self, parameters: Mapping[str, float]):
        self.parameters = dict(parameters)

    def simulate(self, subbasin: "Subbasin", precipitation: List[float]) -> List[float]:
        raise NotImplementedError


@dataclass
class RunoffModelConfig:
    """Configuration for constructing a runoff model."""

    id: str
    model_type: str
    parameters: Dict[str, float]

    REGISTRY: ClassVar[Dict[str, Type[RunoffModel]]] = {}

    @classmethod
    def register(cls, key: str, model: Type[RunoffModel]) -> None:
        cls.REGISTRY[key] = model

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "RunoffModelConfig":
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

    def build(self) -> RunoffModel:
        try:
            model_cls = self.REGISTRY[self.model_type]
        except KeyError as exc:
            raise KeyError(f"Runoff model {self.model_type} not registered") from exc
        return model_cls(self.parameters)
