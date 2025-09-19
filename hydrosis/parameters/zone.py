"""Parameter zoning based on hydrological control structures."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set

from ..model import Subbasin


@dataclass
class ParameterZone:
    id: str
    description: str
    controllers: Sequence[str]
    controlled_subbasins: Sequence[str]
    parameters: Dict[str, float]


@dataclass
class ParameterZoneConfig:
    id: str
    description: str
    control_points: Sequence[str]
    parameters: Mapping[str, float]
    explicit_subbasins: Optional[Sequence[str]] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ParameterZoneConfig":
        return cls(
            id=data["id"],
            description=data.get("description", ""),
            control_points=list(data.get("control_points", [])),
            parameters=dict(data.get("parameters", {})),
            explicit_subbasins=list(data.get("explicit_subbasins", [])) or None,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "description": self.description,
            "control_points": list(self.control_points),
            "parameters": dict(self.parameters),
            "explicit_subbasins": list(self.explicit_subbasins) if self.explicit_subbasins else None,
        }


class ParameterZoneBuilder:
    """Builds parameter zones while avoiding overlap in upstream areas."""

    @staticmethod
    def from_config(
        configs: Iterable[ParameterZoneConfig],
        subbasins: Iterable[Subbasin],
    ) -> List[ParameterZone]:
        sub_list = list(subbasins)
        sub_dict = {sub.id: sub for sub in sub_list}
        upstream_index: Dict[str, List[str]] = {}
        for sub in sub_list:
            if sub.downstream:
                upstream_index.setdefault(sub.downstream, []).append(sub.id)

        assigned: Set[str] = set()
        zones: List[ParameterZone] = []
        for cfg in configs:
            candidates: Set[str] = set(cfg.explicit_subbasins or [])
            for control in cfg.control_points:
                if control not in sub_dict:
                    raise KeyError(f"Control point {control} not in delineated subbasins")
                candidates |= ParameterZoneBuilder._collect_upstream(control, upstream_index)

            unique = sorted(candidates - assigned)
            assigned.update(unique)
            zones.append(
                ParameterZone(
                    id=cfg.id,
                    description=cfg.description,
                    controllers=list(cfg.control_points),
                    controlled_subbasins=unique,
                    parameters=dict(cfg.parameters),
                )
            )
        return zones

    @staticmethod
    def _collect_upstream(node: str, upstream_index: Mapping[str, Sequence[str]]) -> Set[str]:
        stack = [node]
        visited: Set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            for upstream in upstream_index.get(current, []):
                stack.append(upstream)
        return visited


__all__ = ["ParameterZone", "ParameterZoneBuilder", "ParameterZoneConfig"]
