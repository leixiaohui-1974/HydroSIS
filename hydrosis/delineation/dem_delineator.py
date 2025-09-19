"""DEM based delineation logic for HydroSIS."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

from ..model import Subbasin


@dataclass
class DelineationConfig:
    dem_path: Path
    pour_points_path: Path
    accumulation_threshold: float = 1000.0
    burn_streams_path: Optional[Path] = None
    precomputed_subbasins: Optional[List[Mapping[str, object]]] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "DelineationConfig":
        return cls(
            dem_path=Path(data["dem_path"]),
            pour_points_path=Path(data["pour_points_path"]),
            accumulation_threshold=float(data.get("accumulation_threshold", 1000.0)),
            burn_streams_path=Path(data["burn_streams_path"]) if data.get("burn_streams_path") else None,
            precomputed_subbasins=list(data.get("precomputed_subbasins", [])) or None,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "dem_path": str(self.dem_path),
            "pour_points_path": str(self.pour_points_path),
            "accumulation_threshold": self.accumulation_threshold,
            "burn_streams_path": str(self.burn_streams_path) if self.burn_streams_path else None,
            "precomputed_subbasins": list(self.precomputed_subbasins) if self.precomputed_subbasins else None,
        }

    def to_subbasins(self) -> List[Subbasin]:
        if self.precomputed_subbasins is None:
            raise RuntimeError(
                "Automatic DEM processing is not implemented in this lightweight build."
                " Provide `precomputed_subbasins` in the configuration."
            )

        subbasins: List[Subbasin] = []
        for entry in self.precomputed_subbasins:
            subbasins.append(
                Subbasin(
                    id=str(entry["id"]),
                    area_km2=float(entry.get("area_km2", 0.0)),
                    downstream=entry.get("downstream"),
                    parameters=dict(entry.get("parameters", {})),
                )
            )
        return subbasins


__all__ = ["DelineationConfig"]
