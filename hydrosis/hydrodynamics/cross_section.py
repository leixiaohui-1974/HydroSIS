"""Utilities for working with extracted channel cross-sections.

The helpers in this module are deliberately lightweight: they convert the
discrete sample points from ``channel_cross_sections_corrected.csv`` into
analytic functions for wetted area, top width, conveyance, etc.  These pieces
will be re-used by the upcoming 1D hydraulic solver to derive rating curves
and segment-specific wave parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

GRAVITY = 9.80665  # m/s^2


@dataclass(slots=True)
class CrossSection:
    """Representation of a single cross-section sampled at equal spacing."""

    zone_id: str
    segment_id: str
    station_local_m: float
    station_global_m: float
    offsets: np.ndarray  # distance_from_center_m
    elevations: np.ndarray  # elevation_m
    mannings_n: float = 0.04

    @property
    def bed_elevation(self) -> float:
        return float(np.min(self.elevations))

    def _submerged_profile(self, stage: float) -> tuple[np.ndarray, np.ndarray]:
        """Return submerged segment lengths and depths for a given stage elevation."""

        depth = stage - self.elevations
        wet_mask = depth > 0.0
        if not np.any(wet_mask):
            return np.empty(0), np.empty(0)

        x = self.offsets[wet_mask]
        z = depth[wet_mask]
        return x, z

    def top_width(self, stage: float) -> float:
        x, z = self._submerged_profile(stage)
        if x.size < 2:
            return 0.0
        return float(x[-1] - x[0])

    def area(self, stage: float) -> float:
        x, z = self._submerged_profile(stage)
        if x.size < 2:
            return 0.0
        return float(np.trapz(z, x))

    def wetted_perimeter(self, stage: float) -> float:
        x, depth = self._submerged_profile(stage)
        if x.size < 2:
            return 0.0
        z = stage - depth
        dx = np.diff(x)
        dz = np.diff(z)
        return float(np.sum(np.hypot(dx, dz)))

    def hydraulic_radius(self, stage: float) -> float:
        area = self.area(stage)
        wp = self.wetted_perimeter(stage)
        if area <= 0.0 or wp <= 0.0:
            return 0.0
        return float(area / wp)

    def conveyance(self, stage: float, energy_slope: float) -> float:
        area = self.area(stage)
        if area <= 0.0:
            return 0.0
        radius = self.hydraulic_radius(stage)
        return (
            (1.0 / self.mannings_n)
            * area
            * (radius ** (2.0 / 3.0))
            * np.sqrt(max(energy_slope, 1e-8))
        )

    def discharge(self, stage: float, energy_slope: float) -> float:
        """Compute discharge with Manning's equation for a given stage elevation."""

        area = self.area(stage)
        if area <= 0.0:
            return 0.0
        conveyance = self.conveyance(stage, energy_slope)
        return float(conveyance)

    def stage_for_discharge(
        self,
        target_q: float,
        energy_slope: float,
        initial_guess: float | None = None,
        max_iterations: int = 50,
        tolerance: float = 1e-3,
    ) -> float:
        """Solve for stage elevation that matches the target discharge.

        Uses a simple Newton-Raphson iteration on the Manning formulation.
        """

        bed = self.bed_elevation
        stage = float(initial_guess or (bed + 1.0))
        stage = max(stage, bed + 0.01)
        target = max(target_q, 0.0)

        for _ in range(max_iterations):
            q = self.discharge(stage, energy_slope)
            if q <= 0.0:
                stage += 0.5
                continue
            area = self.area(stage)
            top_width = self.top_width(stage)
            if top_width <= 0.0:
                stage += 0.5
                continue
            dq_dh = q * (1.0 / area + (2.0 / 3.0) / (self.hydraulic_radius(stage) * top_width))
            stage = stage + (target - q) / max(dq_dh, 1e-6)
            if abs(target - q) < tolerance:
                return float(stage)

        return float(stage)

    def sample_geometry(
        self,
        *,
        min_depth: float = 0.05,
        depth_step: float = 0.25,
        padding_depth: float = 0.5,
        max_depth: float | None = None,
        active_half_width: float | None = 75.0,
        depth_cap: float | None = 20.0,
    ) -> pd.DataFrame:
        """Tabulate wetted geometry across a depth range for this section."""

        bed = self.bed_elevation
        crest_source = self.elevations
        if active_half_width is not None:
            mask = np.abs(self.offsets) <= active_half_width
            if np.count_nonzero(mask) >= 3:
                crest_source = self.elevations[mask]
        crest = float(np.max(crest_source))
        target_max = max_depth if max_depth is not None else (crest - bed + padding_depth)
        if depth_cap is not None:
            target_max = min(target_max, depth_cap)
        target_max = max(target_max, min_depth)

        depths = np.arange(min_depth, target_max + (depth_step * 0.5), depth_step, dtype=float)
        records: list[dict[str, float]] = []
        for depth in depths:
            stage = bed + depth
            area = self.area(stage)
            if area <= 0.0:
                continue
            wetted = self.wetted_perimeter(stage)
            if wetted <= 0.0:
                continue
            records.append(
                {
                    "zone_id": self.zone_id,
                    "segment_id": self.segment_id,
                    "station_local_m": self.station_local_m,
                    "station_global_m": self.station_global_m,
                    "bed_elevation_m": bed,
                    "stage_elevation_m": stage,
                    "depth_m": depth,
                    "area_m2": area,
                    "wetted_perimeter_m": wetted,
                    "top_width_m": self.top_width(stage),
                    "hydraulic_radius_m": self.hydraulic_radius(stage),
                    "mannings_n": self.mannings_n,
                }
            )

        if not records:
            return pd.DataFrame(
                columns=[
                    "zone_id",
                    "segment_id",
                    "station_local_m",
                    "station_global_m",
                    "bed_elevation_m",
                    "stage_elevation_m",
                    "depth_m",
                    "area_m2",
                    "wetted_perimeter_m",
                    "top_width_m",
                    "hydraulic_radius_m",
                    "mannings_n",
                ]
            )

        return pd.DataFrame.from_records(records)


def load_cross_sections(
    df: pd.DataFrame,
    zone_id: str,
    mannings_n: float = 0.04,
) -> list[CrossSection]:
    """Build ``CrossSection`` objects for a given zone from the corrected CSV."""

    zone_df = df[df["zone_id"] == zone_id].copy()
    if zone_df.empty:
        return []

    sections: list[CrossSection] = []
    for (segment_id, station_global), group in zone_df.groupby(["segment_id", "global_station_m"]):
        ordered = group.sort_values("distance_from_center_m")
        station_local = float(group["station_m"].iloc[0])
        sections.append(
            CrossSection(
                zone_id=zone_id,
                segment_id=str(segment_id),
                station_local_m=station_local,
                station_global_m=float(station_global),
                offsets=ordered["distance_from_center_m"].to_numpy(dtype=float),
                elevations=ordered["elevation_m"].to_numpy(dtype=float),
                mannings_n=mannings_n,
            )
        )
    sections.sort(key=lambda cs: cs.station_global_m)
    return sections


def sample_cross_section_geometry(
    sections: Sequence[CrossSection],
    *,
    min_depth: float = 0.05,
    depth_step: float = 0.25,
    padding_depth: float = 0.5,
    max_depth: float | None = None,
    active_half_width: float | None = 75.0,
    depth_cap: float | None = 20.0,
) -> pd.DataFrame:
    """Sample wetted geometry for all sections in a zone."""

    tables: list[pd.DataFrame] = []
    for section in sections:
        table = section.sample_geometry(
            min_depth=min_depth,
            depth_step=depth_step,
            padding_depth=padding_depth,
            max_depth=max_depth,
            active_half_width=active_half_width,
            depth_cap=depth_cap,
        )
        if not table.empty:
            tables.append(table)

    if not tables:
        return pd.DataFrame()

    combined = pd.concat(tables, ignore_index=True)
    combined.sort_values(["station_global_m", "depth_m"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


def build_geometry_table(
    cross_sections: pd.DataFrame,
    zone_id: str,
    *,
    mannings_n: float = 0.04,
    min_depth: float = 0.05,
    depth_step: float = 0.25,
    padding_depth: float = 0.5,
    max_depth: float | None = None,
    active_half_width: float | None = 75.0,
    depth_cap: float | None = 20.0,
) -> pd.DataFrame:
    """Load and sample cross-sections for a zone into a ready-to-use table."""

    sections = load_cross_sections(cross_sections, zone_id=zone_id, mannings_n=mannings_n)
    if not sections:
        return pd.DataFrame()
    return sample_cross_section_geometry(
        sections,
        min_depth=min_depth,
        depth_step=depth_step,
        padding_depth=padding_depth,
        max_depth=max_depth,
        active_half_width=active_half_width,
        depth_cap=depth_cap,
    )
