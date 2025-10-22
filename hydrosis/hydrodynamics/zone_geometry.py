"""Utilities to assemble solver-ready geometry for a channel zone."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, MutableMapping

import numpy as np
import pandas as pd

from .cross_section import build_geometry_table


def _ensure_dataframe(data: pd.DataFrame | Path) -> pd.DataFrame:
    if isinstance(data, Path):
        return pd.read_csv(data)
    return data.copy()


def load_zone_centerline(centerline: pd.DataFrame | Path, zone_id: str) -> pd.DataFrame:
    """Return ordered centerline profile for a given zone."""

    df = _ensure_dataframe(centerline)
    zone = df[df["zone_id"] == zone_id].copy()
    if zone.empty:
        raise ValueError(f"Zone {zone_id} not found in centerline data.")

    zone.sort_values("global_station_m", inplace=True)
    zone.reset_index(drop=True, inplace=True)

    bed = zone.get("global_corrected_elevation_m")
    if bed is None:
        bed = zone.get("local_corrected_elevation_m")
    if bed is None:
        bed = zone.get("smoothed_elevation_m")
    if bed is None:
        raise KeyError("Centerline data missing corrected elevation columns.")

    zone["bed_elevation_m"] = bed.astype(float)
    station = zone["global_station_m"].to_numpy(dtype=float)
    bed_vals = zone["bed_elevation_m"].to_numpy(dtype=float)

    if bed_vals.size > 1:
        gradient = np.gradient(bed_vals, station, edge_order=1)
        slope = -gradient
        if slope.size > 1:
            slope[0] = slope[1]
    else:
        slope = np.array([0.0], dtype=float)

    zone["centerline_bed_slope"] = slope
    zone["chainage_m"] = station - station.min()
    return zone


@dataclass(slots=True)
class ZoneGeometry:
    zone_id: str
    centerline: pd.DataFrame
    cross_section_table: pd.DataFrame
    metadata: Mapping[str, float]


def build_zone_geometry(
    *,
    zone_id: str,
    centerline: pd.DataFrame | Path,
    cross_sections: pd.DataFrame | Path,
    mannings_n: float = 0.04,
    min_depth: float = 0.05,
    depth_step: float = 0.25,
    padding_depth: float = 0.5,
    max_depth: float | None = None,
    active_half_width: float | None = 75.0,
    depth_cap: float | None = 20.0,
) -> ZoneGeometry:
    """Assemble centerline and cross-section data into a geometry package."""

    centerline_df = load_zone_centerline(centerline, zone_id=zone_id)
    cross_df = _ensure_dataframe(cross_sections)
    metrics = build_geometry_table(
        cross_df,
        zone_id,
        mannings_n=mannings_n,
        min_depth=min_depth,
        depth_step=depth_step,
        padding_depth=padding_depth,
        max_depth=max_depth,
        active_half_width=active_half_width,
        depth_cap=depth_cap,
    )

    if metrics.empty:
        raise ValueError(f"No cross-section samples available for zone {zone_id}.")

    if "global_station_m" not in metrics.columns:
        metrics["global_station_m"] = metrics["station_global_m"]

    attach = centerline_df[["global_station_m", "bed_elevation_m", "centerline_bed_slope"]]
    metrics = metrics.merge(attach, on="global_station_m", how="left", suffixes=("", "_centerline"))
    metrics.rename(
        columns={
            "bed_elevation_m_centerline": "centerline_bed_elevation_m",
        },
        inplace=True,
    )

    missing = metrics["centerline_bed_slope"].isna()
    if missing.any():
        raise ValueError("Cross-section samples missing matching centerline entries.")

    meta: MutableMapping[str, float] = {
        "station_start_m": float(centerline_df["global_station_m"].min()),
        "station_end_m": float(centerline_df["global_station_m"].max()),
        "channel_length_m": float(centerline_df["chainage_m"].max()),
        "cross_section_count": float(metrics["station_global_m"].nunique()),
        "depth_step_m": float(depth_step),
        "default_mannings_n": float(mannings_n),
    }

    return ZoneGeometry(
        zone_id=zone_id,
        centerline=centerline_df,
        cross_section_table=metrics,
        metadata=meta,
    )
