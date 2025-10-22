"""Channel cross-section processing utilities.

This module consolidates the algorithms for:

* loading cross-section data and filtering noise;
* attaching cumulative along-channel distances (`global_station_m`);
* extracting thalweg/centerline profiles; and
* enforcing downstream monotonic behaviour suitable for hydraulic modelling.

It is designed to be reusable by CLI tools and future “product” APIs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


__all__ = [
    "ChannelProfileConfig",
    "PreprocessResult",
    "ChannelProfileResult",
    "load_cross_sections",
    "drop_flat_sections",
    "attach_global_station",
    "build_zone_grid",
    "extract_centerline",
    "enforce_downhill_trend",
    "preprocess_cross_sections",
    "run_channel_profile_model",
    "apply_monotonic_corrections",
    "generate_channel_profiles",
]


@dataclass(slots=True)
class ChannelProfileConfig:
    """Configuration options for channel profile extraction."""

    min_variation: float = 0.1  # metres – min elevation spread to retain a section
    band_width: float = 40.0  # metres – width around channel centre considered “thalweg”
    target_spacing: float = 150.0  # metres – default along-channel grid spacing for gridded outputs
    monotonic_tolerance: float = 0.0  # metres – permitted rise between consecutive stations


@dataclass
class PreprocessResult:
    """Outputs from the preprocessing stage."""

    raw_sections: pd.DataFrame
    cross_sections: pd.DataFrame
    segments: pd.DataFrame
    zones: Sequence[str]
    config: ChannelProfileConfig


@dataclass
class ChannelProfileResult:
    """Outputs from the model (monotonic correction) stage."""

    cross_sections: pd.DataFrame
    centerlines: Dict[str, pd.DataFrame]
    config: ChannelProfileConfig


def load_cross_sections(path: Path, zones: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Load cross-section CSV and optionally filter to selected zones."""
    df = pd.read_csv(path)
    if zones:
        zone_list = list(zones)
        df = df[df["zone_id"].isin(zone_list)].copy()
    if df.empty:
        raise FileNotFoundError(
            f"No cross-section data found in {path} for zones {zones or 'all'}"
        )
    return df


def drop_flat_sections(data: pd.DataFrame, min_variation: float) -> pd.DataFrame:
    """Remove sections whose elevation range is below `min_variation`."""
    variation = (
        data.groupby(["zone_id", "station_m"])["elevation_m"]
        .agg(lambda series: series.max() - series.min())
        .reset_index(name="variation")
    )
    valid = variation[variation["variation"] >= min_variation][["zone_id", "station_m"]]
    cleaned = data.merge(valid, on=["zone_id", "station_m"], how="inner")
    cleaned.sort_values(by=["zone_id", "station_m", "distance_from_center_m"], inplace=True)
    cleaned.reset_index(drop=True, inplace=True)
    return cleaned


def _ensure_dataframe(obj: pd.DataFrame | Path) -> pd.DataFrame:
    if isinstance(obj, Path):
        return pd.read_csv(obj)
    return obj.copy()


def attach_global_station(
    data: pd.DataFrame,
    segments: pd.DataFrame | Path,
    zone_order: Iterable[str],
) -> pd.DataFrame:
    """Attach cumulative chainage (`global_station_m`) to each cross-section row."""
    segments_df = _ensure_dataframe(segments)
    if segments_df.empty:
        raise FileNotFoundError("Segments data is empty, cannot compute global_station_m")

    segments_df["segment_start"] = segments_df["cumulative_length_m"] - segments_df["length_m"]

    zone_lengths = segments_df.groupby("zone_id")["length_m"].sum()
    offset = 0.0
    zone_offsets: Dict[str, float] = {}
    for zone in zone_order:
        zone_offsets[zone] = offset
        offset += float(zone_lengths.get(zone, 0.0))

    merged = data.merge(segments_df[["segment_id", "segment_start"]], on="segment_id", how="left")
    merged["global_station_m"] = (
        merged["station_m"]
        + merged["segment_start"].fillna(0.0)
        + merged["zone_id"].map(zone_offsets).fillna(0.0)
    )
    merged.drop(columns=["segment_start"], inplace=True)
    return merged


def build_zone_grid(
    zone_df: pd.DataFrame,
    *,
    station_field: str = "global_station_m",
    target_spacing: float = 150.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pivot cross-section points into a regular grid for meshing/visualisation."""
    pivot = zone_df.pivot_table(
        index="distance_from_center_m",
        columns=station_field,
        values="elevation_m",
        aggfunc="mean",
    )
    pivot.sort_index(inplace=True)
    pivot.sort_index(axis=1, inplace=True)

    xs = pivot.columns.to_numpy(dtype=float)
    ys = pivot.index.to_numpy(dtype=float)
    Z = pivot.to_numpy(dtype=float)

    if xs.size > 1:
        diffs = np.diff(xs)
        if diffs.size and np.median(diffs) > target_spacing:
            spacing = target_spacing
            steps = int(np.ceil((xs[-1] - xs[0]) / spacing)) + 1
            new_xs = np.linspace(xs[0], xs[-1], steps, dtype=float)
            resampled = np.empty((Z.shape[0], new_xs.size), dtype=float)
            for idx, row in enumerate(Z):
                resampled[idx] = np.interp(new_xs, xs, row)
            xs = new_xs
            Z = resampled

    return xs, ys, Z


def extract_centerline(zone_df: pd.DataFrame, band_width: float) -> pd.DataFrame:
    """Extract the thalweg centreline based on minimum elevation within the band."""
    subset = zone_df[zone_df["distance_from_center_m"].abs() <= band_width].copy()
    if subset.empty:
        subset = zone_df.copy()

    subset = subset.dropna(subset=["global_station_m"])
    if subset.empty:
        raise ValueError("Missing global_station_m column, cannot extract centerline")

    center_idx = (
        subset.groupby("global_station_m")["distance_from_center_m"]
        .apply(lambda s: s.abs().idxmin())
        .to_numpy()
    )
    centerline = subset.loc[center_idx].copy()
    centerline.sort_values("global_station_m", inplace=True)
    centerline.rename(columns={"elevation_m": "base_elevation_m"}, inplace=True)

    rolling_median = (
        centerline["base_elevation_m"].rolling(window=5, center=True, min_periods=1).median()
    )
    smoothed = rolling_median.rolling(window=7, center=True, min_periods=1).mean()
    centerline["smoothed_elevation_m"] = smoothed
    centerline["elevation_m"] = centerline["smoothed_elevation_m"]
    return centerline


def enforce_downhill_trend(values: np.ndarray, tolerance: float = 0.0) -> np.ndarray:
    """Enforce a downstream non-increasing trend on a 1-D elevation array."""
    if values.size == 0:
        return values
    monotonic = values.copy()
    for idx in range(1, monotonic.size):
        allowed = monotonic[idx - 1] - tolerance
        if monotonic[idx] > allowed:
            monotonic[idx] = allowed
    return monotonic


def preprocess_cross_sections(
    cross_sections: pd.DataFrame | Path,
    segments: pd.DataFrame | Path,
    zones: Iterable[str],
    config: Optional[ChannelProfileConfig] = None,
) -> PreprocessResult:
    """Run the preprocessing stage: clean + attach global station."""
    cfg = config or ChannelProfileConfig()
    raw_df = _ensure_dataframe(cross_sections)
    filtered = drop_flat_sections(raw_df, cfg.min_variation)
    filtered = attach_global_station(filtered, segments, zones)
    segments_df = _ensure_dataframe(segments)
    return PreprocessResult(
        raw_sections=raw_df,
        cross_sections=filtered,
        segments=segments_df,
        zones=list(zones),
        config=cfg,
    )


def apply_monotonic_corrections(
    data: pd.DataFrame,
    zones: Iterable[str],
    *,
    band_width: float,
    tolerance: float = 0.0,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Apply centreline extraction and monotonic corrections for each zone."""
    zone_store: Dict[str, Dict[str, pd.DataFrame]] = {}
    for zone in zones:
        zone_df = data[data["zone_id"] == zone].copy()
        if zone_df.empty:
            continue
        centerline = extract_centerline(zone_df, band_width=band_width)
        centerline["zone_id"] = zone
        centerline["local_corrected_elevation_m"] = enforce_downhill_trend(
            centerline["elevation_m"].to_numpy(), tolerance=tolerance
        )
        zone_store[zone] = {"data": zone_df, "centerline": centerline}

    if not zone_store:
        return data, {}

    corrected_frames: list[pd.DataFrame] = []
    centerline_records: Dict[str, pd.DataFrame] = {}
    previous_end: Optional[float] = None

    for zone in zones:
        info = zone_store.get(zone)
        if info is None:
            continue

        centerline = info["centerline"].copy()
        centerline.sort_values("global_station_m", inplace=True)

        local_values = centerline["local_corrected_elevation_m"].to_numpy()
        base_values = centerline["base_elevation_m"].to_numpy()

        if previous_end is None:
            adjusted_values = np.minimum(local_values, base_values)
            adjusted_values = enforce_downhill_trend(adjusted_values, tolerance=tolerance)
        else:
            target_start = min(previous_end, local_values[0])
            shift = max(0.0, local_values[0] - target_start)
            candidate = local_values - shift
            adjusted_values = np.minimum(candidate, base_values)
            adjusted_values = enforce_downhill_trend(adjusted_values, tolerance=tolerance)

        centerline["global_corrected_elevation_m"] = adjusted_values
        centerline["global_adjustment_m"] = (
            centerline["base_elevation_m"] - centerline["global_corrected_elevation_m"]
        ).clip(lower=0.0)

        zone_df = info["data"].copy()
        zone_df["raw_elevation_m"] = zone_df["elevation_m"]
        adjustment = centerline.set_index("global_station_m")["global_adjustment_m"]
        zone_df = zone_df.join(adjustment.rename("station_adjustment_m"), on="global_station_m")
        zone_df["station_adjustment_m"] = zone_df["station_adjustment_m"].fillna(0.0)

        weights = 1.0 - np.clip(np.abs(zone_df["distance_from_center_m"]) / band_width, 0.0, 1.0)
        zone_df["applied_adjustment_m"] = zone_df["station_adjustment_m"] * weights
        zone_df["elevation_m"] = zone_df["elevation_m"] - zone_df["applied_adjustment_m"]
        zone_df.drop(columns=["station_adjustment_m"], inplace=True)

        previous_end = adjusted_values[-1]
        centerline_records[zone] = centerline
        corrected_frames.append(zone_df)

    corrected_data = pd.concat(corrected_frames, axis=0)
    return corrected_data, centerline_records


def run_channel_profile_model(
    preprocess_result: PreprocessResult,
    *,
    band_width: Optional[float] = None,
    tolerance: Optional[float] = None,
) -> ChannelProfileResult:
    """Run the centreline monotonic correction stage."""
    cfg = preprocess_result.config
    bw = band_width if band_width is not None else cfg.band_width
    tol = tolerance if tolerance is not None else cfg.monotonic_tolerance

    corrected, centerlines = apply_monotonic_corrections(
        preprocess_result.cross_sections,
        preprocess_result.zones,
        band_width=bw,
        tolerance=tol,
    )

    updated_cfg = replace(cfg, band_width=bw, monotonic_tolerance=tol)
    return ChannelProfileResult(
        cross_sections=corrected,
        centerlines=centerlines,
        config=updated_cfg,
    )


def generate_channel_profiles(
    cross_sections: pd.DataFrame | Path,
    segments: pd.DataFrame | Path,
    zones: Iterable[str],
    config: Optional[ChannelProfileConfig] = None,
) -> ChannelProfileResult:
    """Full pipeline helper that runs preprocessing + modelling stages."""
    preprocess_result = preprocess_cross_sections(
        cross_sections,
        segments,
        zones,
        config=config,
    )
    return run_channel_profile_model(preprocess_result)
