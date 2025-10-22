"""Synthetic rain gauge generation utilities for temporary testing workflows."""
from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from .thiessen import (
    compute_subbasin_station_weights,
    generate_station_ids,
    interpolate_station_series,
    sample_station_positions,
    thiessen_polygons_for_stations,
)


@dataclass
class RainGaugeInputs:
    """Container holding synthetic rain gauge artefacts."""

    station_series: pd.DataFrame
    subbasin_series: pd.DataFrame
    station_positions: Dict[str, Point]
    thiessen_polygons: Dict[str, BaseGeometry]
    station_weights: Dict[str, Dict[str, float]]

    def write(
        self,
        directory: PathLike[str] | str | Path,
        gauges_filename: str = "rain_gauge_forcing.csv",
        subbasin_filename: str = "subbasin_areal_precipitation.csv",
        stations_geojson: str = "rain_gauge_locations.geojson",
        thiessen_geojson: str = "rain_gauge_thiessen_polygons.geojson",
        weights_json: str = "rain_gauge_weights.json",
    ) -> Dict[str, "Path"]:
        import json
        from shapely.geometry import mapping
        directory_path = Path(directory)
        directory_path.mkdir(parents=True, exist_ok=True)

        gauges_path = directory_path / gauges_filename
        subbasin_path = directory_path / subbasin_filename
        stations_path = directory_path / stations_geojson
        thiessen_path = directory_path / thiessen_geojson
        weights_path = directory_path / weights_json

        self.station_series.to_csv(gauges_path)
        self.subbasin_series.to_csv(subbasin_path)

        stations_feature = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": mapping(point),
                    "properties": {"id": station_id},
                }
                for station_id, point in self.station_positions.items()
            ],
        }
        stations_path.write_text(json.dumps(stations_feature, indent=2), encoding="utf-8")

        thiessen_feature = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": mapping(polygon),
                    "properties": {"station_id": station_id},
                }
                for station_id, polygon in self.thiessen_polygons.items()
            ],
        }
        thiessen_path.write_text(json.dumps(thiessen_feature, indent=2), encoding="utf-8")

        weights_payload = {
            sub_id: {station_id: float(weight) for station_id, weight in weights.items()}
            for sub_id, weights in self.station_weights.items()
        }
        weights_path.write_text(json.dumps(weights_payload, indent=2), encoding="utf-8")

        return {
            "gauges": gauges_path,
            "subbasin": subbasin_path,
            "stations": stations_path,
            "thiessen": thiessen_path,
            "weights": weights_path,
        }


def _smooth_series(series: np.ndarray, alpha: float) -> np.ndarray:
    if not 0 < alpha <= 1:
        raise ValueError("Smoothing parameter alpha must lie in (0, 1].")
    smoothed = np.empty_like(series)
    smoothed[0] = series[0]
    for index in range(1, series.shape[0]):
        smoothed[index] = alpha * smoothed[index - 1] + (1.0 - alpha) * series[index]
    return smoothed


def _station_gradient_factor(
    station_point: Point,
    bounds: Tuple[float, float, float, float],
    max_gradient: float,
) -> float:
    minx, miny, maxx, maxy = bounds
    if maxx == minx or maxy == miny:
        return 1.0

    centre_x = (minx + maxx) * 0.5
    centre_y = (miny + maxy) * 0.5
    span_x = max(maxx - minx, 1e-6)
    span_y = max(maxy - miny, 1e-6)

    dx = (station_point.x - centre_x) / span_x
    dy = (station_point.y - centre_y) / span_y
    radial = np.hypot(dx, dy)
    directional = 0.6 * dx + 0.4 * dy
    combined = directional + 0.35 * radial * np.sign(directional or 1.0)
    factor = 1.0 + max_gradient * np.clip(combined, -1.0, 1.0)
    return max(0.25, factor)


def _create_local_bursts(
    base_peak: float,
    steps: int,
    rng: np.random.Generator,
    min_events: int,
    max_events: int,
) -> np.ndarray:
    bursts = np.zeros(steps, dtype=float)
    event_count = int(rng.integers(min_events, max_events + 1))
    for _ in range(event_count):
        centre = int(rng.integers(0, steps))
        width = float(rng.uniform(0.8, 6.0))
        magnitude = float(rng.uniform(0.2, 0.75)) * base_peak
        indices = np.arange(steps, dtype=float)
        bursts += magnitude * np.exp(-0.5 * ((indices - centre) / max(width, 1.0)) ** 2)
    return bursts


def _apply_time_shift(values: np.ndarray, lag: int) -> np.ndarray:
    if lag == 0:
        return values
    shifted = np.zeros_like(values)
    if lag > 0:
        shifted[lag:] = values[:-lag]
    else:
        shifted[:lag] = values[-lag:]
    return shifted


def _generate_station_series(
    base_series: pd.Series,
    station_positions: Mapping[str, Point],
    rng: np.random.Generator,
    heterogeneity: float,
    min_events: int,
    max_events: int,
) -> pd.DataFrame:
    values = base_series.to_numpy(dtype=float)
    steps = values.shape[0]
    if steps == 0:
        raise ValueError("Base precipitation series must not be empty.")
    peak_intensity = float(np.max(values))
    if peak_intensity <= 0:
        peak_intensity = 1.0

    bounds = (
        min(point.x for point in station_positions.values()),
        min(point.y for point in station_positions.values()),
        max(point.x for point in station_positions.values()),
        max(point.y for point in station_positions.values()),
    )

    time = np.linspace(0.0, 1.0, steps)
    seasonal = 0.15 * np.sin(2.0 * np.pi * time)

    station_ids = list(station_positions.keys())
    output = np.zeros((steps, len(station_ids)), dtype=float)

    for index, station_id in enumerate(station_ids):
        gradient_factor = _station_gradient_factor(station_positions[station_id], bounds, heterogeneity)
        amplitude = float(rng.uniform(0.5, 1.7))
        lag = int(rng.integers(-4, 5))

        multiplicative_noise = rng.normal(0.0, 0.25, size=steps)
        multiplicative_noise = _smooth_series(multiplicative_noise, alpha=0.45)

        burst_component = _create_local_bursts(
            base_peak=peak_intensity,
            steps=steps,
            rng=rng,
            min_events=min_events,
            max_events=max_events,
        )

        base_shifted = _apply_time_shift(values, lag)
        perturbed = base_shifted * (1.0 + multiplicative_noise + seasonal) * amplitude * gradient_factor
        perturbed += burst_component

        additive_noise = rng.normal(0.0, 0.07 * peak_intensity, size=steps)
        additive_noise = _smooth_series(additive_noise, alpha=0.35)

        station_bias = rng.normal(0.0, heterogeneity * 0.35) * peak_intensity

        series = perturbed + additive_noise + station_bias

        dryness_bias = np.clip(0.12 + (1.1 - gradient_factor) * 0.35, 0.05, 0.65)
        dryness_events = rng.uniform(size=steps) < dryness_bias
        dryness_pattern = _smooth_series(dryness_events.astype(float), alpha=0.25)
        dry_mask = np.clip(1.0 - dryness_pattern, 0.0, 1.0)
        series *= dry_mask

        output[:, index] = np.clip(series, a_min=0.0, a_max=None)

    df = pd.DataFrame(output, index=base_series.index, columns=station_ids)
    df.index.name = base_series.index.name or "Timestamp"
    return df


def generate_rain_gauge_inputs(
    base_precipitation: pd.Series,
    subbasin_geometries: Mapping[str, BaseGeometry],
    boundary: Optional[BaseGeometry] = None,
    station_count: int = 10,
    rng_seed: Optional[int] = None,
    heterogeneity_strength: float = 0.4,
    min_burst_events: int = 1,
    max_burst_events: int = 3,
) -> RainGaugeInputs:
    """Generate synthetic rain gauge forcing with enhanced spatial variability."""

    if station_count < 2:
        raise ValueError("At least two stations are required to generate Thiessen polygons.")
    if not subbasin_geometries:
        raise ValueError("Subbasin geometries must be provided.")

    from shapely.ops import unary_union

    rng = np.random.default_rng(rng_seed)
    basins_union = unary_union(list(subbasin_geometries.values()))
    target_boundary = boundary or basins_union
    station_ids = generate_station_ids(station_count)
    station_positions = sample_station_positions(target_boundary, station_ids, rng=rng)
    thiessen_polygons = thiessen_polygons_for_stations(station_positions, target_boundary)

    station_series = _generate_station_series(
        base_precipitation,
        station_positions,
        rng=rng,
        heterogeneity=heterogeneity_strength,
        min_events=min_burst_events,
        max_events=max_burst_events,
    )

    station_weights = compute_subbasin_station_weights(subbasin_geometries, thiessen_polygons)
    subbasin_series = interpolate_station_series(station_series, station_weights)
    subbasin_series.index.name = station_series.index.name

    return RainGaugeInputs(
        station_series=station_series,
        subbasin_series=subbasin_series,
        station_positions=dict(station_positions),
        thiessen_polygons=dict(thiessen_polygons),
        station_weights=station_weights,
    )


__all__ = ["RainGaugeInputs", "generate_rain_gauge_inputs"]
