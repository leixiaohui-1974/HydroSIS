"""Precipitation processing utilities for HydroSIS."""

from .rain_gauge_generator import RainGaugeInputs, generate_rain_gauge_inputs
from .thiessen import (
    generate_station_ids,
    perturb_precipitation_series,
    sample_station_positions,
    thiessen_polygons_for_stations,
    compute_subbasin_station_weights,
    interpolate_station_series,
)

__all__ = [
    "generate_station_ids",
    "perturb_precipitation_series",
    "sample_station_positions",
    "thiessen_polygons_for_stations",
    "compute_subbasin_station_weights",
    "interpolate_station_series",
    "RainGaugeInputs",
    "generate_rain_gauge_inputs",
]
