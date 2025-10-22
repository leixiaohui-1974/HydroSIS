"""Generate rainfall distribution visualisations for HydroSIS results.

This script produces subbasin-centric visualisations:

1. Rain gauge distribution map with subbasin polygons providing the background
   while Thiessen regions and gauge locations are overlaid for reference.
2. Animated GIF illustrating the evolution of subbasin areal precipitation.
3. Static cumulative precipitation map summarising total depth per subbasin.

Usage example::

    python examples/visualize_areal_rainfall.py \
        --thiessen results/upper_truckee_channel_demo/intermediate/rain_gauge_thiessen_polygons.geojson \
        --gauges results/upper_truckee_channel_demo/intermediate/rain_gauge_locations.geojson \
        --subbasins results/upper_truckee_channel_demo/intermediate/subbasins.geojson \
        --precip-csv results/upper_truckee_channel_demo/intermediate/subbasin_areal_precipitation.csv \
        --output-dir results/upper_truckee_channel_demo/intermediate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon as MplPolygon

# Optional GIF backends
Image = None  # type: ignore[assignment]
_HAS_PIL = False
try:
    import imageio.v2 as imageio  # type: ignore[import]

    _HAS_IMAGEIO = True
except Exception:  # pragma: no cover - optional dependency
    imageio = None  # type: ignore[assignment]
    _HAS_IMAGEIO = False
    try:
        from PIL import Image  # type: ignore[import]

        _HAS_PIL = True
    except Exception:  # pragma: no cover - optional dependency
        Image = None  # type: ignore[assignment]
        _HAS_PIL = False
from shapely.geometry import MultiPolygon, Point, Polygon, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


def _load_polygon_geometries(
    path: Path,
    id_field: str,
    fallback_prefix: str,
) -> Dict[str, BaseGeometry]:
    data = json.loads(path.read_text(encoding="utf-8"))
    geometries: Dict[str, BaseGeometry] = {}
    for index, feature in enumerate(data.get("features", []), start=1):
        properties = feature.get("properties", {}) or {}
        feature_id = str(properties.get(id_field) or feature.get("id") or f"{fallback_prefix}_{index}")
        geometries[feature_id] = shape(feature.get("geometry"))
    if not geometries:
        raise ValueError(f"No polygon features found in {path}")
    return geometries


def _load_point_geometries(path: Path, id_field: str = "id") -> Dict[str, Point]:
    data = json.loads(path.read_text(encoding="utf-8"))
    points: Dict[str, Point] = {}
    for index, feature in enumerate(data.get("features", []), start=1):
        properties = feature.get("properties", {}) or {}
        feature_id = str(properties.get(id_field) or feature.get("id") or f"gauge_{index}")
        geometry = shape(feature.get("geometry"))
        if not isinstance(geometry, Point):
            raise TypeError(f"Gauge feature '{feature_id}' is not a Point geometry.")
        points[feature_id] = geometry
    if not points:
        raise ValueError(f"No point features found in {path}")
    return points


def _iter_polygons(geometry: BaseGeometry) -> Iterable[Polygon]:
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        for polygon in geometry.geoms:
            yield polygon
    else:
        raise TypeError(f"Unsupported geometry type: {geometry.geom_type}")


def _polygon_to_patch(polygon: Polygon) -> MplPolygon:
    return MplPolygon(np.asarray(polygon.exterior.coords), closed=True)


def _set_extent(ax, geometries: Sequence[BaseGeometry]) -> None:
    union = unary_union(list(geometries))
    minx, miny, maxx, maxy = union.bounds
    border_x = (maxx - minx) * 0.05 or 1.0
    border_y = (maxy - miny) * 0.05 or 1.0
    ax.set_xlim(minx - border_x, maxx + border_x)
    ax.set_ylim(miny - border_y, maxy + border_y)
    ax.set_aspect("equal", adjustable="box")


def plot_rain_gauge_distribution(
    subbasins: Dict[str, BaseGeometry],
    thiessen_polygons: Dict[str, BaseGeometry],
    gauge_points: Dict[str, Point],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    # Subbasin fill and boundaries
    sub_patches: List[MplPolygon] = []
    sub_indices: List[int] = []
    for geometry in subbasins.values():
        for polygon in _iter_polygons(geometry):
            sub_patches.append(_polygon_to_patch(polygon))
            sub_indices.append(len(sub_indices))
    if sub_patches:
        sub_collection = PatchCollection(
            sub_patches,
            cmap=plt.get_cmap("Pastel2"),
            alpha=0.6,
            edgecolor="white",
            linewidth=0.6,
        )
        sub_collection.set_array(np.asarray(sub_indices, dtype=float))
        ax.add_collection(sub_collection)
        for polygon in subbasins.values():
            for geom in _iter_polygons(polygon):
                coords = np.asarray(geom.exterior.coords)
                ax.plot(coords[:, 0], coords[:, 1], color="black", linewidth=1.0, alpha=0.95)

    # Thiessen polygons overlay
    for station_id, geometry in thiessen_polygons.items():
        for polygon in _iter_polygons(geometry):
            patch = _polygon_to_patch(polygon)
            ax.add_patch(
                MplPolygon(
                    patch.get_xy(),
                    fill=False,
                    edgecolor="black",
                    linewidth=1.2,
                    alpha=0.9,
                    label="_nolegend_",
                )
            )

    # Gauge locations
    xs = [point.x for point in gauge_points.values()]
    ys = [point.y for point in gauge_points.values()]
    ax.scatter(xs, ys, s=80, c="red", edgecolors="white", linewidths=1.0, zorder=5, label="Rain Gauge")
    for label, point in gauge_points.items():
        ax.text(point.x, point.y, label, fontsize=8, ha="left", va="bottom", color="black", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    _set_extent(ax, list(subbasins.values()))
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    ax.set_title("Rain Gauge Distribution with Thiessen Polygons")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def generate_subbasin_rainfall_gif(
    subbasins: Dict[str, BaseGeometry],
    precipitation: pd.DataFrame,
    output_path: Path,
    duration: float = 0.2,
    dynamic_scale: bool = False,
) -> None:
    patches: List[MplPolygon] = []
    patch_sub_ids: List[str] = []
    boundary_coords: List[np.ndarray] = []
    for sub_id, geometry in subbasins.items():
        for polygon in _iter_polygons(geometry):
            coords = np.asarray(polygon.exterior.coords)
            boundary_coords.append(coords)
            patches.append(_polygon_to_patch(polygon))
            patch_sub_ids.append(sub_id)

    frames: List[np.ndarray] = []
    global_vmax = float(np.nanmax(precipitation.to_numpy()))
    if not np.isfinite(global_vmax) or global_vmax <= 0:
        global_vmax = 1.0
    global_norm = Normalize(vmin=0.0, vmax=global_vmax)
    cmap = plt.get_cmap("Blues")

    geometries = list(subbasins.values())
    bounds_union = unary_union(geometries).bounds
    minx, miny, maxx, maxy = bounds_union
    pad_x = (maxx - minx) * 0.05 or 1.0
    pad_y = (maxy - miny) * 0.05 or 1.0

    for timestamp, row in precipitation.iterrows():
        fig, ax = plt.subplots(figsize=(8, 6))
        if dynamic_scale:
            frame_max = float(np.nanmax(row.to_numpy(dtype=float)))
            if not np.isfinite(frame_max) or frame_max <= 0:
                frame_max = global_vmax
            norm = Normalize(vmin=0.0, vmax=frame_max)
        else:
            norm = global_norm

        patch_collection = PatchCollection(
            patches,
            cmap=cmap,
            norm=norm,
            edgecolor="gray",
            linewidth=0.4,
        )
        values = np.asarray([row[sub_id] for sub_id in patch_sub_ids], dtype=float)
        patch_collection.set_array(values)
        ax.add_collection(patch_collection)
        for coords in boundary_coords:
            ax.plot(coords[:, 0], coords[:, 1], color="black", linewidth=0.8, alpha=0.9)
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Areal Precipitation\n{timestamp}")
        ax.set_axis_off()

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Precipitation (mm/hr)")

        fig.tight_layout()
        canvas = fig.canvas
        canvas.draw()
        buffer = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        width, height = canvas.get_width_height()
        image = buffer.reshape((height, width, 4))[:, :, :3].copy()
        frames.append(image)
        plt.close(fig)

    if not frames:
        raise ValueError("No frames generated for GIF output.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if _HAS_IMAGEIO:
        imageio.mimsave(output_path, frames, duration=duration)  # type: ignore[arg-type]
    elif _HAS_PIL:
        pil_frames = [Image.fromarray(frame) for frame in frames]  # type: ignore[operator]
        pil_frames[0].save(
            output_path,
            format="GIF",
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(duration * 1000),
            loop=0,
        )
    else:
        raise RuntimeError("Neither imageio nor Pillow is available to write GIF animations.")


def plot_cumulative_areal_precipitation(
    subbasins: Dict[str, BaseGeometry],
    precipitation: pd.DataFrame,
    output_path: Path,
    cmap_name: str = "viridis",
) -> None:
    patches: List[MplPolygon] = []
    patch_sub_ids: List[str] = []
    boundary_coords: List[np.ndarray] = []
    for sub_id, geometry in subbasins.items():
        for polygon in _iter_polygons(geometry):
            coords = np.asarray(polygon.exterior.coords)
            boundary_coords.append(coords)
            patches.append(_polygon_to_patch(polygon))
            patch_sub_ids.append(sub_id)

    # Determine time-step duration in hours for each record
    timestamps = precipitation.index.to_series()
    delta_hours = timestamps.diff().dt.total_seconds().fillna(0.0) / 3600.0
    delta_matrix = pd.DataFrame(
        np.tile(delta_hours.to_numpy().reshape(-1, 1), (1, precipitation.shape[1])),
        index=precipitation.index,
        columns=precipitation.columns,
    )
    cumulative_depth = (precipitation * delta_matrix).sum(axis=0)

    values = np.asarray([float(cumulative_depth[sub_id]) for sub_id in patch_sub_ids])
    vmax = float(np.nanmax(values))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    fig, ax = plt.subplots(figsize=(8, 6))
    patch_collection = PatchCollection(
        patches,
        cmap=cmap,
        norm=norm,
        edgecolor="gray",
        linewidth=0.6,
    )
    patch_collection.set_array(values)
    ax.add_collection(patch_collection)
    for coords in boundary_coords:
        ax.plot(coords[:, 0], coords[:, 1], color="black", linewidth=0.9, alpha=0.9)

    _set_extent(ax, list(subbasins.values()))
    ax.set_title("Cumulative Areal Precipitation")
    ax.set_axis_off()

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cumulative Precipitation (mm)")


    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_station_timeseries(
    station_csv: Path,
    output_path: Path,
) -> None:
    df = pd.read_csv(station_csv, index_col=0, parse_dates=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.empty:
        raise ValueError(f"Station CSV {station_csv} contains no data.")

    fig, ax = plt.subplots(figsize=(12, 4))
    for column in df.columns:
        ax.plot(df.index, df[column], label=column, linewidth=1.3)
    ax.set_xlabel("Time")
    ax.set_ylabel("Precipitation (mm/hr)")
    ax.set_title("Rain Gauge Precipitation Time Series")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def _load_precipitation(csv_path: Path, subbasin_ids: Sequence[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    missing = [sub_id for sub_id in subbasin_ids if sub_id not in df.columns]
    if missing:
        raise KeyError(f"Precipitation series missing subbasin columns: {missing}")
    return df[subbasin_ids]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise rain gauge distribution and subbasin rainfall dynamics.")
    parser.add_argument("--thiessen", type=Path, required=True, help="GeoJSON file containing rain gauge Thiessen polygons.")
    parser.add_argument("--gauges", type=Path, required=True, help="GeoJSON file containing rain gauge point locations.")
    parser.add_argument("--subbasins", type=Path, required=True, help="GeoJSON file containing subbasin polygons.")
    parser.add_argument("--precip-csv", type=Path, required=True, help="CSV file containing subbasin areal precipitation time series.")
    parser.add_argument("--station-csv", type=Path, default=None, help="Optional CSV containing rainfall station time series (defaults to subbasin CSV directory / rain_gauge_forcing.csv).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for generated visualisations.")
    parser.add_argument("--map-name", type=str, default="rain_gauge_distribution.png", help="Filename for the rain gauge distribution map.")
    parser.add_argument("--gif-name", type=str, default="subbasin_areal_precipitation.gif", help="Filename for the animated precipitation GIF.")
    parser.add_argument("--cumulative-name", type=str, default="subbasin_cumulative_precipitation.png", help="Filename for the cumulative precipitation map.")
    parser.add_argument("--station-timeseries-name", type=str, default="rain_gauge_timeseries.png", help="Filename for the rain gauge time-series comparison plot.")
    parser.add_argument("--frame-duration", type=float, default=0.2, help="Frame duration in seconds for the GIF animation.")
    parser.add_argument("--dynamic-scale", action="store_true", help="Allow GIF colour scale to adapt per timestep (default keeps a fixed legend).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    thiessen_polygons = _load_polygon_geometries(args.thiessen, id_field="station_id", fallback_prefix="station")
    gauge_points = _load_point_geometries(args.gauges, id_field="id")
    subbasins = _load_polygon_geometries(args.subbasins, id_field="subzone_id", fallback_prefix="subbasin")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    map_output = output_dir / args.map_name
    gif_output = output_dir / args.gif_name
    cumulative_output = output_dir / args.cumulative_name
    station_ts_output = output_dir / args.station_timeseries_name

    plot_rain_gauge_distribution(subbasins, thiessen_polygons, gauge_points, map_output)

    precipitation = _load_precipitation(args.precip_csv, list(subbasins.keys()))
    plot_cumulative_areal_precipitation(subbasins, precipitation, cumulative_output)
    generate_subbasin_rainfall_gif(
        subbasins,
        precipitation,
        gif_output,
        duration=args.frame_duration,
        dynamic_scale=args.dynamic_scale,
    )

    station_csv = args.station_csv
    if station_csv is None:
        station_csv = args.precip_csv.parent / "rain_gauge_forcing.csv"
    if station_csv.exists():
        plot_station_timeseries(station_csv, station_ts_output)
        station_msg = f"Rain gauge time series plot saved to {station_ts_output}"
    else:
        station_msg = f"Rain gauge time series skipped (file not found: {station_csv})"

    print(f"Rain gauge map saved to {map_output}")
    print(f"Cumulative precipitation map saved to {cumulative_output}")
    print(f"Subbasin rainfall GIF saved to {gif_output}")
    print(station_msg)


if __name__ == "__main__":
    main()
