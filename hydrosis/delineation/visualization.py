"""Delineation utilities - Visualization and Output

Plotting, statistics, and output utilities for delineation results.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


try:
    import rasterio
    from affine import Affine
except ImportError:
    rasterio = None
    Affine = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from shapely.geometry import Polygon
from .pour_points import PourPoint


def compute_statistics(
    pour_points: Sequence[PourPoint],
    masks: Dict[str, np.ndarray],
    dem: np.ndarray,
    accumulation: np.ndarray,
    transform: rasterio.Affine,
) -> List[Dict[str, float]]:
    stats: List[Dict[str, float]] = []
    cell_area = cell_area_sqkm(transform)
    for pp in pour_points:
        mask = masks[pp.id]
        if not mask.any():
            continue
        area_km2 = mask.sum() * cell_area
        elevations = dem[mask]
        accum_values = accumulation[mask]
        rows, cols = np.nonzero(mask)
        xs = transform.c + cols * transform.a + rows * transform.b
        ys = transform.f + cols * transform.d + rows * transform.e
        stats.append(
            {
                "id": pp.id,
                "area_km2": float(area_km2),
                "accumulation": float(pp.accumulation),
                "mean_elevation": float(np.nanmean(elevations)),
                "max_accumulation": float(np.nanmax(accum_values)),
                "min_elevation": float(np.nanmin(elevations)),
                "max_elevation": float(np.nanmax(elevations)),
                "centroid_x": float(np.nanmean(xs)),
                "centroid_y": float(np.nanmean(ys)),
            }
        )
    return stats


def write_summary(statistics: Sequence[Dict[str, float]], output_dir: Path) -> None:
    lines = [
        "# Watershed summary",
        "| ID | Area (km^2) | Mean Elev. | Max Elev. | Min Elev. | Max Accumulation |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in statistics:
        lines.append(
            f"| {item['id']} | {item['area_km2']:.3f} | {item['mean_elevation']:.1f} | "
            f"{item['max_elevation']:.1f} | {item['min_elevation']:.1f} | {item['max_accumulation']:.0f} |"
        )
    (output_dir / "delineation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_attributes_csv(statistics: Sequence[Dict[str, float]], output_dir: Path) -> None:
    fieldnames = [
        "id",
        "area_km2",
        "accumulation",
        "mean_elevation",
        "min_elevation",
        "max_elevation",
        "max_accumulation",
        "centroid_x",
        "centroid_y",
    ]
    csv_path = output_dir / "delineation_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in statistics:
            writer.writerow(item)


def plot_raster(data: np.ndarray, title: str, output: Path, cmap: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data, origin="upper", cmap=cmap)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.85)
    ax.set_xlabel("column index")
    ax.set_ylabel("row index")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_flow_direction(data: np.ndarray, output: Path) -> None:
    direction = np.where(np.isin(data, list(D8_OFFSETS.keys())), data, np.nan)
    cmap = ListedColormap(FLOWDIR_COLORS, name="d8_flow")
    bounds = np.arange(0.5, 8.6, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(direction, origin="upper", cmap=cmap, norm=norm)
    ax.set_title("Flow direction (D8 codes)")
    ax.set_xlabel("column index")
    ax.set_ylabel("row index")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, ticks=np.arange(1, 9))
    cbar.ax.set_yticklabels(["East", "North-East", "North", "North-West", "West", "South-West", "South", "South-East"])
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_masks(masks: Dict[str, np.ndarray], pour_points: Sequence[PourPoint], output: Path) -> None:
    sample = next(iter(masks.values()))
    canvas = np.zeros(sample.shape, dtype=float)
    for idx, (pp_id, mask) in enumerate(masks.items(), start=1):
        canvas[mask] = idx

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(canvas, origin="upper", cmap="tab20")
    ax.set_title("Watershed masks")
    fig.colorbar(im, ax=ax, shrink=0.85, label="Subbasin index")
    for pp in pour_points:
        ax.scatter(pp.col, pp.row, c="black", marker="x", s=40)
        ax.text(pp.col + 4, pp.row + 4, pp.id, color="white")
    ax.set_xlabel("column index")
    ax.set_ylabel("row index")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def cell_area_sqkm(transform: rasterio.Affine) -> float:
    width = abs(transform.a)
    height = abs(transform.e)
    sqft = width * height
    return sqft * (0.304800609601219 ** 2) / 1_000_000.0


def plot_overview_map(
    dem: np.ndarray,
    transform: rasterio.Affine,
    polygons: Dict[str, List[Sequence[Tuple[float, float]]]],
    pour_points: Sequence[PourPoint],
    output: Path,
    dem_crs,
) -> None:
    dem_display = np.where(np.isfinite(dem), dem, np.nan)
    rows, cols = dem_display.shape
    left, bottom, right, top = array_bounds(rows, cols, transform)

    cell_height = abs(transform.e)
    cell_width = abs(transform.a)
    gy, gx = np.gradient(dem_display, cell_height, cell_width, edge_order=2)
    slope = np.pi / 2.0 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    azimuth = np.deg2rad(315.0)
    altitude = np.deg2rad(45.0)
    hillshade = np.sin(altitude) * np.sin(slope) + np.cos(altitude) * np.cos(slope) * np.cos(azimuth - aspect)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(hillshade, origin="upper", cmap="gray", extent=(left, right, bottom, top))
    im = ax.imshow(dem_display, origin="upper", cmap="terrain", extent=(left, right, bottom, top), alpha=0.6)
    ax.set_title("DEM with delineated watersheds")
    fig.colorbar(im, ax=ax, shrink=0.75, label="Elevation")

    for coords_list in polygons.values():
        for coords in coords_list:
            polygon = MplPolygon(coords, fill=False, edgecolor="white", linewidth=1.3)
            ax.add_patch(polygon)

    transformer = Transformer.from_crs(dem_crs, "EPSG:4326", always_xy=True)
    for pp in pour_points:
        ax.scatter(pp.x, pp.y, c="black", marker="x", s=60)
        lon, lat = transformer.transform(pp.x, pp.y)
        ax.text(pp.x + 150, pp.y + 150, f"{pp.id}\n({lat:.3f}, {lon:.3f})", color="black", fontsize=9, weight="bold")

    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
