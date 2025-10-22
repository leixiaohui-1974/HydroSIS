# -*- coding: utf-8 -*-
"""Upper Truckee River watershed delineation example."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import rasterio

from hydrosis.delineation import utils as dutils
from hydrosis.delineation.channel_analysis import (
    compute_channel_mask,
    segments_to_feature_collection,
    trace_channel_segments,
)
from hydrosis.model import ChannelNetwork


def write_geojson(features: Dict, path: Path) -> None:
    path.write_text(json.dumps(features, indent=2), encoding="utf-8")


def build_pour_point_geojson(pour_points: Sequence[dutils.PourPoint]) -> Dict:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [pp.x, pp.y]},
                "properties": {
                    "id": pp.id,
                    "row": pp.row,
                    "col": pp.col,
                    "accumulation": pp.accumulation,
                },
            }
            for pp in pour_points
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Subbasin delineation workflow")
    parser.add_argument("--dem", type=Path, default=Path("data/Upper_Truckee_River/terrain/UpTruckeeRv_S10_NED_30m/00/elevation.tif"), help="DEM GeoTIFF path")
    parser.add_argument("--flow-acc", type=Path, default=Path("data/Upper_Truckee_River/terrain/UpTruckeeRv_S10_NED_30m/00/flowaccum.tif"), help="Flow accumulation raster path")
    parser.add_argument("--flow-dir", type=Path, default=Path("data/Upper_Truckee_River/terrain/UpTruckeeRv_S10_NED_30m/00/flowdir.tif"), help="Flow direction raster path")
    parser.add_argument("--output", type=Path, default=Path("results/delineation_upper_truckee"), help="Output directory")
    parser.add_argument("--pour-count", type=int, default=4, help="Number of pour points to select")
    parser.add_argument("--threshold", type=float, default=2000.0, help="Minimum accumulation threshold")
    parser.add_argument("--min-spacing", type=float, default=2000.0, help="Minimum spacing between pour points (map units)")
    parser.add_argument(
        "--channel-threshold",
        type=float,
        default=2000.0,
        help="Flow accumulation threshold used to derive the channel network (in accumulation units)",
    )
    args = parser.parse_args()

    dutils.ensure_inputs(args.dem, args.flow_acc, args.flow_dir, args.output)

    pour_points = dutils.derive_pour_points(
        args.flow_acc,
        args.pour_count,
        args.threshold,
        args.min_spacing,
    )
    write_geojson(build_pour_point_geojson(pour_points), args.output / "pour_points.geojson")

    flowdir, upstream, shape = dutils.build_flow_network(args.flow_dir)

    with rasterio.open(args.dem) as dem_ds:
        transform = dem_ds.transform
        dem_data = dem_ds.read(1)
        dem_data = np.where(np.isfinite(dem_data), dem_data, np.nan)
        dem_crs = dem_ds.crs

    masks = {pp.id: dutils.delineate_watershed(pp, upstream, shape) for pp in pour_points}
    polygons = dutils.masks_to_polygons(masks, transform)

    features = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "MultiPolygon", "coordinates": [coords]},
                "properties": {"id": basin_id},
            }
            for basin_id, coords in polygons.items()
            if coords
        ],
    }
    write_geojson(features, args.output / "subbasins.geojson")
    dutils.plot_overview_map(dem_data, transform, polygons, pour_points, args.output / "overview_map.png", dem_crs)

    with rasterio.open(args.flow_acc) as acc_ds:
        acc = acc_ds.read(1)
    acc = np.where(np.isfinite(acc), acc, 0.0)
    acc_plot = np.where(acc > 0, acc, np.nan)
    dutils.plot_raster(np.log1p(acc_plot), "Log1p flow accumulation", args.output / "flow_accumulation.png", "inferno")
    dutils.plot_flow_direction(flowdir, args.output / "flow_direction.png")
    dutils.plot_masks(masks, pour_points, args.output / "watershed_masks.png")

    channel_mask = compute_channel_mask(acc, args.channel_threshold)
    segments = trace_channel_segments(flowdir, upstream, channel_mask, transform, dem_data)
    network = ChannelNetwork()
    for segment in segments:
        network.add_segment(segment)
    channel_features = segments_to_feature_collection(segments, transform)
    write_geojson(channel_features, args.output / "channel_network.geojson")

    segment_records = [
        {
            "id": seg.id,
            "length_m": float(seg.length_m),
            "drop_m": float(seg.drop_m) if seg.drop_m is not None else None,
            "slope": float(seg.slope) if seg.slope is not None else None,
            "downstream_id": seg.downstream,
            "upstream_ids": seg.upstream_ids,
            "start_elevation": float(seg.start_elevation) if seg.start_elevation is not None else None,
            "end_elevation": float(seg.end_elevation) if seg.end_elevation is not None else None,
            "cells": seg.cells,
        }
        for seg in segments
    ]
    (args.output / "channel_segments.json").write_text(json.dumps(segment_records, indent=2), encoding="utf-8")

    channel_mask_plot = np.where(channel_mask, 1.0, np.nan)
    dutils.plot_raster(channel_mask_plot, "Channel mask (acc threshold)", args.output / "channel_mask.png", "Blues")

    channel_summary = network.summary()
    channel_summary["threshold"] = args.channel_threshold

    statistics = dutils.compute_statistics(pour_points, masks, dem_data, acc, transform)
    dutils.write_summary(statistics, args.output)
    dutils.write_attributes_csv(statistics, args.output)

    diagnostics = {
        "dem_path": str(args.dem),
        "flow_acc_path": str(args.flow_acc),
        "flow_dir_path": str(args.flow_dir),
        "pour_points": [pp.__dict__ for pp in pour_points],
        "total_cells": int(sum(mask.sum() for mask in masks.values())),
        "statistics": statistics,
        "channel_summary": channel_summary,
    }
    write_geojson(diagnostics, args.output / "delineation_diagnostics.json")

    print("Watershed delineation complete. Outputs ->", args.output.resolve())


if __name__ == "__main__":
    main()
