"""Delineation utilities - Flow Network Analysis

Flow network construction and watershed delineation.
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
except ImportError:
    rasterio = None

from .pour_points import PourPoint

# D8 direction offsets
D8_OFFSETS = {
    1: (0, 1),     # East
    2: (1, 1),     # Southeast
    4: (1, 0),     # South
    8: (1, -1),    # Southwest
    16: (0, -1),   # West
    32: (-1, -1),  # Northwest
    64: (-1, 0),   # North
    128: (-1, 1),  # Northeast
}

def build_flow_network(flow_dir_path: Path, offsets: Dict[int, Tuple[int, int]] = D8_OFFSETS) -> Tuple[np.ndarray, List[List[Tuple[int, int]]], Tuple[int, int]]:
    with rasterio.open(flow_dir_path) as dir_ds:
        flowdir = dir_ds.read(1)
    rows, cols = flowdir.shape

    upstream: List[List[Tuple[int, int]]] = [[] for _ in range(rows * cols)]
    for row in range(rows):
        for col in range(cols):
            code = int(flowdir[row, col])
            if code not in offsets:
                continue
            dr, dc = offsets[code]
            nr, nc = row + dr, col + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                upstream[nr * cols + nc].append((row, col))
    return flowdir, upstream, (rows, cols)


def delineate_watershed(
    pour_point: PourPoint,
    upstream: Sequence[List[Tuple[int, int]]],
    shape: Tuple[int, int],
) -> np.ndarray:
    rows, cols = shape
    mask = np.zeros((rows, cols), dtype=bool)
    stack = [(pour_point.row, pour_point.col)]
    while stack:
        r, c = stack.pop()
        if not (0 <= r < rows and 0 <= c < cols):
            continue
        if mask[r, c]:
            continue
        mask[r, c] = True
        idx = r * cols + c
        for nr, nc in upstream[idx]:
            if not mask[nr, nc]:
                stack.append((nr, nc))
    return mask


def masks_to_polygons(mask_map: Dict[str, np.ndarray], transform: rasterio.Affine) -> Dict[str, List[Sequence[Tuple[float, float]]]]:
    polygons: Dict[str, List[Sequence[Tuple[float, float]]]] = {}
    for basin_id, mask in mask_map.items():
        coords: List[Sequence[Tuple[float, float]]] = []
        for geom, value in rasterio.features.shapes(mask.astype(np.uint8), transform=transform):
            if value != 1:
                continue
            coords.append(geom["coordinates"][0])
        polygons[basin_id] = coords
    return polygons

FLOWDIR_COLORS = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#a65628",
    "#f781bf",
    "#999999",
]

