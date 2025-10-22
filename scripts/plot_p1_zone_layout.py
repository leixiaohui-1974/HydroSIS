"""
Generate an updated P1 mainstem layout map that:

* Extends the main channel to the upstream zone outlet that feeds P1.
* Highlights cross-section nodes used for the hydrodynamic solver.
* Marks every subbasin (across all zones) whose flow eventually enters the mainstem,
  alongside the inferred inflow point on the channel.

Run from the repository root:
    python scripts/plot_p1_zone_layout.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import Point


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results" / "upper_truckee_project"


def load_datasets() -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame]:
    """Load subbasins, channel geometries, and cross-section samples."""
    subbasins = gpd.read_file(RESULTS_ROOT / "03_partitioning" / "parameter_subbasins.geojson")
    channels = gpd.read_file(RESULTS_ROOT / "03_partitioning" / "parameter_channels.geojson")
    cross_sections = pd.read_csv(
        RESULTS_ROOT / "04_channel_profile" / "channel_cross_sections_corrected.csv"
    )

    # Stored geometries are already projected in a planar CRS; clear the incorrect EPSG tag.
    subbasins = subbasins.set_crs(None, allow_override=True)
    channels = channels.set_crs(None, allow_override=True)

    return subbasins, channels, cross_sections


def _is_missing(value: object) -> bool:
    """Return True when a downstream identifier should be treated as missing."""
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def segment_suffix(segment_id: str) -> str:
    """Extract a short suffix for display from a segment identifier."""
    if not segment_id:
        return segment_id
    zone_part = segment_id.split("_")[0]
    suffix_part = segment_id.split("_")[-1]
    zone_digits = "".join(ch for ch in zone_part if ch.isdigit())
    suffix_digits = "".join(ch for ch in suffix_part if ch.isdigit())
    if not suffix_digits:
        suffix_digits = suffix_part
    if zone_digits:
        return f"{zone_digits}{suffix_digits}"
    return suffix_digits


def format_suffix_list(segments: Sequence[str]) -> List[str]:
    """Format a list of segment ids as compact labels preserving order."""
    labels: List[str] = []
    seen: set[str] = set()
    counts: Dict[str, int] = {}
    for seg in segments:
        label = segment_suffix(seg)
        counts[label] = counts.get(label, 0) + 1
    for seg in segments:
        label = segment_suffix(seg)
        if label in seen:
            continue
        seen.add(label)
        count = counts[label]
        labels.append(f"{label}×{count}" if count > 1 else label)
    return labels


def parse_upstream(value: object) -> List[str]:
    """Parse the stored upstream identifiers into a clean list of segment ids."""
    if _is_missing(value):
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    cleaned = str(value).strip()
    if cleaned.startswith("[") and cleaned.endswith("]"):
        inner = cleaned[1:-1].replace('"', "").replace("'", "")
        return [part.strip() for part in inner.split(",") if part.strip()]
    return [part.strip() for part in cleaned.split(";") if part.strip()]


def build_downstream_map(channels_meta: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Create a downstream mapping for quick traversal."""
    mapping: Dict[str, Optional[str]] = {}
    for seg, ds in channels_meta["downstream_id"].items():
        if _is_missing(ds):
            mapping[seg] = None
        else:
            mapping[seg] = str(ds).strip()
    return mapping


def traverse_path(
    start: str,
    outlet: str,
    downstream_map: Dict[str, Optional[str]],
) -> Optional[List[str]]:
    """Follow the downstream chain from start to outlet; return path if reachable."""
    path: List[str] = [start]
    current = start
    visited = {start}
    while current != outlet:
        next_seg = downstream_map.get(current)
        if not next_seg or next_seg in visited:
            return None
        path.append(next_seg)
        visited.add(next_seg)
        current = next_seg
    return path


def compute_main_path(
    zone_id: str,
    channels_meta: pd.DataFrame,
) -> List[str]:
    """
    Determine the mainstem from the upstream zone outlet to the zone outlet.

    The algorithm searches for upstream segments that belong to other zones but whose
    downstream chain terminates at the zone outlet. The longest such path is selected
    to represent the mainstem feeding the target zone.
    """
    zone_rows = channels_meta[channels_meta["zone_id"] == zone_id]
    if zone_rows.empty:
        raise ValueError(f"No channel segments found for zone {zone_id}.")

    outlet_mask = zone_rows["downstream_id"].apply(_is_missing)
    outlet_rows = zone_rows[outlet_mask]
    if outlet_rows.empty:
        raise ValueError(f"Unable to locate outlet segment for zone {zone_id}.")
    outlet_segment = outlet_rows.index[0]

    downstream_map = build_downstream_map(channels_meta)
    length_map = channels_meta["length_m"].to_dict()

    candidate_paths: List[Tuple[float, List[str]]] = []
    for seg_id, row in zone_rows.iterrows():
        for upstream_id in parse_upstream(row["upstream_ids"]):
            if upstream_id not in channels_meta.index:
                continue
            upstream_zone = channels_meta.loc[upstream_id, "zone_id"]
            if upstream_zone == zone_id:
                continue
            path = traverse_path(upstream_id, outlet_segment, downstream_map)
            if path is None:
                continue
            total_length = float(sum(length_map.get(s, 0.0) for s in path))
            candidate_paths.append((total_length, path))

    if candidate_paths:
        _, best_path = max(candidate_paths, key=lambda item: item[0])
        return best_path

    # Fallback: default to the internal zone order if no upstream zones feed into it.
    fallback_start = outlet_segment
    for seg_id, row in zone_rows.iterrows():
        if _is_missing(row["downstream_id"]):
            continue
        if seg_id not in downstream_map:
            continue
        path = traverse_path(seg_id, outlet_segment, downstream_map)
        if path:
            fallback_start = seg_id
            break
    fallback_path = traverse_path(fallback_start, outlet_segment, downstream_map)
    if fallback_path is None:
        raise ValueError("Failed to identify a valid mainstem path.")
    return fallback_path


def collect_contributing_segments(
    main_path: Sequence[str],
    channels_meta: pd.DataFrame,
) -> List[Dict[str, object]]:
    """
    Identify every segment whose downstream chain feeds the main path.

    Returns a list of dictionaries with:
        - segment_id: contributing segment
        - zone_id: owning parameter zone
        - connect_segment: first segment on the main path that receives the flow
        - link_segment: segment whose downstream node touches the main path
        - path: the ordered list from the contributing segment to (but excluding) the main path
    """
    downstream_map = build_downstream_map(channels_meta)
    main_set = set(main_path)
    contributions: List[Dict[str, object]] = []

    for seg_id in channels_meta.index:
        if seg_id in main_set:
            continue

        path_sequence: List[str] = []
        current = seg_id
        visited: set[str] = set()

        while current and current not in main_set and current not in visited:
            path_sequence.append(current)
            visited.add(current)
            current = downstream_map.get(current)

        if not current or current not in main_set:
            continue

        contributions.append(
            {
                "segment_id": seg_id,
                "zone_id": channels_meta.loc[seg_id, "zone_id"],
                "connect_segment": current,
                "link_segment": path_sequence[-1] if path_sequence else seg_id,
                "path": path_sequence,
            }
        )

    return contributions


def entry_point_from_segments(
    link_segment: str,
    connect_segment: str,
    channel_geoms: gpd.GeoSeries,
) -> Point:
    """Compute the inflow point where a tributary branch joins the mainstem."""
    link_geom = channel_geoms.get(link_segment)
    connect_geom = channel_geoms.get(connect_segment)
    if link_geom is None or connect_geom is None:
        raise ValueError(f"Missing geometry for {link_segment} or {connect_segment}.")

    # Use the endpoint of the link segment closest to the mainstem geometry.
    endpoints = list(link_geom.boundary.geoms) if hasattr(link_geom.boundary, "geoms") else [link_geom.boundary]
    if not endpoints:
        return Point(link_geom.coords[-1])
    closest = min(endpoints, key=lambda pt: pt.distance(connect_geom))
    return Point(closest.x, closest.y)


def plot_zone(zone_id: str = "P1") -> Path:
    """Create the updated overview plot for the specified parameter zone."""
    subbasins, channels, cross_sections = load_datasets()

    channels_meta = (
        channels.drop(columns="geometry")
        .set_index("segment_id")
        .copy()
    )
    channel_geoms = channels.set_index("segment_id").geometry

    main_path = compute_main_path(zone_id, channels_meta)
    contributions = collect_contributing_segments(main_path, channels_meta)

    main_start = main_path[0]
    start_zone = channels_meta.loc[main_start, "zone_id"]
    filtered: List[Dict[str, object]] = []
    for item in contributions:
        connect_seg = item["connect_segment"]
        seg_zone = item["zone_id"]
        if connect_seg == main_start and seg_zone != start_zone:
            continue
        filtered.append(item)
    contributions = filtered

    # Gather auxiliary collections for plotting.
    main_set = set(main_path)
    branch_segments: set[str] = set()
    for item in contributions:
        branch_segments.update(item["path"])
    branch_segments -= main_set

    contributing_ids = {item["segment_id"] for item in contributions}
    relevant_segments = main_set | branch_segments | contributing_ids

    main_subbasins = subbasins[subbasins["subzone_id"].isin(main_set)]
    contrib_subbasins = subbasins[subbasins["subzone_id"].isin(contributing_ids)]
    base_subbasins = subbasins[subbasins["subzone_id"].isin(relevant_segments)]

    xs_center = cross_sections[
        (cross_sections["segment_id"].isin(main_set)) & (cross_sections["distance_from_center_m"] == 0)
    ].copy()
    xs_gdf = gpd.GeoDataFrame(
        xs_center,
        geometry=gpd.points_from_xy(xs_center["x"], xs_center["y"]),
        crs=subbasins.crs,
    )

    # Aggregate inflow sources by their receiving mainstem segment to simplify labelling.
    grouped_entries: Dict[str, Dict[str, object]] = {}
    for item in contributions:
        connect_seg = item["connect_segment"]
        grouped = grouped_entries.setdefault(
            connect_seg,
            {"sources": [], "link_segment": item["link_segment"]},
        )
        grouped["sources"].append(item["segment_id"])
        # Prefer a tributary link segment (not on the main path) when available.
        if grouped["link_segment"] in main_set and item["link_segment"] not in main_set:
            grouped["link_segment"] = item["link_segment"]

    entry_records = []
    for connect_seg, data in grouped_entries.items():
        link_seg = data["link_segment"]
        entry_point = entry_point_from_segments(link_seg, connect_seg, channel_geoms)
        entry_records.append(
            {
                "connect_segment": connect_seg,
                "sources": sorted(data["sources"]),
                "geometry": entry_point,
            }
        )
    entry_gdf = gpd.GeoDataFrame(entry_records, geometry="geometry", crs=subbasins.crs)

    fig, ax = plt.subplots(figsize=(11, 10))

    if not base_subbasins.empty:
        base_subbasins.plot(ax=ax, color="#f7f7f7", edgecolor="#bdbdbd", linewidth=0.4, alpha=1.0)

    if not contrib_subbasins.empty:
        contrib_subbasins.plot(ax=ax, color="#c6dbef", edgecolor="#6baed6", linewidth=0.9, alpha=0.85)

    if not main_subbasins.empty:
        main_subbasins = main_subbasins.set_index("subzone_id").loc[list(main_path)].reset_index()
        main_subbasins.plot(ax=ax, color="#fee8c8", edgecolor="#e6550d", linewidth=1.1, alpha=0.9)

    branch_lines = channels[channels["segment_id"].isin(branch_segments)]
    if not branch_lines.empty:
        branch_lines.plot(ax=ax, color="#66a9c9", linewidth=1.5, linestyle="--", alpha=0.9)

    main_lines = channels[channels["segment_id"].isin(main_path)]
    if not main_lines.empty:
        main_lines = main_lines.set_index("segment_id").loc[list(main_path)].reset_index()
        main_lines.plot(ax=ax, color="#0b3954", linewidth=2.4, alpha=1.0)

    if not xs_gdf.empty:
        xs_gdf.plot(
            ax=ax,
            marker="o",
            color="#d73027",
            edgecolor="white",
            linewidth=0.4,
            markersize=30,
            zorder=6,
        )

    if not entry_gdf.empty:
        entry_gdf.plot(
            ax=ax,
            marker="*",
            color="#f03b20",
            edgecolor="white",
            linewidth=0.6,
            markersize=120,
            zorder=7,
        )
        for _, row in entry_gdf.iterrows():
            label = ", ".join(format_suffix_list(row["sources"]))
            ax.annotate(
                label,
                xy=(row.geometry.x, row.geometry.y),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=7,
                color="#4d4d4d",
                ha="left",
                va="bottom",
            )

    ax.set_title(
        f"{zone_id} Mainstem with Cross Sections and Tributary Inflows",
        fontsize=15,
    )
    ax.set_xlabel("Projected X")
    ax.set_ylabel("Projected Y")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.5)

    legend_elements = [
        Patch(facecolor="#fee8c8", edgecolor="#e6550d", label="Mainstem subbasins"),
        Patch(facecolor="#c6dbef", edgecolor="#6baed6", label="Inflow subbasins"),
        Line2D([0], [0], color="#0b3954", lw=2.4, label="Mainstem channel"),
        Line2D([0], [0], color="#66a9c9", lw=1.5, linestyle="--", label="Tributary channels"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            markerfacecolor="#d73027",
            markersize=8,
            linewidth=0,
            label="Cross-section nodes",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="white",
            markerfacecolor="#f03b20",
            markersize=12,
            linewidth=0,
            label="Aggregated inflow point",
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower left")

    plt.tight_layout()

    output_path = RESULTS_ROOT / "10_hydrodynamic_run" / "figures" / f"{zone_id.lower()}_zone_layout.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=320)
    plt.close(fig)

    print("Mainstem segments (upstream -> downstream):")
    print("  " + " -> ".join(segment_suffix(seg) for seg in main_path))
    if contributions:
        print("\nContributing subbasins grouped by mainstem connection:")
        for entry in entry_records:
            connect = segment_suffix(entry["connect_segment"])
            sources = ", ".join(format_suffix_list(entry["sources"]))
            print(f"  {connect} <= {sources}")

    return output_path


def main():
    output_path = plot_zone("P1")
    print(f"\nSaved plot to: {output_path}")


if __name__ == "__main__":
    main()
