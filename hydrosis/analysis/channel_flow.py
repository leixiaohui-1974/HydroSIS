from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _read_csv_series(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a two-column CSV (time,value) as numpy arrays."""
    data = np.loadtxt(path, delimiter=",", dtype=float)
    if data.ndim == 1:
        # Single row fallback
        return np.array([data[0]], dtype=float), np.array([data[1]], dtype=float)
    return data[:, 0], data[:, 1]


def _align_series(series: np.ndarray, length: int) -> np.ndarray:
    """Ensure time series match the reference simulation length."""

    if len(series) == length:
        return series.copy()
    aligned = np.zeros(length, dtype=float)
    count = min(len(series), length)
    aligned[:count] = series[:count]
    return aligned


def compute_channel_flows(
    parameter_dir: Path,
    baseline_dir: Path,
    intermediate_dir: Path,
    local_dir: Optional[Path] = None,
) -> Path:
    """Compute channel flow timeseries using subzone runoff and network topology.

    When ``local_dir`` is provided (or defaults to ``baseline_local``), routed
    discharge for each parameter subzone is used directly.  Otherwise the
    routine falls back to distributing zone totals by subzone area ratios.
    """
    subbasin_csv = parameter_dir / "parameter_subbasins.csv"
    channel_csv = parameter_dir / "parameter_channels.csv"

    if not subbasin_csv.exists() or not channel_csv.exists():
        raise FileNotFoundError("Parameter directory missing required CSV files.")

    # Load subzone metadata (areas, zones)
    subzone_area: Dict[str, float] = {}
    zone_totals: Dict[str, float] = defaultdict(float)
    with subbasin_csv.open("r", encoding="utf-8") as handle:
        next(handle)  # header
        for line in handle:
            if not line.strip():
                continue
            parts = line.strip().split(",")
            zone_id, subzone_id = parts[0], parts[1]
            area_cells = float(parts[2])
            zone_totals[zone_id] += area_cells
            subzone_area[subzone_id] = area_cells

    # Load zone runoff timeseries
    zone_series: Dict[str, np.ndarray] = {}
    time_index: np.ndarray | None = None
    if not baseline_dir.exists():
        raise FileNotFoundError("Baseline runoff directory not generated.")

    for csv_file in baseline_dir.glob("*.csv"):
        zone_id = csv_file.stem.upper()
        time, values = _read_csv_series(csv_file)
        if time_index is None:
            time_index = time
        elif len(time_index) != len(time):
            raise ValueError("Inconsistent time index length across zone runoff CSV files.")
        zone_series[zone_id] = values

    if not zone_series:
        raise RuntimeError(f"No zone runoff CSV files found in {baseline_dir}.")

    time_index = np.asarray(time_index, dtype=float)
    dt = float(time_index[1] - time_index[0]) if len(time_index) > 1 else 1.0

    # Load channel topology
    segment_info: Dict[str, Dict[str, object]] = {}
    downstream_map: Dict[str, str | None] = {}
    adjacency: Dict[str, List[str]] = defaultdict(list)
    upstream_counts: Dict[str, int] = {}

    with channel_csv.open("r", encoding="utf-8") as handle:
        next(handle)  # header
        for line in handle:
            if not line.strip():
                continue
            parts = line.strip().split(",")
            segment_id = parts[0]
            zone_id = parts[1]
            subzone_id = parts[2]
            downstream_id = parts[6] if parts[6] else None
            upstream_raw = parts[7] if len(parts) > 7 else ""
            upstream_ids = [u for u in upstream_raw.split(";") if u]

            segment_info[segment_id] = {
                "zone": zone_id,
                "subzone": subzone_id,
                "downstream": downstream_id,
                "upstream": upstream_ids,
            }
            downstream_map[segment_id] = downstream_id
            upstream_counts[segment_id] = len(upstream_ids)
            for upstream in upstream_ids:
                adjacency[upstream].append(segment_id)

    # Determine processing order (topological)
    queue: deque[str] = deque(seg for seg, degree in upstream_counts.items() if degree == 0)
    topo_order: List[str] = []
    if not queue:
        raise RuntimeError("Channel network does not contain head segments.")

    while queue:
        seg = queue.popleft()
        topo_order.append(seg)
        for downstream in adjacency.get(seg, []):
            upstream_counts[downstream] -= 1
            if upstream_counts[downstream] == 0:
                queue.append(downstream)

    if len(topo_order) != len(segment_info):
        raise RuntimeError("Cycle detected in channel network, unable to compute flows.")

    reference_series = next(iter(zone_series.values()))
    step_count = len(reference_series)
    zeros = np.zeros(step_count, dtype=float)

    if local_dir is None:
        local_dir = baseline_dir.with_name("baseline_local")
    local_dir = Path(local_dir)

    local_series: Dict[str, np.ndarray] = {}
    if local_dir.exists():
        for csv_path in local_dir.glob("*.csv"):
            sub_id = csv_path.stem.upper()
            _, values = _read_csv_series(csv_path)
            local_series[sub_id] = _align_series(values, step_count)

    local_runoff: Dict[str, np.ndarray] = {}

    for segment_id, info in segment_info.items():
        zone_id = info["zone"]
        subzone_id = info["subzone"]
        sub_key = subzone_id.upper()
        if sub_key in local_series:
            local_runoff[segment_id] = local_series[sub_key].copy()
            continue
        zone_total = zone_totals.get(zone_id)
        if not zone_total:
            local_runoff[segment_id] = zeros.copy()
            continue
        zone_flow = zone_series.get(zone_id)
        if zone_flow is None:
            local_runoff[segment_id] = zeros.copy()
            continue
        area = subzone_area.get(subzone_id, 0.0)
        ratio = area / zone_total if zone_total > 0 else 0.0
        local_runoff[segment_id] = _align_series(zone_flow, step_count) * ratio

    flow_series: Dict[str, np.ndarray] = {
        segment_id: local_runoff.get(segment_id, zeros.copy()).copy() for segment_id in segment_info
    }

    for segment_id in topo_order:
        total_flow = flow_series[segment_id]
        for downstream in adjacency.get(segment_id, []):
            flow_series[downstream] = flow_series.get(downstream, zeros.copy()) + total_flow

    # Write outputs
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    flow_path = intermediate_dir / "channel_flow_timeseries.csv"
    df = pd.DataFrame({"time_index": time_index})
    for segment_id in topo_order:
        df[segment_id] = flow_series[segment_id]
    df.to_csv(flow_path, index=False)

    summary_records = []
    for segment_id in topo_order:
        flows = flow_series[segment_id]
        local = local_runoff.get(segment_id, zeros)
        summary_records.append(
            {
                "segment_id": segment_id,
                "zone_id": segment_info[segment_id]["zone"],
                "subzone_id": segment_info[segment_id]["subzone"],
                "local_peak": float(local.max()) if len(local) else 0.0,
                "flow_peak": float(flows.max()) if len(flows) else 0.0,
                "flow_volume": float(flows.sum() * dt),
            }
        )

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(intermediate_dir / "channel_flow_summary.csv", index=False)

    return flow_path


def compare_channel_flows(
    parameter_dir: Path,
    baseline_dir: Path,
    aggregated_dir: Path,
    intermediate_dir: Path,
    local_dir: Optional[Path] = None,
    zones: Iterable[str] | None = None,
    output_filename: str = "channel_flow_comparison.png",
) -> Path:
    """Plot channel-flow recomputation against model aggregated outputs."""

    channel_csv = parameter_dir / "parameter_channels.csv"
    if not channel_csv.exists():
        raise FileNotFoundError(f"Channel CSV not found: {channel_csv}")

    aggregated_dir = Path(aggregated_dir)
    if not aggregated_dir.exists():
        raise FileNotFoundError(f"Aggregated results directory not found: {aggregated_dir}")

    intermediate_dir.mkdir(parents=True, exist_ok=True)
    flows_path = intermediate_dir / "channel_flow_timeseries.csv"
    if not flows_path.exists():
        flows_path = compute_channel_flows(
            parameter_dir,
            baseline_dir,
            intermediate_dir,
            local_dir=local_dir,
        )

    channel_df = pd.read_csv(flows_path)
    if "time_index" not in channel_df.columns:
        raise ValueError("Channel flow file missing 'time_index' column.")
    time_index = channel_df["time_index"].to_numpy(dtype=float)

    aggregated_series: Dict[str, np.ndarray] = {}
    aggregated_time: np.ndarray | None = None
    for csv_path in sorted(aggregated_dir.glob("*.csv")):
        zone_id = csv_path.stem.upper()
        steps, values = _read_csv_series(csv_path)
        if aggregated_time is None:
            aggregated_time = steps
        elif len(aggregated_time) != len(steps):
            raise ValueError(f"Aggregated series length mismatch in {csv_path}")
        aggregated_series[zone_id] = values

    if not aggregated_series:
        raise RuntimeError(f"No aggregated flow series found in {aggregated_dir}")

    if aggregated_time is not None and len(aggregated_time) != len(time_index):
        raise ValueError("Aggregated time index length differs from channel flow length.")

    channels_df = pd.read_csv(channel_csv)
    seg_zone = {row["segment_id"]: row["zone_id"] for _, row in channels_df.iterrows()}

    outlet_segments: Dict[str, str] = {}
    for zone_id in aggregated_series:
        zone_rows = channels_df[channels_df["zone_id"] == zone_id]
        candidate: str | None = None
        for _, row in zone_rows.iterrows():
            downstream = row["downstream_id"]
            if isinstance(downstream, float) and np.isnan(downstream):
                candidate = row["segment_id"]
                break
            downstream = str(downstream) if downstream is not None else ""
            if not downstream or seg_zone.get(downstream) != zone_id:
                candidate = row["segment_id"]
                break
        if candidate is None and not zone_rows.empty:
            candidate = zone_rows.iloc[0]["segment_id"]
        if candidate is None:
            continue
        outlet_segments[zone_id] = candidate

    if not outlet_segments:
        raise RuntimeError("Unable to identify outlet segments for any parameter zone.")

    requested_zones = None
    if zones is not None:
        requested_zones = {zone.upper() for zone in zones}

    zones = sorted(outlet_segments.keys())
    if requested_zones is not None:
        filtered = [zone for zone in zones if zone.upper() in requested_zones]
        if not filtered:
            raise ValueError(
                "Requested zones not found in outlet segments: "
                + ", ".join(sorted(requested_zones))
            )
        zones = filtered

    if not zones:
        raise RuntimeError("No parameter zones available for comparison.")

    computed_series: Dict[str, np.ndarray] = {}
    for zone_id in zones:
        segment = outlet_segments[zone_id]
        if segment not in channel_df.columns:
            raise KeyError(f"Segment {segment} not found in channel flow data.")
        computed_series[zone_id] = channel_df[segment].to_numpy(dtype=float)

    output_path = intermediate_dir / output_filename
    fig, axes = plt.subplots(len(zones), 1, figsize=(10, 3 * len(zones)), sharex=True)
    if len(zones) == 1:
        axes = [axes]  # type: ignore[assignment]

    stats_records: List[Dict[str, float | str]] = []
    diff_df = pd.DataFrame({"time_index": time_index})
    dt = float(time_index[1] - time_index[0]) if len(time_index) > 1 else 1.0
    dt_seconds = dt * 3600.0

    for ax, zone_id in zip(axes, zones):
        aggregated = aggregated_series.get(zone_id)
        computed = computed_series.get(zone_id)
        if aggregated is None or computed is None:
            continue
        if len(aggregated) != len(computed):
            raise ValueError(f"Length mismatch for zone {zone_id}")
        ax.plot(time_index, aggregated, label="HydroSIS aggregated", color="tab:green")
        ax.plot(
            time_index,
            computed,
            label=f"Recomputed ({outlet_segments[zone_id]})",
            color="tab:blue",
            linestyle="--",
        )
        ax.set_ylabel("Flow (m³/s)")
        ax.set_title(f"Zone {zone_id}")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right")

        diff = computed - aggregated
        diff_df[zone_id] = diff
        max_abs = float(np.max(np.abs(diff))) if len(diff) else 0.0
        max_idx = int(np.argmax(np.abs(diff))) if len(diff) else 0
        aggregated_peak = float(np.max(np.abs(aggregated))) if len(aggregated) else 0.0
        signed_volume_diff = float(np.sum(diff) * dt_seconds)
        abs_volume_diff = float(np.sum(np.abs(diff)) * dt_seconds)
        relative_peak_diff = max_abs / aggregated_peak if aggregated_peak > 0 else None

        stats_records.append(
            {
                "zone_id": zone_id,
                "outlet_segment": outlet_segments[zone_id],
                "max_abs_diff": max_abs,
                "rmse": float(np.sqrt(np.mean(diff**2))),
                "mean_bias": float(np.mean(diff)),
                "mean_abs_diff": float(np.mean(np.abs(diff))),
                "time_of_max_abs_diff": float(time_index[max_idx]) if len(time_index) else 0.0,
                "abs_volume_diff_m3": abs_volume_diff,
                "signed_volume_diff_m3": signed_volume_diff,
                "relative_peak_diff": relative_peak_diff,
            }
        )

    axes[-1].set_xlabel("Time index")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    stats_df = pd.DataFrame(stats_records)
    stats_df.to_csv(intermediate_dir / "channel_flow_comparison_stats.csv", index=False)
    diff_df.to_csv(intermediate_dir / "channel_flow_difference_timeseries.csv", index=False)

    return output_path


__all__ = ["compute_channel_flows", "compare_channel_flows"]
