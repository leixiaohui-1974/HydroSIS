from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Set, Tuple

import numpy as np
import pandas as pd


def _read_time_series(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """读取 `write_time_series` 格式的两列 CSV。"""
    data = np.loadtxt(path, delimiter=",", dtype=float)
    if data.ndim == 1:
        return np.array([data[0]], dtype=float), np.array([data[1]], dtype=float)
    return data[:, 0], data[:, 1]


def _detect_precip_column(df: pd.DataFrame) -> str:
    """返回降雨强度所在的第一个数值列。"""
    for column in df.columns[1:]:
        if pd.api.types.is_numeric_dtype(df[column]):
            return column
    raise ValueError("无法在降雨文件中找到数值型降雨列。")


def _compute_time_step_hours(timestamps: pd.Series) -> float:
    """估算连续时间序列的步长（单位：小时）。"""
    deltas = timestamps.diff().dropna()
    if deltas.empty:
        return 1.0
    delta = deltas.iloc[0]
    if isinstance(delta, pd.Timedelta):
        seconds = delta.total_seconds()
        return seconds / 3600.0 if seconds > 0 else 1.0
    return 1.0


def _collect_upstream(zone_id: str, upstream_map: Mapping[str, Sequence[str]]) -> Set[str]:
    """回溯 zone 的所有上游 zone（含自身）。"""
    stack: List[str] = [zone_id]
    visited: Set[str] = set()
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        for upstream in upstream_map.get(current, []):
            if upstream not in visited:
                stack.append(upstream)
    return visited


def _align_series(series: np.ndarray, length: int) -> np.ndarray:
    """调整时序长度（超长截断，缺失补零）。"""
    if len(series) == length:
        return series
    aligned = np.zeros(length, dtype=float)
    count = min(len(series), length)
    aligned[:count] = series[:count]
    return aligned


def compute_zone_runoff_coefficients(
    parameter_dir: Path,
    aggregated_dir: Path,
    local_dir: Path,
    precipitation_path: Path,
    output_path: Path,
) -> Path:
    """根据参数区拓扑、局地径流与降雨，计算每个参数区的径流系数。"""

    parameter_dir = Path(parameter_dir)
    aggregated_dir = Path(aggregated_dir)
    local_dir = Path(local_dir)
    precipitation_path = Path(precipitation_path)
    output_path = Path(output_path)

    zone_csv = parameter_dir / "parameter_zones.csv"
    subzone_csv = parameter_dir / "parameter_subbasins.csv"

    if not zone_csv.exists() or not subzone_csv.exists():
        raise FileNotFoundError("参数目录缺少 parameter_zones.csv 或 parameter_subbasins.csv。")
    if not aggregated_dir.exists():
        raise FileNotFoundError(f"聚合结果目录不存在: {aggregated_dir}")
    if not local_dir.exists():
        raise FileNotFoundError(f"局地径流目录不存在: {local_dir}")
    if not precipitation_path.exists():
        raise FileNotFoundError(f"降雨时序文件不存在: {precipitation_path}")

    zones_df = pd.read_csv(zone_csv)
    subzones_df = pd.read_csv(subzone_csv)

    zone_area_km2: Dict[str, float] = {}
    zone_downstream: Dict[str, str | None] = {}
    for _, row in zones_df.iterrows():
        zone_id = str(row["zone_id"])
        zone_area_km2[zone_id] = float(row["area_km2"])
        downstream = row.get("downstream_id")
        if isinstance(downstream, float) and np.isnan(downstream):
            downstream = None
        elif downstream in ("", "None"):
            downstream = None
        else:
            downstream = str(downstream)
        zone_downstream[zone_id] = downstream

    zone_subzones: Dict[str, List[str]] = defaultdict(list)
    for _, row in subzones_df.iterrows():
        zone_id = str(row["zone_id"])
        subzone_id = str(row["subzone_id"])
        zone_subzones[zone_id].append(subzone_id)

    aggregated_series: Dict[str, np.ndarray] = {}
    time_index: np.ndarray | None = None
    for csv_path in sorted(aggregated_dir.glob("*.csv")):
        zone_id = csv_path.stem.upper()
        steps, values = _read_time_series(csv_path)
        if time_index is None:
            time_index = steps
        elif len(time_index) != len(steps):
            raise ValueError(f"聚合结果 {csv_path} 的时间长度与其他序列不一致。")
        aggregated_series[zone_id] = values

    if not aggregated_series:
        raise RuntimeError(f"聚合目录 {aggregated_dir} 中没有可用的 CSV 时序。")

    step_count = len(next(iter(aggregated_series.values())))

    local_series_raw: Dict[str, np.ndarray] = {}
    for csv_path in local_dir.glob("*.csv"):
        key = csv_path.stem.upper()
        _, values = _read_time_series(csv_path)
        local_series_raw[key] = values

    precip_df = pd.read_csv(precipitation_path)
    if precip_df.empty:
        raise ValueError(f"降雨文件 {precipitation_path} 为空。")

    time_col = precip_df.columns[0]
    try:
        timestamps = pd.to_datetime(precip_df[time_col])
        dt_hours = _compute_time_step_hours(pd.Series(timestamps))
    except (ValueError, TypeError):
        dt_hours = 1.0
    precip_column = _detect_precip_column(precip_df)
    precip_series = precip_df[precip_column].astype(float).to_numpy()
    if len(precip_series) != step_count:
        raise ValueError("降雨序列长度与聚合序列长度不匹配。")

    rainfall_depth_mm = float(precip_series.sum() * dt_hours)
    rainfall_depth_m = rainfall_depth_mm / 1000.0
    dt_seconds = dt_hours * 3600.0

    upstream_map: Dict[str, List[str]] = defaultdict(list)
    for zone_id, downstream in zone_downstream.items():
        if downstream:
            upstream_map[downstream].append(zone_id)

    records: List[Dict[str, object]] = []
    for zone_id in sorted(zone_area_km2.keys()):
        zone_key = zone_id.upper()
        if zone_key in local_series_raw:
            local_combined = _align_series(local_series_raw[zone_key], step_count)
        else:
            local_subzones = zone_subzones.get(zone_id, [])
            combined = np.zeros(step_count, dtype=float)
            found = False
            for sub_id in local_subzones:
                sub_key = sub_id.upper()
                if sub_key not in local_series_raw:
                    continue
                combined += _align_series(local_series_raw[sub_key], step_count)
                found = True
            local_combined = combined if found else np.zeros(step_count, dtype=float)

        aggregated = aggregated_series.get(zone_id)
        if aggregated is None:
            aggregated = np.zeros(step_count, dtype=float)

        zone_area = zone_area_km2.get(zone_id, 0.0)
        local_runoff_volume = float(local_combined.sum() * dt_seconds)
        aggregated_volume = float(aggregated.sum() * dt_seconds)

        local_rainfall_volume = rainfall_depth_m * zone_area * 1_000_000.0

        upstream_zones = _collect_upstream(zone_id, upstream_map)
        upstream_area = sum(zone_area_km2.get(z, 0.0) for z in upstream_zones)
        upstream_rainfall_volume = rainfall_depth_m * upstream_area * 1_000_000.0

        local_coeff = (
            local_runoff_volume / local_rainfall_volume if local_rainfall_volume > 0 else np.nan
        )
        upstream_coeff = (
            aggregated_volume / upstream_rainfall_volume if upstream_rainfall_volume > 0 else np.nan
        )

        records.append(
            {
                "zone_id": zone_id,
                "local_area_km2": zone_area,
                "upstream_area_km2": upstream_area,
                "rainfall_depth_mm": rainfall_depth_mm,
                "local_runoff_volume_m3": local_runoff_volume,
                "upstream_runoff_volume_m3": aggregated_volume,
                "local_rainfall_volume_m3": local_rainfall_volume,
                "upstream_rainfall_volume_m3": upstream_rainfall_volume,
                "local_runoff_coeff": local_coeff,
                "upstream_runoff_coeff": upstream_coeff,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)

    history_path = output_path.parent / "zone_runoff_coefficients_history.csv"
    run_timestamp = pd.Timestamp.utcnow().isoformat()
    history_df = df.copy()
    history_df.insert(0, "timestamp", run_timestamp)
    if history_path.exists():
        history_df.to_csv(history_path, mode="a", header=False, index=False)
    else:
        history_df.to_csv(history_path, index=False)

    return output_path


__all__ = ["compute_zone_runoff_coefficients"]
