"""Generate longitudinal, cross-section, and hydrograph plots for Upper Truckee."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


def _load_channel_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    table.set_index("segment_id", inplace=True)
    return table


def _configure_font() -> None:
    preferred = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    for name in preferred:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams["font.family"] = name
            break
        except ValueError:
            continue
    plt.rcParams["axes.unicode_minus"] = False


def _load_profile(profile_dir: Path, segment_id: str) -> Tuple[np.ndarray, np.ndarray]:
    csv_path = profile_dir / f"{segment_id}_profile.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    return df["distance_m"].to_numpy(), df["elevation_m"].to_numpy()


def _load_cross_section(cross_dir: Path, segment_id: str) -> pd.DataFrame:
    csv_path = cross_dir / f"{segment_id}_cross_sections.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    return df


def _wetted_properties(xs: np.ndarray, ys: np.ndarray, elev: np.ndarray, water_level: float) -> Tuple[float, float]:
    area = 0.0
    wetted = 0.0
    for i in range(len(xs) - 1):
        x0, y0, z0 = xs[i], ys[i], elev[i]
        x1, y1, z1 = xs[i + 1], ys[i + 1], elev[i + 1]
        seg_len = math.hypot(x1 - x0, y1 - y0)
        if seg_len == 0:
            continue
        zmin = min(z0, z1)
        zmax = max(z0, z1)
        if water_level <= zmin:
            continue
        if water_level >= zmax:
            depth0 = water_level - z0
            depth1 = water_level - z1
            area += (depth0 + depth1) * 0.5 * seg_len
            wetted += seg_len
        else:
            if z0 < water_level < z1:
                frac = (water_level - z0) / max(z1 - z0, 1e-6)
                submerged = seg_len * frac
                depth0 = water_level - z0
                area += depth0 * 0.5 * submerged
                wetted += submerged
            elif z1 < water_level < z0:
                frac = (water_level - z1) / max(z0 - z1, 1e-6)
                submerged = seg_len * frac
                depth1 = water_level - z1
                area += depth1 * 0.5 * submerged
                wetted += submerged
    return area, wetted


def solve_stage(xs: np.ndarray, ys: np.ndarray, elev: np.ndarray, slope: float, discharge: float, n: float = 0.035) -> float:
    if discharge <= 0:
        return float(np.min(elev))
    slope = max(slope, 1e-4)
    zmin = float(np.min(elev))
    zmax = float(np.max(elev))
    lower = zmin + 0.01
    upper = zmax + 20.0

    def manning(level: float) -> float:
        area, wetted = _wetted_properties(xs, ys, elev, level)
        if area == 0 or wetted == 0:
            return 0.0
        radius = area / wetted
        return (area * (radius ** (2 / 3)) * (slope ** 0.5)) / n

    q_upper = manning(upper)
    while q_upper < discharge:
        upper += 10.0
        if upper > zmax + 200:
            return upper
        q_upper = manning(upper)

    for _ in range(60):
        mid = 0.5 * (lower + upper)
        q_mid = manning(mid)
        if abs(q_mid - discharge) < 1e-3:
            return mid
        if q_mid < discharge:
            lower = mid
        else:
            upper = mid
    return 0.5 * (lower + upper)


def select_station(section: pd.DataFrame) -> pd.DataFrame:
    grouped = section.groupby("station_m")
    stations = grouped.size().index.to_numpy()
    representative = float(np.median(stations))
    nearest = min(stations, key=lambda s: abs(s - representative))
    return grouped.get_group(nearest)


def create_longitudinal_plot(segments: Iterable[str], profile_dir: Path, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    for seg in segments:
        distance, elevation = _load_profile(profile_dir, seg)
        ax.plot(distance, elevation, label=seg)
    ax.set_xlabel("沿河长 (m)")
    ax.set_ylabel("高程 (m)")
    ax.set_title("P3/P4 主干河道纵剖面")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def create_cross_section_plot(section: pd.DataFrame, stages: Dict[str, float], segment_id: str, output_path: Path) -> None:
    d = section["distance_from_center_m"].to_numpy()
    elev = section["elevation_m"].to_numpy()
    order = np.argsort(d)
    d = d[order]
    elev = elev[order]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(d, elev, label=f"{segment_id} 河床")
    for label, stage in stages.items():
        ax.hlines(stage, d.min(), d.max(), linestyles="--", label=label)
    ax.set_xlabel("距河心距离 (m)")
    ax.set_ylabel("高程 (m)")
    ax.set_title(f"{segment_id} 横断面")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def create_hydrograph_plot(
    label: str,
    discharge: pd.Series,
    stage: pd.Series,
    output_path: Path,
) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(stage.index, stage.values, color="tab:blue", label="水位 (m)")
    ax1.set_ylabel("水位 (m)")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax2 = ax1.twinx()
    ax2.plot(discharge.index, discharge.values, color="tab:orange", label="流量 (m$^3$/s)")
    ax2.set_ylabel("流量 (m$^3$/s)")
    ax1.set_title(label)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def load_discharge_series(flow_path: Path, forcing_path: Path) -> pd.Series:
    flows = pd.read_csv(flow_path, header=None, names=["index", "discharge_cms"])["discharge_cms"]
    try:
        index = pd.read_csv(forcing_path, parse_dates=["Timestamp"])["Timestamp"]
    except Exception:
        index = pd.date_range(start="2023-01-01", periods=len(flows), freq="H")
    if len(index) >= len(flows):
        index = index[: len(flows)]
    else:
        index = pd.date_range(start=index.iloc[0], periods=len(flows), freq="H")
    return pd.Series(flows.to_numpy(), index=index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Upper Truckee diagnostic plots.")
    parser.add_argument("--results-root", type=Path, default=Path("results/upper_truckee_channel_demo"))
    parser.add_argument("--entry-segment", type=str, default="P3_sub51")
    parser.add_argument("--exit-segment", type=str, default="P4_sub2")
    parser.add_argument("--manning-n", type=float, default=0.035)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    results_root = args.results_root
    intermediate_dir = results_root / "intermediate"
    param_dir = results_root / "parameters"
    figures_dir = (
        args.output_dir if args.output_dir is not None else results_root / "hydro_project" / "figures"
    )
    figures_dir.mkdir(parents=True, exist_ok=True)

    _configure_font()

    profile_dir = intermediate_dir / "channel_profiles"
    cross_dir = intermediate_dir / "channel_cross_sections"
    channel_table = _load_channel_table(param_dir / "parameter_channels.csv")

    # Longitudinal profiles
    create_longitudinal_plot(
        [args.entry_segment, args.exit_segment],
        profile_dir,
        figures_dir / "upper_truckee_longitudinal_profiles.png",
    )

    # Discharge series
    discharge_entry = load_discharge_series(
        results_root / "hydro_project" / "baseline" / "P3.csv",
        intermediate_dir / "storm_forcing.csv",
    )
    discharge_exit = load_discharge_series(
        results_root / "hydro_project" / "baseline" / "P4.csv",
        intermediate_dir / "storm_forcing.csv",
    )

    # Stage series using Manning approximation
    def compute_stage_series(segment_id: str, discharge: pd.Series) -> pd.Series:
        section = select_station(_load_cross_section(cross_dir, segment_id))
        xs = section["x"].to_numpy()
        ys = section["y"].to_numpy()
        elev = section["elevation_m"].to_numpy()
        slope = float(channel_table.loc[segment_id, "slope"]) if segment_id in channel_table.index else 0.001
        stages = [
            solve_stage(xs, ys, elev, slope, float(q), n=args.manning_n) for q in discharge.to_numpy()
        ]
        return pd.Series(stages, index=discharge.index)

    stage_entry = compute_stage_series(args.entry_segment, discharge_entry)
    stage_exit = compute_stage_series(args.exit_segment, discharge_exit)

    create_hydrograph_plot(
        "P3 入口流量-水位过程线",
        discharge_entry,
        stage_entry,
        figures_dir / "P3_entry_hydrograph.png",
    )
    create_hydrograph_plot(
        "P4 出口流量-水位过程线",
        discharge_exit,
        stage_exit,
        figures_dir / "P4_exit_hydrograph.png",
    )

    # Cross-section plots (using representative stages)
    entry_section = select_station(_load_cross_section(cross_dir, args.entry_segment))
    exit_section = select_station(_load_cross_section(cross_dir, args.exit_segment))
    entry_levels = {
        "低水位": float(np.percentile(stage_entry, 10)),
        "平均水位": float(stage_entry.mean()),
        "高水位": float(np.percentile(stage_entry, 90)),
    }
    exit_levels = {
        "低水位": float(np.percentile(stage_exit, 10)),
        "平均水位": float(stage_exit.mean()),
        "高水位": float(np.percentile(stage_exit, 90)),
    }
    create_cross_section_plot(
        entry_section,
        entry_levels,
        args.entry_segment,
        figures_dir / "P3_entry_cross_section.png",
    )
    create_cross_section_plot(
        exit_section,
        exit_levels,
        args.exit_segment,
        figures_dir / "P4_exit_cross_section.png",
    )

    print("Figures saved to:", figures_dir)


if __name__ == "__main__":
    main()
