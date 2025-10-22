"""Compare baseline and scenario routing results for selected subbasins."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd


def configure_font() -> None:
    preferred = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    for name in preferred:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams["font.family"] = name
            break
        except ValueError:
            continue
    plt.rcParams["axes.unicode_minus"] = False


def load_series(csv_path: Path, forcing_path: Path) -> pd.Series:
    flow = pd.read_csv(csv_path, header=None, names=["index", "discharge_cms"])["discharge_cms"]
    try:
        time_index = pd.read_csv(forcing_path, parse_dates=["Timestamp"])["Timestamp"]
    except Exception:
        time_index = pd.date_range(start="2023-01-01", periods=len(flow), freq="H")
    if len(time_index) >= len(flow):
        time_index = time_index[: len(flow)]
    else:
        time_index = pd.date_range(start=time_index.iloc[0], periods=len(flow), freq="H")
    return pd.Series(flow.to_numpy(), index=time_index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot routing method comparison for P3/P4 outlets.")
    parser.add_argument("--results-root", type=Path, default=Path("results/upper_truckee_channel_demo"))
    parser.add_argument("--scenario", type=str, default="hydraulic_p3p4", help="Scenario identifier to compare.")
    parser.add_argument(
        "--subbasins",
        nargs="*",
        default=["P3", "P4"],
        help="Subbasin identifiers to compare (default: P3 P4).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the figure (defaults to figures directory under results).",
    )
    args = parser.parse_args()

    root = args.results_root
    forcing_path = root / "intermediate" / "storm_forcing.csv"
    baseline_dir = root / "hydro_project" / "baseline"
    scenario_dir = root / "hydro_project" / args.scenario
    figures_dir = root / "hydro_project" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output if args.output is not None else figures_dir / f"{args.scenario}_flow_comparison.png"

    configure_font()

    num = len(args.subbasins)
    fig, axes = plt.subplots(num, 1, figsize=(10, 4 * num), sharex=True)
    if num == 1:
        axes = [axes]

    for ax, sub_id in zip(axes, args.subbasins):
        baseline_csv = baseline_dir / f"{sub_id}.csv"
        scenario_csv = scenario_dir / f"{sub_id}.csv"
        if not baseline_csv.exists() or not scenario_csv.exists():
            ax.text(0.5, 0.5, f"{sub_id} 缺少结果文件", transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            continue
        base_series = load_series(baseline_csv, forcing_path)
        scenario_series = load_series(scenario_csv, forcing_path)
        ax.plot(base_series.index, base_series.values, label="Muskingum（基线）", color="tab:blue")
        ax.plot(scenario_series.index, scenario_series.values, label=f"{args.scenario}", color="tab:orange", linestyle="--")
        ax.set_ylabel("流量 (m$^3$/s)")
        ax.set_title(f"{sub_id} 出口流量对比")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
    axes[-1].set_xlabel("时间")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print("Saved comparison to", output_path)


if __name__ == "__main__":
    main()
def configure_font() -> None:
    preferred = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    for name in preferred:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams["font.family"] = name
            break
        except ValueError:
            continue
    plt.rcParams["axes.unicode_minus"] = False
