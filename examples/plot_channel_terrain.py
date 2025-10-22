"""可视化与校正主干河道断面的 CLI 工具。

该脚本依赖 `hydrosis.analysis.channel_profile` 模块提供的通用算法，
生成单分区与多分区联合的三维可视化、二维热力图、纵剖面及中心线结果。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from matplotlib import font_manager

from hydrosis.analysis.channel_profile import (
    ChannelProfileConfig,
    build_zone_grid,
    extract_centerline,
    load_cross_sections,
    preprocess_cross_sections,
    run_channel_profile_model,
)


# ---------------------------------------------------------------------------
# 基础绘图辅助
# ---------------------------------------------------------------------------


def configure_matplotlib_font() -> None:
    """配置 matplotlib 字体，避免中文乱码。"""
    preferred = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    for name in preferred:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams["font.family"] = name
            break
        except ValueError:
            continue
    plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 三维与二维可视化
# ---------------------------------------------------------------------------


def make_zone_figure(zone_id: str, data: pd.DataFrame, output_dir: Path) -> Path:
    zone_df = data[data["zone_id"] == zone_id].copy()
    zone_df.sort_values(by=["global_station_m", "distance_from_center_m"], inplace=True)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=zone_df.get("global_station_m", zone_df["station_m"]),
            y=zone_df["distance_from_center_m"],
            z=zone_df["elevation_m"],
            mode="markers",
            marker=dict(size=2, color=zone_df["elevation_m"], colorscale="Viridis"),
        )
    )
    fig.update_layout(
        title=f"{zone_id} 主干河道三维断面点云",
        scene=dict(
            xaxis_title="沿程 (m)",
            yaxis_title="距河心距离 (m)",
            zaxis_title="高程 (m)",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    output_path = output_dir / f"{zone_id}_channel_terrain.html"
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def make_combined_figure(zones: Iterable[str], data: pd.DataFrame, output_dir: Path) -> Path:
    fig = go.Figure()
    for zone in zones:
        zone_df = data[data["zone_id"] == zone].copy()
        if zone_df.empty:
            continue
        zone_df.sort_values(by=["global_station_m", "distance_from_center_m"], inplace=True)
        fig.add_trace(
            go.Scatter3d(
                x=zone_df.get("global_station_m", zone_df["station_m"]),
                y=zone_df["distance_from_center_m"],
                z=zone_df["elevation_m"],
                mode="markers",
                marker=dict(size=2, opacity=0.7),
                name=zone,
            )
        )
    fig.update_layout(
        title="主干河道三维断面点云对比",
        scene=dict(
            xaxis_title="沿程 (m)",
            yaxis_title="距河心距离 (m)",
            zaxis_title="高程 (m)",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    output_path = output_dir / "combined_channel_terrain.html"
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def make_zone_surface(
    zone_id: str,
    data: pd.DataFrame,
    output_dir: Path,
    target_spacing: float,
) -> Path:
    zone_df = data[data["zone_id"] == zone_id].copy()
    if zone_df.empty:
        raise FileNotFoundError(f"未找到 {zone_id} 的断面数据")
    xs, ys, Z = build_zone_grid(
        zone_df,
        station_field="global_station_m",
        target_spacing=target_spacing,
    )

    surface = go.Surface(x=xs, y=ys, z=Z, colorscale="Viridis", showscale=True)
    fig = go.Figure(data=[surface])
    fig.update_layout(
        title=f"{zone_id} 主干河道三维网格",
        scene=dict(
            xaxis_title="沿程 (m)",
            yaxis_title="距河心距离 (m)",
            zaxis_title="高程 (m)",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    output_path = output_dir / f"{zone_id}_channel_surface.html"
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def make_combined_surface(zones: Iterable[str], data: pd.DataFrame, output_dir: Path) -> Path:
    combined_frames: list[pd.DataFrame] = []
    for zone in zones:
        zone_df = data[data["zone_id"] == zone].copy()
        if zone_df.empty:
            continue
        combined_frames.append(zone_df)

    if not combined_frames:
        raise FileNotFoundError("未找到任何可用于生成联合河道网格的断面数据。")

    combined_df = pd.concat(combined_frames, axis=0)
    combined_df.sort_values(by=["global_station_m", "distance_from_center_m"], inplace=True)

    xs, ys, Z = build_zone_grid(
        combined_df,
        station_field="global_station_m",
        target_spacing=200.0,
    )

    surface = go.Surface(x=xs, y=ys, z=Z, colorscale="Viridis", showscale=True)
    fig = go.Figure(data=[surface])
    fig.update_layout(
        title="主干河道三维网格（分区拼接）",
        scene=dict(
            xaxis_title="沿程 (m)",
            yaxis_title="距河心距离 (m)",
            zaxis_title="高程 (m)",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    output_path = output_dir / "combined_channel_surface.html"
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def make_static_visual(
    zone_id: str,
    data: pd.DataFrame,
    output_dir: Path,
    band_width: float,
    target_spacing: float,
) -> Path:
    """生成 2D 热力图 + 纵剖面，更直观呈现河槽地形。"""
    zone_df = data[data["zone_id"] == zone_id].copy()
    if zone_df.empty:
        raise FileNotFoundError(f"未找到 {zone_id} 的断面数据")

    xs, ys, Z = build_zone_grid(
        zone_df,
        station_field="global_station_m",
        target_spacing=target_spacing,
    )
    centerline = extract_centerline(zone_df, band_width=band_width)
    chain_m = centerline["global_station_m"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    im = axes[0].pcolormesh(xs, ys, Z, shading="auto", cmap="terrain")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("沿程 (m)")
    axes[0].set_ylabel("距河心距离 (m)")
    axes[0].set_title(f"{zone_id} 河槽地形（热力图）")
    fig.colorbar(im, ax=axes[0], label="高程 (m)")

    axes[1].plot(chain_m, centerline["elevation_m"], color="tab:blue")
    axes[1].set_xlabel("沿程 (m)")
    axes[1].set_ylabel("高程 (m)")
    axes[1].set_title(f"{zone_id} 主干纵剖面")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    output_path = output_dir / f"{zone_id}_channel_static.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def make_combined_centerline_plot(
    centerlines: dict[str, pd.DataFrame],
    output_dir: Path,
) -> tuple[Path, Path]:
    """生成联通的 P3-P4 河槽纵剖图，并导出数据表。"""
    if not centerlines:
        raise FileNotFoundError("缺少中心线数据，无法绘制联合纵剖面。")

    output_dir.mkdir(parents=True, exist_ok=True)

    combined_records: list[pd.DataFrame] = []
    fig, ax = plt.subplots(figsize=(12, 5))

    boundary_marks: list[float] = []
    for zone, profile in centerlines.items():
        df = profile.sort_values("global_station_m").copy()
        df["zone_id"] = zone
        combined_records.append(
            df[
                [
                    "zone_id",
                    "global_station_m",
                    "base_elevation_m",
                    "local_corrected_elevation_m",
                    "global_corrected_elevation_m",
                ]
            ]
        )

        chain_km = df["global_station_m"] / 1000.0
        ax.plot(chain_km, df["base_elevation_m"], linestyle="--", linewidth=1.1, label=f"{zone} 原始")
        ax.plot(chain_km, df["global_corrected_elevation_m"], linewidth=2.0, label=f"{zone} 修正")
        boundary_marks.append(chain_km.iloc[-1])

    for boundary in boundary_marks[:-1]:
        ax.axvline(boundary, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)

    ax.set_xlabel("沿程 (km)")
    ax.set_ylabel("河槽中心线高程 (m)")
    ax.set_title("P3-P4 主干河槽纵剖面对比（原始 vs 修正）")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(ncol=2, fontsize=9)

    figure_path = output_dir / "combined_centerline_profile.png"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)

    combined_df = pd.concat(combined_records, axis=0, ignore_index=True)
    combined_csv = output_dir / "combined_centerline_profile.csv"
    combined_df.to_csv(combined_csv, index=False)

    return figure_path, combined_csv


# ---------------------------------------------------------------------------
# CLI 执行入口
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="生成主干河道三维与二维可视化")
    parser.add_argument(
        "--zones",
        nargs="*",
        default=["P3", "P4"],
        help="需要可视化的分区标识，默认 P3 与 P4",
    )
    parser.add_argument(
        "--cross-sections",
        type=Path,
        default=Path("results/upper_truckee_channel_demo/intermediate/analysis/main_channel_cross_sections.csv"),
        help="包含汇总断面信息的 CSV",
    )
    parser.add_argument(
        "--segments",
        type=Path,
        default=None,
        help="主干分段列表（CSV）。若未提供，则与断面文件同目录下的 main_channel_segments.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/upper_truckee_channel_demo/intermediate/analysis"),
        help="输出目录，默认写入分析目录",
    )
    parser.add_argument(
        "--min-variation",
        type=float,
        default=0.1,
        help="剔除断面时允许的最小高程变化（m），默认 0.1",
    )
    parser.add_argument(
        "--centerline-bandwidth",
        type=float,
        default=40.0,
        help="提取中心线时的横向带宽（m），默认 40",
    )
    parser.add_argument(
        "--monotonic-tolerance",
        type=float,
        default=0.0,
        help="沿程单调约束允许的最小降幅（m）；0 表示严格不升高",
    )
    parser.add_argument(
        "--grid-spacing",
        type=float,
        default=200.0,
        help="热力图/三维网格的沿程插值步长（m），默认 200",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_matplotlib_font()

    cross_sections = load_cross_sections(args.cross_sections, args.zones)
    segments_path = args.segments or args.cross_sections.with_name("main_channel_segments.csv")
    config = ChannelProfileConfig(
        min_variation=args.min_variation,
        band_width=args.centerline_bandwidth,
        target_spacing=args.grid_spacing,
        monotonic_tolerance=args.monotonic_tolerance,
    )
    preprocess_result = preprocess_cross_sections(
        cross_sections,
        segments_path,
        args.zones,
        config=config,
    )
    model_result = run_channel_profile_model(preprocess_result)
    data = model_result.cross_sections
    centerlines = model_result.centerlines
    profile_cfg = model_result.config

    produced_html: list[Path] = []
    produced_surface: list[Path] = []
    for zone in args.zones:
        try:
            produced_html.append(make_zone_figure(zone, data, output_dir))
            produced_surface.append(make_zone_surface(zone, data, output_dir, profile_cfg.target_spacing))
        except FileNotFoundError:
            continue
    combined_html = make_combined_figure(args.zones, data, output_dir)
    combined_surface = make_combined_surface(args.zones, data, output_dir)

    static_zone_outputs: list[Path] = []
    for zone in args.zones:
        try:
            static_zone_outputs.append(
                    make_static_visual(
                        zone,
                        data,
                        output_dir,
                        band_width=profile_cfg.band_width,
                        target_spacing=profile_cfg.target_spacing,
                    )
            )
        except FileNotFoundError:
            continue

    # 输出子分区热力图（仍按 station_m 展开，便于诊断）
    subzones = data[["zone_id", "subzone_id"]].drop_duplicates()
    static_subzone_outputs: list[Path] = []
    for _, row in subzones.iterrows():
        zone_id = row["zone_id"]
        sub_id = row["subzone_id"]
        sub_df = data[(data["zone_id"] == zone_id) & (data["subzone_id"] == sub_id)].copy()
        if sub_df.empty:
            continue
        pivot = sub_df.pivot_table(
            index="distance_from_center_m",
            columns="station_m",
            values="elevation_m",
            aggfunc="mean",
        )
        pivot.sort_index(inplace=True)
        pivot.sort_index(axis=1, inplace=True)
        xs = pivot.columns.to_numpy()
        ys = pivot.index.to_numpy()
        Z = pivot.to_numpy()
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.pcolormesh(xs, ys, Z, shading="auto", cmap="terrain")
        ax.invert_yaxis()
        ax.set_xlabel("沿程 (m)")
        ax.set_ylabel("距河心距离 (m)")
        ax.set_title(f"{zone_id} - {sub_id} 河槽断面")
        fig.colorbar(im, ax=ax, label="高程 (m)")
        output_path = output_dir / f"{zone_id}_{sub_id}_section.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        static_subzone_outputs.append(output_path)

    centerline_outputs: list[Path] = []
    for zone_id, profile in centerlines.items():
        if profile.empty:
            continue
        output_path = output_dir / f"{zone_id}_centerline_profile.csv"
        profile.to_csv(output_path, index=False)
        centerline_outputs.append(output_path)

    combined_profile_plot: Optional[Path] = None
    combined_profile_csv: Optional[Path] = None
    try:
        combined_profile_plot, combined_profile_csv = make_combined_centerline_plot(centerlines, output_dir)
    except FileNotFoundError:
        combined_profile_plot = None
        combined_profile_csv = None

    print("生成的三维可视化文件：")
    for path in produced_html:
        print("  ", path)
    print("  ", combined_html)
    print("生成的三维网格文件：")
    for path in produced_surface:
        print("  ", path)
    print("  ", combined_surface)
    print("生成的二维热力图/纵剖面：")
    for path in static_zone_outputs:
        print("  ", path)
    print("生成的子流域断面热力图：")
    for path in static_subzone_outputs:
        print("  ", path)
    if centerline_outputs:
        print("生成的中心线校正成果：")
        for path in centerline_outputs:
            print("  ", path)
    if combined_profile_plot:
        print("  ", combined_profile_plot)
    if combined_profile_csv:
        print("  ", combined_profile_csv)


if __name__ == "__main__":
    main()
