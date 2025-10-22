# -*- coding: utf-8 -*-
"""Multi-model storm comparison showcasing extended rainfall response and channel-aware routing."""
import argparse
import os
import warnings
from pathlib import Path
import math
import copy
from typing import Dict, Iterable, List, Literal, Tuple, Callable
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from hydrosis.config import (
    DelineationConfig,
    IOConfig,
    ModelConfig,
    ParameterZoneConfig,
    RoutingModelConfig,
    RunoffModelConfig,
)
from hydrosis.model import Subbasin
import hydrosis.runoff.distributed_green_ampt  # noqa: F401
import hydrosis.runoff.simple  # noqa: F401
from hydrosis.reporting.markdown import MarkdownReportBuilder, TableData
import hydrosis.routing.simple  # noqa: F401  # ensure simple routing is registered
from hydrosis.routing.dynamic_wave import DynamicWaveRouting
from hydrosis.routing.muskingum import MuskingumRouting
from hydrosis.workflow import run_workflow
from hydrosis.testing.synthetic_datasets import write_synthetic_delineation_inputs

BASIN_AREA_KM2 = 50.0
MM_TO_M3_PER_KM2 = 1_000.0  # 1æ¯«ç±³è¦ç1å¹³æ¹å¬éç­äº1,000ç«æ¹ç±³
PRECIP_COLUMN = "éé¨å¼ºåº¦_æ¯«ç±³æ¯å°æ¶"
DEFAULT_STAGE_CURVE = {
    "coefficient": 28.0,
    "exponent": 0.58,
    "base_level": 0.35,
    "offset": 0.05,
}
STAGE_TIMESERIES_FILENAME = "water_level_timeseries.csv"
STAGE_PLOT_FILENAME = "water_level_timeseries.png"
FONT_CANDIDATES = [
    "Microsoft YaHei",
    "Microsoft YaHei UI",
    "SimHei",
    "SimSun",
    "NSimSun",
    "Source Han Sans SC",
    "Source Han Sans CN",
    "Noto Sans CJK SC",
    "Noto Sans SC",
    "PingFang SC",
    "PingFangSC-Regular",
    "WenQuanYi Micro Hei",
    "Arial Unicode MS",
]
FONT_FILE_CANDIDATES = [
    "msyh.ttc",
    "msyh.ttf",
    "msyhbd.ttc",
    "simhei.ttf",
    "simhei.ttc",
    "simsun.ttc",
    "nsimsun.ttc",
    "sourcehansanssc-regular.otf",
    "sourcehansanscn-regular.otf",
    "notosanscjksc-regular.otf",
    "notosanssc-regular.otf",
]


def configure_plot_fonts() -> None:
    """配置中文字体，避免图中汉字出现乱码。"""
    available_fonts = {f.name.lower(): f.name for f in font_manager.fontManager.ttflist}
    chosen: str | None = None
    for candidate in FONT_CANDIDATES:
        key = candidate.lower()
        for name_lower, original in available_fonts.items():
            if key in name_lower:
                chosen = original
                break
        if chosen:
            break

    if chosen is None:
        for font_path in font_manager.findSystemFonts() or []:
            filename = Path(font_path).name.lower()
            if filename in FONT_FILE_CANDIDATES:
                font_manager.fontManager.addfont(font_path)
                chosen = font_manager.FontProperties(fname=font_path).get_name()
                break

    if chosen is None:
        chosen = "DejaVu Sans"
        warnings.warn(
            "未在系统中找到常见中文字体，图形可能出现汉字缺失。可安装思源黑体或微软雅黑以获得更好的显示效果。",
            UserWarning,
            stacklevel=2,
        )

    plt.rcParams["font.family"] = [chosen, "sans-serif"]
    plt.rcParams["font.sans-serif"] = [chosen] + FONT_CANDIDATES + ["sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False


configure_plot_fonts()

warnings.filterwarnings("ignore", message="Glyph .* missing")

RunoffUnit = Literal["depth", "areal_depth", "discharge"]

UNIT_DEPTH: RunoffUnit = "depth"
UNIT_AREAL_DEPTH: RunoffUnit = "areal_depth"
UNIT_DISCHARGE: RunoffUnit = "discharge"

RUNOFF_LIBRARY: Dict[str, Dict[str, object]] = {
    "distributed_green_ampt": {
        "label": "DistributedGreenAmpt",
        "label_cn": "分布式Green-Ampt",
        "parameters": {
            "saturated_conductivity": 12.0,
            "wetting_front_suction": 110.0,
            "initial_moisture": 0.22,
            "saturated_moisture": 0.4,
            "porosity": 0.45,
            "zones": 4,
            "zone_distribution": "clustered",
        },
        "unit": UNIT_DISCHARGE,
    },
    "hbv": {
        "label": "HBV",
        "label_cn": "HBV",
        "parameters": {
            "field_capacity": 140.0,
            "k0": 0.18,
            "k1": 0.06,
            "k2": 0.012,
            "percolation": 2.5,
        },
        "unit": UNIT_AREAL_DEPTH,
    },
    "hymod": {
        "label": "HYMOD",
        "label_cn": "HYMOD",
        "parameters": {
            "max_storage": 85.0,
            "beta": 1.2,
            "quickflow_ratio": 0.55,
            "quick_k": 0.45,
            "slow_k": 0.05,
        },
        "unit": UNIT_AREAL_DEPTH,
    },
    "linear_reservoir": {
        "label": "LinearReservoir",
        "label_cn": "线性水库",
        "parameters": {
            "recession": 0.94,
            "conversion": 0.7,
        },
        "unit": UNIT_AREAL_DEPTH,
    },
    "scs_curve_number": {
        "label": "SCS",
        "label_cn": "SCS曲线数",
        "parameters": {
            "curve_number": 78.0,
            "initial_abstraction_ratio": 0.08,
        },
        "unit": UNIT_AREAL_DEPTH,
    },
    "simple": {
        "label": "Simple",
        "label_cn": "简单直接",
        "parameters": {},
        "unit": UNIT_DEPTH,
    },
    "vic": {
        "label": "VIC",
        "label_cn": "VIC",
        "parameters": {
            "infiltration_shape": 1.0,
            "max_soil_moisture": 135.0,
            "baseflow_coefficient": 0.012,
            "recession": 0.92,
        },
        "unit": UNIT_AREAL_DEPTH,
    },
    "wetspa": {
        "label": "WETSPA",
        "label_cn": "WETSPA",
        "parameters": {
            "soil_storage_max": 210.0,
            "infiltration_coefficient": 0.62,
            "surface_runoff_coefficient": 0.32,
            "percolation_coefficient": 0.06,
            "baseflow_constant": 0.045,
        },
        "unit": UNIT_DEPTH,
    },
    "xin_an_jiang": {
        "label": "XinAnJiang",
        "label_cn": "新安江",
        "parameters": {
            "wm": 700.0,
            "b": 0.85,
            "imp": 0.0,
            "recession": 0.999,
            "initial_tension_water": 15.0,
            "initial_groundwater": 0.05,
        },
        "unit": UNIT_DEPTH,
    },
}


def _terminal_subbasin_id(subbasins: Iterable["Subbasin"]) -> str:
    for sub in subbasins:
        if getattr(sub, "downstream", None) is None:
            return sub.id
    raise ValueError("No terminal subbasin found in the delineation result.")


def _build_parameter_zones(
    subbasins: Iterable["Subbasin"],
    runoff_model_id: str,
    routing_selector: Callable[["Subbasin"], str],
) -> Tuple[List[ParameterZoneConfig], Dict[str, str]]:
    zones: List[ParameterZoneConfig] = []
    assignment: Dict[str, str] = {}
    for index, sub in enumerate(subbasins, start=1):
        routing_id = routing_selector(sub)
        assignment[sub.id] = routing_id
        zones.append(
            ParameterZoneConfig(
                id=f"zone_{index}",
                description=f"Parameters for {sub.id}",
                control_points=[sub.id],
                parameters={
                    "runoff_model": runoff_model_id,
                    "routing_model": routing_id,
                },
                explicit_subbasins=[sub.id],
            )
        )
    return zones, assignment


def _channel_routing_details(
    subbasins: Iterable[Subbasin],
    assignment_map: Dict[str, str],
    id_to_slug: Dict[str, str],
    routing_param_map: Dict[str, Dict[str, float]],
) -> List[Dict[str, float]]:
    """Build diagnostic entries for each channel-aware routed subbasin."""
    details: List[Dict[str, float]] = []
    if not assignment_map:
        return details
    sub_lookup = {sub.id: sub for sub in subbasins}
    for sub_id, routing_id in assignment_map.items():
        sub = sub_lookup.get(sub_id)
        if sub is None:
            continue
        slug = id_to_slug.get(routing_id, routing_id)
        params = routing_param_map.get(routing_id, {})
        if slug == "dynamic_wave":
            router = DynamicWaveRouting(params)
            dt, segments, dx, wave_celerity, diffusivity = router._resolve_channel_properties(sub)
            reach_length = float(sub.channel_length_m or (segments * dx))
            travel_time_h = (segments * dx) / (wave_celerity * 3600.0) if wave_celerity else None
            details.append(
                {
                    "subbasin": sub_id,
                    "routing": "dynamic_wave",
                    "length_m": reach_length,
                    "segments": segments,
                    "wave_celerity": wave_celerity,
                    "diffusivity": diffusivity,
                    "travel_time_h": travel_time_h,
                }
            )
        elif slug == "muskingum":
            router = MuskingumRouting(params)
            resolved = router.resolved_parameters(sub)
            details.append(
                {
                    "subbasin": sub_id,
                    "routing": "muskingum",
                    "length_m": float(sub.channel_length_m or 0.0),
                    "travel_time_h": resolved.get("travel_time"),
                    "weighting_factor": resolved.get("weighting_factor"),
                }
            )
    return details


ROUTING_LIBRARY: Dict[str, Dict[str, object]] = {
    "simple": {
        "label": "Simple",
        "label_cn": "简易汇流",
        "parameters": {},
    },
    "lag": {
        "label": "Lag",
        "label_cn": "延迟汇流",
        "parameters": {
            "lag_steps": 6,
        },
    },
    "muskingum": {
        "label": "Muskingum",
        "label_cn": "马斯京根",
        "parameters": {
            "travel_time": 10.0,
            "weighting_factor": 0.05,
        },
        "stage_curve": {
            "coefficient": 32.0,
            "exponent": 0.54,
            "base_level": 0.38,
            "offset": 0.05,
        },
    },
    "dynamic_wave": {
        "label": "DynamicWave",
        "label_cn": "动力波",
        "parameters": {
            "reach_length": 24.0,
            "wave_celerity": 1.35,
            "diffusivity": 0.18,
            "segments": 8,
            "substeps": 1,
            "auto_substeps": True,
            "max_substeps": 18,
        },
        "stage_curve": {
            "coefficient": 24.0,
            "exponent": 0.60,
            "base_level": 0.32,
            "offset": 0.04,
        },
    },
    "channel_aware": {
        "label": "ChannelAware",
        "label_cn": "分段汇流",
        "parameters": {
            "threshold_m": 1200.0,
            "dynamic_wave": {
                "segments": 8,
                "wave_celerity": 1.35,
                "diffusivity": 0.18,
                "substeps": 1,
                "auto_substeps": True,
                "max_substeps": 18,
            },
            "muskingum": {
                "weighting_factor": 0.10,
                "travel_time": 8.0,
            },
        },
        "stage_curve": {
            "coefficient": 26.0,
            "exponent": 0.57,
            "base_level": 0.34,
            "offset": 0.04,
        },
    },
}

def evaluate_dynamic_wave_stability(parameters: Dict[str, float]) -> Dict[str, float | int | bool]:
    reach_length = float(parameters.get("reach_length", 10.0))
    segments = max(2, int(parameters.get("segments", 5)))
    dx = reach_length / segments if segments else float("inf")
    dt = float(parameters.get("time_step", 1.0))
    wave_celerity = float(parameters.get("wave_celerity", 2.0))
    diffusivity = float(parameters.get("diffusivity", 0.1))
    auto_substeps = bool(parameters.get("auto_substeps", True))
    max_substeps = max(1, int(parameters.get("max_substeps", 16)))

    base_courant = wave_celerity * dt / max(dx, 1e-6)
    base_diffusion = diffusivity * dt / max(dx**2, 1e-6)

    substeps = 1
    adjusted = False
    effective_courant = base_courant
    effective_diffusion = base_diffusion
    if auto_substeps and (base_courant > 1.0 or base_diffusion > 0.5):
        required = max(base_courant, base_diffusion / 0.5)
        substeps = max(1, min(max_substeps, math.ceil(required)))
        adjusted = substeps > 1
        effective_courant = base_courant / substeps
        effective_diffusion = base_diffusion / substeps

    return {
        "courant": base_courant,
        "diffusion": base_diffusion,
        "effective_courant": effective_courant,
        "effective_diffusion": effective_diffusion,
        "substeps": substeps,
        "adjusted": adjusted,
        "dx": dx,
    }


def generate_storm_forcing(
    total_hours: int = 240,
    storm_hours: int = 24,
    lead_hours: int = 72,
    tail_hours: int = 120,
    time_step_minutes: int = 60,
) -> pd.DataFrame:
    """构造一个包含充足前置和后置干旱期的合成暴雨过程。"""
    if lead_hours + storm_hours + tail_hours > total_hours:
        raise ValueError("The sum of lead, storm and tail hours must not exceed total duration.")

    steps_per_hour = max(1, 60 // time_step_minutes)
    total_steps = total_hours * steps_per_hour
    series = np.zeros(total_steps, dtype=float)

    event_steps = storm_hours * steps_per_hour
    start_step = lead_hours * steps_per_hour
    end_step = start_step + event_steps

    time_in_event = np.arange(event_steps)
    peak_time_fraction = 0.35
    peak_intensity = 22.0  # 峰值强度（毫米/小时）

    rising_mask = time_in_event <= event_steps * peak_time_fraction
    rising = peak_intensity * (time_in_event[rising_mask] / (event_steps * peak_time_fraction)) ** 2
    falling = peak_intensity * np.exp(
        -0.1 * (time_in_event[~rising_mask] - event_steps * peak_time_fraction)
    )
    event_profile = np.concatenate([rising, falling])[: event_steps]
    series[start_step:end_step] = event_profile

    timestamps = pd.date_range(start="2023-01-01", periods=total_steps, freq=f"{time_step_minutes}min")
    forcing_df = pd.DataFrame({"Timestamp": timestamps, PRECIP_COLUMN: series})
    forcing_df.set_index("Timestamp", inplace=True)
    return forcing_df


def calculate_rainfall_statistics(forcing_df: pd.DataFrame, area_km2: float) -> tuple[float, float, float]:
    if len(forcing_df) < 2:
        raise ValueError("Forcing series must contain at least two records.")

    dt_seconds = (forcing_df.index[1] - forcing_df.index[0]).total_seconds()
    dt_hours = dt_seconds / 3600.0
    rainfall_depth_mm = (forcing_df[PRECIP_COLUMN] * dt_hours).sum()
    rainfall_volume_m3 = rainfall_depth_mm / 1000.0 * area_km2 * 1_000_000
    return dt_seconds, rainfall_depth_mm, rainfall_volume_m3


def to_discharge_series(
    raw_series: pd.Series,
    unit: RunoffUnit,
    dt_seconds: float,
    area_km2: float,
) -> pd.Series:
    """将模型输出统一转换为瞬时流量（立方米/秒）。"""
    series = raw_series.astype(float)
    if unit == UNIT_DEPTH:
        return series * area_km2 * MM_TO_M3_PER_KM2 / 3600.0
    if unit == UNIT_AREAL_DEPTH:
        return series * MM_TO_M3_PER_KM2 / 3600.0
    if unit == UNIT_DISCHARGE:
        return series
    raise ValueError(f"Unsupported unit '{unit}' for discharge conversion.")


def to_stage_series(discharge: pd.Series, stage_curve: Dict[str, float] | None = None) -> pd.Series:
    """Convert discharge (m³/s) into an estimated water level (m) using a rating curve."""
    series = discharge.astype(float).clip(lower=0.0)
    curve = dict(DEFAULT_STAGE_CURVE)
    if stage_curve:
        curve.update({k: float(v) for k, v in stage_curve.items()})

    coefficient = max(1e-6, float(curve.get("coefficient", DEFAULT_STAGE_CURVE["coefficient"])))
    exponent = float(curve.get("exponent", DEFAULT_STAGE_CURVE["exponent"]))
    base_level = float(curve.get("base_level", DEFAULT_STAGE_CURVE.get("base_level", 0.0)))
    offset = float(curve.get("offset", DEFAULT_STAGE_CURVE.get("offset", 0.0)))

    discharge_np = series.to_numpy(dtype=float, copy=True)
    stage_component = np.zeros_like(discharge_np)
    positive_mask = discharge_np > 0.0
    if np.any(positive_mask):
        stage_component[positive_mask] = (discharge_np[positive_mask] / coefficient) ** exponent

    stages = base_level + stage_component + offset
    return pd.Series(stages, index=series.index, name="stage_m")


def collect_time_series(
    forcing_df: pd.DataFrame,
    results: Dict[str, object],
    dt_seconds: float,
    model_meta: Dict[str, Dict[str, object]],
    target_subbasin: str,
    basin_area_km2: float,
) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
    """Generate discharge and stage lookup tables for downstream plotting/reporting."""

    discharge_lookup: Dict[str, pd.Series] = {}
    stage_lookup: Dict[str, pd.Series] = {}

    for model_name, result in results.items():
        result_df = pd.DataFrame(result.baseline.aggregated, index=forcing_df.index)
        raw_series = result_df[target_subbasin].astype(float)
        unit = model_meta[model_name]["unit"]
        discharge_series = to_discharge_series(raw_series, unit, dt_seconds, basin_area_km2)
        stage_curve = model_meta.get(model_name, {}).get("stage_curve")
        stage_series = to_stage_series(discharge_series, stage_curve)
        discharge_lookup[model_name] = discharge_series
        stage_lookup[model_name] = stage_series

    return discharge_lookup, stage_lookup


def plot_comparison(
    forcing_df: pd.DataFrame,
    results: Dict[str, object],
    output_path: str,
    dt_seconds: float,
    model_meta: Dict[str, Dict[str, object]],
    target_subbasin: str,
    basin_area_km2: float,
    discharge_lookup: Dict[str, pd.Series] | None = None,
) -> None:
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.bar(
        forcing_df.index,
        forcing_df[PRECIP_COLUMN],
        width=0.04,
        color="#1f77b4",
        alpha=0.35,
        label="Precipitation (mm/hr)",
    )
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Precipitation (mm/hr)")
    ax1.invert_yaxis()
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)

    ax2 = ax1.twinx()
    channel_models: List[str] = []
    for model_name, result in results.items():
        discharge = None
        if discharge_lookup is not None:
            discharge = discharge_lookup.get(model_name)
        if discharge is None:
            result_df = pd.DataFrame(result.baseline.aggregated, index=forcing_df.index)
            raw_series = result_df[target_subbasin].astype(float)
            meta = model_meta[model_name]
            unit = meta["unit"]
            discharge = to_discharge_series(raw_series, unit, dt_seconds, basin_area_km2)
        meta = model_meta[model_name]
        is_channel = meta.get("routing_slug") == "channel_aware"
        linestyle = "--" if is_channel else "-"
        linewidth = 2.2 if is_channel else 1.4
        label_text = meta.get("label") or meta.get("label_cn") or model_name
        if is_channel:
            label_text = f"{label_text} [ChannelAware]"
            channel_models.append(label_text)
        ax2.plot(discharge.index, discharge.values, linestyle=linestyle, linewidth=linewidth, label=label_text)

    ax2.set_ylabel("Discharge (m³/s)")
    ax2.set_ylim(bottom=0)
    ax2.legend(loc="upper right", ncol=2, title="Runoff-Routing combinations")

    if channel_models:
        ax2.text(
            0.01,
            0.02,
            "Channel-aware selections shown dashed.",
            transform=ax2.transAxes,
            fontsize=10,
            color="#333333",
        )

    plt.title("Runoff routing comparison at outlet subbasin")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_stage_timeseries(
    forcing_df: pd.DataFrame,
    stage_lookup: Dict[str, pd.Series],
    output_path: str,
    model_meta: Dict[str, Dict[str, object]],
) -> None:
    if not stage_lookup:
        return

    fig, ax1 = plt.subplots(figsize=(14, 6))
    channel_models: List[str] = []
    for model_name, stage_series in stage_lookup.items():
        meta = model_meta.get(model_name, {})
        is_channel = meta.get("routing_slug") == "channel_aware"
        linestyle = "--" if is_channel else "-"
        linewidth = 2.2 if is_channel else 1.5
        label_text = meta.get("label") or meta.get("label_cn") or model_name
        if is_channel:
            label_text = f"{label_text} [ChannelAware]"
            channel_models.append(label_text)
        ax1.plot(stage_series.index, stage_series.values, linestyle=linestyle, linewidth=linewidth, label=label_text)

    ax1.set_ylabel("Stage (m)")
    ax1.set_xlabel("Time")
    ax1.set_ylim(bottom=0.0)
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    ax1.legend(loc="upper left", ncol=2, title="Runoff-Routing combinations")

    if channel_models:
        ax1.text(
            0.01,
            0.02,
            "Channel-aware selections shown dashed.",
            transform=ax1.transAxes,
            fontsize=10,
            color="#333333",
        )

    ax2 = ax1.twinx()
    ax2.bar(
        forcing_df.index,
        forcing_df[PRECIP_COLUMN],
        width=0.04,
        color="#1f77b4",
        alpha=0.3,
        label="Precipitation (mm/hr)",
    )
    ax2.set_ylabel("Precipitation (mm/hr)")
    ax2.invert_yaxis()

    plt.title("Water level comparison at outlet subbasin")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_metrics_summary(
    metrics_summary: Dict[str, Dict[str, float | str]],
    model_meta: Dict[str, Dict[str, object]],
    output_path: str,
) -> None:
    """Render peak-flow bars with runoff coefficients overlaid for quick screening."""
    if not metrics_summary:
        return

    model_names = list(metrics_summary.keys())
    labels = [model_meta.get(name, {}).get("label", name) for name in model_names]
    peak_flows = [float(metrics_summary[name].get("peak_flow", float("nan"))) for name in model_names]
    runoff_coeffs = [float(metrics_summary[name].get("runoff_coeff", float("nan"))) for name in model_names]

    positions = np.arange(len(model_names), dtype=float)
    fig, ax1 = plt.subplots(figsize=(14, 6))
    colors = plt.cm.Blues(np.linspace(0.45, 0.85, len(model_names))) if len(model_names) > 1 else ["#4F81BD"]
    bars = ax1.bar(positions, peak_flows, color=colors, alpha=0.85)
    ax1.set_ylabel("Peak flow (m³/s)")
    ax1.set_xticks(positions)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.set_title("Peak flows and runoff coefficients by model")
    ax1.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)

    ax2 = ax1.twinx()
    coeff_line = ax2.plot(
        positions,
        runoff_coeffs,
        color="#E15759",
        marker="o",
        linewidth=2.0,
        label="Runoff coefficient",
    )
    ax2.set_ylabel("Runoff coefficient (-)")
    ax2.set_ylim(bottom=0.0)

    for idx, coeff in enumerate(runoff_coeffs):
        if not np.isnan(coeff):
            ax2.text(
                positions[idx],
                coeff + 0.02,
                f"{coeff:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#E15759",
            )

    handles = [bars[0], coeff_line[0]] if bars and coeff_line else []
    labels_for_legend = ["Peak flow", "Runoff coefficient"] if handles else []
    if handles:
        ax1.legend(handles, labels_for_legend, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def prepare_metrics(
    forcing_df: pd.DataFrame,
    results: Dict[str, object],
    dt_seconds: float,
    rainfall_volume_m3: float,
    model_meta: Dict[str, Dict[str, object]],
    target_subbasin: str,
    basin_area_km2: float,
    discharge_lookup: Dict[str, pd.Series] | None = None,
    stage_lookup: Dict[str, pd.Series] | None = None,
) -> Dict[str, Dict[str, float | str]]:
    metrics_summary: Dict[str, Dict[str, float | str]] = {}
    for model_name, result in results.items():
        label_text = model_meta.get(model_name, {}).get("label", model_name)
        discharge_series = None
        if discharge_lookup is not None:
            discharge_series = discharge_lookup.get(model_name)
        if discharge_series is None:
            result_df = pd.DataFrame(result.baseline.aggregated, index=forcing_df.index)
            raw_series = result_df[target_subbasin].astype(float)
            unit = model_meta[model_name]["unit"]
            discharge_series = to_discharge_series(raw_series, unit, dt_seconds, basin_area_km2)
        stage_series = None
        if stage_lookup is not None:
            stage_series = stage_lookup.get(model_name)
        if stage_series is None:
            stage_curve = model_meta.get(model_name, {}).get("stage_curve")
            stage_series = to_stage_series(discharge_series, stage_curve)

        peak_flow = float(discharge_series.max())
        time_to_peak = discharge_series.idxmax()
        runoff_volume_m3 = float((discharge_series * dt_seconds).sum())
        runoff_coeff = runoff_volume_m3 / rainfall_volume_m3 if rainfall_volume_m3 else float("nan")
        balance_error_m3 = rainfall_volume_m3 - runoff_volume_m3
        balance_ratio = balance_error_m3 / rainfall_volume_m3 if rainfall_volume_m3 else float("nan")
        negative_flows = int((discharge_series < -1e-8).sum())
        peak_stage = float(stage_series.max())
        stage_time_to_peak = stage_series.idxmax()

        notes: List[str] = []
        if negative_flows:
            notes.append(f"{label_text} produced {negative_flows} negative discharge timesteps; check model parameters.")
        if not np.isnan(balance_ratio) and balance_error_m3 < -0.05 * rainfall_volume_m3:
            notes.append(
                f"{label_text} runoff volume exceeds rainfall by {abs(balance_ratio):.1%}; review rainfall inputs, area, or groundwater settings."
            )
        if model_meta[model_name]["runoff_slug"] == "simple" and not np.isnan(runoff_coeff):
            if abs(runoff_coeff - 1.0) > 0.02:
                notes.append(
                    f"{label_text} should match rainfall volume; current runoff coefficient is {runoff_coeff:.2f}."
                )

        time_to_peak_str = (
            time_to_peak.strftime("%Y-%m-%d %H:%M") if hasattr(time_to_peak, "strftime") else str(time_to_peak)
        )
        stage_time_to_peak_str = (
            stage_time_to_peak.strftime("%Y-%m-%d %H:%M")
            if hasattr(stage_time_to_peak, "strftime")
            else str(stage_time_to_peak)
        )

        summary_entry: Dict[str, float | str | List[Dict[str, float]]] = {
            "peak_flow": peak_flow,
            "time_to_peak": time_to_peak_str,
            "runoff_volume_m3": runoff_volume_m3,
            "runoff_coeff": runoff_coeff,
            "balance_error_m3": balance_error_m3,
            "balance_ratio": balance_ratio,
            "negative_flows": negative_flows,
            "notes": notes,
            "courant": None,
            "diffusion": None,
            "substeps": None,
            "channel_details": model_meta[model_name].get("channel_details"),
            "peak_stage_m": peak_stage,
            "stage_time_to_peak": stage_time_to_peak_str,
        }

        stability = model_meta[model_name].get("stability")
        if stability:
            effective_courant = stability.get("effective_courant", float("nan"))
            effective_diffusion = stability.get("effective_diffusion", float("nan"))
            substeps = stability.get("substeps")
            summary_entry["courant"] = effective_courant
            summary_entry["diffusion"] = effective_diffusion
            summary_entry["substeps"] = substeps

            if stability.get("adjusted"):
                notes.append(
                    f"{model_name} dynamic wave solver subdivided into {substeps} sub-steps for stability."
                )
            if (
                effective_courant is not None
                and effective_diffusion is not None
                and (effective_courant > 1.0 or effective_diffusion > 0.5)
            ):
                notes.append(
                    f"{label_text} dynamic wave still exceeds stability limits (C={effective_courant:.2f}, D={effective_diffusion:.2f})."
                )

        channel_details = summary_entry.get("channel_details") or []
        if channel_details and runoff_coeff > 1.05:
            for detail in channel_details:
                if detail["routing"] == "dynamic_wave":
                    notes.append(
                        f"{label_text} dynamic-wave reach {detail['subbasin']}: length {detail['length_m']:.0f} m, celerity {detail['wave_celerity']:.2f} m/s, travel time ~{(detail['travel_time_h'] or 0):.2f} h."
                    )
                elif detail["routing"] == "muskingum":
                    notes.append(
                        f"{label_text} Muskingum reach {detail['subbasin']}: estimated travel time {detail['travel_time_h']:.2f} h (X={detail.get('weighting_factor', 0.0):.2f})."
                    )

        metrics_summary[model_name] = summary_entry

    return metrics_summary

def generate_report(
    plot_path: str,
    output_dir: str,
    rainfall_depth_mm: float,
    rainfall_volume_m3: float,
    basin_area_km2: float,
    metrics_summary: Dict[str, Dict[str, float | str]],
    metrics_plot_path: str | None = None,
    stage_plot_path: str | None = None,
    stage_csv_path: str | None = None,
) -> None:
    report = MarkdownReportBuilder(title="Multi-model storm comparison")
    report.add_heading("Hydrograph comparison", level=2)
    report.add_image(os.path.relpath(plot_path, output_dir), "hydrograph_comparison.png")

    if stage_plot_path:
        report.add_heading("Water level comparison", level=2)
        report.add_image(os.path.relpath(stage_plot_path, output_dir), os.path.basename(stage_plot_path))
        if stage_csv_path:
            report.add_paragraph(
                f"Estimated water level series exported to `{os.path.basename(stage_csv_path)}`."
            )

    rainfall_volume_million = rainfall_volume_m3 / 1e6
    report.add_heading("Rainfall summary", level=2)
    report.add_list(
        [
            f"Total rainfall depth: {rainfall_depth_mm:.1f} mm",
            f"Catchment area: {basin_area_km2:.1f} km^2",
            f"Rainfall volume: {rainfall_volume_million:.2f} x10^6 m^3",
        ]
    )

    report.add_heading("Key hydrologic metrics", level=2)
    rows: List[List[str]] = []
    issues: List[str] = []
    for model_name, metrics in metrics_summary.items():
        runoff_volume_million = metrics["runoff_volume_m3"] / 1e6
        runoff_coeff = metrics["runoff_coeff"]
        balance_error_million = metrics["balance_error_m3"] / 1e6
        courant = metrics.get("courant")
        diffusion = metrics.get("diffusion")
        substeps = metrics.get("substeps")
        peak_stage = metrics.get("peak_stage_m")
        stage_time_to_peak = metrics.get("stage_time_to_peak", "-")
        rows.append(
            [
                model_name,
                f"{metrics['peak_flow']:.2f}",
                metrics["time_to_peak"],
                f"{runoff_volume_million:.2f}",
                f"{balance_error_million:.2f}",
                f"{runoff_coeff:.2f}",
                f"{courant:.2f}" if courant is not None else "-",
                f"{diffusion:.2f}" if diffusion is not None else "-",
                str(substeps) if substeps is not None else "-",
                f"{peak_stage:.2f}" if peak_stage is not None else "-",
                stage_time_to_peak,
            ]
        )
        issues.extend(metrics.get("notes", []))

    if metrics_plot_path:
        report.add_heading("Peak flow and runoff summary", level=2)
        report.add_image(os.path.relpath(metrics_plot_path, output_dir), os.path.basename(metrics_plot_path))

    report.add_table(
        TableData(
            headers=[
                "Model",
                "Peak flow (m3/s)",
                "Time to peak",
                "Runoff volume (x10^6 m3)",
                "Water balance error (x10^6 m3)",
                "Runoff coefficient",
                "Courant",
                "Diffusion",
                "Sub-steps",
                "Peak stage (m)",
                "Stage time to peak",
            ],
            rows=rows,
        )
    )

    channel_rows: List[List[str]] = []
    for model_name, metrics in metrics_summary.items():
        details = metrics.get("channel_details") or []
        for detail in details:
            routing = detail.get("routing", "")
            length = detail.get("length_m")
            travel_time = detail.get("travel_time_h")
            if routing == "dynamic_wave":
                note = f"celerity {detail.get('wave_celerity', float('nan')):.2f} m/s"
            elif routing == "muskingum":
                note = f"X={detail.get('weighting_factor', float('nan')):.2f}"
            else:
                note = "-"
            channel_rows.append(
                [
                    model_name,
                    detail.get("subbasin", "-"),
                    routing,
                    f"{length:.0f}" if length is not None else "-",
                    f"{travel_time:.2f}" if travel_time is not None else "-",
                    note,
                ]
            )

    if channel_rows:
        report.add_heading("Channel-aware routing breakdown", level=2)
        report.add_table(
            TableData(
                headers=[
                    "Model",
                    "Subbasin",
                    "Routing",
                    "Reach length (m)",
                    "Travel time (h)",
                    "Notes",
                ],
                rows=channel_rows,
            )
        )

    report.add_heading("Diagnostic summary", level=2)
    if issues:
        report.add_list(issues)
    else:
        report.add_paragraph("All model configurations passed the coarse checks (mass balance and sign tests).")

    report.add_heading("Suggested actions", level=2)
    report.add_list(
        [
            "Include a quick mass-balance review when screening new scenarios.",
            "If runoff coefficients exceed 1.0, review rainfall inputs, catchment area configuration, or groundwater settings.",
            "If runoff coefficients are very low, revisit infiltration/storage parameters or soil moisture assumptions.",
        ]
    )

    report_path = os.path.join(output_dir, "multi_model_report.md")
    report.write(report_path)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare runoff/routing configurations and demonstrate channel-aware switching.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/multi_model_storm_comparison"),
        help="Output directory for synthetic delineation, forcing and reports.",
    )
    parser.add_argument(
        "--accumulation-threshold",
        type=float,
        default=2.0,
        help="Minimum accumulation threshold when delineating from the JSON DEM.",
    )
    parser.add_argument(
        "--channel-threshold",
        type=float,
        default=None,
        help="Flow accumulation threshold used when extracting channels (auto-tuned if omitted).",
    )
    parser.add_argument(
        "--dynamic-wave-length",
        type=float,
        default=1200.0,
        help="Enable dynamic-wave routing for reaches longer than this length (meters).",
    )
    args = parser.parse_args()

    output_dir: Path = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    storm_forcing = generate_storm_forcing()
    forcing_path = output_dir / "storm_forcing.csv"
    storm_forcing.to_csv(forcing_path)

    synthetic_dir = output_dir / "synthetic_inputs"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    dem_path, pour_points_path = write_synthetic_delineation_inputs(synthetic_dir)

    delineation_config = DelineationConfig(
        dem_path=dem_path,
        pour_points_path=pour_points_path,
        accumulation_threshold=float(args.accumulation_threshold),
        channel_threshold=(None if args.channel_threshold is None else float(args.channel_threshold)),
    )

    subbasins = delineation_config.to_subbasins()
    basin_area_km2 = sum(sub.area_km2 for sub in subbasins) or BASIN_AREA_KM2
    terminal_subbasin = _terminal_subbasin_id(subbasins)
    sub_lookup = {sub.id: sub for sub in subbasins}

    dt_seconds, rainfall_depth_mm, rainfall_volume_m3 = calculate_rainfall_statistics(
        storm_forcing, basin_area_km2
    )
    dt_hours = dt_seconds / 3600.0

    results: Dict[str, object] = {}
    model_meta: Dict[str, Dict[str, object]] = {}
    forcing_series = storm_forcing[PRECIP_COLUMN].tolist()

    for runoff_slug, runoff_info in RUNOFF_LIBRARY.items():
        runoff_label = str(runoff_info.get("label", runoff_slug))
        runoff_label_cn = str(runoff_info.get("label_cn", runoff_label))
        for routing_slug, routing_info in ROUTING_LIBRARY.items():
            runoff_parameters = dict(runoff_info.get("parameters", {}))
            routing_label = str(routing_info.get("label", routing_slug))
            routing_label_cn = str(routing_info.get("label_cn", routing_label))
            label = f"{runoff_label_cn}-{routing_label_cn}"

            runoff_config = RunoffModelConfig(
                id=f"{runoff_slug}_{routing_slug}_runoff",
                model_type=runoff_slug,
                parameters=runoff_parameters,
            )

            routing_parameters = copy.deepcopy(routing_info.get("parameters", {}))
            routing_configs: List[RoutingModelConfig] = []
            stability_info = None
            id_to_slug: Dict[str, str] = {}
            routing_param_map: Dict[str, Dict[str, float]] = {}

            if routing_slug == "channel_aware":
                threshold = float(args.dynamic_wave_length)
                dynamic_params = copy.deepcopy(routing_parameters.get("dynamic_wave", {}))
                dynamic_params["time_step"] = dt_hours
                muskingum_params = copy.deepcopy(routing_parameters.get("muskingum", {}))
                muskingum_params["time_step"] = dt_hours

                dynamic_config = RoutingModelConfig(
                    id=f"{runoff_slug}_{routing_slug}_dynamic",
                    model_type="dynamic_wave",
                    parameters=dynamic_params,
                )
                muskingum_config = RoutingModelConfig(
                    id=f"{runoff_slug}_{routing_slug}_muskingum",
                    model_type="muskingum",
                    parameters=muskingum_params,
                )
                routing_configs = [dynamic_config, muskingum_config]
                id_to_slug = {
                    dynamic_config.id: "dynamic_wave",
                    muskingum_config.id: "muskingum",
                }
                routing_param_map = {
                    dynamic_config.id: dynamic_params,
                    muskingum_config.id: muskingum_params,
                }

                def select_routing(sub: Subbasin, _threshold: float = threshold) -> str:
                    length = float(getattr(sub, "channel_length_m", 0.0) or 0.0)
                    return dynamic_config.id if length >= _threshold else muskingum_config.id

            else:
                if routing_slug in {"muskingum", "dynamic_wave"}:
                    routing_parameters["time_step"] = dt_hours
                routing_config = RoutingModelConfig(
                    id=f"{runoff_slug}_{routing_slug}",
                    model_type=routing_slug,
                    parameters=routing_parameters,
                )
                routing_configs = [routing_config]
                id_to_slug = {routing_config.id: routing_slug}
                routing_param_map = {routing_config.id: routing_parameters}
                if routing_slug == "dynamic_wave":
                    stability_info = evaluate_dynamic_wave_stability(routing_parameters)

                def select_routing(sub: Subbasin, routing_id: str = routing_config.id) -> str:
                    return routing_id

            parameter_zones, assignment_map = _build_parameter_zones(
                subbasins,
                runoff_config.id,
                select_routing,
            )
            assignment_labels = {
                sub_id: id_to_slug.get(routing_id, routing_id)
                for sub_id, routing_id in assignment_map.items()
            }
            channel_details = []
            if routing_slug == "channel_aware":
                channel_details = _channel_routing_details(
                    subbasins,
                    assignment_map,
                    id_to_slug,
                    routing_param_map,
                )

            config = ModelConfig(
                delineation=delineation_config,
                runoff_models=[runoff_config],
                routing_models=routing_configs,
                parameter_zones=parameter_zones,
                io=IOConfig(precipitation=forcing_path),
                scenarios=[],
                evaluation=None,
            )

            forcing_data = {sub.id: list(forcing_series) for sub in subbasins}
            result = run_workflow(config, forcing_data)
            results[label] = result
            model_meta[label] = {
                "unit": runoff_info["unit"],
                "runoff_slug": runoff_slug,
                "routing_slug": routing_slug,
                "label_cn": label,
                "label": f"{runoff_label}-{routing_label}",
                "stability": stability_info,
                "channel_assignment": assignment_labels,
                "channel_details": channel_details,
                "target_subbasin": terminal_subbasin,
                "stage_curve": routing_parameters.get("stage_curve", routing_info.get("stage_curve")),
            }

    discharge_lookup, stage_lookup = collect_time_series(
        storm_forcing,
        results,
        dt_seconds,
        model_meta,
        terminal_subbasin,
        basin_area_km2,
    )

    stage_df = pd.DataFrame(stage_lookup)
    stage_df.index.name = "Timestamp"
    stage_csv_path = output_dir / STAGE_TIMESERIES_FILENAME
    stage_df.to_csv(stage_csv_path)

    metrics_summary = prepare_metrics(
        storm_forcing,
        results,
        dt_seconds,
        rainfall_volume_m3,
        model_meta,
        terminal_subbasin,
        basin_area_km2,
        discharge_lookup=discharge_lookup,
        stage_lookup=stage_lookup,
    )

    plot_path = output_dir / "hydrograph_comparison.png"
    plot_comparison(
        storm_forcing,
        results,
        str(plot_path),
        dt_seconds,
        model_meta,
        terminal_subbasin,
        basin_area_km2,
        discharge_lookup=discharge_lookup,
    )

    stage_plot_path = output_dir / STAGE_PLOT_FILENAME
    plot_stage_timeseries(storm_forcing, stage_lookup, str(stage_plot_path), model_meta)

    metrics_plot_path = output_dir / "metric_summary.png"
    plot_metrics_summary(metrics_summary, model_meta, str(metrics_plot_path))

    generate_report(
        str(plot_path),
        str(output_dir),
        rainfall_depth_mm,
        rainfall_volume_m3,
        basin_area_km2,
        metrics_summary,
        str(metrics_plot_path),
        stage_plot_path=str(stage_plot_path),
        stage_csv_path=str(stage_csv_path),
    )
    print(f"\nTotal rainfall volume: {rainfall_volume_m3 / 1e6:.2f} x10^6 m^3")
    print(f"Total contributing area: {basin_area_km2:.1f} km^2; outlet subbasin: {terminal_subbasin}")
    for name, metrics in metrics_summary.items():
        label_text = model_meta.get(name, {}).get("label", name)
        print(f"  - {label_text}: runoff coefficient {metrics['runoff_coeff']:.2f}")
    print(f"Water level figure: {stage_plot_path.name}")
    print(f"Water level timeseries: {stage_csv_path.name}")
    channel_configs = {
        name: meta
        for name, meta in model_meta.items()
        if meta.get("channel_assignment") and meta["routing_slug"] == "channel_aware"
    }
    if channel_configs:
        print("\nChannel-aware routing selections:")
        for name, meta in channel_configs.items():
            assignment = meta.get("channel_assignment", {})
            dynamic_segments = [sid for sid, slug in assignment.items() if slug == "dynamic_wave"]
            if dynamic_segments:
                segs = ", ".join(dynamic_segments)
                label_text = model_meta.get(name, {}).get("label", name)
                print(f"  - {label_text} dynamic-wave reaches: {segs}")
    diagnostic_notes: List[str] = []
    for metrics in metrics_summary.values():
        for note in metrics.get("notes", []):
            if note not in diagnostic_notes:
                diagnostic_notes.append(note)
    if diagnostic_notes:
        print("\nDiagnostic notes:")
        for note in diagnostic_notes:
            print(f"  - {note}")
    else:
        print("\nDiagnostic notes: all combinations passed water balance and sign checks.")
    print(f"Results written to {output_dir / 'multi_model_report.md'}")
if __name__ == "__main__":
    main()


