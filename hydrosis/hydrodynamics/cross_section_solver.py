"""基于 Step04 断面几何的简化水动力求解工具。

该模块将 `ZoneGeometry` 生成的断面采样转化为曼宁公式
的流量-水深评级曲线，并据此把 Step09 的分段流量时序
转换为水位、流速等水动力指标。求解假定能量坡度≈床坡，
适合用于快速对比水文学与断面驱动的结果。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from .zone_geometry import ZoneGeometry

GRAVITY = 9.80665  # m/s^2


def _as_array(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=float)


@dataclass(slots=True)
class RatingCurve:
    """单个断面的流量-水深评级曲线。"""

    zone_id: str
    segment_id: str
    station_global_m: float
    station_local_m: float
    chainage_m: float
    bed_elevation_m: float
    slope: float
    mannings_n: float
    depths: np.ndarray
    discharges: np.ndarray
    areas: np.ndarray
    wetted_perimeters: np.ndarray
    top_widths: np.ndarray

    def _interp(self, series: np.ndarray, discharge: float) -> float:
        """对指定序列按流量插值，并在外推时使用末段线性估计。"""
        if series.size == 0:
            return float("nan")
        q = max(discharge, 0.0)
        if q <= self.discharges[0]:
            return float(series[0])
        if q >= self.discharges[-1]:
            if series.size == 1:
                return float(series[0])
            q1, q0 = self.discharges[-1], self.discharges[-2]
            s1, s0 = series[-1], series[-2]
            gradient = (s1 - s0) / max(q1 - q0, 1e-6)
            return float(s1 + gradient * (q - q1))
        return float(np.interp(q, self.discharges, series))

    def depth_from_discharge(self, discharge: float) -> float:
        return self._interp(self.depths, discharge)

    def area_from_discharge(self, discharge: float) -> float:
        return max(self._interp(self.areas, discharge), 1e-6)

    def wetted_perimeter_from_discharge(self, discharge: float) -> float:
        return max(self._interp(self.wetted_perimeters, discharge), 1e-6)

    def top_width_from_discharge(self, discharge: float) -> float:
        return max(self._interp(self.top_widths, discharge), 1e-6)

    def evaluate(self, discharge: float) -> Mapping[str, float]:
        """给定流量 (m³/s)，返回水深、断面面积等派生量。"""

        q = max(float(discharge), 0.0)
        depth = self.depth_from_discharge(q)
        area = self.area_from_discharge(q)
        wetted = self.wetted_perimeter_from_discharge(q)
        top_width = self.top_width_from_discharge(q)

        stage = self.bed_elevation_m + depth
        velocity = q / area if area > 1e-6 else 0.0
        hydraulic_radius = area / wetted if wetted > 0.0 else 0.0
        hydraulic_depth = area / top_width if top_width > 0.0 else depth
        froude = (
            velocity / np.sqrt(GRAVITY * hydraulic_depth)
            if hydraulic_depth > 1e-6
            else 0.0
        )

        return {
            "zone_id": self.zone_id,
            "segment_id": self.segment_id,
            "station_global_m": self.station_global_m,
            "station_local_m": self.station_local_m,
            "chainage_m": self.chainage_m,
            "bed_elevation_m": self.bed_elevation_m,
            "slope": self.slope,
            "mannings_n": self.mannings_n,
            "discharge_m3s": q,
            "depth_m": depth,
            "stage_m": stage,
            "area_m2": area,
            "wetted_perimeter_m": wetted,
            "top_width_m": top_width,
            "hydraulic_radius_m": hydraulic_radius,
            "hydraulic_depth_m": hydraulic_depth,
            "velocity_mps": velocity,
            "froude_number": froude,
        }


class CrossSectionSolver:
    """利用 Step04 断面数据和曼宁公式的简化 1D 求解器。"""

    def __init__(
        self,
        ratings: Dict[str, RatingCurve],
        *,
        zone_id: str,
        metadata: Mapping[str, float],
    ) -> None:
        if not ratings:
            raise ValueError("缺少断面评级曲线，无法初始化求解器。")
        self.zone_id = zone_id
        self.ratings = ratings
        self.metadata = dict(metadata)

    @classmethod
    def from_zone_geometry(
        cls,
        geometry: ZoneGeometry,
        *,
        segments: Optional[Iterable[str]] = None,
        slope_floor: float = 1e-6,
        slope_cap: Optional[float] = None,
    ) -> "CrossSectionSolver":
        """根据 ZoneGeometry 构建求解器。

        参数:
            geometry: `build_zone_geometry` 的输出。
            segments: 可选，指定需要构建的 segment_id 子集。
            slope_floor: 当中心线坡度过小时的最小取值，防止评级曲线退化。
        """

        cross_df = geometry.cross_section_table.copy()
        if cross_df.empty:
            raise ValueError("cross_section_table 为空，无法构建评级曲线。")

        if segments is not None:
            segment_set = set(segments)
            cross_df = cross_df[cross_df["segment_id"].isin(segment_set)]
            if cross_df.empty:
                raise ValueError("所选 segment_id 无对应断面数据。")

        slopes = np.abs(cross_df["centerline_bed_slope"].to_numpy(dtype=float))
        slopes = np.maximum(slopes, slope_floor)
        if slope_cap is not None:
            slopes = np.minimum(slopes, slope_cap)
        cross_df["slope_for_rating"] = slopes
        cross_df["discharge_m3s"] = (
            (1.0 / cross_df["mannings_n"])
            * cross_df["area_m2"]
            * np.power(cross_df["hydraulic_radius_m"], 2.0 / 3.0)
            * np.sqrt(cross_df["slope_for_rating"])
        )

        centerline = geometry.centerline[
            [
                "segment_id",
                "global_station_m",
                "chainage_m",
                "bed_elevation_m",
                "centerline_bed_slope",
            ]
        ].drop_duplicates()

        ratings: Dict[str, RatingCurve] = {}

        for segment_id, seg_group in cross_df.groupby("segment_id"):
            # 选取该段最下游的 global station（对应 Step09 的段尾流量）
            target_station = seg_group["station_global_m"].max()
            station_df = seg_group[
                seg_group["station_global_m"] == target_station
            ].copy()
            station_df.sort_values("depth_m", inplace=True)

            center_subset = centerline[
                (centerline["segment_id"] == segment_id)
                & (
                    np.isclose(
                        centerline["global_station_m"],
                        target_station,
                        atol=1e-6,
                    )
                )
            ]
            if center_subset.empty:
                chainage = float(station_df["station_global_m"].iloc[0])
                bed_elevation = float(station_df["centerline_bed_elevation_m"].iloc[0])
            else:
                chainage = float(center_subset["chainage_m"].iloc[0])
                bed_elevation = float(center_subset["bed_elevation_m"].iloc[0])

            curve = RatingCurve(
                zone_id=geometry.zone_id,
                segment_id=segment_id,
                station_global_m=float(target_station),
                station_local_m=float(station_df["station_local_m"].iloc[0]),
                chainage_m=chainage,
                bed_elevation_m=bed_elevation,
                slope=float(station_df["slope_for_rating"].iloc[0]),
                mannings_n=float(station_df["mannings_n"].iloc[0]),
                depths=_as_array(station_df["depth_m"]),
                discharges=_as_array(station_df["discharge_m3s"]),
                areas=_as_array(station_df["area_m2"]),
                wetted_perimeters=_as_array(station_df["wetted_perimeter_m"]),
                top_widths=_as_array(station_df["top_width_m"]),
            )
            ratings[segment_id] = curve

        return cls(ratings=ratings, zone_id=geometry.zone_id, metadata=geometry.metadata)

    def available_segments(self) -> Iterable[str]:
        return self.ratings.keys()

    def summary(self) -> pd.DataFrame:
        rows = [
            {
                "zone_id": curve.zone_id,
                "segment_id": seg,
                "station_global_m": curve.station_global_m,
                "chainage_m": curve.chainage_m,
                "bed_elevation_m": curve.bed_elevation_m,
                "slope": curve.slope,
                "mannings_n": curve.mannings_n,
                "max_depth_m": float(curve.depths[-1]),
                "max_discharge_m3s": float(curve.discharges[-1]),
            }
            for seg, curve in self.ratings.items()
        ]
        return pd.DataFrame(rows).sort_values("chainage_m").reset_index(drop=True)

    def evaluate_timeseries(
        self,
        flow_df: pd.DataFrame,
        *,
        time_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """将各段流量时序转换为水动力指标。

        参数:
            flow_df: 列为 segment_id，行为时间的 DataFrame。
            time_column: 可选，若 flow_df 已含时间列，可指定其列名；
                         否则将使用索引名（若为空则自动命名为 'time'）。
        返回:
            长表格 DataFrame，包含 time、segment_id、stage、velocity 等字段。
        """

        data = flow_df.copy()

        if time_column:
            if time_column not in data.columns:
                raise KeyError(f"{time_column} 不存在于流量表。")
        else:
            if data.index.name is None:
                data = data.reset_index(drop=False).rename(columns={"index": "time"})
                time_column = "time"
            else:
                time_column = data.index.name
                data = data.reset_index()

        value_columns = [c for c in data.columns if c != time_column]
        if not value_columns:
            raise ValueError("流量表缺少 segment_id 列。")

        melted = data.melt(
            id_vars=time_column, var_name="segment_id", value_name="discharge_m3s"
        )
        melted = melted.dropna(subset=["discharge_m3s"])

        results = []
        for row in melted.itertuples(index=False):
            segment_id = row.segment_id
            if segment_id not in self.ratings:
                continue  # 无对应断面时跳过
            rating = self.ratings[segment_id]
            evaluation = rating.evaluate(float(row.discharge_m3s))
            evaluation[time_column] = getattr(row, time_column)
            results.append(evaluation)

        if not results:
            return pd.DataFrame(
                columns=[
                    time_column,
                    "zone_id",
                    "segment_id",
                    "discharge_m3s",
                    "stage_m",
                    "depth_m",
                    "velocity_mps",
                ]
            )

        result_df = pd.DataFrame(results)
        result_df.rename(columns={time_column: "time"}, inplace=True)
        return result_df.sort_values(["time", "chainage_m"]).reset_index(drop=True)

    def summarize_timeseries(self, result_df: pd.DataFrame) -> pd.DataFrame:
        """对求解结果进行段级统计，输出极值及对应时间。"""

        if result_df.empty:
            return pd.DataFrame(
                columns=[
                    "zone_id",
                    "segment_id",
                    "station_global_m",
                    "chainage_m",
                    "bed_elevation_m",
                    "max_stage_m",
                    "time_of_max_stage",
                    "max_depth_m",
                    "time_of_max_depth",
                    "max_velocity_mps",
                    "time_of_max_velocity",
                    "max_discharge_m3s",
                ]
            )

        groups = result_df.groupby("segment_id", as_index=False)
        summaries = []
        for segment_id, group in groups:
            idx_stage = group["stage_m"].idxmax()
            idx_depth = group["depth_m"].idxmax()
            idx_velocity = group["velocity_mps"].idxmax()
            idx_discharge = group["discharge_m3s"].idxmax()

            representative = group.iloc[0]
            summaries.append(
                {
                    "zone_id": representative["zone_id"],
                    "segment_id": segment_id,
                    "station_global_m": representative["station_global_m"],
                    "chainage_m": representative["chainage_m"],
                    "bed_elevation_m": representative["bed_elevation_m"],
                    "max_stage_m": group.loc[idx_stage, "stage_m"],
                    "time_of_max_stage": group.loc[idx_stage, "time"],
                    "max_depth_m": group.loc[idx_depth, "depth_m"],
                    "time_of_max_depth": group.loc[idx_depth, "time"],
                    "max_velocity_mps": group.loc[idx_velocity, "velocity_mps"],
                    "time_of_max_velocity": group.loc[idx_velocity, "time"],
                    "max_discharge_m3s": group.loc[idx_discharge, "discharge_m3s"],
                }
            )

        return pd.DataFrame(summaries).sort_values("chainage_m").reset_index(drop=True)
