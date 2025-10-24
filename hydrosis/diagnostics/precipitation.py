"""降雨空间分布诊断

诊断流域降雨的空间分布问题，识别降雨异常分区和数据质量问题。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .base import BaseDiagnostic, DiagnosticResult, IssueSeverity


class PrecipitationDiagnostic(BaseDiagnostic):
    """降雨空间分布诊断器

    检查流域降雨的空间分布，识别以下问题：
    1. 分区间降雨差异过大
    2. 子流域降雨异常
    3. 降雨数据质量问题
    4. 雨量站覆盖不足

    Parameters
    ----------
    output_dir : Path, optional
        输出目录
    verbose : bool, default=True
        是否输出详细信息
    anomaly_threshold : float, default=0.3
        异常阈值（相对差异），超过此值认为存在异常
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        verbose: bool = True,
        anomaly_threshold: float = 0.3
    ):
        super().__init__(output_dir, verbose)
        self.anomaly_threshold = anomaly_threshold

        # 数据
        self._precipitation_df: Optional[pd.DataFrame] = None
        self._subbasins_df: Optional[pd.DataFrame] = None
        self._zone_stats: Dict[int, Dict] = {}

    @property
    def diagnostic_name(self) -> str:
        return "降雨空间分布诊断"

    def load_data(
        self,
        precipitation_df: pd.DataFrame,
        subbasins_df: pd.DataFrame
    ) -> None:
        """加载降雨和子流域数据

        Parameters
        ----------
        precipitation_df : pd.DataFrame
            降雨时间序列，列为子流域ID，值为降雨强度 (mm/h)
        subbasins_df : pd.DataFrame
            子流域信息，必须包含列: subzone_id, zone_id, area_km2
        """
        required_cols = {'subzone_id', 'zone_id', 'area_km2'}
        if not required_cols.issubset(subbasins_df.columns):
            raise ValueError(
                f"subbasins_df 缺少必需列: {required_cols - set(subbasins_df.columns)}"
            )

        self._precipitation_df = precipitation_df
        self._subbasins_df = subbasins_df

        if self.verbose:
            print(f"  ✓ 已加载 {len(precipitation_df)} 小时, {len(precipitation_df.columns)} 个子流域的降雨数据")
            print(f"  ✓ 已加载 {len(subbasins_df)} 个子流域信息")

    def analyze(self) -> DiagnosticResult:
        """执行降雨空间分布分析"""
        # 初始化结果
        self._result = DiagnosticResult(diagnostic_name=self.diagnostic_name)

        # 1. 分析各分区降雨
        self._analyze_zones()

        # 2. 检查分区间差异
        self._check_inter_zone_anomalies()

        # 3. 检查数据质量
        self._check_data_quality()

        # 4. 生成建议
        self._generate_recommendations()

        return self._result

    def _analyze_zones(self) -> None:
        """分析各分区降雨"""
        # 获取所有分区ID
        zone_ids = sorted(self._subbasins_df['zone_id'].unique())

        for zone_id in zone_ids:
            # 获取该分区的子流域
            zone_subbasins = self._subbasins_df[
                self._subbasins_df['zone_id'] == zone_id
            ]
            total_area = zone_subbasins['area_km2'].sum()

            # 计算面积加权平均降雨
            subbasin_precip = []
            for _, subbasin in zone_subbasins.iterrows():
                subbasin_id = str(int(subbasin['subzone_id']))
                area = subbasin['area_km2']
                weight = area / total_area

                if subbasin_id in self._precipitation_df.columns:
                    precip_series = self._precipitation_df[subbasin_id]
                    total_precip = precip_series.sum()
                    mean_intensity = precip_series.mean()

                    subbasin_precip.append({
                        'subbasin_id': subbasin_id,
                        'area_km2': area,
                        'total_precip_mm': total_precip,
                        'mean_intensity_mm_h': mean_intensity,
                        'weight': weight,
                        'weighted_precip': total_precip * weight
                    })

            # 计算分区统计
            if subbasin_precip:
                weighted_avg = sum(s['weighted_precip'] for s in subbasin_precip)
                precip_values = [s['total_precip_mm'] for s in subbasin_precip]
                min_precip = min(precip_values)
                max_precip = max(precip_values)
                cv = np.std(precip_values) / np.mean(precip_values)

                self._zone_stats[zone_id] = {
                    'zone_id': zone_id,
                    'num_subbasins': len(zone_subbasins),
                    'total_area_km2': total_area,
                    'weighted_avg_mm': weighted_avg,
                    'min_mm': min_precip,
                    'max_mm': max_precip,
                    'cv': cv,
                    'subbasin_details': subbasin_precip
                }

                # 添加指标
                self.add_metric(f"zone_{zone_id}_weighted_avg_mm", weighted_avg)
                self.add_metric(f"zone_{zone_id}_cv", cv)

    def _check_inter_zone_anomalies(self) -> None:
        """检查分区间降雨异常"""
        if len(self._zone_stats) < 2:
            return

        # 计算所有分区的平均降雨
        all_zone_avgs = [z['weighted_avg_mm'] for z in self._zone_stats.values()]
        overall_mean = np.mean(all_zone_avgs)
        overall_std = np.std(all_zone_avgs)

        self.add_metric("overall_mean_precipitation_mm", overall_mean)
        self.add_metric("overall_std_precipitation_mm", overall_std)

        # 检查每个分区是否异常
        for zone_id, stats in self._zone_stats.items():
            weighted_avg = stats['weighted_avg_mm']
            relative_diff = (weighted_avg - overall_mean) / overall_mean

            # 相对差异过大
            if abs(relative_diff) > self.anomaly_threshold:
                severity = (
                    IssueSeverity.ERROR if abs(relative_diff) > 0.5
                    else IssueSeverity.WARNING
                )

                self.add_issue(
                    category="zone_precipitation_anomaly",
                    severity=severity,
                    message=(
                        f"分区 {zone_id} 降雨量 {weighted_avg:.2f} mm "
                        f"与平均值 {overall_mean:.2f} mm 差异 {relative_diff*100:.1f}%"
                    ),
                    details={
                        "zone_id": zone_id,
                        "zone_precipitation": weighted_avg,
                        "overall_mean": overall_mean,
                        "relative_difference": relative_diff
                    },
                    suggestion=(
                        "检查雨量站分布、Thiessen多边形权重或降雨插值方法"
                        if relative_diff < 0
                        else "检查该分区是否有异常的雨量站数据"
                    )
                )

            # 分区内变异系数过大
            if stats['cv'] > 0.5:
                self.add_issue(
                    category="zone_variability",
                    severity=IssueSeverity.WARNING,
                    message=f"分区 {zone_id} 内部降雨变异系数 {stats['cv']:.4f} 过大",
                    details={
                        "zone_id": zone_id,
                        "cv": stats['cv'],
                        "min_mm": stats['min_mm'],
                        "max_mm": stats['max_mm']
                    },
                    suggestion="检查子流域降雨插值方法或雨量站覆盖"
                )

    def _check_data_quality(self) -> None:
        """检查降雨数据质量"""
        # 检查缺失值
        missing_count = self._precipitation_df.isnull().sum().sum()
        if missing_count > 0:
            self.add_issue(
                category="data_quality",
                severity=IssueSeverity.WARNING,
                message=f"降雨数据存在 {missing_count} 个缺失值",
                details={"missing_count": missing_count},
                suggestion="填补缺失值或检查数据源"
            )

        # 检查负值
        negative_count = (self._precipitation_df < 0).sum().sum()
        if negative_count > 0:
            self.add_issue(
                category="data_quality",
                severity=IssueSeverity.ERROR,
                message=f"降雨数据存在 {negative_count} 个负值",
                details={"negative_count": negative_count},
                suggestion="检查数据处理流程"
            )

        # 检查异常大值（> 100 mm/h）
        extreme_count = (self._precipitation_df > 100).sum().sum()
        if extreme_count > 0:
            max_value = self._precipitation_df.max().max()
            self.add_issue(
                category="data_quality",
                severity=IssueSeverity.WARNING,
                message=f"降雨数据存在 {extreme_count} 个异常大值 (最大 {max_value:.2f} mm/h)",
                details={
                    "extreme_count": extreme_count,
                    "max_value": max_value
                },
                suggestion="检查是否为数据错误或极端降雨事件"
            )

    def _generate_recommendations(self) -> None:
        """生成修复建议"""
        # 如果有降雨异常分区
        anomaly_zones = [
            issue.details['zone_id']
            for issue in self._result.issues
            if issue.category == "zone_precipitation_anomaly"
        ]

        if anomaly_zones:
            self.add_recommendation(
                "检查雨量站位置和Thiessen多边形权重分配"
            )
            self.add_recommendation(
                "考虑使用真实降雨数据替代合成数据（如适用）"
            )
            self.add_recommendation(
                "优化降雨空间插值方法（如IDW、Kriging或PRISM）"
            )
            self.add_recommendation(
                f"增加降雨异常分区（{', '.join(map(str, anomaly_zones))}）的雨量站密度"
            )

        # 如果有数据质量问题
        quality_issues = [
            issue for issue in self._result.issues
            if issue.category == "data_quality"
        ]
        if quality_issues:
            self.add_recommendation(
                "对降雨数据进行质量控制和清洗"
            )

    def visualize(self) -> None:
        """生成降雨空间分布可视化"""
        if self._result is None or not self._zone_stats:
            return

        # 准备分区对比数据
        zone_ids = sorted(self._zone_stats.keys())
        zone_labels = [f"Zone {z}" for z in zone_ids]
        zone_precip = [self._zone_stats[z]['weighted_avg_mm'] for z in zone_ids]
        zone_cv = [self._zone_stats[z]['cv'] for z in zone_ids]

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 子图1: 分区降雨对比
        ax1 = axes[0, 0]
        colors = [
            'red' if p < np.mean(zone_precip) * 0.7
            else 'orange' if p < np.mean(zone_precip) * 0.9
            else 'green'
            for p in zone_precip
        ]
        bars = ax1.bar(zone_labels, zone_precip, color=colors, alpha=0.7, edgecolor='black')
        ax1.axhline(
            y=np.mean(zone_precip),
            color='blue',
            linestyle='--',
            linewidth=2,
            label=f'平均: {np.mean(zone_precip):.1f}mm'
        )
        ax1.set_xlabel('分区')
        ax1.set_ylabel('面积加权平均降雨 (mm)')
        ax1.set_title('各分区降雨对比')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar, val in zip(bars, zone_precip):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{val:.1f}',
                ha='center',
                va='bottom',
                fontsize=9
            )

        # 子图2: 分区内变异系数
        ax2 = axes[0, 1]
        bars = ax2.bar(zone_labels, zone_cv, color='skyblue', alpha=0.7, edgecolor='black')
        ax2.axhline(
            y=0.5,
            color='red',
            linestyle='--',
            linewidth=2,
            label='异常阈值: 0.5'
        )
        ax2.set_xlabel('分区')
        ax2.set_ylabel('变异系数 (CV)')
        ax2.set_title('各分区降雨变异系数')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # 子图3: 详细分析第一个分区（示例）
        ax3 = axes[1, 0]
        if zone_ids:
            first_zone = zone_ids[0]
            subbasin_data = self._zone_stats[first_zone]['subbasin_details']
            if subbasin_data:
                subbasin_precip = [s['total_precip_mm'] for s in subbasin_data]
                ax3.hist(subbasin_precip, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
                ax3.axvline(
                    x=np.mean(subbasin_precip),
                    color='red',
                    linestyle='--',
                    linewidth=2,
                    label=f'平均: {np.mean(subbasin_precip):.1f}mm'
                )
                ax3.set_xlabel('降雨量 (mm)')
                ax3.set_ylabel('子流域数量')
                ax3.set_title(f'Zone {first_zone} 子流域降雨分布')
                ax3.legend()
                ax3.grid(True, alpha=0.3)

        # 子图4: 降雨时间序列（选取第一个子流域示例）
        ax4 = axes[1, 1]
        if len(self._precipitation_df.columns) > 0:
            # 随机选择最多5个子流域
            sample_cols = self._precipitation_df.columns[:min(5, len(self._precipitation_df.columns))]
            for col in sample_cols:
                ax4.plot(
                    self._precipitation_df[col].values,
                    label=f'子流域 {col}',
                    alpha=0.7
                )
            ax4.set_xlabel('时间步 (小时)')
            ax4.set_ylabel('降雨强度 (mm/h)')
            ax4.set_title('降雨时间序列示例')
            ax4.legend(fontsize=8)
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图表
        fig_path = self.output_dir / f"{self.diagnostic_name}_visualization.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        self._result.figures["precipitation_visualization"] = fig_path

        if self.verbose:
            print(f"  ✓ 可视化已保存: {fig_path}")

        # 为每个异常分区生成详细诊断图
        for zone_id in zone_ids:
            self._visualize_zone_detail(zone_id)

    def _visualize_zone_detail(self, zone_id: int) -> None:
        """为指定分区生成详细诊断图"""
        if zone_id not in self._zone_stats:
            return

        stats = self._zone_stats[zone_id]
        subbasin_data = stats['subbasin_details']

        if not subbasin_data:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 子图1: 子流域降雨柱状图
        ax1 = axes[0, 0]
        subbasin_ids = [s['subbasin_id'] for s in subbasin_data]
        precip_values = [s['total_precip_mm'] for s in subbasin_data]
        weighted_avg = stats['weighted_avg_mm']

        colors = [
            'red' if p < weighted_avg * 0.7
            else 'orange' if p < weighted_avg * 0.9
            else 'green'
            for p in precip_values
        ]

        ax1.bar(range(len(subbasin_ids)), precip_values, color=colors, alpha=0.7)
        ax1.axhline(
            y=weighted_avg,
            color='blue',
            linestyle='--',
            linewidth=2,
            label=f'加权平均: {weighted_avg:.1f}mm'
        )
        ax1.set_xlabel('子流域索引')
        ax1.set_ylabel('总降雨 (mm)')
        ax1.set_title(f'Zone {zone_id} 各子流域降雨分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 子图2: 降雨-面积权重散点图
        ax2 = axes[0, 1]
        areas = [s['area_km2'] for s in subbasin_data]
        weights = [s['weight'] for s in subbasin_data]
        scatter = ax2.scatter(
            precip_values,
            weights,
            s=[a * 10 for a in areas],
            c=precip_values,
            cmap='RdYlGn',
            alpha=0.6
        )
        ax2.set_xlabel('总降雨 (mm)')
        ax2.set_ylabel('面积权重')
        ax2.set_title(f'Zone {zone_id} 降雨与权重关系 (气泡大小=面积)')
        plt.colorbar(scatter, ax=ax2, label='降雨量(mm)')
        ax2.grid(True, alpha=0.3)

        # 子图3: 时间序列（前5个子流域）
        ax3 = axes[1, 0]
        for s in subbasin_data[:5]:
            subbasin_id = s['subbasin_id']
            if subbasin_id in self._precipitation_df.columns:
                ax3.plot(
                    self._precipitation_df[subbasin_id].values,
                    label=f'子流域 {subbasin_id}',
                    alpha=0.7
                )
        ax3.set_xlabel('时间步 (小时)')
        ax3.set_ylabel('降雨强度 (mm/h)')
        ax3.set_title(f'Zone {zone_id} 降雨时间序列 (前5个子流域)')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # 子图4: 降雨分布直方图
        ax4 = axes[1, 1]
        ax4.hist(precip_values, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax4.axvline(
            x=weighted_avg,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'加权平均: {weighted_avg:.1f}mm'
        )
        ax4.set_xlabel('总降雨 (mm)')
        ax4.set_ylabel('子流域数量')
        ax4.set_title(f'Zone {zone_id} 降雨分布直方图')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图表
        fig_path = self.output_dir / f"zone_{zone_id}_precipitation_detail.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        self._result.figures[f"zone_{zone_id}_detail"] = fig_path
