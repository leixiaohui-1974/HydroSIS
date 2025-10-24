"""水量平衡诊断

诊断HBV模型的水量平衡问题，特别是径流系数异常和初始储量配置问题。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .base import BaseDiagnostic, DiagnosticResult, IssueSeverity


class WaterBalanceDiagnostic(BaseDiagnostic):
    """水量平衡诊断器

    检查HBV模型的水量平衡，识别以下问题：
    1. 径流系数异常（> 1.0）
    2. 初始储量(initial_lower)配置不合理
    3. 基流参数(k2)配置问题
    4. 参数合理性检查

    Parameters
    ----------
    output_dir : Path, optional
        输出目录
    verbose : bool, default=True
        是否输出详细信息
    target_runoff_coefficient : float, default=0.5
        目标径流系数（用于生成修正建议）
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        verbose: bool = True,
        target_runoff_coefficient: float = 0.5
    ):
        super().__init__(output_dir, verbose)
        self.target_runoff_coefficient = target_runoff_coefficient

        # 数据
        self._precipitation: Optional[np.ndarray] = None
        self._runoff: Optional[np.ndarray] = None
        self._initial_lower: Optional[float] = None
        self._k2: Optional[float] = None

    @property
    def diagnostic_name(self) -> str:
        return "水量平衡诊断"

    def load_data(
        self,
        precipitation: np.ndarray,
        runoff: np.ndarray,
        initial_lower: float,
        k2: float
    ) -> None:
        """加载水量平衡数据

        Parameters
        ----------
        precipitation : np.ndarray
            降雨时间序列 (mm/h)
        runoff : np.ndarray
            径流时间序列 (mm/h)
        initial_lower : float
            初始下层储量 (mm)
        k2 : float
            基流系数 (1/h)
        """
        if len(precipitation) != len(runoff):
            raise ValueError(
                f"降雨和径流序列长度不一致: {len(precipitation)} vs {len(runoff)}"
            )

        self._precipitation = precipitation
        self._runoff = runoff
        self._initial_lower = initial_lower
        self._k2 = k2

        if self.verbose:
            print(f"  ✓ 已加载 {len(precipitation)} 小时的数据")

    def analyze(self) -> DiagnosticResult:
        """执行水量平衡分析"""
        # 初始化结果
        self._result = DiagnosticResult(diagnostic_name=self.diagnostic_name)

        # 基本统计
        total_precip = self._precipitation.sum()
        total_runoff = self._runoff.sum()
        runoff_coefficient = total_runoff / total_precip
        excess_runoff = total_runoff - total_precip

        # 添加基本指标
        self.add_metric("total_precipitation_mm", total_precip)
        self.add_metric("total_runoff_mm", total_runoff)
        self.add_metric("runoff_coefficient", runoff_coefficient)
        self.add_metric("excess_runoff_mm", excess_runoff)

        # 1. 检查径流系数
        self._check_runoff_coefficient(runoff_coefficient, excess_runoff)

        # 2. 分析initial_lower的影响
        self._analyze_initial_lower(excess_runoff)

        # 3. 检查参数合理性
        self._check_parameter_reasonability()

        # 4. 生成修正建议
        self._generate_recommendations(runoff_coefficient, excess_runoff)

        return self._result

    def _check_runoff_coefficient(
        self,
        runoff_coefficient: float,
        excess_runoff: float
    ) -> None:
        """检查径流系数"""
        if runoff_coefficient > 1.0:
            self.add_issue(
                category="runoff_coefficient",
                severity=IssueSeverity.ERROR,
                message=f"径流系数 {runoff_coefficient:.4f} > 1.0 (不合理)",
                details={
                    "runoff_coefficient": runoff_coefficient,
                    "excess_runoff_mm": excess_runoff
                },
                suggestion=f"初始储量释放了 {excess_runoff:.2f} mm 的额外径流，建议降低 initial_lower"
            )
        elif runoff_coefficient > 0.9:
            self.add_issue(
                category="runoff_coefficient",
                severity=IssueSeverity.WARNING,
                message=f"径流系数 {runoff_coefficient:.4f} 偏高 (典型值 0.3-0.7)",
                details={"runoff_coefficient": runoff_coefficient},
                suggestion="检查initial_lower配置"
            )
        elif runoff_coefficient < 0.1:
            self.add_issue(
                category="runoff_coefficient",
                severity=IssueSeverity.WARNING,
                message=f"径流系数 {runoff_coefficient:.4f} 偏低",
                details={"runoff_coefficient": runoff_coefficient},
                suggestion="可能需要增加initial_lower或调整HBV参数"
            )

    def _analyze_initial_lower(self, excess_runoff: float) -> None:
        """分析initial_lower的影响"""
        hours = len(self._precipitation)

        # 计算储量释放: S(t) = S0 * exp(-k2 * t)
        # 释放量 = S0 * (1 - exp(-k2 * t))
        decay_factor = np.exp(-self._k2 * hours)
        released_storage = self._initial_lower * (1 - decay_factor)
        remaining_storage = self._initial_lower * decay_factor

        # 添加指标
        self.add_metric("initial_lower_mm", self._initial_lower)
        self.add_metric("k2_per_hour", self._k2)
        self.add_metric("released_storage_mm", released_storage)
        self.add_metric("remaining_storage_mm", remaining_storage)
        self.add_metric("simulation_hours", hours)

        # 检查释放量是否与额外径流匹配
        if abs(released_storage - excess_runoff) > 50:  # 50mm容差
            self.add_issue(
                category="storage_release",
                severity=IssueSeverity.WARNING,
                message=(
                    f"理论释放储量 {released_storage:.2f} mm "
                    f"与实际额外径流 {excess_runoff:.2f} mm 差异较大"
                ),
                details={
                    "theoretical_release": released_storage,
                    "actual_excess": excess_runoff,
                    "difference": abs(released_storage - excess_runoff)
                },
                suggestion="检查k2参数或模型配置"
            )

    def _check_parameter_reasonability(self) -> None:
        """检查参数合理性"""
        initial_baseflow = self._k2 * self._initial_lower

        self.add_metric("initial_baseflow_mm_per_hour", initial_baseflow)

        if initial_baseflow > 100:
            self.add_issue(
                category="parameter_reasonability",
                severity=IssueSeverity.CRITICAL,
                message=f"初始基流 {initial_baseflow:.2f} mm/h 过大 (典型值 < 10 mm/h)",
                details={"initial_baseflow": initial_baseflow},
                suggestion="大幅降低 initial_lower 或调整 k2"
            )
        elif initial_baseflow > 10:
            self.add_issue(
                category="parameter_reasonability",
                severity=IssueSeverity.WARNING,
                message=f"初始基流 {initial_baseflow:.2f} mm/h 偏大 (典型值 < 10 mm/h)",
                details={"initial_baseflow": initial_baseflow},
                suggestion="考虑降低 initial_lower"
            )

    def _generate_recommendations(
        self,
        runoff_coefficient: float,
        excess_runoff: float
    ) -> None:
        """生成修正建议"""
        if runoff_coefficient <= 1.0:
            return  # 无需修正

        hours = len(self._precipitation)
        decay_factor = np.exp(-self._k2 * hours)
        total_precip = self._precipitation.sum()

        # 方案1: 基于目标径流系数计算
        target_total_runoff = total_precip * self.target_runoff_coefficient
        excess_to_remove = self._runoff.sum() - target_total_runoff
        recommended_initial_lower = (
            self._initial_lower - excess_to_remove / (1 - decay_factor)
        )

        self.add_recommendation(
            f"方案1 (推荐): 设置 initial_lower = {max(0, recommended_initial_lower):.2f} mm "
            f"(预期径流系数: {self.target_runoff_coefficient})"
        )

        # 方案2: 保守值
        conservative_value = 100.0
        self.add_recommendation(
            f"方案2 (保守): 设置 initial_lower = {conservative_value} mm "
            f"(保守值，适用于缺乏观测数据的情况)"
        )

        # 方案3: 从零开始
        self.add_recommendation(
            "方案3 (从零开始): 设置 initial_lower = 0.0 mm "
            "(让模型自然达到稳定状态，建议配合warmup period)"
        )

    def visualize(self) -> None:
        """生成水量平衡可视化"""
        if self._result is None:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 子图1: 降雨-径流时间序列
        ax1 = axes[0, 0]
        hours = np.arange(len(self._precipitation))
        ax1.plot(hours, self._precipitation, label='降雨', alpha=0.7)
        ax1.plot(hours, self._runoff, label='径流', alpha=0.7)
        ax1.set_xlabel('时间 (小时)')
        ax1.set_ylabel('强度 (mm/h)')
        ax1.set_title('降雨-径流时间序列')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 子图2: 累积量对比
        ax2 = axes[0, 1]
        cumulative_precip = np.cumsum(self._precipitation)
        cumulative_runoff = np.cumsum(self._runoff)
        ax2.plot(hours, cumulative_precip, label='累积降雨', linewidth=2)
        ax2.plot(hours, cumulative_runoff, label='累积径流', linewidth=2)
        ax2.axhline(
            y=cumulative_precip[-1],
            color='blue',
            linestyle='--',
            alpha=0.5,
            label=f'总降雨: {cumulative_precip[-1]:.1f}mm'
        )
        ax2.axhline(
            y=cumulative_runoff[-1],
            color='orange',
            linestyle='--',
            alpha=0.5,
            label=f'总径流: {cumulative_runoff[-1]:.1f}mm'
        )
        ax2.set_xlabel('时间 (小时)')
        ax2.set_ylabel('累积量 (mm)')
        ax2.set_title('累积降雨-径流对比')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 子图3: 储量释放模拟
        ax3 = axes[1, 0]
        storage = self._initial_lower * np.exp(-self._k2 * hours)
        released = self._initial_lower - storage
        ax3.plot(hours, storage, label='剩余储量', linewidth=2)
        ax3.plot(hours, released, label='已释放储量', linewidth=2)
        ax3.axhline(
            y=self._initial_lower,
            color='green',
            linestyle='--',
            alpha=0.5,
            label=f'初始储量: {self._initial_lower:.1f}mm'
        )
        ax3.set_xlabel('时间 (小时)')
        ax3.set_ylabel('储量 (mm)')
        ax3.set_title('下层储量变化')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 子图4: 水量平衡分析
        ax4 = axes[1, 1]
        components = ['总降雨', '总径流', '额外径流\n(来自储量)']
        values = [
            self._precipitation.sum(),
            self._runoff.sum(),
            self._runoff.sum() - self._precipitation.sum()
        ]
        colors = ['skyblue', 'orange', 'red' if values[2] > 0 else 'green']
        bars = ax4.bar(components, values, color=colors, alpha=0.7, edgecolor='black')

        # 添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{val:.1f}mm',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )

        ax4.set_ylabel('水量 (mm)')
        ax4.set_title('水量平衡分解')
        ax4.grid(True, alpha=0.3, axis='y')

        # 添加径流系数文本
        rc = self._runoff.sum() / self._precipitation.sum()
        ax4.text(
            0.5, 0.95,
            f'径流系数: {rc:.4f}',
            transform=ax4.transAxes,
            ha='center',
            va='top',
            fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

        plt.tight_layout()

        # 保存图表
        fig_path = self.output_dir / f"{self.diagnostic_name}_visualization.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        self._result.figures["water_balance_visualization"] = fig_path

        if self.verbose:
            print(f"  ✓ 可视化已保存: {fig_path}")
