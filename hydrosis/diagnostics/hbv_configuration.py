"""HBV模型配置诊断

全面检查HBV模型配置的合理性，包括：
- 单位转换验证
- 水量平衡分析
- 初始状态估计
- 参数合理性检查
- 模型性能评估
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .base import BaseDiagnostic, DiagnosticResult, DiagnosticIssue, IssueSeverity


class HBVConfigurationDiagnostic(BaseDiagnostic):
    """HBV模型配置诊断器

    检查HBV模型配置的各个方面，包括单位转换、水量平衡、
    初始状态和参数设置的合理性。

    Parameters
    ----------
    output_dir : Path
        输出目录路径
    verbose : bool
        是否显示详细信息
    runoff_coeff_range : tuple
        合理的径流系数范围，默认(0.3, 0.7)
    k2_estimate : float
        用于估算初始下层储量的K2值，默认0.02
    area_km2 : float
        流域面积(km²)
    timestep_hours : float
        时间步长(小时)
    """

    def __init__(
        self,
        output_dir: Path,
        verbose: bool = True,
        runoff_coeff_range: tuple = (0.3, 0.7),
        k2_estimate: float = 0.02,
        area_km2: float = None,
        timestep_hours: float = 1.0
    ):
        super().__init__(
            output_dir=output_dir,
            verbose=verbose
        )
        self.runoff_coeff_range = runoff_coeff_range
        self.k2_estimate = k2_estimate
        self.area_km2 = area_km2
        self.timestep_hours = timestep_hours

        # 诊断数据
        self.precipitation_mmh = None
        self.observed_m3s = None
        self.hbv_params = None
        self.hbv_runoff_m3s = None
        self.estimated_initial_lower = None

    @property
    def diagnostic_name(self) -> str:
        """诊断名称"""
        return "HBV配置诊断"

    def load_data(
        self,
        precipitation_mmh: np.ndarray,
        observed_m3s: np.ndarray,
        area_km2: float,
        hbv_params: Optional[Dict[str, float]] = None,
        timestep_hours: float = 1.0
    ) -> None:
        """加载数据

        Parameters
        ----------
        precipitation_mmh : np.ndarray
            降雨数据 (mm/h)
        observed_m3s : np.ndarray
            观测径流 (m³/s)
        area_km2 : float
            流域面积 (km²)
        hbv_params : dict, optional
            HBV参数，如果提供则会运行模拟
        timestep_hours : float
            时间步长(小时)
        """
        self.precipitation_mmh = precipitation_mmh
        self.observed_m3s = observed_m3s
        self.area_km2 = area_km2
        self.timestep_hours = timestep_hours
        self.hbv_params = hbv_params

        if self.verbose:
            print(f"\n✓ 数据加载:")
            print(f"  降雨序列: {len(precipitation_mmh)} 时步")
            print(f"  观测径流: {len(observed_m3s)} 时步")
            print(f"  流域面积: {area_km2:.2f} km²")
            print(f"  时间步长: {timestep_hours:.2f} h")

    def analyze(self) -> DiagnosticResult:
        """执行诊断分析"""
        result = DiagnosticResult(
            diagnostic_name=self.diagnostic_name,
            issues=[],
            metrics={},
            recommendations=[]
        )

        # 1. 单位转换检查
        self._check_unit_conversion(result)

        # 2. 水量平衡分析
        self._check_water_balance(result)

        # 3. 初始状态估计
        self._estimate_initial_state(result)

        # 4. HBV模型模拟（如果提供了参数）
        if self.hbv_params is not None:
            self._run_hbv_simulation(result)
            self._check_hbv_performance(result)

        # 5. 生成建议
        self._generate_recommendations(result)

        return result

    def _check_unit_conversion(self, result: DiagnosticResult) -> None:
        """检查单位转换"""
        # mm/h → m³/s 转换
        # Flow (m³/s) = Precip (mm/h) * Area (km²) * 1000 / 3600
        precipitation_m3s = self.precipitation_mmh * self.area_km2 * 1000 / 3600

        result.metrics['max_precipitation_mmh'] = float(self.precipitation_mmh.max())
        result.metrics['max_precipitation_m3s'] = float(precipitation_m3s.max())
        result.metrics['max_observed_m3s'] = float(self.observed_m3s.max())

        # 检查转换后的降雨是否合理
        if precipitation_m3s.max() < self.observed_m3s.max():
            issue = DiagnosticIssue(
                category="unit_conversion",
                severity=IssueSeverity.INFO,
                message=f"最大降雨流量 ({precipitation_m3s.max():.2f} m³/s) < 最大观测流量 ({self.observed_m3s.max():.2f} m³/s)",
                details={
                    'max_precip_m3s': float(precipitation_m3s.max()),
                    'max_obs_m3s': float(self.observed_m3s.max())
                },
                suggestion="这是正常的，考虑了土壤蓄水和前期降雨的累积效应"
            )
            result.issues.append(issue)

    def _check_water_balance(self, result: DiagnosticResult) -> None:
        """检查水量平衡"""
        # 计算累积水量
        total_precip_mm = self.precipitation_mmh.sum() * self.timestep_hours
        total_precip_volume_m3 = total_precip_mm * self.area_km2 * 1000

        total_runoff_volume_m3 = self.observed_m3s.sum() * self.timestep_hours * 3600
        total_runoff_mm = total_runoff_volume_m3 / (self.area_km2 * 1000)

        # 径流系数
        runoff_coefficient = total_runoff_mm / total_precip_mm if total_precip_mm > 0 else 0

        result.metrics['total_precipitation_mm'] = float(total_precip_mm)
        result.metrics['total_runoff_mm'] = float(total_runoff_mm)
        result.metrics['runoff_coefficient'] = float(runoff_coefficient)
        result.metrics['storage_change_mm'] = float(total_precip_mm - total_runoff_mm)

        # 评估径流系数
        min_rc, max_rc = self.runoff_coeff_range
        if runoff_coefficient < min_rc:
            issue = DiagnosticIssue(
                category="runoff_coefficient",
                severity=IssueSeverity.WARNING,
                message=f"径流系数偏低 ({runoff_coefficient:.4f} < {min_rc})",
                details={
                    'runoff_coefficient': float(runoff_coefficient),
                    'expected_range': self.runoff_coeff_range
                },
                suggestion="可能原因：土壤蓄水能力强、蒸散发量大或观测径流低估"
            )
            result.issues.append(issue)
        elif runoff_coefficient > max_rc:
            issue = DiagnosticIssue(
                category="runoff_coefficient",
                severity=IssueSeverity.WARNING,
                message=f"径流系数偏高 ({runoff_coefficient:.4f} > {max_rc})",
                details={
                    'runoff_coefficient': float(runoff_coefficient),
                    'expected_range': self.runoff_coeff_range
                },
                suggestion="可能原因：土壤饱和、不透水面积大或观测径流高估"
            )
            result.issues.append(issue)

        if runoff_coefficient > 1.0:
            issue = DiagnosticIssue(
                category="runoff_coefficient",
                severity=IssueSeverity.CRITICAL,
                message=f"径流系数 > 1.0 ({runoff_coefficient:.4f})，配置存在严重问题",
                details={'runoff_coefficient': float(runoff_coefficient)},
                suggestion="检查单位转换、面积计算和数据质量"
            )
            result.issues.append(issue)

    def _estimate_initial_state(self, result: DiagnosticResult) -> None:
        """估计初始状态"""
        # 从初始基流反推初始下层储量
        # HBV: Q_base = K2 * S_lower
        # 因此: S_lower = Q_base / K2
        initial_baseflow = self.observed_m3s[0]
        estimated_initial_lower = initial_baseflow / self.k2_estimate

        self.estimated_initial_lower = estimated_initial_lower

        result.metrics['initial_baseflow_m3s'] = float(initial_baseflow)
        result.metrics['k2_estimate'] = float(self.k2_estimate)
        result.metrics['estimated_initial_lower_mm'] = float(estimated_initial_lower)

        # 检查估计值的合理性
        if estimated_initial_lower > 5000:
            issue = DiagnosticIssue(
                category="initial_state",
                severity=IssueSeverity.WARNING,
                message=f"反推的初始下层储量很大 ({estimated_initial_lower:.1f} mm)",
                details={
                    'estimated_initial_lower': float(estimated_initial_lower),
                    'initial_baseflow': float(initial_baseflow),
                    'k2': float(self.k2_estimate)
                },
                suggestion="可能说明初始基流很高，或K2参数设置不当"
            )
            result.issues.append(issue)

        # 如果HBV参数中有initial_lower，检查差异
        if self.hbv_params and 'initial_lower' in self.hbv_params:
            configured_initial_lower = self.hbv_params['initial_lower']
            relative_diff = abs(configured_initial_lower - estimated_initial_lower) / estimated_initial_lower

            result.metrics['configured_initial_lower_mm'] = float(configured_initial_lower)
            result.metrics['initial_lower_relative_diff'] = float(relative_diff)

            if relative_diff > 0.5:
                issue = DiagnosticIssue(
                    category="initial_state",
                    severity=IssueSeverity.ERROR,
                    message=f"配置的initial_lower ({configured_initial_lower:.1f} mm) 与反推值相差 > 50%",
                    details={
                        'configured': float(configured_initial_lower),
                        'estimated': float(estimated_initial_lower),
                        'relative_diff': float(relative_diff)
                    },
                    suggestion=f"建议使用反推值: {estimated_initial_lower:.1f} mm"
                )
                result.issues.append(issue)

    def _run_hbv_simulation(self, result: DiagnosticResult) -> None:
        """运行HBV模拟"""
        try:
            from hydrosis.runoff.hbv import HBVRunoff

            # 创建模拟子流域
            class MockSubbasin:
                def __init__(self, area_km2):
                    self.area_km2 = area_km2

            subbasin = MockSubbasin(self.area_km2)

            # 运行HBV
            hbv = HBVRunoff(self.hbv_params)
            self.hbv_runoff_m3s = np.array(hbv.simulate(subbasin, self.precipitation_mmh.tolist()))

            result.metrics['hbv_simulation_completed'] = True

            if self.verbose:
                print(f"\n✓ HBV模拟完成")
                print(f"  输出长度: {len(self.hbv_runoff_m3s)}")
                print(f"  流量范围: {self.hbv_runoff_m3s.min():.2f} - {self.hbv_runoff_m3s.max():.2f} m³/s")

        except Exception as e:
            issue = DiagnosticIssue(
                category="hbv_simulation",
                severity=IssueSeverity.ERROR,
                message=f"HBV模拟失败: {str(e)}",
                details={'error': str(e)},
                suggestion="检查HBV参数配置和输入数据"
            )
            result.issues.append(issue)
            result.metrics['hbv_simulation_completed'] = False

    def _check_hbv_performance(self, result: DiagnosticResult) -> None:
        """检查HBV模型性能"""
        if self.hbv_runoff_m3s is None:
            return

        # 计算HBV水量平衡
        hbv_total_runoff_m3 = self.hbv_runoff_m3s.sum() * self.timestep_hours * 3600
        hbv_total_runoff_mm = hbv_total_runoff_m3 / (self.area_km2 * 1000)

        total_precip_mm = result.metrics['total_precipitation_mm']
        hbv_runoff_coefficient = hbv_total_runoff_mm / total_precip_mm if total_precip_mm > 0 else 0

        result.metrics['hbv_total_runoff_mm'] = float(hbv_total_runoff_mm)
        result.metrics['hbv_runoff_coefficient'] = float(hbv_runoff_coefficient)

        # 对比观测
        obs_runoff_coefficient = result.metrics['runoff_coefficient']
        rc_diff = abs(hbv_runoff_coefficient - obs_runoff_coefficient)

        result.metrics['runoff_coefficient_difference'] = float(rc_diff)

        if rc_diff > 0.1:
            severity = IssueSeverity.WARNING if rc_diff < 0.2 else IssueSeverity.ERROR

            if hbv_runoff_coefficient > obs_runoff_coefficient:
                reason = "HBV产流偏多，可能FC过小或BETA过大"
            else:
                reason = "HBV产流偏少，可能FC过大或BETA过小"

            issue = DiagnosticIssue(
                category="hbv_performance",
                severity=severity,
                message=f"HBV径流系数 ({hbv_runoff_coefficient:.4f}) 与观测 ({obs_runoff_coefficient:.4f}) 差异较大",
                details={
                    'hbv_rc': float(hbv_runoff_coefficient),
                    'obs_rc': float(obs_runoff_coefficient),
                    'difference': float(rc_diff)
                },
                suggestion=reason
            )
            result.issues.append(issue)

        # 计算误差指标
        residuals = self.observed_m3s - self.hbv_runoff_m3s
        rmse = np.sqrt((residuals ** 2).mean())
        mean_residual = residuals.mean()

        result.metrics['rmse_m3s'] = float(rmse)
        result.metrics['mean_residual_m3s'] = float(mean_residual)
        result.metrics['relative_rmse'] = float(rmse / self.observed_m3s.mean()) if self.observed_m3s.mean() > 0 else 0

    def _generate_recommendations(self, result: DiagnosticResult) -> None:
        """生成修复建议"""
        recommendations = []

        # 基于径流系数的建议
        if 'runoff_coefficient' in result.metrics:
            rc = result.metrics['runoff_coefficient']
            if rc < self.runoff_coeff_range[0] or rc > self.runoff_coeff_range[1]:
                recommendations.append(
                    f"径流系数 ({rc:.4f}) 不在合理范围，建议验证数据质量和单位转换"
                )

        # 基于初始状态的建议
        if 'estimated_initial_lower_mm' in result.metrics:
            est_lower = result.metrics['estimated_initial_lower_mm']
            if 'configured_initial_lower_mm' in result.metrics:
                conf_lower = result.metrics['configured_initial_lower_mm']
                if abs(conf_lower - est_lower) / est_lower > 0.3:
                    recommendations.append(
                        f"建议将 initial_lower 从 {conf_lower:.1f} mm 改为 {est_lower:.1f} mm (基于初始基流反推)"
                    )

        # 基于HBV性能的建议
        if 'runoff_coefficient_difference' in result.metrics:
            rc_diff = result.metrics['runoff_coefficient_difference']
            if rc_diff > 0.1:
                recommendations.append(
                    "HBV径流系数与观测差异较大，建议调整FC和BETA参数"
                )

        # 通用建议
        if len(self.precipitation_mmh) < 720:  # 少于30天
            recommendations.append(
                f"当前时间序列较短 ({len(self.precipitation_mmh)} h)，建议使用30-90天数据以充分识别参数"
            )

        result.recommendations = recommendations

    def visualize(self) -> None:
        """生成可视化图表"""
        if self.precipitation_mmh is None or self.observed_m3s is None:
            return

        # 创建3-4个子图
        n_plots = 4 if self.hbv_runoff_m3s is not None else 3
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots))

        if n_plots == 3:
            axes = list(axes)

        # 子图1: 降雨和径流对比
        ax1 = axes[0]
        ax1_twin = ax1.twinx()
        ax1_twin.bar(range(len(self.precipitation_mmh)), self.precipitation_mmh,
                     alpha=0.3, color='blue', label='Precipitation')
        ax1.plot(self.observed_m3s, 'o-', label='Observed Runoff',
                color='orange', linewidth=1.5, markersize=3)

        if self.hbv_runoff_m3s is not None:
            ax1.plot(self.hbv_runoff_m3s, 's-', label='HBV Simulated',
                    color='green', linewidth=1.5, markersize=3, alpha=0.7)

        ax1.set_ylabel('Discharge (m³/s)', fontsize=10)
        ax1_twin.set_ylabel('Precipitation (mm/h)', fontsize=10, color='blue')
        ax1_twin.tick_params(axis='y', labelcolor='blue')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        ax1.set_title('Precipitation and Runoff Comparison', fontsize=11)
        ax1.grid(True, alpha=0.3)

        # 子图2: 累积水量对比
        ax2 = axes[1]
        cumulative_precip = np.cumsum(self.precipitation_mmh) * self.timestep_hours
        cumulative_obs = np.cumsum(
            self.observed_m3s * self.timestep_hours * 3600 / (self.area_km2 * 1000)
        )

        ax2.plot(cumulative_precip, '-', label='Cumulative Precipitation',
                color='blue', linewidth=2)
        ax2.plot(cumulative_obs, '-', label='Cumulative Observed Runoff',
                color='orange', linewidth=2)

        if self.hbv_runoff_m3s is not None:
            cumulative_hbv = np.cumsum(
                self.hbv_runoff_m3s * self.timestep_hours * 3600 / (self.area_km2 * 1000)
            )
            ax2.plot(cumulative_hbv, '-', label='Cumulative HBV Runoff',
                    color='green', linewidth=2)

        ax2.set_ylabel('Cumulative Depth (mm)', fontsize=10)
        ax2.set_title('Cumulative Water Balance', fontsize=11)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 子图3: 径流系数对比
        ax3 = axes[2]
        rc_obs = self.observed_m3s.cumsum() * self.timestep_hours * 3600 / (self.area_km2 * 1000)
        rc_obs = rc_obs / cumulative_precip
        rc_obs = np.nan_to_num(rc_obs, nan=0, posinf=0, neginf=0)

        ax3.plot(rc_obs, '-', label='Observed Runoff Coefficient',
                color='orange', linewidth=2)

        if self.hbv_runoff_m3s is not None:
            rc_hbv = self.hbv_runoff_m3s.cumsum() * self.timestep_hours * 3600 / (self.area_km2 * 1000)
            rc_hbv = rc_hbv / cumulative_precip
            rc_hbv = np.nan_to_num(rc_hbv, nan=0, posinf=0, neginf=0)
            ax3.plot(rc_hbv, '-', label='HBV Runoff Coefficient',
                    color='green', linewidth=2)

        ax3.axhline(y=self.runoff_coeff_range[0], color='red', linestyle='--',
                   linewidth=1, label=f'Reasonable Range')
        ax3.axhline(y=self.runoff_coeff_range[1], color='red', linestyle='--',
                   linewidth=1)
        ax3.set_ylabel('Runoff Coefficient', fontsize=10)
        ax3.set_title('Runoff Coefficient Evolution', fontsize=11)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 子图4: 残差分析（如果有HBV模拟）
        if self.hbv_runoff_m3s is not None and n_plots == 4:
            ax4 = axes[3]
            residuals = self.observed_m3s - self.hbv_runoff_m3s
            ax4.bar(range(len(residuals)), residuals, alpha=0.6, color='red',
                   label='Observed - HBV')
            ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
            ax4.axhline(y=residuals.mean(), color='blue', linestyle='--',
                       linewidth=1, label=f'Mean = {residuals.mean():.2f} m³/s')
            ax4.set_xlabel('Time Step (hour)', fontsize=10)
            ax4.set_ylabel('Residual (m³/s)', fontsize=10)
            rmse = np.sqrt((residuals ** 2).mean())
            ax4.set_title(f'Residual Analysis (RMSE = {rmse:.2f} m³/s)', fontsize=11)
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图表
        output_file = self.output_dir / f"{self.diagnostic_name}_visualization.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        if self.verbose:
            print(f"\n✓ 可视化图表已保存: {output_file.name}")
