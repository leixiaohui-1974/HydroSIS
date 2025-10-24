"""集成敏感性分析模块

将敏感性分析与GenericHydrologicCalibrator集成，
为所有水文模型提供统一的敏感性分析能力。

主要特点：
- 与GenericHydrologicCalibrator无缝集成
- 支持所有产流、汇流、以及组合模型
- 多种敏感性分析方法
- 自动参数筛选和边界调整
- 可视化和报告生成

Author: Claude Code
Date: 2025-01-24
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Dict, List, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
import json

import numpy as np
import pandas as pd

from .sensitivity import (
    one_at_a_time_sensitivity,
    morris_sensitivity,
    print_sensitivity_report,
    SensitivityResult as BaseSensitivityResult
)
from .base import CalibrationData

if TYPE_CHECKING:
    from .generic_calibrator import GenericHydrologicCalibrator


@dataclass
class ModelSensitivityResult:
    """水文模型敏感性分析结果"""
    model_name: str
    model_type: str  # 'runoff', 'routing', or 'coupled'
    sensitivity_result: BaseSensitivityResult
    parameter_recommendations: Dict[str, str]  # 参数建议
    suggested_bounds: Dict[str, Tuple[float, float]]  # 建议的参数边界
    metadata: Optional[Dict[str, Any]] = None

    def get_critical_params(self, threshold: float = 0.7) -> List[str]:
        """获取关键参数（高敏感性）"""
        return [
            param for param in self.sensitivity_result.param_names
            if self.sensitivity_result.sensitivity_indices[param] > threshold
        ]

    def get_insensitive_params(self, threshold: float = 0.3) -> List[str]:
        """获取不敏感参数（低敏感性）"""
        return [
            param for param in self.sensitivity_result.param_names
            if self.sensitivity_result.sensitivity_indices[param] < threshold
        ]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'critical_params': self.get_critical_params(),
            'insensitive_params': self.get_insensitive_params(),
            'parameter_recommendations': self.parameter_recommendations,
            'suggested_bounds': {
                k: [v[0], v[1]] for k, v in self.suggested_bounds.items()
            },
            'sensitivity_indices': self.sensitivity_result.sensitivity_indices,
            'sensitivity_rankings': self.sensitivity_result.sensitivity_rankings,
            'metadata': self.metadata or {}
        }


class HydrologicSensitivityAnalyzer:
    """水文模型敏感性分析器

    与GenericHydrologicCalibrator集成，提供统一的敏感性分析接口。

    Parameters
    ----------
    calibrator : GenericHydrologicCalibrator
        已配置的校准器实例
    method : str
        敏感性分析方法 ('oat' 或 'morris')
    n_samples : int, optional
        采样数量
    seed : int, optional
        随机种子

    Examples
    --------
    >>> from hydrosis.calibration import GenericHydrologicCalibrator
    >>> calibrator = GenericHydrologicCalibrator.create_for_model(
    ...     data=calib_data,
    ...     runoff_model_type='xin_an_jiang'
    ... )
    >>> analyzer = HydrologicSensitivityAnalyzer(calibrator, method='morris')
    >>> result = analyzer.analyze()
    >>> analyzer.print_report(result)
    """

    def __init__(
        self,
        calibrator: 'GenericHydrologicCalibrator',
        method: str = 'morris',
        n_samples: Optional[int] = None,
        seed: Optional[int] = None
    ):
        self.calibrator = calibrator
        self.method = method.lower()
        self.seed = seed or calibrator.config.seed

        # 确定采样数量
        if n_samples is None:
            n_params = len(calibrator.config.param_bounds)
            if self.method == 'oat':
                self.n_samples = 10  # 每个参数10个采样点
            elif self.method == 'morris':
                self.n_samples = max(10, n_params * 4)  # Morris轨迹数
            else:
                self.n_samples = 20
        else:
            self.n_samples = n_samples

        # 模型信息
        self.model_name = getattr(calibrator, 'runoff_model_type', 'unknown')
        self.model_type = calibrator.mode.value if hasattr(calibrator, 'mode') else 'unknown'

    def analyze(self) -> ModelSensitivityResult:
        """执行敏感性分析

        Returns
        -------
        ModelSensitivityResult
            敏感性分析结果
        """
        print(f"开始 {self.method.upper()} 敏感性分析...")
        print(f"  模型: {self.model_name}")
        print(f"  参数数量: {len(self.calibrator.config.param_bounds)}")
        print(f"  采样数量: {self.n_samples}")

        # 创建模型运行函数
        def model_function(params_list: List[float]) -> float:
            """将参数列表转换为字典并运行模型"""
            param_dict = dict(zip(
                self.calibrator.config.param_bounds.keys(),
                params_list
            ))

            # 创建并运行模型
            model = self.calibrator.create_model(param_dict)
            simulated = self.calibrator.run_model(model)

            # 计算目标函数值
            objective = self.calibrator.calculate_objective(simulated)

            # 如果是最小化，取负值使大值更好
            if not self.calibrator.config.maximize:
                objective = -objective

            return objective

        # 准备参数边界
        param_names = list(self.calibrator.config.param_bounds.keys())
        param_bounds = list(self.calibrator.config.param_bounds.values())

        # 执行敏感性分析
        if self.method == 'oat':
            sensitivity_result = one_at_a_time_sensitivity(
                model_function=model_function,
                param_names=param_names,
                param_bounds=param_bounds,
                n_samples=self.n_samples
            )
        elif self.method == 'morris':
            sensitivity_result = morris_sensitivity(
                model_function=model_function,
                param_names=param_names,
                param_bounds=param_bounds,
                n_trajectories=self.n_samples,
                n_levels=4
            )
        else:
            raise ValueError(f"未知的敏感性分析方法: {self.method}")

        print(f"✓ 敏感性分析完成")
        print(f"  模型评估次数: {sensitivity_result.n_samples}")
        print(f"  计算时间: {sensitivity_result.computation_time:.2f}秒")

        # 生成参数建议
        recommendations = self._generate_recommendations(sensitivity_result)

        # 生成建议的参数边界
        suggested_bounds = self._generate_suggested_bounds(sensitivity_result)

        return ModelSensitivityResult(
            model_name=self.model_name,
            model_type=self.model_type,
            sensitivity_result=sensitivity_result,
            parameter_recommendations=recommendations,
            suggested_bounds=suggested_bounds,
            metadata={
                'method': self.method,
                'n_samples': self.n_samples,
                'seed': self.seed
            }
        )

    def _generate_recommendations(
        self,
        sensitivity_result: BaseSensitivityResult
    ) -> Dict[str, str]:
        """生成参数校准建议"""
        recommendations = {}

        for param in sensitivity_result.param_names:
            sens = sensitivity_result.sensitivity_indices[param]

            if sens > 0.7:
                recommendations[param] = "高敏感性 - 重点校准，需要精确估计"
            elif sens > 0.5:
                recommendations[param] = "中高敏感性 - 重要参数，建议校准"
            elif sens > 0.3:
                recommendations[param] = "中等敏感性 - 包含在校准中"
            elif sens > 0.1:
                recommendations[param] = "低敏感性 - 可固定为合理值"
            else:
                recommendations[param] = "极低敏感性 - 建议固定为默认值"

        return recommendations

    def _generate_suggested_bounds(
        self,
        sensitivity_result: BaseSensitivityResult
    ) -> Dict[str, Tuple[float, float]]:
        """基于敏感性生成建议的参数边界"""
        suggested = {}

        for param in sensitivity_result.param_names:
            original_bounds = sensitivity_result.parameter_ranges[param]
            sens = sensitivity_result.sensitivity_indices[param]

            original_min, original_max = original_bounds
            original_mid = (original_min + original_max) / 2
            original_range = original_max - original_min

            # 高敏感性：保持或扩大范围
            # 低敏感性：缩小范围
            if sens > 0.7:
                scale = 1.0  # 保持原范围
            elif sens > 0.5:
                scale = 0.9
            elif sens > 0.3:
                scale = 0.7
            elif sens > 0.1:
                scale = 0.5
            else:
                scale = 0.3

            new_range = original_range * scale
            new_min = original_mid - new_range / 2
            new_max = original_mid + new_range / 2

            # 确保在原边界内
            new_min = max(original_min, new_min)
            new_max = min(original_max, new_max)

            suggested[param] = (new_min, new_max)

        return suggested

    def print_report(self, result: ModelSensitivityResult):
        """打印详细报告"""
        print("\n" + "=" * 80)
        print(f"水文模型敏感性分析报告 - {result.model_name.upper()}")
        print("=" * 80)

        # 打印基础敏感性报告
        print_sensitivity_report(result.sensitivity_result)

        # 打印参数建议
        print("\n参数校准建议:")
        print("-" * 80)
        print(f"{'参数':<20s} {'敏感性':<10s} {'建议':<50s}")
        print("-" * 80)

        for param in result.sensitivity_result.sensitivity_rankings:
            sens = result.sensitivity_result.sensitivity_indices[param]
            rec = result.parameter_recommendations[param]
            print(f"{param:<20s} {sens:>8.4f}   {rec}")

        # 打印建议的参数边界
        print("\n建议的参数边界（基于敏感性调整）:")
        print("-" * 80)
        print(f"{'参数':<20s} {'原始范围':<30s} {'建议范围':<30s} {'调整':<10s}")
        print("-" * 80)

        for param in result.sensitivity_result.param_names:
            original = result.sensitivity_result.parameter_ranges[param]
            suggested = result.suggested_bounds[param]

            original_str = f"[{original[0]:.2f}, {original[1]:.2f}]"
            suggested_str = f"[{suggested[0]:.2f}, {suggested[1]:.2f}]"

            # 计算范围变化
            original_range = original[1] - original[0]
            suggested_range = suggested[1] - suggested[0]
            change_pct = (suggested_range / original_range - 1) * 100

            print(f"{param:<20s} {original_str:<30s} {suggested_str:<30s} {change_pct:>6.1f}%")

        print("\n" + "=" * 80)

    def save_results(
        self,
        result: ModelSensitivityResult,
        output_dir: Path,
        prefix: str = ""
    ):
        """保存敏感性分析结果

        Parameters
        ----------
        result : ModelSensitivityResult
            分析结果
        output_dir : Path
            输出目录
        prefix : str
            文件名前缀
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存JSON
        json_file = output_dir / f"{prefix}sensitivity_analysis.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        # 保存CSV（参数排名）
        ranking_df = pd.DataFrame([
            {
                'rank': i + 1,
                'parameter': param,
                'sensitivity': result.sensitivity_result.sensitivity_indices[param],
                'recommendation': result.parameter_recommendations[param],
                'original_min': result.sensitivity_result.parameter_ranges[param][0],
                'original_max': result.sensitivity_result.parameter_ranges[param][1],
                'suggested_min': result.suggested_bounds[param][0],
                'suggested_max': result.suggested_bounds[param][1],
            }
            for i, param in enumerate(result.sensitivity_result.sensitivity_rankings)
        ])
        csv_file = output_dir / f"{prefix}parameter_sensitivity.csv"
        ranking_df.to_csv(csv_file, index=False)

        # 保存文本报告
        report_file = output_dir / f"{prefix}sensitivity_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            # 重定向print到文件
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = StringIO()

            self.print_report(result)

            report_text = sys.stdout.getvalue()
            sys.stdout = old_stdout

            f.write(report_text)

        print(f"✓ 敏感性分析结果已保存到: {output_dir}")
        print(f"  - {json_file.name}")
        print(f"  - {csv_file.name}")
        print(f"  - {report_file.name}")


def analyze_model_sensitivity(
    calibrator: 'GenericHydrologicCalibrator',
    method: str = 'morris',
    n_samples: Optional[int] = None,
    output_dir: Optional[Path] = None,
    **kwargs
) -> ModelSensitivityResult:
    """便捷函数：对水文模型进行敏感性分析

    Parameters
    ----------
    calibrator : GenericHydrologicCalibrator
        已配置的校准器
    method : str
        分析方法 ('oat' 或 'morris')
    n_samples : int, optional
        采样数量
    output_dir : Path, optional
        输出目录（如果提供，自动保存结果）
    **kwargs
        传递给HydrologicSensitivityAnalyzer的其他参数

    Returns
    -------
    ModelSensitivityResult
        敏感性分析结果

    Examples
    --------
    >>> from hydrosis.calibration import GenericHydrologicCalibrator, analyze_model_sensitivity
    >>>
    >>> calibrator = GenericHydrologicCalibrator.create_for_model(
    ...     data=calib_data,
    ...     runoff_model_type='xin_an_jiang'
    ... )
    >>> result = analyze_model_sensitivity(
    ...     calibrator,
    ...     method='morris',
    ...     output_dir='results/sensitivity'
    ... )
    """
    analyzer = HydrologicSensitivityAnalyzer(
        calibrator=calibrator,
        method=method,
        n_samples=n_samples,
        **kwargs
    )

    result = analyzer.analyze()
    analyzer.print_report(result)

    if output_dir is not None:
        analyzer.save_results(result, output_dir)

    return result
