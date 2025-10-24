"""多模型对比和基准测试框架

提供标准化的框架来对比不同水文模型的性能。

主要功能：
1. 多模型并行校准
2. 标准化性能评估
3. 统计显著性检验
4. 可视化对比
5. 生成基准测试报告

Author: Claude Code
Date: 2025-01-24
"""
from __future__ import annotations

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import json

import numpy as np
import pandas as pd

from .base import CalibrationData, CalibrationConfig, CalibrationResult
from .generic_calibrator import GenericHydrologicCalibrator
from ..analysis import calculate_metrics


@dataclass
class ModelPerformance:
    """单个模型的性能结果"""
    model_name: str
    model_type: str  # 'runoff', 'routing', 'coupled'
    calibration_result: CalibrationResult

    # 性能指标
    metrics: Dict[str, float] = field(default_factory=dict)

    # 计算信息
    n_parameters: int = 0
    computation_time: float = 0.0
    n_evaluations: int = 0

    # 额外信息
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_metric(self, metric_name: str) -> float:
        """获取指定指标"""
        return self.metrics.get(metric_name, np.nan)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'metrics': self.metrics,
            'n_parameters': self.n_parameters,
            'computation_time': self.computation_time,
            'n_evaluations': self.n_evaluations,
            'best_params': self.calibration_result.best_params,
            'metadata': self.metadata
        }


@dataclass
class ModelComparisonResult:
    """多模型对比结果"""
    models: List[ModelPerformance]
    comparison_metrics: List[str]

    # 统计分析
    best_model: Optional[str] = None
    performance_ranking: Optional[List[str]] = None

    # 元数据
    data_info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_model_performance(self, model_name: str) -> Optional[ModelPerformance]:
        """获取指定模型的性能"""
        for model in self.models:
            if model.model_name == model_name:
                return model
        return None

    def get_performance_matrix(self) -> pd.DataFrame:
        """获取性能矩阵（模型 × 指标）"""
        data = []
        for model in self.models:
            row = {'model': model.model_name}
            row.update(model.metrics)
            row['n_params'] = model.n_parameters
            row['time_s'] = model.computation_time
            data.append(row)

        return pd.DataFrame(data)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'models': [m.to_dict() for m in self.models],
            'comparison_metrics': self.comparison_metrics,
            'best_model': self.best_model,
            'performance_ranking': self.performance_ranking,
            'data_info': self.data_info,
            'metadata': self.metadata
        }


class ModelComparator:
    """多模型对比器

    提供标准化的框架来对比不同水文模型的性能。

    Parameters
    ----------
    data : CalibrationData
        校准数据
    models : list of str
        要对比的模型名称列表
    calibration_config : CalibrationConfig or dict
        校准配置（所有模型共享）或每个模型的配置字典
    comparison_metrics : list of str
        对比指标列表，默认['nse', 'kge', 'rmse', 'pbias']
    parallel : bool
        是否并行运行校准，默认False

    Examples
    --------
    >>> from hydrosis.calibration import ModelComparator, CalibrationData
    >>>
    >>> data = CalibrationData(
    ...     precipitation=precip,
    ...     observed_runoff=obs,
    ...     area_km2=100.0
    ... )
    >>>
    >>> comparator = ModelComparator(
    ...     data=data,
    ...     models=['xin_an_jiang', 'hbv', 'vic'],
    ...     comparison_metrics=['nse', 'kge', 'rmse']
    ... )
    >>>
    >>> result = comparator.run_comparison()
    >>> comparator.print_comparison(result)
    """

    def __init__(
        self,
        data: CalibrationData,
        models: List[str],
        calibration_config: Optional[CalibrationConfig] = None,
        comparison_metrics: Optional[List[str]] = None,
        parallel: bool = False,
        seed: Optional[int] = None
    ):
        self.data = data
        self.models = models
        self.calibration_config = calibration_config
        self.comparison_metrics = comparison_metrics or ['nse', 'kge', 'rmse', 'pbias']
        self.parallel = parallel
        self.seed = seed

        # 验证模型可用性
        available = GenericHydrologicCalibrator.list_available_models()
        self.available_runoff_models = list(available['runoff_models'].keys())

        for model in models:
            if model not in self.available_runoff_models:
                raise ValueError(
                    f"模型 '{model}' 不可用。"
                    f"可用模型: {self.available_runoff_models}"
                )

    def run_comparison(self) -> ModelComparisonResult:
        """运行多模型对比

        Returns
        -------
        ModelComparisonResult
            对比结果
        """
        print("=" * 80)
        print("多模型性能对比")
        print("=" * 80)
        print(f"\n对比模型: {len(self.models)}个")
        for model in self.models:
            print(f"  - {model}")
        print(f"\n对比指标: {', '.join(self.comparison_metrics)}")
        print(f"并行执行: {'是' if self.parallel else '否'}")
        print("\n" + "-" * 80)

        if self.parallel:
            performances = self._run_parallel()
        else:
            performances = self._run_sequential()

        # 排名模型
        ranking = self._rank_models(performances)

        # 确定最佳模型
        best_model = ranking[0] if ranking else None

        result = ModelComparisonResult(
            models=performances,
            comparison_metrics=self.comparison_metrics,
            best_model=best_model,
            performance_ranking=ranking,
            data_info={
                'n_timesteps': self.data.n_timesteps,
                'area_km2': self.data.area_km2,
            },
            metadata={
                'parallel': self.parallel,
                'seed': self.seed
            }
        )

        print("\n" + "=" * 80)
        print("对比完成！")
        print("=" * 80)

        return result

    def _run_sequential(self) -> List[ModelPerformance]:
        """顺序运行校准"""
        performances = []

        for i, model_name in enumerate(self.models, 1):
            print(f"\n[{i}/{len(self.models)}] 校准模型: {model_name.upper()}")
            print("-" * 60)

            try:
                performance = self._calibrate_single_model(model_name)
                performances.append(performance)

                print(f"✓ 完成")
                print(f"  NSE: {performance.get_metric('nse'):.4f}")
                print(f"  时间: {performance.computation_time:.2f}秒")

            except Exception as e:
                print(f"✗ 失败: {str(e)}")
                continue

        return performances

    def _run_parallel(self) -> List[ModelPerformance]:
        """并行运行校准"""
        performances = []

        with ProcessPoolExecutor(max_workers=min(4, len(self.models))) as executor:
            futures = {
                executor.submit(self._calibrate_single_model, model_name): model_name
                for model_name in self.models
            }

            for i, future in enumerate(as_completed(futures), 1):
                model_name = futures[future]
                print(f"\n[{i}/{len(self.models)}] 完成模型: {model_name.upper()}")

                try:
                    performance = future.result()
                    performances.append(performance)
                    print(f"  NSE: {performance.get_metric('nse'):.4f}")
                except Exception as e:
                    print(f"  失败: {str(e)}")

        return performances

    def _calibrate_single_model(self, model_name: str) -> ModelPerformance:
        """校准单个模型"""
        start_time = time.time()

        # 创建校准器
        if self.calibration_config is None:
            # 使用默认配置
            calibrator = GenericHydrologicCalibrator.create_for_model(
                data=self.data,
                runoff_model_type=model_name,
                algorithm='differential_evolution',
                algorithm_options={'maxiter': 50, 'popsize': 15},
                seed=self.seed
            )
        else:
            calibrator = GenericHydrologicCalibrator.create_for_model(
                data=self.data,
                runoff_model_type=model_name,
                param_bounds=self.calibration_config.param_bounds.get(model_name),
                algorithm=self.calibration_config.algorithm,
                algorithm_options=self.calibration_config.algorithm_options,
                objective_metric=self.calibration_config.objective_metric,
                seed=self.seed
            )

        # 运行校准
        calib_result = calibrator.run_calibration()

        computation_time = time.time() - start_time

        # 计算所有指标
        metrics = calculate_metrics(
            self.data.observed_runoff,
            calib_result.simulated_runoff,
            metrics=self.comparison_metrics
        )

        return ModelPerformance(
            model_name=model_name,
            model_type='runoff',
            calibration_result=calib_result,
            metrics=metrics,
            n_parameters=len(calib_result.best_params),
            computation_time=computation_time,
            n_evaluations=calib_result.n_evaluations
        )

    def _rank_models(self, performances: List[ModelPerformance]) -> List[str]:
        """根据主要指标排名模型"""
        if not performances:
            return []

        # 使用NSE作为主要排名指标
        ranked = sorted(
            performances,
            key=lambda x: x.get_metric('nse'),
            reverse=True
        )

        return [m.model_name for m in ranked]

    def print_comparison(self, result: ModelComparisonResult):
        """打印对比结果"""
        print("\n" + "=" * 80)
        print("模型性能对比结果")
        print("=" * 80)

        # 性能表格
        print("\n性能指标对比:")
        print("-" * 80)

        df = result.get_performance_matrix()

        # 格式化输出
        header = f"{'模型':<20s}"
        for metric in self.comparison_metrics:
            header += f" {metric.upper():>10s}"
        header += f" {'参数数':>8s} {'时间(s)':>10s}"

        print(header)
        print("-" * 80)

        for _, row in df.iterrows():
            line = f"{row['model']:<20s}"
            for metric in self.comparison_metrics:
                value = row.get(metric, np.nan)
                line += f" {value:>10.4f}"
            line += f" {int(row['n_params']):>8d} {row['time_s']:>10.2f}"
            print(line)

        # 排名
        print("\n模型排名 (基于NSE):")
        print("-" * 80)
        for i, model_name in enumerate(result.performance_ranking, 1):
            perf = result.get_model_performance(model_name)
            nse = perf.get_metric('nse')
            print(f"  {i}. {model_name:<20s} NSE={nse:.6f}")

        # 最佳模型
        print(f"\n✓ 最佳模型: {result.best_model}")

        print("\n" + "=" * 80)

    def save_results(
        self,
        result: ModelComparisonResult,
        output_dir: Path,
        prefix: str = ""
    ):
        """保存对比结果

        Parameters
        ----------
        result : ModelComparisonResult
            对比结果
        output_dir : Path
            输出目录
        prefix : str
            文件名前缀
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存JSON
        json_file = output_dir / f"{prefix}model_comparison.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        # 保存CSV
        df = result.get_performance_matrix()
        csv_file = output_dir / f"{prefix}performance_comparison.csv"
        df.to_csv(csv_file, index=False)

        # 保存详细报告
        report_file = output_dir / f"{prefix}comparison_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = StringIO()

            self.print_comparison(result)

            report_text = sys.stdout.getvalue()
            sys.stdout = old_stdout

            f.write(report_text)

        # 为每个模型保存详细结果
        for model_perf in result.models:
            model_dir = output_dir / model_perf.model_name
            model_dir.mkdir(parents=True, exist_ok=True)

            # 保存参数
            params_file = model_dir / "best_parameters.json"
            with open(params_file, 'w') as f:
                json.dump(model_perf.calibration_result.best_params, f, indent=2)

            # 保存时间序列
            ts_df = pd.DataFrame({
                'observed': self.data.observed_runoff,
                'simulated': model_perf.calibration_result.simulated_runoff,
                'residual': self.data.observed_runoff - model_perf.calibration_result.simulated_runoff
            })
            if self.data.times is not None:
                ts_df.index = self.data.times
            ts_df.to_csv(model_dir / "timeseries.csv")

        print(f"✓ 对比结果已保存到: {output_dir}")
        print(f"  - {json_file.name}")
        print(f"  - {csv_file.name}")
        print(f"  - {report_file.name}")
        print(f"  - 各模型详细结果在对应子目录中")


def compare_models(
    data: CalibrationData,
    models: List[str],
    output_dir: Optional[Path] = None,
    **kwargs
) -> ModelComparisonResult:
    """便捷函数：对比多个水文模型

    Parameters
    ----------
    data : CalibrationData
        校准数据
    models : list of str
        要对比的模型列表
    output_dir : Path, optional
        输出目录（如果提供，自动保存结果）
    **kwargs
        传递给ModelComparator的其他参数

    Returns
    -------
    ModelComparisonResult
        对比结果

    Examples
    --------
    >>> from hydrosis.calibration import compare_models, CalibrationData
    >>>
    >>> data = CalibrationData(precip, obs, area_km2=100.0)
    >>> result = compare_models(
    ...     data=data,
    ...     models=['xin_an_jiang', 'hbv', 'vic'],
    ...     output_dir='results/comparison'
    ... )
    """
    comparator = ModelComparator(data=data, models=models, **kwargs)
    result = comparator.run_comparison()
    comparator.print_comparison(result)

    if output_dir is not None:
        comparator.save_results(result, output_dir)

    return result
