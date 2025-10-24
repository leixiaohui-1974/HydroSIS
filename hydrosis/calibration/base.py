"""校准框架基础类

提供统一的校准接口和数据结构。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class CalibrationData:
    """校准数据容器
    
    Attributes
    ----------
    precipitation : np.ndarray
        降雨时间序列 (mm/h)
    observed_runoff : np.ndarray
        观测径流时间序列 (m³/s)
    area_km2 : float
        流域面积 (km²)
    temperature : np.ndarray, optional
        温度时间序列 (°C)，某些模型需要
    times : pd.DatetimeIndex, optional
        时间索引
    metadata : dict, optional
        额外的元数据
    """
    precipitation: np.ndarray
    observed_runoff: np.ndarray
    area_km2: float
    temperature: Optional[np.ndarray] = None
    times: Optional[pd.DatetimeIndex] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """验证数据一致性"""
        if len(self.precipitation) != len(self.observed_runoff):
            raise ValueError(
                f"降雨和径流长度不一致: "
                f"{len(self.precipitation)} vs {len(self.observed_runoff)}"
            )
        
        if self.temperature is not None:
            if len(self.temperature) != len(self.precipitation):
                raise ValueError("温度序列长度与降雨不一致")
    
    @property
    def n_timesteps(self) -> int:
        """时间步数"""
        return len(self.precipitation)


@dataclass
class CalibrationConfig:
    """校准配置
    
    Attributes
    ----------
    param_bounds : dict
        参数边界字典 {param_name: [min, max]}
    fixed_params : dict
        固定参数字典 {param_name: value}
    algorithm : str
        优化算法 ('sce_ua', 'pso', 'de', etc.)
    algorithm_params : dict
        算法特定参数
    objective_metric : str
        目标指标 ('nse', 'kge', 'rmse', etc.)
    maximize : bool
        是否最大化目标函数（True for NSE/KGE, False for RMSE）
    warmup_steps : int
        预热期步数
    seed : int, optional
        随机种子
    """
    param_bounds: Dict[str, List[float]]
    fixed_params: Dict[str, Any] = field(default_factory=dict)
    algorithm: str = "sce_ua"
    algorithm_params: Dict[str, Any] = field(default_factory=dict)
    objective_metric: str = "nse"
    maximize: bool = True
    warmup_steps: int = 0
    seed: Optional[int] = None
    
    def __post_init__(self):
        """验证配置"""
        # 检查参数边界格式
        for param, bounds in self.param_bounds.items():
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(
                    f"参数边界必须是[min, max]格式: {param}"
                )
            if bounds[0] >= bounds[1]:
                raise ValueError(
                    f"参数边界无效 {param}: min={bounds[0]} >= max={bounds[1]}"
                )
    
    @property
    def n_params(self) -> int:
        """待率定参数数量"""
        return len(self.param_bounds)


@dataclass
class CalibrationResult:
    """校准结果
    
    Attributes
    ----------
    best_params : dict
        最优参数
    best_score : float
        最优目标函数值
    simulated_runoff : np.ndarray
        最优参数的模拟径流
    metrics : dict
        性能指标 (NSE, KGE, RMSE, etc.)
    convergence_history : list
        收敛历史 [(iteration, score), ...]
    n_evaluations : int
        函数评估次数
    computation_time : float
        计算时间 (秒)
    algorithm : str
        使用的算法
    success : bool
        是否成功收敛
    message : str
        状态消息
    """
    best_params: Dict[str, float]
    best_score: float
    simulated_runoff: np.ndarray
    metrics: Dict[str, float]
    convergence_history: List[tuple] = field(default_factory=list)
    n_evaluations: int = 0
    computation_time: float = 0.0
    algorithm: str = ""
    success: bool = True
    message: str = ""
    
    def summary(self) -> str:
        """生成结果摘要"""
        lines = []
        lines.append("=" * 80)
        lines.append("校准结果摘要")
        lines.append("=" * 80)
        lines.append(f"算法: {self.algorithm}")
        lines.append(f"状态: {'✓ 成功' if self.success else '✗ 失败'}")
        lines.append(f"目标函数值: {self.best_score:.6f}")
        lines.append(f"函数评估次数: {self.n_evaluations}")
        lines.append(f"计算时间: {self.computation_time:.2f}秒")
        
        lines.append("\n性能指标:")
        for metric, value in self.metrics.items():
            lines.append(f"  {metric.upper()}: {value:.6f}")
        
        lines.append("\n最优参数:")
        for param, value in self.best_params.items():
            lines.append(f"  {param}: {value:.6f}")
        
        lines.append("=" * 80)
        return "\n".join(lines)


class BaseCalibrator(ABC):
    """校准器抽象基类
    
    定义统一的校准流程接口。子类需要实现模型特定的方法。
    
    Parameters
    ----------
    data : CalibrationData
        校准数据
    config : CalibrationConfig
        校准配置
    output_dir : Path, optional
        输出目录
    """
    
    def __init__(
        self,
        data: CalibrationData,
        config: CalibrationConfig,
        output_dir: Optional[Path] = None
    ):
        """初始化校准器"""
        self.data = data
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else Path("calibration_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 内部状态
        self._best_params: Optional[Dict[str, float]] = None
        self._best_simulated: Optional[np.ndarray] = None
        self._evaluation_count = 0
    
    @abstractmethod
    def create_model(self, params: Dict[str, Any]) -> Any:
        """创建模型实例
        
        Parameters
        ----------
        params : dict
            模型参数
        
        Returns
        -------
        model
            模型实例
        """
        pass
    
    @abstractmethod
    def run_model(self, model: Any) -> np.ndarray:
        """运行模型
        
        Parameters
        ----------
        model
            模型实例
        
        Returns
        -------
        np.ndarray
            模拟径流 (m³/s)
        """
        pass
    
    def create_objective_function(self) -> Callable:
        """创建目标函数
        
        Returns
        -------
        callable
            接受参数列表，返回目标函数值的函数
        """
        param_names = list(self.config.param_bounds.keys())
        
        def objective(params_list: Union[list, np.ndarray]) -> float:
            """目标函数"""
            self._evaluation_count += 1
            
            try:
                # 构建参数字典
                params_dict = dict(zip(param_names, params_list))
                
                # 创建和运行模型
                model = self.create_model(params_dict)
                simulated = self.run_model(model)
                
                # 应用预热期
                if self.config.warmup_steps > 0:
                    observed = self.data.observed_runoff[self.config.warmup_steps:]
                    simulated = simulated[self.config.warmup_steps:]
                else:
                    observed = self.data.observed_runoff
                
                # 计算目标指标
                score = self.calculate_metric(
                    observed,
                    simulated,
                    self.config.objective_metric
                )
                
                # 缓存最优结果
                if self._best_params is None or (
                    self.config.maximize and score > self._best_simulated_score
                ) or (
                    not self.config.maximize and score < self._best_simulated_score
                ):
                    self._best_params = params_dict
                    self._best_simulated = simulated
                    self._best_simulated_score = score
                
                return score if not np.isnan(score) else (-999 if self.config.maximize else 999)
            
            except Exception as e:
                # 模型运行失败
                return -999 if self.config.maximize else 999
        
        return objective
    
    @staticmethod
    def calculate_metric(observed: np.ndarray, simulated: np.ndarray, metric: str) -> float:
        """计算性能指标
        
        Parameters
        ----------
        observed : np.ndarray
            观测值
        simulated : np.ndarray
            模拟值
        metric : str
            指标名称
        
        Returns
        -------
        float
            指标值
        """
        from hydrosis.evaluation.metrics import (
            nash_sutcliffe_efficiency,
            kling_gupta_efficiency,
            rmse as calc_rmse,
        )
        
        metric = metric.lower()
        if metric == "nse":
            return nash_sutcliffe_efficiency(simulated, observed)
        elif metric == "kge":
            return kling_gupta_efficiency(simulated, observed)
        elif metric == "rmse":
            return calc_rmse(simulated, observed)
        else:
            raise ValueError(f"未知指标: {metric}")
    
    def run_calibration(self) -> CalibrationResult:
        """运行校准
        
        Returns
        -------
        CalibrationResult
            校准结果
        """
        import time
        from hydrosis.analysis import calibrate_model
        
        print(f"\n开始校准 ({self.config.algorithm})...")
        print(f"  参数数量: {self.config.n_params}")
        print(f"  数据点数: {self.data.n_timesteps}")
        print(f"  目标指标: {self.config.objective_metric.upper()}")
        
        # 重置评估计数
        self._evaluation_count = 0
        self._best_params = None
        self._best_simulated = None
        self._best_simulated_score = -999 if self.config.maximize else 999
        
        # 创建目标函数
        objective = self.create_objective_function()
        
        # 运行优化
        start_time = time.time()
        
        optimization_result = calibrate_model(
            objective_function=objective,
            param_bounds=self.config.param_bounds,
            maximize=self.config.maximize,
            method=self.config.algorithm,
            seed=self.config.seed,
            **self.config.algorithm_params
        )
        
        computation_time = time.time() - start_time
        
        # 使用最优参数重新运行以获取完整输出
        model = self.create_model(self._best_params)
        best_simulated = self.run_model(model)
        
        # 计算所有指标
        metrics = {
            "nse": self.calculate_metric(self.data.observed_runoff, best_simulated, "nse"),
            "kge": self.calculate_metric(self.data.observed_runoff, best_simulated, "kge"),
            "rmse": self.calculate_metric(self.data.observed_runoff, best_simulated, "rmse"),
        }
        
        # 构建结果
        result = CalibrationResult(
            best_params=self._best_params,
            best_score=optimization_result.best_score,
            simulated_runoff=best_simulated,
            metrics=metrics,
            convergence_history=getattr(optimization_result, 'history', []),
            n_evaluations=self._evaluation_count,
            computation_time=computation_time,
            algorithm=self.config.algorithm,
            success=True,
            message="校准成功完成"
        )
        
        print(f"\n✓ 校准完成!")
        print(f"  最优 {self.config.objective_metric.upper()}: {result.best_score:.6f}")
        print(f"  计算时间: {computation_time:.2f}秒")
        print(f"  函数评估: {self._evaluation_count}次")
        
        return result
    
    def save_results(self, result: CalibrationResult, prefix: str = "") -> None:
        """保存校准结果
        
        Parameters
        ----------
        result : CalibrationResult
            校准结果
        prefix : str, optional
            文件名前缀
        """
        import json
        
        # 保存参数
        params_file = self.output_dir / f"{prefix}best_parameters.json"
        with open(params_file, 'w') as f:
            json.dump(result.best_params, f, indent=2)
        
        # 保存指标
        metrics_file = self.output_dir / f"{prefix}metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(result.metrics, f, indent=2)
        
        # 保存时间序列
        output_df = pd.DataFrame({
            'observed': self.data.observed_runoff,
            'simulated': result.simulated_runoff,
            'precipitation': self.data.precipitation
        })
        if self.data.times is not None:
            output_df.index = self.data.times
        
        series_file = self.output_dir / f"{prefix}timeseries.csv"
        output_df.to_csv(series_file)
        
        # 保存摘要
        summary_file = self.output_dir / f"{prefix}summary.txt"
        summary_file.write_text(result.summary())
        
        print(f"\n✓ 结果已保存到: {self.output_dir}")
        print(f"  参数: {params_file.name}")
        print(f"  指标: {metrics_file.name}")
        print(f"  时间序列: {series_file.name}")
