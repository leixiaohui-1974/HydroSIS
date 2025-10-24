"""通用产流模型校准器

统一的校准框架，支持所有产流模型（HBV, XinAnJiang, VIC, HYMOD等）。

主要特点：
1. 通过模型注册表动态加载任意产流模型
2. 统一的参数率定接口
3. 支持所有优化算法
4. 自动敏感性分析
5. 无需为每个模型编写单独的Calibrator类

Author: Claude Code
Date: 2025-01-24
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Type, Union

import numpy as np

from .base import BaseCalibrator, CalibrationData, CalibrationConfig
from ..runoff.base import RunoffModel, RunoffModelConfig


class MockSubbasin:
    """模拟Subbasin对象（产流模型需要）"""
    def __init__(self, area_km2: float):
        self.area_km2 = area_km2


class GenericRunoffCalibrator(BaseCalibrator):
    """通用产流模型校准器

    支持所有注册的产流模型，无需为每个模型单独编写Calibrator类。

    Parameters
    ----------
    data : CalibrationData
        校准数据
    config : CalibrationConfig
        校准配置
    model_type : str
        模型类型（已注册的模型名称）
        可选值: 'hbv', 'xin_an_jiang', 'vic', 'hymod', 'wetspa'等
    model_class : Type[RunoffModel], optional
        直接提供模型类（如果未注册）
    temperature : np.ndarray, optional
        温度数据（某些模型需要，如HBV）
    output_dir : Path, optional
        输出目录

    Examples
    --------
    使用XinAnJiang模型：

    >>> data = CalibrationData(
    ...     precipitation=precip,
    ...     observed_runoff=obs,
    ...     area_km2=100.0
    ... )
    >>> config = CalibrationConfig(
    ...     param_bounds={
    ...         'wm': [50, 250],
    ...         'b': [0.1, 0.5],
    ...         'imp': [0.0, 0.3],
    ...         'recession': [0.3, 0.9]
    ...     }
    ... )
    >>> calibrator = GenericRunoffCalibrator(
    ...     data=data,
    ...     config=config,
    ...     model_type='xin_an_jiang'
    ... )
    >>> result = calibrator.run_calibration()

    使用VIC模型：

    >>> config_vic = CalibrationConfig(
    ...     param_bounds={
    ...         'infiltration_shape': [0.1, 1.0],
    ...         'max_soil_moisture': [50, 300],
    ...         'baseflow_coefficient': [0.001, 0.1]
    ...     }
    ... )
    >>> calibrator_vic = GenericRunoffCalibrator(
    ...     data=data,
    ...     config=config_vic,
    ...     model_type='vic'
    ... )
    >>> result_vic = calibrator_vic.run_calibration()

    Notes
    -----
    这个统一的校准器消除了为每个模型编写单独Calibrator类的需要，
    通过模型注册表和统一接口支持所有产流模型。
    """

    def __init__(
        self,
        data: CalibrationData,
        config: CalibrationConfig,
        model_type: Optional[str] = None,
        model_class: Optional[Type[RunoffModel]] = None,
        temperature: Optional[np.ndarray] = None,
        **kwargs
    ):
        super().__init__(data, config, **kwargs)

        # 确定使用哪个模型
        if model_type is None and model_class is None:
            raise ValueError("必须提供 model_type 或 model_class 之一")

        if model_type is not None:
            # 从注册表获取模型类
            if model_type not in RunoffModelConfig.REGISTRY:
                available = list(RunoffModelConfig.REGISTRY.keys())
                raise ValueError(
                    f"模型类型 '{model_type}' 未注册。"
                    f"可用模型: {available}"
                )
            self.model_class = RunoffModelConfig.REGISTRY[model_type]
            self.model_type = model_type
        else:
            # 直接使用提供的模型类
            self.model_class = model_class
            self.model_type = model_class.__name__

        # 创建Mock Subbasin
        self._subbasin = MockSubbasin(self.data.area_km2)

        # 温度数据（某些模型需要）
        self.temperature = temperature
        if self.temperature is not None:
            if len(self.temperature) != self.data.n_timesteps:
                raise ValueError(
                    f"温度序列长度 ({len(self.temperature)}) "
                    f"与降雨序列长度 ({self.data.n_timesteps}) 不一致"
                )

        # 记录模型类型到元数据
        self.metadata['model_type'] = self.model_type
        self.metadata['model_class'] = self.model_class.__name__

    def create_model(self, params: Dict[str, Any]) -> RunoffModel:
        """创建产流模型实例

        Parameters
        ----------
        params : dict
            模型参数（待率定参数）

        Returns
        -------
        RunoffModel
            产流模型实例
        """
        # 合并待率定参数和固定参数
        full_params = {**self.config.fixed_params, **params}

        # 创建模型实例
        try:
            model = self.model_class(parameters=full_params)
        except Exception as e:
            # 提供更详细的错误信息
            raise RuntimeError(
                f"创建 {self.model_type} 模型失败: {str(e)}\n"
                f"参数: {full_params}"
            ) from e

        return model

    def run_model(self, model: RunoffModel) -> np.ndarray:
        """运行产流模型

        Parameters
        ----------
        model : RunoffModel
            产流模型实例

        Returns
        -------
        np.ndarray
            模拟径流 (m³/s)
        """
        # 准备输入数据
        precipitation_list = self.data.precipitation.tolist()

        # 运行模型
        try:
            simulated = model.simulate(self._subbasin, precipitation_list)
        except Exception as e:
            raise RuntimeError(
                f"运行 {self.model_type} 模型失败: {str(e)}"
            ) from e

        return np.array(simulated)

    @classmethod
    def get_default_param_bounds(cls, model_type: str) -> Dict[str, tuple]:
        """获取模型的默认参数边界

        Parameters
        ----------
        model_type : str
            模型类型

        Returns
        -------
        dict
            参数边界字典 {param_name: (lower, upper)}

        Notes
        -----
        基于文献和实践经验的推荐参数范围
        """
        # 常见模型的默认参数边界
        default_bounds = {
            'hbv': {
                'FC': (200.0, 600.0),
                'BETA': (1.0, 3.0),
                'K0': (0.05, 0.4),
                'K1': (0.01, 0.15),
                'K2': (0.001, 0.05),
                'PERC': (0.5, 5.0),
            },
            'xin_an_jiang': {
                'wm': (50.0, 250.0),
                'b': (0.1, 0.5),
                'imp': (0.0, 0.3),
                'recession': (0.3, 0.9),
            },
            'vic': {
                'infiltration_shape': (0.1, 1.0),
                'max_soil_moisture': (50.0, 300.0),
                'baseflow_coefficient': (0.001, 0.1),
                'recession': (0.7, 0.99),
            },
            'hymod': {
                'max_storage': (50.0, 200.0),
                'beta': (0.5, 2.0),
                'quickflow_ratio': (0.3, 0.9),
                'quick_k': (0.3, 0.9),
                'slow_k': (0.01, 0.2),
            },
        }

        # 处理别名
        model_type_lower = model_type.lower()

        if model_type_lower in default_bounds:
            return default_bounds[model_type_lower]

        # 未找到默认边界
        raise ValueError(
            f"模型类型 '{model_type}' 没有预定义的参数边界。"
            f"请在 CalibrationConfig 中明确指定 param_bounds。"
            f"可用的预定义模型: {list(default_bounds.keys())}"
        )

    @classmethod
    def list_available_models(cls) -> Dict[str, Type[RunoffModel]]:
        """列出所有已注册的产流模型

        Returns
        -------
        dict
            {model_name: model_class} 字典
        """
        return RunoffModelConfig.REGISTRY.copy()

    @classmethod
    def create_for_model(
        cls,
        model_type: str,
        data: CalibrationData,
        param_bounds: Optional[Dict[str, tuple]] = None,
        **config_kwargs
    ) -> GenericRunoffCalibrator:
        """便捷方法：为指定模型创建校准器

        自动使用默认参数边界（如果未提供）

        Parameters
        ----------
        model_type : str
            模型类型
        data : CalibrationData
            校准数据
        param_bounds : dict, optional
            参数边界（如果为None，使用默认值）
        **config_kwargs
            传递给CalibrationConfig的其他参数

        Returns
        -------
        GenericRunoffCalibrator
            配置好的校准器

        Examples
        --------
        >>> calibrator = GenericRunoffCalibrator.create_for_model(
        ...     model_type='xin_an_jiang',
        ...     data=calib_data,
        ...     algorithm='differential_evolution'
        ... )
        """
        # 如果未提供参数边界，使用默认值
        if param_bounds is None:
            param_bounds = cls.get_default_param_bounds(model_type)

        # 创建配置
        config = CalibrationConfig(
            param_bounds=param_bounds,
            **config_kwargs
        )

        # 创建校准器
        return cls(
            data=data,
            config=config,
            model_type=model_type
        )


# 便捷函数
def calibrate_runoff_model(
    model_type: str,
    precipitation: np.ndarray,
    observed_runoff: np.ndarray,
    area_km2: float,
    param_bounds: Optional[Dict[str, tuple]] = None,
    algorithm: str = 'differential_evolution',
    **kwargs
) -> 'CalibrationResult':
    """快速校准任意产流模型的便捷函数

    Parameters
    ----------
    model_type : str
        模型类型 ('hbv', 'xin_an_jiang', 'vic', 'hymod'等)
    precipitation : np.ndarray
        降雨时间序列 (mm/h)
    observed_runoff : np.ndarray
        观测径流 (m³/s)
    area_km2 : float
        流域面积 (km²)
    param_bounds : dict, optional
        参数边界（如果为None，使用默认值）
    algorithm : str
        优化算法
    **kwargs
        传递给校准器的其他参数

    Returns
    -------
    CalibrationResult
        校准结果

    Examples
    --------
    >>> result = calibrate_runoff_model(
    ...     model_type='xin_an_jiang',
    ...     precipitation=precip,
    ...     observed_runoff=obs,
    ...     area_km2=100.0
    ... )
    >>> print(f"NSE = {result.best_score:.4f}")
    >>> print(f"最优参数: {result.best_params}")
    """
    # 创建数据
    data = CalibrationData(
        precipitation=precipitation,
        observed_runoff=observed_runoff,
        area_km2=area_km2
    )

    # 创建校准器
    calibrator = GenericRunoffCalibrator.create_for_model(
        model_type=model_type,
        data=data,
        param_bounds=param_bounds,
        algorithm=algorithm,
        **kwargs
    )

    # 运行校准
    return calibrator.run_calibration()
