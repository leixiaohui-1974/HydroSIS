"""通用水文模型校准器

统一的校准框架，支持：
1. 产流模型（Runoff Models）：HBV, XinAnJiang, VIC, HYMOD等
2. 汇流模型（Routing Models）：Muskingum, Dynamic Wave, Lag等
3. 产汇流组合（Coupled Models）：任意产流+汇流组合

消除了为每个模型或组合编写单独Calibrator类的需要。

主要特点：
- 通过模型注册表动态加载任意水文模型
- 统一的参数率定接口
- 支持多模型联合率定
- 自动敏感性分析
- 支持所有优化算法

Author: Claude Code
Date: 2025-01-24
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, Union
from enum import Enum

import numpy as np

from .base import BaseCalibrator, CalibrationData, CalibrationConfig
from ..runoff.base import RunoffModel, RunoffModelConfig
from ..routing.base import RoutingModel, RoutingModelConfig


class ModelMode(Enum):
    """模型模式"""
    RUNOFF_ONLY = "runoff_only"      # 仅产流
    ROUTING_ONLY = "routing_only"    # 仅汇流
    COUPLED = "coupled"                # 产汇流耦合


class MockSubbasin:
    """模拟Subbasin对象"""
    def __init__(self, area_km2: float):
        self.area_km2 = area_km2


class GenericHydrologicCalibrator(BaseCalibrator):
    """通用水文模型校准器

    支持产流、汇流、以及产汇流组合的统一校准框架。

    Parameters
    ----------
    data : CalibrationData
        校准数据
    config : CalibrationConfig
        校准配置
    runoff_model_type : str, optional
        产流模型类型（已注册的模型名称）
    routing_model_type : str, optional
        汇流模型类型（已注册的模型名称）
    runoff_model_class : Type[RunoffModel], optional
        直接提供产流模型类
    routing_model_class : Type[RoutingModel], optional
        直接提供汇流模型类
    temperature : np.ndarray, optional
        温度数据（某些产流模型需要，如HBV）
    output_dir : Path, optional
        输出目录

    Examples
    --------
    仅产流模型：

    >>> calibrator = GenericHydrologicCalibrator(
    ...     data=data,
    ...     config=config,
    ...     runoff_model_type='xin_an_jiang'
    ... )

    仅汇流模型：

    >>> calibrator = GenericHydrologicCalibrator(
    ...     data=data,
    ...     config=config,
    ...     routing_model_type='muskingum'
    ... )

    产汇流组合（HBV + Muskingum）：

    >>> config_coupled = CalibrationConfig(
    ...     param_bounds={
    ...         # 产流参数
    ...         'FC': [200, 600],
    ...         'BETA': [1.0, 3.0],
    ...         # 汇流参数
    ...         'K': [0.1, 10.0],
    ...         'x': [0.0, 0.5],
    ...     }
    ... )
    >>> calibrator = GenericHydrologicCalibrator(
    ...     data=data,
    ...     config=config_coupled,
    ...     runoff_model_type='hbv',
    ...     routing_model_type='muskingum'
    ... )
    """

    def __init__(
        self,
        data: CalibrationData,
        config: CalibrationConfig,
        runoff_model_type: Optional[str] = None,
        routing_model_type: Optional[str] = None,
        runoff_model_class: Optional[Type[RunoffModel]] = None,
        routing_model_class: Optional[Type[RoutingModel]] = None,
        temperature: Optional[np.ndarray] = None,
        **kwargs
    ):
        super().__init__(data, config, **kwargs)

        # 确定模型模式
        has_runoff = runoff_model_type is not None or runoff_model_class is not None
        has_routing = routing_model_type is not None or routing_model_class is not None

        if not has_runoff and not has_routing:
            raise ValueError("必须至少提供一个模型（产流或汇流）")

        if has_runoff and has_routing:
            self.mode = ModelMode.COUPLED
        elif has_runoff:
            self.mode = ModelMode.RUNOFF_ONLY
        else:
            self.mode = ModelMode.ROUTING_ONLY

        # 设置产流模型
        self.runoff_model_class = None
        self.runoff_model_type = None
        if has_runoff:
            if runoff_model_type is not None:
                if runoff_model_type not in RunoffModelConfig.REGISTRY:
                    available = list(RunoffModelConfig.REGISTRY.keys())
                    raise ValueError(
                        f"产流模型类型 '{runoff_model_type}' 未注册。"
                        f"可用模型: {available}"
                    )
                self.runoff_model_class = RunoffModelConfig.REGISTRY[runoff_model_type]
                self.runoff_model_type = runoff_model_type
            else:
                self.runoff_model_class = runoff_model_class
                self.runoff_model_type = runoff_model_class.__name__

        # 设置汇流模型
        self.routing_model_class = None
        self.routing_model_type = None
        if has_routing:
            if routing_model_type is not None:
                if routing_model_type not in RoutingModelConfig.REGISTRY:
                    available = list(RoutingModelConfig.REGISTRY.keys())
                    raise ValueError(
                        f"汇流模型类型 '{routing_model_type}' 未注册。"
                        f"可用模型: {available}"
                    )
                self.routing_model_class = RoutingModelConfig.REGISTRY[routing_model_type]
                self.routing_model_type = routing_model_type
            else:
                self.routing_model_class = routing_model_class
                self.routing_model_type = routing_model_class.__name__

        # 创建Mock Subbasin
        self._subbasin = MockSubbasin(self.data.area_km2)

        # 温度数据（某些产流模型需要）
        self.temperature = temperature
        if self.temperature is not None:
            if len(self.temperature) != self.data.n_timesteps:
                raise ValueError(
                    f"温度序列长度 ({len(self.temperature)}) "
                    f"与降雨序列长度 ({self.data.n_timesteps}) 不一致"
                )

        # 记录模型信息到元数据
        self.metadata['model_mode'] = self.mode.value
        if self.runoff_model_type:
            self.metadata['runoff_model_type'] = self.runoff_model_type
            self.metadata['runoff_model_class'] = self.runoff_model_class.__name__
        if self.routing_model_type:
            self.metadata['routing_model_type'] = self.routing_model_type
            self.metadata['routing_model_class'] = self.routing_model_class.__name__

        # 自动分离产流和汇流参数
        self._separate_parameters()

    def _separate_parameters(self):
        """自动分离产流参数和汇流参数"""
        if self.mode != ModelMode.COUPLED:
            # 非耦合模式，无需分离
            self.runoff_param_names = list(self.config.param_bounds.keys())
            self.routing_param_names = list(self.config.param_bounds.keys())
            return

        # 耦合模式：尝试自动分离参数
        # 通过检查默认参数边界来区分
        runoff_bounds = self.get_default_param_bounds(
            model_type=self.runoff_model_type,
            model_category='runoff'
        )
        routing_bounds = self.get_default_param_bounds(
            model_type=self.routing_model_type,
            model_category='routing'
        )

        self.runoff_param_names = []
        self.routing_param_names = []

        for param_name in self.config.param_bounds.keys():
            if param_name in runoff_bounds:
                self.runoff_param_names.append(param_name)
            elif param_name in routing_bounds:
                self.routing_param_names.append(param_name)
            else:
                # 无法自动识别，放入产流参数
                print(f"警告: 参数 '{param_name}' 无法自动分类，默认为产流参数")
                self.runoff_param_names.append(param_name)

    def create_model(self, params: Dict[str, Any]) -> Union[RunoffModel, RoutingModel, tuple]:
        """创建水文模型实例

        Parameters
        ----------
        params : dict
            模型参数（待率定参数）

        Returns
        -------
        model : RunoffModel, RoutingModel, or tuple
            模型实例（耦合模式返回(runoff_model, routing_model)）
        """
        # 合并待率定参数和固定参数
        full_params = {**self.config.fixed_params, **params}

        if self.mode == ModelMode.RUNOFF_ONLY:
            return self._create_runoff_model(full_params)
        elif self.mode == ModelMode.ROUTING_ONLY:
            return self._create_routing_model(full_params)
        else:  # COUPLED
            # 分离参数
            runoff_params = {k: v for k, v in full_params.items()
                           if k in self.runoff_param_names or k in self.config.fixed_params}
            routing_params = {k: v for k, v in full_params.items()
                            if k in self.routing_param_names or k in self.config.fixed_params}

            runoff_model = self._create_runoff_model(runoff_params)
            routing_model = self._create_routing_model(routing_params)
            return (runoff_model, routing_model)

    def _create_runoff_model(self, params: Dict[str, Any]) -> RunoffModel:
        """创建产流模型"""
        try:
            return self.runoff_model_class(parameters=params)
        except Exception as e:
            raise RuntimeError(
                f"创建产流模型 {self.runoff_model_type} 失败: {str(e)}\n"
                f"参数: {params}"
            ) from e

    def _create_routing_model(self, params: Dict[str, Any]) -> RoutingModel:
        """创建汇流模型"""
        try:
            return self.routing_model_class(parameters=params)
        except Exception as e:
            raise RuntimeError(
                f"创建汇流模型 {self.routing_model_type} 失败: {str(e)}\n"
                f"参数: {params}"
            ) from e

    def run_model(self, model: Union[RunoffModel, RoutingModel, tuple]) -> np.ndarray:
        """运行水文模型

        Parameters
        ----------
        model : RunoffModel, RoutingModel, or tuple
            模型实例

        Returns
        -------
        np.ndarray
            模拟径流 (m³/s)
        """
        precipitation_list = self.data.precipitation.tolist()

        if self.mode == ModelMode.RUNOFF_ONLY:
            return self._run_runoff_model(model, precipitation_list)
        elif self.mode == ModelMode.ROUTING_ONLY:
            # 汇流模型需要输入流量（使用观测数据或简化产流）
            # 这里假设data包含inflow，否则使用降雨的简单转换
            if hasattr(self.data, 'inflow') and self.data.inflow is not None:
                inflow = self.data.inflow.tolist()
            else:
                # 简化：将降雨转换为径流（径流系数0.4）
                inflow = (self.data.precipitation * 0.4 * self.data.area_km2).tolist()
            return self._run_routing_model(model, inflow)
        else:  # COUPLED
            runoff_model, routing_model = model
            # 先运行产流
            runoff = self._run_runoff_model(runoff_model, precipitation_list)
            # 再运行汇流
            return self._run_routing_model(routing_model, runoff.tolist())

    def _run_runoff_model(self, model: RunoffModel, precipitation: List[float]) -> np.ndarray:
        """运行产流模型"""
        try:
            simulated = model.simulate(self._subbasin, precipitation)
            return np.array(simulated)
        except Exception as e:
            raise RuntimeError(
                f"运行产流模型 {self.runoff_model_type} 失败: {str(e)}"
            ) from e

    def _run_routing_model(self, model: RoutingModel, inflow: List[float]) -> np.ndarray:
        """运行汇流模型"""
        try:
            routed = model.route(self._subbasin, inflow)
            return np.array(routed)
        except Exception as e:
            raise RuntimeError(
                f"运行汇流模型 {self.routing_model_type} 失败: {str(e)}"
            ) from e

    @classmethod
    def get_default_param_bounds(
        cls,
        model_type: str,
        model_category: str = 'runoff'
    ) -> Dict[str, tuple]:
        """获取模型的默认参数边界

        Parameters
        ----------
        model_type : str
            模型类型
        model_category : str
            模型类别 ('runoff' 或 'routing')

        Returns
        -------
        dict
            参数边界字典 {param_name: (lower, upper)}
        """
        if model_category == 'runoff':
            return cls._get_runoff_default_bounds(model_type)
        else:
            return cls._get_routing_default_bounds(model_type)

    @classmethod
    def _get_runoff_default_bounds(cls, model_type: str) -> Dict[str, tuple]:
        """产流模型默认参数边界"""
        bounds = {
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
        return bounds.get(model_type.lower(), {})

    @classmethod
    def _get_routing_default_bounds(cls, model_type: str) -> Dict[str, tuple]:
        """汇流模型默认参数边界"""
        bounds = {
            'muskingum': {
                'K': (0.1, 10.0),       # 传播时间（小时）
                'x': (0.0, 0.5),        # 权重系数
            },
            'lag': {
                'lag_time': (1.0, 24.0),  # 滞后时间（小时）
            },
            'dynamic_wave': {
                'manning_n': (0.02, 0.1),  # Manning糙率系数
                'slope': (0.0001, 0.05),    # 河道坡度
            },
        }
        return bounds.get(model_type.lower(), {})

    @classmethod
    def list_available_models(cls) -> Dict[str, Any]:
        """列出所有已注册的水文模型

        Returns
        -------
        dict
            包含产流和汇流模型列表的字典
        """
        return {
            'runoff_models': RunoffModelConfig.REGISTRY.copy(),
            'routing_models': RoutingModelConfig.REGISTRY.copy(),
        }

    @classmethod
    def create_for_model(
        cls,
        data: CalibrationData,
        runoff_model_type: Optional[str] = None,
        routing_model_type: Optional[str] = None,
        param_bounds: Optional[Dict[str, tuple]] = None,
        **config_kwargs
    ) -> 'GenericHydrologicCalibrator':
        """便捷方法：创建校准器

        自动使用默认参数边界（如果未提供）

        Examples
        --------
        仅产流：
        >>> calibrator = GenericHydrologicCalibrator.create_for_model(
        ...     data=data,
        ...     runoff_model_type='xin_an_jiang'
        ... )

        产汇流组合：
        >>> calibrator = GenericHydrologicCalibrator.create_for_model(
        ...     data=data,
        ...     runoff_model_type='hbv',
        ...     routing_model_type='muskingum'
        ... )
        """
        # 如果未提供参数边界，使用默认值
        if param_bounds is None:
            param_bounds = {}
            if runoff_model_type:
                param_bounds.update(cls._get_runoff_default_bounds(runoff_model_type))
            if routing_model_type:
                param_bounds.update(cls._get_routing_default_bounds(routing_model_type))

        # 创建配置
        config = CalibrationConfig(
            param_bounds=param_bounds,
            **config_kwargs
        )

        # 创建校准器
        return cls(
            data=data,
            config=config,
            runoff_model_type=runoff_model_type,
            routing_model_type=routing_model_type
        )
