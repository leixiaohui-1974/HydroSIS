"""HBV模型校准器

实现HBV模型特定的校准逻辑。
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import BaseCalibrator, CalibrationData, CalibrationConfig
from ..runoff.hbv import HBVRunoff


class MockSubbasin:
    """模拟Subbasin对象（HBV模型需要）"""
    def __init__(self, area_km2: float):
        self.area_km2 = area_km2


class HBVCalibrator(BaseCalibrator):
    """HBV模型校准器
    
    Parameters
    ----------
    data : CalibrationData
        校准数据（必须包含temperature）
    config : CalibrationConfig
        校准配置
    output_dir : Path, optional
        输出目录
    
    Examples
    --------
    >>> data = CalibrationData(
    ...     precipitation=precip_array,
    ...     observed_runoff=obs_array,
    ...     area_km2=100.0,
    ...     temperature=temp_array
    ... )
    >>> config = CalibrationConfig(
    ...     param_bounds={
    ...         'FC': [200, 600],
    ...         'BETA': [1.0, 3.0],
    ...         'K0': [0.05, 0.4],
    ...         'K1': [0.01, 0.15],
    ...         'K2': [0.001, 0.05],
    ...         'PERC': [0.5, 5.0],
    ...     },
    ...     fixed_params={'LP': 0.7, 'TT': 0.0},
    ...     algorithm='sce_ua',
    ...     objective_metric='nse'
    ... )
    >>> calibrator = HBVCalibrator(data, config)
    >>> result = calibrator.run_calibration()
    >>> calibrator.save_results(result)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 验证HBV需要温度数据
        if self.data.temperature is None:
            print("警告: HBV模型需要温度数据，将使用默认值")
            self.data.temperature = np.ones(self.data.n_timesteps) * 10.0
        
        # 创建Mock Subbasin
        self._subbasin = MockSubbasin(self.data.area_km2)
    
    def create_model(self, params: Dict[str, Any]) -> HBVRunoff:
        """创建HBV模型实例
        
        Parameters
        ----------
        params : dict
            模型参数（待率定参数）
        
        Returns
        -------
        HBVRunoff
            HBV模型实例
        """
        # 合并待率定参数和固定参数
        full_params = {**self.config.fixed_params, **params}
        
        # 处理参数名称映射（兼容不同命名约定）
        param_mapping = {
            'field_capacity': 'FC',
            'beta': 'BETA',
            'k0': 'K0',
            'k1': 'K1',
            'k2': 'K2',
            'percolation': 'PERC',
            'lp': 'LP',
            'maxbas': 'MAXBAS',
            'tt': 'TT',
            'cfmax': 'CFMAX',
            'cfr': 'CFR',
            'cwh': 'CWH',
        }
        
        standardized_params = {}
        for key, value in full_params.items():
            # 转换为小写进行匹配
            key_lower = key.lower()
            # 如果在映射中，使用标准名称，否则保持原样
            standard_key = param_mapping.get(key_lower, key)
            standardized_params[standard_key] = value
        
        # 处理初始状态参数
        # 如果有initial_soil_ratio，转换为absolute initial_soil
        if 'initial_soil_ratio' in params:
            FC = standardized_params.get('FC', standardized_params.get('field_capacity', 300.0))
            standardized_params['initial_soil'] = params['initial_soil_ratio'] * FC
            standardized_params.pop('initial_soil_ratio', None)
        
        # 设置默认初始状态（如果未提供）
        if 'initial_soil' not in standardized_params:
            FC = standardized_params.get('FC', 300.0)
            standardized_params['initial_soil'] = FC * 0.7
        
        if 'initial_upper' not in standardized_params:
            standardized_params['initial_upper'] = 50.0
        
        if 'initial_lower' not in standardized_params:
            standardized_params['initial_lower'] = 30.0
        
        if 'initial_snow' not in standardized_params:
            standardized_params['initial_snow'] = 0.0
        
        # 创建HBV模型
        return HBVRunoff(parameters=standardized_params)
    
    def run_model(self, model: HBVRunoff) -> np.ndarray:
        """运行HBV模型
        
        Parameters
        ----------
        model : HBVRunoff
            HBV模型实例
        
        Returns
        -------
        np.ndarray
            模拟径流 (m³/s)
        """
        # HBV.simulate 需要降雨列表
        precipitation_list = self.data.precipitation.tolist()
        
        # 运行模型（HBV已经返回m³/s）
        simulated = model.simulate(self._subbasin, precipitation_list)
        
        return np.array(simulated)
    
    @classmethod
    def from_workflow_config(
        cls,
        zone_id: int,
        workflow_config: Dict,
        calibration_config: Dict,
        output_dir: str = None
    ) -> HBVCalibrator:
        """从工作流配置创建校准器
        
        Parameters
        ----------
        zone_id : int
            分区ID
        workflow_config : dict
            工作流配置
        calibration_config : dict
            校准配置
        output_dir : str, optional
            输出目录
        
        Returns
        -------
        HBVCalibrator
            配置好的校准器实例
        """
        from pathlib import Path
        import pandas as pd
        
        # 加载降雨数据
        base_dir = Path(workflow_config['directories']['base_results'])
        precip_file = base_dir / workflow_config['steps']['step_08_areal_rainfall']['output']['areal_precipitation']
        precip_df = pd.read_csv(precip_file, index_col=0)
        
        # 提取该分区的降雨
        # 这里需要根据实际的分区-子流域映射来计算
        # 简化处理：假设已经有分区平均降雨
        precipitation = precip_df.mean(axis=1).values
        
        # 加载观测径流
        obs_file = base_dir / f"estimated_observations/zone_{zone_id}_estimated_runoff.csv"
        obs_df = pd.read_csv(obs_file)
        observed_runoff = obs_df['discharge_m3s'].values
        times = pd.to_datetime(obs_df['datetime'])
        
        # 生成温度（简化）
        temperature = 10 + 8 * np.sin(np.arange(len(precipitation)) / 30)
        
        # 获取流域面积
        area_km2 = workflow_config['zones'][str(zone_id)]['area_km2']
        
        # 创建数据对象
        data = CalibrationData(
            precipitation=precipitation,
            observed_runoff=observed_runoff,
            area_km2=area_km2,
            temperature=temperature,
            times=times
        )
        
        # 创建配置对象
        config = CalibrationConfig(
            param_bounds=calibration_config['param_bounds'],
            fixed_params=calibration_config.get('fixed_params', {}),
            algorithm=calibration_config.get('algorithm', 'sce_ua'),
            algorithm_params=calibration_config.get('algorithm_params', {}),
            objective_metric=calibration_config.get('objective_metric', 'nse'),
            warmup_steps=calibration_config.get('warmup_steps', 0),
            seed=calibration_config.get('seed')
        )
        
        # 创建校准器
        if output_dir is None:
            output_dir = base_dir / f"calibration/zone_{zone_id}"
        
        return cls(data, config, output_dir=Path(output_dir))
