"""新安江(XinAnJiang)模型校准器

实现新安江模型特定的校准逻辑。
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import BaseCalibrator, CalibrationData, CalibrationConfig
from ..runoff.xinanjiang import XinAnJiangRunoff


class MockSubbasin:
    """模拟Subbasin对象（XinAnJiang模型需要）"""
    def __init__(self, area_km2: float):
        self.area_km2 = area_km2


class XinAnJiangCalibrator(BaseCalibrator):
    """新安江模型校准器

    新安江模型是中国最经典的产流模型之一，采用蓄满产流机制，
    适用于湿润和半湿润地区的流域水文模拟。

    Parameters
    ----------
    data : CalibrationData
        校准数据（降雨和观测径流）
    config : CalibrationConfig
        校准配置
    output_dir : Path, optional
        输出目录

    Examples
    --------
    >>> data = CalibrationData(
    ...     precipitation=precip_array,
    ...     observed_runoff=obs_array,
    ...     area_km2=100.0
    ... )
    >>> config = CalibrationConfig(
    ...     param_bounds={
    ...         'wm': [50, 250],        # 张力水容量
    ...         'b': [0.1, 0.5],        # 蓄水容量曲线指数
    ...         'imp': [0.0, 0.3],      # 不透水面积比例
    ...         'recession': [0.3, 0.9] # 地下水消退系数
    ...     },
    ...     algorithm='differential_evolution',
    ...     objective_metric='nse'
    ... )
    >>> calibrator = XinAnJiangCalibrator(data, config)
    >>> result = calibrator.run_calibration()
    >>> calibrator.save_results(result)

    Notes
    -----
    新安江模型主要参数：
    - wm: 张力水容量 (mm)，控制土壤蓄水能力
    - b: 蓄水容量曲线指数，控制蓄水容量的空间分布
    - imp: 不透水面积比例 (0-1)，直接产流部分
    - recession: 地下水消退系数 (0-1)，控制基流退水速度

    References
    ----------
    .. [1] Zhao, R.J. (1992). The Xinanjiang model applied in China.
           Journal of Hydrology, 135(1-4), 371-381.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 创建Mock Subbasin
        self._subbasin = MockSubbasin(self.data.area_km2)

    def create_model(self, params: Dict[str, Any]) -> XinAnJiangRunoff:
        """创建新安江模型实例

        Parameters
        ----------
        params : dict
            模型参数（待率定参数）

        Returns
        -------
        XinAnJiangRunoff
            新安江模型实例
        """
        # 合并待率定参数和固定参数
        full_params = {**self.config.fixed_params, **params}

        # 设置默认初始状态（如果未提供）
        if 'initial_tension_water' not in full_params:
            wm = full_params.get('wm', 150.0)
            full_params['initial_tension_water'] = wm * 0.5  # 初始为50%饱和度

        if 'initial_groundwater' not in full_params:
            full_params['initial_groundwater'] = 0.0

        # 创建新安江模型
        return XinAnJiangRunoff(parameters=full_params)

    def run_model(self, model: XinAnJiangRunoff) -> np.ndarray:
        """运行新安江模型

        Parameters
        ----------
        model : XinAnJiangRunoff
            新安江模型实例

        Returns
        -------
        np.ndarray
            模拟径流 (m³/s)
        """
        # XinAnJiang.simulate 需要降雨列表
        precipitation_list = self.data.precipitation.tolist()

        # 运行模型（返回m³/s）
        simulated = model.simulate(self._subbasin, precipitation_list)

        return np.array(simulated)

    @classmethod
    def get_default_param_bounds(cls) -> Dict[str, tuple]:
        """获取新安江模型的默认参数边界

        Returns
        -------
        dict
            参数边界字典 {param_name: (lower, upper)}

        Notes
        -----
        基于文献和实践经验的推荐参数范围：
        - wm: 50-250 mm (湿润地区偏大，干旱地区偏小)
        - b: 0.1-0.5 (控制非均匀性，平原地区偏小，山区偏大)
        - imp: 0.0-0.3 (不透水面积，城市化地区偏大)
        - recession: 0.3-0.9 (地下水退水系数，岩溶地区偏小)
        """
        return {
            'wm': (50.0, 250.0),           # 张力水容量 (mm)
            'b': (0.1, 0.5),               # 蓄水容量曲线指数
            'imp': (0.0, 0.3),             # 不透水面积比例
            'recession': (0.3, 0.9),       # 地下水消退系数
        }

    @classmethod
    def from_workflow_config(
        cls,
        zone_id: int,
        workflow_config: Dict,
        calibration_config: Dict,
        output_dir: str = None
    ) -> XinAnJiangCalibrator:
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
        XinAnJiangCalibrator
            配置好的校准器实例
        """
        from pathlib import Path
        import pandas as pd

        # 加载降雨数据
        base_dir = Path(workflow_config['directories']['base_results'])
        precip_file = base_dir / workflow_config['steps']['step_08_areal_rainfall']['output']['areal_precipitation']
        precip_df = pd.read_csv(precip_file, index_col=0)

        # 提取该分区的降雨
        precipitation = precip_df.mean(axis=1).values

        # 加载观测径流
        obs_file = base_dir / f"estimated_observations/zone_{zone_id}_estimated_runoff.csv"
        obs_df = pd.read_csv(obs_file)
        observed_runoff = obs_df['discharge_m3s'].values
        times = pd.to_datetime(obs_df['datetime'])

        # 获取流域面积
        area_km2 = workflow_config['zones'][str(zone_id)]['area_km2']

        # 创建数据对象
        data = CalibrationData(
            precipitation=precipitation,
            observed_runoff=observed_runoff,
            area_km2=area_km2,
            times=times
        )

        # 创建配置对象
        config = CalibrationConfig(
            param_bounds=calibration_config['param_bounds'],
            fixed_params=calibration_config.get('fixed_params', {}),
            algorithm=calibration_config.get('algorithm', 'differential_evolution'),
            algorithm_params=calibration_config.get('algorithm_params', {}),
            objective_metric=calibration_config.get('objective_metric', 'nse'),
            warmup_steps=calibration_config.get('warmup_steps', 0),
            seed=calibration_config.get('seed')
        )

        # 创建校准器
        if output_dir is None:
            output_dir = base_dir / f"calibration/zone_{zone_id}"

        return cls(data, config, output_dir=Path(output_dir))
