#!/usr/bin/env python3
"""
增强型径流生成器

相比简单的线性径流系数法，该生成器具有：
1. 土壤水分核算（类似HBV的FC/BETA概念）
2. 多分量径流分离（快速径流、中速径流、基流）
3. 线性水库汇流
4. 与HBV模型更兼容的物理过程

设计理念：
- 保持相对简单，但引入必要的非线性和状态依赖
- 可以通过参数调整生成不同响应特性的径流
- 适合用于生成"观测数据"以验证HBV率定系统
"""
import numpy as np
from typing import Dict, List, Tuple


class EnhancedRunoffGenerator:
    """
    增强型径流生成器

    主要特点：
    1. 土壤水分核算：类似HBV的土壤层，有最大容量和非线性产流
    2. 三分量径流：
       - 快速径流（地表径流）：当土壤接近饱和时产生
       - 中速径流（壤中流）：土壤中层排水
       - 慢速径流（基流）：深层渗透
    3. 线性水库汇流：每个分量使用独立的线性水库
    """

    def __init__(
        self,
        # 土壤参数
        soil_capacity: float = 300.0,      # 土壤最大蓄水容量 (mm)
        soil_beta: float = 2.0,             # 土壤蓄水曲线指数 (无量纲)

        # 径流分配参数
        fast_threshold: float = 0.7,        # 快速径流阈值（土壤湿度比例）
        fast_ratio: float = 0.4,            # 快速径流系数
        inter_ratio: float = 0.35,          # 中速径流系数
        base_ratio: float = 0.25,           # 基流系数

        # 水库参数（单位：1/小时）
        k_fast: float = 0.3,                # 快速水库退水系数
        k_inter: float = 0.08,              # 中速水库退水系数
        k_base: float = 0.02,               # 基流水库退水系数

        # 初始状态
        initial_soil: float = 150.0,        # 初始土壤水分 (mm)
        initial_fast: float = 5.0,          # 初始快速水库 (mm)
        initial_inter: float = 10.0,        # 初始中速水库 (mm)
        initial_base: float = 20.0,         # 初始基流水库 (mm)

        # 蒸散发
        et_rate: float = 0.1,               # 蒸散发速率 (mm/h)

        time_step_hours: float = 1.0        # 时间步长（小时）
    ):
        """
        初始化增强型径流生成器

        Parameters
        ----------
        soil_capacity : float
            土壤最大蓄水容量 (mm)，类似HBV的FC参数
        soil_beta : float
            土壤蓄水曲线指数，控制产流的非线性程度
            - beta=1.0: 线性
            - beta>1.0: 非线性，土壤越湿产流越多
        fast_threshold : float
            快速径流阈值（相对土壤湿度），超过此值开始产生地表径流
        fast_ratio : float
            快速径流分配系数
        inter_ratio : float
            中速径流分配系数
        base_ratio : float
            基流分配系数
        k_fast, k_inter, k_base : float
            各水库的退水系数（1/小时），值越大退水越快
        initial_* : float
            初始状态值
        et_rate : float
            蒸散发速率 (mm/h)
        time_step_hours : float
            时间步长（小时）
        """
        # 土壤参数
        self.soil_capacity = soil_capacity
        self.soil_beta = soil_beta

        # 径流分配
        self.fast_threshold = fast_threshold
        self.fast_ratio = fast_ratio
        self.inter_ratio = inter_ratio
        self.base_ratio = base_ratio

        # 确保分配系数和为1
        total_ratio = fast_ratio + inter_ratio + base_ratio
        if abs(total_ratio - 1.0) > 0.01:
            # 归一化
            self.fast_ratio = fast_ratio / total_ratio
            self.inter_ratio = inter_ratio / total_ratio
            self.base_ratio = base_ratio / total_ratio

        # 水库参数
        self.k_fast = k_fast
        self.k_inter = k_inter
        self.k_base = k_base

        # 蒸散发
        self.et_rate = et_rate

        # 时间步长
        self.dt = time_step_hours

        # 初始状态
        self.soil_storage = initial_soil
        self.fast_storage = initial_fast
        self.inter_storage = initial_inter
        self.base_storage = initial_base

    def step(self, precipitation_mm_h: float) -> Tuple[float, Dict[str, float]]:
        """
        单时间步模拟

        Parameters
        ----------
        precipitation_mm_h : float
            降雨强度 (mm/h)

        Returns
        -------
        runoff_mm_h : float
            总径流深度 (mm/h)
        components : dict
            各分量径流和状态变量
        """
        # 1. 有效降雨（扣除蒸散发）
        effective_precip = max(0, precipitation_mm_h - self.et_rate)

        # 2. 土壤水分更新和产流计算
        # 使用类似HBV的非线性土壤蓄水曲线
        soil_saturation = self.soil_storage / self.soil_capacity
        soil_saturation = max(0.0, min(1.0, soil_saturation))

        # 产流量（使用幂函数关系）
        if effective_precip > 0:
            # 土壤越湿，产流越多
            runoff_fraction = soil_saturation ** self.soil_beta
            direct_runoff = effective_precip * runoff_fraction
            infiltration = effective_precip * (1 - runoff_fraction)

            # 更新土壤水分
            self.soil_storage += infiltration * self.dt

            # 土壤容量限制（超出部分作为超渗径流）
            if self.soil_storage > self.soil_capacity:
                excess = self.soil_storage - self.soil_capacity
                self.soil_storage = self.soil_capacity
                direct_runoff += excess / self.dt
        else:
            direct_runoff = 0

        # 3. 径流分量分配
        # 根据土壤湿度调整分配比例
        if soil_saturation > self.fast_threshold:
            # 土壤湿度高时，快速径流比例增加
            excess_saturation = (soil_saturation - self.fast_threshold) / (1.0 - self.fast_threshold)
            adjusted_fast_ratio = self.fast_ratio + (1 - self.fast_ratio) * excess_saturation * 0.5
            adjusted_inter_ratio = self.inter_ratio * (1 - excess_saturation * 0.3)
            adjusted_base_ratio = 1.0 - adjusted_fast_ratio - adjusted_inter_ratio
        else:
            adjusted_fast_ratio = self.fast_ratio
            adjusted_inter_ratio = self.inter_ratio
            adjusted_base_ratio = self.base_ratio

        # 分配到各水库
        fast_input = direct_runoff * adjusted_fast_ratio
        inter_input = direct_runoff * adjusted_inter_ratio
        base_input = direct_runoff * adjusted_base_ratio

        # 4. 线性水库汇流
        # dS/dt = Input - k*S
        # S(t+dt) = S(t) + dt*(Input - k*S(t))
        # 使用解析解: S(t+dt) = S(t)*exp(-k*dt) + Input/k*(1-exp(-k*dt))

        # 快速水库
        self.fast_storage = self._linear_reservoir_step(
            self.fast_storage, fast_input, self.k_fast, self.dt
        )
        fast_outflow = self.k_fast * self.fast_storage

        # 中速水库
        self.inter_storage = self._linear_reservoir_step(
            self.inter_storage, inter_input, self.k_inter, self.dt
        )
        inter_outflow = self.k_inter * self.inter_storage

        # 基流水库
        self.base_storage = self._linear_reservoir_step(
            self.base_storage, base_input, self.k_base, self.dt
        )
        base_outflow = self.k_base * self.base_storage

        # 5. 总径流
        total_runoff_mm_h = fast_outflow + inter_outflow + base_outflow

        # 6. 土壤蒸散发（从土壤储量中扣除）
        soil_et = self.et_rate * self.dt * soil_saturation
        self.soil_storage = max(0, self.soil_storage - soil_et)

        # 返回结果
        components = {
            'fast_runoff': fast_outflow,
            'inter_runoff': inter_outflow,
            'base_runoff': base_outflow,
            'soil_storage': self.soil_storage,
            'soil_saturation': self.soil_storage / self.soil_capacity,
            'fast_storage': self.fast_storage,
            'inter_storage': self.inter_storage,
            'base_storage': self.base_storage,
            'effective_precip': effective_precip,
            'direct_runoff': direct_runoff,
        }

        return total_runoff_mm_h, components

    def _linear_reservoir_step(
        self, storage: float, input_rate: float, k: float, dt: float
    ) -> float:
        """
        线性水库单步更新（解析解）

        dS/dt = I - k*S
        S(t+dt) = S(t)*exp(-k*dt) + I/k*(1-exp(-k*dt))
        """
        exp_kdt = np.exp(-k * dt)
        if k > 1e-10:
            new_storage = storage * exp_kdt + input_rate / k * (1 - exp_kdt)
        else:
            # k很小时使用一阶近似
            new_storage = storage + dt * (input_rate - k * storage)
        return max(0, new_storage)

    def generate(
        self,
        precipitation_series: np.ndarray,
        area_km2: float,
        return_components: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """
        生成完整的径流时间序列

        Parameters
        ----------
        precipitation_series : np.ndarray
            降雨时间序列 (mm/h)
        area_km2 : float
            流域面积 (km²)
        return_components : bool
            是否返回各分量时间序列

        Returns
        -------
        runoff_m3s : np.ndarray
            径流时间序列 (m³/s)
        stats : dict
            统计信息和分量时间序列（如果 return_components=True）
        """
        n_steps = len(precipitation_series)
        runoff_mm_h = np.zeros(n_steps)

        # 如果需要，保存各分量
        if return_components:
            components_series = {
                'fast_runoff': np.zeros(n_steps),
                'inter_runoff': np.zeros(n_steps),
                'base_runoff': np.zeros(n_steps),
                'soil_storage': np.zeros(n_steps),
                'soil_saturation': np.zeros(n_steps),
            }

        # 逐步模拟
        for i in range(n_steps):
            runoff, comp = self.step(precipitation_series[i])
            runoff_mm_h[i] = runoff

            if return_components:
                components_series['fast_runoff'][i] = comp['fast_runoff']
                components_series['inter_runoff'][i] = comp['inter_runoff']
                components_series['base_runoff'][i] = comp['base_runoff']
                components_series['soil_storage'][i] = comp['soil_storage']
                components_series['soil_saturation'][i] = comp['soil_saturation']

        # 转换为m³/s
        # 1 mm/h × 1 km² = 1000 m³/h = 1000/3600 m³/s
        runoff_m3s = runoff_mm_h * area_km2 / 3.6

        # 计算统计信息
        total_precip_mm = precipitation_series.sum() * self.dt
        total_runoff_mm = runoff_mm_h.sum() * self.dt
        runoff_coefficient = total_runoff_mm / total_precip_mm if total_precip_mm > 0 else 0

        stats = {
            'total_precip_mm': total_precip_mm,
            'total_runoff_mm': total_runoff_mm,
            'runoff_coefficient': runoff_coefficient,
            'mean_runoff_m3s': runoff_m3s.mean() if len(runoff_m3s) > 0 else 0.0,
            'peak_runoff_m3s': runoff_m3s.max() if len(runoff_m3s) > 0 else 0.0,
            'min_runoff_m3s': runoff_m3s.min() if len(runoff_m3s) > 0 else 0.0,
        }

        if return_components:
            stats['components'] = components_series

        return runoff_m3s, stats

    def reset(
        self,
        initial_soil: float = None,
        initial_fast: float = None,
        initial_inter: float = None,
        initial_base: float = None
    ):
        """
        重置模型状态
        """
        if initial_soil is not None:
            self.soil_storage = initial_soil
        if initial_fast is not None:
            self.fast_storage = initial_fast
        if initial_inter is not None:
            self.inter_storage = initial_inter
        if initial_base is not None:
            self.base_storage = initial_base

    def get_parameters(self) -> Dict[str, float]:
        """
        获取所有参数
        """
        return {
            'soil_capacity': self.soil_capacity,
            'soil_beta': self.soil_beta,
            'fast_threshold': self.fast_threshold,
            'fast_ratio': self.fast_ratio,
            'inter_ratio': self.inter_ratio,
            'base_ratio': self.base_ratio,
            'k_fast': self.k_fast,
            'k_inter': self.k_inter,
            'k_base': self.k_base,
            'et_rate': self.et_rate,
            'time_step_hours': self.dt,
        }
