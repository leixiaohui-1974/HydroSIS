#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单的观测数据生成器 - 删除HBV特征

关键差异：
1. ❌ 不用HBV的非线性土壤产流（beta幂函数）
2. ❌ 不用HBV的多层储量结构
3. ✅ 用简单的初损后损法 + 固定径流系数
4. ✅ 用单一线性水库汇流（不是HBV的三层）
5. ✅ 添加更多的随机波动

目标：让HBV拟合时NSE降到0.75-0.85（更真实）
"""

import numpy as np
from typing import Tuple, Dict


class SimpleRunoffGenerator:
    """
    简单径流生成器 - 与HBV完全不同的结构

    原理：
    1. 初损后损法：
       - 初期损失：前面的降雨被吸收
       - 后期损失：固定的渗透率
       - 径流 = max(0, 降雨 - 初损 - 后损)

    2. 单一线性水库汇流

    3. 随机扰动（模拟小尺度异质性）
    """

    def __init__(
        self,
        initial_loss: float = 15.0,        # 初期损失 (mm)
        constant_loss: float = 0.3,         # 固定损失率 (mm/h)
        runoff_coefficient: float = 0.45,   # 径流系数
        reservoir_k: float = 0.15,          # 线性水库系数 (1/h)
        initial_storage: float = 10.0,      # 初始储量 (mm)
        random_noise_level: float = 0.15,   # 随机噪声水平 (15%)
        random_seed: int = 42,
    ):
        """
        初始化简单径流生成器

        完全不同于HBV的参数：
        - 没有field_capacity, beta (HBV核心参数)
        - 没有k0, k1, k2 (HBV多层水库)
        - 没有percolation (HBV特有)
        """
        self.initial_loss_capacity = initial_loss
        self.current_loss = 0.0  # 当前累积损失

        self.constant_loss = constant_loss
        self.runoff_coefficient = runoff_coefficient
        self.reservoir_k = reservoir_k
        self.storage = initial_storage

        self.noise_level = random_noise_level
        self.rng = np.random.RandomState(random_seed)

    def step(self, precipitation: float) -> Tuple[float, Dict]:
        """
        单时间步模拟

        完全不同于HBV的计算逻辑
        """
        # 1. 初期损失（简单减法，不是HBV的非线性土壤）
        if self.current_loss < self.initial_loss_capacity:
            # 还在填充初损
            available_capacity = self.initial_loss_capacity - self.current_loss
            absorbed = min(precipitation, available_capacity)
            self.current_loss += absorbed
            remaining_precip = precipitation - absorbed
        else:
            # 初损已满
            remaining_precip = precipitation

        # 2. 固定损失（简单常数，不是HBV的渗透和蒸散发机制）
        remaining_precip = max(0, remaining_precip - self.constant_loss)

        # 3. 固定径流系数（简单乘法，不是HBV的状态依赖产流）
        direct_runoff = remaining_precip * self.runoff_coefficient

        # 4. 添加随机波动（模拟小尺度过程）
        noise = self.rng.normal(1.0, self.noise_level)
        noise = max(0.5, min(1.5, noise))  # 限制在±50%
        direct_runoff *= noise

        # 5. 单一线性水库汇流（不是HBV的多层水库）
        self.storage += direct_runoff
        outflow = self.reservoir_k * self.storage
        self.storage -= outflow
        self.storage = max(0, self.storage)

        # 6. 慢慢恢复初损（模拟蒸发）
        if precipitation < 0.1:  # 无雨时段
            self.current_loss = max(0, self.current_loss - 0.5)  # 缓慢蒸发

        components = {
            'precipitation': precipitation,
            'initial_loss': self.current_loss,
            'direct_runoff': direct_runoff,
            'storage': self.storage,
            'outflow': outflow,
            'noise_factor': noise,
        }

        return outflow, components

    def generate(self, precipitation_series: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        生成完整径流序列
        """
        n = len(precipitation_series)
        runoff = np.zeros(n)

        for i in range(n):
            runoff[i], _ = self.step(precipitation_series[i])

        # 计算统计
        total_precip = precipitation_series.sum()
        total_runoff = runoff.sum()
        runoff_coeff = total_runoff / total_precip if total_precip > 0 else 0

        stats = {
            'total_precip_mm': total_precip,
            'total_runoff_mm': total_runoff,
            'runoff_coefficient': runoff_coeff,
            'mean_runoff': runoff.mean(),
            'peak_runoff': runoff.max(),
        }

        return runoff, stats


def add_observation_errors(runoff_true: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    添加更多的观测误差

    模拟真实的流量观测问题：
    1. 测量误差（10%随机噪声，比之前的5%更大）
    2. 系统偏差（5%高估）
    3. 评级曲线不确定性（流量依赖的误差）
    4. 数据缺失和插值误差
    """
    rng = np.random.RandomState(seed)
    n = len(runoff_true)

    # 1. 基础测量误差（10%）
    noise = rng.normal(1.0, 0.10, n)
    observed = runoff_true * noise

    # 2. 系统偏差（5%）
    observed *= 1.05

    # 3. 流量依赖的误差（高流量误差更大）
    flow_dependent_error = 1.0 + 0.15 * (runoff_true / runoff_true.max())
    observed *= flow_dependent_error

    # 4. 随机尖峰（仪器故障）
    n_spikes = int(n * 0.02)  # 2%的数据点有尖峰
    spike_indices = rng.choice(n, n_spikes, replace=False)
    observed[spike_indices] *= rng.uniform(0.5, 1.8, n_spikes)

    # 5. 数据缺失（10%）
    missing_rate = 0.10
    missing_indices = rng.choice(n, int(n * missing_rate), replace=False)
    for idx in missing_indices:
        if 0 < idx < n - 1:
            # 简单线性插值
            observed[idx] = (observed[idx-1] + observed[idx+1]) / 2

    # 确保非负
    observed = np.maximum(observed, 0)

    return observed


if __name__ == '__main__':
    # 测试
    print("简单径流生成器测试")
    print("=" * 60)

    # 生成测试降雨
    rainfall = np.array([0, 0, 5, 10, 15, 8, 3, 1, 0, 0])

    generator = SimpleRunoffGenerator()
    runoff, stats = generator.generate(rainfall)

    print("\n降雨:", rainfall)
    print("径流:", np.round(runoff, 2))
    print("\n统计:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")

    print("\n关键差异:")
    print("  ❌ 无HBV的beta幂函数产流")
    print("  ❌ 无HBV的多层水库（upper/lower）")
    print("  ✅ 使用初损后损法")
    print("  ✅ 使用固定径流系数")
    print("  ✅ 添加随机扰动")
