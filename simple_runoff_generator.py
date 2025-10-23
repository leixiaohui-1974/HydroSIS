#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单的观测数据生成器 - 增强非线性版本v2

关键差异（与HBV对比）：
1. ❌ 不用HBV的非线性土壤产流（beta幂函数）
2. ❌ 不用HBV的多层线性储量结构
3. ✅ 用降雨阈值效应（小雨无径流，突变性）
4. ✅ 用土壤饱和超渗机制（饱和后突然增加产流）
5. ✅ 用非线性退水（幂函数，不是线性）
6. ✅ 用随机时间延迟（峰值时间不确定）
7. ✅ 更大的随机波动（±70%~+100%）

目标：让HBV拟合时NSE降到0.75-0.85（更真实）
"""

import numpy as np
from typing import Tuple, Dict


class SimpleRunoffGenerator:
    """
    简单径流生成器 - 与HBV完全不同的结构（增强版v2）

    原理：
    1. 初损后损法 + 降雨阈值效应（突变性）
    2. 土壤饱和超渗机制（非线性产流）
    3. 非线性退水（幂函数，不是HBV的线性）
    4. 随机时间延迟（峰值时间不确定性）
    """

    def __init__(
        self,
        initial_loss: float = 15.0,        # 初期损失 (mm)
        constant_loss: float = 0.3,         # 固定损失率 (mm/h)
        runoff_coefficient: float = 0.45,   # 径流系数
        reservoir_k: float = 0.15,          # 水库系数 (1/h)
        initial_storage: float = 10.0,      # 初始储量 (mm)
        random_noise_level: float = 0.15,   # 随机噪声水平 (15%)
        random_seed: int = 42,
        # 新增非线性参数
        rainfall_threshold: float = 2.0,    # 降雨阈值 (mm/h) - 小于此值无径流
        saturation_capacity: float = 25.0,  # 土壤饱和容量 (mm)
        recession_exponent: float = 1.5,    # 退水指数 (>1非线性)
        time_delay_std: float = 2.0,        # 时间延迟标准差 (小时)
    ):
        """
        初始化简单径流生成器（增强非线性版本）

        与HBV的核心差异：
        - ❌ 无HBV的field_capacity + beta幂函数组合
        - ❌ 无HBV的多层线性水库（k0, k1, k2）
        - ✅ 有降雨阈值（HBV无此机制）
        - ✅ 有非线性退水（HBV是线性的）
        - ✅ 有随机时间延迟（HBV是确定性的）
        """
        self.initial_loss_capacity = initial_loss
        self.current_loss = 0.0
        self.constant_loss = constant_loss
        self.runoff_coefficient = runoff_coefficient
        self.reservoir_k = reservoir_k
        self.storage = initial_storage
        self.noise_level = random_noise_level

        # 新增非线性机制
        self.rainfall_threshold = rainfall_threshold
        self.saturation_capacity = saturation_capacity
        self.current_saturation = 0.0  # 当前饱和度
        self.recession_exponent = recession_exponent
        self.time_delay_std = time_delay_std
        self.delay_buffer = []  # 时间延迟缓冲区

        self.rng = np.random.RandomState(random_seed)

    def step(self, precipitation: float) -> Tuple[float, Dict]:
        """
        单时间步模拟（增强非线性版本）

        与HBV完全不同的计算逻辑
        """
        # 1. 降雨阈值效应（HBV无此机制）
        if precipitation < self.rainfall_threshold:
            # 小降雨完全被截留和蒸发，无径流
            effective_precip = 0.0
            threshold_blocked = precipitation
        else:
            effective_precip = precipitation - self.rainfall_threshold
            threshold_blocked = self.rainfall_threshold

        # 2. 初期损失（简单减法，不是HBV的非线性土壤）
        if self.current_loss < self.initial_loss_capacity:
            available_capacity = self.initial_loss_capacity - self.current_loss
            absorbed = min(effective_precip, available_capacity)
            self.current_loss += absorbed
            remaining_precip = effective_precip - absorbed
        else:
            remaining_precip = effective_precip

        # 3. 固定损失（简单常数，不是HBV的渗透机制）
        remaining_precip = max(0, remaining_precip - self.constant_loss)

        # 4. 土壤饱和超渗机制（非线性产流）
        # 与HBV的beta幂函数完全不同
        self.current_saturation += remaining_precip
        saturation_ratio = min(1.0, self.current_saturation / self.saturation_capacity)

        if saturation_ratio > 0.7:
            # 接近饱和时产流突然增加（突变性）
            saturation_boost = 1.0 + 2.0 * (saturation_ratio - 0.7) / 0.3
        else:
            saturation_boost = 1.0

        direct_runoff = remaining_precip * self.runoff_coefficient * saturation_boost

        # 土壤饱和度衰减（蒸发和深层渗透）
        self.current_saturation *= 0.95
        self.current_saturation = max(0, self.current_saturation)

        # 5. 添加随机波动（更大的随机性）
        noise = self.rng.normal(1.0, self.noise_level * 1.5)
        noise = max(0.3, min(2.0, noise))  # 扩大范围到-70%~+100%
        direct_runoff *= noise

        # 6. 随机时间延迟（HBV是确定性的）
        # 将径流加入延迟缓冲区
        delay_hours = int(max(0, self.rng.normal(1.0, self.time_delay_std)))
        self.delay_buffer.append({'runoff': direct_runoff, 'delay': delay_hours})

        # 处理延迟后的径流
        delayed_runoff = 0.0
        remaining_buffer = []
        for item in self.delay_buffer:
            if item['delay'] <= 0:
                delayed_runoff += item['runoff']
            else:
                item['delay'] -= 1
                remaining_buffer.append(item)
        self.delay_buffer = remaining_buffer

        # 7. 非线性退水（幂函数，不是HBV的线性）
        self.storage += delayed_runoff
        if self.storage > 0:
            # Q = k * S^alpha (alpha > 1 时退水更快)
            outflow = self.reservoir_k * (self.storage ** self.recession_exponent)
            outflow = min(outflow, self.storage)  # 不能超过储量
        else:
            outflow = 0.0
        self.storage -= outflow
        self.storage = max(0, self.storage)

        # 8. 慢慢恢复初损（模拟蒸发）
        if precipitation < 0.1:
            self.current_loss = max(0, self.current_loss - 0.5)

        components = {
            'precipitation': precipitation,
            'threshold_blocked': threshold_blocked,
            'initial_loss': self.current_loss,
            'saturation': self.current_saturation,
            'saturation_boost': saturation_boost,
            'direct_runoff': direct_runoff,
            'delayed_runoff': delayed_runoff,
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

    print("\n关键差异（v2增强版）:")
    print("  ❌ 无HBV的beta幂函数产流")
    print("  ❌ 无HBV的多层线性水库（upper/lower）")
    print("  ✅ 有降雨阈值效应（小雨无径流）")
    print("  ✅ 有土壤饱和超渗机制（突变性产流）")
    print("  ✅ 有非线性退水（幂函数退水）")
    print("  ✅ 有随机时间延迟（峰值不确定性）")
    print("  ✅ 有更大的随机扰动（±70%~+100%）")
