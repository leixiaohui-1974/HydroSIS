#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
流域分区逐级参数率定

⚠️ 注意：这是一个高级研究工具脚本
本脚本实现了复杂的多分区流域校准策略，适用于专业用户和研究场景。
普通用户请使用已重构的简化版本脚本。

参考: scripts/calibration/README_ADVANCED_SCRIPTS.md

策略：
1. 确定流域拓扑顺序（上游到下游）
2. 为每个分区生成"观测数据"（使用增强简化模型v2）
3. 从上游到下游逐级率定HBV参数
4. 上游分区率定后，结果作为下游分区的输入

流域结构：
P3 → P6 → P2 → P4 → P1 (主干)
                 P5 → P1 (支流)

优点：简单直观，计算效率高
缺点：误差可能累积
"""

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from typing import Dict, List, Tuple

from simple_runoff_generator import SimpleRunoffGenerator, add_observation_errors


# ============================================================================
# 流域拓扑结构定义
# ============================================================================

WATERSHED_TOPOLOGY = {
    'zones': ['P3', 'P6', 'P2', 'P5', 'P4', 'P1'],
    'connections': {
        'P3': {'downstream': 'P6', 'junction': None},
        'P6': {'downstream': 'P2', 'junction': None},
        'P2': {'downstream': 'P4', 'junction': None},
        'P5': {'downstream': 'P1', 'junction': 'P4'},  # P5和P4都汇入P1
        'P4': {'downstream': 'P1', 'junction': None},
        'P1': {'downstream': None, 'junction': None},  # 流域出口
    },
    'descriptions': {
        'P3': '最上游分区',
        'P6': '中上游分区（接收P3）',
        'P2': '中游分区（接收P6）',
        'P5': '独立支流',
        'P4': '中下游分区（接收P2）',
        'P1': '出口分区（接收P4和P5）',
    }
}


# ============================================================================
# HBV模型
# ============================================================================

def run_hbv_model(rainfall: np.ndarray, params: dict) -> np.ndarray:
    """运行HBV模型"""
    snow = params.get('initial_snow', 0.0)
    soil = params.get('initial_soil', 0.0)
    upper = params.get('initial_upper', 0.0)
    lower = params.get('initial_lower', 0.0)

    degree_day_factor = params.get('degree_day_factor', 3.0)
    snow_threshold = params.get('snow_threshold', 0.0)
    field_capacity = params.get('field_capacity', 80.0)
    beta = max(1e-6, params.get('beta', 1.0))
    k0 = params.get('k0', 0.12)
    k1 = params.get('k1', 0.08)
    k2 = params.get('k2', 0.02)
    percolation = params.get('percolation', 1.0)

    runoff_list = []

    for p in rainfall:
        rainfall_step = max(0.0, p - snow_threshold)
        snowfall = max(0.0, p - rainfall_step)
        snow += snowfall

        melt = degree_day_factor * max(0.0, rainfall_step - snow_threshold)
        melt = min(melt, snow)
        snow -= melt

        effective_precip = rainfall_step + melt
        soil_deficit = max(0.0, field_capacity - soil)

        if soil > 0 and field_capacity > 0:
            recharge = effective_precip * ((soil / field_capacity) ** beta)
        else:
            recharge = 0.0

        recharge = min(recharge, soil_deficit)
        soil += effective_precip - recharge

        quickflow = k0 * upper
        actual_percolation = min(percolation, max(0.0, upper + recharge - quickflow))
        upper += recharge - quickflow - actual_percolation
        upper = max(0.0, upper)

        lower += actual_percolation - k2 * lower
        lower = max(0.0, lower)

        baseflow = k1 * upper + k2 * lower
        total_runoff = quickflow + baseflow

        runoff_list.append(total_runoff)

    return np.array(runoff_list)


# ============================================================================
# 分区观测数据生成
# ============================================================================

def generate_zone_observations(
    rainfall: np.ndarray,
    zone_id: str,
    upstream_inflow: np.ndarray = None,
    random_seed: int = 42
) -> Dict:
    """
    为指定分区生成观测数据

    参数：
        rainfall: 分区降雨序列
        zone_id: 分区ID
        upstream_inflow: 上游入流（如果有）
        random_seed: 随机种子

    返回：
        包含观测数据和统计信息的字典
    """
    print(f"\n  为分区 {zone_id} 生成观测数据:")
    print(f"    描述: {WATERSHED_TOPOLOGY['descriptions'][zone_id]}")

    # 每个分区使用不同的随机种子
    zone_seed = random_seed + hash(zone_id) % 1000

    # 每个分区的参数略有不同（空间异质性）
    zone_params = {
        'P3': {
            'initial_loss': 15.0,
            'constant_loss': 0.35,
            'runoff_coefficient': 0.40,
            'reservoir_k': 0.16,
            'rainfall_threshold': 2.0,
            'saturation_capacity': 28.0,
            'recession_exponent': 1.6,
            'time_delay_std': 2.5,
        },
        'P6': {
            'initial_loss': 18.0,
            'constant_loss': 0.40,
            'runoff_coefficient': 0.42,
            'reservoir_k': 0.18,
            'rainfall_threshold': 2.5,
            'saturation_capacity': 30.0,
            'recession_exponent': 1.8,
            'time_delay_std': 3.0,
        },
        'P2': {
            'initial_loss': 20.0,
            'constant_loss': 0.45,
            'runoff_coefficient': 0.45,
            'reservoir_k': 0.20,
            'rainfall_threshold': 2.8,
            'saturation_capacity': 32.0,
            'recession_exponent': 1.9,
            'time_delay_std': 3.5,
        },
        'P5': {
            'initial_loss': 16.0,
            'constant_loss': 0.38,
            'runoff_coefficient': 0.43,
            'reservoir_k': 0.17,
            'rainfall_threshold': 2.2,
            'saturation_capacity': 29.0,
            'recession_exponent': 1.7,
            'time_delay_std': 2.8,
        },
        'P4': {
            'initial_loss': 17.0,
            'constant_loss': 0.42,
            'runoff_coefficient': 0.44,
            'reservoir_k': 0.19,
            'rainfall_threshold': 2.6,
            'saturation_capacity': 31.0,
            'recession_exponent': 1.85,
            'time_delay_std': 3.2,
        },
        'P1': {
            'initial_loss': 19.0,
            'constant_loss': 0.43,
            'runoff_coefficient': 0.46,
            'reservoir_k': 0.21,
            'rainfall_threshold': 3.0,
            'saturation_capacity': 33.0,
            'recession_exponent': 2.0,
            'time_delay_std': 3.8,
        },
    }

    params = zone_params[zone_id]
    generator = SimpleRunoffGenerator(
        initial_loss=params['initial_loss'],
        constant_loss=params['constant_loss'],
        runoff_coefficient=params['runoff_coefficient'],
        reservoir_k=params['reservoir_k'],
        initial_storage=8.0,
        random_noise_level=0.15,
        random_seed=zone_seed,
        rainfall_threshold=params['rainfall_threshold'],
        saturation_capacity=params['saturation_capacity'],
        recession_exponent=params['recession_exponent'],
        time_delay_std=params['time_delay_std'],
    )

    # 生成本地产流
    local_runoff, stats = generator.generate(rainfall)

    # 如果有上游入流，叠加
    if upstream_inflow is not None:
        total_runoff = local_runoff + upstream_inflow
        print(f"    包含上游入流: 平均={upstream_inflow.mean():.4f} mm/h")
    else:
        total_runoff = local_runoff

    # 添加观测误差
    observed = add_observation_errors(total_runoff, seed=zone_seed)

    print(f"    本地产流系数: {stats['runoff_coefficient']:.4f}")
    print(f"    出口径流量: {observed.sum():.2f} mm")
    print(f"    峰值流量: {observed.max():.4f} mm/h")

    return {
        'observed': observed,
        'true': total_runoff,
        'local_runoff': local_runoff,
        'upstream_inflow': upstream_inflow if upstream_inflow is not None else np.zeros_like(local_runoff),
        'stats': stats,
        'params': params,
    }


# ============================================================================
# 分区参数率定
# ============================================================================

def calibrate_zone(
    rainfall: np.ndarray,
    observed: np.ndarray,
    upstream_inflow: np.ndarray,
    zone_id: str,
) -> Dict:
    """
    率定单个分区的HBV参数

    参数：
        rainfall: 分区降雨
        observed: 观测出流
        upstream_inflow: 上游入流
        zone_id: 分区ID
    """
    print(f"\n  率定分区 {zone_id}:")

    # 参数空间
    param_bounds = {
        'field_capacity': (40.0, 150.0),
        'beta': (0.5, 2.5),
        'k0': (0.05, 0.30),
        'k1': (0.03, 0.15),
    }

    # 目标函数：最小化 -NSE
    def objective(x):
        params = {
            'field_capacity': x[0],
            'beta': x[1],
            'k0': x[2],
            'k1': x[3],
            'k2': 0.02,
            'percolation': 1.0,
            'initial_snow': 0.0,
            'initial_soil': 0.0,
            'initial_upper': 0.0,
            'initial_lower': 0.0,
        }

        # HBV模拟本地产流
        local_runoff = run_hbv_model(rainfall, params)

        # 叠加上游入流
        total_simulated = local_runoff + upstream_inflow

        # 计算NSE
        mean_obs = np.mean(observed)
        nse = 1 - np.sum((observed - total_simulated)**2) / np.sum((observed - mean_obs)**2)

        return -nse  # 最小化负NSE

    # 差分进化优化
    bounds = [param_bounds[k] for k in ['field_capacity', 'beta', 'k0', 'k1']]

    result = differential_evolution(
        objective,
        bounds,
        maxiter=20,
        popsize=10,
        seed=42,
        polish=True,
        disp=False,
    )

    # 最优参数
    optimal_params = {
        'field_capacity': result.x[0],
        'beta': result.x[1],
        'k0': result.x[2],
        'k1': result.x[3],
        'k2': 0.02,
        'percolation': 1.0,
        'initial_snow': 0.0,
        'initial_soil': 0.0,
        'initial_upper': 0.0,
        'initial_lower': 0.0,
    }

    nse = -result.fun
    print(f"    最优NSE: {nse:.4f}")
    print(f"    field_capacity: {optimal_params['field_capacity']:.2f}")
    print(f"    beta: {optimal_params['beta']:.4f}")
    print(f"    k0: {optimal_params['k0']:.4f}")
    print(f"    k1: {optimal_params['k1']:.4f}")

    # 用最优参数模拟
    local_runoff_calibrated = run_hbv_model(rainfall, optimal_params)
    outlet_runoff_calibrated = local_runoff_calibrated + upstream_inflow

    return {
        'params': optimal_params,
        'nse': nse,
        'local_runoff': local_runoff_calibrated,
        'outlet_runoff': outlet_runoff_calibrated,
    }


# ============================================================================
# 主流程：逐级率定
# ============================================================================

def cascading_calibration(rainfall_60day_path: str, output_dir: str = 'results/watershed_calibration'):
    """
    流域逐级率定主流程
    """
    print("=" * 80)
    print("流域分区逐级参数率定")
    print("=" * 80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. 加载60天降雨数据
    print("\n[1/5] 加载60天降雨数据...")
    df = pd.read_csv(rainfall_60day_path)
    rainfall = df['precipitation_mm_per_hour'].values
    timestamps = pd.to_datetime(df['timestamp'])
    n_steps = len(rainfall)
    print(f"  ✓ 时间步数: {n_steps}")
    print(f"  ✓ 总降雨量: {rainfall.sum():.2f} mm")

    # 2. 为每个分区生成观测数据
    print("\n[2/5] 为6个分区生成观测数据...")

    zones = WATERSHED_TOPOLOGY['zones']
    zone_observations = {}

    # 所有分区都接收相同的降雨（简化假设）
    for zone_id in zones:
        zone_observations[zone_id] = generate_zone_observations(
            rainfall=rainfall,
            zone_id=zone_id,
            upstream_inflow=None,  # 先不考虑上游入流，后面逐级加入
            random_seed=42,
        )

    # 3. 从上游到下游逐级率定
    print("\n[3/5] 逐级率定HBV参数（上游→下游）...")

    calibration_results = {}
    upstream_flows = {}  # 记录每个分区的上游入流

    for i, zone_id in enumerate(zones, 1):
        print(f"\n  === 率定分区 {i}/{len(zones)}: {zone_id} ===")

        # 获取上游入流
        conn = WATERSHED_TOPOLOGY['connections'][zone_id]
        upstream_zone = None
        for uz, uconn in WATERSHED_TOPOLOGY['connections'].items():
            if uconn['downstream'] == zone_id and uconn['junction'] is None:
                upstream_zone = uz
                break

        if upstream_zone and upstream_zone in calibration_results:
            upstream_inflow = calibration_results[upstream_zone]['outlet_runoff']
            print(f"  接收上游 {upstream_zone} 的出流")
        else:
            upstream_inflow = np.zeros(n_steps)
            print(f"  无上游入流（起始分区）")

        # 特殊处理P1：接收P4和P5
        if zone_id == 'P1':
            if 'P4' in calibration_results and 'P5' in calibration_results:
                upstream_inflow = (
                    calibration_results['P4']['outlet_runoff'] +
                    calibration_results['P5']['outlet_runoff']
                )
                print(f"  接收上游 P4 和 P5 的出流（汇流）")

        upstream_flows[zone_id] = upstream_inflow

        # 率定
        result = calibrate_zone(
            rainfall=rainfall,
            observed=zone_observations[zone_id]['observed'],
            upstream_inflow=upstream_inflow,
            zone_id=zone_id,
        )

        calibration_results[zone_id] = result

    # 4. 汇总结果
    print("\n[4/5] 汇总率定结果...")

    summary = []
    for zone_id in zones:
        result = calibration_results[zone_id]
        obs = zone_observations[zone_id]

        summary.append({
            'zone': zone_id,
            'description': WATERSHED_TOPOLOGY['descriptions'][zone_id],
            'NSE': result['nse'],
            'field_capacity': result['params']['field_capacity'],
            'beta': result['params']['beta'],
            'k0': result['params']['k0'],
            'k1': result['params']['k1'],
            'observed_runoff_mm': obs['observed'].sum(),
            'calibrated_runoff_mm': result['outlet_runoff'].sum(),
        })

    summary_df = pd.DataFrame(summary)
    summary_path = output_path / 'calibration_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"  ✓ 汇总表: {summary_path}")

    # 打印汇总
    print("\n  率定结果汇总:")
    print(summary_df.to_string(index=False))

    # 5. 保存详细结果
    print("\n[5/5] 保存详细结果...")

    for zone_id in zones:
        zone_dir = output_path / zone_id
        zone_dir.mkdir(exist_ok=True)

        # 保存参数
        params_path = zone_dir / 'calibrated_params.yaml'
        with open(params_path, 'w', encoding='utf-8') as f:
            yaml.dump(calibration_results[zone_id]['params'], f, allow_unicode=True)

        # 保存时间序列
        result = calibration_results[zone_id]
        obs = zone_observations[zone_id]

        ts_df = pd.DataFrame({
            'timestamp': timestamps,
            'rainfall': rainfall,
            'observed': obs['observed'],
            'true_total': obs['true'],
            'true_local': obs['local_runoff'],
            'upstream_inflow': upstream_flows[zone_id],
            'calibrated_local': result['local_runoff'],
            'calibrated_total': result['outlet_runoff'],
        })

        ts_path = zone_dir / 'timeseries.csv'
        ts_df.to_csv(ts_path, index=False)

        # 绘制对比图
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # 降雨
        axes[0].bar(timestamps, rainfall, width=0.04, alpha=0.6, label='降雨')
        axes[0].set_ylabel('降雨 (mm/h)')
        axes[0].legend()
        axes[0].set_title(f'{zone_id}: {WATERSHED_TOPOLOGY["descriptions"][zone_id]}')

        # 径流对比
        axes[1].plot(timestamps, obs['observed'], 'k-', label='观测出流', linewidth=1.5)
        axes[1].plot(timestamps, result['outlet_runoff'], 'r--', label='HBV模拟出流', linewidth=1.5)
        axes[1].plot(timestamps, obs['true'], 'b:', label='真实出流', linewidth=1, alpha=0.7)
        axes[1].set_ylabel('出流 (mm/h)')
        axes[1].legend()
        axes[1].set_title(f'NSE = {result["nse"]:.4f}')

        # 本地产流 vs 上游入流
        axes[2].plot(timestamps, result['local_runoff'], 'g-', label='本地产流（HBV）', linewidth=1.5)
        axes[2].plot(timestamps, upstream_flows[zone_id], 'm-', label='上游入流', linewidth=1.5, alpha=0.7)
        axes[2].set_ylabel('流量 (mm/h)')
        axes[2].set_xlabel('时间')
        axes[2].legend()

        plt.tight_layout()
        fig_path = zone_dir / 'calibration_plot.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  ✓ {zone_id}: {zone_dir}")

    print("\n" + "=" * 80)
    print("✓ 流域逐级率定完成！")
    print("=" * 80)
    print(f"\n平均NSE: {summary_df['NSE'].mean():.4f}")
    print(f"NSE范围: {summary_df['NSE'].min():.4f} - {summary_df['NSE'].max():.4f}")
    print(f"\n输出目录: {output_path}")

    return summary_df, calibration_results


# ============================================================================
# 主函数
# ============================================================================

if __name__ == '__main__':
    # 使用60天降雨数据
    rainfall_path = 'results/extended_timeseries_60days/timeseries_60days.csv'

    summary_df, results = cascading_calibration(
        rainfall_60day_path=rainfall_path,
        output_dir='results/watershed_calibration',
    )
