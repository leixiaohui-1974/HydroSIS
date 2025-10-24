#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
混合策略参数率定：逐级初始化 + 联合精细优化

策略：
1. 阶段1（快速）：逐级率定获得初始参数（~1分钟）
2. 阶段2（精细）：用初始参数作为起点，联合优化（~2分钟）

优势：
✅ 更好的初始值（避免随机初始化）
✅ 更快收敛（已经接近最优）
✅ 更高质量解（精细优化）
✅ 结合两种方法的优点

对比：
- 纯逐级：快但误差累积，出口NSE=-2.34
- 纯联合：好但可能收敛慢，出口NSE=0.71
- 混合策略：快速+高质量，期望出口NSE>0.71
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
import time

from simple_runoff_generator import SimpleRunoffGenerator, add_observation_errors


# ============================================================================
# 导入已有模块
# ============================================================================

# 流域拓扑（与之前相同）
WATERSHED_TOPOLOGY = {
    'zones': ['P3', 'P6', 'P2', 'P5', 'P4', 'P1'],
    'connections': {
        'P3': {'downstream': 'P6'},
        'P6': {'downstream': 'P2'},
        'P2': {'downstream': 'P4'},
        'P5': {'downstream': 'P1'},
        'P4': {'downstream': 'P1'},
        'P1': {'downstream': None},
    },
    'control_points': {
        'P3': {'observe': True, 'weight': 1.0},
        'P6': {'observe': True, 'weight': 1.0},
        'P2': {'observe': True, 'weight': 1.0},
        'P5': {'observe': True, 'weight': 1.0},
        'P4': {'observe': True, 'weight': 1.5},
        'P1': {'observe': True, 'weight': 2.0},
    },
}


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


def simulate_watershed(rainfall: np.ndarray, zone_params: Dict[str, dict]) -> Dict[str, np.ndarray]:
    """模拟整个流域"""
    zones = WATERSHED_TOPOLOGY['zones']
    results = {}

    for zone_id in zones:
        local_runoff = run_hbv_model(rainfall, zone_params[zone_id])
        upstream_inflow = np.zeros_like(local_runoff)

        for upstream_zone, conn in WATERSHED_TOPOLOGY['connections'].items():
            if conn['downstream'] == zone_id and upstream_zone in results:
                upstream_inflow += results[upstream_zone]['outlet']

        outlet_runoff = local_runoff + upstream_inflow

        results[zone_id] = {
            'local': local_runoff,
            'outlet': outlet_runoff,
            'upstream': upstream_inflow,
        }

    return results


# ============================================================================
# 阶段1：逐级率定（快速获取初始参数）
# ============================================================================

def cascading_calibration_fast(
    rainfall: np.ndarray,
    zone_observations: Dict,
    maxiter: int = 15,  # 减少迭代次数以加快速度
) -> Dict[str, dict]:
    """
    快速逐级率定，用于获取初始参数

    参数：
        rainfall: 降雨序列
        zone_observations: 各分区观测数据
        maxiter: 最大迭代次数（减少以加快速度）

    返回：
        各分区的率定参数
    """
    print("\n=== 阶段1：逐级率定（快速初始化） ===")

    zones = WATERSHED_TOPOLOGY['zones']
    param_bounds = {
        'field_capacity': (40.0, 150.0),
        'beta': (0.5, 2.5),
        'k0': (0.05, 0.30),
        'k1': (0.03, 0.15),
    }

    zone_params = {}
    upstream_flows = {}

    # 初始化无上游的分区
    upstream_flows['P3'] = np.zeros(len(rainfall))
    upstream_flows['P5'] = np.zeros(len(rainfall))

    for i, zone_id in enumerate(zones, 1):
        print(f"\n  [{i}/{len(zones)}] 率定 {zone_id}...", end=' ')

        # 获取上游入流（使用已计算的结果）
        upstream_inflow = np.zeros(len(rainfall))

        # 特殊处理P1（汇流节点）
        if zone_id == 'P1':
            for uz in ['P4', 'P5']:
                if uz in upstream_flows:
                    # P4或P5的出口流量 = 本地产流 + 上游入流
                    local = run_hbv_model(rainfall, zone_params[uz])
                    upstream_inflow += local + upstream_flows[uz]
        else:
            # 普通分区：查找上游分区
            for upstream_zone, conn in WATERSHED_TOPOLOGY['connections'].items():
                if conn['downstream'] == zone_id and upstream_zone in zone_params:
                    # 上游出口流量 = 本地产流 + 上游的上游入流
                    local = run_hbv_model(rainfall, zone_params[upstream_zone])
                    upstream_inflow += local + upstream_flows[upstream_zone]

        upstream_flows[zone_id] = upstream_inflow

        # 目标函数
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

            local_runoff = run_hbv_model(rainfall, params)
            total_simulated = local_runoff + upstream_inflow

            observed = zone_observations[zone_id]['outlet']
            mean_obs = np.mean(observed)
            nse = 1 - np.sum((observed - total_simulated)**2) / np.sum((observed - mean_obs)**2)

            return -nse

        # 快速优化
        bounds = [param_bounds[k] for k in ['field_capacity', 'beta', 'k0', 'k1']]
        result = differential_evolution(
            objective,
            bounds,
            maxiter=maxiter,
            popsize=8,  # 减小种群以加快速度
            seed=42,
            polish=False,  # 不精细优化以加快速度
            disp=False,
        )

        zone_params[zone_id] = {
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
        print(f"NSE={nse:.3f}")

    return zone_params


# ============================================================================
# 阶段2：联合优化（使用初始参数精细优化）
# ============================================================================

class HybridCalibrator:
    """混合策略率定器"""

    def __init__(
        self,
        rainfall: np.ndarray,
        observations: Dict[str, np.ndarray],
        control_points: Dict[str, dict],
        initial_params: Dict[str, dict] = None,
    ):
        """
        初始化

        参数：
            initial_params: 初始参数（来自逐级率定）
        """
        self.rainfall = rainfall
        self.observations = observations
        self.control_points = control_points
        self.zones = WATERSHED_TOPOLOGY['zones']
        self.initial_params = initial_params

        self.param_names = ['field_capacity', 'beta', 'k0', 'k1']
        self.n_params_per_zone = len(self.param_names)
        self.n_zones = len(self.zones)
        self.n_params_total = self.n_params_per_zone * self.n_zones

        self.param_bounds_single = {
            'field_capacity': (40.0, 150.0),
            'beta': (0.5, 2.5),
            'k0': (0.05, 0.30),
            'k1': (0.03, 0.15),
        }

        self.bounds = []
        for zone_id in self.zones:
            for param_name in self.param_names:
                self.bounds.append(self.param_bounds_single[param_name])

    def vector_to_params(self, x: np.ndarray) -> Dict[str, dict]:
        """参数向量转字典"""
        zone_params = {}
        for i, zone_id in enumerate(self.zones):
            start_idx = i * self.n_params_per_zone
            params = {
                'field_capacity': x[start_idx + 0],
                'beta': x[start_idx + 1],
                'k0': x[start_idx + 2],
                'k1': x[start_idx + 3],
                'k2': 0.02,
                'percolation': 1.0,
                'initial_snow': 0.0,
                'initial_soil': 0.0,
                'initial_upper': 0.0,
                'initial_lower': 0.0,
            }
            zone_params[zone_id] = params
        return zone_params

    def params_to_vector(self, zone_params: Dict[str, dict]) -> np.ndarray:
        """参数字典转向量"""
        x = []
        for zone_id in self.zones:
            for param_name in self.param_names:
                x.append(zone_params[zone_id][param_name])
        return np.array(x)

    def calculate_spatial_penalty(self, x: np.ndarray, weight: float = 0.1) -> float:
        """空间正则化惩罚"""
        penalty = 0.0
        main_stem = ['P3', 'P6', 'P2', 'P4', 'P1']

        for i in range(len(main_stem) - 1):
            upstream = main_stem[i]
            downstream = main_stem[i + 1]
            idx_up = self.zones.index(upstream)
            idx_down = self.zones.index(downstream)

            for j in range(self.n_params_per_zone):
                param_up = x[idx_up * self.n_params_per_zone + j]
                param_down = x[idx_down * self.n_params_per_zone + j]
                bound = self.bounds[idx_up * self.n_params_per_zone + j]
                range_val = bound[1] - bound[0]
                normalized_diff = (param_up - param_down) / range_val
                penalty += normalized_diff ** 2

        return weight * penalty

    def objective(self, x: np.ndarray) -> float:
        """目标函数"""
        zone_params = self.vector_to_params(x)
        results = simulate_watershed(self.rainfall, zone_params)

        total_weighted_nse = 0.0
        total_weight = 0.0

        for zone_id, config in self.control_points.items():
            if not config['observe']:
                continue

            observed = self.observations[zone_id]
            simulated = results[zone_id]['outlet']
            mean_obs = np.mean(observed)
            nse = 1 - np.sum((observed - simulated)**2) / np.sum((observed - mean_obs)**2)

            weight = config['weight']
            total_weighted_nse += weight * nse
            total_weight += weight

        avg_nse = total_weighted_nse / total_weight
        spatial_penalty = self.calculate_spatial_penalty(x, weight=0.1)

        return -avg_nse + spatial_penalty

    def calibrate(self, maxiter: int = 20, popsize: int = 12) -> Dict:
        """
        执行联合优化

        如果提供了initial_params，使用它作为起点
        """
        print("\n=== 阶段2：联合优化（精细优化） ===")
        print(f"  参数维度: {self.n_params_total}")
        print(f"  最大迭代: {maxiter}")

        # 如果有初始参数，构建初始种群
        if self.initial_params:
            print("  使用逐级率定结果作为初始值")
            x0 = self.params_to_vector(self.initial_params)

            # 构建初始种群：围绕x0的小范围扰动
            init_population = [x0]
            for _ in range(popsize - 1):
                perturbation = np.random.randn(len(x0)) * 0.1  # 10%扰动
                x_perturbed = x0 + perturbation

                # 确保在边界内
                for i, (lower, upper) in enumerate(self.bounds):
                    x_perturbed[i] = np.clip(x_perturbed[i], lower, upper)

                init_population.append(x_perturbed)

            init_population = np.array(init_population)
        else:
            print("  使用随机初始值")
            init_population = 'latinhypercube'

        result = differential_evolution(
            self.objective,
            self.bounds,
            maxiter=maxiter,
            popsize=popsize,
            seed=42,
            polish=True,
            disp=True,
            workers=1,
            init=init_population,
        )

        # 解析结果
        optimal_zone_params = self.vector_to_params(result.x)
        optimal_results = simulate_watershed(self.rainfall, optimal_zone_params)

        # 计算性能
        performance = {}
        for zone_id, config in self.control_points.items():
            if not config['observe']:
                continue

            observed = self.observations[zone_id]
            simulated = optimal_results[zone_id]['outlet']
            mean_obs = np.mean(observed)
            nse = 1 - np.sum((observed - simulated)**2) / np.sum((observed - mean_obs)**2)
            rmse = np.sqrt(np.mean((observed - simulated)**2))
            bias = np.mean(simulated - observed)

            performance[zone_id] = {
                'NSE': nse,
                'RMSE': rmse,
                'Bias': bias,
                'observed_sum': observed.sum(),
                'simulated_sum': simulated.sum(),
            }

        return {
            'optimal_params': optimal_zone_params,
            'optimal_results': optimal_results,
            'performance': performance,
            'objective_value': result.fun,
            'n_iterations': result.nit,
            'n_evaluations': result.nfev,
        }


# ============================================================================
# 观测数据生成
# ============================================================================

def generate_zone_observations(rainfall: np.ndarray, zone_id: str, random_seed: int = 42) -> Dict:
    """生成分区观测数据"""
    zone_seed = random_seed + hash(zone_id) % 1000

    zone_params = {
        'P3': {'initial_loss': 15.0, 'constant_loss': 0.35, 'runoff_coefficient': 0.40,
               'reservoir_k': 0.16, 'rainfall_threshold': 2.0, 'saturation_capacity': 28.0,
               'recession_exponent': 1.6, 'time_delay_std': 2.5},
        'P6': {'initial_loss': 18.0, 'constant_loss': 0.40, 'runoff_coefficient': 0.42,
               'reservoir_k': 0.18, 'rainfall_threshold': 2.5, 'saturation_capacity': 30.0,
               'recession_exponent': 1.8, 'time_delay_std': 3.0},
        'P2': {'initial_loss': 20.0, 'constant_loss': 0.45, 'runoff_coefficient': 0.45,
               'reservoir_k': 0.20, 'rainfall_threshold': 2.8, 'saturation_capacity': 32.0,
               'recession_exponent': 1.9, 'time_delay_std': 3.5},
        'P5': {'initial_loss': 16.0, 'constant_loss': 0.38, 'runoff_coefficient': 0.43,
               'reservoir_k': 0.17, 'rainfall_threshold': 2.2, 'saturation_capacity': 29.0,
               'recession_exponent': 1.7, 'time_delay_std': 2.8},
        'P4': {'initial_loss': 17.0, 'constant_loss': 0.42, 'runoff_coefficient': 0.44,
               'reservoir_k': 0.19, 'rainfall_threshold': 2.6, 'saturation_capacity': 31.0,
               'recession_exponent': 1.85, 'time_delay_std': 3.2},
        'P1': {'initial_loss': 19.0, 'constant_loss': 0.43, 'runoff_coefficient': 0.46,
               'reservoir_k': 0.21, 'rainfall_threshold': 3.0, 'saturation_capacity': 33.0,
               'recession_exponent': 2.0, 'time_delay_std': 3.8},
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

    local_runoff, stats = generator.generate(rainfall)
    observed = add_observation_errors(local_runoff, seed=zone_seed)

    return {
        'observed': observed,
        'true': local_runoff,
        'stats': stats,
    }


def generate_all_observations_with_routing(rainfall: np.ndarray) -> Dict:
    """生成所有分区的观测数据（含汇流）"""
    zones = WATERSHED_TOPOLOGY['zones']
    zone_observations = {}
    local_runoff = {}

    for zone_id in zones:
        obs_data = generate_zone_observations(rainfall, zone_id)
        local_runoff[zone_id] = obs_data['true']
        zone_observations[zone_id] = obs_data

    for zone_id in zones:
        upstream_inflow = np.zeros_like(rainfall)
        for upstream_zone, conn in WATERSHED_TOPOLOGY['connections'].items():
            if conn['downstream'] == zone_id and upstream_zone in zone_observations:
                upstream_outlet = zone_observations[upstream_zone]['outlet']
                upstream_inflow += upstream_outlet

        outlet_flow = local_runoff[zone_id] + upstream_inflow
        zone_seed = 42 + hash(zone_id) % 1000
        outlet_observed = add_observation_errors(outlet_flow, seed=zone_seed)

        zone_observations[zone_id]['outlet'] = outlet_observed
        zone_observations[zone_id]['outlet_true'] = outlet_flow
        zone_observations[zone_id]['upstream_inflow'] = upstream_inflow

    return zone_observations


# ============================================================================
# 主函数：混合策略
# ============================================================================

def main():
    print("=" * 80)
    print("混合策略参数率定：逐级初始化 + 联合精细优化")
    print("=" * 80)

    output_dir = Path('results/watershed_calibration_hybrid')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载降雨数据
    print("\n[1/5] 加载60天降雨数据...")
    rainfall_path = 'results/extended_timeseries_60days/timeseries_60days.csv'
    df = pd.read_csv(rainfall_path)
    rainfall = df['precipitation_mm_per_hour'].values
    timestamps = pd.to_datetime(df['timestamp'])
    print(f"  ✓ 时间步数: {len(rainfall)}")
    print(f"  ✓ 总降雨量: {rainfall.sum():.2f} mm")

    # 2. 生成观测数据
    print("\n[2/5] 生成流域观测数据...")
    zone_observations = generate_all_observations_with_routing(rainfall)

    # 3. 阶段1：快速逐级率定
    print("\n[3/5] 阶段1：快速逐级率定（初始化）...")
    start_time = time.time()
    initial_params = cascading_calibration_fast(
        rainfall=rainfall,
        zone_observations=zone_observations,
        maxiter=15,
    )
    phase1_time = time.time() - start_time
    print(f"\n  阶段1完成，耗时: {phase1_time:.1f}秒")

    # 4. 阶段2：联合精细优化
    print("\n[4/5] 阶段2：联合精细优化...")
    observations = {
        zone_id: zone_observations[zone_id]['outlet']
        for zone_id in WATERSHED_TOPOLOGY['zones']
    }

    start_time = time.time()
    calibrator = HybridCalibrator(
        rainfall=rainfall,
        observations=observations,
        control_points=WATERSHED_TOPOLOGY['control_points'],
        initial_params=initial_params,
    )

    calibration_result = calibrator.calibrate(maxiter=20, popsize=12)
    phase2_time = time.time() - start_time
    print(f"\n  阶段2完成，耗时: {phase2_time:.1f}秒")

    total_time = phase1_time + phase2_time

    # 5. 保存结果
    print("\n[5/5] 保存结果...")

    summary = []
    for zone_id in WATERSHED_TOPOLOGY['zones']:
        perf = calibration_result['performance'][zone_id]
        params = calibration_result['optimal_params'][zone_id]

        summary.append({
            'zone': zone_id,
            'NSE': perf['NSE'],
            'RMSE': perf['RMSE'],
            'Bias': perf['Bias'],
            'observed_mm': perf['observed_sum'],
            'simulated_mm': perf['simulated_sum'],
            'field_capacity': params['field_capacity'],
            'beta': params['beta'],
            'k0': params['k0'],
            'k1': params['k1'],
        })

    summary_df = pd.DataFrame(summary)
    summary_path = output_dir / 'hybrid_calibration_summary.csv'
    summary_df.to_csv(summary_path, index=False)

    print("\n  混合策略率定结果:")
    print(summary_df.to_string(index=False))

    # 保存详细结果
    for zone_id in WATERSHED_TOPOLOGY['zones']:
        zone_dir = output_dir / zone_id
        zone_dir.mkdir(exist_ok=True)

        params_path = zone_dir / 'params.yaml'
        with open(params_path, 'w') as f:
            yaml.dump(calibration_result['optimal_params'][zone_id], f)

        result = calibration_result['optimal_results'][zone_id]
        obs = zone_observations[zone_id]

        ts_df = pd.DataFrame({
            'timestamp': timestamps,
            'rainfall': rainfall,
            'observed': obs['outlet'],
            'simulated': result['outlet'],
            'local_runoff': result['local'],
            'upstream_inflow': result['upstream'],
        })
        ts_path = zone_dir / 'timeseries.csv'
        ts_df.to_csv(ts_path, index=False)

    print(f"\n" + "=" * 80)
    print("✓ 混合策略率定完成！")
    print("=" * 80)
    print(f"\n平均NSE: {summary_df['NSE'].mean():.4f}")
    print(f"出口P1 NSE: {summary_df[summary_df['zone']=='P1']['NSE'].values[0]:.4f}")
    print(f"\n总耗时: {total_time:.1f}秒 (阶段1: {phase1_time:.1f}s + 阶段2: {phase2_time:.1f}s)")
    print(f"输出目录: {output_dir}")

    return summary_df, calibration_result, total_time


if __name__ == '__main__':
    summary_df, result, total_time = main()
