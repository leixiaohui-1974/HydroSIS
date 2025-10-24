#!/usr/bin/env python3
"""Zone 1 HBV参数自动率定（多算法对比 + 统一校准框架）

使用统一的HBVCalibrator框架重构版本，支持多算法对比。

本脚本展示如何：
1. 使用估计观测数据进行校准
2. 使用统一的HBVCalibrator框架
3. 运行多个算法并对比结果
4. 自动化的结果保存和报告

重构改进：
- 使用统一的HBVCalibrator接口
- 支持多算法对比
- 更简洁的代码结构
- 自动化的结果保存
- 更好的可维护性

遵循 .claude/AI_DEVELOPMENT_GUIDE.md 中的最佳实践。

Author: Claude Code (Refactored)
Date: 2025-01-24
"""
import sys
sys.path.insert(0, '/home/user/HydroSIS')

from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import yaml

from hydrosis.runoff.hbv import HBVRunoff
from hydrosis.calibration import HBVCalibrator, CalibrationData, CalibrationConfig
from hydrosis.analysis import calculate_metrics


def main():
    """主函数"""
    print("=" * 80)
    print("Zone 1 HBV参数自动率定（多算法对比 + 统一校准框架）")
    print("=" * 80)

    results_dir = Path("results/upper_truckee_complete_11steps")

    # ========================================================================
    # 步骤 1: 加载数据
    # ========================================================================
    print("\n步骤 1: 加载Zone 1的数据")
    print("-" * 80)

    # 1.1 加载估计观测径流
    obs_file = results_dir / "estimated_observations" / "zone_1_estimated_runoff.csv"
    if not obs_file.exists():
        print(f"❌ 错误: 观测数据不存在: {obs_file}")
        print(f"   请先运行: python generate_estimated_runoff.py")
        return 1

    obs_df = pd.read_csv(obs_file)
    observed_runoff = obs_df['discharge_m3s'].values
    times = pd.to_datetime(obs_df['datetime'])

    print(f"✓ 加载观测径流: {obs_file.name}")
    print(f"  时间范围: {times.min()} 到 {times.max()}")
    print(f"  数据点数: {len(observed_runoff)}")
    print(f"  流量范围: {observed_runoff.min():.2f} - {observed_runoff.max():.2f} m³/s")
    print(f"  平均流量: {observed_runoff.mean():.2f} m³/s")

    # 1.2 加载降雨数据
    precip_file = results_dir / "step_08_areal_rainfall" / "8.2_subbasin_areal_precipitation.csv"

    try:
        precip_df = pd.read_csv(precip_file, index_col=0)
    except FileNotFoundError:
        print(f"❌ 错误: 找不到降雨数据: {precip_file}")
        return 1

    # Zone 1的子分区ID: 101-114
    zone1_subbasins = [str(i) for i in range(101, 115)]
    zone1_cols = [col for col in precip_df.columns if col in zone1_subbasins]

    if len(zone1_cols) == 0:
        print(f"❌ 错误: 未找到Zone 1的子分区数据")
        print(f"  可用列: {precip_df.columns[:10].tolist()}")
        return 1

    precipitation = precip_df[zone1_cols].mean(axis=1).values

    print(f"\n✓ 加载降雨数据: {precip_file.name}")
    print(f"  Zone 1子分区: {len(zone1_cols)}个")
    print(f"  降雨范围: {precipitation.min():.2f} - {precipitation.max():.2f} mm/h")
    print(f"  平均降雨: {precipitation.mean():.2f} mm/h")

    # 1.3 生成温度数据
    temperature = 10 + 8 * np.sin(np.arange(len(precipitation)) / 30)
    print(f"\n✓ 生成温度数据（简化）")
    print(f"  温度范围: {temperature.min():.1f} - {temperature.max():.1f} °C")

    # 1.4 Zone 1流域面积
    zone1_area_km2 = 139.995

    print(f"\n✓ Zone 1流域面积: {zone1_area_km2:.2f} km²")

    # ========================================================================
    # 步骤 2: 配置HBV校准器
    # ========================================================================
    print("\n步骤 2: 配置HBV校准器")
    print("-" * 80)

    # HBV参数搜索范围
    param_bounds = {
        'FC': [200, 600],
        'BETA': [1.0, 3.0],
        'K0': [0.08, 0.4],
        'K1': [0.02, 0.15],
        'K2': [0.005, 0.05],
        'PERC': [0.5, 5.0],
        'initial_soil_ratio': [0.6, 0.95],
        'initial_upper': [20, 100],
    }

    print(f"待率定参数: {len(param_bounds)}个")
    for name, (min_val, max_val) in param_bounds.items():
        print(f"  {name:<20s}: [{min_val:>8.3f}, {max_val:>8.3f}]")

    # 固定参数
    fixed_params = {
        'LP': 0.7,
        'MAXBAS': 3.0,
        'TT': 0.0,
        'CFMAX': 3.5,
        'CFR': 0.05,
        'CWH': 0.1,
        'initial_lower': 30.0,
        'initial_snow': 0.0
    }

    print(f"\n固定参数: {len(fixed_params)}个")
    for name, value in fixed_params.items():
        print(f"  {name:<20s}: {value:>8.3f}")

    # 创建校准数据
    calib_data = CalibrationData(
        precipitation=precipitation,
        observed_runoff=observed_runoff,
        area_km2=zone1_area_km2,
        temperature=temperature
    )

    # ========================================================================
    # 步骤 3: 测试HBV模型（使用默认参数）
    # ========================================================================
    print("\n步骤 3: 测试HBV模型")
    print("-" * 80)

    # 使用默认参数测试
    default_params = {
        'FC': 300, 'BETA': 2.0, 'K0': 0.15, 'K1': 0.08,
        'K2': 0.02, 'PERC': 2.0, 'initial_soil': 240,
        'initial_upper': 50, **fixed_params
    }

    class MockSubbasin:
        def __init__(self, area_km2):
            self.area_km2 = area_km2

    subbasin = MockSubbasin(zone1_area_km2)
    hbv_test = HBVRunoff(default_params)
    test_runoff = np.array(hbv_test.simulate(subbasin, precipitation.tolist()))

    from hydrosis.evaluation.metrics import nash_sutcliffe_efficiency
    test_nse = nash_sutcliffe_efficiency(test_runoff, observed_runoff)

    print(f"✓ 模型测试完成")
    print(f"  输出长度: {len(test_runoff)}")
    print(f"  初始NSE (默认参数): {test_nse:.4f}")

    # ========================================================================
    # 步骤 4: 多算法参数率定
    # ========================================================================
    print("\n步骤 4: 多算法参数率定")
    print("-" * 80)

    # 定义多个算法配置
    algorithm_configs = {
        'SCE-UA': {
            'algorithm': 'sce_ua',
            'algorithm_options': {
                'n_complexes': 6,
                'maxiter': 150,
                'patience': 30,
                'seed': 42
            }
        },
        'PSO': {
            'algorithm': 'pso',
            'algorithm_options': {
                'n_particles': 50,
                'maxiter': 150,
                'w': 0.7,
                'c1': 1.5,
                'c2': 1.5,
                'patience': 30,
                'seed': 42
            }
        },
    }

    output_dir = results_dir / "calibration"
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration_results = {}

    for alg_name, alg_config in algorithm_configs.items():
        print(f"\n运行{alg_name}率定...")
        print(f"  这可能需要几分钟时间...")

        # 创建校准配置
        calib_config = CalibrationConfig(
            param_bounds=param_bounds,
            fixed_params=fixed_params,
            algorithm=alg_config['algorithm'],
            objective_metric='nse',
            maximize_objective=True,
            algorithm_options=alg_config['algorithm_options']
        )

        # 创建算法特定的输出目录
        alg_output_dir = output_dir / f"{alg_name.lower().replace('-', '_')}"
        alg_output_dir.mkdir(parents=True, exist_ok=True)

        # 创建校准器并运行
        calibrator = HBVCalibrator(
            data=calib_data,
            config=calib_config,
            output_dir=alg_output_dir
        )

        result = calibrator.run_calibration()
        calibration_results[alg_name] = result

        print(f"  ✓ 完成!")
        print(f"    最优NSE: {result.best_score:.6f}")
        print(f"    函数评估: {result.n_evaluations}")
        print(f"    计算时间: {result.elapsed_time:.2f}秒")

        # 保存算法特定结果
        calibrator.save_results(result)

    # ========================================================================
    # 步骤 5: 结果对比
    # ========================================================================
    print("\n步骤 5: 率定结果对比")
    print("-" * 80)

    print(f"\n{'算法':<15s} {'NSE':>10s} {'时间(s)':>10s} {'评估次数':>10s}")
    print("-" * 50)
    for name, result in calibration_results.items():
        print(f"{name:<15s} {result.best_score:>10.6f} {result.elapsed_time:>10.2f} "
              f"{result.n_evaluations:>10d}")

    # 选择最佳算法
    best_alg_name = max(calibration_results.items(), key=lambda x: x[1].best_score)[0]
    best_result = calibration_results[best_alg_name]

    print(f"\n✓ 最佳算法: {best_alg_name} (NSE={best_result.best_score:.6f})")
    print("\n最优参数:")
    for param, value in best_result.best_params.items():
        if param in param_bounds:
            bounds = param_bounds[param]
            pct = (value - bounds[0]) / (bounds[1] - bounds[0]) * 100
            print(f"  {param:<20s}: {value:>10.4f}  (搜索空间的{pct:>5.1f}%)")

    # ========================================================================
    # 步骤 6: 使用最优参数运行模型
    # ========================================================================
    print("\n步骤 6: 使用最优参数运行HBV模型")
    print("-" * 80)

    # 使用最优参数运行模型
    final_params = {**fixed_params, **best_result.best_params}

    # 处理 initial_soil_ratio
    if 'initial_soil_ratio' in best_result.best_params:
        final_params['initial_soil'] = best_result.best_params['initial_soil_ratio'] * best_result.best_params['FC']
        final_params.pop('initial_soil_ratio')

    hbv_final = HBVRunoff(final_params)
    calibrated_runoff = np.array(hbv_final.simulate(subbasin, precipitation.tolist()))

    # 计算所有指标
    final_metrics = calculate_metrics(
        observed_runoff,
        calibrated_runoff,
        metrics=['nse', 'log_nse', 'kge', 'rmse', 'mae', 'pbias']
    )

    print("\n率定后性能指标:")
    for metric, value in final_metrics.items():
        if 'peak' not in metric and 'time' not in metric:
            print(f"  {metric.upper():10s}: {value:8.6f}")

    # ========================================================================
    # 步骤 7: 保存最优参数到YAML配置文件
    # ========================================================================
    print("\n步骤 7: 保存最优参数到YAML配置文件")
    print("-" * 80)

    # 创建参数率定YAML配置
    calibration_yaml = {
        'description': f'Zone 1 HBV参数率定结果 ({best_alg_name})',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'framework': 'HBVCalibrator (统一校准框架)',
        'calibration_info': {
            'algorithm': best_alg_name,
            'nse': float(best_result.best_score),
            'rmse': float(final_metrics['rmse']),
            'pbias': float(final_metrics['pbias']),
            'n_evaluations': int(best_result.n_evaluations),
            'computation_time_seconds': float(best_result.elapsed_time),
        },
        'algorithms_compared': {
            name: {
                'nse': float(result.best_score),
                'time_seconds': float(result.elapsed_time),
                'n_evaluations': int(result.n_evaluations)
            }
            for name, result in calibration_results.items()
        },
        'global_defaults': {
            'hbv': {
                **{k: float(v) for k, v in best_result.best_params.items() if k in param_bounds},
                **{k: float(v) for k, v in fixed_params.items()},
                'UZL': 5.0,
                'initial_soil': float(best_result.best_params.get('FC', 300) * 0.3),
                'initial_upper': 5.0,
                'initial_lower': 20.0,
            },
            'muskingum': {
                'K': 10.0,
                'x': 0.2,
                'time_step': 1.0,
            }
        },
        'zone_adjustments': [
            {
                'zone_id': 1,
                'description': f'最上游分区 - 率定后参数 (NSE={best_result.best_score:.4f})',
                'runoff_parameters': {
                    'hbv': {
                        param: {
                            'method': 'set',
                            'value': float(value)
                        }
                        for param, value in best_result.best_params.items()
                        if param in param_bounds
                    }
                }
            }
        ],
        'subbasin_overrides': None,
        'refactored': True,
        'benefits': [
            '使用统一的HBVCalibrator接口',
            '支持多算法对比',
            '自动化的结果保存和报告',
            '更简洁的代码结构',
            '更好的可维护性'
        ]
    }

    # 保存YAML文件
    yaml_file = output_dir / "zone1_calibrated_parameters.yaml"
    with open(yaml_file, 'w') as f:
        yaml.dump(calibration_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"✓ 保存参数配置: {yaml_file.name}")

    # ========================================================================
    # 步骤 8: 保存模拟结果
    # ========================================================================
    print("\n步骤 8: 保存模拟结果")
    print("-" * 80)

    # 保存对比数据
    result_df = pd.DataFrame({
        'datetime': times,
        'time_step': np.arange(len(observed_runoff)),
        'observed_m3s': observed_runoff,
        'simulated_m3s': calibrated_runoff,
        'residual_m3s': observed_runoff - calibrated_runoff,
        'relative_error_pct': (calibrated_runoff - observed_runoff) / observed_runoff * 100
    })
    result_df.to_csv(output_dir / "zone1_calibrated_runoff.csv", index=False)
    print(f"✓ 保存径流对比数据: zone1_calibrated_runoff.csv")

    # ========================================================================
    # 总结
    # ========================================================================
    print("\n" + "=" * 80)
    print("Zone 1 HBV参数率定完成！")
    print("=" * 80)

    print(f"\n1. 率定数据:")
    print(f"   - 时间步数: {len(observed_runoff)}")
    print(f"   - 平均流量: {observed_runoff.mean():.2f} m³/s")
    print(f"   - 峰值流量: {observed_runoff.max():.2f} m³/s")

    print(f"\n2. 率定结果:")
    print(f"   - 最佳算法: {best_alg_name}")
    print(f"   - 最优NSE: {best_result.best_score:.6f}")
    print(f"   - 计算时间: {best_result.elapsed_time:.2f}秒")

    print(f"\n3. 性能指标:")
    for metric, value in final_metrics.items():
        if 'peak' not in metric and 'time' not in metric:
            print(f"   - {metric.upper()}: {value:.6f}")

    print(f"\n4. 最优参数:")
    for param, value in best_result.best_params.items():
        if param in param_bounds:
            print(f"   - {param}: {value:.4f}")

    print(f"\n5. 算法对比:")
    for name, result in calibration_results.items():
        print(f"   - {name}: NSE={result.best_score:.6f}, Time={result.elapsed_time:.2f}s")

    print(f"\n6. 输出文件:")
    print(f"   - 参数配置: {yaml_file.name}")
    print(f"   - 径流数据: zone1_calibrated_runoff.csv")
    print(f"   - 各算法完整报告在对应子目录中")

    print(f"\n7. 下一步操作:")
    print(f"   使用率定后的参数重新运行完整工作流:")
    print(f"   python rerun_step09_10.py --calibration {yaml_file}")

    print("\n✨ 重构改进:")
    print("  1. 使用统一的HBVCalibrator框架")
    print("  2. 支持多算法对比")
    print("  3. 代码更简洁，易于维护")
    print("  4. 自动化的结果保存和报告")
    print("  5. 标准化的配置接口")

    print("\n" + "=" * 80)
    print("✓ Zone 1参数率定完成！参数已保存到YAML配置文件。")
    print("✓ 本脚本展示了如何使用HBVCalibrator进行多算法对比")
    print("✓ 遵循 .claude/AI_DEVELOPMENT_GUIDE.md 最佳实践")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
