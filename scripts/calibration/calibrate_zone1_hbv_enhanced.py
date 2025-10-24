#!/usr/bin/env python3
"""Zone 1 HBV参数自动率定（增强型观测 + 统一校准框架）

使用统一的HBVCalibrator框架重构版本。

使用增强型径流生成器生成的观测数据进行HBV参数率定。

增强型生成器特点：
- 土壤水分核算（状态依赖的产流）
- 三分量径流（快速径流、中速径流、基流）
- 线性水库汇流
- 更接近HBV的物理过程

预期：由于增强型观测数据与HBV有更相似的结构，率定精度应该显著提高（NSE > 0.7）

重构改进：
- 使用统一的HBVCalibrator接口
- 移除重复的指标计算代码
- 更简洁的代码结构
- 自动化的结果保存
- 更好的可维护性

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
import json

from hydrosis.runoff.hbv import HBVRunoff
from hydrosis.calibration import HBVCalibrator, CalibrationData, CalibrationConfig
from hydrosis.analysis import calculate_metrics


def main():
    """主函数"""
    print("=" * 80)
    print("Zone 1 HBV参数自动率定（增强型观测 + 统一校准框架）")
    print("=" * 80)

    results_dir = Path("results/upper_truckee_complete_11steps")

    # ========================================================================
    # 步骤 1: 加载数据
    # ========================================================================
    print("\n步骤 1: 加载Zone 1的数据")
    print("-" * 80)

    # 1.1 加载增强型观测径流
    obs_file = results_dir / "enhanced_observations" / "zone_1_enhanced_runoff.csv"
    if not obs_file.exists():
        print(f"❌ 错误: 观测数据不存在: {obs_file}")
        print(f"   请先运行: python test_enhanced_generator.py")
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

    # Zone 1的子分区ID: 10-23
    zone1_subbasins = [str(i) for i in range(10, 24)]
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
    temperature = np.linspace(5, 15, len(precipitation))
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
        'FC': [250, 600],
        'BETA': [1.5, 3.5],
        'K0': [0.1, 0.5],
        'K1': [0.02, 0.15],
        'K2': [0.005, 0.05],
        'PERC': [0.5, 4.0],
        'initial_soil_ratio': [0.3, 0.9],
        'initial_upper': [5, 50],
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
        'FC': 400, 'BETA': 2.0, 'K0': 0.25, 'K1': 0.08,
        'K2': 0.02, 'PERC': 2.0, 'initial_soil': 240,
        'initial_upper': 20, **fixed_params
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
    print(f"  输出范围: {test_runoff.min():.2f} - {test_runoff.max():.2f} m³/s")
    print(f"  初始NSE (默认参数): {test_nse:.4f}")

    # ========================================================================
    # 步骤 4: 运行校准
    # ========================================================================
    print("\n步骤 4: 参数率定")
    print("-" * 80)

    # 创建校准配置
    calib_config = CalibrationConfig(
        param_bounds=param_bounds,
        fixed_params=fixed_params,
        algorithm='differential_evolution',
        objective_metric='nse',
        maximize_objective=True,
        algorithm_options={
            'maxiter': 150,
            'popsize': 20,
            'seed': 42,
            'workers': 1,
            'polish': True,
            'atol': 1e-6,
            'tol': 1e-6
        }
    )

    # 创建校准器
    output_dir = results_dir / "calibration_enhanced"
    output_dir.mkdir(parents=True, exist_ok=True)

    calibrator = HBVCalibrator(
        data=calib_data,
        config=calib_config,
        output_dir=output_dir
    )

    # 运行校准
    print("\n运行Differential Evolution率定...")
    print("  这可能需要几分钟时间...")

    result = calibrator.run_calibration()

    print(f"  ✓ 完成!")
    print(f"    最优NSE: {result.best_score:.6f}")
    print(f"    函数评估: {result.n_evaluations}")
    print(f"    计算时间: {result.elapsed_time:.2f}秒")

    # ========================================================================
    # 步骤 5: 率定结果
    # ========================================================================
    print("\n步骤 5: 率定结果")
    print("-" * 80)

    best_name = "Differential Evolution"
    best_nse = result.best_score

    print(f"\n算法: {best_name}")
    print(f"  最优NSE: {best_nse:.6f}")
    print(f"  函数评估: {result.n_evaluations}")
    print(f"  计算时间: {result.elapsed_time:.2f}秒")

    # 最优参数
    param_names = list(param_bounds.keys())

    print(f"\n最优参数:")
    for name, value in result.best_params.items():
        if name in param_bounds:
            bounds = param_bounds[name]
            range_pct = (value - bounds[0]) / (bounds[1] - bounds[0]) * 100
            print(f"  {name:<20s}: {value:>12.4f}  (搜索空间的{range_pct:>5.1f}%)")

    # ========================================================================
    # 步骤 6: 使用最优参数运行HBV模型
    # ========================================================================
    print("\n步骤 6: 使用最优参数运行HBV模型")
    print("-" * 80)

    # 使用最优参数运行模型
    final_params = {**fixed_params, **result.best_params}

    # 处理 initial_soil_ratio
    if 'initial_soil_ratio' in result.best_params:
        final_params['initial_soil'] = result.best_params['initial_soil_ratio'] * result.best_params['FC']
        final_params.pop('initial_soil_ratio')

    hbv_final = HBVRunoff(final_params)
    final_simulated = np.array(hbv_final.simulate(subbasin, precipitation.tolist()))

    # 计算所有指标
    final_metrics = calculate_metrics(
        observed_runoff,
        final_simulated,
        metrics=['nse', 'log_nse', 'kge', 'rmse', 'mae', 'pbias']
    )

    print(f"\n率定后性能指标:")
    print(f"  NSE       : {final_metrics['nse']:>8.4f}")
    print(f"  RMSE      : {final_metrics['rmse']:>8.4f}")
    print(f"  MAE       : {final_metrics['mae']:>8.4f}")
    print(f"  PBIAS     : {final_metrics['pbias']:>8.4f}")
    print(f"  KGE       : {final_metrics['kge']:>8.4f}")
    print(f"  LOG_NSE   : {final_metrics['log_nse']:>8.4f}")

    # ========================================================================
    # 步骤 7: 保存最优参数到YAML配置文件
    # ========================================================================
    print("\n步骤 7: 保存最优参数到YAML配置文件")
    print("-" * 80)

    # 使用框架的保存功能
    calibrator.save_results(result)
    print(f"✓ 校准结果已由HBVCalibrator自动保存")

    # 保存额外的增强型观测特定信息
    calibration_yaml = {
        'description': f'Zone 1 HBV参数率定结果（增强型观测）({best_name})',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'framework': 'HBVCalibrator (统一校准框架)',
        'calibration_info': {
            'algorithm': best_name,
            'observation_type': 'EnhancedRunoffGenerator',
            'nse': float(final_metrics['nse']),
            'rmse': float(final_metrics['rmse']),
            'kge': float(final_metrics['kge']),
        },
        'zones': {
            1: {
                'runoff_model': 'HBV',
                'parameters': {
                    name: float(result.best_params[name])
                    for name in param_names
                    if name in result.best_params and name in param_bounds
                },
                'fixed_parameters': fixed_params,
                'initial_conditions': {
                    'initial_soil': float(result.best_params.get('FC', 400) * result.best_params.get('initial_soil_ratio', 0.6)),
                    'initial_upper': float(result.best_params.get('initial_upper', 20)),
                    'initial_lower': 30.0,
                    'initial_snow': 0.0
                }
            }
        },
        'refactored': True,
        'benefits': [
            '使用统一的HBVCalibrator接口',
            '移除重复的指标计算代码',
            '自动化的结果保存和报告',
            '更简洁的代码结构',
            '更好的可维护性'
        ]
    }

    yaml_file = output_dir / "zone1_calibrated_parameters_enhanced.yaml"
    with open(yaml_file, 'w') as f:
        yaml.dump(calibration_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"✓ 保存参数配置: {yaml_file.name}")

    # 保存JSON格式的详细结果
    json_data = {
        'algorithm': best_name,
        'nse': float(final_metrics['nse']),
        'function_evaluations': int(result.n_evaluations),
        'time_seconds': float(result.elapsed_time),
        'parameters': {
            name: float(result.best_params[name])
            for name in param_names
            if name in result.best_params
        },
        'metrics': {k: float(v) for k, v in final_metrics.items()},
    }

    json_file = output_dir / f"zone1_calibration_{best_name.lower().replace(' ', '_')}_enhanced.json"
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"✓ 保存率定结果: {json_file.name}")

    # ========================================================================
    # 步骤 8: 生成分析图表
    # ========================================================================
    print("\n步骤 8: 生成分析图表")
    print("-" * 80)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # 1. 水文过程对比图
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(observed_runoff, 'b-', label='Enhanced Observation', linewidth=1.5)
        ax.plot(final_simulated, 'r--', label='HBV Simulation', linewidth=1.5)
        ax.set_xlabel('Time Step (hour)', fontsize=11)
        ax.set_ylabel('Discharge (m³/s)', fontsize=11)
        ax.set_title(f'Zone 1 HBV Calibration (NSE={final_metrics["nse"]:.4f})', fontsize=12, weight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "zone1_hydrograph_enhanced.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 保存: zone1_hydrograph_enhanced.png")

        # 2. 散点图
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(observed_runoff, final_simulated, alpha=0.5, s=20)
        min_val = min(observed_runoff.min(), final_simulated.min())
        max_val = max(observed_runoff.max(), final_simulated.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='1:1 Line')
        ax.set_xlabel('Observed (m³/s)', fontsize=11)
        ax.set_ylabel('Simulated (m³/s)', fontsize=11)
        ax.set_title(f'Scatter Plot (NSE={final_metrics["nse"]:.4f})', fontsize=12, weight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.savefig(output_dir / "zone1_scatter_enhanced.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 保存: zone1_scatter_enhanced.png")

        print(f"\n✓ 所有图表已保存到: {output_dir}")

    except ImportError:
        print("⚠ matplotlib未安装，跳过可视化")

    # ========================================================================
    # 步骤 9: 保存模拟结果
    # ========================================================================
    print("\n步骤 9: 保存模拟结果")
    print("-" * 80)

    # 保存径流对比数据
    comparison_df = pd.DataFrame({
        'datetime': times,
        'observed_m3s': observed_runoff,
        'simulated_m3s': final_simulated,
        'residual_m3s': observed_runoff - final_simulated
    })

    comparison_file = output_dir / "zone1_calibrated_runoff_enhanced.csv"
    comparison_df.to_csv(comparison_file, index=False)
    print(f"✓ 保存径流对比数据: {comparison_file.name}")

    # ========================================================================
    # 步骤 10: 总结
    # ========================================================================
    print("\n" + "=" * 80)
    print("Zone 1 HBV参数率定完成（增强型观测）！")
    print("=" * 80)

    print(f"\n1. 率定数据:")
    print(f"   - 观测类型: 增强型径流生成器")
    print(f"   - 时间步数: {len(observed_runoff)}")
    print(f"   - 平均流量: {observed_runoff.mean():.2f} m³/s")
    print(f"   - 峰值流量: {observed_runoff.max():.2f} m³/s")

    print(f"\n2. 率定结果:")
    print(f"   - 算法: {best_name}")
    print(f"   - 最优NSE: {best_nse:.6f}")
    print(f"   - 计算时间: {result.elapsed_time:.2f}秒")

    print(f"\n3. 性能指标:")
    print(f"   - NSE: {final_metrics['nse']:.6f}")
    print(f"   - RMSE: {final_metrics['rmse']:.4f} m³/s")
    print(f"   - PBIAS: {final_metrics['pbias']:.2f}%")
    print(f"   - KGE: {final_metrics['kge']:.6f}")

    print(f"\n4. 最优参数:")
    for name in param_names:
        if name in result.best_params:
            print(f"   - {name}: {result.best_params[name]:.4f}")

    print(f"\n5. 输出文件:")
    print(f"   - 参数配置: {yaml_file.name}")
    print(f"   - 率定结果: {json_file.name}")
    print(f"   - 径流数据: {comparison_file.name}")
    print(f"   - 分析图表: zone1_*_enhanced.png")
    print(f"   - HBVCalibrator自动生成的完整校准报告")

    print(f"\n6. 对比之前的率定结果:")
    print(f"   之前（简单线性观测）: NSE ≈ 0.085")
    print(f"   现在（增强型观测）  : NSE = {final_metrics['nse']:.6f}")
    print(f"   改进: {'显著提升' if final_metrics['nse'] > 0.5 else '有所改善' if final_metrics['nse'] > 0.2 else '仍需优化'}")

    print("\n✨ 重构改进:")
    print("  1. 使用统一的HBVCalibrator框架")
    print("  2. 移除重复的指标计算代码（~40行）")
    print("  3. 代码更简洁，易于维护")
    print("  4. 自动化的结果保存和报告")
    print("  5. 标准化的配置接口")

    print("\n" + "=" * 80)
    print("✓ Zone 1参数率定完成！")
    print("✓ 本脚本展示了如何使用HBVCalibrator处理增强型观测数据")
    print("✓ 遵循 .claude/AI_DEVELOPMENT_GUIDE.md 最佳实践")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
