#!/usr/bin/env python3
"""Zone 1 HBV参数率定（统一校准框架）

简化版本，使用统一的HBVCalibrator框架。

原脚本展示了敏感性分析加速方法，本重构版本聚焦于：
1. 使用统一的校准框架
2. 标准化的参数配置
3. 自动化的结果保存

注：敏感性分析功能已简化，聚焦核心校准流程。

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
    print("Zone 1 HBV参数率定（统一校准框架）")
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

    # 1.2 加载降雨数据
    precip_file = results_dir / "step_08_areal_rainfall" / "8.2_subbasin_areal_precipitation.csv"

    try:
        precip_df = pd.read_csv(precip_file, index_col=0)
    except FileNotFoundError:
        print(f"❌ 错误: 找不到降雨数据: {precip_file}")
        return 1

    # Zone 1的子分区ID
    zone1_subbasins = [str(i) for i in range(10, 24)]
    zone1_cols = [col for col in precip_df.columns if col in zone1_subbasins]
    precipitation = precip_df[zone1_cols].mean(axis=1).values

    print(f"\n✓ 加载降雨数据: {precip_file.name}")
    print(f"  Zone 1子分区: {len(zone1_cols)}个 ({zone1_subbasins[0]}-{zone1_subbasins[-1]})")
    print(f"  降雨范围: {precipitation.min():.2f} - {precipitation.max():.2f} mm/h")

    # 1.3 生成温度数据
    temperature = np.linspace(5, 15, len(precipitation))

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

    param_names = list(param_bounds.keys())

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

    print(f"待率定参数: {len(param_bounds)}个")
    for name, (min_val, max_val) in param_bounds.items():
        print(f"  {name:<20s}: [{min_val:>8.3f}, {max_val:>8.3f}]")

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
    # 步骤 3: 测试HBV模型
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
    print(f"  初始NSE (默认参数): {test_nse:.4f}")

    # ========================================================================
    # 步骤 4: 参数率定
    # ========================================================================
    print("\n步骤 4: 参数率定")
    print("-" * 80)

    print("\n运行Differential Evolution率定...")
    print("  这可能需要几分钟时间...")

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
    output_dir = results_dir / "calibration_with_sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)

    calibrator = HBVCalibrator(
        data=calib_data,
        config=calib_config,
        output_dir=output_dir
    )

    # 运行校准
    result = calibrator.run_calibration()

    print(f"\n✓ 完成!")
    print(f"  最优NSE: {result.best_score:.6f}")
    print(f"  函数评估: {result.n_evaluations}")
    print(f"  计算时间: {result.elapsed_time:.2f}秒")

    # ========================================================================
    # 步骤 5: 性能评估
    # ========================================================================
    print("\n步骤 5: 性能评估")
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
    for metric, value in final_metrics.items():
        if 'peak' not in metric and 'time' not in metric:
            print(f"  {metric.upper():<10s}: {value:>8.4f}")

    # 打印最优参数
    print(f"\n最优参数:")
    for param_name in param_names:
        if param_name in result.best_params:
            value = result.best_params[param_name]
            if param_name in param_bounds:
                min_val, max_val = param_bounds[param_name]
                range_pct = (value - min_val) / (max_val - min_val) * 100
                print(f"  {param_name:<20s}: {value:>10.4f}  (范围的{range_pct:>5.1f}%)")

    # ========================================================================
    # 步骤 6: 保存结果
    # ========================================================================
    print("\n步骤 6: 保存结果")
    print("-" * 80)

    # 使用框架的保存功能
    calibrator.save_results(result)
    print(f"✓ 校准结果已由HBVCalibrator自动保存")

    # 保存额外的配置信息
    calibration_yaml = {
        'description': 'Zone 1 HBV参数率定结果（统一校准框架）',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'framework': 'HBVCalibrator (统一校准框架)',
        'calibration_info': {
            'algorithm': 'differential_evolution',
            'observation_type': 'EnhancedRunoffGenerator',
            'n_evaluations': int(result.n_evaluations),
            'computation_time': float(result.elapsed_time),
        },
        'metrics': {k: float(v) for k, v in final_metrics.items()},
        'zones': {
            1: {
                'runoff_model': 'HBV',
                'parameters': {
                    name: float(result.best_params[name])
                    for name in param_names[:6]
                    if name in result.best_params
                },
                'initial_conditions': {
                    'initial_soil': float(result.best_params.get('FC', 400) * result.best_params.get('initial_soil_ratio', 0.6)),
                    'initial_upper': float(result.best_params.get('initial_upper', 20)),
                    'initial_lower': 30.0,
                    'initial_snow': 0.0,
                },
                'fixed_parameters': fixed_params,
            }
        },
        'refactored': True,
        'benefits': [
            '使用统一的HBVCalibrator接口',
            '使用标准化的指标计算',
            '自动化的结果保存和报告',
            '简化的代码结构'
        ]
    }

    yaml_file = output_dir / "zone1_calibrated_parameters_adaptive.yaml"
    with open(yaml_file, 'w') as f:
        yaml.dump(calibration_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"✓ 保存参数配置: {yaml_file.name}")

    # 保存径流对比数据
    comparison_df = pd.DataFrame({
        'datetime': times,
        'observed_m3s': observed_runoff,
        'simulated_m3s': final_simulated,
        'residual_m3s': observed_runoff - final_simulated
    })
    comparison_file = output_dir / "zone1_calibrated_runoff_adaptive.csv"
    comparison_df.to_csv(comparison_file, index=False)
    print(f"✓ 保存径流对比数据: {comparison_file.name}")

    # ========================================================================
    # 步骤 7: 生成可视化图表
    # ========================================================================
    print("\n步骤 7: 生成可视化图表")
    print("-" * 80)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # 水文过程线
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(observed_runoff, 'b-', label='Observed', linewidth=1.5)
        ax.plot(final_simulated, 'r--', label='HBV Simulation', linewidth=1.5)
        ax.set_xlabel('Time Step (hour)', fontsize=11)
        ax.set_ylabel('Discharge (m³/s)', fontsize=11)
        ax.set_title(f'Zone 1 HBV Calibration (NSE={final_metrics["nse"]:.4f})',
                    fontsize=12, weight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "zone1_hydrograph_adaptive.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 保存: zone1_hydrograph_adaptive.png")

        # 散点图
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(observed_runoff, final_simulated, alpha=0.5, s=20)
        min_val = min(observed_runoff.min(), final_simulated.min())
        max_val = max(observed_runoff.max(), final_simulated.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='1:1 Line')
        ax.set_xlabel('Observed (m³/s)', fontsize=11)
        ax.set_ylabel('Simulated (m³/s)', fontsize=11)
        ax.set_title(f'Observed vs Simulated (NSE={final_metrics["nse"]:.4f})',
                    fontsize=12, weight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.savefig(output_dir / "zone1_scatter_adaptive.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 保存: zone1_scatter_adaptive.png")

        print(f"\n✓ 所有图表已保存到: {output_dir}")

    except ImportError:
        print("⚠ matplotlib未安装，跳过可视化")

    # ========================================================================
    # 步骤 8: 总结
    # ========================================================================
    print("\n" + "=" * 80)
    print("Zone 1 HBV参数率定完成!")
    print("=" * 80)

    print(f"\n【率定结果】")
    print(f"  最优NSE: {result.best_score:.6f}")
    print(f"  计算时间: {result.elapsed_time:.2f}秒")
    print(f"  函数评估: {result.n_evaluations}")

    print(f"\n【性能指标】")
    for metric, value in final_metrics.items():
        if 'peak' not in metric and 'time' not in metric:
            print(f"  {metric.upper()}: {value:.6f}")

    print(f"\n【输出文件】")
    print(f"  参数配置: {yaml_file.name}")
    print(f"  径流数据: {comparison_file.name}")
    print(f"  HBVCalibrator自动生成的完整校准报告")
    print(f"  可视化图表: zone1_*_adaptive.png")

    print("\n✨ 重构改进:")
    print("  1. 使用统一的HBVCalibrator框架")
    print("  2. 使用标准化的指标计算（calculate_metrics）")
    print("  3. 简化的代码结构（移除复杂的敏感性分析）")
    print("  4. 自动化的结果保存和报告")
    print("  5. 聚焦核心校准流程")

    print("\n" + "=" * 80)
    print("✓ 本脚本展示了统一校准框架的标准用法")
    print("✓ 遵循 .claude/AI_DEVELOPMENT_GUIDE.md 中的最佳实践")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
