#!/usr/bin/env python3
"""Zone 1 HBV参数自动率定（使用增强型观测数据 + 统一校准框架）

使用统一的HBVCalibrator框架重构版本。

本脚本展示如何：
1. 使用增强型观测数据进行校准
2. 使用统一的HBVCalibrator框架
3. 自动化的结果保存和报告

重构改进：
- 使用统一的HBVCalibrator接口
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
    print("Zone 1 HBV参数自动率定（使用增强型观测 + 统一校准框架）")
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
    print(f"  平均降雨: {precipitation.mean():.2f} mm/h")

    # 1.3 生成简化温度数据
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
        'FC': [250, 600],                # 土壤最大容量
        'BETA': [1.5, 3.5],              # 土壤蓄水曲线指数
        'K0': [0.1, 0.5],                # 快速径流退水系数
        'K1': [0.02, 0.15],              # 中速径流退水系数
        'K2': [0.005, 0.05],             # 基流退水系数
        'PERC': [0.5, 4.0],              # 渗透速率
        'initial_soil_ratio': [0.3, 0.9],  # 初始土壤湿度比例
        'initial_upper': [5, 50],        # 初始上层储量
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

    # 创建校准数据
    calib_data = CalibrationData(
        precipitation=precipitation,
        observed_runoff=observed_runoff,
        area_km2=zone1_area_km2,
        temperature=temperature
    )

    # 创建校准配置
    calib_config = CalibrationConfig(
        param_bounds=param_bounds,
        fixed_params=fixed_params,
        algorithm='differential_evolution',
        objective_metric='nse',
        maximize_objective=True,
        algorithm_options={
            'maxiter': 50,
            'seed': 42,
            'workers': 1,
            'atol': 0.001,
            'tol': 0.001
        }
    )

    print(f"\n✓ 校准配置:")
    print(f"  算法: {calib_config.algorithm}")
    print(f"  目标: 最大化 {calib_config.objective_metric.upper()}")
    print(f"  最大迭代: {calib_config.algorithm_options['maxiter']}")

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
    print(f"  初始NSE (默认参数): {test_nse:.4f}")

    # ========================================================================
    # 步骤 4: 运行校准
    # ========================================================================
    print("\n步骤 4: 运行HBV校准")
    print("-" * 80)

    # 创建校准器
    output_dir = results_dir / "calibration_enhanced"
    output_dir.mkdir(parents=True, exist_ok=True)

    calibrator = HBVCalibrator(
        data=calib_data,
        config=calib_config,
        output_dir=output_dir
    )

    # 运行校准
    print("正在运行校准...")
    print(f"使用增强型观测数据: {len(observed_runoff)}个时间步")
    result = calibrator.run_calibration()

    print(f"\n✓ 校准完成!")
    print(f"  最优NSE: {result.best_score:.6f}")
    print(f"  函数评估: {result.n_evaluations}")
    print(f"  计算时间: {result.elapsed_time:.2f}秒")

    # ========================================================================
    # 步骤 5: 结果分析
    # ========================================================================
    print("\n步骤 5: 结果分析")
    print("-" * 80)

    # 显示最优参数
    print("\n最优参数:")
    for param, value in result.best_params.items():
        if param in param_bounds:
            bounds = param_bounds[param]
            print(f"  {param:<20s}: {value:>10.4f}  (范围: [{bounds[0]:.2f}, {bounds[1]:.2f}])")

    # 使用最优参数运行模型
    final_params = {**fixed_params, **result.best_params}

    # 处理 initial_soil_ratio
    if 'initial_soil_ratio' in result.best_params:
        final_params['initial_soil'] = result.best_params['initial_soil_ratio'] * result.best_params['FC']
        final_params.pop('initial_soil_ratio')

    hbv_final = HBVRunoff(final_params)
    final_runoff = np.array(hbv_final.simulate(subbasin, precipitation.tolist()))

    # 计算所有指标
    final_metrics = calculate_metrics(
        observed_runoff,
        final_runoff,
        metrics=['nse', 'log_nse', 'kge', 'rmse', 'mae', 'pbias']
    )

    print("\n性能指标:")
    for metric, value in final_metrics.items():
        if 'peak' not in metric and 'time' not in metric:
            print(f"  {metric.upper():10s}: {value:8.6f}")

    # 性能提升
    nse_improvement = final_metrics['nse'] - test_nse
    print(f"\nNSE提升: {test_nse:.4f} → {final_metrics['nse']:.4f} (Δ{nse_improvement:+.4f})")

    # ========================================================================
    # 步骤 6: 保存结果
    # ========================================================================
    print("\n步骤 6: 保存结果")
    print("-" * 80)

    # 使用框架的保存功能
    calibrator.save_results(result)
    print(f"✓ 校准结果已由HBVCalibrator自动保存")

    # 保存额外的增强型观测特定信息
    enhanced_info = {
        'description': 'Zone 1 HBV校准（使用增强型观测数据）',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'framework': 'HBVCalibrator (统一校准框架)',
        'data_source': 'enhanced_observations',
        'obs_file': str(obs_file),
        'time_range': {
            'start': str(times.min()),
            'end': str(times.max()),
            'n_timesteps': len(observed_runoff)
        },
        'performance': {
            'initial_nse': float(test_nse),
            'final_nse': float(final_metrics['nse']),
            'improvement': float(nse_improvement),
            **{k: float(v) for k, v in final_metrics.items()
               if 'peak' not in k and 'time' not in k}
        },
        'refactored': True,
        'benefits': [
            '使用统一的HBVCalibrator接口',
            '自动化的结果保存和报告',
            '更简洁的代码结构',
            '更好的可维护性'
        ]
    }

    info_file = output_dir / "zone1_enhanced_calibration_info.yaml"
    with open(info_file, 'w') as f:
        yaml.dump(enhanced_info, f, default_flow_style=False, sort_keys=False)

    print(f"✓ 保存增强型校准信息: {info_file.name}")

    # ========================================================================
    # 总结
    # ========================================================================
    print("\n" + "=" * 80)
    print("校准完成总结")
    print("=" * 80)

    print(f"\n📊 校准结果:")
    print(f"  - 初始NSE: {test_nse:.4f}")
    print(f"  - 最优NSE: {final_metrics['nse']:.4f}")
    print(f"  - 提升: {nse_improvement:+.4f}")

    print(f"\n📁 输出文件:")
    print(f"  - {info_file}")
    print(f"  - 以及HBVCalibrator自动生成的完整校准报告")

    print("\n✨ 重构改进:")
    print("  1. 使用统一的HBVCalibrator框架")
    print("  2. 代码更简洁，易于维护")
    print("  3. 自动化的结果保存和报告")
    print("  4. 标准化的配置接口")

    print("\n" + "=" * 80)
    print("✓ 本脚本展示了如何使用HBVCalibrator处理增强型观测数据")
    print("✓ 遵循 .claude/AI_DEVELOPMENT_GUIDE.md 最佳实践")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
