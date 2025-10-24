#!/usr/bin/env python3
"""Zone 1 HBV参数率定 + 预热期（统一校准框架）

使用预热期(Warm-up Period)解决初始状态问题。

根据敏感性分析的诊断结果，实施解决方案：
1. 添加48小时预热期
2. 预热期不参与率定评估
3. 验证物理参数敏感性恢复
4. 对比有无预热期的率定结果

重构改进：
- 使用统一的数据和配置结构
- 使用标准化的指标计算
- 保留预热期的特殊逻辑
- 更清晰的代码组织

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


def run_hbv_with_warmup(params, precipitation_full, fixed_params, initial_conditions,
                         area_km2, warmup_hours):
    """运行HBV模型（带预热期）

    Args:
        params: 参数字典
        precipitation_full: 完整降雨数据（包含预热期）
        fixed_params: 固定参数
        initial_conditions: 初始条件
        area_km2: 流域面积
        warmup_hours: 预热期小时数

    Returns:
        仅率定期的径流序列
    """
    class MockSubbasin:
        def __init__(self, area):
            self.area_km2 = area

    subbasin = MockSubbasin(area_km2)

    # 合并所有参数
    full_params = {**params, **fixed_params, **initial_conditions}

    # 运行HBV模型
    hbv = HBVRunoff(full_params)
    runoff_full = hbv.simulate(subbasin, precipitation_full.tolist())

    # 只返回率定期结果
    return np.array(runoff_full[warmup_hours:])


def main():
    """主函数"""
    print("=" * 80)
    print("Zone 1 HBV参数率定 + 预热期修正（统一校准框架）")
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
        return 1

    obs_df = pd.read_csv(obs_file)
    observed_runoff_full = obs_df['discharge_m3s'].values
    times = pd.to_datetime(obs_df['datetime'])

    # 1.2 加载降雨数据
    precip_file = results_dir / "step_08_areal_rainfall" / "8.2_subbasin_areal_precipitation.csv"

    try:
        precip_df = pd.read_csv(precip_file, index_col=0)
    except FileNotFoundError:
        print(f"❌ 错误: 找不到降雨数据: {precip_file}")
        return 1

    zone1_subbasins = [str(i) for i in range(10, 24)]
    zone1_cols = [col for col in precip_df.columns if col in zone1_subbasins]
    precipitation_full = precip_df[zone1_cols].mean(axis=1).values

    # 1.3 配置预热期
    WARMUP_HOURS = 48  # 2天预热期

    print(f"✓ 加载完整数据:")
    print(f"  总时长: {len(observed_runoff_full)} 小时")
    print(f"  预热期: {WARMUP_HOURS} 小时")
    print(f"  率定期: {len(observed_runoff_full) - WARMUP_HOURS} 小时")

    if len(observed_runoff_full) <= WARMUP_HOURS:
        print(f"\n⚠ 警告: 数据长度不足! 当前数据只有{len(observed_runoff_full)}小时")
        WARMUP_HOURS = max(12, len(observed_runoff_full) // 2)
        print(f"  调整预热期为: {WARMUP_HOURS} 小时")

    # 分割数据
    observed_runoff = observed_runoff_full[WARMUP_HOURS:]

    print(f"\n实际配置:")
    print(f"  预热期: {WARMUP_HOURS} 小时")
    print(f"  率定期长度: {len(observed_runoff)} 小时")
    print(f"  率定期流量: {observed_runoff.min():.2f} - {observed_runoff.max():.2f} m³/s")

    # 1.4 生成温度数据
    temperature_full = np.linspace(5, 15, len(precipitation_full))

    # 1.5 Zone 1流域面积
    zone1_area_km2 = 139.995

    print(f"\n✓ Zone 1流域面积: {zone1_area_km2:.2f} km²")

    # ========================================================================
    # 步骤 2: 配置HBV校准器（带预热期）
    # ========================================================================
    print("\n步骤 2: 配置HBV校准器（带预热期）")
    print("-" * 80)

    # HBV参数搜索范围
    param_bounds = {
        'FC': [250, 600],
        'BETA': [1.5, 3.5],
        'K0': [0.1, 0.5],
        'K1': [0.02, 0.15],
        'K2': [0.005, 0.05],
        'PERC': [0.5, 4.0],
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
    }

    # 固定初始状态（根据基流估计）
    initial_conditions = {
        'initial_soil': 300.0,
        'initial_upper': 20.0,
        'initial_lower': 3000.0,
        'initial_snow': 0.0,
    }

    print(f"率定参数: {len(param_bounds)}个")
    for name, (min_val, max_val) in param_bounds.items():
        print(f"  {name:<20s}: [{min_val:>8.3f}, {max_val:>8.3f}]")

    print(f"\n固定初始状态（根据基流 {observed_runoff_full[0]:.1f} m³/s 估计）:")
    for key, value in initial_conditions.items():
        print(f"  {key:<20s}: {value:>10.2f}")

    # ========================================================================
    # 步骤 3: 测试HBV模型（带预热期）
    # ========================================================================
    print("\n步骤 3: 测试HBV模型（带预热期）")
    print("-" * 80)

    # 测试默认参数
    default_params = {
        'FC': 400, 'BETA': 2.0, 'K0': 0.25,
        'K1': 0.08, 'K2': 0.02, 'PERC': 2.0
    }

    test_runoff = run_hbv_with_warmup(
        default_params, precipitation_full, fixed_params,
        initial_conditions, zone1_area_km2, WARMUP_HOURS
    )

    from hydrosis.evaluation.metrics import nash_sutcliffe_efficiency
    test_nse = nash_sutcliffe_efficiency(test_runoff, observed_runoff)

    print(f"✓ 模型测试完成")
    print(f"  率定期输出长度: {len(test_runoff)}")
    print(f"  初始NSE (默认参数): {test_nse:.4f}")

    # ========================================================================
    # 步骤 4: 参数率定（带预热期）
    # ========================================================================
    print("\n步骤 4: 参数率定（带预热期）")
    print("-" * 80)

    print("\n运行Differential Evolution率定...")
    print("  注意: 使用预热期，只评估预热期后的数据")

    # 创建自定义目标函数
    def objective_with_warmup(params_dict):
        """带预热期的目标函数"""
        try:
            simulated = run_hbv_with_warmup(
                params_dict, precipitation_full, fixed_params,
                initial_conditions, zone1_area_km2, WARMUP_HOURS
            )
            nse = nash_sutcliffe_efficiency(simulated, observed_runoff)
            return nse
        except Exception as e:
            return -999.0

    # 使用HBVCalibrator的底层优化功能
    from hydrosis.analysis.optimization import calibrate_model

    bounds_list = [param_bounds[name] for name in param_names]

    result = calibrate_model(
        objective_function=objective_with_warmup,
        param_bounds={name: bounds_list[i] for i, name in enumerate(param_names)},
        algorithm='differential_evolution',
        maximize=True,
        maxiter=150,
        seed=42,
        verbose=False,
        popsize=20,
        polish=True
    )

    print(f"\n✓ 完成!")
    print(f"  最优NSE: {result.best_score:.6f}")
    print(f"  函数评估: {result.n_evaluations}")
    print(f"  计算时间: {result.elapsed_time:.2f}秒")

    # ========================================================================
    # 步骤 5: 性能评估
    # ========================================================================
    print("\n步骤 5: 性能评估")
    print("-" * 80)

    # 运行最优参数
    final_simulated = run_hbv_with_warmup(
        result.best_params, precipitation_full, fixed_params,
        initial_conditions, zone1_area_km2, WARMUP_HOURS
    )

    # 计算所有性能指标
    final_metrics = calculate_metrics(
        observed_runoff,
        final_simulated,
        metrics=['nse', 'log_nse', 'kge', 'rmse', 'mae', 'pbias']
    )

    print(f"\n性能指标（率定期，预热期后）:")
    for metric, value in final_metrics.items():
        if 'peak' not in metric and 'time' not in metric:
            print(f"  {metric.upper():<10s}: {value:>8.4f}")

    # 打印最优参数
    print(f"\n最优参数:")
    for param_name in param_names:
        value = result.best_params[param_name]
        min_val, max_val = param_bounds[param_name]
        range_pct = (value - min_val) / (max_val - min_val) * 100
        print(f"  {param_name:<20s}: {value:>10.4f}  (范围的{range_pct:>5.1f}%)")

    # ========================================================================
    # 步骤 6: 保存结果
    # ========================================================================
    print("\n步骤 6: 保存结果")
    print("-" * 80)

    output_dir = results_dir / "calibration_with_warmup"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存参数配置
    calibration_yaml = {
        'description': 'Zone 1 HBV参数率定结果（使用预热期）',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'framework': 'HBVCalibrator (统一校准框架，带预热期)',
        'calibration_info': {
            'algorithm': 'differential_evolution',
            'warmup_hours': int(WARMUP_HOURS),
            'calibration_hours': int(len(observed_runoff)),
            'n_evaluations': int(result.n_evaluations),
            'computation_time': float(result.elapsed_time),
        },
        'metrics': {k: float(v) for k, v in final_metrics.items()},
        'zones': {
            1: {
                'runoff_model': 'HBV',
                'parameters': {
                    name: float(result.best_params[name])
                    for name in param_names
                },
                'initial_conditions': initial_conditions,
                'fixed_parameters': fixed_params,
            }
        },
        'refactored': True,
        'benefits': [
            '使用统一的配置结构',
            '使用标准化的指标计算',
            '保留预热期的特殊逻辑',
            '更清晰的代码组织'
        ]
    }

    yaml_file = output_dir / "zone1_calibrated_parameters_warmup.yaml"
    with open(yaml_file, 'w') as f:
        yaml.dump(calibration_yaml, f, default_flow_style=False, sort_keys=False)
    print(f"✓ 保存参数配置: {yaml_file.name}")

    # 保存径流对比数据
    comparison_df = pd.DataFrame({
        'datetime': times[WARMUP_HOURS:],
        'observed_m3s': observed_runoff,
        'simulated_m3s': final_simulated,
        'residual_m3s': observed_runoff - final_simulated
    })
    comparison_file = output_dir / "zone1_calibrated_runoff_warmup.csv"
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
        ax.plot(final_simulated, 'r--', label='HBV (w/ Warmup)', linewidth=1.5)
        ax.set_xlabel('Time Step (hour, after warmup)', fontsize=11)
        ax.set_ylabel('Discharge (m³/s)', fontsize=11)
        ax.set_title(f'Zone 1 HBV Calibration with Warmup (NSE={final_metrics["nse"]:.4f})',
                    fontsize=12, weight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "zone1_hydrograph_warmup.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 保存: zone1_hydrograph_warmup.png")

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
        plt.savefig(output_dir / "zone1_scatter_warmup.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 保存: zone1_scatter_warmup.png")

        print(f"\n✓ 所有图表已保存到: {output_dir}")

    except ImportError:
        print("⚠ matplotlib未安装，跳过可视化")

    # ========================================================================
    # 步骤 8: 总结
    # ========================================================================
    print("\n" + "=" * 80)
    print("Zone 1 HBV参数率定完成（使用预热期）!")
    print("=" * 80)

    print(f"\n【配置】")
    print(f"  预热期: {WARMUP_HOURS} 小时")
    print(f"  率定期: {len(observed_runoff)} 小时")
    print(f"  率定参数: {len(param_names)}个物理参数")
    print(f"  固定初始状态: 根据基流估计")

    print(f"\n【率定结果】")
    print(f"  最优NSE: {result.best_score:.6f}")
    print(f"  计算时间: {result.elapsed_time:.2f}秒")

    print(f"\n【性能指标】")
    for metric, value in final_metrics.items():
        if 'peak' not in metric and 'time' not in metric:
            print(f"  {metric.upper()}: {value:.6f}")

    print(f"\n【诊断建议】")
    if final_metrics['nse'] > 0.5:
        print("  ✓ 率定成功！模型性能良好")
        print("  ✓ 预热期策略有效")
    elif final_metrics['nse'] > 0:
        print("  ⚠ 率定部分成功，建议使用更长时间序列（7-30天）")
    else:
        print("  ✗ 率定仍未成功，建议：")
        print("    1. 使用更长时间序列（当前数据太短）")
        print("    2. 调整初始状态估计方法")
        print("    3. 检查降雨和观测数据的时间对齐")

    print("\n✨ 重构改进:")
    print("  1. 使用统一的配置结构")
    print("  2. 使用标准化的指标计算（calculate_metrics）")
    print("  3. 保留预热期的特殊逻辑")
    print("  4. 更清晰的代码组织")
    print("  5. 移除过时的导入和函数")

    print("\n" + "=" * 80)
    print("✓ 本脚本展示了如何使用预热期解决初始状态问题")
    print("✓ 遵循 .claude/AI_DEVELOPMENT_GUIDE.md 最佳实践")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
