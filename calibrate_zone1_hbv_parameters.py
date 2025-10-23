#!/usr/bin/env python3
"""
Zone 1 HBV参数自动率定脚本

使用参数分析框架对最上游分区(Zone 1)进行自动参数率定。
集成以下模块：
1. hydrosis.calibration - 参数管理和YAML配置
2. hydrosis.analysis - 参数率定优化算法
3. hydrosis.runoff.hbv - HBV水文模型
4. 已有的降雨和观测数据

率定流程：
1. 加载Zone 1的降雨和观测径流数据
2. 定义HBV模型和目标函数
3. 设置参数搜索范围
4. 运行多算法率定（SCE-UA, PSO）
5. 保存最优参数到YAML配置文件
6. 使用最优参数重新运行模型验证
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hydrosis.runoff.hbv import HBVRunoff
from hydrosis.analysis import (
    calculate_metrics,
    calibrate_model,
    plot_hydrograph_comparison,
    plot_convergence_history,
    plot_scatter,
)

print("=" * 80)
print("Zone 1 HBV参数自动率定")
print("=" * 80)

# ============================================================================
# 步骤 1: 加载数据
# ============================================================================
print("\n步骤 1: 加载Zone 1的数据")
print("-" * 80)

results_dir = Path("results/upper_truckee_complete_11steps")

# 1.1 加载Zone 1的观测径流（使用估计径流作为"观测"）
obs_file = results_dir / "estimated_observations" / "zone_1_estimated_runoff.csv"
if not obs_file.exists():
    print(f"错误: 观测数据不存在: {obs_file}")
    print("请先运行 generate_estimated_runoff.py")
    sys.exit(1)

obs_df = pd.read_csv(obs_file)
observed_runoff = obs_df['discharge_m3s'].values
times = pd.to_datetime(obs_df['datetime'])

print(f"✓ 加载观测径流: {obs_file}")
print(f"  时间范围: {times.min()} 到 {times.max()}")
print(f"  数据点数: {len(observed_runoff)}")
print(f"  流量范围: {observed_runoff.min():.2f} - {observed_runoff.max():.2f} m³/s")
print(f"  平均流量: {observed_runoff.mean():.2f} m³/s")

# 1.2 加载Zone 1的面雨量数据
precip_file = results_dir / "step_08_areal_rainfall" / "8.2_subbasin_areal_precipitation.csv"
if not precip_file.exists():
    print(f"错误: 降雨数据不存在: {precip_file}")
    sys.exit(1)

precip_df = pd.read_csv(precip_file, index_col=0)

# Zone 1的子分区ID: 101-114
zone1_subbasins = [str(i) for i in range(101, 115)]
zone1_cols = [col for col in precip_df.columns if col in zone1_subbasins]

if len(zone1_cols) == 0:
    print(f"错误: 未找到Zone 1的子分区数据")
    print(f"  可用列: {precip_df.columns[:10].tolist()}")
    sys.exit(1)

# 计算Zone 1的平均面雨量
precipitation = precip_df[zone1_cols].mean(axis=1).values

print(f"\n✓ 加载降雨数据: {precip_file}")
print(f"  Zone 1子分区: {len(zone1_cols)}个")
print(f"  降雨范围: {precipitation.min():.2f} - {precipitation.max():.2f} mm/h")
print(f"  平均降雨: {precipitation.mean():.2f} mm/h")

# 1.3 生成模拟温度数据（简化处理）
temperature = 10 + 8 * np.sin(np.arange(len(precipitation)) / 30)
print(f"\n✓ 生成温度数据（简化）")
print(f"  温度范围: {temperature.min():.1f} - {temperature.max():.1f} °C")

# 1.4 Zone 1流域面积
zone1_area_km2 = 139.995  # 从parameter_zones.csv

print(f"\n✓ Zone 1流域面积: {zone1_area_km2:.2f} km²")

# ============================================================================
# 步骤 2: 定义HBV模型包装函数
# ============================================================================
print("\n步骤 2: 定义HBV模型和目标函数")
print("-" * 80)

# 固定的HBV参数（不参与率定）
fixed_params = {
    'LP': 0.7,          # 限制蒸散发土壤湿度阈值
    'MAXBAS': 3.0,      # 转换函数的routing参数
    'TT': 0.0,          # 雨雪分界温度
    'CFMAX': 3.5,       # 度日因子
    'CFR': 0.05,        # 冻结系数
    'CWH': 0.1,         # 持水能力
}

# 待率定参数的搜索范围（基于水文学经验）
param_bounds = {
    'FC': [50, 500],       # 田间持水能力 (mm)
    'BETA': [1.0, 6.0],    # 土壤湿度非线性指数
    'K0': [0.05, 0.5],     # 快速消退系数 (1/h)
    'K1': [0.01, 0.3],     # 慢速消退系数 (1/h)
    'K2': [0.001, 0.1],    # 基流消退系数 (1/h)
    'PERC': [0.0, 10.0],   # 渗漏率 (mm/h)
}

print(f"待率定参数: {len(param_bounds)}个")
for name, bounds in param_bounds.items():
    print(f"  {name:10s}: [{bounds[0]:7.3f}, {bounds[1]:7.3f}]")

print(f"\n固定参数: {len(fixed_params)}个")
for name, value in fixed_params.items():
    print(f"  {name:10s}: {value:7.3f}")

def run_hbv_model(FC, BETA, K0, K1, K2, PERC):
    """运行HBV模型并返回径流序列

    Args:
        FC: 田间持水能力 (mm)
        BETA: 土壤湿度非线性指数
        K0: 快速消退系数
        K1: 慢速消退系数
        K2: 基流消退系数
        PERC: 渗漏率 (mm/h)

    Returns:
        径流序列 (m³/s)
    """
    try:
        # 创建完整的HBV参数字典
        params = {
            'FC': FC,
            'BETA': BETA,
            'K0': K0,
            'K1': K1,
            'K2': K2,
            'PERC': PERC,
            # 添加固定参数
            **fixed_params,
            # 初始状态（基于FC）
            'initial_soil': FC * 0.3,
            'initial_upper': 5.0,
            'initial_lower': 20.0,
            'initial_snow': 0.0
        }

        # 创建HBV模型实例
        hbv = HBVRunoff(params)

        # 创建一个模拟的Subbasin对象（HBV.simulate需要）
        class MockSubbasin:
            def __init__(self, area_km2):
                self.area_km2 = area_km2

        subbasin = MockSubbasin(zone1_area_km2)

        # 运行HBV模型（HBV.simulate已经返回m³/s，不需要额外转换！）
        runoff_m3s = hbv.simulate(subbasin, precipitation.tolist())

        return np.array(runoff_m3s)

    except Exception as e:
        print(f"模型运行错误: {e}")
        import traceback
        traceback.print_exc()
        return np.zeros(len(precipitation))

def calibration_objective(FC, BETA, K0, K1, K2, PERC):
    """率定目标函数：最大化NSE

    Returns:
        NSE值（越接近1越好）
    """
    from hydrosis.analysis import nash_sutcliffe_efficiency

    simulated = run_hbv_model(FC, BETA, K0, K1, K2, PERC)
    nse = nash_sutcliffe_efficiency(observed_runoff, simulated)

    return nse

# 测试模型运行
print("\n测试HBV模型运行...")
test_params = {
    'FC': 150.0,
    'BETA': 2.0,
    'K0': 0.2,
    'K1': 0.05,
    'K2': 0.01,
    'PERC': 2.0
}
test_runoff = run_hbv_model(**test_params)
print(f"  模型输出长度: {len(test_runoff)}")
print(f"  输出范围: {test_runoff.min():.2f} - {test_runoff.max():.2f} m³/s")

# 计算初始NSE
initial_nse = calibration_objective(**test_params)
print(f"  初始NSE (默认参数): {initial_nse:.4f}")

# ============================================================================
# 步骤 3: 多算法参数率定
# ============================================================================
print("\n步骤 3: 多算法参数率定")
print("-" * 80)

# 定义率定配置（高精度模式）
calibration_configs = {
    'SCE-UA': {
        'method': 'sce_ua',
        'n_complexes': 5,
        'max_iterations': 100,  # 提高到100以获得更高精度
        'patience': 20,
    },
    'PSO': {
        'method': 'pso',
        'n_particles': 40,  # 增加粒子数
        'max_iterations': 100,  # 提高到100
        'w': 0.7,
        'c1': 1.5,
        'c2': 1.5,
        'patience': 20,
    },
}

calibration_results = {}

for alg_name, config in calibration_configs.items():
    print(f"\n运行{alg_name}率定...")
    print(f"  这可能需要几分钟时间...")

    result = calibrate_model(
        objective_function=calibration_objective,
        param_bounds=param_bounds,
        maximize=True,
        seed=42,
        verbose=False,
        **config
    )

    calibration_results[alg_name] = result

    print(f"  ✓ 完成!")
    print(f"    最优NSE: {result.best_score:.6f}")
    print(f"    迭代次数: {result.n_iterations}")
    print(f"    函数评估: {result.n_evaluations}")
    print(f"    计算时间: {result.computation_time:.2f}秒")

# ============================================================================
# 步骤 4: 结果对比
# ============================================================================
print("\n步骤 4: 率定结果对比")
print("-" * 80)

print(f"\n{'算法':<15s} {'NSE':>10s} {'时间(s)':>10s} {'评估次数':>10s}")
print("-" * 50)
for name, result in calibration_results.items():
    print(f"{name:<15s} {result.best_score:>10.6f} {result.computation_time:>10.2f} "
          f"{result.n_evaluations:>10d}")

# 选择最佳算法
best_alg = max(calibration_results.items(), key=lambda x: x[1].best_score)
best_name = best_alg[0]
best_result = best_alg[1]

print(f"\n✓ 最佳算法: {best_name} (NSE={best_result.best_score:.6f})")
print("\n最优参数:")
for param, value in best_result.best_params.items():
    bounds = param_bounds[param]
    pct = (value - bounds[0]) / (bounds[1] - bounds[0]) * 100
    print(f"  {param:10s}: {value:10.4f}  (搜索空间的{pct:5.1f}%)")

# ============================================================================
# 步骤 5: 使用最优参数运行模型
# ============================================================================
print("\n步骤 5: 使用最优参数运行HBV模型")
print("-" * 80)

calibrated_runoff = run_hbv_model(**best_result.best_params)

# 计算详细性能指标
final_metrics = calculate_metrics(
    observed_runoff,
    calibrated_runoff,
    metrics=['nse', 'rmse', 'mae', 'pbias', 'kge', 'log_nse']
)

print("\n率定后性能指标:")
for metric, value in final_metrics.items():
    if 'peak' not in metric and 'time' not in metric:
        print(f"  {metric.upper():10s}: {value:8.4f}")

# ============================================================================
# 步骤 6: 保存最优参数到YAML配置文件
# ============================================================================
print("\n步骤 6: 保存最优参数到YAML配置文件")
print("-" * 80)

output_dir = results_dir / "calibration"
output_dir.mkdir(parents=True, exist_ok=True)

# 创建参数率定YAML配置
calibration_yaml = {
    'description': f'Zone 1 HBV参数率定结果 ({best_name})',
    'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'calibration_info': {
        'algorithm': best_name,
        'nse': float(best_result.best_score),
        'rmse': float(final_metrics['rmse']),
        'pbias': float(final_metrics['pbias']),
        'n_iterations': int(best_result.n_iterations),
        'n_evaluations': int(best_result.n_evaluations),
        'computation_time_seconds': float(best_result.computation_time),
    },
    'global_defaults': {
        'hbv': {
            **{k: float(v) for k, v in best_result.best_params.items()},
            **{k: float(v) for k, v in fixed_params.items()},
            'UZL': 5.0,
            'initial_soil': float(best_result.best_params['FC'] * 0.3),
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
                }
            }
        }
    ],
    'subbasin_overrides': None,
}

# 保存YAML文件
yaml_file = output_dir / "zone1_calibrated_parameters.yaml"
with open(yaml_file, 'w') as f:
    yaml.dump(calibration_yaml, f, default_flow_style=False, sort_keys=False)

print(f"✓ 保存参数配置: {yaml_file}")

# 同时保存JSON格式（用于程序化读取）
json_file = output_dir / f"zone1_calibration_{best_name.lower().replace('-', '_')}.json"
best_result.save_json(json_file)
print(f"✓ 保存率定结果: {json_file}")

# ============================================================================
# 步骤 7: 可视化
# ============================================================================
print("\n步骤 7: 生成分析图表")
print("-" * 80)

try:
    import matplotlib
    matplotlib.use('Agg')

    # 7.1 水文过程对比图
    print("  1. 水文过程对比图...")
    plot_hydrograph_comparison(
        observed=observed_runoff,
        simulated=calibrated_runoff,
        time_index=np.arange(len(observed_runoff)),
        metrics={'NSE': final_metrics['nse'], 'PBIAS': final_metrics['pbias']},
        title=f"Zone 1 HBV模型率定结果 ({best_name})",
        xlabel="Time Step (hours)",
        ylabel="Discharge (m³/s)",
        save_path=output_dir / "zone1_hydrograph.png",
        show=False
    )

    # 7.2 散点图
    print("  2. 散点图...")
    plot_scatter(
        observed=observed_runoff,
        simulated=calibrated_runoff,
        metrics={'NSE': final_metrics['nse'], 'KGE': final_metrics['kge']},
        save_path=output_dir / "zone1_scatter.png",
        show=False
    )

    # 7.3 收敛历史（每个算法）
    for alg_name, result in calibration_results.items():
        print(f"  3. {alg_name}收敛历史...")
        plot_convergence_history(
            convergence_history=result.convergence_history,
            title=f"{alg_name} Convergence (Zone 1)",
            ylabel="NSE",
            save_path=output_dir / f"zone1_convergence_{alg_name.lower().replace('-', '_')}.png",
            show=False
        )

    print(f"\n✓ 所有图表已保存到: {output_dir}")

except ImportError:
    print("⚠ matplotlib未安装，跳过可视化")

# ============================================================================
# 步骤 8: 保存模拟结果
# ============================================================================
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

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 80)
print("Zone 1 HBV参数率定完成！")
print("=" * 80)

print(f"\n1. 率定数据:")
print(f"   - 时间步数: {len(observed_runoff)}")
print(f"   - 平均流量: {observed_runoff.mean():.2f} m³/s")
print(f"   - 峰值流量: {observed_runoff.max():.2f} m³/s")

print(f"\n2. 率定结果:")
print(f"   - 最佳算法: {best_name}")
print(f"   - 最优NSE: {best_result.best_score:.6f}")
print(f"   - 计算时间: {best_result.computation_time:.2f}秒")

print(f"\n3. 性能指标:")
print(f"   - NSE: {final_metrics['nse']:.6f}")
print(f"   - RMSE: {final_metrics['rmse']:.4f} m³/s")
print(f"   - PBIAS: {final_metrics['pbias']:.2f}%")
print(f"   - KGE: {final_metrics['kge']:.6f}")

print(f"\n4. 最优参数:")
for param, value in best_result.best_params.items():
    print(f"   - {param}: {value:.4f}")

print(f"\n5. 输出文件:")
print(f"   - 参数配置: {yaml_file.name}")
print(f"   - 率定结果: {json_file.name}")
print(f"   - 径流数据: zone1_calibrated_runoff.csv")
print(f"   - 分析图表: zone1_*.png")

print(f"\n6. 下一步操作:")
print(f"   使用率定后的参数重新运行完整工作流:")
print(f"   python rerun_step09_10.py --calibration {yaml_file}")

print("\n" + "=" * 80)
print("✓ Zone 1参数率定完成！参数已保存到YAML配置文件。")
print("=" * 80)
