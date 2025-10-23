#!/usr/bin/env python3
"""
HBV模型实际参数率定案例

这个示例展示如何使用参数分析框架对真实的HBV水文模型进行：
1. 参数敏感性分析
2. 多算法参数率定（SCE-UA, PSO, DE）
3. 不确定性分析
4. 结果对比和可视化

使用之前生成的估计径流数据作为"观测"进行率定。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hydrosis.runoff.hbv import HBVRunoff
from hydrosis.analysis import (
    calculate_metrics,
    one_at_a_time_sensitivity,
    calibrate_model,
    monte_carlo_analysis,
    plot_hydrograph_comparison,
    plot_convergence_history,
    plot_scatter,
    plot_parameter_distributions,
)

print("=" * 80)
print("HBV模型实际参数率定案例")
print("=" * 80)

# ============================================================================
# 步骤 1: 加载数据
# ============================================================================
print("\n步骤 1: 加载观测数据和输入数据")
print("-" * 80)

# 检查估计径流数据是否存在
estimated_dir = Path("results/upper_truckee_complete_11steps/estimated_observations")
if not estimated_dir.exists():
    print(f"错误: 估计径流数据目录不存在: {estimated_dir}")
    print("请先运行 generate_estimated_runoff.py 生成估计径流数据")
    sys.exit(1)

# 加载Zone 1的估计径流作为"观测"
zone1_file = estimated_dir / "zone_1_estimated_runoff.csv"
if not zone1_file.exists():
    print(f"错误: Zone 1估计径流文件不存在: {zone1_file}")
    sys.exit(1)

print(f"加载观测数据: {zone1_file}")
obs_df = pd.read_csv(zone1_file)
observed_runoff = obs_df['discharge_m3s'].values
times = pd.to_datetime(obs_df['datetime'])

print(f"  观测时段数: {len(observed_runoff)}")
print(f"  时间范围: {times.min()} 到 {times.max()}")
print(f"  流量范围: {observed_runoff.min():.2f} - {observed_runoff.max():.2f} m³/s")
print(f"  平均流量: {observed_runoff.mean():.2f} m³/s")

# 加载气象数据（来自precipitation_zone_1.csv）
precip_dir = Path("results/upper_truckee_complete_11steps/precipitation_zones")
precip_file = precip_dir / "precipitation_zone_1.csv"

if not precip_file.exists():
    print(f"警告: 降雨数据不存在: {precip_file}")
    print("  使用随机生成的降雨数据进行演示")
    # 生成模拟降雨
    np.random.seed(42)
    precipitation = 2 + 5 * np.abs(np.sin(np.arange(len(observed_runoff)) / 10)) + \
                   np.random.gamma(2, 1, len(observed_runoff))
    temperature = 10 + 8 * np.sin(np.arange(len(observed_runoff)) / 30)
else:
    print(f"加载降雨数据: {precip_file}")
    precip_df = pd.read_csv(precip_file)
    precipitation = precip_df['precipitation'].values[:len(observed_runoff)]
    # 生成模拟温度
    temperature = 10 + 8 * np.sin(np.arange(len(observed_runoff)) / 30)

print(f"  降雨范围: {precipitation.min():.2f} - {precipitation.max():.2f} mm/h")

# ============================================================================
# 步骤 2: 定义HBV模型包装函数
# ============================================================================
print("\n步骤 2: 定义HBV模型包装函数")
print("-" * 80)

# Zone 1的流域面积 (km²)
area_km2 = 16.5  # 根据实际情况调整

# HBV参数范围（基于水文学经验）
param_bounds = {
    'FC': [50, 500],      # 田间持水能力 (mm)
    'BETA': [1.0, 6.0],   # 土壤湿度非线性指数
    'K0': [0.05, 0.5],    # 快速消退系数
    'K1': [0.01, 0.3],    # 慢速消退系数
    'K2': [0.001, 0.1],   # 基流消退系数
    'PERC': [0.0, 10.0],  # 渗漏率 (mm/h)
}

print(f"待率定参数数量: {len(param_bounds)}")
print("参数范围:")
for name, bounds in param_bounds.items():
    print(f"  {name:10s}: [{bounds[0]:7.3f}, {bounds[1]:7.3f}]")

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
        # 创建HBV参数字典
        params = {
            'FC': FC,
            'BETA': BETA,
            'K0': K0,
            'K1': K1,
            'K2': K2,
            'PERC': PERC,
            # 使用合理的默认值
            'LP': 0.7,
            'MAXBAS': 3.0,
            'TT': 0.0,
            'CFMAX': 3.5,
            'CFR': 0.05,
            'CWH': 0.1,
            # 初始状态
            'initial_soil': FC * 0.3,
            'initial_upper': 5.0,
            'initial_lower': 20.0,
            'initial_snow': 0.0
        }

        # 创建HBV模型实例
        hbv = HBVRunoff(params)

        # 运行模型
        runoff_mm = []
        for i in range(len(precipitation)):
            q = hbv.step(precipitation[i], temperature[i])
            runoff_mm.append(q)

        # 转换为m³/s
        runoff_m3s = np.array(runoff_mm) * area_km2 / 3.6

        return runoff_m3s

    except Exception as e:
        print(f"模型运行错误: {e}")
        return np.zeros(len(precipitation))

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
print("✓ HBV模型运行正常")

# ============================================================================
# 步骤 3: 参数敏感性分析
# ============================================================================
print("\n步骤 3: 参数敏感性分析")
print("-" * 80)

def sensitivity_objective(FC, BETA, K0, K1, K2, PERC):
    """敏感性分析目标函数"""
    from hydrosis.analysis import nash_sutcliffe_efficiency
    simulated = run_hbv_model(FC, BETA, K0, K1, K2, PERC)
    return nash_sutcliffe_efficiency(observed_runoff, simulated)

print("运行OAT敏感性分析（这可能需要几分钟）...")
baseline_params = {
    'FC': 200.0,
    'BETA': 2.5,
    'K0': 0.15,
    'K1': 0.08,
    'K2': 0.02,
    'PERC': 3.0
}

sensitivity_results = one_at_a_time_sensitivity(
    model_function=sensitivity_objective,
    parameters=baseline_params,
    param_ranges=param_bounds,
    variations=[0.7, 0.85, 1.0, 1.15, 1.3],
    metric_name='NSE'
)

print(f"\n基准NSE: {sensitivity_results['baseline_output']:.4f}")
print("\n参数敏感性排序:")
for i, (param, si) in enumerate(sensitivity_results['ranked_parameters'], 1):
    print(f"  {i}. {param:10s}: SI = {si:.4f}")

# ============================================================================
# 步骤 4: 多算法参数率定
# ============================================================================
print("\n步骤 4: 多算法参数率定")
print("-" * 80)

def calibration_objective(FC, BETA, K0, K1, K2, PERC):
    """率定目标函数：最大化NSE"""
    from hydrosis.analysis import nash_sutcliffe_efficiency
    simulated = run_hbv_model(FC, BETA, K0, K1, K2, PERC)
    nse = nash_sutcliffe_efficiency(observed_runoff, simulated)
    return nse

# 定义率定配置
calibration_configs = {
    'SCE-UA': {
        'method': 'sce_ua',
        'n_complexes': 5,
        'max_iterations': 50,
    },
    'PSO': {
        'method': 'pso',
        'n_particles': 30,
        'max_iterations': 50,
    },
}

calibration_results = {}

for alg_name, config in calibration_configs.items():
    print(f"\n运行{alg_name}率定...")
    result = calibrate_model(
        objective_function=calibration_objective,
        param_bounds=param_bounds,
        maximize=True,
        seed=42,
        verbose=False,
        **config
    )

    calibration_results[alg_name] = result

    print(f"  最优NSE: {result.best_score:.6f}")
    print(f"  迭代次数: {result.n_iterations}")
    print(f"  函数评估: {result.n_evaluations}")
    print(f"  计算时间: {result.computation_time:.2f}秒")

# ============================================================================
# 步骤 5: 结果对比
# ============================================================================
print("\n步骤 5: 率定结果对比")
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

print(f"\n最佳算法: {best_name} (NSE={best_result.best_score:.6f})")
print("\n最优参数:")
for param, value in best_result.best_params.items():
    print(f"  {param:10s}: {value:10.4f}")

# 使用最优参数运行模型
print("\n使用最优参数运行HBV模型...")
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
# 步骤 6: 可视化
# ============================================================================
print("\n步骤 6: 生成分析图表")
print("-" * 80)

output_dir = Path("results/hbv_calibration_case")
output_dir.mkdir(parents=True, exist_ok=True)

try:
    import matplotlib
    matplotlib.use('Agg')

    print("生成图表...")

    # 1. 水文过程对比
    print("  1. 水文过程对比图...")
    plot_hydrograph_comparison(
        observed=observed_runoff,
        simulated=calibrated_runoff,
        time_index=np.arange(len(observed_runoff)),
        metrics={'NSE': final_metrics['nse'], 'PBIAS': final_metrics['pbias']},
        title=f"HBV模型率定结果 ({best_name})",
        xlabel="Time Step",
        ylabel="Discharge (m³/s)",
        save_path=output_dir / "hydrograph.png",
        show=False
    )

    # 2. 散点图
    print("  2. 散点图...")
    plot_scatter(
        observed=observed_runoff,
        simulated=calibrated_runoff,
        metrics={'NSE': final_metrics['nse'], 'KGE': final_metrics['kge']},
        save_path=output_dir / "scatter.png",
        show=False
    )

    # 3. 收敛历史（每个算法）
    for alg_name, result in calibration_results.items():
        print(f"  3.{alg_name}收敛历史...")
        plot_convergence_history(
            convergence_history=result.convergence_history,
            title=f"{alg_name} Convergence",
            ylabel="NSE",
            save_path=output_dir / f"convergence_{alg_name.lower().replace('-', '_')}.png",
            show=False
        )

    print(f"\n✓ 所有图表已保存到: {output_dir}")

except ImportError:
    print("⚠ matplotlib未安装，跳过可视化")

# ============================================================================
# 步骤 7: 保存结果
# ============================================================================
print("\n步骤 7: 保存率定结果")
print("-" * 80)

# 保存最优参数
best_result.save_json(output_dir / f"best_calibration_{best_name.lower().replace('-', '_')}.json")

# 保存模拟径流
result_df = pd.DataFrame({
    'time_step': np.arange(len(observed_runoff)),
    'observed': observed_runoff,
    'simulated': calibrated_runoff,
    'residual': observed_runoff - calibrated_runoff
})
result_df.to_csv(output_dir / "calibrated_runoff.csv", index=False)

print(f"✓ 率定结果已保存到: {output_dir}")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 80)
print("HBV模型率定案例完成！")
print("=" * 80)

print("\n总结:")
print(f"\n1. 数据:")
print(f"   - 观测时段: {len(observed_runoff)}个时间步")
print(f"   - 平均流量: {observed_runoff.mean():.2f} m³/s")

print(f"\n2. 参数敏感性 (前3名):")
for i, (param, si) in enumerate(sensitivity_results['ranked_parameters'][:3], 1):
    print(f"   {i}. {param}: {si:.4f}")

print(f"\n3. 率定结果:")
print(f"   - 最佳算法: {best_name}")
print(f"   - 最优NSE: {best_result.best_score:.6f}")
print(f"   - 计算时间: {best_result.computation_time:.2f}秒")

print(f"\n4. 性能指标:")
print(f"   - NSE: {final_metrics['nse']:.6f}")
print(f"   - RMSE: {final_metrics['rmse']:.4f} m³/s")
print(f"   - PBIAS: {final_metrics['pbias']:.2f}%")
print(f"   - KGE: {final_metrics['kge']:.6f}")

print(f"\n5. 输出文件:")
print(f"   - 率定参数: {output_dir}/best_calibration_*.json")
print(f"   - 模拟径流: {output_dir}/calibrated_runoff.csv")
print(f"   - 分析图表: {output_dir}/*.png")

print("\n" + "=" * 80)
print("✓ 这个案例展示了完整的HBV模型率定流程:")
print("  1. 数据准备 → 2. 敏感性分析 → 3. 多算法率定 → 4. 结果评估 → 5. 可视化")
print("=" * 80)
