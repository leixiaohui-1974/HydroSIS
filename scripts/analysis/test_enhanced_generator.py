#!/usr/bin/env python3
"""
测试增强型径流生成器

1. 使用增强型生成器生成"观测"径流
2. 对比简单线性法和增强法的区别
3. 分析增强生成器的径流分量
4. 准备用于HBV率定的观测数据
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hydrosis.runoff.enhanced_generator import EnhancedRunoffGenerator

print("=" * 80)
print("增强型径流生成器测试")
print("=" * 80)

# ============================================================================
# 1. 加载降雨数据
# ============================================================================
print("\n步骤 1: 加载降雨数据")
print("-" * 80)

results_dir = Path("results/upper_truckee_complete_11steps")
precip_file = results_dir / "step_08_areal_rainfall" / "8.2_subbasin_areal_precipitation.csv"

precip_df = pd.read_csv(precip_file, index_col=0)
# Zone 1的子分区ID: 10-23（从3.4_subzone_statistics.csv获取）
zone1_subbasins = [str(i) for i in range(10, 24)]
zone1_cols = [col for col in precip_df.columns if col in zone1_subbasins]

if len(zone1_cols) == 0:
    print(f"  ✗ 错误: 未找到Zone 1的子分区数据")
    print(f"  可用列: {precip_df.columns[:20].tolist()}")
    sys.exit(1)

precipitation = precip_df[zone1_cols].mean(axis=1).values

print(f"✓ 加载降雨数据: {precip_file}")
print(f"  Zone 1子分区: {len(zone1_cols)}个")
print(f"  时间步数: {len(precipitation)}")
print(f"  降雨范围: {precipitation.min():.2f} - {precipitation.max():.2f} mm/h")
print(f"  平均降雨: {precipitation.mean():.2f} mm/h")
print(f"  总降雨量: {precipitation.sum():.2f} mm")

zone1_area_km2 = 139.995

# ============================================================================
# 2. 测试不同配置的增强型生成器
# ============================================================================
print("\n步骤 2: 测试增强型径流生成器")
print("-" * 80)

# 配置1: 低径流系数配置（Rc ≈ 0.35）
config1 = {
    'soil_capacity': 500.0,      # 更大的土壤容量
    'soil_beta': 1.8,            # 更线性
    'fast_threshold': 0.75,      # 更晚产生快速径流
    'fast_ratio': 0.35,
    'inter_ratio': 0.35,
    'base_ratio': 0.30,
    'k_fast': 0.25,
    'k_inter': 0.08,
    'k_base': 0.02,
    'initial_soil': 150.0,       # 较低的初始湿度
    'initial_fast': 3.0,
    'initial_inter': 8.0,
    'initial_base': 15.0,
    'et_rate': 1.2,              # 更高的蒸散发
    'time_step_hours': 1.0
}

# 配置2: 中等径流系数配置（Rc ≈ 0.41）
config2 = {
    'soil_capacity': 420.0,      # 减小土壤容量以提高Rc
    'soil_beta': 2.0,
    'fast_threshold': 0.70,      # 略早产生快速径流
    'fast_ratio': 0.40,          # 略增加快速径流比例
    'inter_ratio': 0.35,
    'base_ratio': 0.25,
    'k_fast': 0.30,              # 略快的响应
    'k_inter': 0.09,
    'k_base': 0.022,
    'initial_soil': 170.0,       # 略高的初始湿度
    'initial_fast': 4.5,
    'initial_inter': 11.0,
    'initial_base': 19.0,
    'et_rate': 0.9,              # 略降低蒸散发
    'time_step_hours': 1.0
}

# 配置3: 高径流系数配置（Rc ≈ 0.50）
config3 = {
    'soil_capacity': 380.0,
    'soil_beta': 2.2,
    'fast_threshold': 0.68,
    'fast_ratio': 0.42,
    'inter_ratio': 0.35,
    'base_ratio': 0.23,
    'k_fast': 0.30,
    'k_inter': 0.10,
    'k_base': 0.024,
    'initial_soil': 180.0,
    'initial_fast': 5.0,
    'initial_inter': 12.0,
    'initial_base': 20.0,
    'et_rate': 0.8,
    'time_step_hours': 1.0
}

configs = {
    '低径流配置(Rc≈0.35)': config1,
    '中等径流配置(Rc≈0.42)': config2,
    '高径流配置(Rc≈0.50)': config3
}

results = {}

for name, config in configs.items():
    print(f"\n测试配置: {name}")
    generator = EnhancedRunoffGenerator(**config)

    # 生成径流
    runoff_m3s, stats = generator.generate(
        precipitation, zone1_area_km2, return_components=True
    )

    print(f"  径流系数: {stats['runoff_coefficient']:.4f}")
    print(f"  平均流量: {stats['mean_runoff_m3s']:.2f} m³/s")
    print(f"  峰值流量: {stats['peak_runoff_m3s']:.2f} m³/s")
    print(f"  最小流量: {stats['min_runoff_m3s']:.2f} m³/s")

    results[name] = {
        'runoff_m3s': runoff_m3s,
        'stats': stats,
        'config': config
    }

# ============================================================================
# 3. 选择最佳配置并生成观测数据
# ============================================================================
print("\n步骤 3: 选择最佳配置生成观测数据")
print("-" * 80)

# 选择径流系数最接近0.41的配置
target_rc = 0.41
best_config_name = min(
    results.keys(),
    key=lambda x: abs(results[x]['stats']['runoff_coefficient'] - target_rc)
)

print(f"✓ 选择配置: {best_config_name}")
print(f"  目标Rc: {target_rc:.4f}")
print(f"  实际Rc: {results[best_config_name]['stats']['runoff_coefficient']:.4f}")

best_result = results[best_config_name]
best_runoff = best_result['runoff_m3s']
best_stats = best_result['stats']
best_config = best_result['config']

# ============================================================================
# 4. 生成可视化对比图
# ============================================================================
print("\n步骤 4: 生成可视化对比")
print("-" * 80)

output_dir = results_dir / "enhanced_observations"
output_dir.mkdir(parents=True, exist_ok=True)

# 图1: 三种配置的径流对比
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax1 = axes[0]
ax1.bar(range(len(precipitation)), precipitation, color='steelblue', alpha=0.5, label='Precipitation')
ax1.set_ylabel('Precipitation (mm/h)', fontsize=10)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
colors = ['red', 'green', 'blue']
for i, (name, result) in enumerate(results.items()):
    runoff = result['runoff_m3s']
    rc = result['stats']['runoff_coefficient']
    ax2.plot(runoff, label=f'{name} (Rc={rc:.3f})', color=colors[i], linewidth=1.5)

ax2.set_xlabel('Time Step (hour)', fontsize=10)
ax2.set_ylabel('Runoff (m³/s)', fontsize=10)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'config_comparison.png', dpi=300, bbox_inches='tight')
print(f"  ✓ 保存: config_comparison.png")
plt.close()

# 图2: 最佳配置的径流分量
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# 降雨
ax1 = axes[0]
ax1.bar(range(len(precipitation)), precipitation, color='steelblue', alpha=0.5)
ax1.set_ylabel('Precipitation (mm/h)', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_title('Enhanced Runoff Generator - Components Analysis', fontsize=12, weight='bold')

# 径流分量
ax2 = axes[1]
components = best_stats['components']
fast = components['fast_runoff'] * zone1_area_km2 / 3.6
inter = components['inter_runoff'] * zone1_area_km2 / 3.6
base = components['base_runoff'] * zone1_area_km2 / 3.6

ax2.plot(fast, label='Fast runoff', color='red', linewidth=1.5)
ax2.plot(inter, label='Interflow', color='orange', linewidth=1.5)
ax2.plot(base, label='Baseflow', color='blue', linewidth=1.5)
ax2.plot(best_runoff, label='Total', color='black', linewidth=2, linestyle='--')
ax2.set_ylabel('Runoff (m³/s)', fontsize=10)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# 土壤湿度
ax3 = axes[2]
soil_sat = components['soil_saturation']
ax3.fill_between(range(len(soil_sat)), 0, soil_sat, color='brown', alpha=0.5)
ax3.axhline(y=best_config['fast_threshold'], color='red', linestyle='--',
            label=f'Fast runoff threshold ({best_config["fast_threshold"]:.2f})')
ax3.set_xlabel('Time Step (hour)', fontsize=10)
ax3.set_ylabel('Soil Saturation', fontsize=10)
ax3.set_ylim(0, 1)
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'components_analysis.png', dpi=300, bbox_inches='tight')
print(f"  ✓ 保存: components_analysis.png")
plt.close()

# ============================================================================
# 5. 保存观测数据和配置
# ============================================================================
print("\n步骤 5: 保存观测数据")
print("-" * 80)

# 保存径流数据
times = pd.date_range('2024-01-01', periods=len(best_runoff), freq='H')
obs_df = pd.DataFrame({
    'datetime': times,
    'discharge_m3s': best_runoff,
    'fast_component_m3s': fast,
    'inter_component_m3s': inter,
    'base_component_m3s': base,
})

obs_file = output_dir / 'zone_1_enhanced_runoff.csv'
obs_df.to_csv(obs_file, index=False)
print(f"✓ 保存径流数据: {obs_file}")

# 保存配置参数
import yaml
config_output = {
    'generator_type': 'EnhancedRunoffGenerator',
    'zone_1': best_config,
    'statistics': {
        'total_precip_mm': float(best_stats['total_precip_mm']),
        'total_runoff_mm': float(best_stats['total_runoff_mm']),
        'runoff_coefficient': float(best_stats['runoff_coefficient']),
        'mean_runoff_m3s': float(best_stats['mean_runoff_m3s']),
        'peak_runoff_m3s': float(best_stats['peak_runoff_m3s']),
    }
}

config_file = output_dir / 'enhanced_generator_config.yaml'
with open(config_file, 'w') as f:
    yaml.dump(config_output, f, default_flow_style=False, sort_keys=False)
print(f"✓ 保存配置文件: {config_file}")

# ============================================================================
# 6. 与简单方法对比
# ============================================================================
print("\n步骤 6: 与简单线性法对比")
print("-" * 80)

# 简单线性法: Q = Rc × P
simple_rc = 0.41
simple_runoff_mm_h = precipitation * simple_rc
simple_runoff_m3s = simple_runoff_mm_h * zone1_area_km2 / 3.6

# 对比统计
print(f"\n简单线性法 (Rc={simple_rc}):")
print(f"  平均流量: {simple_runoff_m3s.mean():.2f} m³/s")
print(f"  峰值流量: {simple_runoff_m3s.max():.2f} m³/s")
print(f"  最小流量: {simple_runoff_m3s.min():.2f} m³/s")

print(f"\n增强型生成器:")
print(f"  平均流量: {best_runoff.mean():.2f} m³/s")
print(f"  峰值流量: {best_runoff.max():.2f} m³/s")
print(f"  最小流量: {best_runoff.min():.2f} m³/s")

# 计算相关性
from scipy import stats
correlation, p_value = stats.pearsonr(simple_runoff_m3s, best_runoff)
print(f"\n两者相关系数: {correlation:.4f} (p={p_value:.4e})")

# 峰值延迟分析
simple_peak_idx = np.argmax(simple_runoff_m3s)
enhanced_peak_idx = np.argmax(best_runoff)
peak_lag = enhanced_peak_idx - simple_peak_idx
print(f"峰值延迟: {peak_lag} 小时")

# 对比图
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax1 = axes[0]
ax1.bar(range(len(precipitation)), precipitation, color='steelblue', alpha=0.5)
ax1.set_ylabel('Precipitation (mm/h)', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_title('Simple Linear vs Enhanced Generator', fontsize=12, weight='bold')

ax2 = axes[1]
ax2.plot(simple_runoff_m3s, label='Simple Linear (Rc=0.41)', color='gray',
         linewidth=2, linestyle='--', alpha=0.7)
ax2.plot(best_runoff, label=f'Enhanced (Rc={best_stats["runoff_coefficient"]:.3f})',
         color='blue', linewidth=2)
ax2.set_xlabel('Time Step (hour)', fontsize=10)
ax2.set_ylabel('Runoff (m³/s)', fontsize=10)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'simple_vs_enhanced.png', dpi=300, bbox_inches='tight')
print(f"\n✓ 保存对比图: simple_vs_enhanced.png")
plt.close()

# ============================================================================
# 7. 总结
# ============================================================================
print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)

print(f"\n增强型生成器的改进:")
print(f"  1. 土壤水分核算: 引入状态依赖的产流过程")
print(f"  2. 径流分量分离: 快速径流、中速径流、基流")
print(f"  3. 线性水库汇流: 更真实的水文过程模拟")
print(f"  4. 峰值延迟: {peak_lag} 小时（简单法无延迟）")
print(f"  5. 基流维持: 最小流量 {best_runoff.min():.2f} m³/s （简单法可能为0）")

print(f"\n输出文件:")
print(f"  - 观测数据: {obs_file}")
print(f"  - 配置参数: {config_file}")
print(f"  - 对比图表: {output_dir}/*.png")

print(f"\n下一步:")
print(f"  使用以下命令进行HBV参数率定:")
print(f"  python calibrate_zone1_hbv_enhanced.py")

print("\n" + "=" * 80)
