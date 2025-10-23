#!/usr/bin/env python3
"""生成各参数分区的估计径流序列

基于面雨量时间序列，使用径流系数法生成估计的径流过程。
用于参数敏感性分析和自动率定的基准数据。
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hydrosis.estimated_runoff import EstimatedRunoffGenerator

print("=" * 80)
print("生成参数分区估计径流序列")
print("=" * 80)

# 配置参数
results_dir = Path("results/upper_truckee_complete_11steps")
precip_file = results_dir / "step_08_areal_rainfall" / "8.2_subbasin_areal_precipitation.csv"
parameters_dir = results_dir / "parameters"
output_dir = results_dir / "estimated_observations"
output_dir.mkdir(parents=True, exist_ok=True)

# 分区特定参数（根据地形和下垫面特征）
ZONE_CONFIGS = {
    1: {
        'runoff_coefficient': 0.35,
        'lag_hours': 1.5,
        'attenuation_factor': 0.85,
        'baseflow_ratio': 0.08,
        'description': "上游源头区 - 陡坡，快速汇流"
    },
    2: {
        'runoff_coefficient': 0.34,
        'lag_hours': 3.0,
        'attenuation_factor': 0.75,
        'baseflow_ratio': 0.12,
        'description': "中游过渡区 - 中等坡度"
    },
    3: {
        'runoff_coefficient': 0.33,
        'lag_hours': 5.0,
        'attenuation_factor': 0.65,
        'baseflow_ratio': 0.15,
        'description': "下游主河道区 - 缓坡，显著削峰"
    },
    4: {
        'runoff_coefficient': 0.345,
        'lag_hours': 3.5,
        'attenuation_factor': 0.70,
        'baseflow_ratio': 0.10,
        'description': "Zone 4 - 默认参数"
    },
    5: {
        'runoff_coefficient': 0.345,
        'lag_hours': 3.5,
        'attenuation_factor': 0.70,
        'baseflow_ratio': 0.10,
        'description': "Zone 5 - 默认参数"
    },
    6: {
        'runoff_coefficient': 0.345,
        'lag_hours': 3.5,
        'attenuation_factor': 0.70,
        'baseflow_ratio': 0.10,
        'description': "Zone 6 - 默认参数"
    },
}


def aggregate_precipitation_for_zone(precip_df, zone_id, subbasin_df):
    """计算指定分区的面积加权平均降雨

    Args:
        precip_df: 子流域面雨量DataFrame
        zone_id: 分区ID
        subbasin_df: 子流域属性DataFrame

    Returns:
        Series: 分区面雨量时间序列 (mm/hr)
    """
    # 获取该分区的所有子流域
    zone_subbasins = subbasin_df[subbasin_df['zone_id'] == zone_id]
    subbasin_ids = zone_subbasins['subzone_id'].astype(str).tolist()

    # 提取这些子流域的降雨数据
    available_ids = [sid for sid in subbasin_ids if sid in precip_df.columns]

    if not available_ids:
        raise ValueError(f"Zone {zone_id} 没有找到对应的降雨数据")

    # 面积加权平均
    total_area = 0
    weighted_precip = pd.Series(0.0, index=precip_df.index)

    for sid in available_ids:
        area = zone_subbasins[zone_subbasins['subzone_id'] == int(sid)]['area_km2'].values[0]
        weighted_precip += precip_df[sid] * area
        total_area += area

    if total_area > 0:
        weighted_precip /= total_area

    return weighted_precip, total_area


print("\n1. 加载数据...")
# 加载降雨数据
if not precip_file.exists():
    print(f"  ❌ 降雨数据文件不存在: {precip_file}")
    sys.exit(1)

precip_df = pd.read_csv(precip_file, index_col=0, parse_dates=True)
print(f"  ✓ 加载降雨数据: {len(precip_df)} 时间步, {len(precip_df.columns)} 子流域")

# 加载子流域属性
subbasin_file = parameters_dir / "parameter_subbasins.csv"
subbasin_df = pd.read_csv(subbasin_file)
print(f"  ✓ 加载子流域属性: {len(subbasin_df)} 个子流域")

# 加载分区信息
zones_file = parameters_dir / "parameter_zones.csv"
zones_df = pd.read_csv(zones_file)
print(f"  ✓ 加载参数分区: {len(zones_df)} 个分区")

print("\n2. 为各参数分区生成估计径流序列...")

all_stats = []
all_runoff = {}

for zone_id in sorted(zones_df['zone_id'].unique()):
    print(f"\n  {'='*70}")
    print(f"  处理分区 {zone_id}")
    print(f"  {'='*70}")

    # 获取分区配置
    if zone_id in ZONE_CONFIGS:
        config = ZONE_CONFIGS[zone_id]
    else:
        # 使用默认配置
        config = ZONE_CONFIGS[1].copy()
        config['description'] = f"Zone {zone_id} - 使用默认参数"

    print(f"  配置: {config['description']}")
    print(f"    径流系数: {config['runoff_coefficient']:.3f}")
    print(f"    时滞: {config['lag_hours']:.1f} 小时")
    print(f"    削峰系数: {config['attenuation_factor']:.2f}")
    print(f"    基流比例: {config['baseflow_ratio']:.2f}")

    # 计算分区面雨量
    try:
        zone_precip, zone_area = aggregate_precipitation_for_zone(precip_df, zone_id, subbasin_df)
        print(f"    流域面积: {zone_area:.2f} km²")
        print(f"    平均降雨: {zone_precip.mean():.2f} mm/hr")
        print(f"    最大降雨: {zone_precip.max():.2f} mm/hr")
    except ValueError as e:
        print(f"  ⚠ 跳过分区 {zone_id}: {e}")
        continue

    # 创建径流生成器
    generator = EstimatedRunoffGenerator(
        runoff_coefficient=config['runoff_coefficient'],
        lag_hours=config['lag_hours'],
        attenuation_factor=config['attenuation_factor'],
        baseflow_ratio=config['baseflow_ratio'],
        time_step_hours=1.0
    )

    # 验证参数
    warnings = generator.validate_parameters()
    if warnings:
        for warning in warnings:
            print(f"  ⚠ {warning}")

    # 生成径流序列
    runoff, stats = generator.generate(
        precipitation_series=zone_precip.values,
        area_km2=zone_area
    )

    # 打印统计信息
    print(f"\n  生成统计:")
    print(f"    总降雨深度: {stats['total_precip_mm']:.2f} mm")
    print(f"    总径流深度: {stats['total_runoff_mm']:.2f} mm")
    print(f"    目标径流系数: {stats['target_rc']:.3f}")
    print(f"    实际径流系数: {stats['actual_rc']:.3f}")
    print(f"    峰值流量: {stats['peak_runoff_final_m3s']:.2f} m³/s")
    print(f"    单位面积峰值: {stats['specific_peak_m3s_per_km2']:.3f} m³/s/km²")
    print(f"    削峰效果: {stats['peak_reduction_pct']:.1f}%")
    print(f"    目标时滞: {stats['target_lag_hr']:.1f} hr")
    print(f"    实际时滞: {stats['actual_lag_hr']:.1f} hr")
    print(f"    平均流量: {stats['mean_runoff_m3s']:.2f} m³/s")
    print(f"    最小流量: {stats['min_runoff_m3s']:.2f} m³/s")

    # 验证合理性
    checks_passed = True
    if stats['actual_rc'] > 1.0:
        print(f"  ❌ 警告: 径流系数 > 1.0，违反水量平衡！")
        checks_passed = False
    elif stats['actual_rc'] < 0.1:
        print(f"  ⚠  警告: 径流系数 < 0.1，偏低")

    if stats['specific_peak_m3s_per_km2'] < 0.05 or stats['specific_peak_m3s_per_km2'] > 3.0:
        print(f"  ⚠  警告: 单位面积峰值不在常见范围 (0.05-3.0 m³/s/km²)")

    if checks_passed:
        print(f"  ✓ 合理性检查通过")

    # 保存径流序列
    runoff_df = pd.DataFrame({
        'datetime': precip_df.index,
        'discharge_m3s': runoff
    })
    runoff_file = output_dir / f"zone_{zone_id}_estimated_runoff.csv"
    runoff_df.to_csv(runoff_file, index=False)
    print(f"  ✓ 保存: {runoff_file.name}")

    # 存储用于汇总
    stats['zone_id'] = zone_id
    stats['description'] = config['description']
    all_stats.append(stats)
    all_runoff[zone_id] = runoff

print("\n3. 保存汇总信息...")

# 保存统计表
stats_df = pd.DataFrame(all_stats)
stats_file = output_dir / "generation_statistics.csv"
stats_df.to_csv(stats_file, index=False)
print(f"  ✓ 保存统计表: {stats_file.name}")

# 保存生成配置
config_dict = {
    'version': '1.0',
    'description': '估计径流序列生成配置',
    'zone_configs': ZONE_CONFIGS
}
config_file = output_dir / "generation_config.yaml"
with open(config_file, 'w') as f:
    yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
print(f"  ✓ 保存配置: {config_file.name}")

# 生成文本摘要
summary_file = output_dir / "generation_summary.txt"
with open(summary_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("估计径流序列生成摘要\n")
    f.write("=" * 80 + "\n\n")

    for _, row in stats_df.iterrows():
        f.write(f"Zone {int(row['zone_id'])}:\n")
        f.write(f"  面积: {row['area_km2']:.2f} km²\n")
        f.write(f"  总降雨: {row['total_precip_mm']:.2f} mm\n")
        f.write(f"  总径流: {row['total_runoff_mm']:.2f} mm\n")
        f.write(f"  径流系数: {row['actual_rc']:.3f}\n")
        f.write(f"  峰值流量: {row['peak_runoff_final_m3s']:.2f} m³/s\n")
        f.write(f"  时滞: {row['actual_lag_hr']:.1f} hr\n")
        f.write("\n")

    f.write("\n汇总统计:\n")
    f.write(f"  平均径流系数: {stats_df['actual_rc'].mean():.3f}\n")
    f.write(f"  径流系数范围: {stats_df['actual_rc'].min():.3f} - {stats_df['actual_rc'].max():.3f}\n")
    f.write(f"  平均时滞: {stats_df['actual_lag_hr'].mean():.1f} 小时\n")

print(f"  ✓ 保存摘要: {summary_file.name}")

print("\n4. 生成对比图...")

# 创建对比图：降雨vs径流
fig, axes = plt.subplots(len(all_runoff), 1, figsize=(12, 3*len(all_runoff)), sharex=True)
if len(all_runoff) == 1:
    axes = [axes]

for idx, (zone_id, runoff) in enumerate(sorted(all_runoff.items())):
    ax = axes[idx]

    # 获取该分区的降雨
    zone_precip, zone_area = aggregate_precipitation_for_zone(precip_df, zone_id, subbasin_df)

    # 双Y轴
    ax2 = ax.twinx()

    # 降雨（反向柱状图）
    ax.bar(range(len(zone_precip)), zone_precip, color='blue', alpha=0.3, label='降雨')
    ax.set_ylabel('降雨 (mm/hr)', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.invert_yaxis()  # 反向Y轴

    # 径流（线图）
    ax2.plot(runoff, color='red', linewidth=2, label='估计径流')
    ax2.set_ylabel('径流 (m³/s)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.grid(True, alpha=0.3)

    # 标题
    rc = stats_df[stats_df['zone_id'] == zone_id]['actual_rc'].values[0]
    ax.set_title(f"Zone {zone_id} - Rc={rc:.3f}")

    if idx == len(all_runoff) - 1:
        ax.set_xlabel('时间步 (小时)')

plt.tight_layout()
plot_file = output_dir / "estimated_runoff_comparison.png"
plt.savefig(plot_file, dpi=200, bbox_inches='tight')
print(f"  ✓ 保存对比图: {plot_file.name}")
plt.close()

print("\n" + "=" * 80)
print("✓ 估计径流序列生成完成！")
print("=" * 80)
print(f"\n生成了 {len(all_runoff)} 个分区的径流序列")
print(f"输出目录: {output_dir}")
print("\n下一步:")
print("  1. 查看对比图: estimated_runoff_comparison.png")
print("  2. 查看统计摘要: generation_summary.txt")
print("  3. 在模拟中使用: python rerun_step09_10.py --use-estimated-obs")
