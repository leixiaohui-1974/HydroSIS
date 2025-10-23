#!/usr/bin/env python3
"""
重新运行Step09-10水文水动力模拟

使用已有的Step1-8结果，只重新运行产流和汇流模拟。
用于验证HBV模型参数修复后的效果。
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hydrosis.config import (
    DelineationConfig,
    EvaluationConfig,
    IOConfig,
    ModelConfig,
    ParameterZoneConfig,
    RoutingModelConfig,
    RunoffModelConfig,
)
from hydrosis.workflow import run_workflow

def load_parameter_zones_from_csv(parameter_dir: Path):
    """从CSV文件加载参数分区配置"""
    zones_file = parameter_dir / "parameter_zones.csv"
    subbasins_file = parameter_dir / "parameter_subbasins.csv"

    zones_df = pd.read_csv(zones_file)
    # 关键：读取时保持ID为整数，转换为字符串
    subbasins_df = pd.read_csv(subbasins_file, dtype={'subzone_id': int, 'zone_id': int})

    # 构建zone到subbasins的映射
    zone_to_subbasins = {}
    for _, row in subbasins_df.iterrows():
        zone_id = str(int(row['zone_id']))
        subbasin_id = str(int(row['subzone_id']))
        if zone_id not in zone_to_subbasins:
            zone_to_subbasins[zone_id] = []
        zone_to_subbasins[zone_id].append(subbasin_id)

    # 创建ParameterZoneConfig对象
    parameter_zones = []
    for _, row in zones_df.iterrows():
        zone_id = str(row['zone_id'])
        subbasin_ids = zone_to_subbasins.get(zone_id, [])

        # 找到outlet subbasin (downstream_id为空的)
        outlet_id = None
        for subbasin_id in subbasin_ids:
            sub_row = subbasins_df[subbasins_df['subzone_id'] == subbasin_id]
            if len(sub_row) > 0:
                downstream = sub_row.iloc[0]['downstream_subzone_id']
                if pd.isna(downstream) or downstream == '' or downstream not in subbasin_ids:
                    outlet_id = subbasin_id
                    break

        if outlet_id is None and subbasin_ids:
            outlet_id = subbasin_ids[0]

        zone_config = ParameterZoneConfig(
            id=zone_id,
            description=f"Zone {zone_id}",
            control_points=[outlet_id] if outlet_id else [],
            parameters={
                'runoff_model': 'hbv',
                'routing_model': 'muskingum',
            },
            explicit_subbasins=subbasin_ids,
        )
        parameter_zones.append(zone_config)

    return parameter_zones, subbasins_df

def main():
    print("=" * 80)
    print("重新运行Step09-10：水文水动力模拟")
    print("=" * 80)

    # 设置路径
    results_dir = Path("results/upper_truckee_complete_11steps")
    parameter_dir = results_dir / "parameters"
    step9_dir = results_dir / "step_09_runoff"
    step10_dir = results_dir / "step_10_routing"
    workflow_dir = results_dir / "workflow_results"

    # 清空旧的结果
    import shutil
    for d in [step9_dir, step10_dir, workflow_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    print("\n1. 加载已有数据...")

    # 加载参数分区配置
    parameter_zones, subbasins_df = load_parameter_zones_from_csv(parameter_dir)
    print(f"  ✓ 加载{len(parameter_zones)}个参数分区")

    # 加载子流域面雨量数据
    precip_file = results_dir / "intermediate" / "subbasin_areal_precipitation.csv"
    precip_df = pd.read_csv(precip_file, index_col=0)
    print(f"  ✓ 加载降雨数据：{len(precip_df)}个时间步，{len(precip_df.columns)}个子流域")

    print("\n2. 配置HBV产流模型...")

    # 配置产流模型 - 使用大写参数名（现在HBV模型已支持）
    runoff_models = [
        RunoffModelConfig(
            id="hbv",
            model_type="hbv",
            parameters={
                "TT": 0.0,          # 雪阈值温度
                "CFMAX": 3.5,       # 度日因子
                "CFR": 0.05,        # 再冻结系数
                "CWH": 0.1,         # 持水能力
                "FC": 250.0,        # 最大土壤含水量 ← 关键参数
                "LP": 0.7,          # 蒸散限制
                "BETA": 2.0,        # 形状系数 ← 关键参数
                "K0": 0.05,         # 快速响应系数 ← 关键参数
                "K1": 0.01,         # 慢速响应系数 ← 关键参数
                "K2": 0.001,        # 基流系数 ← 关键参数
                "PERC": 1.5,        # 渗透率 ← 关键参数
                "UZL": 5.0,         # 上层阈值
                "MAXBAS": 3.0,      # 基流最大值
            }
        ),
    ]

    print("  ✓ HBV参数配置：")
    print(f"     - FC (field_capacity): 250.0 mm")
    print(f"     - BETA: 2.0")
    print(f"     - K0/K1/K2: 0.05/0.01/0.001")
    print(f"     - PERC (percolation): 1.5 mm/day")

    print("\n3. 配置Muskingum汇流模型...")

    routing_models = [
        RoutingModelConfig(
            id="muskingum",
            model_type="muskingum",
            parameters={
                "K": 10.0,          # 蓄量常数
                "x": 0.2,           # 权重系数
                "time_step": 1.0,   # 时间步长
            }
        ),
    ]

    print("  ✓ Muskingum参数：K=10.0, x=0.2")

    print("\n4. 构建子流域列表...")

    # 创建子流域列表
    subbasin_list = []
    for _, row in subbasins_df.iterrows():
        # 确保ID为字符串格式
        downstream_id = None
        if pd.notna(row['downstream_subzone_id']):
            try:
                downstream_id = str(int(row['downstream_subzone_id']))
            except (ValueError, TypeError):
                downstream_id = None

        subbasin_obj = {
            'id': str(int(row['subzone_id'])),
            'area_km2': float(row['area_km2']),
            'downstream': downstream_id,
            'parameters': {},
        }
        subbasin_list.append(subbasin_obj)

    print(f"  ✓ {len(subbasin_list)}个子流域")

    print("\n5. 构建模型配置...")

    # 创建delineation配置
    delineation_cfg = DelineationConfig(
        dem_path=Path("data/Upper_Truckee_River/terrain/UpTruckeeRv_S10_NED_30m/00/elevation.tif"),
        pour_points_path=None,
        flow_direction_path=Path("data/Upper_Truckee_River/terrain/UpTruckeeRv_S10_NED_30m/00/flowdir.tif"),
        flow_accumulation_path=Path("data/Upper_Truckee_River/terrain/UpTruckeeRv_S10_NED_30m/00/flowaccum.tif"),
        accumulation_threshold=15000.0,
        intermediate_directory=results_dir / "intermediate",
        parameter_directory=parameter_dir,
        precomputed_subbasins=subbasin_list,
    )

    # IO配置
    io_config = IOConfig(
        results_directory=workflow_dir,
        precipitation=precip_file,
    )

    # 评估配置
    evaluation_config = EvaluationConfig(
        metrics=["rmse", "mae", "nse", "pbias"],
    )

    # 完整模型配置
    model_config = ModelConfig(
        delineation=delineation_cfg,
        runoff_models=runoff_models,
        routing_models=routing_models,
        parameter_zones=parameter_zones,
        io=io_config,
        evaluation=evaluation_config,
    )

    print("  ✓ 模型配置完成")

    print("\n6. 准备forcing数据...")

    # Forcing数据
    forcing = {col: precip_df[col].tolist() for col in precip_df.columns}
    print(f"  ✓ {len(forcing)}个子流域的降雨序列")

    # 生成合成观测数据
    synthetic_obs = np.concatenate([
        np.zeros(48),
        np.linspace(0, 15, 24),
        15 * np.exp(-np.linspace(0, 3, 48)),
    ])

    # 找到outlet
    outlet_subzone = None
    for sb in subbasin_list:
        if sb['downstream'] is None or sb['downstream'] == '':
            outlet_subzone = sb['id']
            break
    if outlet_subzone is None:
        outlet_subzone = subbasin_list[0]['id']

    observations = {outlet_subzone: list(synthetic_obs)}
    print(f"  ✓ 出口子流域: {outlet_subzone}")

    print("\n7. 运行水文水动力模拟...")
    print("   （这将需要几分钟时间...）")

    # 运行模拟
    workflow_result = run_workflow(
        model_config,
        forcing,
        observations=observations,
        persist_outputs=True,
    )

    print("\n8. 保存结果...")

    # 获取结果
    baseline = workflow_result.baseline
    aggregated = baseline.aggregated

    # 保存流量时间序列
    result_df = pd.DataFrame(aggregated)
    result_csv = step10_dir / "10.1_discharge_timeseries.csv"
    result_df.to_csv(result_csv)
    print(f"  ✓ 保存流量时间序列: {result_csv}")

    # 计算统计
    stats = []
    for sub_id, discharge in aggregated.items():
        stats.append({
            'Subbasin_ID': sub_id,
            'Peak_Discharge_m3s': float(np.max(discharge)),
            'Total_Volume_m3': float(np.sum(discharge) * 3600.0),
            'Mean_Discharge_m3s': float(np.mean(discharge)),
        })

    stats_df = pd.DataFrame(stats)
    stats_csv = step10_dir / "10.2_discharge_statistics.csv"
    stats_df.to_csv(stats_csv, index=False)
    print(f"  ✓ 保存统计表: {stats_csv}")

    # 绘制流量过程线
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    timesteps = np.arange(len(next(iter(aggregated.values()))))
    for sub_id, discharge in list(aggregated.items())[:4]:
        ax1.plot(timesteps, discharge, label=f'{sub_id}', linewidth=1.5)
    ax1.set_ylabel('Discharge (m³/s)')
    ax1.set_title('Subbasin Discharge Hydrographs')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right')

    if outlet_subzone in aggregated:
        outlet_discharge = aggregated[outlet_subzone]
        ax2.fill_between(timesteps, 0, outlet_discharge, alpha=0.3)
        ax2.plot(timesteps, outlet_discharge, linewidth=2, color='blue', label='Simulated')
        if outlet_subzone in observations:
            ax2.plot(timesteps, observations[outlet_subzone], 'r--',
                    linewidth=2, label='Observed (Synthetic)')

    ax2.set_xlabel('Time Step (hours)')
    ax2.set_ylabel('Discharge (m³/s)')
    ax2.set_title(f'Outlet Discharge ({outlet_subzone})')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right')

    fig.tight_layout()
    fig_file = step10_dir / "10.3_discharge_hydrographs.png"
    plt.savefig(fig_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 生成流量过程图: {fig_file}")

    print("\n" + "=" * 80)
    print("Step09-10重新运行完成！")
    print("=" * 80)

    # 计算径流系数验证
    print("\n9. 计算径流系数验证...")

    # 读取Zone1的流量数据
    zone1_file = workflow_dir / "baseline" / "1.csv"
    if zone1_file.exists():
        zone1_discharge = pd.read_csv(zone1_file, header=None)[1].values
        zone1_area = 139.995  # km²

        # 总径流深度
        dt_hours = 1.0
        total_volume_m3 = zone1_discharge.sum() * dt_hours * 3600
        runoff_depth_mm = (total_volume_m3 / (zone1_area * 1e6)) * 1000

        # Zone1平均降雨
        zone1_cols = [col for col in precip_df.columns if col.startswith('1')]
        avg_precip_mm = precip_df[zone1_cols].mean(axis=1).sum() * dt_hours

        # 径流系数
        runoff_coeff = runoff_depth_mm / avg_precip_mm

        print(f"\n  Zone 1 水量平衡分析：")
        print(f"    流域面积: {zone1_area:.2f} km²")
        print(f"    总降雨深度: {avg_precip_mm:.2f} mm")
        print(f"    总径流深度: {runoff_depth_mm:.2f} mm")
        print(f"    径流系数: {runoff_coeff:.3f}")
        print(f"    峰值流量: {zone1_discharge.max():.2f} m³/s")

        if 0.1 <= runoff_coeff <= 0.8:
            print(f"\n  ✅ 径流系数在合理范围内 (0.1-0.8)")
        else:
            print(f"\n  ⚠️  径流系数超出合理范围 (0.1-0.8)")
            if runoff_coeff > 1.0:
                print(f"     问题：径流深度 > 降雨深度（违反水量平衡！）")

    print("\n下一步：运行 create_zone_rainfall_runoff_plots.py 重新生成降雨径流过程图")

if __name__ == "__main__":
    main()
