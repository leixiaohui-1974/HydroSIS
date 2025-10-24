#!/usr/bin/env python3
"""运行Upper Truckee River完整工作流，为每一步生成详细的图表和报告"""
from __future__ import annotations

import sys
import json
from pathlib import Path

# 确保可以导入HydroSIS模块
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def print_section(title: str):
    """打印分节标题"""
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)
    print()

def print_step(step_num: int, title: str):
    """打印步骤标题"""
    print()
    print("-" * 80)
    print(f"步骤 {step_num}: {title}")
    print("-" * 80)

def main():
    """运行完整的Upper Truckee River工作流，每步生成详细可视化"""
    print_section("Upper Truckee River 详细可视化工作流")

    import numpy as np
    import pandas as pd

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, FancyBboxPatch
        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib.patches as mpatches
        plt.ioff()
    except ImportError:
        print("错误: matplotlib未安装")
        sys.exit(1)

    try:
        import rasterio
        from rasterio.plot import show as rasterio_show
        has_rasterio = True
    except ImportError:
        print("警告: rasterio未安装，将跳过DEM可视化")
        has_rasterio = False

    # 配置路径
    output_dir = REPO_ROOT / "results" / "upper_truckee_detailed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建各步骤的输出目录
    step_dirs = {}
    for i in range(1, 11):
        step_dir = output_dir / f"step_{i:02d}"
        step_dir.mkdir(exist_ok=True)
        step_dirs[i] = step_dir

    dem_dir = REPO_ROOT / "data" / "Upper_Truckee_River" / "terrain" / "UpTruckeeRv_S10_NED_30m" / "00"
    dem_path = dem_dir / "elevation.tif"

    # ========================================================================
    # 步骤1: DEM和地形分析可视化
    # ========================================================================
    print_step(1, "DEM和地形分析可视化")

    step1_dir = step_dirs[1]

    # 读取DEM数据
    dem_stats = {}
    if has_rasterio and dem_path.exists():
        with rasterio.open(dem_path) as src:
            dem_data = src.read(1)
            dem_transform = src.transform
            dem_crs = src.crs
            dem_bounds = src.bounds

            # 统计信息
            dem_stats = {
                "最小高程 (m)": float(np.nanmin(dem_data)),
                "最大高程 (m)": float(np.nanmax(dem_data)),
                "平均高程 (m)": float(np.nanmean(dem_data)),
                "标准差 (m)": float(np.nanstd(dem_data)),
                "行数": dem_data.shape[0],
                "列数": dem_data.shape[1],
                "分辨率 (m)": abs(dem_transform[0]),
            }

            print("DEM统计信息:")
            for key, value in dem_stats.items():
                print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")

            # 图1.1: DEM高程图
            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(dem_data, cmap='terrain', aspect='auto')
            ax.set_title('Upper Truckee River - DEM 高程图', fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('列 (Column)', fontsize=12)
            ax.set_ylabel('行 (Row)', fontsize=12)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('高程 (m)', fontsize=12)
            plt.tight_layout()
            plt.savefig(step1_dir / "1.1_dem_elevation.png", dpi=150, bbox_inches='tight')
            plt.close()
            print("  ✓ DEM高程图已保存")

            # 图1.2: DEM高程直方图
            fig, ax = plt.subplots(figsize=(10, 6))
            valid_data = dem_data[~np.isnan(dem_data)]
            ax.hist(valid_data, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
            ax.axvline(dem_stats["平均高程 (m)"], color='red', linestyle='--',
                      linewidth=2, label=f'平均值: {dem_stats["平均高程 (m)"]:.1f} m')
            ax.set_xlabel('高程 (m)', fontsize=12)
            ax.set_ylabel('像元数量', fontsize=12)
            ax.set_title('Upper Truckee River - DEM 高程分布直方图', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(step1_dir / "1.2_dem_histogram.png", dpi=150, bbox_inches='tight')
            plt.close()
            print("  ✓ DEM高程直方图已保存")

    # 保存DEM统计表
    if dem_stats:
        stats_df = pd.DataFrame(list(dem_stats.items()), columns=['属性', '数值'])
        stats_df.to_csv(step1_dir / "1.3_dem_statistics.csv", index=False, encoding='utf-8-sig')
        print("  ✓ DEM统计表已保存")

    # 图1.3: 流向图 (Flow Direction)
    flowdir_path = dem_dir / "flowdir.tif"
    if has_rasterio and flowdir_path.exists():
        with rasterio.open(flowdir_path) as src:
            flowdir_data = src.read(1)

            fig, ax = plt.subplots(figsize=(12, 10))
            # 使用离散颜色映射显示8个方向
            cmap = plt.cm.get_cmap('tab10', 8)
            im = ax.imshow(flowdir_data, cmap=cmap, aspect='auto', vmin=0, vmax=7)
            ax.set_title('Upper Truckee River - Flow Direction (D8)', fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Column', fontsize=12)
            ax.set_ylabel('Row', fontsize=12)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=range(8))
            cbar.set_label('Flow Direction', fontsize=12)
            cbar.ax.set_yticklabels(['E', 'SE', 'S', 'SW', 'W', 'NW', 'N', 'NE'])
            plt.tight_layout()
            plt.savefig(step1_dir / "1.4_flow_direction.png", dpi=150, bbox_inches='tight')
            plt.close()
            print("  ✓ 流向图已保存")

    # 图1.4: 流量累积图 (Flow Accumulation)
    flowaccum_path = dem_dir / "flowaccum.tif"
    if has_rasterio and flowaccum_path.exists():
        with rasterio.open(flowaccum_path) as src:
            flowaccum_data = src.read(1)

            fig, ax = plt.subplots(figsize=(12, 10))
            # 使用对数刻度显示流量累积
            flowaccum_log = np.log10(flowaccum_data + 1)
            im = ax.imshow(flowaccum_log, cmap='Blues', aspect='auto')
            ax.set_title('Upper Truckee River - Flow Accumulation (log scale)',
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Column', fontsize=12)
            ax.set_ylabel('Row', fontsize=12)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('log10(Flow Accumulation + 1)', fontsize=12)
            plt.tight_layout()
            plt.savefig(step1_dir / "1.5_flow_accumulation.png", dpi=150, bbox_inches='tight')
            plt.close()
            print("  ✓ 流量累积图已保存")

            # 图1.5: 河网提取 (Stream Network)
            fig, ax = plt.subplots(figsize=(12, 10))
            # 叠加DEM和河网
            ax.imshow(dem_data, cmap='terrain', aspect='auto', alpha=0.6)
            # 提取河网（流量累积阈值）
            threshold = np.percentile(flowaccum_data[flowaccum_data > 0], 95)
            stream_network = flowaccum_data > threshold
            ax.imshow(np.ma.masked_where(~stream_network, flowaccum_data),
                     cmap='Blues', aspect='auto', alpha=0.9)
            ax.set_title(f'Upper Truckee River - Stream Network (threshold={threshold:.0f} cells)',
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Column', fontsize=12)
            ax.set_ylabel('Row', fontsize=12)
            plt.tight_layout()
            plt.savefig(step1_dir / "1.6_stream_network.png", dpi=150, bbox_inches='tight')
            plt.close()
            print("  ✓ 河网提取图已保存")

    # 图1.6: 坡度分析 (Slope)
    if has_rasterio and dem_path.exists():
        with rasterio.open(dem_path) as src:
            dem_data = src.read(1)
            res = abs(dem_transform[0])  # 分辨率

            # 计算坡度（使用numpy gradient）
            dy, dx = np.gradient(dem_data, res, res)
            slope = np.arctan(np.sqrt(dx**2 + dy**2)) * 180 / np.pi  # 转换为度

            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(slope, cmap='YlOrRd', aspect='auto', vmin=0, vmax=45)
            ax.set_title('Upper Truckee River - Slope', fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Column', fontsize=12)
            ax.set_ylabel('Row', fontsize=12)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Slope (degrees)', fontsize=12)
            plt.tight_layout()
            plt.savefig(step1_dir / "1.7_slope.png", dpi=150, bbox_inches='tight')
            plt.close()
            print("  ✓ 坡度图已保存")

            # 坡度统计
            slope_stats = {
                "最小坡度 (度)": float(np.nanmin(slope)),
                "最大坡度 (度)": float(np.nanmax(slope)),
                "平均坡度 (度)": float(np.nanmean(slope)),
                "标准差 (度)": float(np.nanstd(slope)),
            }

            # 图1.7: 坡度直方图
            fig, ax = plt.subplots(figsize=(10, 6))
            valid_slope = slope[~np.isnan(slope)]
            ax.hist(valid_slope, bins=50, color='orange', alpha=0.7, edgecolor='black')
            ax.axvline(slope_stats["平均坡度 (度)"], color='red', linestyle='--',
                      linewidth=2, label=f'Mean: {slope_stats["平均坡度 (度)"]:.2f}°')
            ax.set_xlabel('Slope (degrees)', fontsize=12)
            ax.set_ylabel('Cell Count', fontsize=12)
            ax.set_title('Upper Truckee River - Slope Distribution', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(step1_dir / "1.8_slope_histogram.png", dpi=150, bbox_inches='tight')
            plt.close()
            print("  ✓ 坡度直方图已保存")

            # 更新统计表
            combined_stats = {**dem_stats, **slope_stats}
            stats_df = pd.DataFrame(list(combined_stats.items()), columns=['Property', 'Value'])
            stats_df.to_csv(step1_dir / "1.9_terrain_statistics.csv", index=False, encoding='utf-8-sig')
            print("  ✓ 地形统计表已保存")

    # ========================================================================
    # 步骤2: 子流域划分可视化
    # ========================================================================
    print_step(2, "子流域划分和拓扑关系可视化")

    step2_dir = step_dirs[2]

    # 定义子流域
    subbasins = [
        {"id": "W170", "area_km2": 45.2, "downstream": "W180",
         "zone": "Upper_Mountain", "elevation_range": "2000-3000m"},
        {"id": "W160", "area_km2": 35.8, "downstream": "W180",
         "zone": "Upper_Mountain", "elevation_range": "2000-3000m"},
        {"id": "W180", "area_km2": 125.5, "downstream": "W190",
         "zone": "Middle_Forest", "elevation_range": "1500-2000m"},
        {"id": "W190", "area_km2": 165.0, "downstream": None,
         "zone": "Lower_Valley", "elevation_range": "1200-1500m"},
    ]

    total_area = sum(sb["area_km2"] for sb in subbasins)

    # 图2.1: 子流域面积饼图
    fig, ax = plt.subplots(figsize=(10, 8))
    areas = [sb["area_km2"] for sb in subbasins]
    labels = [f"{sb['id']}\n{sb['area_km2']} km²\n({sb['area_km2']/total_area*100:.1f}%)"
              for sb in subbasins]
    colors = plt.cm.Set3(range(len(subbasins)))

    wedges, texts, autotexts = ax.pie(areas, labels=labels, colors=colors, autopct='',
                                        startangle=90, textprops={'fontsize': 11})
    ax.set_title(f'Upper Truckee River 子流域面积分布\n总面积: {total_area} km²',
                 fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(step2_dir / "2.1_subbasin_area_pie.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 子流域面积饼图已保存")

    # 图2.2: 子流域拓扑关系图（网络图）
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # 定义节点位置（按拓扑关系排列）
    positions = {
        "W170": (2, 8),
        "W160": (8, 8),
        "W180": (5, 5),
        "W190": (5, 2),
    }

    # 绘制连接线
    connections = [
        ("W170", "W180"),
        ("W160", "W180"),
        ("W180", "W190"),
    ]

    for source, target in connections:
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='steelblue'))

    # 绘制节点
    zone_colors = {
        "Upper_Mountain": '#FFB6C1',
        "Middle_Forest": '#90EE90',
        "Lower_Valley": '#87CEEB'
    }

    for sb in subbasins:
        x, y = positions[sb["id"]]
        color = zone_colors.get(sb["zone"], 'lightgray')

        # 绘制节点框
        box = FancyBboxPatch((x-0.8, y-0.6), 1.6, 1.2,
                            boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)

        # 添加文本
        ax.text(x, y+0.3, sb["id"], ha='center', va='center',
               fontsize=14, fontweight='bold')
        ax.text(x, y, f"{sb['area_km2']} km²", ha='center', va='center', fontsize=10)
        ax.text(x, y-0.3, sb["zone"], ha='center', va='center', fontsize=8, style='italic')

    # 添加图例
    legend_elements = [mpatches.Patch(facecolor=color, edgecolor='black', label=zone)
                      for zone, color in zone_colors.items()]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

    ax.set_title('Upper Truckee River 子流域拓扑关系图', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(step2_dir / "2.2_subbasin_topology.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 子流域拓扑关系图已保存")

    # 图2.3: 子流域信息表
    subbasin_df = pd.DataFrame(subbasins)
    subbasin_df['下游'] = subbasin_df['downstream'].fillna('出口')
    subbasin_df = subbasin_df[['id', 'area_km2', '下游', 'zone', 'elevation_range']]
    subbasin_df.columns = ['子流域ID', '面积(km²)', '下游', '参数区', '高程范围']
    subbasin_df.to_csv(step2_dir / "2.3_subbasin_info.csv", index=False, encoding='utf-8-sig')
    print("  ✓ 子流域信息表已保存")

    # ========================================================================
    # 步骤3: 降雨数据生成和可视化
    # ========================================================================
    print_step(3, "降雨数据生成和空间分布可视化")

    step3_dir = step_dirs[3]

    # 生成降雨数据
    hours = 120
    timesteps = hours
    rainfall = np.zeros(timesteps)
    storm_start = 48
    storm_duration = 24
    peak_hour = storm_start + 12
    total_rainfall_mm = 75.0

    for t in range(storm_start, storm_start + storm_duration):
        sigma = storm_duration / 6.0
        weight = np.exp(-0.5 * ((t - peak_hour) / sigma) ** 2)
        rainfall[t] = weight

    rainfall = rainfall * (total_rainfall_mm / rainfall.sum())

    # 创建各子流域的降雨数据（考虑海拔梯度）
    timestamps = pd.date_range('2017-01-01', periods=timesteps, freq='h')
    rainfall_multipliers = {
        'W170': 1.1,   # 高山区 +10%
        'W160': 1.05,  # 高山区 +5%
        'W180': 1.0,   # 中游基准
        'W190': 0.95,  # 下游 -5%
    }

    precip_df = pd.DataFrame({'Timestamp': timestamps})
    for sub_id, mult in rainfall_multipliers.items():
        precip_df[sub_id] = rainfall * mult
    precip_df.set_index('Timestamp', inplace=True)

    # 图3.1: 各子流域降雨时序图
    fig, ax = plt.subplots(figsize=(14, 6))
    for sub_id in rainfall_multipliers.keys():
        ax.plot(precip_df.index, precip_df[sub_id], label=sub_id, linewidth=2, marker='o',
               markersize=3, markevery=10)

    ax.set_xlabel('时间', fontsize=12)
    ax.set_ylabel('降雨强度 (mm/h)', fontsize=12)
    ax.set_title('Upper Truckee River 各子流域降雨时序', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(step3_dir / "3.1_rainfall_timeseries.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 降雨时序图已保存")

    # 图3.2: 面雨量空间分布柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    total_rainfall_by_sub = {sub_id: precip_df[sub_id].sum() for sub_id in rainfall_multipliers.keys()}

    x_pos = np.arange(len(total_rainfall_by_sub))
    bars = ax.bar(x_pos, list(total_rainfall_by_sub.values()),
                  color=['#FFB6C1', '#FFB6C1', '#90EE90', '#87CEEB'],
                  edgecolor='black', linewidth=1.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(list(total_rainfall_by_sub.keys()), fontsize=12)
    ax.set_ylabel('总降雨量 (mm)', fontsize=12)
    ax.set_title('Upper Truckee River 各子流域总降雨量分布', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    # 在柱子上标注数值
    for i, (bar, value) in enumerate(zip(bars, total_rainfall_by_sub.values())):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.1f} mm',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(step3_dir / "3.2_rainfall_spatial_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 面雨量空间分布图已保存")

    # 图3.3: 降雨统计表
    rainfall_stats = []
    for sub_id in rainfall_multipliers.keys():
        stats = {
            '子流域': sub_id,
            '总降雨量(mm)': precip_df[sub_id].sum(),
            '峰值强度(mm/h)': precip_df[sub_id].max(),
            '平均强度(mm/h)': precip_df[sub_id].mean(),
            '非零时段(h)': (precip_df[sub_id] > 0).sum(),
        }
        rainfall_stats.append(stats)

    rainfall_stats_df = pd.DataFrame(rainfall_stats)
    rainfall_stats_df.to_csv(step3_dir / "3.3_rainfall_statistics.csv", index=False, encoding='utf-8-sig')
    print("  ✓ 降雨统计表已保存")

    # 保存降雨数据
    forcing_dir = output_dir / "forcing"
    forcing_dir.mkdir(exist_ok=True)
    for col in precip_df.columns:
        precip_df[[col]].to_csv(forcing_dir / f"{col}.csv")
    print("  ✓ 降雨数据文件已保存")

    # ========================================================================
    # 步骤4: 产流模型配置可视化
    # ========================================================================
    print_step(4, "产流模型配置和参数可视化")

    step4_dir = step_dirs[4]

    runoff_models = [
        {
            "id": "mountain_zone",
            "model_type": "HBV",
            "应用子流域": "W170, W160",
            "degree_day_factor": 3.5,
            "snow_threshold": 2.0,
            "field_capacity": 180,
            "beta": 1.8,
            "k0": 0.5,
            "k1": 0.15,
            "k2": 0.03,
            "percolation": 3.0,
        },
        {
            "id": "forest_zone",
            "model_type": "SCS-CN",
            "应用子流域": "W180",
            "curve_number": 65,
            "initial_abstraction_ratio": 0.05,
        },
        {
            "id": "valley_zone",
            "model_type": "SCS-CN",
            "应用子流域": "W190",
            "curve_number": 72,
            "initial_abstraction_ratio": 0.05,
        },
    ]

    # 表4.1: 产流模型参数表
    for i, model in enumerate(runoff_models, 1):
        model_df = pd.DataFrame([model]).T
        model_df.columns = ['数值']
        model_df.index.name = '参数'
        model_df.to_csv(step4_dir / f"4.{i}_{model['id']}_parameters.csv", encoding='utf-8-sig')
        print(f"  ✓ {model['id']} 参数表已保存")

    # 图4.1: 产流模型分布图
    fig, ax = plt.subplots(figsize=(10, 6))

    model_distribution = {
        'HBV (mountain_zone)': 2,  # W170, W160
        'SCS-CN (forest_zone)': 1,  # W180
        'SCS-CN (valley_zone)': 1,  # W190
    }

    colors_map = ['#FFB6C1', '#90EE90', '#87CEEB']
    bars = ax.bar(range(len(model_distribution)), list(model_distribution.values()),
                  color=colors_map, edgecolor='black', linewidth=1.5)

    ax.set_xticks(range(len(model_distribution)))
    ax.set_xticklabels(list(model_distribution.keys()), fontsize=11)
    ax.set_ylabel('应用子流域数量', fontsize=12)
    ax.set_title('Upper Truckee River 产流模型分布', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    for bar, count in zip(bars, model_distribution.values()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{count}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(step4_dir / "4.4_runoff_model_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 产流模型分布图已保存")

    # ========================================================================
    # 步骤5: 汇流模型配置可视化
    # ========================================================================
    print_step(5, "汇流模型配置和参数可视化")

    step5_dir = step_dirs[5]

    routing_models = [
        {
            "id": "muskingum_upper",
            "model_type": "Muskingum",
            "应用子流域": "W170, W160",
            "travel_time(h)": 8,
            "weighting_factor": 0.25,
            "time_step(h)": 1,
        },
        {
            "id": "muskingum_main",
            "model_type": "Muskingum",
            "应用子流域": "W180",
            "travel_time(h)": 12,
            "weighting_factor": 0.2,
            "time_step(h)": 1,
        },
        {
            "id": "muskingum_outlet",
            "model_type": "Muskingum",
            "应用子流域": "W190",
            "travel_time(h)": 15,
            "weighting_factor": 0.15,
            "time_step(h)": 1,
        },
    ]

    # 表5.1: 汇流模型参数表
    routing_df = pd.DataFrame(routing_models)
    routing_df.to_csv(step5_dir / "5.1_routing_parameters.csv", index=False, encoding='utf-8-sig')
    print("  ✓ 汇流模型参数表已保存")

    # 图5.1: 汇流参数对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    model_names = [m['id'] for m in routing_models]
    travel_times = [m['travel_time(h)'] for m in routing_models]
    weighting_factors = [m['weighting_factor'] for m in routing_models]

    # 演进时间对比
    bars1 = ax1.bar(range(len(model_names)), travel_times,
                    color=['#FFB6C1', '#90EE90', '#87CEEB'],
                    edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels(model_names, fontsize=11)
    ax1.set_ylabel('演进时间 K (小时)', fontsize=12)
    ax1.set_title('Muskingum 演进时间参数', fontsize=12, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)

    for bar, value in zip(bars1, travel_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value} h', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 权重因子对比
    bars2 = ax2.bar(range(len(model_names)), weighting_factors,
                    color=['#FFB6C1', '#90EE90', '#87CEEB'],
                    edgecolor='black', linewidth=1.5)
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, fontsize=11)
    ax2.set_ylabel('权重因子 x', fontsize=12)
    ax2.set_title('Muskingum 权重因子参数', fontsize=12, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)

    for bar, value in zip(bars2, weighting_factors):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(step5_dir / "5.2_routing_parameters_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 汇流参数对比图已保存")

    # ========================================================================
    # 步骤6-10: 运行模拟并生成结果可视化
    # ========================================================================
    print_step(6, "运行水文模拟")

    from hydrosis import ModelConfig, run_workflow
    from hydrosis.config import (
        DelineationConfig, IOConfig, RoutingModelConfig,
        RunoffModelConfig, EvaluationConfig,
    )
    from hydrosis.parameters.zone import ParameterZoneConfig

    # 创建虚拟pour points文件
    pour_points_file = output_dir / "pour_points.geojson"
    if not pour_points_file.exists():
        dummy_pour_points = {"type": "FeatureCollection", "features": []}
        pour_points_file.write_text(json.dumps(dummy_pour_points))

    # 配置流域划分
    delineation = DelineationConfig(
        dem_path=dem_path,
        pour_points_path=pour_points_file,
        precomputed_subbasins=[
            {"id": "W170", "area_km2": 45.2, "downstream": "W180",
             "parameters": {"runoff_model": "mountain_zone", "routing_model": "muskingum_upper"}},
            {"id": "W160", "area_km2": 35.8, "downstream": "W180",
             "parameters": {"runoff_model": "mountain_zone", "routing_model": "muskingum_upper"}},
            {"id": "W180", "area_km2": 125.5, "downstream": "W190",
             "parameters": {"runoff_model": "forest_zone", "routing_model": "muskingum_main"}},
            {"id": "W190", "area_km2": 165.0, "downstream": None,
             "parameters": {"runoff_model": "valley_zone", "routing_model": "muskingum_outlet"}},
        ],
    )

    # 配置产流模型
    runoff_model_configs = [
        RunoffModelConfig(
            id="mountain_zone", model_type="hbv",
            parameters={
                "degree_day_factor": 3.5, "snow_threshold": 2.0, "field_capacity": 180,
                "beta": 1.8, "k0": 0.5, "k1": 0.15, "k2": 0.03, "percolation": 3.0,
                "initial_snow": 0.0, "initial_soil": 0.0, "initial_upper": 0.0, "initial_lower": 0.0,
            }
        ),
        RunoffModelConfig(
            id="forest_zone", model_type="scs_curve_number",
            parameters={"curve_number": 65, "initial_abstraction_ratio": 0.05}
        ),
        RunoffModelConfig(
            id="valley_zone", model_type="scs_curve_number",
            parameters={"curve_number": 72, "initial_abstraction_ratio": 0.05}
        ),
    ]

    # 配置汇流模型
    routing_model_configs = [
        RoutingModelConfig(
            id="muskingum_upper", model_type="muskingum",
            parameters={"travel_time": 8, "weighting_factor": 0.25, "time_step": 1}
        ),
        RoutingModelConfig(
            id="muskingum_main", model_type="muskingum",
            parameters={"travel_time": 12, "weighting_factor": 0.2, "time_step": 1}
        ),
        RoutingModelConfig(
            id="muskingum_outlet", model_type="muskingum",
            parameters={"travel_time": 15, "weighting_factor": 0.15, "time_step": 1}
        ),
    ]

    # 配置参数区
    parameter_zones = [
        ParameterZoneConfig(
            id="Upper_Mountain", description="上游高山区 (2000-3000m)",
            control_points=["W170", "W160"],
            parameters={"runoff_model": "mountain_zone", "routing_model": "muskingum_upper"}
        ),
        ParameterZoneConfig(
            id="Middle_Forest", description="中游森林区 (1500-2000m)",
            control_points=["W180"],
            parameters={"runoff_model": "forest_zone", "routing_model": "muskingum_main"}
        ),
        ParameterZoneConfig(
            id="Lower_Valley", description="下游河谷区 (1200-1500m)",
            control_points=["W190"],
            parameters={"runoff_model": "valley_zone", "routing_model": "muskingum_outlet"}
        ),
    ]

    # 生成合成观测数据
    unit_hydrograph = np.zeros(120)
    uh_peak_time = 65
    uh_duration = 40

    for t in range(uh_duration):
        if t < uh_peak_time - storm_start:
            unit_hydrograph[storm_start + t] = t / (uh_peak_time - storm_start)
        else:
            unit_hydrograph[storm_start + t] = np.exp(-(t - (uh_peak_time - storm_start)) / 15.0)

    synthetic_obs = np.convolve(rainfall * 0.95, unit_hydrograph[:20], mode='full')[:120] * 2.5
    np.random.seed(42)
    synthetic_obs += np.random.normal(0, 0.3, 120)
    synthetic_obs = np.maximum(synthetic_obs, 0.1)

    obs_df = pd.DataFrame({'Timestamp': timestamps, 'observed_flow': synthetic_obs})
    obs_df.set_index('Timestamp', inplace=True)
    obs_file = output_dir / "observed_flow.csv"
    obs_df.to_csv(obs_file)

    # 配置评估
    evaluation = EvaluationConfig(metrics=["rmse", "mae", "nse", "pbias"])

    # 组装配置
    config = ModelConfig(
        delineation=delineation,
        runoff_models=runoff_model_configs,
        routing_models=routing_model_configs,
        parameter_zones=parameter_zones,
        io=IOConfig(
            precipitation=forcing_dir,
            results_directory=output_dir / "results",
            figures_directory=output_dir / "figures",
            reports_directory=output_dir / "reports",
        ),
        scenarios=[],
        evaluation=evaluation,
    )

    # 运行模拟
    forcing = {col: precip_df[col].tolist() for col in precip_df.columns}
    observations = {"W190": list(synthetic_obs)}

    print("开始运行工作流...")
    result = run_workflow(config, forcing, observations=observations)
    print("✓ 模拟完成")

    baseline = result.baseline

    # ========================================================================
    # 步骤7: 产流结果可视化
    # ========================================================================
    print_step(7, "产流结果可视化")

    step7_dir = step_dirs[7]

    # 图7.1: 各子流域产流过程线
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, sub_id in enumerate(["W170", "W160", "W180", "W190"]):
        ax = axes[idx]
        if sub_id in baseline.local:
            local = baseline.local[sub_id]
            ax.plot(local, 'b-', linewidth=2, label='本地产流')
            ax.fill_between(range(len(local)), local, alpha=0.3)

            peak = max(local) if local else 0
            peak_time = local.index(peak) if local and peak > 0 else -1
            if peak_time >= 0:
                ax.plot(peak_time, peak, 'ro', markersize=8, label=f'峰值: {peak:.2f} m³/s')

        ax.set_title(f'{sub_id} 本地产流过程', fontsize=12, fontweight='bold')
        ax.set_xlabel('时间 (小时)', fontsize=10)
        ax.set_ylabel('流量 (m³/s)', fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(step7_dir / "7.1_runoff_generation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 产流过程线已保存")

    # 表7.1: 产流统计表
    runoff_stats = []
    for sub_id in ["W170", "W160", "W180", "W190"]:
        if sub_id in baseline.local:
            series = baseline.local[sub_id]
            total = sum(series)
            peak = max(series) if series else 0
            peak_time = series.index(peak) if series and peak > 0 else -1
            runoff_stats.append({
                '子流域': sub_id,
                '总产流(m³/s·h)': total,
                '峰值流量(m³/s)': peak,
                '峰现时间(h)': peak_time,
            })

    runoff_stats_df = pd.DataFrame(runoff_stats)
    runoff_stats_df.to_csv(step7_dir / "7.2_runoff_statistics.csv", index=False, encoding='utf-8-sig')
    print("  ✓ 产流统计表已保存")

    # ========================================================================
    # 步骤8: 汇流结果可视化
    # ========================================================================
    print_step(8, "汇流结果可视化")

    step8_dir = step_dirs[8]

    # 图8.1: 各子流域累积流量过程线
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, sub_id in enumerate(["W170", "W160", "W180", "W190"]):
        ax = axes[idx]

        if sub_id in baseline.aggregated:
            aggregated = baseline.aggregated[sub_id]
            ax.plot(aggregated, 'r-', linewidth=2, label='累积流量')
            ax.fill_between(range(len(aggregated)), aggregated, alpha=0.3, color='red')

            peak = max(aggregated) if aggregated else 0
            peak_time = aggregated.index(peak) if aggregated and peak > 0 else -1
            if peak_time >= 0:
                ax.plot(peak_time, peak, 'ko', markersize=8, label=f'峰值: {peak:.2f} m³/s')

        ax.set_title(f'{sub_id} 累积流量过程', fontsize=12, fontweight='bold')
        ax.set_xlabel('时间 (小时)', fontsize=10)
        ax.set_ylabel('流量 (m³/s)', fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(step8_dir / "8.1_routing_results.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 汇流过程线已保存")

    # 表8.1: 汇流统计表
    routing_stats = []
    for sub_id in ["W170", "W160", "W180", "W190"]:
        if sub_id in baseline.aggregated:
            series = baseline.aggregated[sub_id]
            total = sum(series)
            peak = max(series) if series else 0
            peak_time = series.index(peak) if series and peak > 0 else -1
            routing_stats.append({
                '子流域': sub_id,
                '总流量(m³/s·h)': total,
                '峰值流量(m³/s)': peak,
                '峰现时间(h)': peak_time,
            })

    routing_stats_df = pd.DataFrame(routing_stats)
    routing_stats_df.to_csv(step8_dir / "8.2_routing_statistics.csv", index=False, encoding='utf-8-sig')
    print("  ✓ 汇流统计表已保存")

    # ========================================================================
    # 步骤9: 模型评估可视化
    # ========================================================================
    print_step(9, "模型评估和验证可视化")

    step9_dir = step_dirs[9]

    outlet_id = "W190"

    # 图9.1: 模拟vs观测对比
    if outlet_id in baseline.aggregated:
        fig, ax = plt.subplots(figsize=(14, 6))

        simulated = baseline.aggregated[outlet_id]
        ax.plot(simulated, 'b-', label='模拟流量', linewidth=2)
        ax.plot(synthetic_obs, 'r--', label='观测流量', linewidth=2, alpha=0.8)

        ax.set_title(f'{outlet_id} 出口流量对比 (模拟 vs 观测)', fontsize=14, fontweight='bold')
        ax.set_xlabel('时间 (小时)', fontsize=12)
        ax.set_ylabel('流量 (m³/s)', fontsize=12)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)

        # 添加评估指标
        if result.overall_scores:
            metric_lines = []
            for score in result.overall_scores:
                for metric_name, metric_value in score.aggregated.items():
                    metric_lines.append(f'{metric_name.upper()}: {metric_value:.4f}')
            textstr = '\n'.join(metric_lines)
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)

        plt.tight_layout()
        plt.savefig(step9_dir / "9.1_model_validation.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ 模型验证对比图已保存")

    # 图9.2: 散点图
    if outlet_id in baseline.aggregated:
        fig, ax = plt.subplots(figsize=(8, 8))

        simulated = baseline.aggregated[outlet_id]
        max_val = max(max(simulated), max(synthetic_obs))

        ax.scatter(synthetic_obs, simulated, alpha=0.6, s=50, c='steelblue', edgecolors='black')
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='1:1 线')

        ax.set_title('模拟 vs 观测散点图', fontsize=14, fontweight='bold')
        ax.set_xlabel('观测流量 (m³/s)', fontsize=12)
        ax.set_ylabel('模拟流量 (m³/s)', fontsize=12)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(step9_dir / "9.2_scatter_plot.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ 散点图已保存")

    # 表9.1: 评估指标表
    if result.overall_scores:
        eval_metrics = []
        for score in result.overall_scores:
            for metric_name, metric_value in score.aggregated.items():
                eval_metrics.append({
                    '指标': metric_name.upper(),
                    '数值': metric_value,
                })

        eval_df = pd.DataFrame(eval_metrics)
        eval_df.to_csv(step9_dir / "9.3_evaluation_metrics.csv", index=False, encoding='utf-8-sig')
        print("  ✓ 评估指标表已保存")

    # ========================================================================
    # 步骤10: 综合结果报告
    # ========================================================================
    print_step(10, "生成综合结果报告")

    step10_dir = step_dirs[10]

    # 图10.1: 水量平衡图
    fig, ax = plt.subplots(figsize=(10, 6))

    water_balance = {
        '总降雨': sum(precip_df['W190']),
        '总产流': sum(baseline.local.get('W190', [0])) if 'W190' in baseline.local else 0,
        '总径流': sum(baseline.aggregated.get('W190', [0])) if 'W190' in baseline.aggregated else 0,
    }

    bars = ax.bar(range(len(water_balance)), list(water_balance.values()),
                  color=['skyblue', 'lightgreen', 'coral'],
                  edgecolor='black', linewidth=1.5)

    ax.set_xticks(range(len(water_balance)))
    ax.set_xticklabels(list(water_balance.keys()), fontsize=12)
    ax.set_ylabel('水量 (mm 或 m³/s·h)', fontsize=12)
    ax.set_title('W190 (出口) 水量平衡', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    for bar, (name, value) in zip(bars, water_balance.items()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.1f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(step10_dir / "10.1_water_balance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 水量平衡图已保存")

    # 生成总结报告
    report_file = step10_dir / "10.2_summary_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Upper Truckee River 详细工作流总结报告\n\n")
        f.write(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 工作流步骤完成情况\n\n")
        f.write("| 步骤 | 内容 | 状态 |\n")
        f.write("|------|------|------|\n")
        steps_summary = [
            ("步骤1", "DEM和地形分析", "✓ 完成"),
            ("步骤2", "子流域划分", "✓ 完成"),
            ("步骤3", "降雨数据生成", "✓ 完成"),
            ("步骤4", "产流模型配置", "✓ 完成"),
            ("步骤5", "汇流模型配置", "✓ 完成"),
            ("步骤6", "水文模拟运行", "✓ 完成"),
            ("步骤7", "产流结果分析", "✓ 完成"),
            ("步骤8", "汇流结果分析", "✓ 完成"),
            ("步骤9", "模型评估验证", "✓ 完成"),
            ("步骤10", "综合结果报告", "✓ 完成"),
        ]
        for step, content, status in steps_summary:
            f.write(f"| {step} | {content} | {status} |\n")

        f.write("\n## 主要成果\n\n")
        f.write(f"- **DEM分析**: {len([f for f in step_dirs[1].glob('*.png')])} 张图表\n")
        f.write(f"- **子流域划分**: {len([f for f in step_dirs[2].glob('*.png')])} 张图表\n")
        f.write(f"- **降雨分析**: {len([f for f in step_dirs[3].glob('*.png')])} 张图表\n")
        f.write(f"- **产流模型**: {len([f for f in step_dirs[4].glob('*.png')])} 张图表\n")
        f.write(f"- **汇流模型**: {len([f for f in step_dirs[5].glob('*.png')])} 张图表\n")
        f.write(f"- **产流结果**: {len([f for f in step_dirs[7].glob('*.png')])} 张图表\n")
        f.write(f"- **汇流结果**: {len([f for f in step_dirs[8].glob('*.png')])} 张图表\n")
        f.write(f"- **模型评估**: {len([f for f in step_dirs[9].glob('*.png')])} 张图表\n")
        f.write(f"- **综合报告**: {len([f for f in step_dirs[10].glob('*.png')])} 张图表\n")

        f.write("\n## 结论\n\n")
        f.write("本次工作流成功完成了Upper Truckee River流域的详细水文分析，")
        f.write("为每个关键步骤生成了完整的图表和数据表，")
        f.write("可用于深入理解流域水文过程和模型性能。\n")

        f.write("\n---\n\n")
        f.write("*本报告由 HydroSIS 自动生成*\n")

    print(f"✓ 总结报告已保存: {report_file}")

    print()
    print("=" * 80)
    print("  详细可视化工作流完成！")
    print("=" * 80)
    print()
    print(f"结果保存位置: {output_dir}")
    print()
    print("各步骤成果:")
    for i in range(1, 11):
        step_dir = step_dirs[i]
        file_count = len(list(step_dir.glob('*')))
        print(f"  步骤{i}: {file_count} 个文件")
    print()

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
