#!/usr/bin/env python3
"""
创建增强的可视化图表
包括：雨量站分布图、改进的子流域降雨图、降雨演变GIF
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import rasterio

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def create_rain_gauge_distribution_map():
    """创建雨量站分布图"""
    print("\n" + "="*80)
    print("创建雨量站分布图")
    print("="*80)

    # 路径设置
    base_dir = Path("results/upper_truckee_complete_11steps")
    dem_path = Path("data/Upper_Truckee_River/terrain/UpTruckeeRv_S10_NED_30m/00/elevation.tif")
    gauge_file = base_dir / "step_07_thiessen" / "7.2_gauge_locations.geojson"
    zones_file = base_dir / "parameters" / "parameter_zones.geojson"
    thiessen_file = base_dir / "step_07_thiessen" / "7.1_thiessen_polygons.geojson"
    output_file = base_dir / "step_07_thiessen" / "7.3_rain_gauge_distribution_map.png"

    if not gauge_file.exists():
        print(f"  ✗ 雨量站文件不存在: {gauge_file}")
        return

    # 加载DEM作为底图
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        dem = np.where(np.isfinite(dem), dem, np.nan)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    # 加载雨量站
    with open(gauge_file, 'r') as f:
        gauge_data = json.load(f)

    gauges = []
    for feature in gauge_data['features']:
        coords = feature['geometry']['coordinates']
        station_id = feature['properties']['id']
        gauges.append({'id': station_id, 'x': coords[0], 'y': coords[1]})

    # 加载参数分区
    zones = []
    if zones_file.exists():
        with open(zones_file, 'r') as f:
            zones_data = json.load(f)
        for feature in zones_data['features']:
            zones.append(feature)

    # 加载泰森多边形
    thiessen_polys = []
    if thiessen_file.exists():
        with open(thiessen_file, 'r') as f:
            thiessen_data = json.load(f)
        for feature in thiessen_data['features']:
            thiessen_polys.append(feature)

    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 12))

    # 绘制DEM底图
    im = ax.imshow(dem, extent=extent, cmap='terrain', alpha=0.6, aspect='auto')

    # 绘制参数分区边界
    if zones:
        for feature in zones:
            geom = feature['geometry']
            zone_id = feature['properties'].get('zone_id', feature['properties'].get('id'))

            if geom['type'] == 'Polygon':
                coords = np.array(geom['coordinates'][0])
                poly = Polygon(coords, fill=False, edgecolor='blue', linewidth=2, alpha=0.8)
                ax.add_patch(poly)
                # 添加分区标签
                centroid_x = np.mean(coords[:, 0])
                centroid_y = np.mean(coords[:, 1])
                ax.text(centroid_x, centroid_y, f'Zone {zone_id}',
                       fontsize=14, fontweight='bold', ha='center',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            elif geom['type'] == 'MultiPolygon':
                for poly_coords in geom['coordinates']:
                    coords = np.array(poly_coords[0])
                    poly = Polygon(coords, fill=False, edgecolor='blue', linewidth=2, alpha=0.8)
                    ax.add_patch(poly)

    # 绘制泰森多边形（轻微透明）
    if thiessen_polys:
        for i, feature in enumerate(thiessen_polys):
            geom = feature['geometry']
            if geom['type'] == 'Polygon':
                coords = np.array(geom['coordinates'][0])
                poly = Polygon(coords, fill=False, edgecolor='gray',
                             linewidth=0.5, alpha=0.3, linestyle='--')
                ax.add_patch(poly)

    # 绘制雨量站点
    xs = [g['x'] for g in gauges]
    ys = [g['y'] for g in gauges]
    ax.scatter(xs, ys, c='red', s=50, marker='o', edgecolors='darkred',
              linewidths=1.5, alpha=0.9, zorder=5, label='Rain Gauges')

    # 添加站点编号（仅显示部分以避免拥挤）
    for i, gauge in enumerate(gauges):
        if i % 5 == 0:  # 每5个显示一个编号
            ax.text(gauge['x'], gauge['y'], gauge['id'],
                   fontsize=6, ha='right', va='bottom', color='darkred')

    # 设置标题和标签
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Rain Gauge Distribution Map ({len(gauges)} stations)\n'
                 f'Upper Truckee River Basin', fontsize=16, fontweight='bold')

    # 添加图例
    ax.legend(loc='upper right', fontsize=10)

    # 添加网格
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    # 添加色标
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
    cbar.set_label('Elevation (m)', fontsize=10)

    # 添加统计信息文本框
    stats_text = f"Total Stations: {len(gauges)}\n"
    if zones:
        stats_text += f"Parameter Zones: {len(zones)}\n"
    stats_text += f"Stratified Sampling: Yes"

    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"  ✓ 保存雨量站分布图: {output_file.name}")
    print(f"  ✓ 共{len(gauges)}个雨量站")


def create_improved_subbasin_precipitation_plot():
    """创建改进的子流域降雨时间序列图"""
    print("\n" + "="*80)
    print("创建改进的子流域降雨时间序列图")
    print("="*80)

    base_dir = Path("results/upper_truckee_complete_11steps")
    data_file = base_dir / "step_08_areal_rainfall" / "8.2_subbasin_areal_precipitation.csv"
    output_file = base_dir / "step_08_areal_rainfall" / "8.4_subbasin_precipitation_hyetograph.png"

    if not data_file.exists():
        print(f"  ✗ 数据文件不存在: {data_file}")
        return

    # 读取数据
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)

    # 选择部分子流域显示（避免太拥挤）
    num_subbasins = len(df.columns)
    sample_cols = df.columns[::max(1, num_subbasins // 10)]  # 最多显示10个

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # 子图1：选择的子流域时间序列
    for col in sample_cols:
        ax1.plot(df.index, df[col], label=f'Subbasin {col}', alpha=0.7, linewidth=1.5)

    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Precipitation (mm/hr)', fontsize=12)
    ax1.set_title(f'Subbasin Areal Precipitation Time Series (Sample: {len(sample_cols)} of {num_subbasins} subbasins)',
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    # 子图2：所有子流域的热力图
    # 转置数据以便时间在x轴
    heatmap_data = df.T.values

    im = ax2.imshow(heatmap_data, aspect='auto', cmap='Blues',
                    extent=[0, len(df), 0, len(df.columns)],
                    interpolation='nearest')

    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Subbasin ID', fontsize=12)
    ax2.set_title('Precipitation Distribution Across All Subbasins (Heatmap)',
                 fontsize=14, fontweight='bold')

    # 设置y轴标签（显示部分子流域ID）
    yticks = np.linspace(0, len(df.columns), min(20, len(df.columns)), dtype=int)
    ax2.set_yticks(yticks)
    ax2.set_yticklabels([df.columns[i] if i < len(df.columns) else '' for i in yticks])

    # 添加色标
    cbar = plt.colorbar(im, ax=ax2, orientation='vertical', pad=0.02)
    cbar.set_label('Precipitation (mm/hr)', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"  ✓ 保存改进的子流域降雨图: {output_file.name}")
    print(f"  ✓ 包含{num_subbasins}个子流域，显示样本{len(sample_cols)}个")


def create_precipitation_animation():
    """创建降雨演变动态GIF"""
    print("\n" + "="*80)
    print("创建降雨演变动态GIF")
    print("="*80)

    base_dir = Path("results/upper_truckee_complete_11steps")
    precip_file = base_dir / "step_08_areal_rainfall" / "8.2_subbasin_areal_precipitation.csv"
    zones_file = base_dir / "parameters" / "parameter_subbasins.geojson"
    output_file = base_dir / "step_08_areal_rainfall" / "8.6_precipitation_animation.gif"

    if not precip_file.exists() or not zones_file.exists():
        print(f"  ✗ 所需文件不存在")
        return

    # 读取降雨数据
    df = pd.read_csv(precip_file, index_col=0, parse_dates=True)

    # 读取子流域几何
    with open(zones_file, 'r') as f:
        subbasin_data = json.load(f)

    # 创建子流域ID到几何的映射
    subbasin_geoms = {}
    for feature in subbasin_data['features']:
        sub_id = feature['properties']['subzone_id']
        subbasin_geoms[sub_id] = feature['geometry']

    # 准备动画
    fig, ax = plt.subplots(figsize=(12, 10))

    # 每10个时间步创建一帧（减少总帧数）
    time_steps = range(0, len(df), 10)

    def update(frame_idx):
        ax.clear()
        time_idx = time_steps[frame_idx]
        current_time = df.index[time_idx]

        # 获取当前时刻的降雨数据
        current_precip = df.iloc[time_idx]

        # 绘制子流域并填充颜色
        patches = []
        colors = []

        for sub_id in df.columns:
            if sub_id in subbasin_geoms:
                geom = subbasin_geoms[sub_id]
                precip_value = current_precip[sub_id]

                if geom['type'] == 'Polygon':
                    coords = np.array(geom['coordinates'][0])
                    poly = Polygon(coords, closed=True)
                    patches.append(poly)
                    colors.append(precip_value)
                elif geom['type'] == 'MultiPolygon':
                    for poly_coords in geom['coordinates']:
                        coords = np.array(poly_coords[0])
                        poly = Polygon(coords, closed=True)
                        patches.append(poly)
                        colors.append(precip_value)

        # 创建颜色映射
        p = PatchCollection(patches, cmap='Blues', alpha=0.8)
        p.set_array(np.array(colors))
        p.set_clim([0, df.max().max()])
        ax.add_collection(p)

        # 设置坐标轴范围
        all_coords = []
        for sub_id in subbasin_geoms:
            geom = subbasin_geoms[sub_id]
            if geom['type'] == 'Polygon':
                all_coords.extend(geom['coordinates'][0])
            elif geom['type'] == 'MultiPolygon':
                for poly in geom['coordinates']:
                    all_coords.extend(poly[0])

        if all_coords:
            all_coords = np.array(all_coords)
            ax.set_xlim(all_coords[:, 0].min(), all_coords[:, 0].max())
            ax.set_ylim(all_coords[:, 1].min(), all_coords[:, 1].max())

        ax.set_aspect('equal')
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title(f'Precipitation Distribution\nTime: {current_time}\n'
                    f'Frame {frame_idx+1}/{len(time_steps)}',
                    fontsize=14, fontweight='bold')

        # 添加色标（仅在第一帧）
        if frame_idx == 0:
            cbar = plt.colorbar(p, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
            cbar.set_label('Precipitation (mm/hr)', fontsize=10)

        return ax,

    # 创建动画
    print(f"  ⚙ 创建动画，共{len(time_steps)}帧...")
    anim = animation.FuncAnimation(fig, update, frames=len(time_steps),
                                   interval=200, blit=False, repeat=True)

    # 保存GIF
    print(f"  ⚙ 保存GIF文件（这可能需要一些时间）...")
    anim.save(output_file, writer='pillow', fps=5, dpi=100)
    plt.close()

    print(f"  ✓ 保存降雨演变GIF: {output_file.name}")
    print(f"  ✓ 共{len(time_steps)}帧，时长约{len(time_steps)/5:.1f}秒")


def main():
    """主函数"""
    print("="*80)
    print("HydroSIS 增强可视化生成器")
    print("="*80)

    # 1. 创建雨量站分布图
    create_rain_gauge_distribution_map()

    # 2. 改进子流域降雨图
    create_improved_subbasin_precipitation_plot()

    # 3. 创建降雨演变动画
    create_precipitation_animation()

    print("\n" + "="*80)
    print("✓ 所有增强可视化已完成！")
    print("="*80)


if __name__ == '__main__':
    main()
