"""增强Step 5输出：雨量站分布图、泰森多边形叠加、密度评价

根据用户需求生成：
1. 雨量站分布图（DEM上叠加雨量站点）
2. 雨量站+泰森多边形叠加图
3. 各参数分区的雨量站点密度评价指标
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import numpy as np
import rasterio
from shapely.geometry import shape as shapely_shape, Point, Polygon
from shapely.ops import unary_union

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_gauge_locations(geojson_path):
    """加载雨量站位置"""
    with open(geojson_path, 'r') as f:
        data = json.load(f)

    gauges = []
    for feature in data['features']:
        coords = feature['geometry']['coordinates']
        props = feature['properties']
        gauges.append({
            'id': props.get('id', props.get('gauge_id', 'unknown')),
            'x': coords[0],
            'y': coords[1],
            'geometry': Point(coords)
        })
    return gauges

def load_thiessen_polygons(geojson_path):
    """加载泰森多边形"""
    with open(geojson_path, 'r') as f:
        data = json.load(f)

    polygons = []
    for feature in data['features']:
        geom = shapely_shape(feature['geometry'])
        props = feature['properties']
        polygons.append({
            'gauge_id': props.get('gauge_id', props.get('id', 'unknown')),
            'geometry': geom
        })
    return polygons

def load_parameter_zones(geojson_path):
    """加载参数分区"""
    with open(geojson_path, 'r') as f:
        data = json.load(f)

    zones = []
    for feature in data['features']:
        geom = shapely_shape(feature['geometry'])
        props = feature['properties']
        zones.append({
            'zone_id': props.get('zone_id', props.get('id', 'unknown')),
            'area_km2': props.get('area_km2', 0),
            'geometry': geom
        })
    return zones

def plot_gauge_distribution_on_dem(dem_path, gauges, output_path):
    """绘制雨量站在DEM上的分布图"""
    # 读取DEM
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        extent = [src.bounds.left, src.bounds.right,
                 src.bounds.bottom, src.bounds.top]

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10))

    # 绘制DEM
    dem_plot = ax.imshow(dem, extent=extent, cmap='terrain', alpha=0.8)
    plt.colorbar(dem_plot, ax=ax, label='高程 (m)', fraction=0.046, pad=0.04)

    # 绘制雨量站
    gauge_x = [g['x'] for g in gauges]
    gauge_y = [g['y'] for g in gauges]

    ax.scatter(gauge_x, gauge_y, c='red', s=150, marker='^',
              edgecolors='black', linewidths=2, zorder=5, label='雨量站')

    # 添加雨量站编号
    for gauge in gauges:
        ax.annotate(str(gauge['id']),
                   xy=(gauge['x'], gauge['y']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    ax.set_xlabel('X坐标 (m)', fontsize=12)
    ax.set_ylabel('Y坐标 (m)', fontsize=12)
    ax.set_title('雨量站分布图', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 生成雨量站分布图: {output_path.name}")

def plot_gauges_with_thiessen(dem_path, gauges, thiessen_polygons, output_path):
    """绘制雨量站+泰森多边形叠加图"""
    # 读取DEM
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        extent = [src.bounds.left, src.bounds.right,
                 src.bounds.bottom, src.bounds.top]

    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 12))

    # 绘制DEM作为背景
    dem_plot = ax.imshow(dem, extent=extent, cmap='terrain', alpha=0.5)

    # 绘制泰森多边形
    patches = []
    colors = plt.cm.Set3(np.linspace(0, 1, len(thiessen_polygons)))

    for i, poly_data in enumerate(thiessen_polygons):
        geom = poly_data['geometry']
        if geom.geom_type == 'Polygon':
            coords = list(geom.exterior.coords)
            patch = mpatches.Polygon(coords, closed=True,
                                    edgecolor='black', linewidth=2,
                                    facecolor=colors[i], alpha=0.3)
            ax.add_patch(patch)

            # 添加泰森多边形编号（在中心）
            centroid = geom.centroid
            ax.text(centroid.x, centroid.y, f"T{poly_data['gauge_id']}",
                   fontsize=10, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.6))

    # 绘制雨量站
    gauge_x = [g['x'] for g in gauges]
    gauge_y = [g['y'] for g in gauges]

    ax.scatter(gauge_x, gauge_y, c='red', s=200, marker='^',
              edgecolors='darkred', linewidths=3, zorder=10, label='雨量站')

    # 添加雨量站编号
    for gauge in gauges:
        ax.annotate(f"G{gauge['id']}",
                   xy=(gauge['x'], gauge['y']),
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                           edgecolor='red', linewidth=2, alpha=0.9),
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    ax.set_xlabel('X坐标 (m)', fontsize=12)
    ax.set_ylabel('Y坐标 (m)', fontsize=12)
    ax.set_title('雨量站分布与泰森多边形', fontsize=14, fontweight='bold')

    # 创建图例
    legend_elements = [
        mpatches.Patch(facecolor='gray', edgecolor='black', alpha=0.3, label='泰森多边形'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red',
                   markeredgecolor='darkred', markersize=12, label='雨量站')
    ]
    ax.legend(handles=legend_elements, fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 生成雨量站泰森多边形叠加图: {output_path.name}")

def evaluate_gauge_density_by_zone(gauges, zones, output_path):
    """评价各参数分区的雨量站点密度"""
    results = []

    for zone in zones:
        zone_id = zone['zone_id']
        zone_geom = zone['geometry']
        zone_area = zone['area_km2']

        # 统计该分区内的雨量站数量
        gauge_count = sum(1 for g in gauges if zone_geom.contains(g['geometry']))

        # 计算密度指标
        density = gauge_count / zone_area if zone_area > 0 else 0  # 站点/km²
        coverage_area = zone_area / gauge_count if gauge_count > 0 else float('inf')  # km²/站点

        # 评价等级
        if density >= 0.02:  # ≥0.02站/km² (每50km²一个站)
            grade = "优秀"
        elif density >= 0.01:  # ≥0.01站/km² (每100km²一个站)
            grade = "良好"
        elif density >= 0.005:  # ≥0.005站/km² (每200km²一个站)
            grade = "中等"
        elif gauge_count > 0:
            grade = "偏低"
        else:
            grade = "无覆盖"

        results.append({
            'zone_id': zone_id,
            'area_km2': zone_area,
            'gauge_count': gauge_count,
            'density_per_km2': density,
            'coverage_area_per_gauge': coverage_area,
            'grade': grade
        })

    # 保存为CSV
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('分区ID,分区面积(km²),雨量站数量,密度(站/km²),覆盖面积(km²/站),密度等级\n')
        for r in results:
            f.write(f"{r['zone_id']},{r['area_km2']:.2f},{r['gauge_count']},"
                   f"{r['density_per_km2']:.6f},{r['coverage_area_per_gauge']:.2f},{r['grade']}\n")

    print(f"  ✓ 生成雨量站密度评价: {output_path.name}")

    # 打印摘要
    total_area = sum(z['area_km2'] for z in zones)
    total_gauges = len(gauges)
    overall_density = total_gauges / total_area if total_area > 0 else 0

    print(f"\n  雨量站密度评价摘要:")
    print(f"    - 流域总面积: {total_area:.2f} km²")
    print(f"    - 雨量站总数: {total_gauges}个")
    print(f"    - 平均密度: {overall_density:.6f} 站/km² ({total_area/total_gauges:.2f} km²/站)")
    print(f"    - 分区详情:")
    for r in sorted(results, key=lambda x: x['zone_id']):
        print(f"      Zone {r['zone_id']}: {r['gauge_count']}站, "
              f"密度={r['density_per_km2']:.6f} 站/km², 等级={r['grade']}")

    return results

def main():
    """主函数"""
    print("\n" + "="*80)
    print("Step 5 输出增强：雨量站分布与密度评价")
    print("="*80)

    # 路径设置
    base_dir = Path("results/upper_truckee_complete_11steps")
    output_dir = base_dir / "step_05_rain_gauges"
    output_dir.mkdir(exist_ok=True)

    # 输入文件
    dem_path = Path("data/Upper_Truckee_River/terrain/UpTruckeeRv_S10_NED_30m/00/elevation.tif")
    gauge_locations_path = base_dir / "step_07_thiessen" / "7.2_gauge_locations.geojson"
    thiessen_path = base_dir / "step_07_thiessen" / "7.1_thiessen_polygons.geojson"
    zones_path = base_dir / "parameters" / "parameter_zones.geojson"

    # 检查文件存在性
    for path in [dem_path, gauge_locations_path, thiessen_path, zones_path]:
        if not path.exists():
            print(f"  ❌ 文件不存在: {path}")
            return

    # 1. 加载数据
    print("\n⚙ 加载数据...")
    gauges = load_gauge_locations(gauge_locations_path)
    thiessen_polygons = load_thiessen_polygons(thiessen_path)
    zones = load_parameter_zones(zones_path)

    print(f"  ✓ 加载{len(gauges)}个雨量站")
    print(f"  ✓ 加载{len(thiessen_polygons)}个泰森多边形")
    print(f"  ✓ 加载{len(zones)}个参数分区")

    # 2. 生成雨量站分布图（DEM叠加）
    print("\n⚙ 生成雨量站分布图...")
    output_path = output_dir / "5.1_gauge_distribution_map.png"
    plot_gauge_distribution_on_dem(dem_path, gauges, output_path)

    # 3. 生成雨量站+泰森多边形叠加图
    print("\n⚙ 生成雨量站泰森多边形叠加图...")
    output_path = output_dir / "5.2_gauge_thiessen_overlay.png"
    plot_gauges_with_thiessen(dem_path, gauges, thiessen_polygons, output_path)

    # 4. 评价各分区雨量站密度
    print("\n⚙ 评价雨量站密度...")
    output_path = output_dir / "5.3_gauge_density_evaluation.csv"
    density_results = evaluate_gauge_density_by_zone(gauges, zones, output_path)

    # 5. 生成结果报告
    print("\n⚙ 生成结果报告...")
    report_path = output_dir / "5.4_enhancement_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Step 5 增强输出报告：雨量站分布与密度评价\n")
        f.write("="*80 + "\n\n")

        f.write("生成文件清单:\n")
        f.write("  1. 5.1_gauge_distribution_map.png - 雨量站分布图(DEM叠加)\n")
        f.write("  2. 5.2_gauge_thiessen_overlay.png - 雨量站泰森多边形叠加图\n")
        f.write("  3. 5.3_gauge_density_evaluation.csv - 雨量站密度评价指标\n")
        f.write("  4. 5.4_enhancement_report.txt - 本报告\n\n")

        f.write("雨量站基本信息:\n")
        f.write(f"  - 雨量站总数: {len(gauges)}个\n")
        f.write(f"  - 泰森多边形数: {len(thiessen_polygons)}个\n")
        f.write(f"  - 参数分区数: {len(zones)}个\n\n")

        total_area = sum(z['area_km2'] for z in zones)
        overall_density = len(gauges) / total_area if total_area > 0 else 0

        f.write("密度评价统计:\n")
        f.write(f"  - 流域总面积: {total_area:.2f} km²\n")
        f.write(f"  - 平均密度: {overall_density:.6f} 站/km²\n")
        f.write(f"  - 平均覆盖: {total_area/len(gauges):.2f} km²/站\n\n")

        f.write("各分区密度详情:\n")
        for r in sorted(density_results, key=lambda x: x['zone_id']):
            f.write(f"  - Zone {r['zone_id']}: {r['gauge_count']}站, "
                   f"密度={r['density_per_km2']:.6f} 站/km², "
                   f"覆盖={r['coverage_area_per_gauge']:.2f} km²/站, "
                   f"等级={r['grade']}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("密度等级标准:\n")
        f.write("  - 优秀: ≥0.02 站/km² (每50km²一个站)\n")
        f.write("  - 良好: ≥0.01 站/km² (每100km²一个站)\n")
        f.write("  - 中等: ≥0.005 站/km² (每200km²一个站)\n")
        f.write("  - 偏低: <0.005 站/km² 但有站点\n")
        f.write("  - 无覆盖: 无雨量站\n")
        f.write("="*80 + "\n")

    print(f"  ✓ 生成结果报告: {report_path.name}")

    print("\n" + "="*80)
    print(f"✅ Step 5输出增强完成！共生成4个文件")
    print(f"📁 输出目录: {output_dir}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
