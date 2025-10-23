"""雨量站分布优化：基于密度评价的闭环优化

根据密度评价报告，优化雨量站分布使其更加均匀。
采用闭环验证逻辑，最多尝试10次以避免无限循环。
"""
import json
import re
import random
import numpy as np
from pathlib import Path
from shapely.geometry import Point, shape as shapely_shape


def load_density_evaluation(csv_path):
    """加载雨量站密度评价结果"""
    zones_need_improvement = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            if not line.strip():
                continue

            parts = line.strip().split(',')
            zone_id = int(parts[0])
            area_km2 = float(parts[1])
            gauge_count = int(parts[2])
            density = float(parts[3])
            grade = parts[5]

            # 需要改进的分区：无覆盖、偏低、或分布不均
            if grade in ['无覆盖', '偏低']:
                zones_need_improvement.append({
                    'zone_id': zone_id,
                    'area_km2': area_km2,
                    'current_count': gauge_count,
                    'density': density,
                    'grade': grade
                })

    return zones_need_improvement


def load_parameter_zones(geojson_path):
    """加载参数分区几何"""
    with open(geojson_path, 'r') as f:
        data = json.load(f)

    zones = {}
    for feature in data['features']:
        props = feature['properties']
        zone_id = int(props.get('zone_id', props.get('id')))
        zones[zone_id] = {
            'geometry': shapely_shape(feature['geometry']),
            'area_km2': props.get('area_km2', 0)
        }

    return zones


def load_existing_gauges(geojson_path):
    """加载现有雨量站"""
    with open(geojson_path, 'r') as f:
        data = json.load(f)

    gauges = []
    for feature in data['features']:
        coords = feature['geometry']['coordinates']
        props = feature['properties']
        gauges.append({
            'id': props.get('id', props.get('gauge_id')),
            'coords': coords,
            'point': Point(coords)
        })

    return gauges


def calculate_target_gauge_count(area_km2, target_density=0.01):
    """计算目标雨量站数量

    目标密度：0.01站/km² (每100km²一个站，达到"良好"等级)
    """
    target_count = int(np.ceil(area_km2 * target_density))
    return max(1, target_count)  # 至少1个站


def generate_random_points_in_polygon(polygon, n_points, existing_points=None, min_distance=1000):
    """在多边形内生成随机点，保证与现有点有最小距离

    Args:
        polygon: Shapely多边形
        n_points: 需要生成的点数
        existing_points: 现有点列表
        min_distance: 与现有点的最小距离(米)

    Returns:
        新生成的点列表
    """
    minx, miny, maxx, maxy = polygon.bounds
    new_points = []
    max_attempts = n_points * 100  # 最多尝试次数
    attempts = 0

    while len(new_points) < n_points and attempts < max_attempts:
        attempts += 1

        # 生成随机点
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        point = Point(x, y)

        # 检查是否在多边形内
        if not polygon.contains(point):
            continue

        # 检查与现有点的距离
        too_close = False
        if existing_points:
            for existing_point in existing_points:
                if point.distance(existing_point) < min_distance:
                    too_close = True
                    break

        if not too_close:
            new_points.append(point)

    if len(new_points) < n_points:
        print(f"  ⚠️  警告: 只生成了{len(new_points)}/{n_points}个点（达到最大尝试次数）")

    return new_points


def optimize_gauge_distribution(zones_need_improvement, parameter_zones, existing_gauges,
                                output_dir, max_iterations=10):
    """优化雨量站分布的主函数

    Args:
        zones_need_improvement: 需要改进的分区列表
        parameter_zones: 参数分区几何数据
        existing_gauges: 现有雨量站
        output_dir: 输出目录
        max_iterations: 最大迭代次数（避免无限循环）
    """
    print("\n" + "="*80)
    print("雨量站分布优化（闭环验证）")
    print("="*80)

    if not zones_need_improvement:
        print("\n✅ 所有分区的雨量站分布均满足要求，无需优化")
        return existing_gauges, 0

    print(f"\n发现{len(zones_need_improvement)}个分区需要改进:")
    for zone_info in zones_need_improvement:
        print(f"  - Zone {zone_info['zone_id']}: {zone_info['grade']}, "
              f"当前{zone_info['current_count']}站, "
              f"面积{zone_info['area_km2']:.1f}km²")

    # 准备优化
    all_gauges = [g['point'] for g in existing_gauges]
    # Extract numeric part from gauge IDs like "station_01"
    gauge_numbers = []
    for g in existing_gauges:
        gid = str(g['id'])
        # Extract number from string like "station_01"
        match = re.search(r'\d+', gid)
        if match:
            gauge_numbers.append(int(match.group()))
        else:
            gauge_numbers.append(0)

    next_gauge_id = max(gauge_numbers) + 1 if gauge_numbers else 1
    new_gauges = []

    # 为每个需要改进的分区添加雨量站
    for zone_info in zones_need_improvement:
        zone_id = zone_info['zone_id']
        current_count = zone_info['current_count']

        if zone_id not in parameter_zones:
            print(f"  ⚠️  警告: 找不到Zone {zone_id}的几何数据，跳过")
            continue

        zone_geom = parameter_zones[zone_id]['geometry']

        # 计算目标站点数
        target_count = calculate_target_gauge_count(zone_info['area_km2'])
        additional_count = target_count - current_count

        if additional_count <= 0:
            continue

        print(f"\n  Zone {zone_id}: 需要增加{additional_count}个雨量站 "
              f"(当前{current_count} → 目标{target_count})")

        # 生成新的雨量站点
        new_points = generate_random_points_in_polygon(
            zone_geom,
            additional_count,
            existing_points=all_gauges,
            min_distance=2000  # 至少相距2km
        )

        for point in new_points:
            new_gauges.append({
                'id': f'station_{next_gauge_id:02d}',  # Format like "station_11"
                'coords': [point.x, point.y],
                'point': point,
                'zone_id': zone_id
            })
            all_gauges.append(point)
            next_gauge_id += 1

        print(f"    ✓ 成功生成{len(new_points)}个新雨量站")

    # 合并原有和新增的雨量站
    optimized_gauges = existing_gauges + new_gauges

    print(f"\n优化结果:")
    print(f"  - 原有雨量站: {len(existing_gauges)}个")
    print(f"  - 新增雨量站: {len(new_gauges)}个")
    print(f"  - 优化后总数: {len(optimized_gauges)}个")

    # 保存优化后的雨量站
    output_path = output_dir / "optimized_gauge_locations.geojson"
    save_gauges_geojson(optimized_gauges, output_path)

    return optimized_gauges, len(new_gauges)


def save_gauges_geojson(gauges, output_path):
    """保存雨量站为GeoJSON"""
    features = []
    for gauge in gauges:
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": gauge['coords']
            },
            "properties": {
                "id": gauge['id'],
                "gauge_id": gauge['id']
            }
        }

        # 如果是新增站点，添加zone_id
        if 'zone_id' in gauge:
            feature['properties']['zone_id'] = gauge['zone_id']

        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, indent=2, ensure_ascii=False)

    print(f"  ✓ 保存优化后的雨量站: {output_path.name}")


def validate_optimization(optimized_gauges, parameter_zones, output_dir):
    """验证优化结果

    Returns:
        validation_passed (bool): 验证是否通过
        new_evaluation (dict): 新的密度评价
    """
    print("\n" + "="*80)
    print("闭环验证：优化后密度评价")
    print("="*80)

    new_evaluation = []
    all_good = True

    for zone_id, zone_data in sorted(parameter_zones.items()):
        zone_geom = zone_data['geometry']
        zone_area = zone_data['area_km2']

        # 统计该分区内的雨量站数量
        gauge_count = sum(1 for g in optimized_gauges if zone_geom.contains(g['point']))

        # 计算密度
        density = gauge_count / zone_area if zone_area > 0 else 0
        coverage_area = zone_area / gauge_count if gauge_count > 0 else float('inf')

        # 评价等级
        if density >= 0.02:
            grade = "优秀"
        elif density >= 0.01:
            grade = "良好"
        elif density >= 0.005:
            grade = "中等"
        elif gauge_count > 0:
            grade = "偏低"
            all_good = False
        else:
            grade = "无覆盖"
            all_good = False

        new_evaluation.append({
            'zone_id': zone_id,
            'area_km2': zone_area,
            'gauge_count': gauge_count,
            'density': density,
            'coverage_area': coverage_area,
            'grade': grade
        })

        print(f"  Zone {zone_id}: {gauge_count}站, "
              f"密度={density:.6f} 站/km², "
              f"等级={grade}")

    # 保存新的评价结果
    csv_path = output_dir / "optimized_gauge_density_evaluation.csv"
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('分区ID,分区面积(km²),雨量站数量,密度(站/km²),覆盖面积(km²/站),密度等级\n')
        for result in sorted(new_evaluation, key=lambda x: x['zone_id']):
            f.write(f"{result['zone_id']},{result['area_km2']:.2f},"
                   f"{result['gauge_count']},{result['density']:.6f},"
                   f"{result['coverage_area']:.2f},{result['grade']}\n")

    print(f"\n  ✓ 保存新的密度评价: {csv_path.name}")

    if all_good:
        print("\n✅ 验证通过：所有分区密度均达到'中等'或以上等级")
    else:
        print("\n⚠️  仍有分区未达标，可能需要进一步优化")

    print("="*80)

    return all_good, new_evaluation


def main():
    """主函数"""
    print("\n" + "="*80)
    print("雨量站分布优化系统（基于闭环验证）")
    print("="*80)

    # 路径设置
    base_dir = Path("results/upper_truckee_complete_11steps")
    step_05_dir = base_dir / "step_05_rain_gauges"
    output_dir = base_dir / "rain_gauge_optimization"
    output_dir.mkdir(exist_ok=True)

    # 输入文件
    density_csv = step_05_dir / "5.3_gauge_density_evaluation.csv"
    zones_geojson = base_dir / "parameters" / "parameter_zones.geojson"
    gauges_geojson = base_dir / "step_07_thiessen" / "7.2_gauge_locations.geojson"

    # 检查文件存在性
    for path in [density_csv, zones_geojson, gauges_geojson]:
        if not path.exists():
            print(f"  ❌ 文件不存在: {path}")
            return

    # 1. 加载密度评价结果
    print("\n⚙ 加载密度评价结果...")
    zones_need_improvement = load_density_evaluation(density_csv)
    print(f"  ✓ 发现{len(zones_need_improvement)}个分区需要改进")

    if len(zones_need_improvement) == 0:
        print("\n✅ 所有分区密度均达标，无需优化")
        return

    # 2. 加载参数分区和现有雨量站
    print("\n⚙ 加载参数分区和现有雨量站...")
    parameter_zones = load_parameter_zones(zones_geojson)
    existing_gauges = load_existing_gauges(gauges_geojson)
    print(f"  ✓ 加载{len(parameter_zones)}个参数分区")
    print(f"  ✓ 加载{len(existing_gauges)}个现有雨量站")

    # 3. 执行优化（最多10次迭代，避免无限循环）
    max_iterations = 10
    iteration = 0
    validation_passed = False

    while iteration < max_iterations and not validation_passed:
        iteration += 1
        print(f"\n{'='*80}")
        print(f"第{iteration}次优化迭代")
        print(f"{'='*80}")

        # 执行优化
        optimized_gauges, added_count = optimize_gauge_distribution(
            zones_need_improvement,
            parameter_zones,
            existing_gauges,
            output_dir,
            max_iterations
        )

        if added_count == 0:
            print("\n⚠️  无法添加更多雨量站，优化终止")
            break

        # 验证优化结果
        validation_passed, new_evaluation = validate_optimization(
            optimized_gauges,
            parameter_zones,
            output_dir
        )

        if validation_passed:
            print(f"\n✅ 优化成功！经过{iteration}次迭代达到目标")
            break

        # 准备下一次迭代
        existing_gauges = optimized_gauges
        zones_need_improvement = [
            eval_result for eval_result in new_evaluation
            if eval_result['grade'] in ['无覆盖', '偏低']
        ]

        if iteration >= max_iterations:
            print(f"\n⚠️  达到最大迭代次数({max_iterations})，优化终止")
            print("  建议：手动调整雨量站位置或降低密度要求")

    # 4. 生成优化报告
    print("\n⚙ 生成优化报告...")
    report_path = output_dir / "optimization_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("雨量站分布优化报告\n")
        f.write("="*80 + "\n\n")

        f.write(f"优化迭代次数: {iteration}\n")
        f.write(f"验证状态: {'✅ 通过' if validation_passed else '⚠️  部分达标'}\n\n")

        f.write("优化前后对比:\n")
        f.write(f"  原始雨量站数: {len(existing_gauges) - added_count}个\n")
        f.write(f"  新增雨量站数: {added_count}个\n")
        f.write(f"  优化后总数: {len(optimized_gauges)}个\n\n")

        f.write("各分区密度评价:\n")
        for result in sorted(new_evaluation, key=lambda x: x['zone_id']):
            f.write(f"  Zone {result['zone_id']}: {result['gauge_count']}站, "
                   f"密度={result['density']:.6f} 站/km², "
                   f"等级={result['grade']}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("优化策略:\n")
        f.write("  - 目标密度: 0.01 站/km² (良好等级)\n")
        f.write("  - 最小站间距: 2000米\n")
        f.write("  - 最大迭代次数: 10次（避免无限循环）\n")
        f.write("="*80 + "\n")

    print(f"  ✓ 生成优化报告: {report_path.name}")

    print("\n" + "="*80)
    print("✅ 雨量站优化完成！")
    print(f"📁 输出目录: {output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    random.seed(42)
    np.random.seed(42)
    main()
