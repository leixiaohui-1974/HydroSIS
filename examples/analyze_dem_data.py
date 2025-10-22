#!/usr/bin/env python3
"""分析DEM数据的合理性"""

import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hydrosis.delineation.simple_grid import load_dem, load_pour_points

def analyze_dem(dem):
    """分析DEM数据"""
    print(f"DEM网格大小: {dem.shape}")
    print(f"单元格大小: {dem.transform.pixel_width}m x {dem.transform.pixel_height}m")
    print(f"单元格面积: {dem.transform.pixel_width * dem.transform.pixel_height} m2 = {dem.transform.pixel_width * dem.transform.pixel_height / 1000000} km2")
    
    # 统计高程信息
    elevations = []
    for row in dem.elevations:
        elevations.extend(row)
    
    min_elev = min(elevations)
    max_elev = max(elevations)
    mean_elev = sum(elevations) / len(elevations)
    
    print(f"\n高程统计:")
    print(f"  最小高程: {min_elev}")
    print(f"  最大高程: {max_elev}")
    print(f"  平均高程: {mean_elev:.2f}")
    print(f"  高程差: {max_elev - min_elev}")
    
    # 检查是否有明显的河道
    print("\n检查可能的河道位置:")
    # 河道应该是局部低点，周围高程较高
    river_candidates = []
    
    for row in range(1, dem.shape[0]-1):
        for col in range(1, dem.shape[1]-1):
            center_elev = dem.elevations[row][col]
            
            # 检查周围8个单元格
            neighbors = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    neighbors.append(dem.elevations[row+dr][col+dc])
            
            # 如果中心点比周围大多数点都低，可能是河道
            if sum(1 for n in neighbors if n > center_elev) >= 6:
                river_candidates.append((row, col, center_elev))
    
    print(f"  发现 {len(river_candidates)} 个可能的河道点")
    
    # 显示一些河道候选点
    if river_candidates:
        print("  前10个河道候选点 (行, 列, 高程):")
        for i, (row, col, elev) in enumerate(river_candidates[:10]):
            x, y = dem.transform.cell_centre(row, col)
            print(f"    {i+1}. ({row}, {col}) -> ({x:.1f}, {y:.1f}), 高程: {elev}")
    
    return river_candidates

def analyze_flow_accumulation(dem, pour_points):
    """分析汇流累积量"""
    from hydrosis.delineation.simple_grid import compute_flow_directions, compute_flow_accumulation
    
    print("\n计算流向和汇流累积量...")
    flow = compute_flow_directions(dem)
    accumulation = compute_flow_accumulation(dem, flow)
    
    # 统计汇流累积量
    acc_values = list(accumulation.values())
    min_acc = min(acc_values)
    max_acc = max(acc_values)
    mean_acc = sum(acc_values) / len(acc_values)
    
    print(f"\n汇流累积量统计:")
    print(f"  最小值: {min_acc}")
    print(f"  最大值: {max_acc}")
    print(f"  平均值: {mean_acc:.2f}")
    
    # 检查汇水点处的汇流累积量
    print("\n汇水点处的汇流累积量:")
    for point_id, (row, col) in pour_points:
        acc_value = accumulation.get((row, col), 0)
        print(f"  {point_id} ({row}, {col}): {acc_value}")
        
        # 检查周围单元格的汇流累积量
        print(f"    周围单元格汇流累积量:")
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < dem.shape[0] and 0 <= nc < dem.shape[1]:
                    acc = accumulation.get((nr, nc), 0)
                    print(f"      ({nr}, {nc}): {acc}")
        print()
    
    return accumulation

def main():
    # 加载DEM
    dem_path = REPO_ROOT / "data/sample/dem/realistic_watershed_dem.json"
    dem = load_dem(dem_path)
    
    # 分析DEM数据
    river_candidates = analyze_dem(dem)
    
    # 加载汇水点
    pour_points_path = REPO_ROOT / "data/sample/gis/realistic_watershed_pour_points.geojson"
    pour_points = load_pour_points(pour_points_path, dem.transform)
    
    # 分析汇流累积量
    accumulation = analyze_flow_accumulation(dem, pour_points)
    
    # 检查汇水点是否在可能的河道上
    print("\n检查汇水点是否在可能的河道上:")
    pour_point_set = set((row, col) for _, (row, col) in pour_points)
    river_set = set((row, col) for row, col, _ in river_candidates)
    
    for point_id, (row, col) in pour_points:
        if (row, col) in river_set:
            print(f"  {point_id}: 在可能的河道上")
        else:
            print(f"  {point_id}: 不在可能的河道上")
            
            # 找到最近的河道点
            min_dist = float('inf')
            nearest_river = None
            for rrow, rcol, _ in river_candidates:
                dist = math.sqrt((row - rrow)**2 + (col - rcol)**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_river = (rrow, rcol)
            
            if nearest_river:
                print(f"    最近的河道点: ({nearest_river[0]}, {nearest_river[1]}), 距离: {min_dist}")

if __name__ == "__main__":
    main()
