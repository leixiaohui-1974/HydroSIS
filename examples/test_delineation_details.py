#!/usr/bin/env python3
"""详细测试流域划分过程"""

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hydrosis.delineation.simple_grid import (
    load_dem, 
    load_pour_points, 
    compute_flow_directions,
    compute_flow_accumulation,
    delineate_watersheds
)

def main():
    # 加载DEM
    dem_path = REPO_ROOT / "data/sample/dem/realistic_watershed_dem.json"
    dem = load_dem(dem_path)
    
    print(f"DEM网格大小: {dem.shape}")
    print(f"单元格面积: {dem.transform.pixel_width * dem.transform.pixel_height / 1000000} km2")
    
    # 加载汇水点
    pour_points_path = REPO_ROOT / "data/sample/gis/realistic_watershed_pour_points.geojson"
    pour_points = load_pour_points(pour_points_path, dem.transform)
    
    print("\n汇水点:")
    for point_id, (row, col) in pour_points:
        print(f"  {point_id}: 行={row}, 列={col}")
    
    # 计算流向
    print("\n计算流向...")
    flow = compute_flow_directions(dem)
    
    # 计算汇流累积量
    print("计算汇流累积量...")
    accumulation = compute_flow_accumulation(dem, flow)
    
    # 检查汇水点处的汇流累积量
    print("\n汇水点处的汇流累积量:")
    for point_id, (row, col) in pour_points:
        acc_value = accumulation.get((row, col), 0)
        print(f"  {point_id}: {acc_value}")
    
    # 划分流域
    print("\n划分流域...")
    watersheds = delineate_watersheds(flow, pour_points)
    
    print("\n流域划分结果:")
    for basin_id, cells in watersheds.items():
        print(f"  {basin_id}: {len(cells)} 个单元格")
        if len(cells) > 0:
            area_km2 = len(cells) * dem.transform.pixel_width * dem.transform.pixel_height / 1000000
            print(f"    面积: {area_km2} km2")
            
            # 显示前几个单元格
            print(f"    前5个单元格: {cells[:5]}")
    
    # 应用累积阈值过滤
    print("\n应用累积阈值过滤 (阈值=5)...")
    filtered = {}
    for basin_id, cells in watersheds.items():
        filtered_cells = [
            cell
            for cell in cells
            if accumulation.get(cell, 0) >= 5
        ]
        filtered[basin_id] = filtered_cells
        print(f"  {basin_id}: {len(filtered_cells)} 个单元格 (过滤前: {len(cells)})")
        if len(filtered_cells) > 0:
            area_km2 = len(filtered_cells) * dem.transform.pixel_width * dem.transform.pixel_height / 1000000
            print(f"    面积: {area_km2} km2")
    
    # 尝试更低的阈值
    print("\n应用累积阈值过滤 (阈值=1)...")
    filtered = {}
    for basin_id, cells in watersheds.items():
        filtered_cells = [
            cell
            for cell in cells
            if accumulation.get(cell, 0) >= 1
        ]
        filtered[basin_id] = filtered_cells
        print(f"  {basin_id}: {len(filtered_cells)} 个单元格 (过滤前: {len(cells)})")
        if len(filtered_cells) > 0:
            area_km2 = len(filtered_cells) * dem.transform.pixel_width * dem.transform.pixel_height / 1000000
            print(f"    面积: {area_km2} km2")
    
    # 不应用阈值
    print("\n不应用累积阈值过滤...")
    for basin_id, cells in watersheds.items():
        print(f"  {basin_id}: {len(cells)} 个单元格")
        if len(cells) > 0:
            area_km2 = len(cells) * dem.transform.pixel_width * dem.transform.pixel_height / 1000000
            print(f"    面积: {area_km2} km2")

if __name__ == "__main__":
    main()
