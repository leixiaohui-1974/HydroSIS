#!/usr/bin/env python3
"""测试汇水点坐标转换"""

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hydrosis.delineation.simple_grid import load_dem, load_pour_points

def main():
    # 加载DEM
    dem_path = REPO_ROOT / "data/sample/dem/large_watershed_dem.json"
    dem = load_dem(dem_path)
    
    print(f"DEM网格大小: {dem.shape}")
    print(f"DEM变换参数:")
    print(f"  x_origin: {dem.transform.x_origin}")
    print(f"  y_origin: {dem.transform.y_origin}")
    print(f"  pixel_width: {dem.transform.pixel_width}")
    print(f"  pixel_height: {dem.transform.pixel_height}")
    
    # 加载汇水点
    pour_points_path = REPO_ROOT / "data/sample/gis/large_watershed_pour_points.geojson"
    pour_points = load_pour_points(pour_points_path, dem.transform)
    
    print("\n汇水点坐标转换:")
    for point_id, (row, col) in pour_points:
        print(f"  {point_id}: 行={row}, 列={col}")
        
        # 检查是否在DEM网格范围内
        if 0 <= row < dem.shape[0] and 0 <= col < dem.shape[1]:
            elevation = dem.elevations[row][col]
            print(f"    高程: {elevation}")
        else:
            print(f"    错误: 超出DEM网格范围!")
    
    # 测试一些坐标转换
    print("\n坐标转换测试:")
    test_coords = [
        (500500.0, 3499500.0),
        (501000.0, 3499500.0),
        (501500.0, 3499000.0),
        (502000.0, 3498000.0),
        (502500.0, 3497000.0),
        (503000.0, 3496000.0)
    ]
    
    for x, y in test_coords:
        row, col = dem.transform.to_grid_location(x, y)
        print(f"  坐标({x}, {y}) -> 行={row}, 列={col}")
        
        # 检查是否在DEM网格范围内
        if 0 <= row < dem.shape[0] and 0 <= col < dem.shape[1]:
            elevation = dem.elevations[row][col]
            print(f"    高程: {elevation}")
        else:
            print(f"    错误: 超出DEM网格范围!")

if __name__ == "__main__":
    main()
