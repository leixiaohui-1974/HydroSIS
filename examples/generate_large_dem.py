#!/usr/bin/env python3
"""生成大型DEM网格数据，用于测试流域划分"""

import json
import math
import random
from pathlib import Path

def generate_dem_grid(width=1000, height=1000):
    """生成一个模拟流域地形的DEM网格"""
    grid = []
    
    # 创建一个从西北到东南倾斜的基础地形
    for row in range(height):
        row_data = []
        for col in range(width):
            # 基础高程：从西北到东南递减
            base_elevation = 150.0 * (1 - row / height) * (1 - col / width)
            
            # 添加一些随机噪声
            noise = random.uniform(-5, 5)
            
            # 添加一条从西北到东南的主河道
            # 河道中心线
            river_center_row = height * 0.2 + row * 0.6  # 从20%高度开始，到80%高度结束
            river_center_col = width * 0.2 + col * 0.6   # 从20%宽度开始，到80%宽度结束
            
            # 计算到河道中心线的距离
            dist_to_river = math.sqrt((row - river_center_row)**2 + (col - river_center_col)**2)
            
            # 河道影响：距离河道越近，高程越低
            river_effect = max(0, 20 - dist_to_river * 0.5)
            
            # 添加一些支流
            tributary_effect = 0
            # 第一条支流
            trib1_center_row = height * 0.3
            trib1_center_col = width * 0.5 + (row - height * 0.3) * 0.3
            dist_to_trib1 = math.sqrt((row - trib1_center_row)**2 + (col - trib1_center_col)**2)
            tributary_effect += max(0, 10 - dist_to_trib1 * 0.3)
            
            # 第二条支流
            trib2_center_row = height * 0.6
            trib2_center_col = width * 0.3 + (row - height * 0.6) * 0.4
            dist_to_trib2 = math.sqrt((row - trib2_center_row)**2 + (col - trib2_center_col)**2)
            tributary_effect += max(0, 10 - dist_to_trib2 * 0.3)
            
            # 计算最终高程
            elevation = base_elevation - river_effect - tributary_effect + noise
            elevation = max(0, elevation)  # 确保高程非负
            
            row_data.append(round(elevation, 1))
        
        grid.append(row_data)
    
    return grid

def save_dem_file(grid, output_path):
    """保存DEM数据到JSON文件"""
    # 设置变换参数
    transform = {
        "x_origin": 500000.0,
        "y_origin": 3500000.0,
        "pixel_width": 10.0,
        "pixel_height": 10.0
    }
    
    # 创建DEM数据结构
    dem_data = {
        "transform": transform,
        "crs": "EPSG:3857",
        "grid": grid
    }
    
    # 保存到文件
    with open(output_path, 'w') as f:
        json.dump(dem_data, f, indent=2)
    
    print(f"DEM数据已保存到: {output_path}")
    print(f"网格大小: {len(grid)} x {len(grid[0])}")

def generate_pour_points(dem_width, dem_height, output_path):
    """生成汇水点数据"""
    # 定义汇水点位置（在河道上）
    # 确保坐标在DEM网格范围内
    pour_points = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [500500.0, 3499500.0]},
            "properties": {"id": "UP1", "name": "上游控制点1"}
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [501000.0, 3499500.0]},
            "properties": {"id": "UP2", "name": "上游控制点2"}
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [501500.0, 3499000.0]},
            "properties": {"id": "MID1", "name": "中游控制点1"}
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [502000.0, 3498000.0]},
            "properties": {"id": "MID2", "name": "中游控制点2"}
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [502500.0, 3497000.0]},
            "properties": {"id": "DOWN1", "name": "下游控制点1"}
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [503000.0, 3496000.0]},
            "properties": {"id": "OUTLET", "name": "流域出口"}
        }
    ]
    
    # 创建GeoJSON结构
    geojson_data = {
        "type": "FeatureCollection",
        "features": pour_points
    }
    
    # 保存到文件
    with open(output_path, 'w') as f:
        json.dump(geojson_data, f, indent=2)
    
    print(f"汇水点数据已保存到: {output_path}")

def main():
    """主函数"""
    # 设置输出路径
    repo_root = Path(__file__).resolve().parents[1]
    dem_output_path = repo_root / "data/sample/dem/large_watershed_dem.json"
    pour_points_output_path = repo_root / "data/sample/gis/large_watershed_pour_points.geojson"
    
    # 生成DEM网格
    print("正在生成DEM网格...")
    grid = generate_dem_grid(1000, 1000)
    
    # 保存DEM文件
    save_dem_file(grid, dem_output_path)
    
    # 生成汇水点
    print("\n正在生成汇水点...")
    generate_pour_points(1000, 1000, pour_points_output_path)
    
    print("\n完成!")

if __name__ == "__main__":
    main()
