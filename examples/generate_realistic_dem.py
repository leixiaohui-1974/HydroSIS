#!/usr/bin/env python3
"""生成更真实的DEM网格数据，用于测试流域划分"""

import json
import math
import random
from pathlib import Path

def generate_realistic_dem_grid(width=1000, height=1000):
    """生成一个更真实的流域地形DEM网格，大部分网格都汇流到汇水点"""
    grid = []
    
    # 创建一个从西北到东南倾斜的基础地形，高程差更大
    base_elevation_max = 1000.0  # 最大高程1000米
    base_elevation_min = 100.0   # 最小高程100米
    
    # 主河道从西北到东南，更明显
    main_river_start_row = int(height * 0.05)
    main_river_start_col = int(width * 0.05)
    main_river_end_row = int(height * 0.95)
    main_river_end_col = int(width * 0.95)
    
    # 支流位置
    tributary1_start = (int(height * 0.1), int(width * 0.9))
    tributary1_join = (int(height * 0.4), int(width * 0.6))
    
    tributary2_start = (int(height * 0.2), int(width * 0.1))
    tributary2_join = (int(height * 0.5), int(width * 0.4))
    
    tributary3_start = (int(height * 0.3), int(width * 0.8))
    tributary3_join = (int(height * 0.6), int(width * 0.7))
    
    # 创建一个汇流方向矩阵，确保大部分网格都流向河道
    flow_directions = {}
    
    for row in range(height):
        row_data = []
        for col in range(width):
            # 基础高程：从西北到东南递减，高程差更大
            progress_row = row / height
            progress_col = col / width
            base_elevation = base_elevation_max * (1 - progress_row * 0.9) * (1 - progress_col * 0.9) + base_elevation_min
            
            # 添加一些随机噪声
            noise = random.uniform(-10, 10)
            
            # 计算到主河道的距离和影响
            # 主河道中心线（使用线性插值）
            t = max(0, min(1, (row - main_river_start_row) / (main_river_end_row - main_river_start_row) if main_river_end_row != main_river_start_row else 0))
            river_center_row = main_river_start_row + t * (main_river_end_row - main_river_start_row)
            river_center_col = main_river_start_col + t * (main_river_end_col - main_river_start_col)
            
            dist_to_main_river = math.sqrt((row - river_center_row)**2 + (col - river_center_col)**2)
            
            # 主河道影响：距离河道越近，高程越低，影响范围更大
            river_width = 100.0  # 河道宽度增加
            river_depth = 300.0  # 河道深度增加
            river_effect = max(0, river_depth * (1 - dist_to_main_river / river_width))
            
            # 计算到支流1的距离和影响
            t1 = max(0, min(1, (row - tributary1_start[0]) / (tributary1_join[0] - tributary1_start[0]) if tributary1_join[0] != tributary1_start[0] else 0))
            trib1_center_row = tributary1_start[0] + t1 * (tributary1_join[0] - tributary1_start[0])
            trib1_center_col = tributary1_start[1] + t1 * (tributary1_join[1] - tributary1_start[1])
            
            dist_to_trib1 = math.sqrt((row - trib1_center_row)**2 + (col - trib1_center_col)**2)
            trib1_width = 50.0
            trib1_depth = 150.0
            trib1_effect = max(0, trib1_depth * (1 - dist_to_trib1 / trib1_width))
            
            # 计算到支流2的距离和影响
            t2 = max(0, min(1, (row - tributary2_start[0]) / (tributary2_join[0] - tributary2_start[0]) if tributary2_join[0] != tributary2_start[0] else 0))
            trib2_center_row = tributary2_start[0] + t2 * (tributary2_join[0] - tributary2_start[0])
            trib2_center_col = tributary2_start[1] + t2 * (tributary2_join[1] - tributary2_start[1])
            
            dist_to_trib2 = math.sqrt((row - trib2_center_row)**2 + (col - trib2_center_col)**2)
            trib2_width = 50.0
            trib2_depth = 150.0
            trib2_effect = max(0, trib2_depth * (1 - dist_to_trib2 / trib2_width))
            
            # 计算到支流3的距离和影响
            t3 = max(0, min(1, (row - tributary3_start[0]) / (tributary3_join[0] - tributary3_start[0]) if tributary3_join[0] != tributary3_start[0] else 0))
            trib3_center_row = tributary3_start[0] + t3 * (tributary3_join[0] - tributary3_start[0])
            trib3_center_col = tributary3_start[1] + t3 * (tributary3_join[1] - tributary3_start[1])
            
            dist_to_trib3 = math.sqrt((row - trib3_center_row)**2 + (col - trib3_center_col)**2)
            trib3_width = 50.0
            trib3_depth = 150.0
            trib3_effect = max(0, trib3_depth * (1 - dist_to_trib3 / trib3_width))
            
            # 添加一个整体的汇流趋势，确保大部分网格都向东南方向倾斜
            flow_trend = (row / height + col / width) * 50  # 向东南方向倾斜的趋势
            
            # 计算最终高程
            elevation = base_elevation - river_effect - trib1_effect - trib2_effect - trib3_effect - flow_trend + noise
            elevation = max(0, elevation)  # 确保高程非负
            
            row_data.append(round(elevation, 1))
        
        grid.append(row_data)
    
    # 后处理：轻微调整高程，确保大部分网格向东南方向汇流，但保留河道特征
    for row in range(1, height-1):
        for col in range(1, width-1):
            # 只对远离河道的区域进行轻微调整
            dist_to_main_river = math.sqrt((row - river_center_row)**2 + (col - river_center_col)**2)
            
            if dist_to_main_river > river_width * 2:  # 只对远离河道的区域进行调整
                # 轻微降低高程，形成向东南方向的汇流趋势
                adjustment = (row / height + col / width) * 2  # 减小调整幅度
                grid[row][col] = max(0, grid[row][col] - adjustment)
    
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

def generate_realistic_pour_points(dem_width, dem_height, output_path):
    """生成更合理的汇水点数据，位于河道上"""
    # 汇水点位置（在主河道和支流上）
    # 上游控制点1 - 主河道上游
    up1_x = 500000 + dem_width * 10 * 0.15
    up1_y = 3500000 - dem_height * 10 * 0.15
    
    # 上游控制点2 - 主河道中上游
    up2_x = 500000 + dem_width * 10 * 0.25
    up2_y = 3500000 - dem_height * 10 * 0.25
    
    # 中游控制点1 - 支流1汇入点
    mid1_x = 500000 + dem_width * 10 * 0.5
    mid1_y = 3500000 - dem_height * 10 * 0.5
    
    # 中游控制点2 - 支流2汇入点
    mid2_x = 500000 + dem_width * 10 * 0.4
    mid2_y = 3500000 - dem_height * 10 * 0.6
    
    # 下游控制点1 - 主河道下游
    down1_x = 500000 + dem_width * 10 * 0.7
    down1_y = 3500000 - dem_height * 10 * 0.7
    
    # 流域出口 - 主河道出口
    outlet_x = 500000 + dem_width * 10 * 0.85
    outlet_y = 3500000 - dem_height * 10 * 0.85
    
    pour_points = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [up1_x, up1_y]},
            "properties": {"id": "UP1", "name": "上游控制点1"}
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [up2_x, up2_y]},
            "properties": {"id": "UP2", "name": "上游控制点2"}
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [mid1_x, mid1_y]},
            "properties": {"id": "MID1", "name": "中游控制点1"}
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [mid2_x, mid2_y]},
            "properties": {"id": "MID2", "name": "中游控制点2"}
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [down1_x, down1_y]},
            "properties": {"id": "DOWN1", "name": "下游控制点1"}
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [outlet_x, outlet_y]},
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
    dem_output_path = repo_root / "data/sample/dem/realistic_watershed_dem.json"
    pour_points_output_path = repo_root / "data/sample/gis/realistic_watershed_pour_points.geojson"
    
    # 生成DEM网格
    print("正在生成真实DEM网格...")
    grid = generate_realistic_dem_grid(1000, 1000)
    
    # 保存DEM文件
    save_dem_file(grid, dem_output_path)
    
    # 生成汇水点
    print("\n正在生成汇水点...")
    generate_realistic_pour_points(1000, 1000, pour_points_output_path)
    
    print("\n完成!")

if __name__ == "__main__":
    main()
