#!/usr/bin/env python3
"""生成子流域和参数分区的静态GIS图片"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_geojson_data(file_path: Path) -> Dict[str, Any]:
    """加载GeoJSON数据"""
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"type": "FeatureCollection", "features": []}

def extract_polygon_bounds(geojson_data: Dict[str, Any]) -> List[Tuple[str, List[Tuple[float, float]]]]:
    """从GeoJSON中提取多边形边界"""
    polygons = []
    
    for feature in geojson_data.get("features", []):
        feature_id = feature.get("properties", {}).get("id", "Unknown")
        geometry = feature.get("geometry", {})
        
        if geometry.get("type") == "MultiPolygon":
            coords = geometry.get("coordinates", [])
            if coords:
                # 取第一个多边形的外环
                outer_ring = coords[0][0] if coords[0] else []
                polygons.append((feature_id, outer_ring))
        elif geometry.get("type") == "Polygon":
            coords = geometry.get("coordinates", [])
            if coords:
                # 取外环
                outer_ring = coords[0]
                polygons.append((feature_id, outer_ring))
    
    return polygons

def create_subbasin_map(output_path: Path) -> None:
    """创建子流域划分地图"""
    # 模拟子流域数据（基于配置文件中的信息）
    subbasins = {
        "SB1": {"area": 120.5, "downstream": "SB3", "color": "#1f77b4", 
                "bounds": [(500000, 3499000), (502000, 3499000), (502000, 3497000), (500000, 3497000)]},
        "SB2": {"area": 80.2, "downstream": "SB3", "color": "#ff7f0e",
                "bounds": [(504000, 3499000), (506000, 3499000), (506000, 3497000), (504000, 3497000)]},
        "SB3": {"area": 210.7, "downstream": "SB4", "color": "#2ca02c",
                "bounds": [(501000, 3497000), (506000, 3497000), (506000, 3495000), (501000, 3495000)]},
        "SB4": {"area": 540.0, "downstream": None, "color": "#d62728",
                "bounds": [(500000, 3495000), (506000, 3495000), (506000, 3493000), (500000, 3493000)]}
    }
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制子流域
    for sub_id, data in subbasins.items():
        bounds = data["bounds"]
        polygon = patches.Polygon(bounds, facecolor=data["color"], 
                                edgecolor='black', alpha=0.7, linewidth=1.5)
        ax.add_patch(polygon)
        
        # 添加标签
        center_x = sum(x for x, y in bounds) / len(bounds)
        center_y = sum(y for x, y in bounds) / len(bounds)
        ax.text(center_x, center_y, f"{sub_id}\n{data['area']:.1f} km²", 
               ha='center', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # 绘制河网连接
    connections = [
        ((501000, 3498000), (503000, 3496000)),  # SB1 -> SB3
        ((505000, 3498000), (503000, 3496000)),  # SB2 -> SB3  
        ((503000, 3496000), (503000, 3494000)),  # SB3 -> SB4
    ]
    
    for start, end in connections:
        ax.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1],
                head_width=500, head_length=300, fc='blue', ec='blue', alpha=0.6)
    
    ax.set_xlim(499000, 507000)
    ax.set_ylim(3492000, 3500000)
    ax.set_xlabel('X坐标 (m)', fontsize=12)
    ax.set_ylabel('Y坐标 (m)', fontsize=12)
    ax.set_title('子流域划分图', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # 添加图例
    legend_elements = [patches.Patch(facecolor=data["color"], label=f"{sub_id} ({data['area']:.1f} km²)")
                      for sub_id, data in subbasins.items()]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"子流域划分图已保存到: {output_path}")

def create_parameter_zone_map(output_path: Path) -> None:
    """创建参数分区划分地图"""
    # 参数分区数据
    zones = {
        "Z1": {"description": "上游山区集水区", "subbasins": ["SB1"], "color": "#9467bd",
               "bounds": [(500000, 3499000), (502000, 3499000), (502000, 3497000), (500000, 3497000)]},
        "Z2": {"description": "中游水库控制区", "subbasins": ["SB2", "SB3"], "color": "#8c564b",
               "bounds": [(501000, 3499000), (506000, 3499000), (506000, 3495000), (501000, 3495000)]},
        "Z3": {"description": "下游出口站控制区", "subbasins": ["SB4"], "color": "#e377c2",
               "bounds": [(500000, 3495000), (506000, 3495000), (506000, 3493000), (500000, 3493000)]}
    }
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制参数分区
    for zone_id, data in zones.items():
        bounds = data["bounds"]
        polygon = patches.Polygon(bounds, facecolor=data["color"], 
                                edgecolor='black', alpha=0.6, linewidth=2)
        ax.add_patch(polygon)
        
        # 添加标签
        center_x = sum(x for x, y in bounds) / len(bounds)
        center_y = sum(y for x, y in bounds) / len(bounds)
        subbasins_text = ", ".join(data["subbasins"])
        ax.text(center_x, center_y, f"{zone_id}\n{subbasins_text}\n{data['description']}", 
               ha='center', va='center', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.9))
    
    # 添加控制点
    control_points = {
        "G1": (501000, 3498000, "Z1"),
        "R1": (503500, 3496000, "Z2"), 
        "G2": (503000, 3494000, "Z3")
    }
    
    for point_id, (x, y, zone) in control_points.items():
        ax.scatter(x, y, s=200, c='red', marker='s', edgecolor='black', linewidth=2, zorder=5)
        ax.text(x+200, y+200, point_id, fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.8))
    
    ax.set_xlim(499000, 507000)
    ax.set_ylim(3492000, 3500000)
    ax.set_xlabel('X坐标 (m)', fontsize=12)
    ax.set_ylabel('Y坐标 (m)', fontsize=12)
    ax.set_title('参数分区划分图', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # 添加图例
    legend_elements = [patches.Patch(facecolor=data["color"], 
                                   label=f"{zone_id}: {data['description']}")
                      for zone_id, data in zones.items()]
    legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                    markerfacecolor='red', markersize=10, 
                                    label='控制点', markeredgecolor='black'))
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.25, 1))
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"参数分区划分图已保存到: {output_path}")

def create_combined_map(output_path: Path) -> None:
    """创建子流域和参数分区的组合图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # 子流域数据
    subbasins = {
        "SB1": {"area": 120.5, "color": "#1f77b4", 
                "bounds": [(500000, 3499000), (502000, 3499000), (502000, 3497000), (500000, 3497000)]},
        "SB2": {"area": 80.2, "color": "#ff7f0e",
                "bounds": [(504000, 3499000), (506000, 3499000), (506000, 3497000), (504000, 3497000)]},
        "SB3": {"area": 210.7, "color": "#2ca02c",
                "bounds": [(501000, 3497000), (506000, 3497000), (506000, 3495000), (501000, 3495000)]},
        "SB4": {"area": 540.0, "color": "#d62728",
                "bounds": [(500000, 3495000), (506000, 3495000), (506000, 3493000), (500000, 3493000)]}
    }
    
    # 参数分区数据
    zones = {
        "Z1": {"description": "上游山区", "color": "#9467bd",
               "bounds": [(500000, 3499000), (502000, 3499000), (502000, 3497000), (500000, 3497000)]},
        "Z2": {"description": "中游水库区", "color": "#8c564b",
               "bounds": [(501000, 3499000), (506000, 3499000), (506000, 3495000), (501000, 3495000)]},
        "Z3": {"description": "下游出口区", "color": "#e377c2",
               "bounds": [(500000, 3495000), (506000, 3495000), (506000, 3493000), (500000, 3493000)]}
    }
    
    # 绘制子流域图
    for sub_id, data in subbasins.items():
        bounds = data["bounds"]
        polygon = patches.Polygon(bounds, facecolor=data["color"], 
                                edgecolor='black', alpha=0.7, linewidth=1.5)
        ax1.add_patch(polygon)
        
        center_x = sum(x for x, y in bounds) / len(bounds)
        center_y = sum(y for x, y in bounds) / len(bounds)
        ax1.text(center_x, center_y, f"{sub_id}\n{data['area']:.1f} km²", 
               ha='center', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # 绘制参数分区图
    for zone_id, data in zones.items():
        bounds = data["bounds"]
        polygon = patches.Polygon(bounds, facecolor=data["color"], 
                                edgecolor='black', alpha=0.6, linewidth=2)
        ax2.add_patch(polygon)
        
        center_x = sum(x for x, y in bounds) / len(bounds)
        center_y = sum(y for x, y in bounds) / len(bounds)
        ax2.text(center_x, center_y, f"{zone_id}\n{data['description']}", 
               ha='center', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
    
    # 设置两个子图的属性
    for ax, title in zip([ax1, ax2], ['子流域划分', '参数分区划分']):
        ax.set_xlim(499000, 507000)
        ax.set_ylim(3492000, 3500000)
        ax.set_xlabel('X坐标 (m)', fontsize=12)
        ax.set_ylabel('Y坐标 (m)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"组合地图已保存到: {output_path}")

def main():
    """主函数"""
    base_dir = Path("results/example_run/figures")
    
    # 生成各种地图
    create_subbasin_map(base_dir / "subbasin_map.png")
    create_parameter_zone_map(base_dir / "parameter_zone_map.png")
    create_combined_map(base_dir / "combined_gis_map.png")
    
    print("\n所有GIS地图已生成完成！")
    print(f"输出目录: {base_dir.absolute()}")

if __name__ == "__main__":
    main()