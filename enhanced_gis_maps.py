#!/usr/bin/env python3
"""增强版GIS地图生成器 - 基于真实数据生成子流域和参数分区图"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

def load_real_gis_data():
    """加载真实的GIS数据"""
    # 从GIS报告中提取的真实坐标数据
    subbasins_data = {
        "SB1": {
            "area_km2": 120.5,
            "downstream": "SB3", 
            "zone": "Z1",
            "coordinates": [[500000.0, 3499000.0], [502000.0, 3499000.0], 
                          [502000.0, 3497000.0], [500000.0, 3497000.0]],
            "color": "#1f77b4",
            "description": "上游山区集水区"
        },
        "SB2": {
            "area_km2": 80.2,
            "downstream": "SB3",
            "zone": "Z2", 
            "coordinates": [[504000.0, 3499000.0], [506000.0, 3499000.0],
                          [506000.0, 3497000.0], [504000.0, 3497000.0]],
            "color": "#ff7f0e",
            "description": "中游支流集水区"
        },
        "SB3": {
            "area_km2": 210.7,
            "downstream": "SB4",
            "zone": "Z2",
            "coordinates": [[501000.0, 3497000.0], [506000.0, 3497000.0],
                          [506000.0, 3495000.0], [501000.0, 3495000.0]],
            "color": "#2ca02c", 
            "description": "中游汇流区"
        },
        "SB4": {
            "area_km2": 540.0,
            "downstream": None,
            "zone": "Z3",
            "coordinates": [[500000.0, 3495000.0], [506000.0, 3495000.0],
                          [506000.0, 3493000.0], [500000.0, 3493000.0]],
            "color": "#d62728",
            "description": "下游出口区"
        }
    }
    
    zones_data = {
        "Z1": {
            "description": "上游山区集水区 (G1控制)",
            "control_points": ["SB1"],
            "subbasins": ["SB1"],
            "color": "#9467bd",
            "runoff_model": "upland",
            "routing_model": "muskingum_main"
        },
        "Z2": {
            "description": "中游水库控制区 (R1控制)",
            "control_points": ["SB3"],
            "subbasins": ["SB2", "SB3"],
            "color": "#8c564b",
            "runoff_model": "midland", 
            "routing_model": "muskingum_main"
        },
        "Z3": {
            "description": "下游水文站控制区 (G2控制)",
            "control_points": ["SB4"],
            "subbasins": ["SB4"],
            "color": "#e377c2",
            "runoff_model": "floodplain",
            "routing_model": "muskingum_outlet"
        }
    }
    
    # 控制点位置
    control_points = {
        "G1": {"coords": (501000, 3498000), "type": "gauge", "zone": "Z1"},
        "R1": {"coords": (503500, 3496000), "type": "reservoir", "zone": "Z2"},
        "G2": {"coords": (503000, 3494000), "type": "gauge", "zone": "Z3"}
    }
    
    # 河网数据
    river_network = [
        {"name": "主河道", "coords": [[500500.0, 3499500.0], [502500.0, 3497500.0], [502500.0, 3494500.0]]},
        {"name": "支流1", "coords": [[505000.0, 3498500.0], [503000.0, 3497000.0]]},
        {"name": "支流2", "coords": [[501500.0, 3498500.0], [502000.0, 3497500.0]]}
    ]
    
    return subbasins_data, zones_data, control_points, river_network

def create_professional_subbasin_map(output_path: Path) -> None:
    """创建专业的子流域划分地图"""
    subbasins_data, _, control_points, river_network = load_real_gis_data()
    
    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor('white')
    
    # 绘制子流域多边形
    for sub_id, data in subbasins_data.items():
        coords = data["coordinates"]
        polygon = patches.Polygon(coords, facecolor=data["color"], 
                                edgecolor='black', alpha=0.7, linewidth=2)
        ax.add_patch(polygon)
        
        # 计算中心点
        center_x = sum(x for x, y in coords) / len(coords)
        center_y = sum(y for x, y in coords) / len(coords)
        
        # 添加子流域标签
        bbox_props = dict(boxstyle="round,pad=0.4", facecolor='white', 
                         edgecolor='black', alpha=0.9, linewidth=1)
        ax.text(center_x, center_y, f"{sub_id}\n{data['area_km2']:.1f} km²\n{data['description']}", 
               ha='center', va='center', fontsize=11, fontweight='bold',
               bbox=bbox_props)
    
    # 绘制河网
    for river in river_network:
        coords = river["coords"]
        x_coords = [coord[0] for coord in coords]
        y_coords = [coord[1] for coord in coords]
        ax.plot(x_coords, y_coords, 'b-', linewidth=3, alpha=0.8, label='河网' if river == river_network[0] else "")
    
    # 绘制流向箭头
    flow_arrows = [
        ((501000, 3498000), (502500, 3496500)),  # SB1 -> SB3
        ((505000, 3498000), (503500, 3496500)),  # SB2 -> SB3
        ((503000, 3496000), (503000, 3494500)),  # SB3 -> SB4
    ]
    
    for start, end in flow_arrows:
        dx, dy = end[0] - start[0], end[1] - start[1]
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='darkblue', alpha=0.8))
    
    # 添加控制点
    for point_id, info in control_points.items():
        x, y = info["coords"]
        marker = 's' if info["type"] == "gauge" else '^'
        color = 'red' if info["type"] == "gauge" else 'orange'
        ax.scatter(x, y, s=300, c=color, marker=marker, edgecolor='black', 
                  linewidth=2, zorder=10, alpha=0.9)
        ax.text(x+300, y+300, point_id, fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.8))
    
    # 设置坐标轴和标题
    ax.set_xlim(499000, 507000)
    ax.set_ylim(3492000, 3500500)
    ax.set_xlabel('东向坐标 (m)', fontsize=14, fontweight='bold')
    ax.set_ylabel('北向坐标 (m)', fontsize=14, fontweight='bold')
    ax.set_title('HydroSIS 子流域划分图\n基于DEM自动划分结果', fontsize=18, fontweight='bold', pad=25)
    
    # 添加网格
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.set_aspect('equal')
    
    # 创建图例
    legend_elements = []
    for sub_id, data in subbasins_data.items():
        legend_elements.append(patches.Patch(facecolor=data["color"], 
                                           label=f"{sub_id} ({data['area_km2']:.1f} km²)"))
    
    legend_elements.extend([
        plt.Line2D([0], [0], color='blue', linewidth=3, label='河网'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
                  markersize=12, label='水文站', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='orange',
                  markersize=12, label='水库', markeredgecolor='black')
    ])
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
             fontsize=11, frameon=True, fancybox=True, shadow=True)
    
    # 添加指北针
    ax.annotate('N', xy=(0.95, 0.95), xycoords='axes fraction', fontsize=16, 
               fontweight='bold', ha='center', va='center',
               bbox=dict(boxstyle="circle,pad=0.3", facecolor='lightblue', alpha=0.8))
    ax.annotate('↑', xy=(0.95, 0.92), xycoords='axes fraction', fontsize=20, 
               ha='center', va='center')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"专业子流域划分图已保存到: {output_path}")

def create_professional_zone_map(output_path: Path) -> None:
    """创建专业的参数分区划分地图"""
    subbasins_data, zones_data, control_points, river_network = load_real_gis_data()
    
    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor('white')
    
    # 为每个参数区绘制合并的多边形
    for zone_id, zone_info in zones_data.items():
        zone_coords = []
        for sub_id in zone_info["subbasins"]:
            zone_coords.extend(subbasins_data[sub_id]["coordinates"])
        
        if zone_coords:
            # 计算包围盒
            min_x = min(coord[0] for coord in zone_coords)
            max_x = max(coord[0] for coord in zone_coords)
            min_y = min(coord[1] for coord in zone_coords)
            max_y = max(coord[1] for coord in zone_coords)
            
            # 创建参数区多边形
            zone_polygon = patches.Rectangle((min_x, min_y), max_x-min_x, max_y-min_y,
                                           facecolor=zone_info["color"], alpha=0.5,
                                           edgecolor='black', linewidth=3)
            ax.add_patch(zone_polygon)
            
            # 添加参数区标签
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            
            label_text = f"{zone_id}\n{zone_info['description']}\n"
            label_text += f"产流模型: {zone_info['runoff_model']}\n"
            label_text += f"汇流模型: {zone_info['routing_model']}\n"
            label_text += f"子流域: {', '.join(zone_info['subbasins'])}"
            
            bbox_props = dict(boxstyle="round,pad=0.5", facecolor='white', 
                             edgecolor=zone_info["color"], alpha=0.95, linewidth=2)
            ax.text(center_x, center_y, label_text, ha='center', va='center', 
                   fontsize=10, fontweight='bold', bbox=bbox_props)
    
    # 绘制子流域边界（细线）
    for sub_id, data in subbasins_data.items():
        coords = data["coordinates"]
        polygon = patches.Polygon(coords, facecolor='none', 
                                edgecolor='gray', alpha=0.7, linewidth=1, linestyle='--')
        ax.add_patch(polygon)
        
        # 添加子流域ID
        center_x = sum(x for x, y in coords) / len(coords)
        center_y = sum(y for x, y in coords) / len(coords)
        ax.text(center_x, center_y-200, sub_id, ha='center', va='center', 
               fontsize=9, style='italic', color='gray')
    
    # 绘制河网
    for river in river_network:
        coords = river["coords"]
        x_coords = [coord[0] for coord in coords]
        y_coords = [coord[1] for coord in coords]
        ax.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.6)
    
    # 绘制控制点
    for point_id, info in control_points.items():
        x, y = info["coords"]
        marker = 's' if info["type"] == "gauge" else '^'
        color = 'red' if info["type"] == "gauge" else 'orange'
        ax.scatter(x, y, s=400, c=color, marker=marker, edgecolor='black', 
                  linewidth=3, zorder=15, alpha=1.0)
        
        # 添加控制点标签和连线
        zone_color = zones_data[info["zone"]]["color"]
        ax.text(x+400, y+400, f"{point_id}\n({info['zone']}控制点)", 
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor=zone_color, alpha=0.8))
    
    # 设置坐标轴和标题
    ax.set_xlim(499000, 507000)
    ax.set_ylim(3492000, 3500500)
    ax.set_xlabel('东向坐标 (m)', fontsize=14, fontweight='bold')
    ax.set_ylabel('北向坐标 (m)', fontsize=14, fontweight='bold')
    ax.set_title('HydroSIS 参数分区划分图\n基于控制点的参数区自动分配', fontsize=18, fontweight='bold', pad=25)
    
    # 添加网格
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.set_aspect('equal')
    
    # 创建图例
    legend_elements = []
    for zone_id, zone_info in zones_data.items():
        legend_elements.append(patches.Patch(facecolor=zone_info["color"], alpha=0.7,
                                           label=f"{zone_id}: {zone_info['description']}"))
    
    legend_elements.extend([
        plt.Line2D([0], [0], color='gray', linewidth=1, linestyle='--', label='子流域边界'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
                  markersize=12, label='水文站', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='orange',
                  markersize=12, label='水库', markeredgecolor='black')
    ])
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
             fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    # 添加指北针
    ax.annotate('N', xy=(0.95, 0.95), xycoords='axes fraction', fontsize=16, 
               fontweight='bold', ha='center', va='center',
               bbox=dict(boxstyle="circle,pad=0.3", facecolor='lightblue', alpha=0.8))
    ax.annotate('↑', xy=(0.95, 0.92), xycoords='axes fraction', fontsize=20, 
               ha='center', va='center')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"专业参数分区划分图已保存到: {output_path}")

def create_comparison_dashboard(output_path: Path) -> None:
    """创建对比仪表板"""
    subbasins_data, zones_data, control_points, river_network = load_real_gis_data()
    
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor('white')
    
    # 创建网格布局
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], width_ratios=[1, 1, 1])
    
    # 子流域图
    ax1 = fig.add_subplot(gs[0, 0])
    for sub_id, data in subbasins_data.items():
        coords = data["coordinates"]
        polygon = patches.Polygon(coords, facecolor=data["color"], 
                                edgecolor='black', alpha=0.7, linewidth=2)
        ax1.add_patch(polygon)
        
        center_x = sum(x for x, y in coords) / len(coords)
        center_y = sum(y for x, y in coords) / len(coords)
        ax1.text(center_x, center_y, f"{sub_id}\n{data['area_km2']:.1f}km²", 
               ha='center', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
    
    ax1.set_title('子流域划分', fontsize=14, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # 参数分区图
    ax2 = fig.add_subplot(gs[0, 1])
    for zone_id, zone_info in zones_data.items():
        zone_coords = []
        for sub_id in zone_info["subbasins"]:
            zone_coords.extend(subbasins_data[sub_id]["coordinates"])
        
        if zone_coords:
            min_x = min(coord[0] for coord in zone_coords)
            max_x = max(coord[0] for coord in zone_coords)
            min_y = min(coord[1] for coord in zone_coords)
            max_y = max(coord[1] for coord in zone_coords)
            
            zone_polygon = patches.Rectangle((min_x, min_y), max_x-min_x, max_y-min_y,
                                           facecolor=zone_info["color"], alpha=0.6,
                                           edgecolor='black', linewidth=2)
            ax2.add_patch(zone_polygon)
            
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            ax2.text(center_x, center_y, f"{zone_id}\n{', '.join(zone_info['subbasins'])}", 
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
    
    ax2.set_title('参数分区划分', fontsize=14, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # 统计图表
    ax3 = fig.add_subplot(gs[0, 2])
    
    # 面积统计
    areas = [data["area_km2"] for data in subbasins_data.values()]
    labels = list(subbasins_data.keys())
    colors = [data["color"] for data in subbasins_data.values()]
    
    wedges, texts, autotexts = ax3.pie(areas, labels=labels, colors=colors, autopct='%1.1f%%',
                                      startangle=90, textprops={'fontsize': 10})
    ax3.set_title('子流域面积分布', fontsize=14, fontweight='bold')
    
    # 设置所有子图的坐标范围
    for ax in [ax1, ax2]:
        ax.set_xlim(499000, 507000)
        ax.set_ylim(3492000, 3500500)
    
    # 添加统计表格
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')
    
    # 创建统计表格数据
    table_data = []
    table_data.append(['子流域', '面积(km²)', '下游', '所属参数区', '产流模型', '汇流模型'])
    
    for sub_id, data in subbasins_data.items():
        zone_id = data["zone"]
        zone_info = zones_data[zone_id]
        table_data.append([
            sub_id,
            f"{data['area_km2']:.1f}",
            data["downstream"] or "出口",
            zone_id,
            zone_info["runoff_model"],
            zone_info["routing_model"]
        ])
    
    # 绘制表格
    table = ax4.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 设置表格样式
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.suptitle('HydroSIS 流域划分与参数分区对比分析', fontsize=20, fontweight='bold', y=0.95)
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"对比仪表板已保存到: {output_path}")

def main():
    """主函数"""
    base_dir = Path("results/example_run/figures")
    
    print("正在生成增强版GIS地图...")
    
    # 生成专业地图
    create_professional_subbasin_map(base_dir / "professional_subbasin_map.png")
    create_professional_zone_map(base_dir / "professional_zone_map.png") 
    create_comparison_dashboard(base_dir / "gis_comparison_dashboard.png")
    
    print("\n✅ 所有增强版GIS地图已生成完成！")
    print(f"📁 输出目录: {base_dir.absolute()}")
    print("\n生成的文件:")
    print("  - professional_subbasin_map.png (专业子流域划分图)")
    print("  - professional_zone_map.png (专业参数分区划分图)")
    print("  - gis_comparison_dashboard.png (对比仪表板)")

if __name__ == "__main__":
    main()