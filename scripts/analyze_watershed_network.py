#!/usr/bin/env python3
"""分析流域水系网络和汇水点分布"""

import numpy as np
import rasterio
import geopandas as gpd
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def analyze_flow_accumulation(flow_acc_path):
    """分析流量累积"""
    print("\n" + "="*80)
    print("流量累积分析")
    print("="*80)
    
    with rasterio.open(flow_acc_path) as src:
        flow_acc = src.read(1)
        transform = src.transform
        
        # 基本统计
        valid_mask = flow_acc > 0
        valid_data = flow_acc[valid_mask]
        
        print(f"\n基本统计:")
        print(f"  DEM尺寸: {flow_acc.shape}")
        print(f"  有效像元数: {np.sum(valid_mask)}")
        print(f"  像元大小: {transform[0]:.2f}m × {-transform[4]:.2f}m")
        
        print(f"\n流量累积统计:")
        print(f"  最小值: {np.min(valid_data):.0f}")
        print(f"  最大值: {np.max(valid_data):.0f}")  
        print(f"  平均值: {np.mean(valid_data):.2f}")
        print(f"  中位数: {np.median(valid_data):.2f}")
        print(f"  标准差: {np.std(valid_data):.2f}")
        
        # 百分位数
        percentiles = [50, 75, 90, 95, 99, 99.9]
        print(f"\n百分位数分析:")
        for p in percentiles:
            val = np.percentile(valid_data, p)
            print(f"  P{p:5.1f}: {val:10.0f}")
        
        # 水系分级
        thresholds = [100, 500, 1000, 2000, 5000]
        print(f"\n水系分级（累积像元数）:")
        for thresh in thresholds:
            count = np.sum(flow_acc >= thresh)
            length_km = count * transform[0] / 1000  # 假设30m像元
            print(f"  >= {thresh:5d}: {count:6d} 像元 (~{length_km:6.1f} km)")
        
        # 找出主要河道点
        max_val = np.max(flow_acc)
        max_idx = np.unravel_index(np.argmax(flow_acc), flow_acc.shape)
        
        print(f"\n主河道出口:")
        print(f"  位置: 行={max_idx[0]}, 列={max_idx[1]}")
        print(f"  累积值: {max_val:.0f} 像元")
        print(f"  控制面积: {max_val * transform[0] * abs(transform[4]) / 1e6:.2f} km²")
        
        # 计算总流域面积
        total_area_km2 = np.sum(valid_mask) * transform[0] * abs(transform[4]) / 1e6
        print(f"  总流域面积: {total_area_km2:.2f} km²")
        
        return {
            'max_acc': float(max_val),
            'max_idx': max_idx,
            'total_area_km2': float(total_area_km2),
            'transform': transform
        }

def analyze_pour_points(pour_points_path, flow_acc_path):
    """分析汇水点分布"""
    print("\n" + "="*80)
    print("汇水点分布分析")
    print("="*80)
    
    # 读取汇水点
    pour_points = gpd.read_file(pour_points_path)
    
    print(f"\n基本信息:")
    print(f"  汇水点总数: {len(pour_points)}")
    print(f"  坐标系: {pour_points.crs}")
    
    # 分析累积值分布
    if 'accumulation' in pour_points.columns:
        acc_vals = pour_points['accumulation'].values
        
        print(f"\n累积值统计:")
        print(f"  最小值: {np.min(acc_vals):.0f}")
        print(f"  最大值: {np.max(acc_vals):.0f}")
        print(f"  平均值: {np.mean(acc_vals):.2f}")
        print(f"  中位数: {np.median(acc_vals):.2f}")
        
        # 分类
        print(f"\n汇水点分类（按累积值）:")
        print(f"  特大（>5000）: {np.sum(acc_vals > 5000)} 个 - 主干流")
        print(f"  大（2000-5000）: {np.sum((acc_vals >= 2000) & (acc_vals <= 5000))} 个 - 大支流")
        print(f"  中（1000-2000）: {np.sum((acc_vals >= 1000) & (acc_vals < 2000))} 个 - 中支流")
        print(f"  小（<1000）: {np.sum(acc_vals < 1000)} 个 - 小支流")
        
        # 控制面积估算
        with rasterio.open(flow_acc_path) as src:
            transform = src.transform
            pixel_area_km2 = transform[0] * abs(transform[4]) / 1e6
        
        print(f"\n控制面积估算（前10个最大汇水点）:")
        sorted_indices = np.argsort(acc_vals)[::-1][:10]
        for i, idx in enumerate(sorted_indices, 1):
            point = pour_points.iloc[idx]
            acc = point['accumulation']
            area = acc * pixel_area_km2
            print(f"  {i:2d}. 累积值={acc:7.0f}, 面积={area:6.2f} km², 位置=({point.geometry.x:.2f}, {point.geometry.y:.2f})")
    
    return pour_points

def visualize_network(flow_acc_path, pour_points_path, output_dir):
    """可视化河网和汇水点"""
    print("\n" + "="*80)
    print("生成可视化")
    print("="*80)
    
    with rasterio.open(flow_acc_path) as src:
        flow_acc = src.read(1)
        
    pour_points = gpd.read_file(pour_points_path)
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 图1: 流量累积（对数刻度）
    ax = axes[0]
    log_acc = np.log10(flow_acc + 1)
    im1 = ax.imshow(log_acc, cmap='Blues', origin='upper')
    plt.colorbar(im1, ax=ax, label='log10(Flow Accumulation + 1)')
    ax.set_title('Flow Accumulation (log scale)')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    # 图2: 河网和汇水点
    ax = axes[1]
    # 只显示高累积区域
    river_mask = flow_acc > 500
    river_display = np.where(river_mask, flow_acc, np.nan)
    im2 = ax.imshow(river_display, cmap='Blues', origin='upper', alpha=0.7)
    
    # 叠加汇水点
    if 'accumulation' in pour_points.columns:
        acc_vals = pour_points['accumulation'].values
        
        # 按累积值分组绘制
        large = acc_vals > 2000
        medium = (acc_vals >= 1000) & (acc_vals <= 2000)
        small = acc_vals < 1000
        
        # 转换坐标
        with rasterio.open(flow_acc_path) as src:
            transform = src.transform
            
        def lonlat_to_rowcol(lon, lat):
            col = (lon - transform[2]) / transform[0]
            row = (lat - transform[5]) / transform[4]
            return row, col
        
        for mask, color, size, label in [
            (large, 'red', 100, 'Large (>2000)'),
            (medium, 'orange', 50, 'Medium (1000-2000)'),
            (small, 'yellow', 20, 'Small (<1000)')
        ]:
            if np.any(mask):
                points_subset = pour_points[mask]
                rows = []
                cols = []
                for _, point in points_subset.iterrows():
                    r, c = lonlat_to_rowcol(point.geometry.x, point.geometry.y)
                    rows.append(r)
                    cols.append(c)
                ax.scatter(cols, rows, c=color, s=size, alpha=0.8, 
                          edgecolors='black', linewidths=1, label=label)
    
    ax.legend(loc='upper right')
    ax.set_title('River Network and Pour Points')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im2, ax=ax, label='Flow Accumulation')
    
    plt.tight_layout()
    output_path = output_dir / 'watershed_network_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 可视化已保存: {output_path}")
    plt.close()

def main():
    print("="*80)
    print("流域水系网络完整分析")
    print("="*80)
    
    # 使用测试结果 - 尝试多个位置
    possible_flow_paths = [
        "results/enhanced_workflow_tests/02_两步基础测试/outputs/terrain/flow_accumulation.tif",
        "results/enhanced_workflow_tests/01_最小测试-仅地形/outputs/terrain/flow_accumulation.tif",
        "results/richdem_verification/flow_acc_recompiled.tif",
    ]
    
    possible_pour_paths = [
        "results/enhanced_workflow_tests/02_两步基础测试/outputs/pour_points/pour_points.geojson",
        "results/enhanced_workflow_tests/03_三步流域划分/outputs/pour_points/pour_points.geojson",
    ]
    
    flow_acc_path = None
    for path in possible_flow_paths:
        if Path(path).exists():
            flow_acc_path = path
            break
            
    pour_points_path = None
    for path in possible_pour_paths:
        if Path(path).exists():
            pour_points_path = path
            break
    
    if not Path(flow_acc_path).exists():
        print(f"错误: 未找到流量累积文件: {flow_acc_path}")
        return
    
    if not Path(pour_points_path).exists():
        print(f"错误: 未找到汇水点文件: {pour_points_path}")
        return
    
    # 分析流量累积
    flow_stats = analyze_flow_accumulation(flow_acc_path)
    
    # 分析汇水点
    pour_points = analyze_pour_points(pour_points_path, flow_acc_path)
    
    # 生成可视化
    output_dir = Path("results/watershed_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualize_network(flow_acc_path, pour_points_path, output_dir)
    
    # 保存分析结果
    results = {
        'flow_accumulation': flow_stats,
        'pour_points': {
            'total_count': int(len(pour_points)),
            'has_accumulation': bool('accumulation' in pour_points.columns)
        }
    }
    
    with open(output_dir / 'network_analysis.json', 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x))
    
    print(f"\n✅ 分析结果已保存到: {output_dir}")
    print("\n" + "="*80)
    print("分析完成")
    print("="*80)

if __name__ == "__main__":
    main()
