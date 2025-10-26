#!/usr/bin/env python3
"""
改进的汇水点提取策略

策略：
1. 找到最大累积数点作为流域出口
2. 在干流上均匀提取n个汇水点
3. 在每个干流区间提取m个支流汇水点
"""

import numpy as np
import rasterio
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json

def extract_main_stream(flow_acc, outlet_point, threshold=1000):
    """
    从出口点回溯提取干流
    
    参数:
        flow_acc: 流量累积数组
        outlet_point: 出口点坐标 (row, col)
        threshold: 干流阈值
    
    返回:
        干流点列表 [(row, col, acc), ...]
    """
    print(f"\n提取干流...")
    print(f"  出口点: {outlet_point}")
    print(f"  干流阈值: {threshold}")
    
    # 使用8邻域回溯
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    main_stream = [(*outlet_point, flow_acc[outlet_point])]
    current = outlet_point
    visited = {outlet_point}
    
    rows, cols = flow_acc.shape
    
    while True:
        row, col = current
        
        # 查找上游累积数最大的点
        max_acc = 0
        next_point = None
        
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            
            # 边界检查
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            
            # 避免重复
            if (nr, nc) in visited:
                continue
            
            # 累积数检查
            acc = flow_acc[nr, nc]
            if acc >= threshold and acc > max_acc:
                max_acc = acc
                next_point = (nr, nc)
        
        if next_point is None:
            break
        
        current = next_point
        visited.add(current)
        main_stream.append((*current, flow_acc[current]))
        
        # 防止无限循环
        if len(main_stream) > 10000:
            print(f"  ⚠️ 干流点数超过10000，终止回溯")
            break
    
    # 按累积数从大到小排序（从出口到源头）
    main_stream.sort(key=lambda x: x[2], reverse=True)
    
    print(f"  ✅ 提取到 {len(main_stream)} 个干流点")
    
    return main_stream

def select_main_stream_points(main_stream, n_points=3):
    """
    在干流上均匀选择n个汇水点
    
    参数:
        main_stream: 干流点列表
        n_points: 要选择的点数
    
    返回:
        选中的点列表
    """
    print(f"\n在干流上均匀选择 {n_points} 个汇水点...")
    
    if len(main_stream) < n_points:
        print(f"  ⚠️ 干流点数({len(main_stream)})少于目标数({n_points})，使用所有点")
        return main_stream
    
    # 均匀间隔
    indices = np.linspace(0, len(main_stream) - 1, n_points, dtype=int)
    selected = [main_stream[i] for i in indices]
    
    print(f"  ✅ 已选择 {len(selected)} 个干流汇水点")
    for i, (row, col, acc) in enumerate(selected):
        print(f"     {i+1}. 位置({row}, {col}), 累积={acc:.0f}")
    
    return selected

def extract_tributary_points(flow_acc, main_stream_point, next_main_point, 
                             m_points=1, threshold=500):
    """
    在两个干流点之间提取支流汇水点
    
    参数:
        flow_acc: 流量累积数组
        main_stream_point: 当前干流点 (row, col, acc)
        next_main_point: 下一个干流点 (row, col, acc)
        m_points: 要提取的支流点数
        threshold: 支流阈值
    
    返回:
        支流点列表
    """
    row1, col1, _ = main_stream_point
    row2, col2, _ = next_main_point if next_main_point else (row1, col1, 0)
    
    # 定义搜索区域（两个干流点之间的矩形）
    row_min = min(row1, row2)
    row_max = max(row1, row2)
    col_min = min(col1, col2)
    col_max = max(col1, col2)
    
    # 扩展搜索区域
    expand = 50
    row_min = max(0, row_min - expand)
    row_max = min(flow_acc.shape[0] - 1, row_max + expand)
    col_min = max(0, col_min - expand)
    col_max = min(flow_acc.shape[1] - 1, col_max + expand)
    
    # 在区域内查找高累积点
    region = flow_acc[row_min:row_max+1, col_min:col_max+1]
    
    # 过滤：大于阈值且小于当前干流点（避免选到干流）
    mask = (region >= threshold) & (region < main_stream_point[2] * 0.5)
    
    if not np.any(mask):
        return []
    
    # 找到所有候选点
    candidates = []
    for i in range(region.shape[0]):
        for j in range(region.shape[1]):
            if mask[i, j]:
                global_row = row_min + i
                global_col = col_min + j
                candidates.append((global_row, global_col, region[i, j]))
    
    if not candidates:
        return []
    
    # 按累积数排序，选择前m个
    candidates.sort(key=lambda x: x[2], reverse=True)
    selected = candidates[:m_points]
    
    return selected

def extract_hierarchical_pour_points(flow_acc_path, n_main=3, m_tributary=1, 
                                    main_threshold=1000, tributary_threshold=500,
                                    output_dir="results/improved_pour_points"):
    """
    分层提取汇水点
    
    参数:
        flow_acc_path: 流量累积文件路径
        n_main: 干流汇水点数量
        m_tributary: 每个区间的支流汇水点数量
        main_threshold: 干流阈值
        tributary_threshold: 支流阈值
    """
    print("="*80)
    print("改进的汇水点提取")
    print("="*80)
    print(f"\n配置:")
    print(f"  干流汇水点数: {n_main}")
    print(f"  每区间支流数: {m_tributary}")
    print(f"  干流阈值: {main_threshold}")
    print(f"  支流阈值: {tributary_threshold}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取流量累积
    print(f"\n读取流量累积: {flow_acc_path}")
    with rasterio.open(flow_acc_path) as src:
        flow_acc = src.read(1)
        profile = src.profile
        transform = src.transform
    
    print(f"  数组大小: {flow_acc.shape}")
    print(f"  最大累积: {np.max(flow_acc):.0f}")
    print(f"  平均累积: {np.mean(flow_acc):.2f}")
    
    # 步骤1: 找出口点
    print("\n" + "-"*80)
    print("步骤1: 识别流域出口（最大累积数点）")
    print("-"*80)
    
    outlet_row, outlet_col = np.unravel_index(np.argmax(flow_acc), flow_acc.shape)
    outlet_acc = flow_acc[outlet_row, outlet_col]
    
    print(f"  出口位置: ({outlet_row}, {outlet_col})")
    print(f"  出口累积: {outlet_acc:.0f}")
    
    # 转换为地理坐标
    outlet_x, outlet_y = transform * (outlet_col, outlet_row)
    print(f"  地理坐标: ({outlet_x:.2f}, {outlet_y:.2f})")
    
    # 步骤2: 提取干流
    print("\n" + "-"*80)
    print("步骤2: 提取干流")
    print("-"*80)
    
    main_stream = extract_main_stream(flow_acc, (outlet_row, outlet_col), main_threshold)
    
    # 步骤3: 选择干流汇水点
    print("\n" + "-"*80)
    print("步骤3: 选择干流汇水点")
    print("-"*80)
    
    main_points = select_main_stream_points(main_stream, n_main)
    
    # 步骤4: 提取支流汇水点
    print("\n" + "-"*80)
    print("步骤4: 提取支流汇水点")
    print("-"*80)
    
    tributary_points = []
    
    for i in range(len(main_points) - 1):
        print(f"\n  区间 {i+1} (在干流点 {i+1} 和 {i+2} 之间):")
        
        tributaries = extract_tributary_points(
            flow_acc, 
            main_points[i], 
            main_points[i+1],
            m_tributary,
            tributary_threshold
        )
        
        if tributaries:
            print(f"    ✅ 提取到 {len(tributaries)} 个支流点")
            for j, (row, col, acc) in enumerate(tributaries):
                print(f"       支流 {j+1}: 位置({row}, {col}), 累积={acc:.0f}")
                tributary_points.append((row, col, acc, f"区间{i+1}"))
        else:
            print(f"    ⚠️ 未找到符合条件的支流点")
    
    # 汇总
    print("\n" + "="*80)
    print("汇水点提取总结")
    print("="*80)
    print(f"  流域出口: 1个")
    print(f"  干流汇水点: {len(main_points)}个")
    print(f"  支流汇水点: {len(tributary_points)}个")
    print(f"  总计: {1 + len(main_points) + len(tributary_points)}个")
    
    # 创建GeoDataFrame
    pour_points = []
    
    # 出口点
    x, y = transform * (outlet_col, outlet_row)
    pour_points.append({
        'geometry': Point(x, y),
        'type': 'outlet',
        'category': '流域出口',
        'accumulation': float(outlet_acc),
        'row': int(outlet_row),
        'col': int(outlet_col),
        'order': 0
    })
    
    # 干流点
    for i, (row, col, acc) in enumerate(main_points):
        x, y = transform * (col, row)
        pour_points.append({
            'geometry': Point(x, y),
            'type': 'mainstream',
            'category': f'干流点{i+1}',
            'accumulation': float(acc),
            'row': int(row),
            'col': int(col),
            'order': i + 1
        })
    
    # 支流点
    trib_order = 100
    for i, (row, col, acc, interval) in enumerate(tributary_points):
        x, y = transform * (col, row)
        pour_points.append({
            'geometry': Point(x, y),
            'type': 'tributary',
            'category': f'{interval}_支流{i+1}',
            'accumulation': float(acc),
            'row': int(row),
            'col': int(col),
            'order': trib_order + i
        })
    
    gdf = gpd.GeoDataFrame(pour_points, crs=profile['crs'])
    
    # 保存
    output_geojson = output_dir / 'improved_pour_points.geojson'
    gdf.to_file(output_geojson, driver='GeoJSON')
    print(f"\n✅ 汇水点已保存: {output_geojson}")
    
    # 保存统计信息
    stats = {
        'total_points': len(pour_points),
        'outlet': 1,
        'mainstream_points': len(main_points),
        'tributary_points': len(tributary_points),
        'outlet_accumulation': float(outlet_acc),
        'main_threshold': main_threshold,
        'tributary_threshold': tributary_threshold
    }
    
    stats_file = output_dir / 'pour_points_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # 可视化
    visualize_pour_points(flow_acc, gdf, transform, output_dir)
    
    return gdf, stats

def visualize_pour_points(flow_acc, gdf, transform, output_dir):
    """可视化汇水点"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 绘制流量累积（对数尺度）
    vmin = max(1, np.min(flow_acc[flow_acc > 0]))
    vmax = np.max(flow_acc)
    
    im = ax.imshow(flow_acc, cmap='Blues', 
                   norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
                   extent=[0, flow_acc.shape[1], flow_acc.shape[0], 0])
    
    plt.colorbar(im, ax=ax, label='Flow Accumulation (log scale)')
    
    # 绘制汇水点
    colors = {'outlet': 'red', 'mainstream': 'blue', 'tributary': 'green'}
    markers = {'outlet': '*', 'mainstream': 'o', 'tributary': '^'}
    sizes = {'outlet': 300, 'mainstream': 150, 'tributary': 100}
    
    for ptype in ['tributary', 'mainstream', 'outlet']:
        subset = gdf[gdf['type'] == ptype]
        if len(subset) > 0:
            rows = subset['row'].values
            cols = subset['col'].values
            ax.scatter(cols, rows, 
                      c=colors[ptype], 
                      marker=markers[ptype],
                      s=sizes[ptype],
                      edgecolors='white',
                      linewidths=2,
                      label=ptype.title(),
                      zorder=10)
    
    ax.set_title('Hierarchical Pour Points Extraction', fontsize=14, fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'improved_pour_points_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 可视化已保存: {output_path}")
    plt.close()

def main():
    # 使用修复后的流量累积
    flow_acc_path = "results/dem_processing_fixed/flow_accumulation_fixed.tif"
    
    if not Path(flow_acc_path).exists():
        print(f"错误: 流量累积文件不存在: {flow_acc_path}")
        return
    
    # 提取汇水点
    gdf, stats = extract_hierarchical_pour_points(
        flow_acc_path,
        n_main=3,           # 3个干流汇水点
        m_tributary=1,      # 每区间1个支流点
        main_threshold=10000,    # 干流阈值
        tributary_threshold=5000  # 支流阈值
    )
    
    print("\n" + "="*80)
    print("完成")
    print("="*80)

if __name__ == "__main__":
    main()
