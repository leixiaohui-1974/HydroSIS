#!/usr/bin/env python3
"""
Upper Truckee River完整11步水文建模工作流

这个脚本实现了从DEM处理到水动力模拟的完整水文建模流程，
包括所有中间步骤的详细可视化和数据输出。

11个工作流步骤：
1. DEM处理 - 地形分析、流向流量累计计算
2. 汇水点生成 - 自动生成或手动指定pour points
3. 参数分区和子流域划分 - 参数区和子流域的层次划分
4. 河道断面提取 - 沿河道提取横断面地形
5. 雨量站分布图生成 - 生成合成雨量站并可视化分布
6. 雨量序列生成 - 生成时间序列雨量数据
7. 泰森多边形计算 - 计算雨量站影响范围
8. 面雨量计算 - 将站点雨量插值到子流域
9. 水文模拟 - 产流模拟（HBV、SCS-CN等）
10. 水动力模拟 - 汇流路由模拟（Muskingum、动力波等）
11. 结果报告 - 生成完整的评估报告和可视化
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# 添加项目根目录到Python路径
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon

# HydroSIS导入
from hydrosis.config import (
    DelineationConfig,
    EvaluationConfig,
    IOConfig,
    ModelConfig,
    ModelStructureConfig,
    OutputArtifactsConfig,
    ParameterPartitionConfig,
    ParameterZoneConfig,
    RoutingModelConfig,
    RunoffModelConfig,
)
from hydrosis.delineation import utils as dutils
from hydrosis.delineation.channel_analysis import (
    compute_channel_mask,
    segments_to_feature_collection,
    trace_channel_segments,
)
from hydrosis.model import ChannelNetwork, Subbasin
from hydrosis.parameters.partition import partition_parameter_zones
# Import runoff and routing models to register them
import hydrosis.runoff  # noqa: F401
import hydrosis.routing  # noqa: F401
from hydrosis.workflow import run_workflow
from hydrosis.workflow.stages import (
    generate_precipitation_for_parameters,
    run_delineation_stage,
    snap_pour_points_to_flow_cells,
    suggest_accumulation_threshold,
)

try:
    import rasterio
    from rasterio import transform as rio_transform
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    warnings.warn("rasterio not available - DEM visualization will be limited")

# ============================================================================
# 第1步：DEM处理和地形分析
# ============================================================================

def step01_dem_processing(
    dem_path: Path,
    flow_dir_path: Path,
    flow_acc_path: Path,
    output_dir: Path,
) -> Dict[str, object]:
    """
    第1步：DEM处理和地形分析

    输入：
    - DEM栅格文件
    - 流向栅格文件
    - 流量累计栅格文件

    输出：
    - DEM高程图
    - 流向分布图
    - 流量累计图（对数尺度）
    - 坡度分布图
    - 地形统计表
    """
    print("\n" + "="*80)
    print("第1步：DEM处理和地形分析")
    print("="*80)

    step_dir = output_dir / "step_01_dem_processing"
    step_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "step": "01_DEM处理",
        "outputs": [],
    }

    if not HAS_RASTERIO:
        print("警告：rasterio不可用，跳过DEM可视化")
        return results

    # 读取DEM
    with rasterio.open(dem_path) as src:
        dem_data = src.read(1)
        transform = src.transform
        crs = src.crs
        # 正确处理nodata值
        nodata = src.nodata
        if nodata is not None:
            dem_data = np.where(dem_data == nodata, np.nan, dem_data)
        dem_data = np.where(np.isfinite(dem_data), dem_data, np.nan)

    # 读取流向
    with rasterio.open(flow_dir_path) as src:
        flowdir_data = src.read(1)

    # 读取流量累计
    with rasterio.open(flow_acc_path) as src:
        flowacc_data = src.read(1)
        flowacc_data = np.where(np.isfinite(flowacc_data), flowacc_data, 0.0)

    # 1.1 DEM高程图
    fig, ax = plt.subplots(figsize=(10, 8))
    valid_dem = dem_data[np.isfinite(dem_data)]
    if len(valid_dem) > 0:
        vmin, vmax = np.percentile(valid_dem, [2, 98])
        im = ax.imshow(dem_data, cmap='terrain', aspect='auto', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label='Elevation (m)')
    else:
        im = ax.imshow(dem_data, cmap='terrain', aspect='auto')
        plt.colorbar(im, ax=ax, label='Elevation (m)')
    ax.set_title('DEM Elevation Map')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    dem_fig = step_dir / "1.1_dem_elevation.png"
    plt.savefig(dem_fig, dpi=200, bbox_inches='tight')
    plt.close()
    results["outputs"].append(str(dem_fig))
    print(f"  ✓ 生成DEM高程图: {dem_fig.name}")

    # 1.2 流向图
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.colormaps.get_cmap('tab10').resampled(8)
    im = ax.imshow(flowdir_data, cmap=cmap, aspect='auto', vmin=0, vmax=7)
    plt.colorbar(im, ax=ax, label='Flow Direction Code', ticks=range(8))
    ax.set_title('D8 Flow Direction')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    flowdir_fig = step_dir / "1.2_flow_direction.png"
    plt.savefig(flowdir_fig, dpi=200, bbox_inches='tight')
    plt.close()
    results["outputs"].append(str(flowdir_fig))
    print(f"  ✓ 生成流向图: {flowdir_fig.name}")

    # 1.3 流量累计图（对数尺度）
    fig, ax = plt.subplots(figsize=(10, 8))
    acc_plot = np.where(flowacc_data > 0, flowacc_data, np.nan)
    im = ax.imshow(np.log1p(acc_plot), cmap='Blues', aspect='auto')
    plt.colorbar(im, ax=ax, label='Log(1 + Flow Accumulation)')
    ax.set_title('Flow Accumulation (Log Scale)')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    flowacc_fig = step_dir / "1.3_flow_accumulation.png"
    plt.savefig(flowacc_fig, dpi=200, bbox_inches='tight')
    plt.close()
    results["outputs"].append(str(flowacc_fig))
    print(f"  ✓ 生成流量累计图: {flowacc_fig.name}")

    # 1.4 坡度计算和可视化
    res = abs(transform[0])  # 分辨率
    dy, dx = np.gradient(dem_data, res, res)
    slope = np.arctan(np.sqrt(dx**2 + dy**2)) * 180 / np.pi
    slope = np.where(np.isfinite(slope), slope, np.nan)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(slope, cmap='YlOrRd', aspect='auto', vmin=0, vmax=45)
    plt.colorbar(im, ax=ax, label='Slope (degrees)')
    ax.set_title('Slope Distribution')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    slope_fig = step_dir / "1.4_slope_map.png"
    plt.savefig(slope_fig, dpi=200, bbox_inches='tight')
    plt.close()
    results["outputs"].append(str(slope_fig))
    print(f"  ✓ 生成坡度图: {slope_fig.name}")

    # 1.5 地形统计表
    stats = pd.DataFrame({
        'Metric': [
            'Min Elevation (m)',
            'Max Elevation (m)',
            'Mean Elevation (m)',
            'Std Elevation (m)',
            'Min Slope (deg)',
            'Max Slope (deg)',
            'Mean Slope (deg)',
            'Max Flow Accumulation',
        ],
        'Value': [
            float(np.nanmin(valid_dem)),
            float(np.nanmax(valid_dem)),
            float(np.nanmean(valid_dem)),
            float(np.nanstd(valid_dem)),
            float(np.nanmin(slope)),
            float(np.nanmax(slope)),
            float(np.nanmean(slope)),
            float(np.max(flowacc_data)),
        ]
    })
    stats_file = step_dir / "1.5_terrain_statistics.csv"
    stats.to_csv(stats_file, index=False)
    results["outputs"].append(str(stats_file))
    print(f"  ✓ 保存地形统计: {stats_file.name}")

    results["transform"] = transform
    results["crs"] = crs
    results["shape"] = dem_data.shape

    print(f"第1步完成：生成{len(results['outputs'])}个输出文件")
    return results

# ============================================================================
# 第2步：汇水点生成
# ============================================================================

def step02_pour_point_generation(
    flow_dir_path: Path,
    flow_acc_path: Path,
    output_dir: Path,
    main_stream_count: int = 3,
) -> Dict[str, object]:
    """
    第2步：汇水点生成（Pfafstetter编码方案）

    策略：
    1. 找到流域出口点（最大累积数点）
    2. 从出口点向上追溯主干流
    3. 在主干流上选3个点，使它们控制的流域面积基本3等分
    4. 对每个主干流分区，选择1个最大支流汇入点
    5. 使用Pfafstetter风格的数字编码

    编码方案：
    - 主干流（从下游到上游）：10, 20, 30
    - 对应的支流：11, 21, 31

    输入：
    - 流向栅格
    - 流量累计栅格

    输出：
    - 6个汇水点：3个干流 + 3个支流
    - Pour points GeoJSON文件
    - Pour points统计表
    - Pour points位置图
    """
    print("\n" + "="*80)
    print(f"第2步：汇水点生成（{main_stream_count}个干流 + {main_stream_count}个支流，Pfafstetter编码）")
    print("="*80)

    step_dir = output_dir / "step_02_pour_points"
    step_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "step": "02_汇水点生成",
        "outputs": [],
    }

    # 读取流量累计数据
    with rasterio.open(flow_acc_path) as src:
        flowacc = src.read(1)
        flowacc = np.where(np.isfinite(flowacc), flowacc, 0.0)
        transform = src.transform
        rows, cols = flowacc.shape
        cell_area_km2 = abs(src.res[0] * src.res[1]) / 1_000_000.0

    # 读取流向数据
    with rasterio.open(flow_dir_path) as src:
        flowdir = src.read(1)

    print("  ⚙ 分析流域结构...")

    # 1. 找到出口点（流量累计最大的点）
    outlet_row, outlet_col = np.unravel_index(np.argmax(flowacc), flowacc.shape)
    outlet_acc = float(flowacc[outlet_row, outlet_col])
    total_basin_area_km2 = outlet_acc * cell_area_km2
    print(f"  ✓ 识别出口点: ({outlet_row}, {outlet_col}), 累计={outlet_acc:.0f}, 流域面积={total_basin_area_km2:.2f} km²")

    # 2. D8流向编码（Richdem使用1-8）
    D8_OFFSETS = {
        1: (0, 1),   # East
        2: (-1, 1),  # NE
        3: (-1, 0),  # North
        4: (-1, -1), # NW
        5: (0, -1),  # West
        6: (1, -1),  # SW
        7: (1, 0),   # South
        8: (1, 1),   # SE
    }

    # 反向追溯：找到所有流向当前点的上游点
    def find_all_upstream(r, c):
        """找到所有流向(r,c)的上游点"""
        upstream_cells = []
        # 检查周围8个格网
        for code, (dr, dc) in D8_OFFSETS.items():
            # 邻居格网位置
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                # 获取邻居的流向代码
                neighbor_flowdir = int(flowdir[nr, nc])
                # 计算邻居流向的目标位置
                if neighbor_flowdir in D8_OFFSETS:
                    target_dr, target_dc = D8_OFFSETS[neighbor_flowdir]
                    target_r, target_c = nr + target_dr, nc + target_dc
                    # 如果邻居流向当前格网，则它是上游
                    if target_r == r and target_c == c:
                        upstream_cells.append((nr, nc, flowacc[nr, nc]))
        return upstream_cells

    # 3. 追溯主干流（沿流量累计最大的路径）
    main_stream_cells = [(outlet_row, outlet_col)]
    current = (outlet_row, outlet_col)
    visited = {(outlet_row, outlet_col)}

    while True:
        upstream = find_all_upstream(current[0], current[1])
        if not upstream:
            break

        # 选择流量累计最大的上游点（主干流）
        upstream.sort(key=lambda x: x[2], reverse=True)
        next_cell = (upstream[0][0], upstream[0][1])

        if next_cell in visited:
            break

        main_stream_cells.append(next_cell)
        visited.add(next_cell)
        current = next_cell

        if len(main_stream_cells) > 10000:  # 防止无限循环
            break

    print(f"  ✓ 追溯主干流: {len(main_stream_cells)}个格网")

    # 4. 在主干流上选择3个点
    # 修正：第一个点（10号）应该在流域出口（最大累积数点）
    # 上游两个点（20号、30号）分别控制约1/3和2/3的流域面积
    main_stream_points = []

    # 第一个干流点（10号）：直接设在流域出口
    x, y = transform * (outlet_col, outlet_row)
    main_stream_points.append({
        'id': '10',
        'row': int(outlet_row),
        'col': int(outlet_col),
        'x': float(x),
        'y': float(y),
        'accumulation': float(outlet_acc),
        'controlled_area_km2': float(total_basin_area_km2),
        'type': 'main_stream',
        'pfafstetter_code': 10,
    })

    # 上游两个点（20号、30号）：分别控制约1/3和2/3的流域面积
    # 这里的"控制面积"是指从该点向上的流域面积
    target_areas = [
        total_basin_area_km2 * 1.0 / 3.0,  # 20号点：约1/3流域
        total_basin_area_km2 * 2.0 / 3.0,  # 30号点：约2/3流域
    ]

    for idx, target_area in enumerate(target_areas):
        # 在主干流上找到最接近目标面积的点（排除出口点）
        best_cell = None
        best_diff = float('inf')

        for r, c in main_stream_cells[1:]:  # 跳过第一个点（出口点）
            cell_area = flowacc[r, c] * cell_area_km2
            diff = abs(cell_area - target_area)
            if diff < best_diff:
                best_diff = diff
                best_cell = (r, c)

        if best_cell:
            r, c = best_cell
            x, y = transform * (c, r)
            controlled_area = flowacc[r, c] * cell_area_km2

            # Pfafstetter编码：20, 30（从下游到上游）
            pfaf_code = (idx + 2) * 10

            main_stream_points.append({
                'id': str(pfaf_code),
                'row': int(r),
                'col': int(c),
                'x': float(x),
                'y': float(y),
                'accumulation': float(flowacc[r, c]),
                'controlled_area_km2': float(controlled_area),
                'type': 'main_stream',
                'pfafstetter_code': pfaf_code,
            })

    # 按照accumulation从大到小排序，确保顺序是从下游到上游
    # 这样在后续分区计算时才能正确处理
    main_stream_points.sort(key=lambda p: p['accumulation'], reverse=True)

    print(f"  ✓ 选择{len(main_stream_points)}个干流汇水点（按流量从下游到上游排列）")
    for p in main_stream_points:
        pct = p['controlled_area_km2'] / total_basin_area_km2 * 100
        print(f"    - {p['id']}号点: 累积={p['accumulation']:.0f}, 控制面积={p['controlled_area_km2']:.2f} km² ({pct:.1f}%)")

    # 5. 为每个干流分区找到1个最大支流汇入点
    main_stream_set = set(main_stream_cells)

    # 定义函数：追溯一个点的所有上游格网
    def delineate_watershed(pour_r, pour_c):
        """追溯汇水点的所有上游格网"""
        watershed = set()
        queue = [(pour_r, pour_c)]
        visited_ws = {(pour_r, pour_c)}

        while queue:
            r, c = queue.pop(0)
            watershed.add((r, c))

            # 找到所有流向当前点的上游点
            upstream = find_all_upstream(r, c)
            for ur, uc, _ in upstream:
                if (ur, uc) not in visited_ws:
                    visited_ws.add((ur, uc))
                    queue.append((ur, uc))

            if len(watershed) > 100000:  # 防止无限循环
                break

        return watershed

    tributary_points = []

    # 正向处理（从下游到上游），main_stream_points已按accumulation从大到小排序
    # 每个干流点的分区 = 本点的上游 - 上游干流点的上游
    for main_idx in range(len(main_stream_points)):
        main_point = main_stream_points[main_idx]
        main_id = main_point['id']
        main_r, main_c = main_point['row'], main_point['col']

        print(f"  ⚙ 为干流点{main_id}寻找最大支流...")

        # 追溯该干流点的流域范围
        watershed = delineate_watershed(main_r, main_c)

        # 如果不是最上游的点，需要排除上游干流点的流域
        if main_idx < len(main_stream_points) - 1:
            upstream_point = main_stream_points[main_idx + 1]
            upstream_watershed = delineate_watershed(upstream_point['row'], upstream_point['col'])
            watershed = watershed - upstream_watershed

        print(f"    - 本分区流域范围: {len(watershed)}个格网")

        # 在流域内寻找候选支流点（不在主干流上的高流量点）
        threshold = main_point['accumulation'] * 0.05  # 至少是干流点流量的5%
        candidate_tribs = []

        for wr, wc in watershed:
            acc = flowacc[wr, wc]
            # 必须满足：在流域内、不在主干流上、流量足够大
            if (wr, wc) not in main_stream_set and acc > threshold:
                candidate_tribs.append((wr, wc, acc))

        # 按流量排序，选择最大的
        if candidate_tribs:
            candidate_tribs.sort(key=lambda x: x[2], reverse=True)
            tr, tc, tacc = candidate_tribs[0]

            x, y = transform * (tc, tr)
            trib_area = tacc * cell_area_km2

            # Pfafstetter编码：11, 21, 31（与对应干流相关）
            pfaf_code = int(main_id) + 1

            tributary_points.append({
                'id': str(pfaf_code),
                'row': int(tr),
                'col': int(tc),
                'x': float(x),
                'y': float(y),
                'accumulation': float(tacc),
                'controlled_area_km2': float(trib_area),
                'type': 'tributary',
                'main_stream_id': main_id,
                'pfafstetter_code': pfaf_code,
            })

            print(f"    - 选择支流{pfaf_code}: 控制面积={trib_area:.2f} km²")
        else:
            print(f"    - 未找到合适的支流")

    # 6. 合并所有汇水点
    all_points = main_stream_points + tributary_points
    print(f"  ✓ 生成总共{len(all_points)}个汇水点（{len(main_stream_points)}干流 + {len(tributary_points)}支流）")

    # 7. 保存为GeoJSON
    features = []
    for point in all_points:
        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [point['x'], point['y']]
            },
            'properties': {
                'id': point['id'],
                'type': point['type'],
                'row': point['row'],
                'col': point['col'],
                'accumulation': point['accumulation'],
                'controlled_area_km2': point['controlled_area_km2'],
                'pfafstetter_code': point['pfafstetter_code'],
            }
        }
        if point['type'] == 'tributary':
            feature['properties']['main_stream_id'] = point['main_stream_id']
        features.append(feature)

    geojson = {
        'type': 'FeatureCollection',
        'features': features
    }

    geojson_path = step_dir / "2.1_pour_points.geojson"
    with open(geojson_path, 'w') as f:
        json.dump(geojson, f, indent=2)
    results["outputs"].append(str(geojson_path))
    print(f"  ✓ 保存汇水点GeoJSON: {geojson_path.name}")

    # 8. 保存统计表
    stats_df = pd.DataFrame(all_points)
    stats_df = stats_df[['id', 'type', 'pfafstetter_code', 'row', 'col', 'x', 'y',
                          'accumulation', 'controlled_area_km2']]
    stats_path = step_dir / "2.2_pour_points_table.csv"
    stats_df.to_csv(stats_path, index=False)
    results["outputs"].append(str(stats_path))
    print(f"  ✓ 保存汇水点统计表: {stats_path.name}")

    # 9. 可视化
    fig, ax = plt.subplots(figsize=(12, 10))

    # 绘制流量累计作为背景
    flowacc_log = np.log10(flowacc + 1)
    im = ax.imshow(flowacc_log, cmap='Blues', aspect='auto')
    plt.colorbar(im, ax=ax, label='Log10(Flow Accumulation + 1)', shrink=0.8)

    # 绘制主干流
    if main_stream_cells:
        stream_rows = [c[0] for c in main_stream_cells]
        stream_cols = [c[1] for c in main_stream_cells]
        ax.plot(stream_cols, stream_rows, 'r-', linewidth=2, alpha=0.7, label='Main Stream')

    # 绘制汇水点
    for point in all_points:
        if point['type'] == 'main_stream':
            ax.plot(point['col'], point['row'], 'ro', markersize=12,
                   markeredgecolor='white', markeredgewidth=2, label='Main Stream Point' if point == main_stream_points[0] else '')
            ax.text(point['col'] + 5, point['row'], point['id'], fontsize=12, fontweight='bold',
                   color='red', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.plot(point['col'], point['row'], 'go', markersize=10,
                   markeredgecolor='white', markeredgewidth=2, label='Tributary Point' if point == tributary_points[0] else '')
            ax.text(point['col'] + 5, point['row'], point['id'], fontsize=10, fontweight='bold',
                   color='green', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_title('Upper Truckee River - Pour Points (Pfafstetter Encoding)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    map_path = step_dir / "2.3_pour_points_map.png"
    plt.savefig(map_path, dpi=200, bbox_inches='tight')
    plt.close()
    results["outputs"].append(str(map_path))
    print(f"  ✓ 生成汇水点位置图: {map_path.name}")

    results["pour_points"] = all_points
    results["main_stream_cells"] = main_stream_cells
    results["pour_points_path"] = geojson_path

    print(f"第2步完成：生成{len(results['outputs'])}个输出文件")
    return results
# ============================================================================
# 第3步：参数分区和子流域划分
# ============================================================================

def step03_parameter_zones_and_subbasins(
    dem_path: Path,
    flow_dir_path: Path,
    flow_acc_path: Path,
    pour_points_path: Path,
    output_dir: Path,
) -> Dict[str, object]:
    """
    第3步：参数分区和子流域划分

    输入：
    - DEM
    - 流向
    - 流量累计
    - Pour points

    输出：
    - 子流域边界GeoJSON
    - 参数分区GeoJSON
    - 河道网络GeoJSON
    - 子流域统计表
    - 参数区统计表
    - 可视化地图
    """
    print("\n" + "="*80)
    print("第3步：参数分区和子流域划分")
    print("="*80)

    step_dir = output_dir / "step_03_zones_subbasins"
    step_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir = output_dir / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    parameter_dir = output_dir / "parameters"
    parameter_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "step": "03_参数分区和子流域划分",
        "outputs": [],
    }

    # 配置
    delineation_cfg = DelineationConfig(
        dem_path=dem_path,
        pour_points_path=pour_points_path,
        flow_direction_path=flow_dir_path,
        flow_accumulation_path=flow_acc_path,
        accumulation_threshold=15000.0,
        intermediate_directory=intermediate_dir,
        parameter_directory=parameter_dir,
    )

    partition_cfg = ParameterPartitionConfig(
        pour_points_path=pour_points_path,
        target_subzone_area_km2=10.0,  # Smaller target for more subzones
        min_subzone_area_km2=2.0,       # Lower minimum
        max_subzones_per_zone=None,     # No limit on subzones
        area_balance_tolerance=0.5,      # More flexible
        subzone_accumulation_threshold=None,  # Use area-based subdivision
    )

    model_structure = ModelStructureConfig(
        default_runoff_model="hbv",
        default_routing_model="muskingum",
    )

    outputs_cfg = OutputArtifactsConfig()

    # Step 3A: 基础子流域划分（基于pour points）
    print("  ⚙ 运行基础流域划分...")
    pour_points = dutils.read_pour_points_geojson(pour_points_path)

    # 读取DEM和流向数据
    with rasterio.open(dem_path) as src:
        dem_array = src.read(1)
        dem_transform = src.transform
        dem_crs = src.crs

    # 构建流向网络
    flowdir, upstream, shape = dutils.build_flow_network(flow_dir_path)
    print(f"  ✓ 构建流向网络: {shape}")

    # 读取流量累计
    with rasterio.open(flow_acc_path) as src:
        flowacc = src.read(1)
        cell_area_km2 = abs(src.res[0] * src.res[1]) / 1_000_000.0

    # 划分基础子流域（每个pour point一个）
    subbasins = []
    sorted_pps = sorted(pour_points, key=lambda pp: -pp.accumulation)

    for i, pp in enumerate(sorted_pps):
        sub_id = pp.id
        mask = dutils.delineate_watershed(pp, upstream, shape)
        area_km2 = float(mask.sum() * cell_area_km2)

        # 确定下游子流域
        downstream_id = None
        for j in range(i + 1, len(sorted_pps)):
            other_pp = sorted_pps[j]
            other_mask = dutils.delineate_watershed(other_pp, upstream, shape)
            if other_mask[pp.row, pp.col]:
                downstream_id = other_pp.id
                break

        subbasin = Subbasin(
            id=sub_id,
            area_km2=area_km2,
            downstream=downstream_id,
            parameters={},
        )
        subbasins.append(subbasin)

    print(f"  ✓ 成功划分{len(subbasins)}个基础子流域")

    # Step 3B: 参数分区细化（使用partition_parameter_zones进行子区划分）
    print("  ⚙ 运行参数分区细化...")

    # 更新delineation_cfg使用预计算的subbasins
    delineation_cfg_updated = DelineationConfig(
        dem_path=dem_path,
        pour_points_path=pour_points_path,
        flow_direction_path=flow_dir_path,
        flow_accumulation_path=flow_acc_path,
        accumulation_threshold=delineation_cfg.accumulation_threshold,
        intermediate_directory=intermediate_dir,
        parameter_directory=parameter_dir,
        precomputed_subbasins=[
            {'id': sub.id, 'area_km2': sub.area_km2, 'downstream': sub.downstream,
             'parameters': sub.parameters}
            for sub in subbasins
        ],
    )

    # 调用partition_parameter_zones进行细化
    partition_outputs = partition_parameter_zones(
        delineation_cfg=delineation_cfg_updated,
        partition_cfg=partition_cfg,
        model_structure=model_structure,
        outputs_cfg=outputs_cfg,
    )

    if partition_outputs:
        subzone_count = len(partition_outputs.subzone_summaries)
        zone_count = len(partition_outputs.parameter_zones)
        print(f"  ✓ 生成{subzone_count}个参数子区")
        print(f"  ✓ 生成{zone_count}个参数区")

        # 重新编码子流域ID为Pfafstetter数字编码
        # Zone 10 -> 100, 101, 102, ...
        # Zone 11 -> 110, 111, 112, ...
        # Zone 20 -> 200, 201, 202, ...
        print("  ⚙ 应用Pfafstetter数字编码...")

        # 按zone分组子流域
        zone_subzones = {}
        for subzone in partition_outputs.subzone_summaries:
            zone_id = subzone.zone_id
            if zone_id not in zone_subzones:
                zone_subzones[zone_id] = []
            zone_subzones[zone_id].append(subzone)

        # 创建ID映射：old_id -> new_id
        id_mapping = {}
        for zone_id in sorted(zone_subzones.keys()):
            subzones = zone_subzones[zone_id]
            # 按原ID排序保持一致性
            subzones.sort(key=lambda sz: sz.subzone_id)

            # 生成新ID：zone_code * 10 + index
            # 例如：zone 10 -> 100, 101, 102, ...
            zone_code = int(zone_id) if zone_id.isdigit() else int(zone_id.split('_')[0])
            for idx, subzone in enumerate(subzones):
                new_id = str(zone_code * 10 + idx)
                id_mapping[subzone.subzone_id] = new_id

        # 更新subzone_summaries中的IDs
        for subzone in partition_outputs.subzone_summaries:
            subzone.subzone_id = id_mapping[subzone.subzone_id]
            # 更新downstream引用
            if subzone.downstream_subzone_id and subzone.downstream_subzone_id in id_mapping:
                subzone.downstream_subzone_id = id_mapping[subzone.downstream_subzone_id]

        print(f"  ✓ 重新编码{len(id_mapping)}个子流域为Pfafstetter数字编码")

        # 更新parameter目录下的GeoJSON和CSV文件中的IDs
        print("  ⚙ 更新输出文件中的IDs...")

        # 更新parameter_subbasins.geojson
        param_subbasin_geojson = parameter_dir / "parameter_subbasins.geojson"
        if param_subbasin_geojson.exists():
            with open(param_subbasin_geojson, 'r', encoding='utf-8') as f:
                geojson_data = json.load(f)
            for feature in geojson_data['features']:
                old_id = feature['properties'].get('subzone_id') or feature['properties'].get('id')
                if old_id and old_id in id_mapping:
                    new_id = id_mapping[old_id]
                    if 'subzone_id' in feature['properties']:
                        feature['properties']['subzone_id'] = new_id
                    if 'id' in feature['properties']:
                        feature['properties']['id'] = new_id
                    # 更新downstream引用
                    if 'downstream_subzone_id' in feature['properties']:
                        ds_id = feature['properties']['downstream_subzone_id']
                        if ds_id and ds_id in id_mapping:
                            feature['properties']['downstream_subzone_id'] = id_mapping[ds_id]
            with open(param_subbasin_geojson, 'w', encoding='utf-8') as f:
                json.dump(geojson_data, f, indent=2)

        # 更新parameter_channels.geojson
        channel_geojson = parameter_dir / "parameter_channels.geojson"
        if channel_geojson.exists():
            with open(channel_geojson, 'r', encoding='utf-8') as f:
                geojson_data = json.load(f)
            for feature in geojson_data['features']:
                props = feature['properties']
                # 更新segment_id (河道ID)
                if 'segment_id' in props and props['segment_id'] in id_mapping:
                    props['segment_id'] = id_mapping[props['segment_id']]
                if 'subzone_id' in props and props['subzone_id'] in id_mapping:
                    props['subzone_id'] = id_mapping[props['subzone_id']]
                # 更新downstream_id
                if 'downstream_id' in props and props['downstream_id'] and props['downstream_id'] in id_mapping:
                    props['downstream_id'] = id_mapping[props['downstream_id']]
                # 更新upstream_ids（可能是分号分隔的列表）
                if 'upstream_ids' in props and props['upstream_ids']:
                    upstream_list = str(props['upstream_ids']).split(';')
                    new_upstream = [id_mapping.get(uid.strip(), uid.strip()) for uid in upstream_list if uid.strip()]
                    props['upstream_ids'] = ';'.join(new_upstream) if new_upstream else None
            with open(channel_geojson, 'w', encoding='utf-8') as f:
                json.dump(geojson_data, f, indent=2)

        # 更新parameter_subbasins.csv
        param_subbasin_csv = parameter_dir / "parameter_subbasins.csv"
        if param_subbasin_csv.exists():
            df = pd.read_csv(param_subbasin_csv)
            if 'subzone_id' in df.columns:
                df['subzone_id'] = df['subzone_id'].map(lambda x: id_mapping.get(x, x))
            if 'downstream_subzone_id' in df.columns:
                df['downstream_subzone_id'] = df['downstream_subzone_id'].map(lambda x: id_mapping.get(x, x) if pd.notna(x) else x)
            df.to_csv(param_subbasin_csv, index=False)

        # 更新parameter_channels.csv
        channel_csv = parameter_dir / "parameter_channels.csv"
        if channel_csv.exists():
            df = pd.read_csv(channel_csv)
            if 'segment_id' in df.columns:
                df['segment_id'] = df['segment_id'].map(lambda x: id_mapping.get(x, x))
            if 'subzone_id' in df.columns:
                df['subzone_id'] = df['subzone_id'].map(lambda x: id_mapping.get(x, x))
            if 'downstream_id' in df.columns:
                df['downstream_id'] = df['downstream_id'].map(lambda x: id_mapping.get(x, x) if pd.notna(x) else x)
            if 'upstream_ids' in df.columns:
                def update_upstream(val):
                    if pd.isna(val) or not val:
                        return val
                    ids = str(val).split(';')
                    return ';'.join([id_mapping.get(i.strip(), i.strip()) for i in ids if i.strip()])
                df['upstream_ids'] = df['upstream_ids'].map(update_upstream)
            df.to_csv(channel_csv, index=False)

        print(f"  ✓ 更新GeoJSON和CSV文件中的ID引用")

    else:
        print("  ⚠ 未生成参数分区输出")

    # 复制由run_delineation_stage生成的文件到step目录
    import shutil

    # 查找并复制子流域GeoJSON文件
    subbasin_geojson = intermediate_dir / "subbasins.geojson"
    if subbasin_geojson.exists():
        dest = step_dir / "3.1_subbasins.geojson"
        shutil.copy2(subbasin_geojson, dest)
        results["outputs"].append(str(dest))
        print(f"  ✓ 保存子流域边界: {dest.name}")
    elif (parameter_dir / "subzone_geometries.geojson").exists():
        # 如果subbasins.geojson不存在，使用subzone_geometries
        subbasin_geojson = parameter_dir / "subzone_geometries.geojson"
        dest = step_dir / "3.1_subbasins.geojson"
        shutil.copy2(subbasin_geojson, dest)
        results["outputs"].append(str(dest))
        print(f"  ✓ 保存子流域边界: {dest.name}")

    # 复制参数子流域文件
    param_subbasin_geojson = parameter_dir / "parameter_subbasins.geojson"
    if param_subbasin_geojson.exists():
        dest = step_dir / "3.2_parameter_subbasins.geojson"
        shutil.copy2(param_subbasin_geojson, dest)
        results["outputs"].append(str(dest))
        print(f"  ✓ 保存参数子流域: {dest.name}")

    # 复制subzone_geometries文件
    subzone_geojson = parameter_dir / "subzone_geometries.geojson"
    if subzone_geojson.exists():
        dest = step_dir / "3.2_subzone_geometries.geojson"
        shutil.copy2(subzone_geojson, dest)
        results["outputs"].append(str(dest))
        print(f"  ✓ 保存参数子区几何: {dest.name}")

    # 复制河道网络文件
    channel_geojson = parameter_dir / "parameter_channels.geojson"
    if not channel_geojson.exists():
        channel_geojson = intermediate_dir / "channel_network.geojson"
    if channel_geojson.exists():
        dest = step_dir / "3.3_channel_network.geojson"
        shutil.copy2(channel_geojson, dest)
        results["outputs"].append(str(dest))
        print(f"  ✓ 保存河道网络: {dest.name}")

    # 生成统计表
    if partition_outputs.subzone_summaries:
        subzone_stats = pd.DataFrame([
            {
                'Subzone_ID': s.subzone_id,
                'Zone_ID': s.zone_id,
                'Area_km2': s.area_km2,
                'Downstream_ID': s.downstream_subzone_id,
            }
            for s in partition_outputs.subzone_summaries
        ])
        stats_file = step_dir / "3.4_subzone_statistics.csv"
        subzone_stats.to_csv(stats_file, index=False)
        results["outputs"].append(str(stats_file))
        print(f"  ✓ 保存子区统计: {stats_file.name}")

    # ========================================================================
    # 生成可视化图片
    # ========================================================================
    print("  ⚙ 生成可视化图片...")

    # 3.5 子流域分区可视化（显示183个细化子流域）
    # 修复：使用param_subbasin_geojson而不是subbasin_geojson
    if param_subbasin_geojson.exists():
        geojson_data = json.loads(param_subbasin_geojson.read_text(encoding='utf-8'))
        fig, ax = plt.subplots(figsize=(14, 12))

        # 绘制DEM作为背景
        with rasterio.open(dem_path) as src:
            dem_array = src.read(1)
            # 正确处理nodata值
            nodata = src.nodata
            if nodata is not None:
                dem_array = np.where(dem_array == nodata, np.nan, dem_array)
            dem_array = np.where(np.isfinite(dem_array), dem_array, np.nan)
            # 过滤异常值
            valid_data = dem_array[np.isfinite(dem_array)]
            if len(valid_data) > 0:
                vmin, vmax = np.percentile(valid_data, [2, 98])
                extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
                im = ax.imshow(dem_array, cmap='terrain', extent=extent, alpha=0.5, vmin=vmin, vmax=vmax)
                plt.colorbar(im, ax=ax, label='Elevation (m)', shrink=0.8)

        # 绘制子流域边界
        from shapely.geometry import shape as shapely_shape
        import matplotlib.cm as cm
        n_features = len(geojson_data['features'])

        # 如果子流域数量较多，使用渐变色而不是离散颜色
        if n_features > 20:
            cmap = cm.get_cmap('tab20', n_features)
            show_labels = True  # 修改：始终显示子流域编码标注
            show_legend = False
        else:
            cmap = cm.get_cmap('tab10')
            show_labels = True
            show_legend = True

        for i, feature in enumerate(geojson_data['features']):
            geom = shapely_shape(feature['geometry'])
            sub_id = feature['properties'].get('id', feature['properties'].get('subzone_id', f'Sub_{i}'))
            area = feature['properties'].get('area_km2', 0)
            color = cmap(i / max(1, n_features - 1) if n_features > 20 else i % 10)

            if geom.geom_type == 'Polygon':
                x, y = geom.exterior.xy
                label = f'{sub_id} ({area:.1f} km²)' if show_legend else None
                ax.plot(x, y, linewidth=1.5, color=color, label=label)
                ax.fill(x, y, alpha=0.3, color=color)
            elif geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    x, y = poly.exterior.xy
                    label = f'{sub_id} ({area:.1f} km²)' if show_legend and poly == geom.geoms[0] else None
                    ax.plot(x, y, linewidth=1.5, color=color, label=label)
                    ax.fill(x, y, alpha=0.3, color=color)

            # 添加子流域编码标签
            if show_labels:
                centroid = geom.centroid
                # 根据子流域数量调整字体大小
                fontsize = 6 if n_features > 100 else (7 if n_features > 50 else 9)
                ax.text(centroid.x, centroid.y, sub_id, fontsize=fontsize, fontweight='bold',
                       ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, pad=0.2))

        ax.set_title(f'Upper Truckee River - Subbasin Delineation ({n_features} subbasins)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        if show_legend:
            ax.legend(loc='best', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)

        subbasin_map = step_dir / "3.5_subbasin_map.png"
        plt.savefig(subbasin_map, dpi=200, bbox_inches='tight')
        plt.close()
        results["outputs"].append(str(subbasin_map))
        print(f"  ✓ 生成子流域分区图 ({n_features}个子流域): {subbasin_map.name}")

    # 3.6 参数分区可视化（显示12个参数分区）
    # 修复：使用parameter_zones.geojson而不是param_subbasin_geojson
    param_zones_geojson = parameter_dir / "parameter_zones.geojson"
    if param_zones_geojson.exists():
        geojson_data = json.loads(param_zones_geojson.read_text(encoding='utf-8'))
        fig, ax = plt.subplots(figsize=(12, 10))

        # 绘制DEM作为背景
        with rasterio.open(dem_path) as src:
            dem_array = src.read(1)
            # 正确处理nodata值
            nodata = src.nodata
            if nodata is not None:
                dem_array = np.where(dem_array == nodata, np.nan, dem_array)
            dem_array = np.where(np.isfinite(dem_array), dem_array, np.nan)
            # 过滤异常值
            valid_data = dem_array[np.isfinite(dem_array)]
            if len(valid_data) > 0:
                vmin, vmax = np.percentile(valid_data, [2, 98])
                extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
                im = ax.imshow(dem_array, cmap='terrain', extent=extent, alpha=0.5, vmin=vmin, vmax=vmax)

        # 绘制参数分区（从parameter_zones.geojson）
        from shapely.geometry import shape as shapely_shape
        colors = plt.colormaps.get_cmap('Set3')
        n_zones = len(geojson_data['features'])

        for i, feature in enumerate(geojson_data['features']):
            geom = shapely_shape(feature['geometry'])
            zone_id = feature['properties'].get('zone_id', f'Zone_{i}')
            area = feature['properties'].get('area_km2', 0)
            color = colors(i / max(1, n_zones - 1))

            if geom.geom_type == 'Polygon':
                x, y = geom.exterior.xy
                ax.plot(x, y, linewidth=2, color=color, label=f'{zone_id} ({area:.1f} km²)')
                ax.fill(x, y, alpha=0.4, color=color)
            elif geom.geom_type == 'MultiPolygon':
                for j, poly in enumerate(geom.geoms):
                    x, y = poly.exterior.xy
                    label = f'{zone_id} ({area:.1f} km²)' if j == 0 else None
                    ax.plot(x, y, linewidth=2, color=color, label=label)
                    ax.fill(x, y, alpha=0.4, color=color)

        ax.set_title(f'Upper Truckee River - Parameter Zones ({n_zones} zones)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)

        param_zone_map = step_dir / "3.6_parameter_zones_map.png"
        plt.savefig(param_zone_map, dpi=200, bbox_inches='tight')
        plt.close()
        results["outputs"].append(str(param_zone_map))
        print(f"  ✓ 生成参数分区图 ({n_zones}个分区): {param_zone_map.name}")

    # 3.7 河道网络可视化
    if channel_geojson.exists():
        geojson_data = json.loads(channel_geojson.read_text(encoding='utf-8'))
        fig, ax = plt.subplots(figsize=(12, 10))

        # 绘制流量累计作为背景
        with rasterio.open(flow_acc_path) as src:
            flowacc = src.read(1)
            flowacc = np.where(np.isfinite(flowacc), flowacc, 0)
            flowacc_log = np.log10(flowacc + 1)
            extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
            im = ax.imshow(flowacc_log, cmap='Blues', extent=extent, alpha=0.6)
            plt.colorbar(im, ax=ax, label='Log10(Flow Accumulation + 1)', shrink=0.8)

        # 绘制河道网络
        from shapely.geometry import shape as shapely_shape
        for feature in geojson_data['features']:
            geom = shapely_shape(feature['geometry'])
            if geom.geom_type == 'LineString':
                x, y = geom.xy
                ax.plot(x, y, 'r-', linewidth=2, alpha=0.8)
            elif geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    x, y = line.xy
                    ax.plot(x, y, 'r-', linewidth=2, alpha=0.8)

        ax.set_title('Upper Truckee River - Channel Network', fontsize=14, fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, alpha=0.3)

        channel_map = step_dir / "3.7_channel_network_map.png"
        plt.savefig(channel_map, dpi=200, bbox_inches='tight')
        plt.close()
        results["outputs"].append(str(channel_map))
        print(f"  ✓ 生成河道网络图: {channel_map.name}")

    # 从partition_parameter_zones()生成的GeoJSON文件中加载子流域几何形状
    from shapely.geometry import shape as shapely_shape
    subbasin_geometries = {}

    # 加载parameter_subbasins.geojson
    param_subbasin_geojson = parameter_dir / "parameter_subbasins.geojson"
    if param_subbasin_geojson.exists():
        geojson_data = json.loads(param_subbasin_geojson.read_text(encoding='utf-8'))
        for feature in geojson_data['features']:
            sub_id = feature['properties'].get('subzone_id', feature['properties'].get('id'))
            if sub_id:
                geom = shapely_shape(feature['geometry'])
                subbasin_geometries[sub_id] = geom

    print(f"  ✓ 加载{len(subbasin_geometries)}个子流域几何形状")

    results["partition_outputs"] = partition_outputs
    results["delineation_cfg"] = delineation_cfg_updated
    results["intermediate_dir"] = intermediate_dir
    results["parameter_dir"] = parameter_dir
    results["subbasins"] = subbasins
    results["subbasin_geometries"] = subbasin_geometries

    print(f"第3步完成：生成{len(results['outputs'])}个输出文件")
    return results

# ============================================================================
# 第4步：河道断面提取
# ============================================================================

def step04_channel_cross_sections(
    dem_path: Path,
    parameter_dir: Path,
    output_dir: Path,
    spacing_m: float = 500.0,
    half_width_m: float = 150.0,
    n_points: int = 41,
) -> Dict[str, object]:
    """
    第4步：河道断面提取

    输入：
    - DEM
    - 参数河道GeoJSON

    输出：
    - 河道断面CSV文件（每条河道一个）
    - 断面统计汇总表
    """
    print("\n" + "="*80)
    print("第4步：河道断面提取")
    print("="*80)

    step_dir = output_dir / "step_04_cross_sections"
    step_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "step": "04_河道断面提取",
        "outputs": [],
    }

    channel_geojson = parameter_dir / "parameter_channels.geojson"
    if not channel_geojson.exists():
        print("  ⚠ 参数河道文件不存在，跳过断面提取")
        return results

    if not HAS_RASTERIO:
        print("  ⚠ rasterio不可用，跳过断面提取")
        return results

    # 读取河道
    with channel_geojson.open('r', encoding='utf-8') as f:
        channel_data = json.load(f)

    from shapely.geometry import LineString, shape

    print(f"  ⚙ 提取河道断面 (间距={spacing_m}m, 半宽={half_width_m}m)...")

    with rasterio.open(dem_path) as dem:
        transform = dem.transform
        crs = dem.crs

        for feature in channel_data.get('features', []):
            props = feature.get('properties', {})
            segment_id = props.get('segment_id') or props.get('subzone_id')
            if not segment_id:
                continue

            geom = shape(feature.get('geometry'))
            if not isinstance(geom, LineString):
                continue

            coords = list(geom.coords)
            if len(coords) < 2:
                continue

            # 计算累积距离
            cumulative = np.concatenate([
                [0.0],
                np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))
            ])

            if cumulative[-1] == 0:
                continue

            n_sections = max(1, int(cumulative[-1] // spacing_m))
            distances = np.linspace(0.0, cumulative[-1], n_sections)

            sections = []
            for dist in distances:
                idx = np.searchsorted(cumulative, dist, side='right') - 1
                idx = max(0, min(idx, len(coords) - 2))

                local_start = np.array(coords[idx])
                local_end = np.array(coords[idx + 1])
                local_frac = ((dist - cumulative[idx]) /
                             max(cumulative[idx + 1] - cumulative[idx], 1e-6))

                centre = local_start + (local_end - local_start) * local_frac

                # 计算法线方向
                vec = local_end - local_start
                length = np.linalg.norm(vec)
                if length > 0:
                    tangent = vec / length
                    normal = np.array([-tangent[1], tangent[0]])
                else:
                    normal = np.array([0.0, 0.0])

                # 断面端点
                start = centre - normal * half_width_m
                end = centre + normal * half_width_m

                # 采样高程
                xs = np.linspace(start[0], end[0], n_points)
                ys = np.linspace(start[1], end[1], n_points)
                elevations = []
                for x, y in zip(xs, ys):
                    sample = next(dem.sample([(x, y)]), [np.nan])
                    elevations.append(float(sample[0]))

                rel_dist = np.linspace(-1.0, 1.0, n_points) * half_width_m

                df = pd.DataFrame({
                    'x': xs,
                    'y': ys,
                    'distance_from_center_m': rel_dist,
                    'elevation_m': elevations,
                    'station_m': dist,
                    'segment_id': segment_id,
                })
                sections.append(df)

            if sections:
                profile_df = pd.concat(sections, ignore_index=True)
                output_path = step_dir / f"4.1_{segment_id}_cross_sections.csv"
                profile_df.to_csv(output_path, index=False)
                results["outputs"].append(str(output_path))
                print(f"  ✓ 提取{segment_id}断面: {len(sections)}个")

    print(f"第4步完成：生成{len(results['outputs'])}个断面文件")
    return results

# ============================================================================
# 第5步：雨量站分布图生成
# ============================================================================

def step05_rain_gauge_distribution(
    partition_outputs,
    intermediate_dir: Path,
    output_dir: Path,
    station_count: int = 10,
    rng_seed: int = 42,
) -> Dict[str, object]:
    """
    第5步：雨量站分布图生成

    输入：
    - 参数分区输出

    输出：
    - 雨量站位置GeoJSON
    - 泰森多边形GeoJSON
    - 雨量站分布图
    """
    print("\n" + "="*80)
    print("第5步：雨量站分布图生成")
    print("="*80)

    step_dir = output_dir / "step_05_rain_gauges"
    step_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "step": "05_雨量站分布图",
        "outputs": [],
        "station_count": station_count,
    }

    # 这一步只是准备，实际生成在第6步
    print(f"  ✓ 配置{station_count}个合成雨量站")
    print(f"  ✓ 随机种子: {rng_seed}")

    results["rng_seed"] = rng_seed

    print(f"第5步完成：雨量站配置完成")
    return results

# ============================================================================
# 第6-8步：雨量序列生成、泰森多边形计算、面雨量计算（合并）
# ============================================================================

def step06_to_08_precipitation_processing(
    partition_outputs,
    subbasins: Sequence[Subbasin],
    subbasin_geometries: Dict,
    intermediate_dir: Path,
    output_dir: Path,
    station_count: int = 10,
    rng_seed: int = 42,
    total_hours: int = 120,
) -> Dict[str, object]:
    """
    第6-8步：雨量处理（合并执行）
    - 第6步：雨量序列生成
    - 第7步：泰森多边形计算
    - 第8步：面雨量计算

    输入：
    - 参数分区输出
    - 子流域信息

    输出：
    - 雨量站时间序列CSV
    - 雨量站位置GeoJSON
    - 泰森多边形GeoJSON
    - 权重JSON
    - 参数子区面雨量CSV
    - 子流域面雨量CSV
    - 流域平均雨量CSV
    - 可视化图表
    """
    print("\n" + "="*80)
    print("第6-8步：雨量处理（序列生成+泰森多边形+面雨量计算）")
    print("="*80)

    step6_dir = output_dir / "step_06_rain_series"
    step7_dir = output_dir / "step_07_thiessen"
    step8_dir = output_dir / "step_08_areal_rainfall"
    for d in [step6_dir, step7_dir, step8_dir]:
        d.mkdir(parents=True, exist_ok=True)

    results = {
        "step": "06_07_08_雨量处理",
        "outputs": [],
    }

    # 使用传入的子流域几何
    parameter_geometries = subbasin_geometries
    parameter_to_zone = {sub.id: f"Zone_{sub.id}" for sub in subbasins}
    parameter_areas = {sub.id: sub.area_km2 for sub in subbasins}

    print(f"  ✓ 加载{len(parameter_geometries)}个参数子区")

    # 生成基础降雨序列（合成）
    timestamps = pd.date_range('2024-01-01', periods=total_hours, freq='h')

    # 简单的三角形暴雨过程
    peak_hour = 48 + 12  # 峰值在第60小时
    base_precip = np.zeros(total_hours)
    for i in range(total_hours):
        if i < peak_hour:
            base_precip[i] = (i / peak_hour) * 15.0  # 上升到15 mm/hr
        else:
            remaining = total_hours - peak_hour
            if remaining > 0:
                base_precip[i] = 15.0 * (1.0 - (i - peak_hour) / remaining)

    base_precip = np.maximum(base_precip, 0.0)
    base_series = pd.Series(base_precip, index=timestamps, name='precipitation_mm_per_hr')

    print(f"  ✓ 生成基础降雨序列：{total_hours}小时，总雨量{base_precip.sum():.1f}mm")

    # 使用HydroSIS的降雨生成功能
    from hydrosis.precipitation import generate_rain_gauge_inputs

    print(f"  ⚙ 生成{station_count}个合成雨量站...")
    rain_inputs = generate_rain_gauge_inputs(
        base_series,
        parameter_geometries,
        station_count=station_count,
        rng_seed=rng_seed,
        heterogeneity_strength=0.6,
        min_burst_events=2,
        max_burst_events=4,
    )

    # 保存雨量站输出
    gauge_paths = rain_inputs.write(
        intermediate_dir,
        gauges_filename="rain_gauge_forcing.csv",
        subbasin_filename="parameter_subbasin_areal_precipitation.csv",
        stations_geojson="rain_gauge_locations.geojson",
        thiessen_geojson="rain_gauge_thiessen_polygons.geojson",
        weights_json="rain_gauge_weights.json",
    )

    print(f"  ✓ 生成{len(rain_inputs.station_positions)}个雨量站")

    # 复制到step目录
    import shutil

    # 第6步：雨量序列
    gauge_csv = intermediate_dir / "rain_gauge_forcing.csv"
    if gauge_csv.exists():
        dest = step6_dir / "6.1_rain_gauge_forcing.csv"
        shutil.copy2(gauge_csv, dest)
        results["outputs"].append(str(dest))
        print(f"  ✓ 雨量站序列: {dest.name}")

    # 第7步：泰森多边形
    thiessen_geojson = intermediate_dir / "rain_gauge_thiessen_polygons.geojson"
    if thiessen_geojson.exists():
        dest = step7_dir / "7.1_thiessen_polygons.geojson"
        shutil.copy2(thiessen_geojson, dest)
        results["outputs"].append(str(dest))
        print(f"  ✓ 泰森多边形: {dest.name}")

    stations_geojson = intermediate_dir / "rain_gauge_locations.geojson"
    if stations_geojson.exists():
        dest = step7_dir / "7.2_gauge_locations.geojson"
        shutil.copy2(stations_geojson, dest)
        results["outputs"].append(str(dest))
        print(f"  ✓ 雨量站位置: {dest.name}")

    # 第8步：面雨量
    param_precip_csv = intermediate_dir / "parameter_subbasin_areal_precipitation.csv"
    if param_precip_csv.exists():
        dest = step8_dir / "8.1_parameter_areal_precipitation.csv"
        shutil.copy2(param_precip_csv, dest)
        results["outputs"].append(str(dest))
        print(f"  ✓ 参数区面雨量: {dest.name}")

    # 计算子流域面雨量
    from hydrosis.workflow.stages import aggregate_parameter_precipitation

    # RainGaugeInputs返回的是subbasin_series，直接使用
    subbasin_series = rain_inputs.subbasin_series

    subbasin_csv = step8_dir / "8.2_subbasin_areal_precipitation.csv"
    subbasin_series.to_csv(subbasin_csv)
    results["outputs"].append(str(subbasin_csv))
    print(f"  ✓ 子流域面雨量: {subbasin_csv.name}")

    # 同时保存到intermediate目录供后续使用
    subbasin_series.to_csv(intermediate_dir / "subbasin_areal_precipitation.csv")

    # 计算流域平均雨量（使用实际的subzone IDs）
    # 从subbasin_series的列中获取实际存在的ID
    actual_ids = set(subbasin_series.columns)

    # 使用partition_outputs中的subzone信息
    if partition_outputs and partition_outputs.subzone_summaries:
        area_lookup = {s.subzone_id: float(s.area_km2) for s in partition_outputs.subzone_summaries
                      if s.subzone_id in actual_ids}
    else:
        # 回退：使用base subbasins
        area_lookup = {sub.id: float(sub.area_km2) for sub in subbasins if sub.id in actual_ids}

    total_area = sum(area_lookup.values())
    weighted_series = sum(
        subbasin_series[sub_id] * area for sub_id, area in area_lookup.items()
    ) / total_area
    basin_series = pd.DataFrame(
        {'precipitation_mm_per_hr': weighted_series},
        index=subbasin_series.index
    )

    basin_csv = step8_dir / "8.3_basin_average_precipitation.csv"
    basin_series.to_csv(basin_csv)
    results["outputs"].append(str(basin_csv))
    print(f"  ✓ 流域平均雨量: {basin_csv.name}")

    # 可视化雨量站时间序列
    fig, ax = plt.subplots(figsize=(12, 4))
    station_df = rain_inputs.station_series
    for col in station_df.columns[:min(5, len(station_df.columns))]:  # 只显示前5个站
        ax.plot(station_df.index, station_df[col], label=col, linewidth=1.0)
    ax.set_xlabel('Time')
    ax.set_ylabel('Precipitation (mm/hr)')
    ax.set_title('Rain Gauge Time Series (Sample)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')

    ts_fig = step6_dir / "6.2_gauge_timeseries_sample.png"
    plt.savefig(ts_fig, dpi=200, bbox_inches='tight')
    plt.close()
    results["outputs"].append(str(ts_fig))
    print(f"  ✓ 雨量站时间序列图: {ts_fig.name}")

    # 可视化子流域面雨量时间序列
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.colormaps.get_cmap('tab10')
    for i, sub in enumerate(subbasins):
        if sub.id in subbasin_series.columns:
            ax.plot(subbasin_series.index, subbasin_series[sub.id],
                   label=f'{sub.id} ({sub.area_km2:.1f} km²)',
                   linewidth=1.5, color=colors(i))
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Precipitation (mm/hr)', fontsize=12)
    ax.set_title('Subbasin Areal Precipitation Time Series', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    subbasin_fig = step8_dir / "8.4_subbasin_precipitation_hyetograph.png"
    plt.savefig(subbasin_fig, dpi=200, bbox_inches='tight')
    plt.close()
    results["outputs"].append(str(subbasin_fig))
    print(f"  ✓ 子流域雨量过程图: {subbasin_fig.name}")

    # 可视化流域平均过程
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(basin_series.index, 0, basin_series['precipitation_mm_per_hr'],
                    alpha=0.3, label='Basin Average')
    ax.plot(basin_series.index, basin_series['precipitation_mm_per_hr'],
           linewidth=2, color='blue')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Precipitation (mm/hr)', fontsize=12)
    ax.set_title('Basin-Average Precipitation', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=10)

    basin_fig = step8_dir / "8.5_basin_precipitation_hyetograph.png"
    plt.savefig(basin_fig, dpi=200, bbox_inches='tight')
    plt.close()
    results["outputs"].append(str(basin_fig))
    print(f"  ✓ 流域雨量过程图: {basin_fig.name}")

    results["subbasin_series"] = subbasin_series
    results["basin_series"] = basin_series
    results["rain_inputs"] = rain_inputs

    print(f"第6-8步完成：生成{len(results['outputs'])}个输出文件")
    return results

# ============================================================================
# 第9-10步：水文模拟和水动力模拟（合并）
# ============================================================================

def step09_to_10_hydrologic_and_hydraulic_simulation(
    delineation_cfg: DelineationConfig,
    partition_outputs,
    subbasin_series: pd.DataFrame,
    subbasins: Sequence[Subbasin],
    output_dir: Path,
) -> Dict[str, object]:
    """
    第9-10步：水文模拟和水动力模拟
    - 第9步：产流模拟（HBV、SCS-CN）
    - 第10步：汇流路由模拟（Muskingum、动力波）

    输入：
    - 流域配置
    - 子流域面雨量

    输出：
    - 各子流域流量过程线
    - 出口断面流量过程线
    - 模拟结果统计表
    - 流量过程图
    """
    print("\n" + "="*80)
    print("第9-10步：水文水动力模拟")
    print("="*80)

    step9_dir = output_dir / "step_09_runoff"
    step10_dir = output_dir / "step_10_routing"
    for d in [step9_dir, step10_dir]:
        d.mkdir(parents=True, exist_ok=True)

    results = {
        "step": "09_10_水文水动力模拟",
        "outputs": [],
    }

    # 配置产流模型（ID必须与parameters中的runoff_model匹配）
    runoff_models = [
        RunoffModelConfig(
            id="hbv",  # 与parameter_zones中的runoff_model键匹配
            model_type="hbv",
            parameters={
                "TT": 0.0,
                "CFMAX": 3.5,
                "CFR": 0.05,
                "CWH": 0.1,
                "FC": 250.0,
                "LP": 0.7,
                "BETA": 2.0,
                "K0": 0.05,
                "K1": 0.01,
                "K2": 0.001,
                "PERC": 1.5,
                "UZL": 5.0,
                "MAXBAS": 3.0,
            }
        ),
        RunoffModelConfig(
            id="scs_curve_number",
            model_type="scs_curve_number",
            parameters={
                "curve_number": 75.0,
                "initial_abstraction_ratio": 0.2,
            }
        ),
    ]

    # 配置汇流模型（ID必须与parameters中的routing_model匹配）
    routing_models = [
        RoutingModelConfig(
            id="muskingum",  # 与parameter_zones中的routing_model键匹配
            model_type="muskingum",
            parameters={
                "K": 10.0,
                "x": 0.2,
                "time_step": 1.0,
            }
        ),
    ]

    # 构建zone到subzones的映射
    zone_to_subzones = {}
    for subzone in partition_outputs.subzone_summaries:
        zone_id = subzone.zone_id
        if zone_id not in zone_to_subzones:
            zone_to_subzones[zone_id] = []
        zone_to_subzones[zone_id].append(subzone.subzone_id)

    # 创建修正后的parameter_zones，使用实际的subzone IDs
    # 关键：parameter zones提供models和parameters，subbasins只是计算单元
    updated_parameter_zones = []

    # 创建subzone的downstream映射，用于找到每个zone的outlet subzone
    subzone_downstream = {sz.subzone_id: sz.downstream_subzone_id
                         for sz in partition_outputs.subzone_summaries}

    for zone in partition_outputs.parameter_zones:
        # 获取该zone下的所有subzone IDs
        subzone_ids = zone_to_subzones.get(zone.id, [])
        if not subzone_ids:
            continue

        # 找到该zone的outlet subzone（下游不在本zone内的subzone，即控制点位置）
        # 这是监测数据的位置
        outlet_subzone = None
        for sz_id in subzone_ids:
            downstream_id = subzone_downstream.get(sz_id)
            # 如果downstream不在本zone内，或为None，则为outlet
            if downstream_id is None or downstream_id not in subzone_ids:
                outlet_subzone = sz_id
                break

        # 如果没找到（理论上不应该），使用第一个
        if outlet_subzone is None:
            outlet_subzone = subzone_ids[0]

        updated_zone = ParameterZoneConfig(
            id=zone.id,
            description=zone.description,
            control_points=[outlet_subzone],  # 使用zone的outlet subzone作为控制点
            parameters=zone.parameters,  # 包含runoff_model和routing_model的参数
            explicit_subbasins=subzone_ids,  # 明确列出该zone的所有subzones
        )
        updated_parameter_zones.append(updated_zone)

    # 从partition_outputs创建包含183个subzones的delineation配置
    # 关键修正：subzones只是计算单元，不应有自己的模型参数
    # 模型参数应该来自它们所属的parameter zone
    subzone_list = []

    for subzone in partition_outputs.subzone_summaries:
        # Subbasins只包含基本信息：id, area, downstream
        # 不包含runoff_model和routing_model - 这些来自parameter zones
        subzone_obj = {
            'id': subzone.subzone_id,
            'area_km2': subzone.area_km2,
            'downstream': subzone.downstream_subzone_id,
            'parameters': {},  # 空参数字典 - 参数来自zone
        }
        subzone_list.append(subzone_obj)

    # 创建新的delineation配置，使用183个subzones作为subbasins
    simulation_delineation_cfg = DelineationConfig(
        dem_path=delineation_cfg.dem_path,
        pour_points_path=delineation_cfg.pour_points_path,
        flow_direction_path=delineation_cfg.flow_direction_path,
        flow_accumulation_path=delineation_cfg.flow_accumulation_path,
        accumulation_threshold=delineation_cfg.accumulation_threshold,
        intermediate_directory=delineation_cfg.intermediate_directory,
        parameter_directory=delineation_cfg.parameter_directory,
        precomputed_subbasins=subzone_list,
    )

    # 配置IO
    io_config = IOConfig(
        results_directory=output_dir / "workflow_results",
        precipitation=output_dir / "intermediate" / "subbasin_areal_precipitation.csv",
    )

    # 配置评估指标
    evaluation_config = EvaluationConfig(
        metrics=["rmse", "mae", "nse", "pbias"],
    )

    # 构建模型配置
    model_config = ModelConfig(
        delineation=simulation_delineation_cfg,
        runoff_models=runoff_models,
        routing_models=routing_models,
        parameter_zones=updated_parameter_zones,
        io=io_config,
        evaluation=evaluation_config,
    )

    print(f"  ⚙ 配置完成（正确的概念模型）：")
    print(f"     - 产流模型库: {len(runoff_models)}个")
    print(f"     - 汇流模型库: {len(routing_models)}个")
    print(f"     - 参数区（率定单元）: {len(updated_parameter_zones)}个 [提供参数]")
    print(f"     - 子流域（计算单元）: {len(subzone_list)}个 [用于降水和产汇流计算]")
    print(f"  ℹ 概念：{len(subzone_list)}个子流域分组在{len(updated_parameter_zones)}个参数区下，参数区提供模型参数")

    # 准备forcing数据（使用实际的subbasin_series列）
    # subbasin_series包含所有subzone的降水数据
    forcing = {col: subbasin_series[col].tolist() for col in subbasin_series.columns}

    # 生成合成观测数据（用于演示）
    synthetic_obs = np.concatenate([
        np.zeros(48),
        np.linspace(0, 15, 24),
        15 * np.exp(-np.linspace(0, 3, 48)),
    ])
    # 找到outlet subzone（downstream为None的）
    outlet_subzone = None
    for sz in partition_outputs.subzone_summaries:
        if sz.downstream_subzone_id is None or sz.downstream_subzone_id == "":
            outlet_subzone = sz.subzone_id
            break
    if outlet_subzone is None:
        outlet_subzone = subzone_list[0]['id']  # 后备方案
    observations = {outlet_subzone: list(synthetic_obs)}

    # 运行模拟
    print(f"  ⚙ 运行水文水动力模拟...")
    workflow_result = run_workflow(
        model_config,
        forcing,
        observations=observations,
        persist_outputs=True,
    )

    print(f"  ✓ 模拟完成")

    # 获取结果
    baseline = workflow_result.baseline
    aggregated = baseline.aggregated

    # 保存模拟结果
    result_df = pd.DataFrame(aggregated)
    result_csv = step10_dir / "10.1_discharge_timeseries.csv"
    result_df.to_csv(result_csv)
    results["outputs"].append(str(result_csv))
    print(f"  ✓ 保存流量时间序列: {result_csv.name}")

    # 计算统计
    stats = []
    for sub_id, discharge in aggregated.items():
        stats.append({
            'Subbasin_ID': sub_id,
            'Peak_Discharge_m3s': float(np.max(discharge)),
            'Total_Volume_m3': float(np.sum(discharge) * 3600.0),
            'Mean_Discharge_m3s': float(np.mean(discharge)),
        })

    stats_df = pd.DataFrame(stats)
    stats_csv = step10_dir / "10.2_discharge_statistics.csv"
    stats_df.to_csv(stats_csv, index=False)
    results["outputs"].append(str(stats_csv))
    print(f"  ✓ 保存统计表: {stats_csv.name}")

    # 绘制流量过程线
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 上图：主要子流域流量
    timesteps = np.arange(len(next(iter(aggregated.values()))))
    for sub_id, discharge in list(aggregated.items())[:4]:  # 只显示前4个
        ax1.plot(timesteps, discharge, label=f'{sub_id}', linewidth=1.5)
    ax1.set_ylabel('Discharge (m³/s)')
    ax1.set_title('Subbasin Discharge Hydrographs')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right')

    # 下图：出口流量
    if outlet_subzone in aggregated:
        outlet_discharge = aggregated[outlet_subzone]
        ax2.fill_between(timesteps, 0, outlet_discharge, alpha=0.3, label='Simulated')
        ax2.plot(timesteps, outlet_discharge, linewidth=2, color='blue', label='Simulated')
        if outlet_subzone in observations:
            ax2.plot(timesteps, observations[outlet_subzone], 'r--',
                    linewidth=2, label='Observed (Synthetic)')

    ax2.set_xlabel('Time Step (hours)')
    ax2.set_ylabel('Discharge (m³/s)')
    ax2.set_title(f'Outlet Discharge ({outlet_subzone})')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right')

    fig.tight_layout()
    fig_file = step10_dir / "10.3_discharge_hydrographs.png"
    plt.savefig(fig_file, dpi=200, bbox_inches='tight')
    plt.close()
    results["outputs"].append(str(fig_file))
    print(f"  ✓ 生成流量过程图: {fig_file.name}")

    # 评估指标
    if hasattr(baseline, 'evaluation') and baseline.evaluation:
        eval_df = pd.DataFrame([
            {'Metric': k, 'Value': v}
            for k, v in baseline.evaluation.items()
        ])
        eval_csv = step10_dir / "10.4_evaluation_metrics.csv"
        eval_df.to_csv(eval_csv, index=False)
        results["outputs"].append(str(eval_csv))
        print(f"  ✓ 评估指标: {eval_csv.name}")

        for metric, value in baseline.evaluation.items():
            print(f"     - {metric.upper()}: {value:.4f}")

    results["workflow_result"] = workflow_result
    results["peak_discharge"] = float(np.max(list(aggregated.values())[0])) if aggregated else 0.0

    print(f"第9-10步完成：生成{len(results['outputs'])}个输出文件")
    return results

# ============================================================================
# 第11步：结果报告
# ============================================================================

def step11_final_report(
    all_results: Dict[str, object],
    output_dir: Path,
) -> Dict[str, object]:
    """
    第11步：生成完整的结果报告

    输入：
    - 所有前面步骤的结果

    输出：
    - 工作流总结报告（Markdown）
    - 数据文件索引
    - 可视化汇总
    """
    print("\n" + "="*80)
    print("第11步：生成结果报告")
    print("="*80)

    step_dir = output_dir / "step_11_final_report"
    step_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "step": "11_结果报告",
        "outputs": [],
    }

    # 生成Markdown报告
    report_lines = [
        "# Upper Truckee River 完整11步水文建模工作流报告",
        "",
        f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 工作流概览",
        "",
        "本报告展示了从DEM处理到水动力模拟的完整水文建模过程。",
        "",
        "### 工作流步骤",
        "",
    ]

    # 汇总所有输出
    total_outputs = 0
    for step_name, step_result in all_results.items():
        if isinstance(step_result, dict) and 'outputs' in step_result:
            step_num = step_result.get('step', step_name)
            outputs = step_result['outputs']
            total_outputs += len(outputs)

            report_lines.append(f"#### {step_num}")
            report_lines.append(f"- 输出文件数: {len(outputs)}")
            if outputs:
                report_lines.append("- 主要输出:")
                for output in outputs[:5]:  # 只列出前5个
                    report_lines.append(f"  - `{Path(output).name}`")
                if len(outputs) > 5:
                    report_lines.append(f"  - ... 及其他{len(outputs) - 5}个文件")
            report_lines.append("")

    report_lines.extend([
        "",
        f"**总输出文件数**: {total_outputs}",
        "",
        "## 关键结果",
        "",
    ])

    # 添加关键结果
    if 'step09_10' in all_results:
        sim_result = all_results['step09_10']
        if 'peak_discharge' in sim_result:
            report_lines.append(f"- **峰值流量**: {sim_result['peak_discharge']:.2f} m³/s")

    if 'step06_to_08' in all_results:
        precip_result = all_results['step06_to_08']
        if 'basin_series' in precip_result:
            basin_series = precip_result['basin_series']
            total_precip = (basin_series['precipitation_mm_per_hr'].sum())
            report_lines.append(f"- **总降雨量**: {total_precip:.1f} mm")

    if 'step03' in all_results:
        partition_result = all_results['step03']
        if 'partition_outputs' in partition_result:
            partition_outputs = partition_result['partition_outputs']
            n_subzones = len(partition_outputs.subzone_summaries)
            n_zones = len(partition_outputs.parameter_zones)
            report_lines.append(f"- **参数区数**: {n_zones}")
            report_lines.append(f"- **参数子区数**: {n_subzones}")

    report_lines.extend([
        "",
        "## 文件组织结构",
        "",
        "```",
        "results/upper_truckee_complete_11steps/",
        "├── step_01_dem_processing/          # DEM和地形分析",
        "├── step_02_pour_points/              # 汇水点",
        "├── step_03_zones_subbasins/          # 分区和子流域",
        "├── step_04_cross_sections/           # 河道断面",
        "├── step_05_rain_gauges/              # 雨量站配置",
        "├── step_06_rain_series/              # 雨量序列",
        "├── step_07_thiessen/                 # 泰森多边形",
        "├── step_08_areal_rainfall/           # 面雨量",
        "├── step_09_runoff/                   # 产流模拟",
        "├── step_10_routing/                  # 汇流模拟",
        "├── step_11_final_report/             # 本报告",
        "├── intermediate/                     # 中间文件",
        "└── parameters/                       # 参数文件",
        "```",
        "",
        "## 使用说明",
        "",
        "1. 所有可视化图片为PNG格式，可直接查看",
        "2. 所有数据表为CSV格式，可用Excel或Python读取",
        "3. 所有空间数据为GeoJSON格式，可用QGIS或GIS软件查看",
        "",
        "---",
        "",
        "*本报告由HydroSIS自动生成*",
    ])

    report_md = step_dir / "11.1_workflow_report.md"
    report_md.write_text('\n'.join(report_lines), encoding='utf-8')
    results["outputs"].append(str(report_md))
    print(f"  ✓ 生成工作流报告: {report_md.name}")

    # 生成文件索引
    index_data = []
    for step_name, step_result in all_results.items():
        if isinstance(step_result, dict) and 'outputs' in step_result:
            step_num = step_result.get('step', step_name)
            for output in step_result['outputs']:
                output_path = Path(output)
                index_data.append({
                    'Step': step_num,
                    'File': output_path.name,
                    'Path': str(output_path.relative_to(output_dir)),
                    'Type': output_path.suffix[1:].upper() if output_path.suffix else 'N/A',
                })

    index_df = pd.DataFrame(index_data)
    index_csv = step_dir / "11.2_file_index.csv"
    index_df.to_csv(index_csv, index=False)
    results["outputs"].append(str(index_csv))
    print(f"  ✓ 生成文件索引: {index_csv.name}")

    print(f"第11步完成：生成{len(results['outputs'])}个输出文件")
    print("")
    print("="*80)
    print(f"✅ 全部11个步骤完成！总输出文件数: {total_outputs}")
    print(f"📁 结果目录: {output_dir}")
    print(f"📄 工作流报告: {report_md}")
    print("="*80)

    return results

# ============================================================================
# 主程序
# ============================================================================

def main():
    """运行完整的11步工作流"""
    print("\n")
    print("=" * 80)
    print("Upper Truckee River 完整11步水文建模工作流")
    print("=" * 80)
    print()

    # 输入数据路径
    dem_dir = Path("data/Upper_Truckee_River/terrain/UpTruckeeRv_S10_NED_30m/00")
    dem_path = dem_dir / "elevation.tif"
    flow_dir_path = dem_dir / "flowdir.tif"
    flow_acc_path = dem_dir / "flowaccum.tif"

    # 输出目录
    output_root = Path("results/upper_truckee_complete_11steps")
    output_root.mkdir(parents=True, exist_ok=True)

    # 检查输入
    for path in [dem_path, flow_dir_path, flow_acc_path]:
        if not path.exists():
            print(f"错误：输入文件不存在: {path}")
            return 1

    print(f"✓ 输入数据检查完成")
    print(f"✓ 输出目录: {output_root}")
    print()

    all_results = {}

    try:
        # 第1步：DEM处理
        result1 = step01_dem_processing(
            dem_path, flow_dir_path, flow_acc_path, output_root
        )
        all_results['step01'] = result1

        # 第2步：汇水点生成（6个：3个干流 + 3个支流，Pfafstetter编码）
        result2 = step02_pour_point_generation(
            flow_dir_path, flow_acc_path, output_root,
            main_stream_count=3
        )
        all_results['step02'] = result2
        pour_points_path = result2['pour_points_path']

        # 第3步：参数分区和子流域划分
        result3 = step03_parameter_zones_and_subbasins(
            dem_path, flow_dir_path, flow_acc_path, pour_points_path, output_root
        )
        all_results['step03'] = result3
        partition_outputs = result3['partition_outputs']
        delineation_cfg = result3['delineation_cfg']
        intermediate_dir = result3['intermediate_dir']
        parameter_dir = result3['parameter_dir']

        # 获取子流域列表
        subbasins = delineation_cfg.to_subbasins()
        print(f"\n✓ 加载{len(subbasins)}个子流域")

        # 第4步：河道断面提取
        result4 = step04_channel_cross_sections(
            dem_path, parameter_dir, output_root
        )
        all_results['step04'] = result4

        # 第5步：雨量站配置
        result5 = step05_rain_gauge_distribution(
            partition_outputs, intermediate_dir, output_root, station_count=10
        )
        all_results['step05'] = result5

        # 第6-8步：雨量处理
        result6_to_8 = step06_to_08_precipitation_processing(
            partition_outputs, subbasins, result3['subbasin_geometries'],
            intermediate_dir, output_root,
            station_count=10, rng_seed=42, total_hours=120
        )
        all_results['step06_to_08'] = result6_to_8
        subbasin_series = result6_to_8['subbasin_series']

        # 第9-10步：水文水动力模拟
        result9_to_10 = step09_to_10_hydrologic_and_hydraulic_simulation(
            delineation_cfg, partition_outputs, subbasin_series, subbasins, output_root
        )
        all_results['step09_10'] = result9_to_10

        # 第11步：结果报告
        result11 = step11_final_report(all_results, output_root)
        all_results['step11'] = result11

        return 0

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
