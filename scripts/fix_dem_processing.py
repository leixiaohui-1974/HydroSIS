#!/usr/bin/env python3
"""
修复DEM处理 - 添加平坦区域处理

参考：
1. Barnes et al. (2014) - Priority-Flood处理flats
2. RichDEM文档关于ResolveFlats
"""

import numpy as np
import rasterio
import richdem as rd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def process_dem_with_flats(dem_path, output_dir):
    """
    完整的DEM处理流程，包括平坦区域处理
    
    流程:
    1. 加载DEM
    2. 填充坑洼 (FillDepressions)
    3. 解决平坦区域 (ResolveFlats) - 关键步骤！
    4. 计算流向
    5. 计算流量累积
    6. 提取河网
    """
    print("="*80)
    print("DEM处理（含平坦区域处理）")
    print("="*80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载DEM
    print("\n步骤1: 加载DEM")
    with rasterio.open(dem_path) as src:
        dem_data = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        
        print(f"  DEM尺寸: {dem_data.shape}")
        print(f"  DEM范围: [{np.nanmin(dem_data):.2f}, {np.nanmax(dem_data):.2f}]米")
        print(f"  NoData值: {src.nodata}")
    
    # 清理NoData - 将异常值替换为NaN
    nodata_value = src.nodata
    if nodata_value and abs(nodata_value) > 1e10:
        print(f"  清理NoData值: {nodata_value}")
        dem_data = np.where(np.abs(dem_data) > 1e10, np.nan, dem_data)
        nodata_value = -9999
    
    # 转换为RichDEM数组
    rd_dem = rd.rdarray(dem_data, no_data=nodata_value if nodata_value else -9999)
    rd_dem.geotransform = transform.to_gdal()
    
    print(f"  转换后DEM范围: [{np.nanmin(rd_dem):.2f}, {np.nanmax(rd_dem):.2f}]米")
    print(f"  有效像元: {np.sum(~np.isnan(dem_data))}")
    
    # 2. 填充坑洼
    print("\n步骤2: 填充坑洼")
    print("  算法: Priority-Flood (Zhou2016)")
    rd.FillDepressions(rd_dem, in_place=True)
    
    # 保存填充后的DEM
    filled_path = output_dir / "filled_dem.tif"
    with rasterio.open(filled_path, 'w', **profile) as dst:
        dst.write(rd_dem, 1)
    print(f"  ✅ 已保存: {filled_path}")
    
    # 3. 解决平坦区域 - 关键！
    print("\n步骤3: 解决平坦区域（Resolve Flats）")
    print("  这是关键步骤，会显著提升流量累积的准确性")
    
    # 方法1: 使用BreachDepressions（推荐）
    try:
        print("  方法1: BreachDepressions (Lindsay2016)...")
        rd_dem_breached = rd_dem.copy()
        rd.BreachDepressions(rd_dem_breached, in_place=True)
        print("  ✅ BreachDepressions完成")
        
        # 方法2: 再使用ResolveFlats处理剩余的平坦区域
        try:
            print("  方法2: ResolveFlats (Barnes2014)...")
            rd.ResolveFlats(rd_dem_breached, in_place=True)
            print("  ✅ ResolveFlats完成")
        except Exception as e:
            print(f"  ⚠️  ResolveFlats不可用: {e}")
        
        rd_dem = rd_dem_breached
        
    except Exception as e:
        print(f"  ⚠️  Breach/Resolve方法失败: {e}")
        print("  使用填充后的DEM继续...")
    
    # 4. 计算流向
    print("\n步骤4: 计算流向")
    print("  算法: D8")
    
    # 使用FlowProportions获取更准确的流向信息
    flow_props = rd.FlowProportions(rd_dem, method='D8')
    
    # 5. 计算流量累积
    print("\n步骤5: 计算流量累积")
    print("  算法: D8 Flow Accumulation")
    
    flow_acc = rd.FlowAccumulation(rd_dem, method='D8')
    
    # 统计
    print(f"\n流量累积统计:")
    print(f"  最小值: {np.min(flow_acc):.0f}")
    print(f"  最大值: {np.max(flow_acc):.0f}")
    print(f"  平均值: {np.mean(flow_acc):.2f}")
    print(f"  中位数: {np.median(flow_acc):.2f}")
    
    # 检查是否有改善
    pixel_area = abs(transform[0] * transform[4])  # m²
    max_area_km2 = np.max(flow_acc) * pixel_area / 1e6
    total_area_km2 = np.sum(flow_acc > 0) * pixel_area / 1e6
    
    print(f"\n面积统计:")
    print(f"  最大控制面积: {max_area_km2:.2f} km²")
    print(f"  总流域面积: {total_area_km2:.2f} km²")
    print(f"  控制比例: {max_area_km2/total_area_km2*100:.1f}%")
    
    # 保存流量累积
    profile.update(dtype=rasterio.float32)
    flow_acc_path = output_dir / "flow_accumulation_fixed.tif"
    with rasterio.open(flow_acc_path, 'w', **profile) as dst:
        dst.write(flow_acc.astype(np.float32), 1)
    print(f"\n✅ 已保存: {flow_acc_path}")
    
    # 6. 提取河网
    print("\n步骤6: 提取河网")
    thresholds = [100, 500, 1000, 5000, 10000]
    
    for thresh in thresholds:
        count = np.sum(flow_acc >= thresh)
        if count > 0:
            print(f"  阈值{thresh:6d}: {count:8d} 像元")
    
    # 可视化对比
    print("\n步骤7: 生成对比可视化")
    visualize_comparison(dem_data, rd_dem, flow_acc, output_dir)
    
    print("\n" + "="*80)
    print("DEM处理完成")
    print("="*80)
    
    return {
        'flow_acc': flow_acc,
        'dem': rd_dem,
        'max_acc': float(np.max(flow_acc)),
        'max_area_km2': float(max_area_km2),
        'total_area_km2': float(total_area_km2)
    }

def visualize_comparison(original_dem, filled_dem, flow_acc, output_dir):
    """可视化对比"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 原始DEM
    ax = axes[0, 0]
    im = ax.imshow(original_dem, cmap='terrain')
    ax.set_title('Original DEM')
    plt.colorbar(im, ax=ax, label='Elevation (m)')
    
    # 填充后DEM
    ax = axes[0, 1]
    im = ax.imshow(filled_dem, cmap='terrain')
    ax.set_title('Filled DEM (after depression filling)')
    plt.colorbar(im, ax=ax, label='Elevation (m)')
    
    # 流量累积（线性）
    ax = axes[1, 0]
    im = ax.imshow(flow_acc, cmap='Blues', norm=matplotlib.colors.LogNorm(vmin=1, vmax=np.max(flow_acc)))
    ax.set_title('Flow Accumulation (log scale)')
    plt.colorbar(im, ax=ax, label='Accumulated cells (log)')
    
    # 河网提取
    ax = axes[1, 1]
    river_mask = flow_acc > 1000
    river_display = np.where(river_mask, flow_acc, np.nan)
    im = ax.imshow(river_display, cmap='Blues')
    ax.set_title('River Network (threshold > 1000)')
    plt.colorbar(im, ax=ax, label='Flow Accumulation')
    
    plt.tight_layout()
    output_path = output_dir / 'dem_processing_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ 可视化已保存: {output_path}")
    plt.close()

def main():
    # 使用Upper Truckee River DEM
    dem_path = "data/Upper_Truckee_River/terrain/UpTruckeeRv_S10_NED_30m/00/elevation.tif"
    output_dir = "results/dem_processing_fixed"
    
    if not Path(dem_path).exists():
        print(f"错误: DEM文件不存在: {dem_path}")
        return
    
    results = process_dem_with_flats(dem_path, output_dir)
    
    print(f"\n最终结果:")
    print(f"  最大累积数: {results['max_acc']:.0f} 像元")
    print(f"  最大控制面积: {results['max_area_km2']:.2f} km²")
    print(f"  总流域面积: {results['total_area_km2']:.2f} km²")
    
    # 检查改善
    if results['max_acc'] > 10000:
        print(f"\n✅ 流量累积显著改善！最大值达到 {results['max_acc']:.0f}")
    else:
        print(f"\n⚠️  流量累积仍然偏小，可能需要:")
        print(f"     1. 检查DEM质量和分辨率")
        print(f"     2. 尝试不同的平坦区域处理方法")
        print(f"     3. 检查流向算法参数")

if __name__ == "__main__":
    main()
