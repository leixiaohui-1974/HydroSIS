#!/usr/bin/env python3
"""
深入研究RichDEM的流量累积算法

测试不同参数和方法的效果
"""

import numpy as np
import rasterio
import richdem as rd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def test_flow_accumulation_methods(dem_path, output_dir):
    """
    测试不同的流量累积方法
    """
    print("="*80)
    print("RichDEM流量累积方法研究")
    print("="*80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载DEM
    print("\n1. 加载DEM")
    with rasterio.open(dem_path) as src:
        dem_data = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        nodata = src.nodata
    
    # 清理NoData
    if nodata and abs(nodata) > 1e10:
        dem_data = np.where(np.abs(dem_data) > 1e10, np.nan, dem_data)
        nodata = -9999
    
    print(f"   DEM大小: {dem_data.shape}")
    print(f"   高程范围: [{np.nanmin(dem_data):.1f}, {np.nanmax(dem_data):.1f}] m")
    
    results = {}
    
    # 方法1: 仅填充坑洼
    print("\n" + "-"*80)
    print("方法1: FillDepressions")
    print("-"*80)
    rd_dem1 = rd.rdarray(dem_data.copy(), no_data=nodata)
    rd_dem1.geotransform = transform.to_gdal()
    
    rd.FillDepressions(rd_dem1, in_place=True)
    flow_acc1 = rd.FlowAccumulation(rd_dem1, method='D8')
    
    results['fill_only'] = {
        'max': float(np.max(flow_acc1)),
        'mean': float(np.mean(flow_acc1)),
        'method': 'FillDepressions only'
    }
    print(f"   最大累积: {results['fill_only']['max']:.0f}")
    
    # 保存
    profile.update(dtype=rasterio.float32)
    with rasterio.open(output_dir / 'flow_acc_fill_only.tif', 'w', **profile) as dst:
        dst.write(flow_acc1.astype(np.float32), 1)
    
    # 方法2: 填充 + Breach
    print("\n" + "-"*80)
    print("方法2: FillDepressions + BreachDepressions")
    print("-"*80)
    rd_dem2 = rd.rdarray(dem_data.copy(), no_data=nodata)
    rd_dem2.geotransform = transform.to_gdal()
    
    rd.FillDepressions(rd_dem2, in_place=True)
    rd.BreachDepressions(rd_dem2, in_place=True)
    flow_acc2 = rd.FlowAccumulation(rd_dem2, method='D8')
    
    results['fill_breach'] = {
        'max': float(np.max(flow_acc2)),
        'mean': float(np.mean(flow_acc2)),
        'method': 'FillDepressions + BreachDepressions'
    }
    print(f"   最大累积: {results['fill_breach']['max']:.0f}")
    
    with rasterio.open(output_dir / 'flow_acc_fill_breach.tif', 'w', **profile) as dst:
        dst.write(flow_acc2.astype(np.float32), 1)
    
    # 方法3: Breach + 填充（顺序反转）
    print("\n" + "-"*80)
    print("方法3: BreachDepressions + FillDepressions (reversed)")
    print("-"*80)
    rd_dem3 = rd.rdarray(dem_data.copy(), no_data=nodata)
    rd_dem3.geotransform = transform.to_gdal()
    
    rd.BreachDepressions(rd_dem3, in_place=True)
    rd.FillDepressions(rd_dem3, in_place=True)
    flow_acc3 = rd.FlowAccumulation(rd_dem3, method='D8')
    
    results['breach_fill'] = {
        'max': float(np.max(flow_acc3)),
        'mean': float(np.mean(flow_acc3)),
        'method': 'BreachDepressions + FillDepressions'
    }
    print(f"   最大累积: {results['breach_fill']['max']:.0f}")
    
    with rasterio.open(output_dir / 'flow_acc_breach_fill.tif', 'w', **profile) as dst:
        dst.write(flow_acc3.astype(np.float32), 1)
    
    # 方法4: 仅Breach
    print("\n" + "-"*80)
    print("方法4: BreachDepressions only")
    print("-"*80)
    rd_dem4 = rd.rdarray(dem_data.copy(), no_data=nodata)
    rd_dem4.geotransform = transform.to_gdal()
    
    rd.BreachDepressions(rd_dem4, in_place=True)
    flow_acc4 = rd.FlowAccumulation(rd_dem4, method='D8')
    
    results['breach_only'] = {
        'max': float(np.max(flow_acc4)),
        'mean': float(np.mean(flow_acc4)),
        'method': 'BreachDepressions only'
    }
    print(f"   最大累积: {results['breach_only']['max']:.0f}")
    
    with rasterio.open(output_dir / 'flow_acc_breach_only.tif', 'w', **profile) as dst:
        dst.write(flow_acc4.astype(np.float32), 1)
    
    # 方法5: 尝试ResolveFlats（如果可用）
    print("\n" + "-"*80)
    print("方法5: FillDepressions + ResolveFlats")
    print("-"*80)
    try:
        rd_dem5 = rd.rdarray(dem_data.copy(), no_data=nodata)
        rd_dem5.geotransform = transform.to_gdal()
        
        rd.FillDepressions(rd_dem5, in_place=True)
        rd.ResolveFlats(rd_dem5, in_place=True)
        flow_acc5 = rd.FlowAccumulation(rd_dem5, method='D8')
        
        results['fill_resolve'] = {
            'max': float(np.max(flow_acc5)),
            'mean': float(np.mean(flow_acc5)),
            'method': 'FillDepressions + ResolveFlats'
        }
        print(f"   最大累积: {results['fill_resolve']['max']:.0f}")
        
        with rasterio.open(output_dir / 'flow_acc_fill_resolve.tif', 'w', **profile) as dst:
            dst.write(flow_acc5.astype(np.float32), 1)
    except Exception as e:
        print(f"   ⚠️ ResolveFlats不可用: {e}")
    
    # 对比分析
    print("\n" + "="*80)
    print("对比分析")
    print("="*80)
    print(f"{'方法':<40} {'最大累积':<15} {'平均累积':<15}")
    print("-"*80)
    
    for key, data in sorted(results.items(), key=lambda x: x[1]['max'], reverse=True):
        print(f"{data['method']:<40} {data['max']:<15.0f} {data['mean']:<15.2f}")
    
    # 可视化对比
    visualize_comparison(flow_acc1, flow_acc2, flow_acc3, flow_acc4, results, output_dir)
    
    return results

def visualize_comparison(acc1, acc2, acc3, acc4, results, output_dir):
    """可视化不同方法的对比"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    methods = [
        (acc1, 'Method 1: Fill only', results['fill_only']['max']),
        (acc2, 'Method 2: Fill + Breach', results['fill_breach']['max']),
        (acc3, 'Method 3: Breach + Fill', results['breach_fill']['max']),
        (acc4, 'Method 4: Breach only', results['breach_only']['max'])
    ]
    
    for idx, (ax, (acc, title, max_val)) in enumerate(zip(axes.flat, methods)):
        # 使用对数尺度
        vmin = max(1, np.min(acc[acc > 0]))
        vmax = np.max(acc)
        
        im = ax.imshow(acc, cmap='Blues', 
                      norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
        ax.set_title(f'{title}\nMax: {max_val:.0f}', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Flow Accumulation (log scale)')
        ax.axis('off')
    
    plt.tight_layout()
    output_path = output_dir / 'flow_accumulation_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 对比图已保存: {output_path}")
    plt.close()

def main():
    dem_path = "data/Upper_Truckee_River/terrain/UpTruckeeRv_S10_NED_30m/00/elevation.tif"
    output_dir = "results/richdem_research"
    
    if not Path(dem_path).exists():
        print(f"错误: DEM文件不存在: {dem_path}")
        return
    
    results = test_flow_accumulation_methods(dem_path, output_dir)
    
    # 找出最佳方法
    best_method = max(results.items(), key=lambda x: x[1]['max'])
    
    print("\n" + "="*80)
    print("结论")
    print("="*80)
    print(f"✅ 最佳方法: {best_method[1]['method']}")
    print(f"   最大累积: {best_method[1]['max']:.0f} 像元")
    print(f"   平均累积: {best_method[1]['mean']:.2f} 像元")

if __name__ == "__main__":
    main()
