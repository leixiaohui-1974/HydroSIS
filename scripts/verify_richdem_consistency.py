#!/usr/bin/env python3
"""验证重新编译前后richdem结果一致性"""

import numpy as np
import rasterio
from pathlib import Path
import json

def compare_rasters(path1, path2, name):
    """对比两个栅格文件"""
    print(f"\n对比 {name}:")
    
    with rasterio.open(path1) as src1, rasterio.open(path2) as src2:
        data1 = src1.read(1)
        data2 = src2.read(1)
        
        # 基本统计
        print(f"  形状对比: {data1.shape} vs {data2.shape}")
        
        if data1.shape != data2.shape:
            print(f"  ❌ 形状不一致！")
            return False
        
        # 数据对比
        diff = np.abs(data1 - data2)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        # 忽略nodata值
        mask = (data1 != src1.nodata) & (data2 != src2.nodata)
        if np.any(mask):
            diff_valid = diff[mask]
            max_diff_valid = np.max(diff_valid)
            mean_diff_valid = np.mean(diff_valid)
            
            print(f"  最大差异: {max_diff_valid}")
            print(f"  平均差异: {mean_diff_valid}")
            print(f"  数据范围1: [{np.min(data1[mask]):.2f}, {np.max(data1[mask]):.2f}]")
            print(f"  数据范围2: [{np.min(data2[mask]):.2f}, {np.max(data2[mask]):.2f}]")
            
            # 判断一致性
            if max_diff_valid < 1e-5:
                print(f"  ✅ 完全一致！")
                return True
            elif max_diff_valid < 0.01:
                print(f"  ✅ 基本一致（浮点误差）")
                return True
            else:
                print(f"  ⚠️  存在差异")
                return False
        else:
            print(f"  ⚠️  无有效数据")
            return False

def main():
    print("="*80)
    print("RichDEM重新编译前后结果一致性验证")
    print("="*80)
    
    # 假设之前的结果在某个备份目录
    # 这里我们直接运行两次来对比
    
    # 运行terrain处理并对比
    import sys
    sys.path.insert(0, '/workspace')
    
    import richdem as rd
    
    # 测试DEM路径
    dem_path = "data/Upper_Truckee_River/terrain/UpTruckeeRv_S10_NED_30m/00/elevation.tif"
    
    print("\n测试1: 加载DEM并分析")
    with rasterio.open(dem_path) as src:
        dem_data = src.read(1)
        transform = src.transform
        profile = src.profile.copy()
        
        print(f"DEM尺寸: {dem_data.shape}")
        print(f"DEM范围: [{np.min(dem_data):.2f}, {np.max(dem_data):.2f}]米")
        print(f"有效像元数: {np.sum(~np.isnan(dem_data))}")
        
    print("\n测试2: RichDEM处理")
    rd_dem = rd.rdarray(dem_data, no_data=-9999)
    rd_dem.geotransform = transform.to_gdal()
    
    # 填充坑洼
    print("  执行FillDepressions...")
    rd.FillDepressions(rd_dem, in_place=True)
    
    # 流量累积
    print("  执行FlowAccumulation...")
    flow_acc = rd.FlowAccumulation(rd_dem, method='D8')
    
    print(f"\n流量累积统计:")
    print(f"  最小值: {np.min(flow_acc)}")
    print(f"  最大值: {np.max(flow_acc)}")
    print(f"  平均值: {np.mean(flow_acc):.2f}")
    print(f"  中位数: {np.median(flow_acc):.2f}")
    
    # 找出高累积区域
    high_acc = flow_acc > 1000
    num_high = np.sum(high_acc)
    print(f"\n高累积区域（>1000）: {num_high} 个像元")
    
    # 最高累积点
    max_idx = np.unravel_index(np.argmax(flow_acc), flow_acc.shape)
    max_val = flow_acc[max_idx]
    print(f"最大累积点: {max_idx}, 累积值: {max_val}")
    
    # 保存对比结果
    output_dir = Path("results/richdem_verification")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存流量累积
    profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(output_dir / "flow_acc_recompiled.tif", 'w', **profile) as dst:
        dst.write(flow_acc.astype(np.float32), 1)
    
    print(f"\n✅ 结果已保存到: {output_dir}")
    
    # 对比之前的结果
    prev_result = "results/enhanced_workflow_tests/01_最小测试-仅地形/outputs/terrain/flow_accumulation.tif"
    if Path(prev_result).exists():
        print(f"\n对比之前的结果...")
        is_consistent = compare_rasters(
            output_dir / "flow_acc_recompiled.tif",
            prev_result,
            "流量累积"
        )
        
        if is_consistent:
            print(f"\n✅ 重新编译后结果一致！")
        else:
            print(f"\n⚠️  结果存在差异，需要检查")
    
    print("\n" + "="*80)
    print("验证完成")
    print("="*80)

if __name__ == "__main__":
    main()
