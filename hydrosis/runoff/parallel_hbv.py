"""HBV模型并行化模块

提供多分区HBV模拟的并行化支持，提升大规模流域的计算效率

用法:
    from hydrosis.runoff.parallel_hbv import run_hbv_parallel

    results = run_hbv_parallel(
        zones_data,
        precipitation_data,
        hbv_parameters,
        max_workers=4
    )
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass

from ..model import Subbasin
from .hbv import HBVRunoff


@dataclass
class ParallelHBVConfig:
    """并行HBV配置"""
    max_workers: int = 4  # 最大并行worker数
    chunk_size: int = 100  # 子流域分组大小
    use_multiprocessing: bool = True  # 使用多进程(vs多线程)
    show_progress: bool = True  # 显示进度


def _run_single_zone_hbv(args):
    """运行单个分区的HBV模拟 (用于并行化)

    Args:
        args: (zone_id, precipitation, area_km2, hbv_params)

    Returns:
        (zone_id, result_dict)
    """
    zone_id, precipitation, area_km2, hbv_params = args

    # 创建Subbasin对象
    subbasin = Subbasin(
        id=str(zone_id),
        area_km2=area_km2,
        downstream=None
    )

    # 创建HBV模型
    model = HBVRunoff(parameters=hbv_params)

    # 运行模拟
    simulated_runoff = np.array(model.simulate(subbasin, precipitation.tolist()))

    # 计算统计指标
    total_precip_mm = precipitation.sum() * 1.0
    total_runoff_mm = simulated_runoff.sum() * 3600 / (area_km2 * 1e6) * 1000
    runoff_coefficient = total_runoff_mm / total_precip_mm if total_precip_mm > 0 else 0

    result = {
        'zone_id': zone_id,
        'runoff_coefficient': runoff_coefficient,
        'total_precip_mm': total_precip_mm,
        'total_runoff_mm': total_runoff_mm,
        'peak_runoff_m3s': simulated_runoff.max(),
        'mean_runoff_m3s': simulated_runoff.mean(),
        'runoff_series': simulated_runoff
    }

    return zone_id, result


def run_hbv_parallel(
    zones: List[Dict],
    precipitation_data: Dict[int, np.ndarray],
    hbv_params: Dict,
    config: Optional[ParallelHBVConfig] = None
) -> Dict[int, Dict]:
    """并行运行多个分区的HBV模拟

    Args:
        zones: 分区列表 [{'zone_id': 1, 'area_km2': 100}, ...]
        precipitation_data: 分区降雨字典 {zone_id: precipitation_array}
        hbv_params: HBV参数字典
        config: 并行配置

    Returns:
        结果字典 {zone_id: result_dict}
    """
    if config is None:
        config = ParallelHBVConfig()

    # 准备参数
    tasks = []
    for zone in zones:
        zone_id = zone['zone_id']
        area_km2 = zone['area_km2']
        precipitation = precipitation_data.get(zone_id)

        if precipitation is None:
            print(f"  ⚠️  跳过分区 {zone_id}: 未找到降雨数据")
            continue

        tasks.append((zone_id, precipitation, area_km2, hbv_params))

    results = {}

    if config.max_workers == 1 or len(tasks) == 1:
        # 串行执行
        print(f"  ⚙ 串行模拟 {len(tasks)} 个分区...")
        for task in tasks:
            zone_id, result = _run_single_zone_hbv(task)
            results[zone_id] = result
            if config.show_progress:
                print(f"    ✓ 分区 {zone_id} 完成")

    else:
        # 并行执行
        print(f"  ⚙ 并行模拟 {len(tasks)} 个分区 (workers={config.max_workers})...")

        with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
            # 提交所有任务
            futures = {executor.submit(_run_single_zone_hbv, task): task[0]
                      for task in tasks}

            # 收集结果
            completed = 0
            for future in as_completed(futures):
                zone_id = futures[future]
                try:
                    result_zone_id, result = future.result()
                    results[result_zone_id] = result
                    completed += 1

                    if config.show_progress:
                        print(f"    ✓ 分区 {zone_id} 完成 ({completed}/{len(tasks)})")

                except Exception as e:
                    print(f"    ❌ 分区 {zone_id} 失败: {e}")

    print(f"  ✓ 并行模拟完成，成功 {len(results)}/{len(tasks)} 个分区")

    return results


def benchmark_parallel_performance(
    zones: List[Dict],
    precipitation_data: Dict[int, np.ndarray],
    hbv_params: Dict,
    worker_counts: List[int] = [1, 2, 4, 8]
) -> Dict[int, float]:
    """基准测试不同并行度的性能

    Args:
        zones: 分区列表
        precipitation_data: 降雨数据
        hbv_params: HBV参数
        worker_counts: 要测试的worker数量列表

    Returns:
        {worker_count: elapsed_time} 字典
    """
    import time

    results = {}

    for num_workers in worker_counts:
        print(f"\n测试 {num_workers} workers...")
        config = ParallelHBVConfig(
            max_workers=num_workers,
            show_progress=False
        )

        start_time = time.time()
        run_hbv_parallel(zones, precipitation_data, hbv_params, config)
        elapsed = time.time() - start_time

        results[num_workers] = elapsed
        print(f"  完成时间: {elapsed:.2f} 秒")

        # 计算加速比
        if num_workers > 1 and 1 in results:
            speedup = results[1] / elapsed
            efficiency = speedup / num_workers
            print(f"  加速比: {speedup:.2f}x")
            print(f"  并行效率: {efficiency:.2%}")

    return results
