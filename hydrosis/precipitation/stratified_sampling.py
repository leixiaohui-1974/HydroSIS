"""分层采样方法生成雨量站，确保在各分区均匀分布"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry


def stratified_station_sampling(
    zone_geometries: Dict[str, BaseGeometry],
    total_stations: int,
    zone_ids: Optional[Sequence[str]] = None,
    min_stations_per_zone: int = 3,
    allocation_method: str = "proportional",  # "proportional" or "equal"
    rng: Optional[np.random.Generator] = None,
    max_attempts_per_station: int = 10000,
) -> Dict[str, Point]:
    """
    使用分层采样在各分区均匀分布雨量站

    Args:
        zone_geometries: 分区几何体字典 {zone_id: geometry}
        total_stations: 总雨量站数量
        zone_ids: 要包含的分区ID列表（None表示所有分区）
        min_stations_per_zone: 每个分区最少雨量站数
        allocation_method: 分配方法
            - "proportional": 按面积比例分配
            - "equal": 均等分配
        rng: 随机数生成器
        max_attempts_per_station: 每个站点的最大采样尝试次数

    Returns:
        雨量站位置字典 {station_id: Point}
    """
    if rng is None:
        rng = np.random.default_rng()

    if not zone_geometries:
        raise ValueError("至少需要提供一个分区")

    # 确定要使用的分区
    if zone_ids is None:
        zone_ids = list(zone_geometries.keys())
    else:
        zone_ids = [zid for zid in zone_ids if zid in zone_geometries]

    if not zone_ids:
        raise ValueError("没有有效的分区")

    n_zones = len(zone_ids)

    # 检查总站点数是否足够
    min_total = n_zones * min_stations_per_zone
    if total_stations < min_total:
        raise ValueError(
            f"总雨量站数({total_stations})不足以满足每个分区的最小数量要求 "
            f"({n_zones}个分区 × {min_stations_per_zone}站/分区 = {min_total}站)"
        )

    # 计算每个分区的面积
    zone_areas = {}
    for zid in zone_ids:
        geom = zone_geometries[zid]
        zone_areas[zid] = geom.area

    total_area = sum(zone_areas.values())

    # 分配雨量站到各分区
    zone_allocations = {}

    if allocation_method == "equal":
        # 均等分配
        base_count = total_stations // n_zones
        remainder = total_stations % n_zones
        for i, zid in enumerate(zone_ids):
            zone_allocations[zid] = base_count + (1 if i < remainder else 0)

    elif allocation_method == "proportional":
        # 按面积比例分配
        # 首先给每个分区分配最小数量
        for zid in zone_ids:
            zone_allocations[zid] = min_stations_per_zone

        remaining = total_stations - min_total

        # 剩余的按面积比例分配
        if remaining > 0 and total_area > 0:
            # 计算每个分区应得的额外站点数
            for zid in zone_ids:
                proportion = zone_areas[zid] / total_area
                extra = int(round(remaining * proportion))
                zone_allocations[zid] += extra

            # 调整以确保总数正确
            current_total = sum(zone_allocations.values())
            diff = total_stations - current_total

            # 如果有差异，按面积从大到小调整
            if diff != 0:
                sorted_zones = sorted(zone_ids, key=lambda z: zone_areas[z], reverse=True)
                for i in range(abs(diff)):
                    zid = sorted_zones[i % len(sorted_zones)]
                    zone_allocations[zid] += 1 if diff > 0 else -1

    else:
        raise ValueError(f"未知的分配方法: {allocation_method}")

    # 验证分配结果
    for zid, count in zone_allocations.items():
        if count < min_stations_per_zone:
            zone_allocations[zid] = min_stations_per_zone

    # 调整总数
    current_total = sum(zone_allocations.values())
    if current_total != total_stations:
        diff = total_stations - current_total
        # 从最大的分区调整
        sorted_zones = sorted(zone_ids, key=lambda z: zone_areas[z], reverse=True)
        if diff > 0:
            for i in range(diff):
                zone_allocations[sorted_zones[i % len(sorted_zones)]] += 1
        else:
            for i in range(abs(diff)):
                zid = sorted_zones[i % len(sorted_zones)]
                if zone_allocations[zid] > min_stations_per_zone:
                    zone_allocations[zid] -= 1

    print(f"  ✓ 雨量站分配方案 (方法={allocation_method}):")
    for zid in zone_ids:
        area_pct = (zone_areas[zid] / total_area * 100) if total_area > 0 else 0
        print(f"    - Zone {zid}: {zone_allocations[zid]}个站点 "
              f"(面积={zone_areas[zid]/1e6:.2f} km², {area_pct:.1f}%)")

    # 在每个分区内采样雨量站
    station_positions = {}
    station_counter = 1

    for zid in zone_ids:
        geom = zone_geometries[zid]
        n_stations = zone_allocations[zid]

        # 获取分区边界
        bounds = geom.bounds  # (minx, miny, maxx, maxy)
        minx, miny, maxx, maxy = bounds
        extent_x = maxx - minx
        extent_y = maxy - miny

        if extent_x <= 0 or extent_y <= 0:
            print(f"    ⚠ Zone {zid} 边界无效，跳过")
            continue

        # 在分区内随机采样点
        for _ in range(n_stations):
            attempts = 0
            while attempts < max_attempts_per_station:
                attempts += 1
                # 在边界框内随机采样
                x = rng.uniform(minx, maxx)
                y = rng.uniform(miny, maxy)
                candidate = Point(x, y)

                # 检查点是否在分区内
                if geom.covers(candidate):
                    station_id = f"S{station_counter:03d}"
                    station_positions[station_id] = candidate
                    station_counter += 1
                    break
            else:
                print(f"    ⚠ Zone {zid}: 无法找到有效位置（尝试{attempts}次）")

    return station_positions


def generate_stratified_station_ids(
    zone_allocations: Dict[str, int]
) -> Dict[str, List[str]]:
    """
    生成分层的雨量站ID

    Args:
        zone_allocations: 每个分区的雨量站数量 {zone_id: count}

    Returns:
        每个分区的站点ID列表 {zone_id: [station_ids]}
    """
    zone_station_ids = {}
    global_counter = 1

    for zone_id, count in zone_allocations.items():
        station_ids = []
        for _ in range(count):
            station_ids.append(f"S{global_counter:03d}")
            global_counter += 1
        zone_station_ids[zone_id] = station_ids

    return zone_station_ids


__all__ = [
    "stratified_station_sampling",
    "generate_stratified_station_ids",
]
