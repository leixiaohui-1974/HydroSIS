"""空间数据验证模块

提供流域空间数据的验证功能，包括：
- 流域边界完整性
- 河网拓扑一致性
- 空间关系验证
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from shapely.geometry import shape

from .base import ValidationCriteria, ValidationResult


@dataclass
class SpatialCriteria(ValidationCriteria):
    """空间验证标准"""
    # 面积检查
    min_area_km2: float = 1.0
    max_area_km2: float = 10000.0

    # 几何有效性
    allow_invalid_geometry: bool = False
    min_valid_geometry_ratio: float = 0.95

    # 拓扑一致性
    allow_gaps: bool = False
    allow_overlaps: bool = False
    overlap_tolerance_m: float = 1.0

    @classmethod
    def from_dict(cls, data: Dict) -> "SpatialCriteria":
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


def validate_basin_geometry(
    basin_features: List[Dict],
    criteria: Optional[SpatialCriteria] = None,
    step_name: str = "流域几何验证"
) -> ValidationResult:
    """验证流域几何数据

    Args:
        basin_features: GeoJSON features列表
        criteria: 验证标准
        step_name: 步骤名称

    Returns:
        验证结果
    """
    if criteria is None:
        criteria = SpatialCriteria()

    result = ValidationResult(step_name=step_name)

    if not basin_features:
        result.add_error("未找到流域要素")
        return result

    # 1. 几何有效性检查
    valid_count = 0
    invalid_geoms = []

    for i, feature in enumerate(basin_features):
        geom = shape(feature['geometry'])

        if not geom.is_valid:
            invalid_geoms.append(i)
            result.add_error(f"要素 {i}: 几何无效")
        else:
            valid_count += 1

        # 面积检查
        area_km2 = geom.area / 1e6  # 假设坐标为米
        if area_km2 < criteria.min_area_km2:
            result.add_warning(f"要素 {i}: 面积过小 ({area_km2:.2f} km²)")
        elif area_km2 > criteria.max_area_km2:
            result.add_warning(f"要素 {i}: 面积过大 ({area_km2:.2f} km²)")

    valid_ratio = valid_count / len(basin_features)
    result.metrics['valid_geometry_ratio'] = valid_ratio
    result.metrics['total_features'] = len(basin_features)
    result.metrics['valid_features'] = valid_count

    if valid_ratio < criteria.min_valid_geometry_ratio:
        result.add_error(
            f"有效几何比例过低: {valid_ratio:.2%} < {criteria.min_valid_geometry_ratio:.2%}"
        )

    return result


def validate_network_topology(
    network_features: List[Dict],
    criteria: Optional[SpatialCriteria] = None,
    step_name: str = "河网拓扑验证"
) -> ValidationResult:
    """验证河网拓扑一致性

    Args:
        network_features: 河网要素列表
        criteria: 验证标准
        step_name: 步骤名称

    Returns:
        验证结果
    """
    if criteria is None:
        criteria = SpatialCriteria()

    result = ValidationResult(step_name=step_name)

    # 提取上下游关系
    topology = {}
    for feature in network_features:
        props = feature['properties']
        feature_id = props.get('id')
        downstream = props.get('downstream')

        if feature_id is not None:
            topology[feature_id] = downstream

    # 检查拓扑一致性
    all_ids = set(topology.keys())
    downstream_ids = set(v for v in topology.values() if v is not None)

    # 1. 检查下游引用
    invalid_refs = downstream_ids - all_ids
    if invalid_refs:
        result.add_error(f"存在无效的下游引用: {invalid_refs}")

    # 2. 检查环路
    def has_cycle(node_id, visited, rec_stack):
        visited.add(node_id)
        rec_stack.add(node_id)

        downstream = topology.get(node_id)
        if downstream and downstream in topology:
            if downstream in rec_stack:
                return True
            if downstream not in visited:
                if has_cycle(downstream, visited, rec_stack):
                    return True

        rec_stack.remove(node_id)
        return False

    visited = set()
    for node_id in topology:
        if node_id not in visited:
            if has_cycle(node_id, visited, set()):
                result.add_error(f"拓扑结构存在环路,起点: {node_id}")
                break

    # 3. 统计出口点(没有下游的节点)
    outlets = [k for k, v in topology.items() if v is None]
    result.metrics['num_outlets'] = len(outlets)
    result.metrics['total_segments'] = len(topology)

    if len(outlets) == 0:
        result.add_warning("未找到出口点(所有节点都有下游)")
    elif len(outlets) > 1:
        result.add_warning(f"存在多个出口点({len(outlets)}个)")

    return result
