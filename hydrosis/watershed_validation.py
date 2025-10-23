"""Watershed delineation validation module.

This module provides validation logic to ensure watershed delineation outputs
meet expected criteria, preventing inconsistent zone counts and invalid configurations.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np


@dataclass
class ValidationCriteria:
    """Validation criteria for watershed delineation outputs."""

    # Zone validation
    expected_zone_count: int = 6
    zone_count_tolerance: int = 0  # Must be exactly 6 by default
    min_zone_area_km2: float = 10.0  # Minimum area per zone
    max_zone_area_km2: float = 1000.0  # Maximum area per zone

    # Subbasin validation
    expected_total_subbasins: Optional[int] = 157
    subbasin_count_tolerance: int = 10  # Allow ±10 subbasins
    min_subbasin_area_km2: float = 2.0
    min_subbasins_per_zone: int = 5  # Each zone should have at least 5 subbasins

    # Pour point validation
    expected_pour_point_count: int = 6
    require_main_stream_points: int = 3  # Must have exactly 3 main stream points
    require_tributary_points: int = 3  # Must have exactly 3 tributary points

    # Connectivity validation
    validate_downstream_connectivity: bool = True
    allow_disconnected_zones: bool = False


@dataclass
class ValidationResult:
    """Result of watershed validation."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def summary(self) -> str:
        """Generate a summary report."""
        lines = []
        lines.append("=" * 80)
        lines.append("流域划分验证结果")
        lines.append("=" * 80)
        lines.append(f"验证状态: {'✅ 通过' if self.is_valid else '❌ 失败'}")
        lines.append("")

        if self.metrics:
            lines.append("关键指标:")
            for key, value in self.metrics.items():
                lines.append(f"  - {key}: {value}")
            lines.append("")

        if self.errors:
            lines.append(f"错误 ({len(self.errors)}):")
            for error in self.errors:
                lines.append(f"  ❌ {error}")
            lines.append("")

        if self.warnings:
            lines.append(f"警告 ({len(self.warnings)}):")
            for warning in self.warnings:
                lines.append(f"  ⚠️  {warning}")
            lines.append("")

        lines.append("=" * 80)
        return "\n".join(lines)


class WatershedValidator:
    """Validator for watershed delineation outputs."""

    def __init__(self, criteria: Optional[ValidationCriteria] = None):
        """Initialize validator with criteria.

        Args:
            criteria: Validation criteria. If None, uses default criteria.
        """
        self.criteria = criteria or ValidationCriteria()

    def validate_all(
        self,
        parameter_dir: Path,
        pour_points_path: Path,
    ) -> ValidationResult:
        """Validate all watershed delineation outputs.

        Args:
            parameter_dir: Directory containing parameter zone outputs
            pour_points_path: Path to pour points GeoJSON

        Returns:
            ValidationResult with overall validation status
        """
        result = ValidationResult(is_valid=True)

        # 1. Validate pour points
        self._validate_pour_points(pour_points_path, result)

        # 2. Validate parameter zones
        self._validate_parameter_zones(parameter_dir, result)

        # 3. Validate subbasins
        self._validate_subbasins(parameter_dir, result)

        # 4. Validate connectivity
        if self.criteria.validate_downstream_connectivity:
            self._validate_connectivity(parameter_dir, result)

        return result

    def _validate_pour_points(
        self,
        pour_points_path: Path,
        result: ValidationResult,
    ) -> None:
        """Validate pour points."""
        if not pour_points_path.exists():
            result.add_error(f"汇水点文件不存在: {pour_points_path}")
            return

        try:
            with open(pour_points_path, 'r', encoding='utf-8') as f:
                pour_points_data = json.load(f)

            features = pour_points_data.get('features', [])
            total_count = len(features)

            # Count by type
            main_stream_count = sum(
                1 for f in features
                if f.get('properties', {}).get('type') == 'main_stream'
            )
            tributary_count = sum(
                1 for f in features
                if f.get('properties', {}).get('type') == 'tributary'
            )

            result.metrics['pour_point_总数'] = total_count
            result.metrics['pour_point_主流数'] = main_stream_count
            result.metrics['pour_point_支流数'] = tributary_count

            # Validate counts
            if total_count != self.criteria.expected_pour_point_count:
                result.add_error(
                    f"汇水点数量错误: 期望{self.criteria.expected_pour_point_count}个, "
                    f"实际{total_count}个"
                )

            if main_stream_count != self.criteria.require_main_stream_points:
                result.add_error(
                    f"主流汇水点数量错误: 期望{self.criteria.require_main_stream_points}个, "
                    f"实际{main_stream_count}个"
                )

            if tributary_count != self.criteria.require_tributary_points:
                result.add_error(
                    f"支流汇水点数量错误: 期望{self.criteria.require_tributary_points}个, "
                    f"实际{tributary_count}个"
                )

            # Validate zone_id assignment for main stream points
            main_stream_points = [
                f for f in features
                if f.get('properties', {}).get('type') == 'main_stream'
            ]
            zone_ids = set()
            for pp in main_stream_points:
                props = pp.get('properties', {})
                if 'zone_id' in props:
                    zone_ids.add(props['zone_id'])

            if len(zone_ids) != main_stream_count:
                result.add_warning("主流汇水点的zone_id重复或缺失")

        except Exception as e:
            result.add_error(f"读取汇水点文件失败: {e}")

    def _validate_parameter_zones(
        self,
        parameter_dir: Path,
        result: ValidationResult,
    ) -> None:
        """Validate parameter zones."""
        zones_csv = parameter_dir / "parameter_zones.csv"
        if not zones_csv.exists():
            result.add_error(f"参数分区文件不存在: {zones_csv}")
            return

        try:
            # Read zone data
            zones = []
            with open(zones_csv, 'r', encoding='utf-8') as f:
                lines = f.readlines()[1:]  # Skip header
                for line in lines:
                    if not line.strip():
                        continue
                    parts = line.strip().split(',')
                    if len(parts) >= 4:
                        zones.append({
                            'zone_id': parts[0],
                            'downstream_id': parts[1],
                            'area_km2': float(parts[3]) if parts[3] else 0.0,
                        })

            zone_count = len(zones)
            result.metrics['参数分区数量'] = zone_count

            # Validate zone count
            expected = self.criteria.expected_zone_count
            tolerance = self.criteria.zone_count_tolerance
            if not (expected - tolerance <= zone_count <= expected + tolerance):
                result.add_error(
                    f"参数分区数量错误: 期望{expected}个(±{tolerance}), 实际{zone_count}个"
                )

            # Validate zone areas
            for zone in zones:
                area = zone['area_km2']
                if area < self.criteria.min_zone_area_km2:
                    result.add_warning(
                        f"分区{zone['zone_id']}面积过小: {area:.2f} km² "
                        f"(最小值: {self.criteria.min_zone_area_km2} km²)"
                    )
                if area > self.criteria.max_zone_area_km2:
                    result.add_warning(
                        f"分区{zone['zone_id']}面积过大: {area:.2f} km² "
                        f"(最大值: {self.criteria.max_zone_area_km2} km²)"
                    )

            # Calculate statistics
            if zones:
                areas = [z['area_km2'] for z in zones]
                result.metrics['参数分区_平均面积_km2'] = np.mean(areas)
                result.metrics['参数分区_最小面积_km2'] = np.min(areas)
                result.metrics['参数分区_最大面积_km2'] = np.max(areas)

        except Exception as e:
            result.add_error(f"读取参数分区文件失败: {e}")

    def _validate_subbasins(
        self,
        parameter_dir: Path,
        result: ValidationResult,
    ) -> None:
        """Validate subbasins."""
        subbasins_csv = parameter_dir / "parameter_subbasins.csv"
        if not subbasins_csv.exists():
            result.add_error(f"子流域文件不存在: {subbasins_csv}")
            return

        try:
            # Read subbasin data
            subbasins = []
            with open(subbasins_csv, 'r', encoding='utf-8') as f:
                lines = f.readlines()[1:]  # Skip header
                for line in lines:
                    if not line.strip():
                        continue
                    parts = line.strip().split(',')
                    if len(parts) >= 4:
                        subbasins.append({
                            'zone_id': parts[0],
                            'subzone_id': parts[1],
                            'area_km2': float(parts[3]) if parts[3] else 0.0,
                        })

            total_count = len(subbasins)
            result.metrics['子流域总数'] = total_count

            # Validate total count
            if self.criteria.expected_total_subbasins:
                expected = self.criteria.expected_total_subbasins
                tolerance = self.criteria.subbasin_count_tolerance
                if not (expected - tolerance <= total_count <= expected + tolerance):
                    result.add_warning(
                        f"子流域总数偏离期望值: 期望{expected}个(±{tolerance}), "
                        f"实际{total_count}个"
                    )

            # Validate subbasins per zone
            zone_subbasin_counts: Dict[str, int] = {}
            for sub in subbasins:
                zone_id = sub['zone_id']
                zone_subbasin_counts[zone_id] = zone_subbasin_counts.get(zone_id, 0) + 1

            for zone_id, count in zone_subbasin_counts.items():
                if count < self.criteria.min_subbasins_per_zone:
                    result.add_warning(
                        f"分区{zone_id}的子流域数量过少: {count}个 "
                        f"(最小值: {self.criteria.min_subbasins_per_zone})"
                    )

            # Validate subbasin areas
            for sub in subbasins:
                area = sub['area_km2']
                if area < self.criteria.min_subbasin_area_km2:
                    result.add_warning(
                        f"子流域{sub['subzone_id']}面积过小: {area:.2f} km² "
                        f"(最小值: {self.criteria.min_subbasin_area_km2} km²)"
                    )

            # Calculate statistics
            if subbasins:
                areas = [s['area_km2'] for s in subbasins]
                result.metrics['子流域_平均面积_km2'] = np.mean(areas)
                result.metrics['子流域_最小面积_km2'] = np.min(areas)
                result.metrics['子流域_最大面积_km2'] = np.max(areas)

        except Exception as e:
            result.add_error(f"读取子流域文件失败: {e}")

    def _validate_connectivity(
        self,
        parameter_dir: Path,
        result: ValidationResult,
    ) -> None:
        """Validate downstream connectivity."""
        zones_csv = parameter_dir / "parameter_zones.csv"
        if not zones_csv.exists():
            return

        try:
            # Build connectivity graph
            zones = {}
            with open(zones_csv, 'r', encoding='utf-8') as f:
                lines = f.readlines()[1:]  # Skip header
                for line in lines:
                    if not line.strip():
                        continue
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        zone_id = parts[0]
                        downstream_id = parts[1] if parts[1] else None
                        zones[zone_id] = downstream_id

            # Find roots (zones with no downstream)
            roots = [zid for zid, down in zones.items() if not down]

            if len(roots) == 0:
                result.add_error("没有找到出口分区(所有分区都有下游)")
            elif len(roots) > 1:
                if not self.criteria.allow_disconnected_zones:
                    result.add_error(
                        f"发现多个出口分区: {roots} (流域不连通)"
                    )
                else:
                    result.add_warning(
                        f"发现多个出口分区: {roots} (可能有多个独立流域)"
                    )

            # Check for cycles
            visited = set()
            for zone_id in zones:
                path = []
                current = zone_id
                while current and current not in visited:
                    if current in path:
                        result.add_error(f"发现循环引用: {' -> '.join(path + [current])}")
                        break
                    path.append(current)
                    current = zones.get(current)
                visited.update(path)

        except Exception as e:
            result.add_error(f"验证连通性失败: {e}")
