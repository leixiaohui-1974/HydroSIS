"""
外部数据加载模块
支持从CSV、GeoJSON和Shapefile读取汇水点和雨量站数据
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from shapely.geometry import Point


def load_pour_points_from_csv(
    csv_path: Path | str,
    x_col: str = "x",
    y_col: str = "y",
    id_col: str = "id",
) -> List[Dict[str, Any]]:
    """
    从CSV文件加载汇水点数据

    Args:
        csv_path: CSV文件路径
        x_col: X坐标列名
        y_col: Y坐标列名
        id_col: ID列名

    Returns:
        汇水点列表，每个元素是一个字典包含id, x, y等属性
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV文件不存在: {csv_path}")

    df = pd.read_csv(csv_path)

    # 检查必需的列
    required_cols = [id_col, x_col, y_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV文件缺少必需的列: {missing_cols}")

    pour_points = []
    for _, row in df.iterrows():
        point_dict = {
            'id': str(row[id_col]),
            'x': float(row[x_col]),
            'y': float(row[y_col]),
        }

        # 添加其他可选列
        optional_cols = ['type', 'zone_id', 'depth', 'accumulation', 'controlled_area_km2',
                        'row', 'col', 'main_stream_id']
        for col in optional_cols:
            if col in df.columns and pd.notna(row[col]):
                point_dict[col] = row[col]

        pour_points.append(point_dict)

    return pour_points


def load_pour_points_from_geojson(
    geojson_path: Path | str
) -> List[Dict[str, Any]]:
    """
    从GeoJSON文件加载汇水点数据

    Args:
        geojson_path: GeoJSON文件路径

    Returns:
        汇水点列表，每个元素是一个字典包含id, x, y等属性
    """
    geojson_path = Path(geojson_path)
    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON文件不存在: {geojson_path}")

    with open(geojson_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if data.get('type') != 'FeatureCollection':
        raise ValueError("GeoJSON文件必须是FeatureCollection类型")

    pour_points = []
    for feature in data.get('features', []):
        if feature.get('type') != 'Feature':
            continue

        geometry = feature.get('geometry', {})
        if geometry.get('type') != 'Point':
            continue

        coords = geometry.get('coordinates', [])
        if len(coords) < 2:
            continue

        props = feature.get('properties', {})
        point_dict = {
            'id': str(props.get('id', '')),
            'x': float(coords[0]),
            'y': float(coords[1]),
        }

        # 添加其他属性
        for key in ['type', 'zone_id', 'depth', 'accumulation', 'controlled_area_km2',
                   'row', 'col', 'main_stream_id']:
            if key in props:
                point_dict[key] = props[key]

        pour_points.append(point_dict)

    return pour_points


def load_rain_gauges_from_csv(
    csv_path: Path | str,
    x_col: str = "x",
    y_col: str = "y",
    id_col: str = "id",
) -> Dict[str, Point]:
    """
    从CSV文件加载雨量站数据

    Args:
        csv_path: CSV文件路径
        x_col: X坐标列名
        y_col: Y坐标列名
        id_col: ID列名

    Returns:
        雨量站位置字典 {station_id: Point}
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV文件不存在: {csv_path}")

    df = pd.read_csv(csv_path)

    # 检查必需的列
    required_cols = [id_col, x_col, y_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV文件缺少必需的列: {missing_cols}")

    station_positions = {}
    for _, row in df.iterrows():
        station_id = str(row[id_col])
        x = float(row[x_col])
        y = float(row[y_col])
        station_positions[station_id] = Point(x, y)

    return station_positions


def load_rain_gauges_from_geojson(
    geojson_path: Path | str
) -> Dict[str, Point]:
    """
    从GeoJSON文件加载雨量站数据

    Args:
        geojson_path: GeoJSON文件路径

    Returns:
        雨量站位置字典 {station_id: Point}
    """
    geojson_path = Path(geojson_path)
    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON文件不存在: {geojson_path}")

    with open(geojson_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if data.get('type') != 'FeatureCollection':
        raise ValueError("GeoJSON文件必须是FeatureCollection类型")

    station_positions = {}
    for feature in data.get('features', []):
        if feature.get('type') != 'Feature':
            continue

        geometry = feature.get('geometry', {})
        if geometry.get('type') != 'Point':
            continue

        coords = geometry.get('coordinates', [])
        if len(coords) < 2:
            continue

        props = feature.get('properties', {})
        station_id = str(props.get('id', ''))
        if station_id:
            station_positions[station_id] = Point(coords[0], coords[1])

    return station_positions


def detect_file_format(file_path: Path | str) -> str:
    """
    检测文件格式

    Args:
        file_path: 文件路径

    Returns:
        文件格式: 'csv', 'geojson', 'shp', 或 'unknown'
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    format_map = {
        '.csv': 'csv',
        '.geojson': 'geojson',
        '.json': 'geojson',
        '.shp': 'shp',
    }

    return format_map.get(suffix, 'unknown')


def load_pour_points(file_path: Path | str) -> List[Dict[str, Any]]:
    """
    自动检测格式并加载汇水点数据

    Args:
        file_path: 文件路径（CSV或GeoJSON）

    Returns:
        汇水点列表
    """
    file_path = Path(file_path)
    fmt = detect_file_format(file_path)

    if fmt == 'csv':
        return load_pour_points_from_csv(file_path)
    elif fmt == 'geojson':
        return load_pour_points_from_geojson(file_path)
    elif fmt == 'shp':
        raise NotImplementedError("Shapefile格式暂未实现，请使用CSV或GeoJSON格式")
    else:
        raise ValueError(f"不支持的文件格式: {file_path.suffix}")


def load_rain_gauges(file_path: Path | str) -> Dict[str, Point]:
    """
    自动检测格式并加载雨量站数据

    Args:
        file_path: 文件路径（CSV或GeoJSON）

    Returns:
        雨量站位置字典 {station_id: Point}
    """
    file_path = Path(file_path)
    fmt = detect_file_format(file_path)

    if fmt == 'csv':
        return load_rain_gauges_from_csv(file_path)
    elif fmt == 'geojson':
        return load_rain_gauges_from_geojson(file_path)
    elif fmt == 'shp':
        raise NotImplementedError("Shapefile格式暂未实现，请使用CSV或GeoJSON格式")
    else:
        raise ValueError(f"不支持的文件格式: {file_path.suffix}")


__all__ = [
    'load_pour_points',
    'load_rain_gauges',
    'load_pour_points_from_csv',
    'load_pour_points_from_geojson',
    'load_rain_gauges_from_csv',
    'load_rain_gauges_from_geojson',
    'detect_file_format',
]
