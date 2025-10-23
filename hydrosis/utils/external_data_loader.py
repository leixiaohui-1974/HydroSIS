"""
外部数据加载模块
支持从CSV、GeoJSON和Shapefile读取汇水点和雨量站数据
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from shapely.geometry import Point, shape as shapely_shape


# ==============================================================================
# 数据验证函数
# ==============================================================================

def validate_pour_points(
    pour_points: List[Dict[str, Any]],
    check_unique_ids: bool = True,
    check_coordinate_range: bool = True,
    coord_bounds: Optional[Tuple[float, float, float, float]] = None,
) -> List[str]:
    """
    验证汇水点数据的有效性

    Args:
        pour_points: 汇水点列表
        check_unique_ids: 是否检查ID唯一性
        check_coordinate_range: 是否检查坐标范围
        coord_bounds: 坐标边界 (min_x, min_y, max_x, max_y)，None表示使用全球范围

    Returns:
        验证错误信息列表，空列表表示验证通过
    """
    errors = []

    if not pour_points:
        errors.append("汇水点列表为空")
        return errors

    # 检查ID唯一性
    if check_unique_ids:
        ids = [p.get('id') for p in pour_points]
        id_counts = {}
        for pid in ids:
            id_counts[pid] = id_counts.get(pid, 0) + 1

        duplicates = [pid for pid, count in id_counts.items() if count > 1]
        if duplicates:
            errors.append(f"发现重复的汇水点ID: {duplicates}")

    # 检查坐标范围
    if check_coordinate_range:
        # 默认全球坐标范围（Web Mercator投影）
        if coord_bounds is None:
            min_x, min_y, max_x, max_y = -20037508.34, -20037508.34, 20037508.34, 20037508.34
        else:
            min_x, min_y, max_x, max_y = coord_bounds

        for i, point in enumerate(pour_points):
            x = point.get('x')
            y = point.get('y')
            pid = point.get('id', f'point_{i}')

            if x is None or y is None:
                errors.append(f"汇水点 {pid} 缺少坐标")
                continue

            try:
                x = float(x)
                y = float(y)
            except (ValueError, TypeError):
                errors.append(f"汇水点 {pid} 坐标格式错误: x={x}, y={y}")
                continue

            if not (min_x <= x <= max_x):
                errors.append(f"汇水点 {pid} X坐标超出范围: {x} (范围: {min_x} 到 {max_x})")
            if not (min_y <= y <= max_y):
                errors.append(f"汇水点 {pid} Y坐标超出范围: {y} (范围: {min_y} 到 {max_y})")

    # 检查必需字段
    for i, point in enumerate(pour_points):
        pid = point.get('id', f'point_{i}')
        if not point.get('id'):
            errors.append(f"汇水点 {i} 缺少ID字段")
        if 'x' not in point or 'y' not in point:
            errors.append(f"汇水点 {pid} 缺少坐标字段")

    return errors


def validate_rain_gauges(
    station_positions: Dict[str, Point],
    check_coordinate_range: bool = True,
    coord_bounds: Optional[Tuple[float, float, float, float]] = None,
) -> List[str]:
    """
    验证雨量站数据的有效性

    Args:
        station_positions: 雨量站位置字典 {station_id: Point}
        check_coordinate_range: 是否检查坐标范围
        coord_bounds: 坐标边界 (min_x, min_y, max_x, max_y)，None表示使用全球范围

    Returns:
        验证错误信息列表，空列表表示验证通过
    """
    errors = []

    if not station_positions:
        errors.append("雨量站列表为空")
        return errors

    # 检查坐标范围
    if check_coordinate_range:
        # 默认全球坐标范围（Web Mercator投影）
        if coord_bounds is None:
            min_x, min_y, max_x, max_y = -20037508.34, -20037508.34, 20037508.34, 20037508.34
        else:
            min_x, min_y, max_x, max_y = coord_bounds

        for station_id, point in station_positions.items():
            if not isinstance(point, Point):
                errors.append(f"雨量站 {station_id} 不是Point类型")
                continue

            x, y = point.x, point.y

            if not (min_x <= x <= max_x):
                errors.append(f"雨量站 {station_id} X坐标超出范围: {x} (范围: {min_x} 到 {max_x})")
            if not (min_y <= y <= max_y):
                errors.append(f"雨量站 {station_id} Y坐标超出范围: {y} (范围: {min_y} 到 {max_y})")

    # 检查空ID
    if '' in station_positions:
        errors.append("发现空ID的雨量站")

    return errors


# ==============================================================================
# 数据加载函数
# ==============================================================================

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


def load_pour_points_from_shapefile(
    shp_path: Path | str,
    id_field: str = "id",
) -> List[Dict[str, Any]]:
    """
    从Shapefile加载汇水点数据

    Args:
        shp_path: Shapefile路径（.shp文件）
        id_field: ID字段名

    Returns:
        汇水点列表，每个元素是一个字典包含id, x, y等属性

    Raises:
        ImportError: 如果pyshp库未安装
        FileNotFoundError: 如果文件不存在
        ValueError: 如果数据格式不正确
    """
    try:
        import shapefile
    except ImportError:
        raise ImportError(
            "读取Shapefile需要安装pyshp库。请运行: pip install pyshp"
        )

    shp_path = Path(shp_path)
    if not shp_path.exists():
        raise FileNotFoundError(f"Shapefile不存在: {shp_path}")

    # 读取shapefile
    try:
        sf = shapefile.Reader(str(shp_path))
    except Exception as e:
        raise ValueError(f"无法读取Shapefile: {e}")

    # 获取字段名
    field_names = [field[0] for field in sf.fields[1:]]  # 跳过第一个DeletionFlag字段

    if id_field not in field_names:
        raise ValueError(f"Shapefile中未找到ID字段 '{id_field}'，可用字段: {field_names}")

    pour_points = []
    for record in sf.shapeRecords():
        shape = record.shape
        props = dict(zip(field_names, record.record))

        # 只处理Point类型
        if shape.shapeType != shapefile.POINT:
            warnings.warn(f"跳过非Point类型的要素: {shape.shapeTypeName}")
            continue

        # 获取坐标
        x, y = shape.points[0]

        point_dict = {
            'id': str(props.get(id_field, '')),
            'x': float(x),
            'y': float(y),
        }

        # 添加其他属性（自动识别常用字段）
        optional_fields = ['type', 'zone_id', 'depth', 'accumulation', 'controlled_area_km2',
                          'row', 'col', 'main_stream_id', 'TYPE', 'ZONE_ID', 'DEPTH']
        for field in optional_fields:
            if field in props and props[field] is not None:
                # 转换为小写键
                key = field.lower()
                point_dict[key] = props[field]

        pour_points.append(point_dict)

    sf.close()
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


def load_rain_gauges_from_shapefile(
    shp_path: Path | str,
    id_field: str = "id",
) -> Dict[str, Point]:
    """
    从Shapefile加载雨量站数据

    Args:
        shp_path: Shapefile路径（.shp文件）
        id_field: ID字段名

    Returns:
        雨量站位置字典 {station_id: Point}

    Raises:
        ImportError: 如果pyshp库未安装
        FileNotFoundError: 如果文件不存在
        ValueError: 如果数据格式不正确
    """
    try:
        import shapefile
    except ImportError:
        raise ImportError(
            "读取Shapefile需要安装pyshp库。请运行: pip install pyshp"
        )

    shp_path = Path(shp_path)
    if not shp_path.exists():
        raise FileNotFoundError(f"Shapefile不存在: {shp_path}")

    # 读取shapefile
    try:
        sf = shapefile.Reader(str(shp_path))
    except Exception as e:
        raise ValueError(f"无法读取Shapefile: {e}")

    # 获取字段名
    field_names = [field[0] for field in sf.fields[1:]]

    if id_field not in field_names:
        raise ValueError(f"Shapefile中未找到ID字段 '{id_field}'，可用字段: {field_names}")

    station_positions = {}
    for record in sf.shapeRecords():
        shape = record.shape
        props = dict(zip(field_names, record.record))

        # 只处理Point类型
        if shape.shapeType != shapefile.POINT:
            warnings.warn(f"跳过非Point类型的要素: {shape.shapeTypeName}")
            continue

        # 获取坐标
        x, y = shape.points[0]
        station_id = str(props.get(id_field, ''))

        if station_id:
            station_positions[station_id] = Point(x, y)

    sf.close()
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


def load_pour_points(
    file_path: Path | str,
    id_field: str = "id",
    validate: bool = True,
    coord_bounds: Optional[Tuple[float, float, float, float]] = None,
) -> List[Dict[str, Any]]:
    """
    自动检测格式并加载汇水点数据

    Args:
        file_path: 文件路径（CSV、GeoJSON或Shapefile）
        id_field: ID字段名（仅用于Shapefile）
        validate: 是否进行数据验证
        coord_bounds: 坐标边界 (min_x, min_y, max_x, max_y)，用于验证

    Returns:
        汇水点列表

    Raises:
        ValueError: 数据验证失败时抛出
    """
    file_path = Path(file_path)
    fmt = detect_file_format(file_path)

    if fmt == 'csv':
        pour_points = load_pour_points_from_csv(file_path)
    elif fmt == 'geojson':
        pour_points = load_pour_points_from_geojson(file_path)
    elif fmt == 'shp':
        pour_points = load_pour_points_from_shapefile(file_path, id_field=id_field)
    else:
        raise ValueError(f"不支持的文件格式: {file_path.suffix}")

    # 数据验证
    if validate:
        errors = validate_pour_points(pour_points, coord_bounds=coord_bounds)
        if errors:
            error_msg = "汇水点数据验证失败:\n" + "\n".join(f"  - {err}" for err in errors)
            raise ValueError(error_msg)

    return pour_points


def load_rain_gauges(
    file_path: Path | str,
    id_field: str = "id",
    validate: bool = True,
    coord_bounds: Optional[Tuple[float, float, float, float]] = None,
) -> Dict[str, Point]:
    """
    自动检测格式并加载雨量站数据

    Args:
        file_path: 文件路径（CSV、GeoJSON或Shapefile）
        id_field: ID字段名（仅用于Shapefile）
        validate: 是否进行数据验证
        coord_bounds: 坐标边界 (min_x, min_y, max_x, max_y)，用于验证

    Returns:
        雨量站位置字典 {station_id: Point}

    Raises:
        ValueError: 数据验证失败时抛出
    """
    file_path = Path(file_path)
    fmt = detect_file_format(file_path)

    if fmt == 'csv':
        station_positions = load_rain_gauges_from_csv(file_path)
    elif fmt == 'geojson':
        station_positions = load_rain_gauges_from_geojson(file_path)
    elif fmt == 'shp':
        station_positions = load_rain_gauges_from_shapefile(file_path, id_field=id_field)
    else:
        raise ValueError(f"不支持的文件格式: {file_path.suffix}")

    # 数据验证
    if validate:
        errors = validate_rain_gauges(station_positions, coord_bounds=coord_bounds)
        if errors:
            error_msg = "雨量站数据验证失败:\n" + "\n".join(f"  - {err}" for err in errors)
            raise ValueError(error_msg)

    return station_positions


__all__ = [
    'load_pour_points',
    'load_rain_gauges',
    'load_pour_points_from_csv',
    'load_pour_points_from_geojson',
    'load_pour_points_from_shapefile',
    'load_rain_gauges_from_csv',
    'load_rain_gauges_from_geojson',
    'load_rain_gauges_from_shapefile',
    'validate_pour_points',
    'validate_rain_gauges',
    'detect_file_format',
]
