"""
数据格式转换工具
支持汇水点和雨量站数据在CSV、GeoJSON和Shapefile之间的相互转换
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from shapely.geometry import Point, mapping

from .external_data_loader import (
    load_pour_points,
    load_rain_gauges,
    detect_file_format,
)


def convert_pour_points(
    input_path: Path | str,
    output_path: Path | str,
    id_field: str = "id",
    crs: Optional[str] = None,
) -> None:
    """
    转换汇水点数据格式

    支持的格式转换：
    - CSV → GeoJSON
    - CSV → Shapefile
    - GeoJSON → CSV
    - GeoJSON → Shapefile
    - Shapefile → CSV
    - Shapefile → GeoJSON

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        id_field: ID字段名（用于Shapefile输入）
        crs: 坐标参考系统（用于GeoJSON和Shapefile输出），例如 "EPSG:3857"

    Raises:
        ValueError: 不支持的格式转换
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    input_fmt = detect_file_format(input_path)
    output_fmt = detect_file_format(output_path)

    if input_fmt == 'unknown':
        raise ValueError(f"不支持的输入格式: {input_path.suffix}")
    if output_fmt == 'unknown':
        raise ValueError(f"不支持的输出格式: {output_path.suffix}")

    # 加载数据（不进行验证，因为可能需要转换后再验证）
    pour_points = load_pour_points(input_path, id_field=id_field, validate=False)

    # 转换为目标格式
    if output_fmt == 'csv':
        _save_pour_points_to_csv(pour_points, output_path)
    elif output_fmt == 'geojson':
        _save_pour_points_to_geojson(pour_points, output_path, crs=crs)
    elif output_fmt == 'shp':
        _save_pour_points_to_shapefile(pour_points, output_path, crs=crs)

    print(f"✓ 成功转换: {input_path} ({input_fmt}) → {output_path} ({output_fmt})")
    print(f"  共转换 {len(pour_points)} 个汇水点")


def convert_rain_gauges(
    input_path: Path | str,
    output_path: Path | str,
    id_field: str = "id",
    crs: Optional[str] = None,
) -> None:
    """
    转换雨量站数据格式

    支持的格式转换：
    - CSV → GeoJSON
    - CSV → Shapefile
    - GeoJSON → CSV
    - GeoJSON → Shapefile
    - Shapefile → CSV
    - Shapefile → GeoJSON

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        id_field: ID字段名（用于Shapefile输入）
        crs: 坐标参考系统（用于GeoJSON和Shapefile输出），例如 "EPSG:3857"

    Raises:
        ValueError: 不支持的格式转换
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    input_fmt = detect_file_format(input_path)
    output_fmt = detect_file_format(output_path)

    if input_fmt == 'unknown':
        raise ValueError(f"不支持的输入格式: {input_path.suffix}")
    if output_fmt == 'unknown':
        raise ValueError(f"不支持的输出格式: {output_path.suffix}")

    # 加载数据
    station_positions = load_rain_gauges(input_path, id_field=id_field, validate=False)

    # 转换为目标格式
    if output_fmt == 'csv':
        _save_rain_gauges_to_csv(station_positions, output_path)
    elif output_fmt == 'geojson':
        _save_rain_gauges_to_geojson(station_positions, output_path, crs=crs)
    elif output_fmt == 'shp':
        _save_rain_gauges_to_shapefile(station_positions, output_path, crs=crs)

    print(f"✓ 成功转换: {input_path} ({input_fmt}) → {output_path} ({output_fmt})")
    print(f"  共转换 {len(station_positions)} 个雨量站")


# ==============================================================================
# 内部辅助函数 - CSV输出
# ==============================================================================

def _save_pour_points_to_csv(pour_points: List[Dict[str, Any]], csv_path: Path) -> None:
    """保存汇水点数据到CSV"""
    df = pd.DataFrame(pour_points)
    # 确保ID, x, y列在最前面
    cols = ['id', 'x', 'y']
    other_cols = [c for c in df.columns if c not in cols]
    df = df[cols + other_cols]
    df.to_csv(csv_path, index=False)


def _save_rain_gauges_to_csv(station_positions: Dict[str, Point], csv_path: Path) -> None:
    """保存雨量站数据到CSV"""
    data = []
    for station_id, point in station_positions.items():
        data.append({
            'id': station_id,
            'x': point.x,
            'y': point.y,
        })
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)


# ==============================================================================
# 内部辅助函数 - GeoJSON输出
# ==============================================================================

def _save_pour_points_to_geojson(
    pour_points: List[Dict[str, Any]],
    geojson_path: Path,
    crs: Optional[str] = None,
) -> None:
    """保存汇水点数据到GeoJSON"""
    features = []
    for point in pour_points:
        x = point['x']
        y = point['y']

        # 提取properties（排除x, y坐标）
        properties = {k: v for k, v in point.items() if k not in ['x', 'y']}

        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [x, y]
            },
            'properties': properties
        }
        features.append(feature)

    geojson = {
        'type': 'FeatureCollection',
        'features': features
    }

    # 添加CRS信息
    if crs:
        geojson['crs'] = {
            'type': 'name',
            'properties': {'name': crs}
        }

    with open(geojson_path, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, indent=2, ensure_ascii=False)


def _save_rain_gauges_to_geojson(
    station_positions: Dict[str, Point],
    geojson_path: Path,
    crs: Optional[str] = None,
) -> None:
    """保存雨量站数据到GeoJSON"""
    features = []
    for station_id, point in station_positions.items():
        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [point.x, point.y]
            },
            'properties': {'id': station_id}
        }
        features.append(feature)

    geojson = {
        'type': 'FeatureCollection',
        'features': features
    }

    # 添加CRS信息
    if crs:
        geojson['crs'] = {
            'type': 'name',
            'properties': {'name': crs}
        }

    with open(geojson_path, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, indent=2, ensure_ascii=False)


# ==============================================================================
# 内部辅助函数 - Shapefile输出
# ==============================================================================

def _save_pour_points_to_shapefile(
    pour_points: List[Dict[str, Any]],
    shp_path: Path,
    crs: Optional[str] = None,
) -> None:
    """保存汇水点数据到Shapefile"""
    try:
        import shapefile
    except ImportError:
        raise ImportError(
            "保存Shapefile需要安装pyshp库。请运行: pip install pyshp"
        )

    # 创建Shapefile writer
    w = shapefile.Writer(str(shp_path), shapeType=shapefile.POINT)

    # 确定字段
    if pour_points:
        sample = pour_points[0]
        for key in sample.keys():
            if key in ['x', 'y']:
                continue
            # 简化字段类型判断
            value = sample[key]
            if isinstance(value, (int, float)):
                w.field(key, 'N', decimal=6)
            else:
                w.field(key, 'C', size=100)

    # 写入记录
    for point in pour_points:
        x = point['x']
        y = point['y']
        w.point(x, y)

        # 写入属性
        record = []
        for key in w.fields[1:]:  # 跳过DeletionFlag
            field_name = key[0]
            if field_name in point:
                record.append(point[field_name])
            else:
                record.append(None)
        w.record(*record)

    w.close()

    # 写入PRJ文件（投影信息）
    if crs:
        _write_prj_file(shp_path, crs)


def _save_rain_gauges_to_shapefile(
    station_positions: Dict[str, Point],
    shp_path: Path,
    crs: Optional[str] = None,
) -> None:
    """保存雨量站数据到Shapefile"""
    try:
        import shapefile
    except ImportError:
        raise ImportError(
            "保存Shapefile需要安装pyshp库。请运行: pip install pyshp"
        )

    # 创建Shapefile writer
    w = shapefile.Writer(str(shp_path), shapeType=shapefile.POINT)
    w.field('id', 'C', size=50)

    # 写入记录
    for station_id, point in station_positions.items():
        w.point(point.x, point.y)
        w.record(station_id)

    w.close()

    # 写入PRJ文件
    if crs:
        _write_prj_file(shp_path, crs)


def _write_prj_file(shp_path: Path, crs: str) -> None:
    """写入PRJ文件（投影信息）"""
    prj_path = shp_path.with_suffix('.prj')

    # 简化的CRS定义（仅支持常见的EPSG代码）
    crs_wkt_map = {
        'EPSG:3857': 'PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Mercator_1SP"],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1]]',
        'EPSG:4326': 'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]]',
    }

    wkt = crs_wkt_map.get(crs.upper())
    if wkt:
        with open(prj_path, 'w') as f:
            f.write(wkt)


__all__ = [
    'convert_pour_points',
    'convert_rain_gauges',
]
