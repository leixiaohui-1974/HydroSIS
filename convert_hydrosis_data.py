#!/usr/bin/env python3
"""
HydroSIS数据格式转换命令行工具

使用示例：
  # 转换汇水点数据
  python convert_hydrosis_data.py pour-points input.csv output.geojson

  # 转换雨量站数据
  python convert_hydrosis_data.py rain-gauges input.geojson output.shp --crs EPSG:3857

  # 指定Shapefile的ID字段
  python convert_hydrosis_data.py pour-points input.shp output.csv --id-field station_id
"""
import argparse
import sys
from pathlib import Path

from hydrosis.utils.data_converter import convert_pour_points, convert_rain_gauges


def main():
    parser = argparse.ArgumentParser(
        description="HydroSIS数据格式转换工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
支持的格式：
  - CSV (.csv)
  - GeoJSON (.geojson, .json)
  - Shapefile (.shp)

示例：
  %(prog)s pour-points data/pour_points.csv output/pour_points.geojson
  %(prog)s rain-gauges data/stations.shp output/stations.csv --id-field ID
  %(prog)s pour-points input.csv output.shp --crs EPSG:3857
        """
    )

    parser.add_argument(
        'data_type',
        choices=['pour-points', 'rain-gauges'],
        help='数据类型'
    )
    parser.add_argument(
        'input',
        type=Path,
        help='输入文件路径'
    )
    parser.add_argument(
        'output',
        type=Path,
        help='输出文件路径'
    )
    parser.add_argument(
        '--id-field',
        default='id',
        help='ID字段名（用于Shapefile输入，默认: id）'
    )
    parser.add_argument(
        '--crs',
        default=None,
        help='坐标参考系统（用于GeoJSON和Shapefile输出），例如: EPSG:3857'
    )

    args = parser.parse_args()

    # 检查输入文件是否存在
    if not args.input.exists():
        print(f"错误：输入文件不存在: {args.input}", file=sys.stderr)
        return 1

    # 创建输出目录
    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        if args.data_type == 'pour-points':
            convert_pour_points(
                input_path=args.input,
                output_path=args.output,
                id_field=args.id_field,
                crs=args.crs,
            )
        else:  # rain-gauges
            convert_rain_gauges(
                input_path=args.input,
                output_path=args.output,
                id_field=args.id_field,
                crs=args.crs,
            )
    except Exception as e:
        print(f"错误：转换失败 - {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
