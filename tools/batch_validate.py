#!/usr/bin/env python3
"""批量验证工具

对项目中的所有关键数据进行批量质量检查。

用法:
    python tools/batch_validate.py --project upper_truckee_complete_11steps
    python tools/batch_validate.py --config config/workflow_config.yaml --report

功能:
- 验证降雨数据质量
- 验证流域几何数据
- 验证河网拓扑
- 验证时间序列数据
- 生成验证报告
"""
import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hydrosis.config import load_workflow_config
from hydrosis.validation import (
    ValidationResult,
    validate_precipitation_data,
    validate_basin_geometry,
    validate_network_topology,
    validate_time_series,
    PrecipitationCriteria,
    SpatialCriteria,
    TimeSeriesCriteria,
)


class BatchValidator:
    """批量验证器"""

    def __init__(self, config_path: Path, validation_config_path: Path = None):
        """初始化

        Args:
            config_path: 工作流配置文件路径
            validation_config_path: 验证标准配置文件路径
        """
        self.config = load_workflow_config(config_path)
        self.base_dir = Path(self.config['directories']['base_results'])

        # 加载验证标准
        if validation_config_path and validation_config_path.exists():
            import yaml
            with open(validation_config_path) as f:
                val_config = yaml.safe_load(f)
            self.precip_criteria = PrecipitationCriteria.from_dict(val_config['precipitation'])
            self.spatial_criteria = SpatialCriteria.from_dict(val_config['spatial'])
            self.timeseries_criteria = TimeSeriesCriteria.from_dict(val_config['timeseries'])
        else:
            # 使用默认标准
            self.precip_criteria = PrecipitationCriteria()
            self.spatial_criteria = SpatialCriteria()
            self.timeseries_criteria = TimeSeriesCriteria()

        self.results: Dict[str, ValidationResult] = {}

    def validate_precipitation(self) -> ValidationResult:
        """验证降雨数据"""
        print("\n⚙ 验证降雨数据...")

        # 查找降雨数据文件
        precip_file = self.base_dir / 'step_08_areal_precipitation' / '8.2_subbasin_areal_precipitation.csv'

        if not precip_file.exists():
            result = ValidationResult(step_name="降雨数据验证")
            result.add_error(f"降雨数据文件不存在: {precip_file}")
            return result

        # 加载降雨数据
        import pandas as pd
        precip_df = pd.read_csv(precip_file, index_col=0)

        # 验证
        result = validate_precipitation_data(
            precip_df,
            criteria=self.precip_criteria,
            step_name="批量验证 - 降雨数据"
        )

        return result

    def validate_basin_geometries(self) -> ValidationResult:
        """验证流域几何"""
        print("\n⚙ 验证流域几何...")

        # 查找流域GeoJSON
        basin_file = self.base_dir / 'step_03_parameter_zones' / '3.2_parameter_subbasins.geojson'

        if not basin_file.exists():
            result = ValidationResult(step_name="流域几何验证")
            result.add_error(f"流域文件不存在: {basin_file}")
            return result

        # 加载GeoJSON
        import json
        with open(basin_file) as f:
            data = json.load(f)
            features = data['features']

        # 验证
        result = validate_basin_geometry(
            features,
            criteria=self.spatial_criteria,
            step_name="批量验证 - 流域几何"
        )

        return result

    def validate_network(self) -> ValidationResult:
        """验证河网拓扑"""
        print("\n⚙ 验证河网拓扑...")

        # 查找河网GeoJSON
        network_file = self.base_dir / 'step_03_parameter_zones' / '3.3_channel_network.geojson'

        if not network_file.exists():
            result = ValidationResult(step_name="河网拓扑验证")
            result.add_error(f"河网文件不存在: {network_file}")
            return result

        # 加载GeoJSON
        import json
        with open(network_file) as f:
            data = json.load(f)
            features = data['features']

        # 验证
        result = validate_network_topology(
            features,
            criteria=self.spatial_criteria,
            step_name="批量验证 - 河网拓扑"
        )

        return result

    def validate_runoff_timeseries(self) -> Dict[str, ValidationResult]:
        """验证径流时间序列"""
        print("\n⚙ 验证径流时间序列...")

        # 查找径流结果目录
        runoff_dir = self.base_dir / 'workflow_results' / 'baseline_local'

        if not runoff_dir.exists():
            result = ValidationResult(step_name="径流时序验证")
            result.add_error(f"径流结果目录不存在: {runoff_dir}")
            return {"summary": result}

        # 验证每个分区的径流序列
        import pandas as pd
        results = {}

        for csv_file in runoff_dir.glob('*.csv'):
            zone_id = csv_file.stem
            df = pd.read_csv(csv_file)

            if 'runoff_m3s' in df.columns:
                series = df['runoff_m3s']
                result = validate_time_series(
                    series,
                    criteria=self.timeseries_criteria,
                    series_name=f"Zone {zone_id}",
                    step_name="批量验证 - 径流时序"
                )
                results[zone_id] = result

        return results

    def run_all_validations(self) -> None:
        """运行所有验证"""
        print("=" * 80)
        print("批量验证工具")
        print("=" * 80)
        print(f"项目目录: {self.base_dir}")

        # 1. 验证降雨
        try:
            result = self.validate_precipitation()
            self.results['precipitation'] = result
            self._print_result("降雨数据", result)
        except Exception as e:
            print(f"  ❌ 降雨验证失败: {e}")

        # 2. 验证流域几何
        try:
            result = self.validate_basin_geometries()
            self.results['basin_geometry'] = result
            self._print_result("流域几何", result)
        except Exception as e:
            print(f"  ❌ 流域几何验证失败: {e}")

        # 3. 验证河网拓扑
        try:
            result = self.validate_network()
            self.results['network_topology'] = result
            self._print_result("河网拓扑", result)
        except Exception as e:
            print(f"  ❌ 河网拓扑验证失败: {e}")

        # 4. 验证径流时序
        try:
            timeseries_results = self.validate_runoff_timeseries()
            self.results['runoff_timeseries'] = timeseries_results

            # 统计
            total = len(timeseries_results)
            valid = sum(1 for r in timeseries_results.values() if r.is_valid)
            print(f"\n✓ 径流时序验证完成: {valid}/{total} 个分区通过")
        except Exception as e:
            print(f"  ❌ 径流时序验证失败: {e}")

    def _print_result(self, name: str, result: ValidationResult) -> None:
        """打印验证结果"""
        if result.is_valid:
            print(f"✓ {name}验证通过")
        else:
            print(f"✗ {name}验证失败:")
            for error in result.errors:
                print(f"  - {error}")

        if result.warnings:
            print(f"⚠ {name}警告:")
            for warning in result.warnings:
                print(f"  - {warning}")

    def generate_report(self, output_path: Path) -> None:
        """生成验证报告

        Args:
            output_path: 报告输出路径
        """
        print(f"\n⚙ 生成验证报告: {output_path}")

        report = {
            'project_dir': str(self.base_dir),
            'validation_summary': {},
            'details': {}
        }

        # 汇总统计
        total_checks = 0
        passed_checks = 0

        for check_name, result in self.results.items():
            if check_name == 'runoff_timeseries':
                # 处理时序结果
                timeseries_results = result
                total = len(timeseries_results)
                valid = sum(1 for r in timeseries_results.values() if r.is_valid)

                report['validation_summary'][check_name] = {
                    'total': total,
                    'passed': valid,
                    'pass_rate': f"{valid/total:.1%}" if total > 0 else "N/A"
                }

                total_checks += total
                passed_checks += valid
            else:
                # 处理单个结果
                report['validation_summary'][check_name] = {
                    'is_valid': result.is_valid,
                    'errors': len(result.errors),
                    'warnings': len(result.warnings)
                }

                report['details'][check_name] = {
                    'errors': result.errors,
                    'warnings': result.warnings,
                    'metrics': result.metrics
                }

                total_checks += 1
                if result.is_valid:
                    passed_checks += 1

        report['overall'] = {
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'pass_rate': f"{passed_checks/total_checks:.1%}" if total_checks > 0 else "N/A"
        }

        # 保存JSON报告
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"✓ 报告已保存: {output_path}")
        print(f"\n总体通过率: {report['overall']['pass_rate']}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='批量验证工具')
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('config/workflow_config.yaml'),
        help='工作流配置文件路径'
    )
    parser.add_argument(
        '--validation-config',
        type=Path,
        default=Path('config/validation_criteria.yaml'),
        help='验证标准配置文件路径'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='生成JSON验证报告'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('validation_report.json'),
        help='报告输出路径'
    )

    args = parser.parse_args()

    # 创建验证器
    validator = BatchValidator(args.config, args.validation_config)

    # 运行验证
    validator.run_all_validations()

    # 生成报告
    if args.report:
        validator.generate_report(args.output)

    print("\n" + "=" * 80)
    print("✅ 批量验证完成")
    print("=" * 80)


if __name__ == '__main__':
    main()
