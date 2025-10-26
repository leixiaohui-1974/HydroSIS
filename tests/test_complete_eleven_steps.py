"""完整的十一步工作流测试

使用Upper Truckee River实际项目的全部参数和配置进行测试。
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ElevenStepWorkflowTest:
    """十一步工作流完整测试"""
    
    def __init__(self, config_path: Path, output_dir: Path):
        """初始化测试
        
        Args:
            config_path: Upper Truckee项目配置文件路径
            output_dir: 测试输出目录
        """
        self.config_path = config_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载配置
        self.config = self._load_config()
        
        # 存储每一步的结果
        self.step_results: Dict[str, Any] = {}
        
        # 验证结果
        self.validation_results: Dict[str, Dict[str, Any]] = {}
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            import yaml
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except ImportError:
            logger.error("需要安装PyYAML")
            raise
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            raise
    
    def step01_terrain_processing(self):
        """步骤1: DEM地形处理"""
        logger.info("\n" + "=" * 80)
        logger.info("步骤1: DEM地形处理")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            from hydrosis.modules import TerrainModule
            
            # 获取配置
            delineation_config = self.config.get('delineation', {})
            dem_path = delineation_config.get('dem_path', '')
            
            # 转换相对路径
            if not Path(dem_path).is_absolute():
                dem_path = self.config_path.parent / dem_path
            
            if not Path(dem_path).exists():
                raise FileNotFoundError(f"DEM文件不存在: {dem_path}")
            
            output_dir = self.output_dir / "01_terrain"
            
            # 执行
            terrain = TerrainModule()
            output = terrain.run({
                "dem_path": str(dem_path),
                "method": "d8",
                "fill_depressions": True,
                "compute_slope": True,
                "output_dir": str(output_dir)
            })
            
            # 保存结果
            self.step_results['step01'] = {
                "flow_direction": output.flow_direction,
                "flow_accumulation": output.flow_accumulation,
                "filled_dem": output.filled_dem,
                "slope": output.slope,
                "metadata": output.metadata
            }
            
            # 验证
            self._validate_step01(output_dir)
            
            elapsed = time.time() - start_time
            logger.info(f"✓ 步骤1完成，耗时: {elapsed:.2f}秒")
            logger.info(f"  流向: {output.flow_direction}")
            logger.info(f"  流量累积: {output.flow_accumulation}")
            
            return True
        
        except Exception as e:
            logger.error(f"✗ 步骤1失败: {e}")
            self.validation_results['step01'] = {
                "passed": False,
                "error": str(e)
            }
            return False
    
    def _validate_step01(self, output_dir: Path):
        """验证步骤1"""
        validation = {"passed": True, "checks": []}
        
        # 检查文件存在
        required_files = [
            "flow_direction.tif",
            "flow_accumulation.tif"
        ]
        
        for filename in required_files:
            file_path = output_dir / filename
            if file_path.exists():
                validation["checks"].append({
                    "name": f"{filename}存在",
                    "passed": True
                })
            else:
                validation["checks"].append({
                    "name": f"{filename}存在",
                    "passed": False,
                    "error": f"文件不存在: {file_path}"
                })
                validation["passed"] = False
        
        # 验证栅格数据
        try:
            import rasterio
            import numpy as np
            
            flow_acc_path = output_dir / "flow_accumulation.tif"
            with rasterio.open(flow_acc_path) as src:
                data = src.read(1)
                
                # 检查无效值
                valid_data = data[data >= 0]
                if len(valid_data) == 0:
                    validation["checks"].append({
                        "name": "流量累积有效性",
                        "passed": False,
                        "error": "所有值都无效"
                    })
                    validation["passed"] = False
                else:
                    validation["checks"].append({
                        "name": "流量累积有效性",
                        "passed": True,
                        "stats": {
                            "min": float(valid_data.min()),
                            "max": float(valid_data.max()),
                            "mean": float(valid_data.mean())
                        }
                    })
        
        except Exception as e:
            validation["checks"].append({
                "name": "栅格数据验证",
                "passed": False,
                "error": str(e)
            })
            validation["passed"] = False
        
        self.validation_results['step01'] = validation
        
        if validation["passed"]:
            logger.info("  ✓ 验证通过")
        else:
            logger.warning("  ⚠ 验证发现问题")
    
    def step02_pour_points(self):
        """步骤2: 汇水点生成"""
        logger.info("\n" + "=" * 80)
        logger.info("步骤2: 汇水点生成")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            from hydrosis.modules import PourPointsModule
            
            # 获取上一步输出
            flow_acc = self.step_results.get('step01', {}).get('flow_accumulation')
            if not flow_acc:
                raise ValueError("需要先运行步骤1")
            
            # 获取配置
            partition_config = self.config.get('partition', {})
            threshold = partition_config.get('subzone_accumulation_threshold', 1000.0)
            
            output_dir = self.output_dir / "02_pour_points"
            
            # 执行
            pour_points = PourPointsModule()
            output = pour_points.run({
                "flow_accumulation": flow_acc,
                "method": "auto",
                "threshold": threshold,
                "output_dir": str(output_dir)
            })
            
            # 保存结果
            self.step_results['step02'] = {
                "pour_points": output.pour_points_geojson,
                "points": [
                    {
                        "id": pt.id,
                        "lon": pt.lon,
                        "lat": pt.lat,
                        "accumulation": pt.accumulation
                    }
                    for pt in output.points
                ],
                "metadata": output.metadata
            }
            
            # 验证
            self._validate_step02(output_dir)
            
            elapsed = time.time() - start_time
            logger.info(f"✓ 步骤2完成，耗时: {elapsed:.2f}秒")
            logger.info(f"  识别到 {len(output.points)} 个汇水点")
            
            return True
        
        except Exception as e:
            logger.error(f"✗ 步骤2失败: {e}")
            self.validation_results['step02'] = {
                "passed": False,
                "error": str(e)
            }
            return False
    
    def _validate_step02(self, output_dir: Path):
        """验证步骤2"""
        validation = {"passed": True, "checks": []}
        
        # 检查文件
        pour_points_file = output_dir / "pour_points.geojson"
        if not pour_points_file.exists():
            validation["checks"].append({
                "name": "汇水点文件存在",
                "passed": False,
                "error": "文件不存在"
            })
            validation["passed"] = False
            self.validation_results['step02'] = validation
            return
        
        # 验证GeoJSON
        try:
            with open(pour_points_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            features = data.get('features', [])
            point_count = len(features)
            
            validation["checks"].append({
                "name": "汇水点数量",
                "passed": point_count > 0,
                "value": point_count
            })
            
            if point_count == 0:
                validation["passed"] = False
            
            # 验证每个点
            for i, feature in enumerate(features):
                props = feature.get('properties', {})
                acc = props.get('accumulation')
                
                if acc is None or acc <= 0:
                    validation["checks"].append({
                        "name": f"汇水点{i}流量累积",
                        "passed": False,
                        "error": f"无效的流量累积值: {acc}"
                    })
                    validation["passed"] = False
        
        except Exception as e:
            validation["checks"].append({
                "name": "GeoJSON验证",
                "passed": False,
                "error": str(e)
            })
            validation["passed"] = False
        
        self.validation_results['step02'] = validation
        
        if validation["passed"]:
            logger.info("  ✓ 验证通过")
        else:
            logger.warning("  ⚠ 验证发现问题")
    
    def run_all_steps(self):
        """运行所有步骤"""
        logger.info("\n" + "#" * 80)
        logger.info("开始执行完整的十一步工作流测试")
        logger.info("#" * 80)
        
        total_start = time.time()
        
        # 步骤1-2已实现，其他步骤待实现
        steps = [
            ("步骤1: 地形处理", self.step01_terrain_processing),
            ("步骤2: 汇水点生成", self.step02_pour_points),
            # 步骤3-11待实现...
        ]
        
        passed_steps = 0
        failed_steps = 0
        
        for step_name, step_func in steps:
            try:
                if step_func():
                    passed_steps += 1
                else:
                    failed_steps += 1
            except Exception as e:
                logger.error(f"{step_name}执行异常: {e}")
                failed_steps += 1
        
        total_elapsed = time.time() - total_start
        
        # 生成报告
        self._generate_report(total_elapsed, passed_steps, failed_steps)
        
        return failed_steps == 0
    
    def _generate_report(self, total_time: float, passed: int, failed: int):
        """生成测试报告"""
        logger.info("\n" + "#" * 80)
        logger.info("测试完成报告")
        logger.info("#" * 80)
        
        logger.info(f"\n总耗时: {total_time:.2f} 秒")
        logger.info(f"通过步骤: {passed}")
        logger.info(f"失败步骤: {failed}")
        logger.info(f"通过率: {passed/(passed+failed)*100:.1f}%")
        
        # 保存JSON报告
        report = {
            "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config_file": str(self.config_path),
            "output_dir": str(self.output_dir),
            "total_time_seconds": total_time,
            "passed_steps": passed,
            "failed_steps": failed,
            "step_results": self.step_results,
            "validation_results": self.validation_results
        }
        
        report_file = self.output_dir / "test_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n测试报告已保存: {report_file}")
        
        # 生成Markdown报告
        self._generate_markdown_report(report)
    
    def _generate_markdown_report(self, report: Dict[str, Any]):
        """生成Markdown格式的测试报告"""
        md_file = self.output_dir / "test_report.md"
        
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write("# HydroSIS 十一步工作流测试报告\n\n")
            
            f.write("## 测试概览\n\n")
            f.write(f"- **测试时间**: {report['test_date']}\n")
            f.write(f"- **配置文件**: {report['config_file']}\n")
            f.write(f"- **输出目录**: {report['output_dir']}\n")
            f.write(f"- **总耗时**: {report['total_time_seconds']:.2f} 秒\n")
            f.write(f"- **通过步骤**: {report['passed_steps']}\n")
            f.write(f"- **失败步骤**: {report['failed_steps']}\n")
            f.write(f"- **通过率**: {report['passed_steps']/(report['passed_steps']+report['failed_steps'])*100:.1f}%\n\n")
            
            f.write("## 详细结果\n\n")
            
            for step_id, validation in report['validation_results'].items():
                status = "✅" if validation.get('passed', False) else "❌"
                f.write(f"### {step_id} {status}\n\n")
                
                if not validation.get('passed', False) and 'error' in validation:
                    f.write(f"**错误**: {validation['error']}\n\n")
                
                if 'checks' in validation:
                    f.write("**检查项**:\n\n")
                    for check in validation['checks']:
                        check_status = "✅" if check.get('passed', False) else "❌"
                        f.write(f"- {check_status} {check['name']}\n")
                        
                        if 'error' in check:
                            f.write(f"  - 错误: {check['error']}\n")
                        
                        if 'stats' in check:
                            f.write(f"  - 统计: {check['stats']}\n")
                    
                    f.write("\n")
            
            f.write("## 输出文件\n\n")
            for step_id, result in report['step_results'].items():
                f.write(f"### {step_id}\n\n")
                for key, value in result.items():
                    if key != 'metadata':
                        f.write(f"- **{key}**: `{value}`\n")
                f.write("\n")
        
        logger.info(f"Markdown报告已保存: {md_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="HydroSIS十一步工作流完整测试")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/upper_truckee_project.yml"),
        help="项目配置文件路径"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/eleven_steps_test"),
        help="测试输出目录"
    )
    
    args = parser.parse_args()
    
    # 运行测试
    test = ElevenStepWorkflowTest(args.config, args.output)
    success = test.run_all_steps()
    
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
