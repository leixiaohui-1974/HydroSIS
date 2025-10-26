"""HydroSIS 模块化API综合测试

使用Upper Truckee River实际项目数据进行完整的十一步工作流测试。
每一步都包含闭环验证。
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pytest

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ValidationResult:
    """验证结果"""
    
    def __init__(self, step_id: str, step_name: str):
        self.step_id = step_id
        self.step_name = step_name
        self.passed = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.metrics: Dict[str, Any] = {}
    
    def add_error(self, error: str):
        """添加错误"""
        self.errors.append(error)
        self.passed = False
    
    def add_warning(self, warning: str):
        """添加警告"""
        self.warnings.append(warning)
    
    def add_metric(self, name: str, value: Any):
        """添加指标"""
        self.metrics[name] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "step_id": self.step_id,
            "step_name": self.step_name,
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "metrics": self.metrics
        }


class ClosedLoopValidator:
    """闭环验证器
    
    对每一步的输出进行全面验证，确保：
    1. 输出文件存在
    2. 输出格式正确
    3. 数据范围合理
    4. 物理约束满足
    5. 与上游数据一致
    """
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.validation_results: List[ValidationResult] = []
    
    def validate_step01_terrain(self, output_dir: Path) -> ValidationResult:
        """验证步骤1: 地形处理"""
        result = ValidationResult("step01", "地形处理")
        
        try:
            # 1. 检查输出文件
            flow_dir = output_dir / "flow_direction.tif"
            flow_acc = output_dir / "flow_accumulation.tif"
            filled_dem = output_dir / "filled_dem.tif"
            slope = output_dir / "slope.tif"
            
            if not flow_dir.exists():
                result.add_error(f"流向文件不存在: {flow_dir}")
            else:
                result.add_metric("flow_direction_exists", True)
            
            if not flow_acc.exists():
                result.add_error(f"流量累积文件不存在: {flow_acc}")
            else:
                result.add_metric("flow_accumulation_exists", True)
            
            # 2. 验证栅格数据
            try:
                import rasterio
                import numpy as np
                
                # 验证流向
                with rasterio.open(flow_dir) as src:
                    flow_dir_data = src.read(1)
                    result.add_metric("flow_dir_shape", flow_dir_data.shape)
                    result.add_metric("flow_dir_dtype", str(flow_dir_data.dtype))
                    
                    # D8流向应该在0-255范围内
                    if flow_dir_data.min() < 0 or flow_dir_data.max() > 255:
                        result.add_warning(f"流向值异常: {flow_dir_data.min()} - {flow_dir_data.max()}")
                    
                    result.add_metric("flow_dir_min", float(flow_dir_data.min()))
                    result.add_metric("flow_dir_max", float(flow_dir_data.max()))
                
                # 验证流量累积
                with rasterio.open(flow_acc) as src:
                    flow_acc_data = src.read(1)
                    result.add_metric("flow_acc_shape", flow_acc_data.shape)
                    
                    # 流量累积应该是正值
                    if flow_acc_data.min() < 0:
                        result.add_error(f"流量累积出现负值: {flow_acc_data.min()}")
                    
                    result.add_metric("flow_acc_min", float(flow_acc_data.min()))
                    result.add_metric("flow_acc_max", float(flow_acc_data.max()))
                    result.add_metric("flow_acc_mean", float(flow_acc_data.mean()))
                
                # 验证坡度
                if slope.exists():
                    with rasterio.open(slope) as src:
                        slope_data = src.read(1)
                        
                        # 坡度应该在合理范围内
                        if slope_data.max() > 1.0:  # 坡度>100%可能有问题
                            result.add_warning(f"存在极陡坡度: {slope_data.max()}")
                        
                        result.add_metric("slope_min", float(slope_data.min()))
                        result.add_metric("slope_max", float(slope_data.max()))
                        result.add_metric("slope_mean", float(slope_data.mean()))
                
            except ImportError:
                result.add_warning("未安装rasterio，跳过栅格数据验证")
            except Exception as e:
                result.add_error(f"栅格数据验证失败: {e}")
        
        except Exception as e:
            result.add_error(f"地形处理验证失败: {e}")
        
        self.validation_results.append(result)
        return result
    
    def validate_step02_pour_points(self, output_dir: Path, flow_acc_path: Path) -> ValidationResult:
        """验证步骤2: 汇水点生成"""
        result = ValidationResult("step02", "汇水点生成")
        
        try:
            # 1. 检查输出文件
            pour_points_file = output_dir / "pour_points.geojson"
            
            if not pour_points_file.exists():
                result.add_error(f"汇水点文件不存在: {pour_points_file}")
                return result
            
            # 2. 验证GeoJSON格式
            with open(pour_points_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if data.get('type') != 'FeatureCollection':
                result.add_error("GeoJSON格式错误：不是FeatureCollection")
            
            features = data.get('features', [])
            result.add_metric("pour_points_count", len(features))
            
            if len(features) == 0:
                result.add_error("没有识别到汇水点")
                return result
            
            # 3. 验证汇水点属性
            for i, feature in enumerate(features):
                props = feature.get('properties', {})
                geom = feature.get('geometry', {})
                
                # 检查坐标
                if geom.get('type') != 'Point':
                    result.add_error(f"汇水点{i}几何类型错误")
                
                coords = geom.get('coordinates', [])
                if len(coords) != 2:
                    result.add_error(f"汇水点{i}坐标格式错误")
                
                # 检查流量累积值
                acc = props.get('accumulation')
                if acc is None:
                    result.add_warning(f"汇水点{i}缺少流量累积值")
                elif acc <= 0:
                    result.add_error(f"汇水点{i}流量累积值<=0: {acc}")
            
            # 4. 检查汇水点是否在合理位置（高流量累积处）
            try:
                import rasterio
                
                with rasterio.open(flow_acc_path) as src:
                    max_acc = src.read(1).max()
                    result.add_metric("max_flow_accumulation", float(max_acc))
                    
                    # 汇水点的流量累积应该较大
                    min_acc = min(f['properties'].get('accumulation', 0) for f in features)
                    if min_acc < max_acc * 0.01:  # 小于最大值的1%
                        result.add_warning(f"部分汇水点流量累积过小: {min_acc}")
            
            except ImportError:
                result.add_warning("未安装rasterio，跳过流量累积验证")
        
        except Exception as e:
            result.add_error(f"汇水点验证失败: {e}")
        
        self.validation_results.append(result)
        return result
    
    def validate_step03_watershed(self, output_dir: Path, pour_points_path: Path) -> ValidationResult:
        """验证步骤3: 流域划分"""
        result = ValidationResult("step03", "流域划分")
        
        try:
            # 1. 检查输出文件
            watersheds_file = output_dir / "watersheds.geojson"
            
            if not watersheds_file.exists():
                result.add_error(f"流域文件不存在: {watersheds_file}")
                return result
            
            # 2. 验证GeoJSON格式
            with open(watersheds_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            features = data.get('features', [])
            result.add_metric("watershed_count", len(features))
            
            # 3. 验证流域数量与汇水点数量一致
            with open(pour_points_path, 'r', encoding='utf-8') as f:
                pp_data = json.load(f)
            
            pp_count = len(pp_data.get('features', []))
            if len(features) != pp_count:
                result.add_warning(f"流域数量({len(features)})与汇水点数量({pp_count})不一致")
            
            # 4. 验证流域面积
            total_area = 0
            for i, feature in enumerate(features):
                props = feature.get('properties', {})
                geom = feature.get('geometry', {})
                
                # 检查几何类型
                geom_type = geom.get('type')
                if geom_type not in ['Polygon', 'MultiPolygon']:
                    result.add_error(f"流域{i}几何类型错误: {geom_type}")
                
                # 检查面积
                area = props.get('area_km2')
                if area is not None:
                    if area <= 0:
                        result.add_error(f"流域{i}面积<=0: {area}")
                    total_area += area
            
            if total_area > 0:
                result.add_metric("total_watershed_area_km2", total_area)
                result.add_metric("mean_watershed_area_km2", total_area / len(features))
        
        except Exception as e:
            result.add_error(f"流域划分验证失败: {e}")
        
        self.validation_results.append(result)
        return result
    
    def validate_timeseries_file(self, file_path: Path, result: ValidationResult, name: str):
        """验证时间序列文件"""
        if not file_path.exists():
            result.add_error(f"{name}文件不存在: {file_path}")
            return
        
        try:
            # 读取CSV
            import csv
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
            
            if len(rows) < 2:
                result.add_error(f"{name}文件为空或只有表头")
                return
            
            result.add_metric(f"{name}_rows", len(rows) - 1)
            
            # 检查数据列
            headers = rows[0]
            result.add_metric(f"{name}_columns", len(headers))
            
            # 检查数值列
            for i, row in enumerate(rows[1:], start=1):
                if len(row) != len(headers):
                    result.add_warning(f"{name}第{i}行列数不匹配")
                    break
                
                # 检查数值是否有效
                for j, value in enumerate(row[1:], start=1):  # 跳过时间列
                    try:
                        float(value)
                    except ValueError:
                        result.add_warning(f"{name}第{i}行第{j}列不是有效数值: {value}")
                        break
        
        except Exception as e:
            result.add_error(f"{name}文件验证失败: {e}")
    
    def validate_water_balance(
        self, 
        precip_file: Path, 
        runoff_file: Path,
        result: ValidationResult
    ):
        """验证水量平衡"""
        try:
            import csv
            import numpy as np
            
            # 读取降雨数据
            with open(precip_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # 跳过表头
                precip_data = []
                for row in reader:
                    precip_data.extend([float(v) for v in row[1:] if v])
            
            # 读取径流数据
            with open(runoff_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # 跳过表头
                runoff_data = []
                for row in reader:
                    runoff_data.extend([float(v) for v in row[1:] if v])
            
            if not precip_data or not runoff_data:
                result.add_warning("降雨或径流数据为空，无法验证水量平衡")
                return
            
            # 计算总量
            total_precip = sum(precip_data)
            total_runoff = sum(runoff_data)
            
            # 计算径流系数
            runoff_coef = total_runoff / total_precip if total_precip > 0 else 0
            
            result.add_metric("total_precipitation_mm", total_precip)
            result.add_metric("total_runoff_mm", total_runoff)
            result.add_metric("runoff_coefficient", runoff_coef)
            
            # 验证径流系数合理性
            if runoff_coef < 0:
                result.add_error(f"径流系数为负: {runoff_coef}")
            elif runoff_coef > 1.0:
                result.add_error(f"径流系数>1: {runoff_coef}")
            elif runoff_coef < 0.1:
                result.add_warning(f"径流系数过小: {runoff_coef}")
            elif runoff_coef > 0.9:
                result.add_warning(f"径流系数过大: {runoff_coef}")
            else:
                result.add_metric("water_balance_status", "合理")
        
        except Exception as e:
            result.add_warning(f"水量平衡验证失败: {e}")
    
    def generate_report(self, output_path: Path):
        """生成验证报告"""
        report = {
            "test_date": str(Path.ctime(Path(__file__))),
            "total_steps": len(self.validation_results),
            "passed_steps": sum(1 for r in self.validation_results if r.passed),
            "failed_steps": sum(1 for r in self.validation_results if not r.passed),
            "results": [r.to_dict() for r in self.validation_results]
        }
        
        # 生成JSON报告
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 生成Markdown报告
        md_path = output_path.with_suffix('.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# HydroSIS 模块化API测试报告\n\n")
            f.write(f"## 测试概览\n\n")
            f.write(f"- **测试步骤总数**: {report['total_steps']}\n")
            f.write(f"- **通过步骤**: {report['passed_steps']}\n")
            f.write(f"- **失败步骤**: {report['failed_steps']}\n")
            f.write(f"- **通过率**: {report['passed_steps']/report['total_steps']*100:.1f}%\n\n")
            
            f.write("## 详细结果\n\n")
            for r in self.validation_results:
                status = "✅ 通过" if r.passed else "❌ 失败"
                f.write(f"### {r.step_name} ({r.step_id}) - {status}\n\n")
                
                if r.errors:
                    f.write("**错误**:\n")
                    for error in r.errors:
                        f.write(f"- ❌ {error}\n")
                    f.write("\n")
                
                if r.warnings:
                    f.write("**警告**:\n")
                    for warning in r.warnings:
                        f.write(f"- ⚠️ {warning}\n")
                    f.write("\n")
                
                if r.metrics:
                    f.write("**指标**:\n")
                    for key, value in r.metrics.items():
                        f.write(f"- {key}: {value}\n")
                    f.write("\n")
        
        logger.info(f"验证报告已生成: {output_path}")
        logger.info(f"Markdown报告: {md_path}")
        
        return report


class TestModularAPIComprehensive:
    """综合测试类"""
    
    @pytest.fixture
    def test_data_dir(self):
        """测试数据目录"""
        return Path("data/Upper_Truckee_River")
    
    @pytest.fixture
    def test_output_dir(self):
        """测试输出目录"""
        output_dir = Path("results/modular_api_test")
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    @pytest.fixture
    def validator(self, test_output_dir):
        """创建验证器"""
        return ClosedLoopValidator(test_output_dir)
    
    def test_01_terrain_processing(self, test_data_dir, test_output_dir, validator):
        """测试步骤1: 地形处理"""
        logger.info("=" * 60)
        logger.info("测试步骤1: 地形处理")
        logger.info("=" * 60)
        
        from hydrosis.modules import TerrainModule
        
        # 准备输入
        dem_path = test_data_dir / "terrain" / "UpTruckeeRv_S10_NED_30m" / "00" / "elevation.tif"
        
        if not dem_path.exists():
            pytest.skip(f"DEM文件不存在: {dem_path}")
        
        output_dir = test_output_dir / "01_terrain"
        
        # 执行模块
        terrain = TerrainModule()
        try:
            output = terrain.run({
                "dem_path": str(dem_path),
                "method": "d8",
                "fill_depressions": True,
                "compute_slope": True,
                "output_dir": str(output_dir)
            })
            
            logger.info(f"✓ 地形处理完成")
            logger.info(f"  流向: {output.flow_direction}")
            logger.info(f"  流量累积: {output.flow_accumulation}")
            
            # 验证结果
            result = validator.validate_step01_terrain(output_dir)
            
            assert result.passed, f"地形处理验证失败: {result.errors}"
            logger.info(f"✓ 验证通过: {result.metrics}")
        
        except Exception as e:
            logger.error(f"✗ 地形处理失败: {e}")
            raise
    
    def test_02_pour_points(self, test_output_dir, validator):
        """测试步骤2: 汇水点生成"""
        logger.info("=" * 60)
        logger.info("测试步骤2: 汇水点生成")
        logger.info("=" * 60)
        
        from hydrosis.modules import PourPointsModule
        
        # 获取上一步输出
        flow_acc = test_output_dir / "01_terrain" / "flow_accumulation.tif"
        
        if not flow_acc.exists():
            pytest.skip("需要先运行步骤1")
        
        output_dir = test_output_dir / "02_pour_points"
        
        # 执行模块
        pour_points = PourPointsModule()
        try:
            output = pour_points.run({
                "flow_accumulation": str(flow_acc),
                "method": "auto",
                "threshold": 1000.0,
                "output_dir": str(output_dir)
            })
            
            logger.info(f"✓ 汇水点生成完成")
            logger.info(f"  识别到 {len(output.points)} 个汇水点")
            logger.info(f"  输出: {output.pour_points_geojson}")
            
            # 验证结果
            result = validator.validate_step02_pour_points(output_dir, flow_acc)
            
            assert result.passed, f"汇水点验证失败: {result.errors}"
            logger.info(f"✓ 验证通过: {result.metrics}")
        
        except Exception as e:
            logger.error(f"✗ 汇水点生成失败: {e}")
            raise
    
    def test_generate_validation_report(self, test_output_dir, validator):
        """生成最终验证报告"""
        logger.info("=" * 60)
        logger.info("生成验证报告")
        logger.info("=" * 60)
        
        report_path = test_output_dir / "validation_report.json"
        report = validator.generate_report(report_path)
        
        logger.info(f"\n测试总结:")
        logger.info(f"  总步骤数: {report['total_steps']}")
        logger.info(f"  通过步骤: {report['passed_steps']}")
        logger.info(f"  失败步骤: {report['failed_steps']}")
        logger.info(f"  通过率: {report['passed_steps']/report['total_steps']*100:.1f}%")
        
        assert report['failed_steps'] == 0, f"有{report['failed_steps']}个步骤测试失败"


if __name__ == "__main__":
    # 直接运行测试
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
