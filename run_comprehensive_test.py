#!/usr/bin/env python3
"""HydroSIS 模块化API综合测试运行脚本

使用Upper Truckee River实际数据进行完整测试。
"""
import sys
import time
import logging
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_run.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def test_module_registry():
    """测试模块注册系统"""
    logger.info("\n" + "=" * 80)
    logger.info("测试1: 模块注册系统")
    logger.info("=" * 80)
    
    try:
        from hydrosis.modules.base import get_registry
        
        registry = get_registry()
        modules = registry.list_modules()
        
        logger.info(f"✓ 已注册 {len(modules)} 个模块:")
        for module_id in modules:
            logger.info(f"  - {module_id}")
        
        # 测试获取模块信息
        if 'terrain' in modules:
            terrain = registry.get_or_create_module('terrain')
            info = terrain.get_info()
            logger.info(f"\n模块 'terrain' 信息:")
            logger.info(f"  名称: {info['name']}")
            logger.info(f"  描述: {info['description']}")
            logger.info(f"  版本: {info['version']}")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ 模块注册系统测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_terrain_module():
    """测试地形处理模块"""
    logger.info("\n" + "=" * 80)
    logger.info("测试2: 地形处理模块")
    logger.info("=" * 80)
    
    try:
        from hydrosis.modules import TerrainModule
        
        # 查找DEM文件
        dem_paths = [
            Path("data/Upper_Truckee_River/terrain/UpTruckeeRv_S10_NED_30m/00/elevation.tif"),
            Path("data/Upper_Truckee_River/terrain/elevation.tif"),
            Path("data/dem.tif")
        ]
        
        dem_path = None
        for path in dem_paths:
            if path.exists():
                dem_path = path
                break
        
        if dem_path is None:
            logger.warning("⚠ 未找到DEM文件，跳过地形处理测试")
            logger.info("  请准备DEM文件到以下位置之一:")
            for path in dem_paths:
                logger.info(f"    - {path}")
            return None
        
        logger.info(f"使用DEM文件: {dem_path}")
        
        # 创建输出目录
        output_dir = Path("results/test_output/terrain")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 执行模块
        terrain = TerrainModule()
        start_time = time.time()
        
        output = terrain.run({
            "dem_path": str(dem_path),
            "method": "d8",
            "fill_depressions": True,
            "compute_slope": True,
            "output_dir": str(output_dir)
        })
        
        elapsed = time.time() - start_time
        
        logger.info(f"✓ 地形处理完成，耗时: {elapsed:.2f}秒")
        logger.info(f"  流向文件: {output.flow_direction}")
        logger.info(f"  流量累积: {output.flow_accumulation}")
        logger.info(f"  填充DEM: {output.filled_dem}")
        logger.info(f"  坡度文件: {output.slope}")
        
        # 验证输出
        if Path(output.flow_direction).exists():
            logger.info("  ✓ 流向文件已生成")
        else:
            logger.error("  ✗ 流向文件未生成")
            return False
        
        if Path(output.flow_accumulation).exists():
            logger.info("  ✓ 流量累积文件已生成")
        else:
            logger.error("  ✗ 流量累积文件未生成")
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"✗ 地形处理模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pour_points_module():
    """测试汇水点生成模块"""
    logger.info("\n" + "=" * 80)
    logger.info("测试3: 汇水点生成模块")
    logger.info("=" * 80)
    
    try:
        from hydrosis.modules import PourPointsModule
        
        # 检查流量累积文件
        flow_acc_path = Path("results/test_output/terrain/flow_accumulation.tif")
        
        if not flow_acc_path.exists():
            logger.warning("⚠ 需要先运行地形处理测试")
            return None
        
        logger.info(f"使用流量累积文件: {flow_acc_path}")
        
        # 创建输出目录
        output_dir = Path("results/test_output/pour_points")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 执行模块
        pour_points = PourPointsModule()
        start_time = time.time()
        
        output = pour_points.run({
            "flow_accumulation": str(flow_acc_path),
            "method": "auto",
            "threshold": 1000.0,
            "output_dir": str(output_dir)
        })
        
        elapsed = time.time() - start_time
        
        logger.info(f"✓ 汇水点生成完成，耗时: {elapsed:.2f}秒")
        logger.info(f"  识别到 {len(output.points)} 个汇水点")
        logger.info(f"  输出文件: {output.pour_points_geojson}")
        
        # 显示汇水点信息
        for i, pt in enumerate(output.points[:5]):  # 只显示前5个
            logger.info(f"  点{i+1}: ID={pt.id}, Acc={pt.accumulation:.0f}, Lon={pt.lon:.4f}, Lat={pt.lat:.4f}")
        
        if len(output.points) > 5:
            logger.info(f"  ... (共{len(output.points)}个点)")
        
        # 验证输出
        if Path(output.pour_points_geojson).exists():
            logger.info("  ✓ 汇水点文件已生成")
        else:
            logger.error("  ✗ 汇水点文件未生成")
            return False
        
        if len(output.points) == 0:
            logger.error("  ✗ 未识别到汇水点")
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"✗ 汇水点生成模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_workflow_engine():
    """测试工作流引擎"""
    logger.info("\n" + "=" * 80)
    logger.info("测试4: 工作流引擎")
    logger.info("=" * 80)
    
    try:
        from hydrosis.workflow_engine import (
            WorkflowEngine,
            WorkflowTemplates,
            WorkflowDefinition
        )
        
        # 列出可用模板
        templates = WorkflowTemplates.list_templates()
        logger.info(f"可用工作流模板: {templates}")
        
        # 检查DEM文件
        dem_paths = [
            Path("data/Upper_Truckee_River/terrain/UpTruckeeRv_S10_NED_30m/00/elevation.tif"),
            Path("data/Upper_Truckee_River/terrain/elevation.tif")
        ]
        
        dem_path = None
        for path in dem_paths:
            if path.exists():
                dem_path = path
                break
        
        if dem_path is None:
            logger.warning("⚠ 未找到DEM文件，跳过工作流测试")
            return None
        
        # 加载工作流模板
        template = WorkflowTemplates.get_template("pour_points_only")
        workflow = WorkflowDefinition.from_dict(template)
        
        logger.info(f"工作流: {workflow.name}")
        logger.info(f"步骤数: {len(workflow.steps)}")
        for step in workflow.steps:
            logger.info(f"  - {step.id}: {step.module}")
        
        # 创建引擎
        engine = WorkflowEngine()
        
        # 执行工作流
        logger.info("\n开始执行工作流...")
        start_time = time.time()
        
        def progress_callback(run, step_result):
            percent = run.progress_percent()
            logger.info(f"  [{percent:5.1f}%] {step_result.step_id:20s} -> {step_result.status}")
        
        run = engine.execute(
            workflow,
            parameters={
                "dem_path": str(dem_path),
                "output_dir": "results/test_output/workflow",
                "threshold": 1500.0
            },
            progress_callback=progress_callback
        )
        
        elapsed = time.time() - start_time
        
        logger.info(f"\n✓ 工作流执行{run.status}")
        logger.info(f"  运行ID: {run.run_id}")
        logger.info(f"  耗时: {elapsed:.2f}秒")
        logger.info(f"  进度: {run.progress_percent():.1f}%")
        
        # 显示每个步骤的结果
        logger.info("\n步骤结果:")
        for step_id, result in run.step_results.items():
            status_icon = "✓" if result.status == "completed" else "✗"
            logger.info(f"  {status_icon} {step_id}: {result.status}")
            if result.duration_seconds():
                logger.info(f"     耗时: {result.duration_seconds():.2f}秒")
        
        # 显示输出
        if run.outputs:
            logger.info("\n工作流输出:")
            for key, value in run.outputs.items():
                logger.info(f"  {key}: {value}")
        
        return run.status == "completed"
    
    except Exception as e:
        logger.error(f"✗ 工作流引擎测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    logger.info("#" * 80)
    logger.info("HydroSIS 模块化API综合测试")
    logger.info("#" * 80)
    
    start_time = time.time()
    
    # 运行测试
    tests = [
        ("模块注册系统", test_module_registry),
        ("地形处理模块", test_terrain_module),
        ("汇水点生成模块", test_pour_points_module),
        ("工作流引擎", test_workflow_engine),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"测试 '{test_name}' 执行异常: {e}")
            results[test_name] = False
    
    total_elapsed = time.time() - start_time
    
    # 统计结果
    logger.info("\n" + "#" * 80)
    logger.info("测试总结")
    logger.info("#" * 80)
    
    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)
    
    logger.info(f"\n总耗时: {total_elapsed:.2f}秒")
    logger.info(f"测试总数: {len(results)}")
    logger.info(f"通过: {passed}")
    logger.info(f"失败: {failed}")
    logger.info(f"跳过: {skipped}")
    
    if passed + skipped == len(results):
        logger.info(f"通过率: 100%")
    else:
        logger.info(f"通过率: {passed/(passed+failed)*100:.1f}%")
    
    logger.info("\n详细结果:")
    for test_name, result in results.items():
        if result is True:
            logger.info(f"  ✅ {test_name}")
        elif result is False:
            logger.info(f"  ❌ {test_name}")
        else:
            logger.info(f"  ⏭️  {test_name} (跳过)")
    
    logger.info("\n" + "#" * 80)
    
    if failed == 0:
        logger.info("🎉 所有测试通过!")
        return 0
    else:
        logger.error(f"❌ 有 {failed} 个测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
