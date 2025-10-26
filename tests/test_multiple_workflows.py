"""多工作流测试套件

测试各种工作流场景，验证模块化架构的灵活性和可靠性。
"""
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WorkflowTestSuite:
    """工作流测试套件"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_results: Dict[str, Dict[str, Any]] = {}
    
    def test_workflow(self, workflow_config: Path, test_name: str) -> bool:
        """测试单个工作流
        
        Args:
            workflow_config: 工作流配置文件路径
            test_name: 测试名称
        
        Returns:
            测试是否通过
        """
        logger.info("\n" + "=" * 80)
        logger.info(f"测试工作流: {test_name}")
        logger.info(f"配置文件: {workflow_config}")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            from hydrosis.workflow_engine import WorkflowEngine, WorkflowDefinition
            
            # 加载工作流
            if not workflow_config.exists():
                logger.error(f"配置文件不存在: {workflow_config}")
                self.test_results[test_name] = {
                    "passed": False,
                    "error": f"配置文件不存在: {workflow_config}",
                    "duration": 0
                }
                return False
            
            workflow = WorkflowDefinition.from_yaml(workflow_config)
            logger.info(f"工作流: {workflow.name}")
            logger.info(f"描述: {workflow.description}")
            logger.info(f"步骤数: {len(workflow.steps)}")
            
            # 列出步骤
            for i, step in enumerate(workflow.steps, 1):
                deps = f" (依赖: {', '.join(step.depends_on)})" if step.depends_on else ""
                logger.info(f"  {i}. {step.id}: {step.module}{deps}")
            
            # 创建引擎
            engine = WorkflowEngine()
            
            # 执行工作流
            logger.info("\n开始执行...")
            
            progress_log = []
            
            def progress_callback(run, step_result):
                percent = run.progress_percent()
                msg = f"[{percent:5.1f}%] {step_result.step_id:25s} -> {step_result.status}"
                logger.info(msg)
                progress_log.append({
                    "percent": percent,
                    "step": step_result.step_id,
                    "status": step_result.status
                })
            
            run = engine.execute(
                workflow,
                progress_callback=progress_callback
            )
            
            elapsed = time.time() - start_time
            
            # 记录结果
            logger.info(f"\n工作流执行{run.status}")
            logger.info(f"运行ID: {run.run_id}")
            logger.info(f"耗时: {elapsed:.2f}秒")
            logger.info(f"进度: {run.progress_percent():.1f}%")
            
            # 步骤详情
            logger.info("\n步骤结果:")
            step_summary = []
            for step_id, result in run.step_results.items():
                status_icon = "✓" if result.status == "completed" else "✗"
                logger.info(f"  {status_icon} {step_id}: {result.status}")
                if result.duration_seconds():
                    logger.info(f"     耗时: {result.duration_seconds():.2f}秒")
                if result.error:
                    logger.error(f"     错误: {result.error}")
                
                step_summary.append({
                    "step_id": step_id,
                    "status": result.status,
                    "duration": result.duration_seconds(),
                    "error": result.error
                })
            
            # 输出
            if run.outputs:
                logger.info("\n工作流输出:")
                for key, value in run.outputs.items():
                    logger.info(f"  {key}: {value}")
            
            # 保存结果
            passed = run.status == "completed"
            self.test_results[test_name] = {
                "passed": passed,
                "workflow_id": workflow.id,
                "workflow_name": workflow.name,
                "duration": elapsed,
                "step_count": len(workflow.steps),
                "step_results": step_summary,
                "outputs": run.outputs,
                "error": run.error,
                "progress_log": progress_log
            }
            
            if passed:
                logger.info(f"\n✅ 测试通过: {test_name}")
            else:
                logger.error(f"\n❌ 测试失败: {test_name}")
                if run.error:
                    logger.error(f"   错误: {run.error}")
            
            return passed
        
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"✗ 测试执行异常: {e}")
            import traceback
            traceback.print_exc()
            
            self.test_results[test_name] = {
                "passed": False,
                "error": str(e),
                "duration": elapsed,
                "exception": traceback.format_exc()
            }
            return False
    
    def run_all_tests(self) -> bool:
        """运行所有工作流测试"""
        logger.info("\n" + "#" * 80)
        logger.info("HydroSIS 多工作流测试套件")
        logger.info("#" * 80)
        
        total_start = time.time()
        
        # 定义测试场景
        test_scenarios = [
            ("01_最小测试-仅地形", "config/workflows/test_scenarios/01_minimal_terrain.yaml"),
            ("02_两步基础测试", "config/workflows/test_scenarios/02_two_step_basic.yaml"),
            ("03_三步流域划分", "config/workflows/test_scenarios/03_three_step_delineation.yaml"),
            ("04_降雨分析", "config/workflows/test_scenarios/04_precipitation_analysis.yaml"),
            ("05_水文模拟", "config/workflows/test_scenarios/05_hydrologic_simulation.yaml"),
            ("06_参数率定", "config/workflows/test_scenarios/06_calibration_workflow.yaml"),
            ("07_并行分析", "config/workflows/test_scenarios/07_parallel_analysis.yaml"),
            ("08_完整十一步", "config/workflows/test_scenarios/08_complete_eleven_steps.yaml"),
        ]
        
        # 运行测试
        for test_name, config_path in test_scenarios:
            config_file = Path(config_path)
            self.test_workflow(config_file, test_name)
        
        total_elapsed = time.time() - total_start
        
        # 生成报告
        self._generate_report(total_elapsed)
        
        # 返回是否所有测试通过
        passed = all(result.get("passed", False) for result in self.test_results.values())
        return passed
    
    def _generate_report(self, total_time: float):
        """生成测试报告"""
        logger.info("\n" + "#" * 80)
        logger.info("测试总结报告")
        logger.info("#" * 80)
        
        passed = sum(1 for r in self.test_results.values() if r.get("passed", False))
        failed = len(self.test_results) - passed
        
        logger.info(f"\n总耗时: {total_time:.2f}秒")
        logger.info(f"测试总数: {len(self.test_results)}")
        logger.info(f"通过: {passed}")
        logger.info(f"失败: {failed}")
        logger.info(f"通过率: {passed/len(self.test_results)*100:.1f}%")
        
        logger.info("\n详细结果:")
        for test_name, result in self.test_results.items():
            icon = "✅" if result.get("passed", False) else "❌"
            duration = result.get("duration", 0)
            logger.info(f"  {icon} {test_name:30s} ({duration:6.2f}秒)")
            if not result.get("passed", False) and "error" in result:
                logger.info(f"      错误: {result['error']}")
        
        # 保存JSON报告
        report = {
            "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_time_seconds": total_time,
            "total_tests": len(self.test_results),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed/len(self.test_results)*100,
            "test_results": self.test_results
        }
        
        report_file = self.output_dir / "workflow_test_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nJSON报告已保存: {report_file}")
        
        # 生成Markdown报告
        self._generate_markdown_report(report)
    
    def _generate_markdown_report(self, report: Dict[str, Any]):
        """生成Markdown报告"""
        md_file = self.output_dir / "workflow_test_report.md"
        
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write("# HydroSIS 多工作流测试报告\n\n")
            
            f.write("## 测试概览\n\n")
            f.write(f"- **测试日期**: {report['test_date']}\n")
            f.write(f"- **总耗时**: {report['total_time_seconds']:.2f}秒\n")
            f.write(f"- **测试总数**: {report['total_tests']}\n")
            f.write(f"- **通过**: {report['passed']}\n")
            f.write(f"- **失败**: {report['failed']}\n")
            f.write(f"- **通过率**: {report['pass_rate']:.1f}%\n\n")
            
            f.write("## 测试结果总览\n\n")
            f.write("| 测试名称 | 状态 | 耗时(秒) | 步骤数 | 说明 |\n")
            f.write("|---------|------|---------|--------|------|\n")
            
            for test_name, result in report['test_results'].items():
                status = "✅" if result.get("passed", False) else "❌"
                duration = result.get("duration", 0)
                step_count = result.get("step_count", 0)
                workflow_name = result.get("workflow_name", "")
                f.write(f"| {test_name} | {status} | {duration:.2f} | {step_count} | {workflow_name} |\n")
            
            f.write("\n## 详细测试结果\n\n")
            
            for test_name, result in report['test_results'].items():
                status = "✅ 通过" if result.get("passed", False) else "❌ 失败"
                f.write(f"### {test_name} - {status}\n\n")
                
                f.write(f"- **工作流名称**: {result.get('workflow_name', 'N/A')}\n")
                f.write(f"- **工作流ID**: {result.get('workflow_id', 'N/A')}\n")
                f.write(f"- **耗时**: {result.get('duration', 0):.2f}秒\n")
                f.write(f"- **步骤数**: {result.get('step_count', 0)}\n\n")
                
                if "error" in result and not result.get("passed", False):
                    f.write(f"**错误信息**:\n```\n{result['error']}\n```\n\n")
                
                if "step_results" in result:
                    f.write("**步骤执行情况**:\n\n")
                    for step in result['step_results']:
                        step_status = "✅" if step['status'] == "completed" else "❌"
                        f.write(f"- {step_status} {step['step_id']}: {step['status']}")
                        if step.get('duration'):
                            f.write(f" ({step['duration']:.2f}秒)")
                        f.write("\n")
                        if step.get('error'):
                            f.write(f"  - 错误: {step['error']}\n")
                    f.write("\n")
                
                if "outputs" in result and result['outputs']:
                    f.write("**工作流输出**:\n\n")
                    for key, value in result['outputs'].items():
                        f.write(f"- `{key}`: {value}\n")
                    f.write("\n")
        
        logger.info(f"Markdown报告已保存: {md_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="HydroSIS多工作流测试")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/workflow_tests"),
        help="测试输出目录"
    )
    parser.add_argument(
        "--test",
        type=str,
        help="只运行指定的测试（测试名称或配置文件路径）"
    )
    
    args = parser.parse_args()
    
    # 创建测试套件
    suite = WorkflowTestSuite(args.output)
    
    # 运行测试
    if args.test:
        # 运行单个测试
        config_path = Path(args.test)
        if not config_path.exists():
            logger.error(f"配置文件不存在: {config_path}")
            return 1
        
        success = suite.test_workflow(config_path, config_path.stem)
        return 0 if success else 1
    else:
        # 运行所有测试
        success = suite.run_all_tests()
        return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
