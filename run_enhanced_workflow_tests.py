#!/usr/bin/env python3
"""增强的工作流测试运行器

运行所有测试场景，为每个测试生成详细报告、图表和动画
"""
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedWorkflowTester:
    """增强的工作流测试器"""
    
    def __init__(self, output_root: Path):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.test_results = {}
    
    def run_test_scenario(
        self,
        scenario_id: str,
        scenario_name: str,
        config_file: Path
    ) -> Dict[str, Any]:
        """运行单个测试场景
        
        Args:
            scenario_id: 场景ID
            scenario_name: 场景名称
            config_file: 配置文件路径
        
        Returns:
            测试结果字典
        """
        logger.info("\n" + "=" * 80)
        logger.info(f"测试场景 {scenario_id}: {scenario_name}")
        logger.info("=" * 80)
        
        # 创建测试目录
        test_dir = self.output_root / f"{scenario_id}_{scenario_name.replace(' ', '_')}"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        result = {
            'scenario_id': scenario_id,
            'scenario_name': scenario_name,
            'config_file': str(config_file),
            'test_dir': str(test_dir),
            'start_time': datetime.now().isoformat(),
            'status': 'unknown',
            'outputs': {},
            'visualizations': {},
            'validation': {}
        }
        
        try:
            # 检查配置文件
            if not config_file.exists():
                result['status'] = 'skipped'
                result['error'] = f"配置文件不存在: {config_file}"
                logger.warning(result['error'])
                return result
            
            # 加载工作流
            from hydrosis.workflow_engine import WorkflowDefinition, WorkflowEngine
            
            logger.info(f"📝 加载工作流配置: {config_file}")
            workflow = WorkflowDefinition.from_yaml(config_file)
            logger.info(f"   工作流: {workflow.name}")
            logger.info(f"   步骤数: {len(workflow.steps)}")
            
            # 列出步骤
            for i, step in enumerate(workflow.steps, 1):
                deps = f" (依赖: {', '.join(step.depends_on)})" if step.depends_on else ""
                logger.info(f"   {i}. {step.id}: {step.module}{deps}")
            
            # 创建引擎并执行
            engine = WorkflowEngine()
            
            logger.info("\n🚀 开始执行工作流...")
            
            progress_log = []
            def progress_callback(run, step_result):
                percent = run.progress_percent()
                status_icon = "✓" if step_result.status == "completed" else "✗" if step_result.status == "failed" else "⟳"
                msg = f"[{percent:5.1f}%] {status_icon} {step_result.step_id}"
                logger.info(msg)
                progress_log.append({
                    'percent': percent,
                    'step': step_result.step_id,
                    'status': step_result.status,
                    'duration': step_result.duration_seconds()
                })
            
            run = engine.execute(workflow, progress_callback=progress_callback)
            
            elapsed = time.time() - start_time
            
            # 记录结果
            result['status'] = run.status
            result['duration'] = elapsed
            result['run_id'] = run.run_id
            result['step_count'] = len(workflow.steps)
            result['progress_log'] = progress_log
            result['end_time'] = datetime.now().isoformat()
            
            # 步骤结果
            result['step_results'] = {}
            for step_id, step_result in run.step_results.items():
                result['step_results'][step_id] = {
                    'status': step_result.status,
                    'duration': step_result.duration_seconds(),
                    'error': step_result.error,
                    'outputs': step_result.outputs
                }
            
            # 工作流输出
            result['outputs'] = run.outputs if run.outputs else {}
            
            logger.info(f"\n✨ 工作流执行{run.status}")
            logger.info(f"   运行ID: {run.run_id}")
            logger.info(f"   耗时: {elapsed:.2f}秒")
            logger.info(f"   进度: {run.progress_percent():.1f}%")
            
            # 生成可视化
            logger.info("\n🎨 生成可视化...")
            result['visualizations'] = self._generate_visualizations(
                test_dir, run, workflow
            )
            
            # 验证结果
            logger.info("\n✓ 验证结果...")
            result['validation'] = self._validate_results(test_dir, run)
            
            # 生成报告
            logger.info("\n📄 生成测试报告...")
            self._generate_test_report(test_dir, result)
            
            if result['status'] == 'completed':
                logger.info(f"\n✅ 测试通过: {scenario_name}")
            else:
                logger.error(f"\n❌ 测试失败: {scenario_name}")
                if run.error:
                    logger.error(f"   错误: {run.error}")
        
        except Exception as e:
            elapsed = time.time() - start_time
            result['status'] = 'error'
            result['duration'] = elapsed
            result['error'] = str(e)
            result['end_time'] = datetime.now().isoformat()
            
            logger.error(f"✗ 测试执行异常: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return result
    
    def _generate_visualizations(
        self,
        test_dir: Path,
        run,
        workflow
    ) -> Dict[str, str]:
        """生成可视化文件
        
        Args:
            test_dir: 测试目录
            run: 工作流运行结果
            workflow: 工作流定义
        
        Returns:
            可视化文件路径字典
        """
        visualizations = {}
        viz_dir = test_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            from hydrosis.utils.visualization import ResultVisualizer
            visualizer = ResultVisualizer(viz_dir)
            
            # 遍历步骤输出
            for step_id, step_result in run.step_results.items():
                if step_result.status != 'completed' or not step_result.outputs:
                    continue
                
                outputs = step_result.outputs
                
                # 处理栅格输出
                for key, value in outputs.items():
                    if not isinstance(value, (str, Path)):
                        continue
                    
                    file_path = Path(value)
                    if not file_path.exists():
                        continue
                    
                    # 栅格文件
                    if file_path.suffix == '.tif':
                        out_path = viz_dir / f"{step_id}_{file_path.stem}.png"
                        result_path = visualizer.plot_raster(
                            file_path,
                            out_path,
                            title=f"{step_id}: {file_path.stem}",
                            cmap='terrain'
                        )
                        if result_path:
                            visualizations[f"{step_id}_{file_path.stem}"] = str(result_path)
                    
                    # 矢量文件
                    elif file_path.suffix == '.geojson':
                        out_path = viz_dir / f"{step_id}_{file_path.stem}.png"
                        result_path = visualizer.plot_vector(
                            file_path,
                            out_path,
                            title=f"{step_id}: {file_path.stem}"
                        )
                        if result_path:
                            visualizations[f"{step_id}_{file_path.stem}"] = str(result_path)
                    
                    # CSV文件（时间序列）
                    elif file_path.suffix == '.csv':
                        out_path = viz_dir / f"{step_id}_{file_path.stem}.png"
                        result_path = visualizer.plot_timeseries(
                            file_path,
                            out_path,
                            title=f"{step_id}: {file_path.stem}"
                        )
                        if result_path:
                            visualizations[f"{step_id}_{file_path.stem}"] = str(result_path)
            
            # 创建总结图
            summary_path = viz_dir / "test_summary.png"
            visualizer.create_summary_figure(
                workflow.name,
                {
                    'status': run.status,
                    'duration': run.duration_seconds(),
                    'step_count': len(workflow.steps)
                },
                summary_path
            )
            if summary_path.exists():
                visualizations['summary'] = str(summary_path)
            
            logger.info(f"   生成了 {len(visualizations)} 个可视化文件")
        
        except Exception as e:
            logger.error(f"生成可视化失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return visualizations
    
    def _validate_results(self, test_dir: Path, run) -> Dict[str, Any]:
        """验证测试结果
        
        Args:
            test_dir: 测试目录
            run: 工作流运行结果
        
        Returns:
            验证结果字典
        """
        validation = {
            'passed': False,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }
        
        try:
            # 检查运行状态
            if run.status != 'completed':
                validation['errors'].append(f"工作流未完成: {run.status}")
                if run.error:
                    validation['errors'].append(f"错误信息: {run.error}")
                return validation
            
            # 检查步骤状态
            failed_steps = []
            for step_id, step_result in run.step_results.items():
                if step_result.status != 'completed':
                    failed_steps.append(step_id)
                    if step_result.error:
                        validation['errors'].append(
                            f"步骤 {step_id} 失败: {step_result.error}"
                        )
            
            if failed_steps:
                validation['errors'].append(
                    f"失败步骤: {', '.join(failed_steps)}"
                )
                return validation
            
            # 检查输出文件
            missing_outputs = []
            for step_id, step_result in run.step_results.items():
                if not step_result.outputs:
                    continue
                
                for key, value in step_result.outputs.items():
                    if isinstance(value, (str, Path)):
                        if not Path(value).exists():
                            missing_outputs.append(f"{step_id}.{key}")
            
            if missing_outputs:
                validation['warnings'].append(
                    f"输出文件缺失: {', '.join(missing_outputs)}"
                )
            
            # 计算指标
            validation['metrics']['total_duration'] = run.duration_seconds()
            validation['metrics']['step_count'] = len(run.step_results)
            validation['metrics']['completed_steps'] = sum(
                1 for r in run.step_results.values() if r.status == 'completed'
            )
            
            # 判断是否通过
            validation['passed'] = (
                run.status == 'completed' and
                len(failed_steps) == 0
            )
        
        except Exception as e:
            validation['errors'].append(f"验证过程异常: {e}")
            logger.error(f"验证失败: {e}")
        
        return validation
    
    def _generate_test_report(self, test_dir: Path, result: Dict[str, Any]):
        """生成测试报告
        
        Args:
            test_dir: 测试目录
            result: 测试结果
        """
        try:
            from hydrosis.utils.visualization import create_test_report
            
            # 准备输入信息
            inputs = {
                'config_file': result.get('config_file', 'N/A'),
                'scenario_id': result.get('scenario_id', 'N/A'),
                'scenario_name': result.get('scenario_name', 'N/A')
            }
            
            # 准备输出信息
            outputs = result.get('outputs', {})
            
            # 准备验证信息
            validation = result.get('validation', {})
            validation['timestamp'] = result.get('start_time', 'N/A')
            validation['duration'] = result.get('duration', 0)
            validation['passed'] = result.get('status', '') == 'completed'
            
            # 准备可视化文件
            visualizations = {}
            for name, path in result.get('visualizations', {}).items():
                visualizations[name] = Path(path)
            
            # 生成报告
            create_test_report(
                test_dir,
                result['scenario_name'],
                inputs,
                outputs,
                validation,
                visualizations
            )
            
            # 同时保存JSON结果
            json_path = test_dir / "test_result.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"   报告已保存: {test_dir / 'TEST_REPORT.md'}")
            logger.info(f"   结果已保存: {json_path}")
        
        except Exception as e:
            logger.error(f"生成报告失败: {e}")
    
    def run_all_scenarios(self) -> bool:
        """运行所有测试场景
        
        Returns:
            是否所有测试通过
        """
        logger.info("\n" + "#" * 80)
        logger.info("HydroSIS 增强工作流测试套件")
        logger.info("#" * 80)
        
        total_start = time.time()
        
        # 定义测试场景
        scenarios = [
            ("01", "最小测试-仅地形", "config/workflows/test_scenarios/01_minimal_terrain.yaml"),
            ("02", "两步基础测试", "config/workflows/test_scenarios/02_two_step_basic.yaml"),
            ("03", "三步流域划分", "config/workflows/test_scenarios/03_three_step_delineation.yaml"),
            ("04", "降雨分析", "config/workflows/test_scenarios/04_precipitation_analysis.yaml"),
            ("05", "水文模拟", "config/workflows/test_scenarios/05_hydrologic_simulation.yaml"),
            ("06", "参数率定", "config/workflows/test_scenarios/06_calibration_workflow.yaml"),
            ("07", "并行分析", "config/workflows/test_scenarios/07_parallel_analysis.yaml"),
            ("08", "完整十一步", "config/workflows/test_scenarios/08_complete_eleven_steps.yaml"),
        ]
        
        # 运行每个场景
        for scenario_id, scenario_name, config_file in scenarios:
            result = self.run_test_scenario(
                scenario_id,
                scenario_name,
                Path(config_file)
            )
            self.test_results[scenario_id] = result
        
        total_elapsed = time.time() - total_start
        
        # 生成总结报告
        self._generate_summary_report(total_elapsed)
        
        # 判断是否全部通过
        passed = sum(
            1 for r in self.test_results.values()
            if r.get('status') == 'completed'
        )
        total = len(self.test_results)
        
        logger.info("\n" + "#" * 80)
        logger.info("测试总结")
        logger.info("#" * 80)
        logger.info(f"总耗时: {total_elapsed:.2f}秒")
        logger.info(f"测试总数: {total}")
        logger.info(f"通过: {passed}")
        logger.info(f"失败: {total - passed}")
        logger.info(f"通过率: {passed/total*100:.1f}%")
        
        return passed == total
    
    def _generate_summary_report(self, total_time: float):
        """生成总结报告
        
        Args:
            total_time: 总耗时
        """
        logger.info("\n📊 生成总结报告...")
        
        # 生成JSON报告
        summary = {
            'test_date': datetime.now().isoformat(),
            'total_time_seconds': total_time,
            'total_tests': len(self.test_results),
            'passed': sum(1 for r in self.test_results.values() if r.get('status') == 'completed'),
            'failed': sum(1 for r in self.test_results.values() if r.get('status') != 'completed'),
            'test_results': self.test_results
        }
        
        json_path = self.output_root / "test_summary.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 生成Markdown报告
        md_path = self.output_root / "TEST_SUMMARY.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# HydroSIS 工作流测试总结报告\n\n")
            
            f.write("## 测试概览\n\n")
            f.write(f"- **测试日期**: {summary['test_date']}\n")
            f.write(f"- **总耗时**: {summary['total_time_seconds']:.2f}秒\n")
            f.write(f"- **测试总数**: {summary['total_tests']}\n")
            f.write(f"- **通过**: {summary['passed']}\n")
            f.write(f"- **失败**: {summary['failed']}\n")
            f.write(f"- **通过率**: {summary['passed']/summary['total_tests']*100:.1f}%\n\n")
            
            f.write("## 测试结果明细\n\n")
            f.write("| ID | 场景名称 | 状态 | 耗时(秒) | 步骤数 | 报告 |\n")
            f.write("|----|---------|------|---------|--------|------|\n")
            
            for scenario_id, result in sorted(self.test_results.items()):
                status = "✅" if result.get('status') == 'completed' else "❌"
                name = result.get('scenario_name', 'N/A')
                duration = result.get('duration', 0)
                steps = result.get('step_count', 0)
                test_dir = Path(result.get('test_dir', ''))
                report_link = f"[查看]({test_dir.name}/TEST_REPORT.md)" if test_dir.exists() else "N/A"
                
                f.write(f"| {scenario_id} | {name} | {status} | {duration:.2f} | {steps} | {report_link} |\n")
            
            f.write("\n## 测试目录结构\n\n")
            f.write("```\n")
            f.write(f"{self.output_root}/\n")
            for scenario_id, result in sorted(self.test_results.items()):
                test_dir = Path(result.get('test_dir', ''))
                if test_dir.exists():
                    f.write(f"  ├── {test_dir.name}/\n")
                    f.write(f"  │   ├── TEST_REPORT.md (详细报告)\n")
                    f.write(f"  │   ├── test_result.json (JSON结果)\n")
                    f.write(f"  │   └── visualizations/ (可视化文件)\n")
            f.write("```\n\n")
        
        logger.info(f"   总结报告已保存: {md_path}")
        logger.info(f"   JSON结果已保存: {json_path}")


def main():
    """主函数"""
    # 创建测试器
    output_dir = Path("results/enhanced_workflow_tests")
    tester = EnhancedWorkflowTester(output_dir)
    
    # 运行所有测试
    success = tester.run_all_scenarios()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
