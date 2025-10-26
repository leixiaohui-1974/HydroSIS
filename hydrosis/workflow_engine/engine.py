"""工作流执行引擎"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Callable
import re

from hydrosis.modules.base import Module, ModuleRegistry, get_registry

logger = logging.getLogger(__name__)


@dataclass
class WorkflowStep:
    """工作流步骤定义"""
    
    id: str
    module: str
    inputs: Dict[str, Any]
    outputs: Dict[str, str] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    condition: Optional[str] = None
    retry: int = 0
    timeout: Optional[int] = None


@dataclass
class WorkflowDefinition:
    """工作流定义"""
    
    id: str
    name: str
    version: str = "1.0"
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    steps: List[WorkflowStep] = field(default_factory=list)
    outputs: Dict[str, str] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkflowDefinition:
        """从字典创建工作流定义"""
        workflow_data = data.get('workflow', data)
        
        steps = [
            WorkflowStep(
                id=step['id'],
                module=step['module'],
                inputs=step.get('inputs', {}),
                outputs=step.get('outputs', {}),
                depends_on=step.get('depends_on', []),
                condition=step.get('condition'),
                retry=step.get('retry', 0),
                timeout=step.get('timeout')
            )
            for step in workflow_data.get('steps', [])
        ]
        
        return cls(
            id=workflow_data.get('id', ''),
            name=workflow_data.get('name', ''),
            version=workflow_data.get('version', '1.0'),
            description=workflow_data.get('description', ''),
            parameters=workflow_data.get('parameters', {}),
            steps=steps,
            outputs=workflow_data.get('outputs', {})
        )
    
    @classmethod
    def from_yaml(cls, path: Path) -> WorkflowDefinition:
        """从YAML文件加载工作流定义"""
        try:
            import yaml
            data = yaml.safe_load(path.read_text(encoding='utf-8'))
            return cls.from_dict(data)
        except ImportError:
            raise RuntimeError("需要安装PyYAML来加载YAML文件")


@dataclass
class StepResult:
    """步骤执行结果"""
    
    step_id: str
    status: str  # pending, running, completed, failed, skipped
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def duration_seconds(self) -> Optional[float]:
        """计算执行时长"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


@dataclass
class WorkflowRun:
    """工作流执行实例"""
    
    run_id: str
    workflow_id: str
    status: str  # pending, running, completed, failed
    start_time: datetime
    end_time: Optional[datetime] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    step_results: Dict[str, StepResult] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def duration_seconds(self) -> Optional[float]:
        """计算总执行时长"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def progress_percent(self) -> float:
        """计算进度百分比"""
        if not self.step_results:
            return 0.0
        
        completed = sum(
            1 for result in self.step_results.values()
            if result.status in ['completed', 'skipped']
        )
        total = len(self.step_results)
        return (completed / total * 100) if total > 0 else 0.0


class WorkflowEngine:
    """工作流执行引擎"""
    
    def __init__(self, registry: Optional[ModuleRegistry] = None):
        """初始化工作流引擎
        
        Args:
            registry: 模块注册表，如果为None则使用全局注册表
        """
        self.registry = registry or get_registry()
        self.logger = logging.getLogger("hydrosis.workflow_engine")
        self._runs: Dict[str, WorkflowRun] = {}
    
    def execute(
        self,
        workflow: WorkflowDefinition,
        parameters: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[WorkflowRun, StepResult], None]] = None
    ) -> WorkflowRun:
        """执行工作流
        
        Args:
            workflow: 工作流定义
            parameters: 运行时参数（覆盖工作流定义中的参数）
            progress_callback: 进度回调函数
        
        Returns:
            工作流执行实例
        """
        # 生成运行ID
        run_id = str(uuid.uuid4())
        
        # 合并参数
        run_params = {**workflow.parameters}
        if parameters:
            run_params.update(parameters)
        
        # 创建运行实例
        run = WorkflowRun(
            run_id=run_id,
            workflow_id=workflow.id,
            status="running",
            start_time=datetime.now(),
            parameters=run_params
        )
        
        self._runs[run_id] = run
        
        self.logger.info(f"[{run_id}] 开始执行工作流: {workflow.name}")
        
        try:
            # 初始化所有步骤状态
            for step in workflow.steps:
                run.step_results[step.id] = StepResult(
                    step_id=step.id,
                    status="pending"
                )
            
            # 构建执行顺序（拓扑排序）
            execution_order = self._topological_sort(workflow.steps)
            
            # 执行上下文（用于变量替换）
            context = {
                "parameters": run_params,
                "steps": {}
            }
            
            # 逐步执行
            for step_id in execution_order:
                step = next(s for s in workflow.steps if s.id == step_id)
                
                # 检查依赖是否都已完成
                deps_satisfied = all(
                    run.step_results[dep].status in ['completed', 'skipped']
                    for dep in step.depends_on
                )
                
                if not deps_satisfied:
                    self.logger.warning(f"步骤 {step_id} 的依赖未满足，跳过")
                    run.step_results[step_id].status = "skipped"
                    continue
                
                # 检查条件
                if step.condition and not self._evaluate_condition(step.condition, context):
                    self.logger.info(f"步骤 {step_id} 不满足执行条件，跳过")
                    run.step_results[step_id].status = "skipped"
                    continue
                
                # 执行步骤
                self._execute_step(step, run, context, progress_callback)
            
            # 提取最终输出
            for output_name, output_expr in workflow.outputs.items():
                run.outputs[output_name] = self._resolve_variable(output_expr, context)
            
            # 检查是否所有步骤都成功
            if all(
                result.status in ['completed', 'skipped']
                for result in run.step_results.values()
            ):
                run.status = "completed"
                self.logger.info(f"[{run_id}] 工作流执行成功")
            else:
                run.status = "failed"
                self.logger.error(f"[{run_id}] 工作流执行失败")
            
        except Exception as e:
            run.status = "failed"
            run.error = str(e)
            self.logger.error(f"[{run_id}] 工作流执行出错: {e}")
            raise
        
        finally:
            run.end_time = datetime.now()
        
        return run
    
    def _execute_step(
        self,
        step: WorkflowStep,
        run: WorkflowRun,
        context: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> None:
        """执行单个步骤"""
        step_result = run.step_results[step.id]
        step_result.status = "running"
        step_result.start_time = datetime.now()
        
        self.logger.info(f"[{run.run_id}] 执行步骤: {step.id} ({step.module})")
        
        try:
            # 获取模块
            module = self.registry.get_or_create_module(step.module)
            
            # 解析输入参数
            resolved_inputs = {}
            for key, value in step.inputs.items():
                resolved_inputs[key] = self._resolve_variable(value, context)
            
            # 执行模块
            output = module.run(resolved_inputs, task_id=f"{run.run_id}:{step.id}")
            
            # 保存输出
            step_result.outputs = output.to_dict() if hasattr(output, 'to_dict') else {}
            step_result.status = "completed"
            step_result.end_time = datetime.now()
            
            # 更新上下文
            context['steps'][step.id] = {
                "outputs": step_result.outputs
            }
            
            self.logger.info(
                f"[{run.run_id}] 步骤 {step.id} 完成，"
                f"耗时 {step_result.duration_seconds():.2f} 秒"
            )
            
            # 调用进度回调
            if progress_callback:
                progress_callback(run, step_result)
            
        except Exception as e:
            step_result.status = "failed"
            step_result.error = str(e)
            step_result.end_time = datetime.now()
            self.logger.error(f"[{run.run_id}] 步骤 {step.id} 失败: {e}")
            raise
    
    def _resolve_variable(self, value: Any, context: Dict[str, Any]) -> Any:
        """解析变量引用
        
        支持格式:
        - ${parameters.dem_path}
        - ${steps.terrain_processing.outputs.flow_direction}
        """
        if not isinstance(value, str):
            return value
        
        # 查找所有变量引用
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, value)
        
        if not matches:
            return value
        
        result = value
        for match in matches:
            # 解析路径
            parts = match.split('.')
            current = context
            
            try:
                for part in parts:
                    current = current[part]
                
                # 替换变量
                result = result.replace(f"${{{match}}}", str(current))
            except (KeyError, TypeError):
                self.logger.warning(f"无法解析变量: {match}")
        
        return result
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """评估条件表达式（简化实现）"""
        # TODO: 实现更复杂的条件评估
        return True
    
    def _topological_sort(self, steps: List[WorkflowStep]) -> List[str]:
        """拓扑排序，确定执行顺序"""
        # 构建邻接表和入度
        graph: Dict[str, Set[str]] = {step.id: set() for step in steps}
        in_degree: Dict[str, int] = {step.id: 0 for step in steps}
        
        for step in steps:
            for dep in step.depends_on:
                if dep in graph:
                    graph[dep].add(step.id)
                    in_degree[step.id] += 1
        
        # 找到所有入度为0的节点
        queue = [step_id for step_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            # 取出一个节点
            current = queue.pop(0)
            result.append(current)
            
            # 更新邻居节点的入度
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # 检查是否存在环
        if len(result) != len(steps):
            raise ValueError("工作流存在循环依赖")
        
        return result
    
    def get_run(self, run_id: str) -> Optional[WorkflowRun]:
        """获取运行实例"""
        return self._runs.get(run_id)
    
    def list_runs(self) -> List[WorkflowRun]:
        """列出所有运行实例"""
        return list(self._runs.values())
