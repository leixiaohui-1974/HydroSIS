"""异步任务处理和进度上报"""

import uuid
import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from enum import Enum
import json

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Task:
    """任务对象"""
    
    def __init__(
        self,
        task_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        user_id: str
    ):
        self.task_id = task_id
        self.tool_name = tool_name
        self.arguments = arguments
        self.user_id = user_id
        self.status = TaskStatus.PENDING
        self.progress = 0.0
        self.result: Optional[Any] = None
        self.error: Optional[str] = None
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.metadata: Dict[str, Any] = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "user_id": self.user_id,
            "status": self.status.value,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata
        }


class TaskManager:
    """
    任务管理器
    
    管理异步任务的创建、执行、状态跟踪等
    """
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.callbacks: Dict[str, Callable] = {}
        logger.info("任务管理器初始化完成")
    
    async def create_task(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        user_id: str,
        callback_url: Optional[str] = None
    ) -> str:
        """
        创建新任务
        
        Args:
            tool_name: 工具名称
            arguments: 工具参数
            user_id: 用户ID
            callback_url: 回调URL（可选）
        
        Returns:
            str: 任务ID
        """
        task_id = str(uuid.uuid4())
        
        task = Task(
            task_id=task_id,
            tool_name=tool_name,
            arguments=arguments,
            user_id=user_id
        )
        
        if callback_url:
            task.metadata['callback_url'] = callback_url
        
        self.tasks[task_id] = task
        
        logger.info(f"创建任务: {task_id} - {tool_name}")
        return task_id
    
    async def submit_task(
        self,
        task_id: str,
        executor: Callable
    ):
        """
        提交任务执行
        
        Args:
            task_id: 任务ID
            executor: 执行函数
        """
        if task_id not in self.tasks:
            raise KeyError(f"任务不存在: {task_id}")
        
        task = self.tasks[task_id]
        
        # 在后台执行任务
        asyncio.create_task(self._execute_task(task, executor))
        
        logger.info(f"提交任务执行: {task_id}")
    
    async def _execute_task(self, task: Task, executor: Callable):
        """
        执行任务（内部方法）
        
        Args:
            task: 任务对象
            executor: 执行函数
        """
        try:
            # 更新状态为运行中
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            
            logger.info(f"开始执行任务: {task.task_id}")
            
            # 创建进度报告器
            progress_reporter = ProgressReporter(task.task_id, self)
            
            # 执行工具
            result = await executor(
                task.arguments,
                progress_reporter=progress_reporter
            )
            
            # 更新为完成状态
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.progress = 100.0
            task.completed_at = datetime.now()
            
            logger.info(f"任务执行完成: {task.task_id}")
            
            # 调用回调
            await self._call_callback(task)
        
        except Exception as e:
            # 更新为失败状态
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            
            logger.error(f"任务执行失败: {task.task_id} - {e}")
            
            # 调用回调
            await self._call_callback(task)
    
    async def _call_callback(self, task: Task):
        """调用任务完成回调"""
        callback_url = task.metadata.get('callback_url')
        if not callback_url:
            return
        
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                await client.post(
                    callback_url,
                    json=task.to_dict(),
                    timeout=10.0
                )
            logger.info(f"回调成功: {callback_url}")
        except Exception as e:
            logger.error(f"回调失败: {callback_url} - {e}")
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        获取任务状态
        
        Args:
            task_id: 任务ID
        
        Returns:
            Dict: 任务状态信息
        """
        if task_id not in self.tasks:
            raise KeyError(f"任务不存在: {task_id}")
        
        return self.tasks[task_id].to_dict()
    
    async def update_progress(self, task_id: str, progress: float, message: str = ""):
        """
        更新任务进度
        
        Args:
            task_id: 任务ID
            progress: 进度（0-100）
            message: 进度消息
        """
        if task_id not in self.tasks:
            return
        
        task = self.tasks[task_id]
        task.progress = max(0.0, min(100.0, progress))
        
        if message:
            if 'progress_messages' not in task.metadata:
                task.metadata['progress_messages'] = []
            task.metadata['progress_messages'].append({
                'timestamp': datetime.now().isoformat(),
                'progress': progress,
                'message': message
            })
        
        logger.debug(f"任务进度更新: {task_id} - {progress}% - {message}")
    
    async def cancel_task(self, task_id: str):
        """
        取消任务
        
        Args:
            task_id: 任务ID
        """
        if task_id not in self.tasks:
            raise KeyError(f"任务不存在: {task_id}")
        
        task = self.tasks[task_id]
        
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            logger.warning(f"任务已结束，无法取消: {task_id}")
            return
        
        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.now()
        
        logger.info(f"任务已取消: {task_id}")
    
    async def list_user_tasks(
        self,
        user_id: str,
        status: Optional[TaskStatus] = None
    ) -> list:
        """
        列出用户的任务
        
        Args:
            user_id: 用户ID
            status: 过滤状态（可选）
        
        Returns:
            list: 任务列表
        """
        tasks = [
            task.to_dict()
            for task in self.tasks.values()
            if task.user_id == user_id and (status is None or task.status == status)
        ]
        
        # 按创建时间倒序排列
        tasks.sort(key=lambda t: t['created_at'], reverse=True)
        
        return tasks
    
    async def cleanup_old_tasks(self, days: int = 7):
        """
        清理旧任务
        
        Args:
            days: 保留天数
        """
        from datetime import timedelta
        cutoff_time = datetime.now() - timedelta(days=days)
        
        old_task_ids = [
            task_id
            for task_id, task in self.tasks.items()
            if task.completed_at and task.completed_at < cutoff_time
        ]
        
        for task_id in old_task_ids:
            del self.tasks[task_id]
        
        logger.info(f"清理了 {len(old_task_ids)} 个旧任务")


class ProgressReporter:
    """
    进度报告器
    
    用于在任务执行过程中报告进度
    """
    
    def __init__(self, task_id: str, task_manager: TaskManager):
        self.task_id = task_id
        self.task_manager = task_manager
    
    async def report(self, progress: float, message: str = ""):
        """
        报告进度
        
        Args:
            progress: 进度值（0-100）
            message: 进度消息
        """
        await self.task_manager.update_progress(
            self.task_id,
            progress,
            message
        )
    
    async def report_stage(self, stage: str, total_stages: int, current_stage: int):
        """
        报告阶段进度
        
        Args:
            stage: 阶段名称
            total_stages: 总阶段数
            current_stage: 当前阶段编号（从1开始）
        """
        progress = (current_stage / total_stages) * 100
        message = f"阶段 {current_stage}/{total_stages}: {stage}"
        await self.report(progress, message)


# 全局任务管理器实例
task_manager = TaskManager()
