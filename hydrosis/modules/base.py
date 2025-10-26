"""模块基础类和接口定义"""
from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class ModuleInput:
    """模块输入数据结构"""
    pass


@dataclass
class ModuleOutput:
    """模块输出数据结构"""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def save(self, path: Path) -> None:
        """保存输出到文件"""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding='utf-8')


@dataclass
class ModuleConfig:
    """模块配置"""
    
    module_id: str
    version: str = "1.0.0"
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModuleConfig:
        """从字典创建配置"""
        return cls(**data)
    
    @classmethod
    def from_yaml(cls, path: Path) -> ModuleConfig:
        """从YAML文件加载配置"""
        try:
            import yaml
            data = yaml.safe_load(path.read_text(encoding='utf-8'))
            return cls.from_dict(data.get('module', {}))
        except ImportError:
            raise RuntimeError("需要安装PyYAML来加载YAML配置")


@dataclass
class ModuleMetadata:
    """模块元数据"""
    
    module_id: str
    name: str
    description: str
    version: str
    author: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_openapi_schema(self) -> Dict[str, Any]:
        """转换为OpenAPI schema"""
        return {
            "summary": self.name,
            "description": self.description,
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": self.input_schema
                    }
                }
            },
            "responses": {
                "200": {
                    "description": "成功",
                    "content": {
                        "application/json": {
                            "schema": self.output_schema
                        }
                    }
                }
            }
        }
    
    def to_mcp_tool(self) -> Dict[str, Any]:
        """转换为MCP工具定义"""
        return {
            "name": f"hydrosis_{self.module_id}",
            "description": self.description,
            "inputSchema": self.input_schema
        }


@dataclass
class ModuleExecutionContext:
    """模块执行上下文"""
    
    task_id: str
    module_id: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: str = "running"  # running, completed, failed
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def mark_completed(self) -> None:
        """标记为完成"""
        self.status = "completed"
        self.end_time = datetime.now()
    
    def mark_failed(self, error: str) -> None:
        """标记为失败"""
        self.status = "failed"
        self.error = error
        self.end_time = datetime.now()
    
    def duration_seconds(self) -> Optional[float]:
        """计算执行时长（秒）"""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()


class Module(abc.ABC, Generic[T]):
    """模块基类
    
    所有功能模块都继承此基类，实现标准化的接口。
    """
    
    def __init__(self, config: Optional[ModuleConfig] = None):
        """初始化模块
        
        Args:
            config: 模块配置，如果为None则使用默认配置
        """
        self.config = config or self._default_config()
        self._setup_logging()
    
    @classmethod
    @abc.abstractmethod
    def module_id(cls) -> str:
        """返回模块唯一标识符"""
        pass
    
    @classmethod
    @abc.abstractmethod
    def metadata(cls) -> ModuleMetadata:
        """返回模块元数据"""
        pass
    
    @classmethod
    def _default_config(cls) -> ModuleConfig:
        """返回默认配置"""
        return ModuleConfig(
            module_id=cls.module_id(),
            name=cls.metadata().name,
            description=cls.metadata().description
        )
    
    def _setup_logging(self) -> None:
        """设置日志"""
        self.logger = logging.getLogger(f"hydrosis.modules.{self.module_id()}")
    
    @abc.abstractmethod
    def execute(self, inputs: ModuleInput, context: Optional[ModuleExecutionContext] = None) -> ModuleOutput:
        """执行模块功能
        
        Args:
            inputs: 模块输入
            context: 执行上下文（可选）
        
        Returns:
            模块输出
        
        Raises:
            ValueError: 输入参数无效
            RuntimeError: 执行过程中出错
        """
        pass
    
    def validate_inputs(self, inputs: ModuleInput) -> List[str]:
        """验证输入参数
        
        Args:
            inputs: 模块输入
        
        Returns:
            错误消息列表，如果为空则验证通过
        """
        errors = []
        # 子类可以重写此方法添加自定义验证
        return errors
    
    def run(self, inputs: ModuleInput, task_id: Optional[str] = None) -> ModuleOutput:
        """运行模块（包含完整的生命周期管理）
        
        Args:
            inputs: 模块输入
            task_id: 任务ID（可选，用于追踪）
        
        Returns:
            模块输出
        
        Raises:
            ValueError: 输入验证失败
            RuntimeError: 执行失败
        """
        # 生成任务ID
        if task_id is None:
            import uuid
            task_id = str(uuid.uuid4())
        
        # 创建执行上下文
        context = ModuleExecutionContext(
            task_id=task_id,
            module_id=self.module_id()
        )
        
        try:
            # 验证输入
            self.logger.info(f"[{task_id}] 开始验证输入...")
            validation_errors = self.validate_inputs(inputs)
            if validation_errors:
                error_msg = "; ".join(validation_errors)
                raise ValueError(f"输入验证失败: {error_msg}")
            
            # 执行模块
            self.logger.info(f"[{task_id}] 开始执行模块 {self.module_id()}...")
            output = self.execute(inputs, context)
            
            # 标记完成
            context.mark_completed()
            self.logger.info(f"[{task_id}] 模块执行成功，耗时 {context.duration_seconds():.2f} 秒")
            
            return output
            
        except Exception as e:
            # 标记失败
            context.mark_failed(str(e))
            self.logger.error(f"[{task_id}] 模块执行失败: {e}")
            raise
    
    def get_info(self) -> Dict[str, Any]:
        """获取模块信息"""
        meta = self.metadata()
        return {
            "module_id": self.module_id(),
            "name": meta.name,
            "description": meta.description,
            "version": meta.version,
            "author": meta.author,
            "config": asdict(self.config),
            "input_schema": meta.input_schema,
            "output_schema": meta.output_schema,
        }


class ModuleRegistry:
    """模块注册表
    
    管理所有可用模块的注册和查找。
    """
    
    def __init__(self):
        self._modules: Dict[str, Type[Module]] = {}
        self._instances: Dict[str, Module] = {}
    
    def register(self, module_class: Type[Module]) -> None:
        """注册模块类
        
        Args:
            module_class: 模块类
        """
        module_id = module_class.module_id()
        if module_id in self._modules:
            logger.warning(f"模块 {module_id} 已注册，将被覆盖")
        self._modules[module_id] = module_class
        logger.info(f"注册模块: {module_id}")
    
    def get_module_class(self, module_id: str) -> Optional[Type[Module]]:
        """获取模块类
        
        Args:
            module_id: 模块ID
        
        Returns:
            模块类，如果未找到返回None
        """
        return self._modules.get(module_id)
    
    def create_module(self, module_id: str, config: Optional[ModuleConfig] = None) -> Module:
        """创建模块实例
        
        Args:
            module_id: 模块ID
            config: 模块配置（可选）
        
        Returns:
            模块实例
        
        Raises:
            KeyError: 模块未注册
        """
        module_class = self.get_module_class(module_id)
        if module_class is None:
            raise KeyError(f"模块 {module_id} 未注册")
        return module_class(config)
    
    def get_or_create_module(self, module_id: str, config: Optional[ModuleConfig] = None) -> Module:
        """获取或创建模块实例（单例模式）
        
        Args:
            module_id: 模块ID
            config: 模块配置（可选）
        
        Returns:
            模块实例
        """
        if module_id not in self._instances:
            self._instances[module_id] = self.create_module(module_id, config)
        return self._instances[module_id]
    
    def list_modules(self) -> List[str]:
        """列出所有注册的模块ID"""
        return list(self._modules.keys())
    
    def get_all_metadata(self) -> Dict[str, ModuleMetadata]:
        """获取所有模块的元数据"""
        return {
            module_id: module_class.metadata()
            for module_id, module_class in self._modules.items()
        }


# 全局模块注册表
_global_registry = ModuleRegistry()


def register_module(module_class: Type[Module]) -> Type[Module]:
    """模块注册装饰器
    
    Usage:
        @register_module
        class MyModule(Module):
            ...
    """
    _global_registry.register(module_class)
    return module_class


def get_registry() -> ModuleRegistry:
    """获取全局模块注册表"""
    return _global_registry
