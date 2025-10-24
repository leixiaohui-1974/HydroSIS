"""统一配置管理模块

提供集中的配置加载、缓存和验证功能。

同时导出原有的配置数据类以保持向后兼容性。
"""
from .manager import ConfigManager
from .models import (
    ComparisonPlanConfig,
    EvaluationConfig,
    IOConfig,
    ScenarioConfig,
    ModelConfig,
    OutputArtifactsConfig,
    SubbasinMethodAssignment,
    ModelStructureConfig,
    ParameterPartitionConfig,
    HydroProjectConfig,
    load_validation_criteria,
    create_hydrologic_criteria,
    load_workflow_config,
    get_step_paths,
)

# Import config classes from their source modules for convenience
from ..runoff.base import RunoffModelConfig
from ..routing.base import RoutingModelConfig
from ..delineation.dem_delineator import DelineationConfig
from ..parameters.zone import ParameterZoneConfig

__all__ = [
    # 新的配置管理器
    "ConfigManager",
    # 原有的配置数据类
    "ComparisonPlanConfig",
    "EvaluationConfig",
    "IOConfig",
    "ScenarioConfig",
    "ModelConfig",
    "OutputArtifactsConfig",
    "SubbasinMethodAssignment",
    "ModelStructureConfig",
    "ParameterPartitionConfig",
    "HydroProjectConfig",
    # 模型配置类
    "RunoffModelConfig",
    "RoutingModelConfig",
    "DelineationConfig",
    "ParameterZoneConfig",
    # 辅助函数
    "load_validation_criteria",
    "create_hydrologic_criteria",
    "load_workflow_config",
    "get_step_paths",
]
