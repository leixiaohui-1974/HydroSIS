"""HydroSIS模块化API系统

提供独立可运行的功能模块，每个模块都可以通过REST API、MCP、CLI等多种接口调用。
"""

from .base import Module, ModuleInput, ModuleOutput, ModuleConfig, ModuleRegistry
from .terrain import TerrainModule
from .pour_points import PourPointsModule
from .watershed import WatershedDelineationModule
from .channel import ChannelNetworkModule
from .rain_gauge import RainGaugeLayoutModule
from .precipitation import PrecipitationGenerationModule
from .areal_precip import ArealPrecipitationModule
from .runoff import RunoffGenerationModule
from .routing import RoutingModule
from .calibration import CalibrationModule
from .evaluation import EvaluationModule

__all__ = [
    "Module",
    "ModuleInput",
    "ModuleOutput",
    "ModuleConfig",
    "ModuleRegistry",
    "TerrainModule",
    "PourPointsModule",
    "WatershedDelineationModule",
    "ChannelNetworkModule",
    "RainGaugeLayoutModule",
    "PrecipitationGenerationModule",
    "ArealPrecipitationModule",
    "RunoffGenerationModule",
    "RoutingModule",
    "CalibrationModule",
    "EvaluationModule",
]
