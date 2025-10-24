"""统一校准框架

提供抽象的模型校准接口，消除重复代码。

主要组件：
- BaseCalibrator: 基础校准器抽象类
- GenericHydrologicCalibrator: 通用水文模型校准器（推荐使用）
  支持所有产流、汇流、以及产汇流组合模型
- GenericRunoffCalibrator: 通用产流模型校准器（简化版本）
- HBVCalibrator: HBV模型专用校准器（向后兼容）
- XinAnJiangCalibrator: 新安江模型专用校准器（向后兼容）

敏感性分析：
- HydrologicSensitivityAnalyzer: 水文模型敏感性分析器
- analyze_model_sensitivity: 便捷函数

模型对比：
- ModelComparator: 多模型性能对比器
- compare_models: 便捷函数
"""
from .base import BaseCalibrator, CalibrationData, CalibrationConfig, CalibrationResult
from .hbv_calibrator import HBVCalibrator
from .xinanjiang_calibrator import XinAnJiangCalibrator
from .generic_runoff_calibrator import GenericRunoffCalibrator, calibrate_runoff_model
from .generic_calibrator import GenericHydrologicCalibrator, ModelMode
from .sensitivity_analyzer import (
    HydrologicSensitivityAnalyzer,
    ModelSensitivityResult,
    analyze_model_sensitivity
)
from .model_comparison import (
    ModelComparator,
    ModelComparisonResult,
    ModelPerformance,
    compare_models
)

__all__ = [
    # 核心类
    "BaseCalibrator",
    "CalibrationData",
    "CalibrationConfig",
    "CalibrationResult",

    # 通用校准器（推荐使用）
    "GenericHydrologicCalibrator",
    "GenericRunoffCalibrator",
    "ModelMode",

    # 便捷函数
    "calibrate_runoff_model",

    # 模型特定校准器（向后兼容）
    "HBVCalibrator",
    "XinAnJiangCalibrator",

    # 敏感性分析
    "HydrologicSensitivityAnalyzer",
    "ModelSensitivityResult",
    "analyze_model_sensitivity",

    # 模型对比
    "ModelComparator",
    "ModelComparisonResult",
    "ModelPerformance",
    "compare_models",
]
