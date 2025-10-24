"""统一校准框架

提供抽象的模型校准接口，消除重复代码。
"""
from .base import BaseCalibrator, CalibrationData, CalibrationConfig, CalibrationResult
from .hbv_calibrator import HBVCalibrator

__all__ = [
    "BaseCalibrator",
    "CalibrationData",
    "CalibrationConfig",
    "CalibrationResult",
    "HBVCalibrator",
]
