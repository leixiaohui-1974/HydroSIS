"""统一诊断框架

提供标准化的诊断接口和报告生成功能。
"""
from .base import BaseDiagnostic, DiagnosticResult, DiagnosticIssue, IssueSeverity
from .water_balance import WaterBalanceDiagnostic
from .precipitation import PrecipitationDiagnostic
from .hbv_configuration import HBVConfigurationDiagnostic

__all__ = [
    "BaseDiagnostic",
    "DiagnosticResult",
    "DiagnosticIssue",
    "IssueSeverity",
    "WaterBalanceDiagnostic",
    "PrecipitationDiagnostic",
    "HBVConfigurationDiagnostic",
]
