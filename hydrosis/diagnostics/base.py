"""诊断框架基础类

提供统一的诊断接口、问题识别和报告生成。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


class IssueSeverity(Enum):
    """问题严重程度"""
    INFO = "info"          # 信息
    WARNING = "warning"    # 警告
    ERROR = "error"        # 错误
    CRITICAL = "critical"  # 严重错误


@dataclass
class DiagnosticIssue:
    """诊断问题
    
    Attributes
    ----------
    category : str
        问题类别 (e.g., "water_balance", "unit_conversion")
    severity : IssueSeverity
        严重程度
    message : str
        问题描述
    details : dict, optional
        详细信息
    suggestion : str, optional
        修复建议
    """
    category: str
    severity: IssueSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    suggestion: str = ""
    
    def __str__(self) -> str:
        """字符串表示"""
        icon = {
            IssueSeverity.INFO: "ℹ",
            IssueSeverity.WARNING: "⚠",
            IssueSeverity.ERROR: "❌",
            IssueSeverity.CRITICAL: "🔴"
        }[self.severity]
        
        result = f"{icon} [{self.category}] {self.message}"
        if self.suggestion:
            result += f"\n   建议: {self.suggestion}"
        return result


@dataclass
class DiagnosticResult:
    """诊断结果
    
    Attributes
    ----------
    diagnostic_name : str
        诊断名称
    issues : list of DiagnosticIssue
        发现的问题列表
    metrics : dict
        诊断指标
    figures : dict
        生成的图表路径
    recommendations : list of str
        修复建议列表
    metadata : dict
        额外元数据
    """
    diagnostic_name: str
    issues: List[DiagnosticIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    figures: Dict[str, Path] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_errors(self) -> bool:
        """是否有错误"""
        return any(
            issue.severity in [IssueSeverity.ERROR, IssueSeverity.CRITICAL]
            for issue in self.issues
        )
    
    @property
    def has_warnings(self) -> bool:
        """是否有警告"""
        return any(issue.severity == IssueSeverity.WARNING for issue in self.issues)
    
    @property
    def error_count(self) -> int:
        """错误数量"""
        return sum(
            1 for issue in self.issues
            if issue.severity in [IssueSeverity.ERROR, IssueSeverity.CRITICAL]
        )
    
    @property
    def warning_count(self) -> int:
        """警告数量"""
        return sum(1 for issue in self.issues if issue.severity == IssueSeverity.WARNING)

    def has_issues(self) -> bool:
        """是否有任何问题（任何严重程度）"""
        return len(self.issues) > 0

    def get_issues_by_severity(self, severity: IssueSeverity) -> List[DiagnosticIssue]:
        """获取指定严重程度的问题列表

        Parameters
        ----------
        severity : IssueSeverity
            问题严重程度

        Returns
        -------
        List[DiagnosticIssue]
            匹配的问题列表
        """
        return [issue for issue in self.issues if issue.severity == severity]

    def summary(self) -> str:
        """生成摘要"""
        lines = []
        lines.append("=" * 80)
        lines.append(f"诊断结果: {self.diagnostic_name}")
        lines.append("=" * 80)
        
        # 问题统计
        lines.append(f"\n问题统计:")
        lines.append(f"  严重错误: {sum(1 for i in self.issues if i.severity == IssueSeverity.CRITICAL)}")
        lines.append(f"  错误: {sum(1 for i in self.issues if i.severity == IssueSeverity.ERROR)}")
        lines.append(f"  警告: {self.warning_count}")
        lines.append(f"  信息: {sum(1 for i in self.issues if i.severity == IssueSeverity.INFO)}")
        
        # 关键指标
        if self.metrics:
            lines.append(f"\n关键指标:")
            for key, value in self.metrics.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.4f}")
                else:
                    lines.append(f"  {key}: {value}")
        
        # 发现的问题
        if self.issues:
            lines.append(f"\n发现的问题:")
            for issue in self.issues:
                lines.append(f"  {issue}")
        
        # 建议
        if self.recommendations:
            lines.append(f"\n修复建议:")
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"  {i}. {rec}")
        
        # 图表
        if self.figures:
            lines.append(f"\n生成的图表:")
            for name, path in self.figures.items():
                lines.append(f"  {name}: {path}")
        
        lines.append("=" * 80)
        return "\n".join(lines)


class BaseDiagnostic(ABC):
    """诊断器抽象基类
    
    Parameters
    ----------
    output_dir : Path, optional
        输出目录
    verbose : bool, default=True
        是否输出详细信息
    """
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        verbose: bool = True
    ):
        """初始化诊断器"""
        self.output_dir = Path(output_dir) if output_dir else Path("diagnostics_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        
        # 内部状态
        self._result: Optional[DiagnosticResult] = None
    
    @property
    @abstractmethod
    def diagnostic_name(self) -> str:
        """诊断名称"""
        pass
    
    @abstractmethod
    def load_data(self, **kwargs) -> None:
        """加载数据
        
        子类实现此方法来加载诊断所需的数据
        """
        pass
    
    @abstractmethod
    def analyze(self) -> DiagnosticResult:
        """执行诊断分析
        
        Returns
        -------
        DiagnosticResult
            诊断结果
        """
        pass
    
    def run(self, **load_kwargs) -> DiagnosticResult:
        """运行完整的诊断流程
        
        Parameters
        ----------
        **load_kwargs
            传递给load_data的参数
        
        Returns
        -------
        DiagnosticResult
            诊断结果
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"{self.diagnostic_name}")
            print(f"{'='*80}")
        
        # 加载数据
        if self.verbose:
            print("\n步骤 1: 加载数据...")
        self.load_data(**load_kwargs)
        
        # 执行分析
        if self.verbose:
            print("步骤 2: 执行诊断分析...")
        self._result = self.analyze()
        
        # 生成可视化
        if self.verbose:
            print("步骤 3: 生成可视化...")
        self.visualize()
        
        # 保存报告
        if self.verbose:
            print("步骤 4: 保存诊断报告...")
        self.save_report()
        
        if self.verbose:
            print(f"\n✓ 诊断完成!")
            print(f"  错误: {self._result.error_count}")
            print(f"  警告: {self._result.warning_count}")
        
        return self._result
    
    def visualize(self) -> None:
        """生成可视化
        
        子类可以覆盖此方法来生成诊断图表
        """
        pass
    
    def save_report(self) -> None:
        """保存诊断报告"""
        if self._result is None:
            raise RuntimeError("请先运行analyze()生成诊断结果")
        
        # 保存文本报告
        report_path = self.output_dir / f"{self.diagnostic_name}_report.txt"
        report_path.write_text(self._result.summary(), encoding='utf-8')
        
        if self.verbose:
            print(f"  报告已保存: {report_path}")
        
        # 保存JSON格式
        import json
        json_path = self.output_dir / f"{self.diagnostic_name}_report.json"
        
        json_data = {
            "diagnostic_name": self._result.diagnostic_name,
            "metrics": self._result.metrics,
            "issues": [
                {
                    "category": issue.category,
                    "severity": issue.severity.value,
                    "message": issue.message,
                    "details": issue.details,
                    "suggestion": issue.suggestion
                }
                for issue in self._result.issues
            ],
            "recommendations": self._result.recommendations,
            "metadata": self._result.metadata
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            print(f"  JSON报告已保存: {json_path}")
    
    def add_issue(
        self,
        category: str,
        severity: IssueSeverity,
        message: str,
        details: Optional[Dict] = None,
        suggestion: str = ""
    ) -> None:
        """添加诊断问题
        
        Parameters
        ----------
        category : str
            问题类别
        severity : IssueSeverity
            严重程度
        message : str
            问题描述
        details : dict, optional
            详细信息
        suggestion : str, optional
            修复建议
        """
        if self._result is None:
            raise RuntimeError("请先初始化DiagnosticResult")
        
        issue = DiagnosticIssue(
            category=category,
            severity=severity,
            message=message,
            details=details or {},
            suggestion=suggestion
        )
        self._result.issues.append(issue)
    
    def add_metric(self, key: str, value: Any) -> None:
        """添加指标"""
        if self._result is None:
            raise RuntimeError("请先初始化DiagnosticResult")
        self._result.metrics[key] = value
    
    def add_recommendation(self, recommendation: str) -> None:
        """添加建议"""
        if self._result is None:
            raise RuntimeError("请先初始化DiagnosticResult")
        self._result.recommendations.append(recommendation)
