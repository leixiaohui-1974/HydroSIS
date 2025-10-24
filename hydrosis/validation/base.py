"""Base classes and data structures for validation framework.

This module provides the foundation for all validation operations in HydroSIS.
It defines common data structures and abstract base classes that ensure
consistent validation behavior across different workflow steps.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class ValidationCriteria:
    """Base class for validation criteria.

    All validation criteria should be loaded from configuration files
    rather than hardcoded. This class provides a common structure for
    organizing validation rules.

    Attributes
    ----------
    name : str
        Name of this validation criteria set
    description : str
        Description of what this criteria validates
    strict_mode : bool
        If True, warnings are treated as errors
    """
    name: str = "default"
    description: str = ""
    strict_mode: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ValidationCriteria:
        """Create ValidationCriteria from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing criteria configuration

        Returns
        -------
        ValidationCriteria
            Initialized criteria object
        """
        return cls(
            name=data.get("name", "default"),
            description=data.get("description", ""),
            strict_mode=data.get("strict_mode", False),
        )


@dataclass
class ValidationResult:
    """Result of a validation operation.

    This class stores the outcome of validation checks, including
    success/failure status, error messages, warnings, and metrics.

    Attributes
    ----------
    is_valid : bool
        Whether validation passed (no errors)
    errors : List[str]
        List of error messages (validation failures)
    warnings : List[str]
        List of warning messages (potential issues)
    metrics : Dict[str, Any]
        Computed metrics and statistics
    step_name : str
        Name of the workflow step being validated
    """
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    step_name: str = ""

    def add_error(self, message: str) -> None:
        """Add an error message and mark validation as failed.

        Parameters
        ----------
        message : str
            Error message to add
        """
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning message.

        Parameters
        ----------
        message : str
            Warning message to add
        """
        self.warnings.append(message)

    def add_metric(self, key: str, value: Any) -> None:
        """Add a computed metric.

        Parameters
        ----------
        key : str
            Metric name
        value : Any
            Metric value
        """
        self.metrics[key] = value

    def summary(self) -> str:
        """Generate a human-readable summary of validation results.

        Returns
        -------
        str
            Formatted summary string
        """
        lines = []
        lines.append("=" * 80)
        if self.step_name:
            lines.append(f"验证结果: {self.step_name}")
        else:
            lines.append("验证结果")
        lines.append("=" * 80)

        # Status
        status = "✅ 通过" if self.is_valid else "❌ 失败"
        lines.append(f"状态: {status}")
        lines.append("")

        # Metrics
        if self.metrics:
            lines.append("关键指标:")
            for key, value in self.metrics.items():
                if isinstance(value, float):
                    lines.append(f"  - {key}: {value:.4f}")
                else:
                    lines.append(f"  - {key}: {value}")
            lines.append("")

        # Errors
        if self.errors:
            lines.append(f"错误 ({len(self.errors)}):")
            for error in self.errors:
                lines.append(f"  ❌ {error}")
            lines.append("")

        # Warnings
        if self.warnings:
            lines.append(f"警告 ({len(self.warnings)}):")
            for warning in self.warnings:
                lines.append(f"  ⚠️  {warning}")
            lines.append("")

        lines.append("=" * 80)
        return "\n".join(lines)

    def save_report(self, output_path: Union[str, Path]) -> None:
        """Save validation report to file.

        Parameters
        ----------
        output_path : str or Path
            Path to save the report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.summary(), encoding='utf-8')


class BaseValidator(ABC):
    """Abstract base class for all validators.

    All validators should inherit from this class and implement
    the validate() method. This ensures consistent API across
    different validation types.

    Parameters
    ----------
    criteria : ValidationCriteria
        Validation criteria to use
    """

    def __init__(self, criteria: Optional[ValidationCriteria] = None):
        """Initialize validator with criteria.

        Parameters
        ----------
        criteria : ValidationCriteria, optional
            Validation criteria. If None, default criteria are used.
        """
        self.criteria = criteria or ValidationCriteria()

    @abstractmethod
    def validate(self, **kwargs) -> ValidationResult:
        """Perform validation.

        This method must be implemented by all concrete validator classes.

        Parameters
        ----------
        **kwargs
            Validation-specific arguments

        Returns
        -------
        ValidationResult
            Validation results
        """
        pass

    def _create_result(self, step_name: str = "") -> ValidationResult:
        """Create a new ValidationResult for this validator.

        Parameters
        ----------
        step_name : str, optional
            Name of the workflow step

        Returns
        -------
        ValidationResult
            New validation result object
        """
        return ValidationResult(step_name=step_name)
