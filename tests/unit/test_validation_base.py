"""Unit tests for validation base classes.

Tests ValidationResult, ValidationCriteria, and BaseValidator.
"""
import pytest
from pathlib import Path
import tempfile

from hydrosis.validation.base import (
    ValidationCriteria,
    ValidationResult,
    BaseValidator,
)


class TestValidationCriteria:
    """Test ValidationCriteria class"""

    def test_default_initialization(self):
        """Test default initialization"""
        criteria = ValidationCriteria()
        assert criteria.name == "default"
        assert criteria.description == ""
        assert criteria.strict_mode is False

    def test_custom_initialization(self):
        """Test initialization with custom values"""
        criteria = ValidationCriteria(
            name="test_criteria",
            description="Test description",
            strict_mode=True
        )
        assert criteria.name == "test_criteria"
        assert criteria.description == "Test description"
        assert criteria.strict_mode is True

    def test_from_dict(self):
        """Test creation from dictionary"""
        data = {
            "name": "dict_criteria",
            "description": "From dict",
            "strict_mode": True,
        }
        criteria = ValidationCriteria.from_dict(data)
        assert criteria.name == "dict_criteria"
        assert criteria.description == "From dict"
        assert criteria.strict_mode is True

    def test_from_dict_partial(self):
        """Test creation from dictionary with missing fields"""
        data = {"name": "partial"}
        criteria = ValidationCriteria.from_dict(data)
        assert criteria.name == "partial"
        assert criteria.description == ""
        assert criteria.strict_mode is False


class TestValidationResult:
    """Test ValidationResult class"""

    def test_default_initialization(self):
        """Test default initialization"""
        result = ValidationResult()
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
        assert len(result.metrics) == 0
        assert result.step_name == ""

    def test_with_step_name(self):
        """Test initialization with step name"""
        result = ValidationResult(step_name="Test Step")
        assert result.step_name == "Test Step"
        assert result.is_valid is True

    def test_add_error(self):
        """Test adding error message"""
        result = ValidationResult()
        assert result.is_valid is True

        result.add_error("Test error 1")
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0] == "Test error 1"

        result.add_error("Test error 2")
        assert len(result.errors) == 2

    def test_add_warning(self):
        """Test adding warning message"""
        result = ValidationResult()
        assert result.is_valid is True

        result.add_warning("Test warning")
        assert result.is_valid is True  # Warnings don't affect validity
        assert len(result.warnings) == 1
        assert result.warnings[0] == "Test warning"

    def test_add_metric(self):
        """Test adding metrics"""
        result = ValidationResult()

        result.add_metric("count", 10)
        result.add_metric("ratio", 0.75)
        result.add_metric("name", "test")

        assert len(result.metrics) == 3
        assert result.metrics["count"] == 10
        assert result.metrics["ratio"] == 0.75
        assert result.metrics["name"] == "test"

    def test_summary_valid(self):
        """Test summary generation for valid result"""
        result = ValidationResult(step_name="Test Validation")
        result.add_metric("test_count", 5)
        result.add_metric("accuracy", 0.95)

        summary = result.summary()
        assert "Test Validation" in summary
        assert "✅ 通过" in summary
        assert "test_count" in summary
        assert "accuracy" in summary

    def test_summary_with_errors(self):
        """Test summary generation with errors"""
        result = ValidationResult(step_name="Error Test")
        result.add_error("Error 1")
        result.add_error("Error 2")

        summary = result.summary()
        assert "❌ 失败" in summary
        assert "错误 (2)" in summary
        assert "Error 1" in summary
        assert "Error 2" in summary

    def test_summary_with_warnings(self):
        """Test summary generation with warnings"""
        result = ValidationResult(step_name="Warning Test")
        result.add_warning("Warning 1")
        result.add_warning("Warning 2")

        summary = result.summary()
        assert "✅ 通过" in summary  # Still valid
        assert "警告 (2)" in summary
        assert "Warning 1" in summary
        assert "Warning 2" in summary

    def test_summary_with_all_types(self):
        """Test summary with errors, warnings, and metrics"""
        result = ValidationResult(step_name="Complete Test")
        result.add_error("Error message")
        result.add_warning("Warning message")
        result.add_metric("metric1", 100)
        result.add_metric("metric2", 0.5)

        summary = result.summary()
        assert "Complete Test" in summary
        assert "❌ 失败" in summary
        assert "Error message" in summary
        assert "Warning message" in summary
        assert "metric1" in summary
        assert "metric2" in summary

    def test_save_report(self):
        """Test saving report to file"""
        result = ValidationResult(step_name="Save Test")
        result.add_metric("test", 123)
        result.add_warning("Test warning")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "report.txt"
            result.save_report(output_path)

            assert output_path.exists()
            content = output_path.read_text(encoding='utf-8')
            assert "Save Test" in content
            assert "test" in content
            assert "Test warning" in content


class ConcreteValidator(BaseValidator):
    """Concrete validator for testing BaseValidator"""

    def validate(self, value: int = 0) -> ValidationResult:
        """Simple validation that checks if value > 0"""
        result = self._create_result("Concrete Validator")
        result.add_metric("input_value", value)

        if value <= 0:
            result.add_error(f"Value must be positive, got {value}")

        return result


class TestBaseValidator:
    """Test BaseValidator abstract class"""

    def test_default_initialization(self):
        """Test validator with default criteria"""
        validator = ConcreteValidator()
        assert validator.criteria is not None
        assert validator.criteria.name == "default"

    def test_custom_criteria(self):
        """Test validator with custom criteria"""
        criteria = ValidationCriteria(name="custom", strict_mode=True)
        validator = ConcreteValidator(criteria=criteria)
        assert validator.criteria.name == "custom"
        assert validator.criteria.strict_mode is True

    def test_validate_valid_input(self):
        """Test validation with valid input"""
        validator = ConcreteValidator()
        result = validator.validate(value=10)

        assert result.is_valid is True
        assert result.metrics["input_value"] == 10
        assert len(result.errors) == 0

    def test_validate_invalid_input(self):
        """Test validation with invalid input"""
        validator = ConcreteValidator()
        result = validator.validate(value=0)

        assert result.is_valid is False
        assert result.metrics["input_value"] == 0
        assert len(result.errors) == 1
        assert "must be positive" in result.errors[0]

    def test_create_result(self):
        """Test _create_result helper method"""
        validator = ConcreteValidator()
        result = validator._create_result("Test Step")

        assert isinstance(result, ValidationResult)
        assert result.step_name == "Test Step"
        assert result.is_valid is True
        assert len(result.errors) == 0
