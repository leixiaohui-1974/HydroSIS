"""敏感性分析器单元测试

测试HydrologicSensitivityAnalyzer的核心功能。
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from hydrosis.calibration import (
    CalibrationData,
    GenericHydrologicCalibrator,
    HydrologicSensitivityAnalyzer,
    analyze_model_sensitivity,
)


@pytest.fixture
def sample_calibration_data():
    """创建示例校准数据"""
    n = 50  # 使用较小的数据集加快测试
    np.random.seed(42)
    precipitation = np.random.gamma(2, 2, size=n)
    observed_runoff = precipitation * 0.3 + np.random.normal(0, 0.5, n)
    observed_runoff = np.maximum(observed_runoff, 0)

    times = pd.date_range('2024-01-01', periods=n, freq='h')

    return CalibrationData(
        precipitation=precipitation,
        observed_runoff=observed_runoff,
        area_km2=100.0,
        times=times
    )


@pytest.fixture
def xinanjiang_calibrator(sample_calibration_data):
    """创建新安江校准器"""
    return GenericHydrologicCalibrator.create_for_model(
        data=sample_calibration_data,
        runoff_model_type='xin_an_jiang',
        algorithm='differential_evolution',
        algorithm_options={'maxiter': 2, 'popsize': 5},
        seed=42
    )


class TestHydrologicSensitivityAnalyzer:
    """测试HydrologicSensitivityAnalyzer基础功能"""

    def test_initialization(self, xinanjiang_calibrator):
        """测试初始化"""
        analyzer = HydrologicSensitivityAnalyzer(
            calibrator=xinanjiang_calibrator,
            method='morris',
            n_samples=5,
            seed=42
        )

        assert analyzer.method == 'morris'
        assert analyzer.n_samples == 5
        assert analyzer.seed == 42
        assert analyzer.model_name == 'xin_an_jiang'

    def test_initialization_auto_samples(self, xinanjiang_calibrator):
        """测试自动确定采样数量"""
        analyzer = HydrologicSensitivityAnalyzer(
            calibrator=xinanjiang_calibrator,
            method='morris'
        )

        # Morris方法应该自动设置为参数数量*4
        n_params = len(xinanjiang_calibrator.config.param_bounds)
        expected_samples = max(10, n_params * 4)
        assert analyzer.n_samples == expected_samples

    def test_invalid_method(self, xinanjiang_calibrator):
        """测试无效的敏感性分析方法"""
        analyzer = HydrologicSensitivityAnalyzer(
            calibrator=xinanjiang_calibrator,
            method='invalid_method'
        )

        with pytest.raises(ValueError, match="未知的敏感性分析方法"):
            analyzer.analyze()


class TestSensitivityAnalysisMorris:
    """测试Morris敏感性分析"""

    def test_morris_analysis(self, xinanjiang_calibrator):
        """测试Morris敏感性分析"""
        analyzer = HydrologicSensitivityAnalyzer(
            calibrator=xinanjiang_calibrator,
            method='morris',
            n_samples=3,  # 使用少量采样加快测试
            seed=42
        )

        result = analyzer.analyze()

        assert result is not None
        assert result.model_name == 'xin_an_jiang'
        assert result.model_type == 'runoff_only'
        assert len(result.sensitivity_result.param_names) == 4
        assert result.sensitivity_result.method == "Morris (Elementary Effects)"

    def test_morris_parameter_ranking(self, xinanjiang_calibrator):
        """测试Morris参数排名"""
        analyzer = HydrologicSensitivityAnalyzer(
            calibrator=xinanjiang_calibrator,
            method='morris',
            n_samples=3,
            seed=42
        )

        result = analyzer.analyze()

        # 检查排名
        rankings = result.sensitivity_result.sensitivity_rankings
        assert len(rankings) == 4
        assert all(param in rankings for param in ['wm', 'b', 'imp', 'recession'])

        # 检查敏感性指数
        for param in result.sensitivity_result.param_names:
            sens = result.sensitivity_result.sensitivity_indices[param]
            assert 0 <= sens <= 1

    def test_morris_critical_params(self, xinanjiang_calibrator):
        """测试识别关键参数"""
        analyzer = HydrologicSensitivityAnalyzer(
            calibrator=xinanjiang_calibrator,
            method='morris',
            n_samples=3,
            seed=42
        )

        result = analyzer.analyze()

        critical_params = result.get_critical_params(threshold=0.5)
        insensitive_params = result.get_insensitive_params(threshold=0.3)

        # 关键参数和不敏感参数不应重叠
        assert len(set(critical_params) & set(insensitive_params)) == 0


class TestSensitivityAnalysisOAT:
    """测试OAT敏感性分析"""

    def test_oat_analysis(self, xinanjiang_calibrator):
        """测试OAT敏感性分析"""
        analyzer = HydrologicSensitivityAnalyzer(
            calibrator=xinanjiang_calibrator,
            method='oat',
            n_samples=5,  # 每个参数5个采样点
            seed=42
        )

        result = analyzer.analyze()

        assert result is not None
        assert result.model_name == 'xin_an_jiang'
        assert result.sensitivity_result.method == "One-at-a-time (OAT)"

    def test_oat_parameter_ranking(self, xinanjiang_calibrator):
        """测试OAT参数排名"""
        analyzer = HydrologicSensitivityAnalyzer(
            calibrator=xinanjiang_calibrator,
            method='oat',
            n_samples=5,
            seed=42
        )

        result = analyzer.analyze()

        rankings = result.sensitivity_result.sensitivity_rankings
        assert len(rankings) == 4

        # 敏感性指数应在[0, 1]范围内
        for param in result.sensitivity_result.param_names:
            sens = result.sensitivity_result.sensitivity_indices[param]
            assert 0 <= sens <= 1


class TestSensitivityRecommendations:
    """测试敏感性分析建议"""

    def test_parameter_recommendations(self, xinanjiang_calibrator):
        """测试参数校准建议"""
        analyzer = HydrologicSensitivityAnalyzer(
            calibrator=xinanjiang_calibrator,
            method='morris',
            n_samples=3,
            seed=42
        )

        result = analyzer.analyze()

        # 每个参数都应该有建议
        assert len(result.parameter_recommendations) == 4

        for param, recommendation in result.parameter_recommendations.items():
            assert isinstance(recommendation, str)
            assert len(recommendation) > 0

    def test_suggested_bounds(self, xinanjiang_calibrator):
        """测试建议的参数边界"""
        analyzer = HydrologicSensitivityAnalyzer(
            calibrator=xinanjiang_calibrator,
            method='morris',
            n_samples=3,
            seed=42
        )

        result = analyzer.analyze()

        # 每个参数都应该有建议的边界
        assert len(result.suggested_bounds) == 4

        original_bounds = xinanjiang_calibrator.config.param_bounds

        for param in result.sensitivity_result.param_names:
            original = original_bounds[param]
            suggested = result.suggested_bounds[param]

            # 建议的边界应该在原始边界内
            assert suggested[0] >= original[0]
            assert suggested[1] <= original[1]
            assert suggested[0] < suggested[1]

    def test_bounds_adjustment_based_on_sensitivity(self, xinanjiang_calibrator):
        """测试基于敏感性的边界调整"""
        analyzer = HydrologicSensitivityAnalyzer(
            calibrator=xinanjiang_calibrator,
            method='morris',
            n_samples=3,
            seed=42
        )

        result = analyzer.analyze()

        original_bounds = xinanjiang_calibrator.config.param_bounds

        for param in result.sensitivity_result.param_names:
            sens = result.sensitivity_result.sensitivity_indices[param]
            original = original_bounds[param]
            suggested = result.suggested_bounds[param]

            original_range = original[1] - original[0]
            suggested_range = suggested[1] - suggested[0]

            # 低敏感性参数的范围应该缩小
            if sens < 0.3:
                assert suggested_range < original_range


class TestSensitivitySaveResults:
    """测试敏感性分析结果保存"""

    def test_save_results(self, xinanjiang_calibrator):
        """测试保存敏感性分析结果"""
        analyzer = HydrologicSensitivityAnalyzer(
            calibrator=xinanjiang_calibrator,
            method='morris',
            n_samples=3,
            seed=42
        )

        result = analyzer.analyze()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            analyzer.save_results(result, output_dir, prefix="test_")

            # 验证文件存在
            assert (output_dir / "test_sensitivity_analysis.json").exists()
            assert (output_dir / "test_parameter_sensitivity.csv").exists()
            assert (output_dir / "test_sensitivity_report.txt").exists()

    def test_to_dict(self, xinanjiang_calibrator):
        """测试转换为字典"""
        analyzer = HydrologicSensitivityAnalyzer(
            calibrator=xinanjiang_calibrator,
            method='morris',
            n_samples=3,
            seed=42
        )

        result = analyzer.analyze()
        result_dict = result.to_dict()

        assert 'model_name' in result_dict
        assert 'model_type' in result_dict
        assert 'critical_params' in result_dict
        assert 'insensitive_params' in result_dict
        assert 'parameter_recommendations' in result_dict
        assert 'suggested_bounds' in result_dict
        assert 'sensitivity_indices' in result_dict


class TestAnalyzeModelSensitivityConvenience:
    """测试analyze_model_sensitivity便捷函数"""

    def test_convenience_function(self, xinanjiang_calibrator):
        """测试便捷函数"""
        result = analyze_model_sensitivity(
            calibrator=xinanjiang_calibrator,
            method='morris',
            n_samples=3
        )

        assert result is not None
        assert result.model_name == 'xin_an_jiang'

    def test_convenience_function_with_output_dir(self, xinanjiang_calibrator):
        """测试便捷函数 - 自动保存结果"""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = analyze_model_sensitivity(
                calibrator=xinanjiang_calibrator,
                method='morris',
                n_samples=3,
                output_dir=tmpdir
            )

            output_path = Path(tmpdir)
            assert (output_path / "sensitivity_analysis.json").exists()


class TestSensitivityWithDifferentModels:
    """测试不同模型的敏感性分析"""

    def test_sensitivity_different_models(self, sample_calibration_data):
        """测试对不同模型进行敏感性分析"""
        models_to_test = ['xin_an_jiang']

        # 如果HBV可用，也测试它
        available = GenericHydrologicCalibrator.list_available_models()
        if 'hbv' in available['runoff_models']:
            sample_calibration_data.temperature = np.ones(50) * 10.0
            models_to_test.append('hbv')

        for model_name in models_to_test:
            calibrator = GenericHydrologicCalibrator.create_for_model(
                data=sample_calibration_data,
                runoff_model_type=model_name,
                seed=42
            )

            result = analyze_model_sensitivity(
                calibrator,
                method='morris',
                n_samples=3
            )

            assert result is not None
            assert result.model_name == model_name
            assert len(result.sensitivity_result.param_names) > 0
