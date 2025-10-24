"""GenericHydrologicCalibrator单元测试

测试通用水文模型校准器的核心功能。
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from hydrosis.calibration import (
    CalibrationData,
    CalibrationConfig,
    GenericHydrologicCalibrator,
    ModelMode,
)
from hydrosis.runoff.base import RunoffModelConfig
from hydrosis.routing.base import RoutingModelConfig


@pytest.fixture
def sample_calibration_data():
    """创建示例校准数据"""
    n = 100
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
def xinanjiang_bounds():
    """新安江模型参数边界"""
    return {
        'wm': [100, 200],
        'b': [0.2, 0.4],
        'imp': [0.0, 0.1],
        'recession': [0.5, 0.7],
    }


@pytest.fixture
def hbv_bounds():
    """HBV模型参数边界"""
    return {
        'FC': [250, 350],
        'BETA': [1.5, 2.5],
        'K0': [0.1, 0.2],
    }


@pytest.fixture
def muskingum_bounds():
    """Muskingum汇流模型参数边界"""
    return {
        'K': [1.0, 5.0],
        'x': [0.1, 0.3],
    }


class TestGenericHydrologicCalibrator:
    """测试GenericHydrologicCalibrator基础功能"""

    def test_initialization_runoff_only(self, sample_calibration_data, xinanjiang_bounds):
        """测试初始化 - 仅产流模型"""
        config = CalibrationConfig(
            param_bounds=xinanjiang_bounds,
            algorithm='differential_evolution',
            algorithm_options={'maxiter': 2, 'popsize': 5},
            seed=42
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            calibrator = GenericHydrologicCalibrator(
                data=sample_calibration_data,
                config=config,
                runoff_model_type='xin_an_jiang',
                output_dir=tmpdir
            )

            assert calibrator.mode == ModelMode.RUNOFF_ONLY
            assert calibrator.runoff_model_type == 'xin_an_jiang'
            assert calibrator.routing_model_type is None
            assert len(calibrator.config.param_bounds) == 4

    def test_initialization_routing_only(self, sample_calibration_data, muskingum_bounds):
        """测试初始化 - 仅汇流模型"""
        config = CalibrationConfig(
            param_bounds=muskingum_bounds,
            algorithm='differential_evolution',
            algorithm_options={'maxiter': 2, 'popsize': 5},
            seed=42
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            calibrator = GenericHydrologicCalibrator(
                data=sample_calibration_data,
                config=config,
                routing_model_type='muskingum',
                output_dir=tmpdir
            )

            assert calibrator.mode == ModelMode.ROUTING_ONLY
            assert calibrator.runoff_model_type is None
            assert calibrator.routing_model_type == 'muskingum'

    def test_initialization_coupled(self, sample_calibration_data, xinanjiang_bounds, muskingum_bounds):
        """测试初始化 - 产汇流组合"""
        coupled_bounds = {**xinanjiang_bounds, **muskingum_bounds}

        config = CalibrationConfig(
            param_bounds=coupled_bounds,
            algorithm='differential_evolution',
            algorithm_options={'maxiter': 2, 'popsize': 5},
            seed=42
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            calibrator = GenericHydrologicCalibrator(
                data=sample_calibration_data,
                config=config,
                runoff_model_type='xin_an_jiang',
                routing_model_type='muskingum',
                output_dir=tmpdir
            )

            assert calibrator.mode == ModelMode.COUPLED
            assert calibrator.runoff_model_type == 'xin_an_jiang'
            assert calibrator.routing_model_type == 'muskingum'
            assert len(calibrator.runoff_param_names) == 4
            assert len(calibrator.routing_param_names) == 2

    def test_error_no_model(self, sample_calibration_data):
        """测试错误处理 - 未提供模型"""
        config = CalibrationConfig(
            param_bounds={'test': [0, 1]},
            seed=42
        )

        with pytest.raises(ValueError, match="必须至少提供一个模型"):
            GenericHydrologicCalibrator(
                data=sample_calibration_data,
                config=config
            )

    def test_error_unknown_runoff_model(self, sample_calibration_data):
        """测试错误处理 - 未知产流模型"""
        config = CalibrationConfig(
            param_bounds={'test': [0, 1]},
            seed=42
        )

        with pytest.raises(ValueError, match="未注册"):
            GenericHydrologicCalibrator(
                data=sample_calibration_data,
                config=config,
                runoff_model_type='nonexistent_model'
            )

    def test_create_model_xinanjiang(self, sample_calibration_data, xinanjiang_bounds):
        """测试创建新安江模型"""
        config = CalibrationConfig(
            param_bounds=xinanjiang_bounds,
            seed=42
        )

        calibrator = GenericHydrologicCalibrator(
            data=sample_calibration_data,
            config=config,
            runoff_model_type='xin_an_jiang'
        )

        params = {'wm': 150, 'b': 0.3, 'imp': 0.05, 'recession': 0.6}
        model = calibrator.create_model(params)

        assert model is not None
        from hydrosis.runoff.xinanjiang import XinAnJiangRunoff
        assert isinstance(model, XinAnJiangRunoff)

    def test_run_model_xinanjiang(self, sample_calibration_data, xinanjiang_bounds):
        """测试运行新安江模型"""
        config = CalibrationConfig(
            param_bounds=xinanjiang_bounds,
            seed=42
        )

        calibrator = GenericHydrologicCalibrator(
            data=sample_calibration_data,
            config=config,
            runoff_model_type='xin_an_jiang'
        )

        params = {'wm': 150, 'b': 0.3, 'imp': 0.05, 'recession': 0.6}
        model = calibrator.create_model(params)
        simulated = calibrator.run_model(model)

        assert len(simulated) == 100
        assert isinstance(simulated, np.ndarray)
        assert not np.any(np.isnan(simulated))
        assert np.all(simulated >= 0)  # 径流应为非负

    def test_get_default_param_bounds_xinanjiang(self):
        """测试获取新安江默认参数边界"""
        bounds = GenericHydrologicCalibrator.get_default_param_bounds(
            'xin_an_jiang', 'runoff'
        )

        assert 'wm' in bounds
        assert 'b' in bounds
        assert 'imp' in bounds
        assert 'recession' in bounds
        assert bounds['wm'] == (50.0, 250.0)
        assert bounds['b'] == (0.1, 0.5)

    def test_get_default_param_bounds_muskingum(self):
        """测试获取Muskingum默认参数边界"""
        bounds = GenericHydrologicCalibrator.get_default_param_bounds(
            'muskingum', 'routing'
        )

        assert 'K' in bounds
        assert 'x' in bounds
        assert bounds['K'] == (0.1, 10.0)
        assert bounds['x'] == (0.0, 0.5)

    def test_list_available_models(self):
        """测试列出可用模型"""
        models = GenericHydrologicCalibrator.list_available_models()

        assert 'runoff_models' in models
        assert 'routing_models' in models
        assert 'xin_an_jiang' in models['runoff_models']
        assert 'muskingum' in models['routing_models']


class TestGenericHydrologicCalibratorConvenience:
    """测试GenericHydrologicCalibrator便捷方法"""

    def test_create_for_model_xinanjiang(self, sample_calibration_data):
        """测试create_for_model便捷方法 - 新安江"""
        calibrator = GenericHydrologicCalibrator.create_for_model(
            data=sample_calibration_data,
            runoff_model_type='xin_an_jiang',
            algorithm='differential_evolution',
            seed=42
        )

        assert calibrator.mode == ModelMode.RUNOFF_ONLY
        assert calibrator.runoff_model_type == 'xin_an_jiang'
        assert len(calibrator.config.param_bounds) == 4  # 默认参数

    def test_create_for_model_with_custom_bounds(self, sample_calibration_data):
        """测试create_for_model - 自定义参数边界"""
        custom_bounds = {
            'wm': [120, 180],
            'b': [0.25, 0.35],
        }

        calibrator = GenericHydrologicCalibrator.create_for_model(
            data=sample_calibration_data,
            runoff_model_type='xin_an_jiang',
            param_bounds=custom_bounds,
            seed=42
        )

        assert calibrator.config.param_bounds == custom_bounds

    def test_create_for_model_coupled(self, sample_calibration_data):
        """测试create_for_model - 产汇流组合"""
        calibrator = GenericHydrologicCalibrator.create_for_model(
            data=sample_calibration_data,
            runoff_model_type='xin_an_jiang',
            routing_model_type='muskingum',
            seed=42
        )

        assert calibrator.mode == ModelMode.COUPLED
        assert len(calibrator.config.param_bounds) == 6  # 4产流 + 2汇流


class TestGenericHydrologicCalibratorCalibration:
    """测试GenericHydrologicCalibrator校准功能"""

    def test_small_calibration_xinanjiang(self, sample_calibration_data):
        """测试小规模校准 - 新安江"""
        config = CalibrationConfig(
            param_bounds={
                'wm': [140, 160],
                'b': [0.25, 0.35],
            },
            fixed_params={
                'imp': 0.05,
                'recession': 0.6,
            },
            algorithm='differential_evolution',
            algorithm_options={'maxiter': 2, 'popsize': 5},
            objective_metric='nse',
            seed=42
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            calibrator = GenericHydrologicCalibrator(
                data=sample_calibration_data,
                config=config,
                runoff_model_type='xin_an_jiang',
                output_dir=tmpdir
            )

            result = calibrator.run_calibration()

            assert result is not None
            assert 'wm' in result.best_params
            assert 'b' in result.best_params
            assert result.best_score is not None
            assert len(result.simulated_runoff) == 100
            assert result.success is True

    def test_calibration_with_multiple_models(self, sample_calibration_data):
        """测试不同模型的校准"""
        models_to_test = ['xin_an_jiang']

        # 如果HBV可用，也测试它
        available = GenericHydrologicCalibrator.list_available_models()
        if 'hbv' in available['runoff_models']:
            # HBV需要温度数据
            sample_calibration_data.temperature = np.ones(100) * 10.0
            models_to_test.append('hbv')

        for model_name in models_to_test:
            calibrator = GenericHydrologicCalibrator.create_for_model(
                data=sample_calibration_data,
                runoff_model_type=model_name,
                algorithm='differential_evolution',
                algorithm_options={'maxiter': 2, 'popsize': 5},
                seed=42
            )

            # 简化参数边界以加快测试
            simplified_bounds = dict(list(calibrator.config.param_bounds.items())[:2])
            calibrator.config.param_bounds = simplified_bounds

            result = calibrator.run_calibration()

            assert result is not None
            assert result.success is True
            assert len(result.best_params) == 2


class TestGenericHydrologicCalibratorSaveResults:
    """测试GenericHydrologicCalibrator结果保存"""

    def test_save_results(self, sample_calibration_data):
        """测试保存校准结果"""
        from hydrosis.calibration import CalibrationResult

        config = CalibrationConfig(
            param_bounds={'wm': [100, 200], 'b': [0.1, 0.5]},
            seed=42
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            calibrator = GenericHydrologicCalibrator(
                data=sample_calibration_data,
                config=config,
                runoff_model_type='xin_an_jiang',
                output_dir=tmpdir
            )

            # 创建模拟结果
            result = CalibrationResult(
                best_params={'wm': 150.0, 'b': 0.3},
                best_score=0.85,
                simulated_runoff=np.ones(100),
                metrics={'nse': 0.85},
                algorithm='differential_evolution'
            )

            calibrator.save_results(result, prefix="test_")

            # 验证文件存在
            output_path = Path(tmpdir)
            assert (output_path / "test_best_parameters.json").exists()
            assert (output_path / "test_metrics.json").exists()
            assert (output_path / "test_timeseries.csv").exists()
