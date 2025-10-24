"""校准框架单元测试

测试BaseCalibrator和HBVCalibrator的核心功能。
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from hydrosis.calibration import (
    CalibrationData,
    CalibrationConfig,
    CalibrationResult,
    BaseCalibrator,
    HBVCalibrator,
)


@pytest.fixture
def sample_calibration_data():
    """创建示例校准数据"""
    n = 100
    precipitation = np.random.gamma(2, 2, size=n)
    observed_runoff = precipitation * 0.3 + np.random.normal(0, 0.5, n)
    observed_runoff = np.maximum(observed_runoff, 0)  # 确保非负
    temperature = 10 + 5 * np.sin(np.arange(n) / 10)
    
    times = pd.date_range('2024-01-01', periods=n, freq='h')
    
    return CalibrationData(
        precipitation=precipitation,
        observed_runoff=observed_runoff,
        area_km2=100.0,
        temperature=temperature,
        times=times
    )


@pytest.fixture
def sample_calibration_config():
    """创建示例校准配置"""
    return CalibrationConfig(
        param_bounds={
            'FC': [200, 400],
            'BETA': [1.0, 3.0],
            'K0': [0.05, 0.3],
        },
        fixed_params={'LP': 0.7, 'TT': 0.0},
        algorithm='sce_ua',
        algorithm_params={'max_iterations': 10, 'n_complexes': 2},
        objective_metric='nse',
        maximize=True,
        warmup_steps=10,
        seed=42
    )


class TestCalibrationData:
    """测试CalibrationData类"""
    
    def test_initialization(self, sample_calibration_data):
        """测试初始化"""
        data = sample_calibration_data
        assert data.n_timesteps == 100
        assert data.area_km2 == 100.0
        assert len(data.precipitation) == 100
        assert len(data.observed_runoff) == 100
    
    def test_length_validation(self):
        """测试长度验证"""
        with pytest.raises(ValueError, match="长度不一致"):
            CalibrationData(
                precipitation=np.array([1, 2, 3]),
                observed_runoff=np.array([1, 2]),  # 长度不匹配
                area_km2=100.0
            )
    
    def test_temperature_validation(self):
        """测试温度长度验证"""
        with pytest.raises(ValueError, match="温度序列长度"):
            CalibrationData(
                precipitation=np.array([1, 2, 3]),
                observed_runoff=np.array([1, 2, 3]),
                area_km2=100.0,
                temperature=np.array([10, 15])  # 长度不匹配
            )
    
    def test_with_metadata(self):
        """测试元数据"""
        data = CalibrationData(
            precipitation=np.array([1, 2, 3]),
            observed_runoff=np.array([1, 2, 3]),
            area_km2=100.0,
            metadata={'zone_id': 1, 'source': 'test'}
        )
        assert data.metadata['zone_id'] == 1
        assert data.metadata['source'] == 'test'


class TestCalibrationConfig:
    """测试CalibrationConfig类"""
    
    def test_initialization(self, sample_calibration_config):
        """测试初始化"""
        config = sample_calibration_config
        assert config.n_params == 3
        assert config.algorithm == 'sce_ua'
        assert config.objective_metric == 'nse'
        assert config.maximize is True
    
    def test_bounds_validation(self):
        """测试参数边界验证"""
        with pytest.raises(ValueError, match="参数边界必须"):
            CalibrationConfig(
                param_bounds={'FC': [200]},  # 格式错误
            )
    
    def test_invalid_bounds(self):
        """测试无效边界"""
        with pytest.raises(ValueError, match="参数边界无效"):
            CalibrationConfig(
                param_bounds={'FC': [400, 200]},  # min > max
            )
    
    def test_default_values(self):
        """测试默认值"""
        config = CalibrationConfig(
            param_bounds={'FC': [200, 400]}
        )
        assert config.algorithm == 'sce_ua'
        assert config.objective_metric == 'nse'
        assert config.maximize is True
        assert config.warmup_steps == 0


class TestCalibrationResult:
    """测试CalibrationResult类"""
    
    def test_initialization(self):
        """测试初始化"""
        result = CalibrationResult(
            best_params={'FC': 300.0, 'BETA': 2.0},
            best_score=0.85,
            simulated_runoff=np.array([1, 2, 3]),
            metrics={'nse': 0.85, 'rmse': 0.5},
            n_evaluations=100,
            computation_time=10.5,
            algorithm='sce_ua'
        )
        
        assert result.best_params['FC'] == 300.0
        assert result.best_score == 0.85
        assert result.success is True
        assert len(result.simulated_runoff) == 3
    
    def test_summary(self):
        """测试摘要生成"""
        result = CalibrationResult(
            best_params={'FC': 300.0},
            best_score=0.85,
            simulated_runoff=np.array([1, 2, 3]),
            metrics={'nse': 0.85},
            algorithm='sce_ua'
        )
        
        summary = result.summary()
        assert '校准结果摘要' in summary
        assert 'sce_ua' in summary
        assert '0.85' in summary


class TestHBVCalibrator:
    """测试HBVCalibrator类"""
    
    def test_initialization(self, sample_calibration_data, sample_calibration_config):
        """测试初始化"""
        with tempfile.TemporaryDirectory() as tmpdir:
            calibrator = HBVCalibrator(
                sample_calibration_data,
                sample_calibration_config,
                output_dir=tmpdir
            )
            
            assert calibrator.data.n_timesteps == 100
            assert calibrator.config.n_params == 3
            assert calibrator.output_dir == Path(tmpdir)
    
    def test_create_model(self, sample_calibration_data, sample_calibration_config):
        """测试创建模型"""
        with tempfile.TemporaryDirectory() as tmpdir:
            calibrator = HBVCalibrator(
                sample_calibration_data,
                sample_calibration_config,
                output_dir=tmpdir
            )
            
            params = {'FC': 300.0, 'BETA': 2.0, 'K0': 0.15}
            model = calibrator.create_model(params)
            
            assert model is not None
            from hydrosis.runoff.hbv import HBVRunoff
            assert isinstance(model, HBVRunoff)
    
    def test_run_model(self, sample_calibration_data, sample_calibration_config):
        """测试运行模型"""
        with tempfile.TemporaryDirectory() as tmpdir:
            calibrator = HBVCalibrator(
                sample_calibration_data,
                sample_calibration_config,
                output_dir=tmpdir
            )
            
            params = {'FC': 300.0, 'BETA': 2.0, 'K0': 0.15}
            model = calibrator.create_model(params)
            simulated = calibrator.run_model(model)
            
            assert len(simulated) == 100
            assert isinstance(simulated, np.ndarray)
            assert not np.any(np.isnan(simulated))
    
    def test_objective_function(self, sample_calibration_data, sample_calibration_config):
        """测试目标函数"""
        with tempfile.TemporaryDirectory() as tmpdir:
            calibrator = HBVCalibrator(
                sample_calibration_data,
                sample_calibration_config,
                output_dir=tmpdir
            )
            
            objective = calibrator.create_objective_function()
            
            # 测试参数列表
            params_list = [300.0, 2.0, 0.15]
            score = objective(params_list)
            
            assert isinstance(score, (int, float))
            assert not np.isnan(score)
    
    def test_calibration_run_small(self, sample_calibration_data):
        """测试小规模校准运行"""
        # 使用极少的迭代次数快速测试
        config = CalibrationConfig(
            param_bounds={
                'FC': [250, 350],
                'BETA': [1.5, 2.5],
            },
            fixed_params={
                'LP': 0.7,
                'K0': 0.15,
                'K1': 0.05,
                'K2': 0.01,
                'PERC': 2.0,
                'TT': 0.0,
                'CFMAX': 3.0,
                'CFR': 0.05,
                'CWH': 0.1,
            },
            algorithm='differential_evolution',
            algorithm_params={'maxiter': 3, 'popsize': 5},
            objective_metric='nse',
            seed=42
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            calibrator = HBVCalibrator(
                sample_calibration_data,
                config,
                output_dir=tmpdir
            )
            
            result = calibrator.run_calibration()
            
            assert result is not None
            assert 'FC' in result.best_params
            assert 'BETA' in result.best_params
            assert result.best_score is not None
            assert len(result.simulated_runoff) == 100
            assert result.success is True
    
    def test_save_results(self, sample_calibration_data, sample_calibration_config):
        """测试保存结果"""
        with tempfile.TemporaryDirectory() as tmpdir:
            calibrator = HBVCalibrator(
                sample_calibration_data,
                sample_calibration_config,
                output_dir=tmpdir
            )
            
            # 创建模拟结果
            result = CalibrationResult(
                best_params={'FC': 300.0, 'BETA': 2.0},
                best_score=0.85,
                simulated_runoff=np.ones(100),
                metrics={'nse': 0.85, 'kge': 0.80},
                algorithm='sce_ua'
            )
            
            calibrator.save_results(result, prefix="test_")
            
            # 验证文件存在
            assert (Path(tmpdir) / "test_best_parameters.json").exists()
            assert (Path(tmpdir) / "test_metrics.json").exists()
            assert (Path(tmpdir) / "test_timeseries.csv").exists()
            assert (Path(tmpdir) / "test_summary.txt").exists()


class TestMetricCalculation:
    """测试指标计算"""
    
    def test_nse_calculation(self):
        """测试NSE计算"""
        observed = np.array([1, 2, 3, 4, 5])
        simulated = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        nse = BaseCalibrator.calculate_metric(observed, simulated, 'nse')
        assert isinstance(nse, float)
        assert 0 <= nse <= 1
    
    def test_kge_calculation(self):
        """测试KGE计算"""
        observed = np.array([1, 2, 3, 4, 5])
        simulated = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        kge = BaseCalibrator.calculate_metric(observed, simulated, 'kge')
        assert isinstance(kge, float)
    
    def test_rmse_calculation(self):
        """测试RMSE计算"""
        observed = np.array([1, 2, 3, 4, 5])
        simulated = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        rmse = BaseCalibrator.calculate_metric(observed, simulated, 'rmse')
        assert isinstance(rmse, float)
        assert rmse >= 0
    
    def test_unknown_metric(self):
        """测试未知指标"""
        observed = np.array([1, 2, 3])
        simulated = np.array([1, 2, 3])
        
        with pytest.raises(ValueError, match="未知指标"):
            BaseCalibrator.calculate_metric(observed, simulated, 'unknown')
