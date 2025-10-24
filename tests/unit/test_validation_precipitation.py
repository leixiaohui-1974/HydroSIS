"""降雨数据验证模块的单元测试

测试 PrecipitationCriteria 和 validate_precipitation_data 功能
"""
import pytest
import numpy as np
import pandas as pd

from hydrosis.validation.precipitation import (
    PrecipitationCriteria,
    validate_precipitation_data,
)


@pytest.fixture
def valid_precipitation_df():
    """创建有效的降雨数据"""
    dates = pd.date_range('2024-01-01', periods=100, freq='h')
    data = {
        'station_1': np.random.gamma(2, 2, size=100),
        'station_2': np.random.gamma(2, 2, size=100),
        'station_3': np.random.gamma(2, 2, size=100),
    }
    df = pd.DataFrame(data, index=dates)
    # 标准化到合理的总降雨量
    df = df / df.sum().mean() * 200  # 200mm 总降雨
    return df


class TestPrecipitationCriteria:
    """测试 PrecipitationCriteria 类"""

    def test_default_initialization(self):
        """测试默认初始化"""
        criteria = PrecipitationCriteria()
        assert criteria.min_value == 0.0
        assert criteria.max_value == 100.0
        assert criteria.max_daily_value == 500.0
        assert criteria.max_spatial_cv == 0.5
        assert criteria.min_spatial_correlation == 0.3
        assert criteria.max_missing_ratio == 0.1
        assert criteria.max_consecutive_zeros == 24

    def test_custom_initialization(self):
        """测试自定义初始化"""
        criteria = PrecipitationCriteria(
            name="strict",
            min_value=0.0,
            max_value=50.0,
            max_spatial_cv=0.3,
            strict_mode=True
        )
        assert criteria.name == "strict"
        assert criteria.max_value == 50.0
        assert criteria.max_spatial_cv == 0.3
        assert criteria.strict_mode is True

    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            "name": "custom",
            "max_value": 80.0,
            "max_spatial_cv": 0.4,
            "strict_mode": False,
        }
        criteria = PrecipitationCriteria.from_dict(data)
        # from_dict 只提取类中定义的字段
        assert criteria.max_value == 80.0
        assert criteria.max_spatial_cv == 0.4


class TestValidatePrecipitationData:
    """测试 validate_precipitation_data 函数"""

    def test_valid_data(self, valid_precipitation_df):
        """测试有效的降雨数据"""
        result = validate_precipitation_data(valid_precipitation_df)
        
        assert result.is_valid is True
        assert 'min_value' in result.metrics
        assert 'max_value' in result.metrics
        assert 'spatial_cv' in result.metrics
        assert result.metrics['min_value'] >= 0

    def test_negative_values(self):
        """测试负降雨值"""
        dates = pd.date_range('2024-01-01', periods=10, freq='h')
        data = {
            'station_1': [1, 2, -1, 4, 5, 6, 7, 8, 9, 10],  # 包含负值
            'station_2': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
        df = pd.DataFrame(data, index=dates)
        
        result = validate_precipitation_data(df)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any('负降雨值' in err for err in result.errors)

    def test_high_intensity_warning(self):
        """测试异常高降雨强度警告"""
        dates = pd.date_range('2024-01-01', periods=10, freq='h')
        data = {
            'station_1': [150, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 异常高值
            'station_2': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
        df = pd.DataFrame(data, index=dates)
        
        criteria = PrecipitationCriteria(max_value=100.0)
        result = validate_precipitation_data(df, criteria=criteria)
        
        # 应该有警告但仍然有效
        assert len(result.warnings) > 0
        assert any('异常高降雨强度' in warn for warn in result.warnings)

    def test_high_spatial_cv_error(self):
        """测试高空间变异系数错误"""
        dates = pd.date_range('2024-01-01', periods=100, freq='h')
        # 创建空间差异极大的数据（使用变化的降雨以避免常数序列导致的相关性计算问题）
        np.random.seed(42)
        base_pattern = np.random.gamma(2, 2, size=100)
        data = {
            'station_1': base_pattern * 0.1,   # 低降雨区
            'station_2': base_pattern * 2.0,   # 高降雨区
            'station_3': base_pattern * 1.0,   # 中等降雨区
        }
        df = pd.DataFrame(data, index=dates)

        criteria = PrecipitationCriteria(max_spatial_cv=0.3)
        result = validate_precipitation_data(df, criteria=criteria)

        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any('空间变异系数过大' in err for err in result.errors)

    def test_custom_criteria(self, valid_precipitation_df):
        """测试自定义验证标准"""
        criteria = PrecipitationCriteria(
            name="strict",
            max_value=50.0,
            max_spatial_cv=0.2,
            strict_mode=True
        )
        
        result = validate_precipitation_data(
            valid_precipitation_df,
            criteria=criteria,
            step_name="严格验证"
        )
        
        assert result.step_name == "严格验证"
        assert 'spatial_cv' in result.metrics

    def test_metrics_calculation(self, valid_precipitation_df):
        """测试指标计算"""
        result = validate_precipitation_data(valid_precipitation_df)
        
        # 验证所有预期的指标都被计算
        expected_metrics = [
            'min_value',
            'max_value',
            'mean_total_precip',
            'std_total_precip',
            'spatial_cv',
        ]
        
        for metric in expected_metrics:
            assert metric in result.metrics, f"缺少指标: {metric}"
        
        # 验证指标值的合理性
        assert result.metrics['min_value'] >= 0
        assert result.metrics['max_value'] >= result.metrics['min_value']
        assert result.metrics['mean_total_precip'] > 0
        assert result.metrics['spatial_cv'] >= 0

    def test_single_station(self):
        """测试单站点数据"""
        dates = pd.date_range('2024-01-01', periods=50, freq='h')
        data = {'station_1': np.random.gamma(2, 2, size=50)}
        df = pd.DataFrame(data, index=dates)

        result = validate_precipitation_data(df)

        # 单站点的空间CV可能为NaN（标准差除以均值）
        # 这是预期行为，单站点没有空间变异性
        assert 'spatial_cv' in result.metrics

    def test_uniform_precipitation(self):
        """测试均匀降雨（所有站点相同）"""
        dates = pd.date_range('2024-01-01', periods=50, freq='h')
        precip = np.random.gamma(2, 2, size=50)
        data = {
            'station_1': precip.copy(),
            'station_2': precip.copy(),
            'station_3': precip.copy(),
        }
        df = pd.DataFrame(data, index=dates)
        
        result = validate_precipitation_data(df)
        
        # 完全均匀的降雨，空间CV应该为0
        assert result.metrics['spatial_cv'] == 0.0
        assert result.is_valid is True

    def test_empty_dataframe(self):
        """测试空DataFrame"""
        df = pd.DataFrame()

        # 空DataFrame会产生NaN值的指标
        # 这是一个边界情况，验证代码应该能够处理
        result = validate_precipitation_data(df)
        # 至少应该有指标被计算（即使是NaN）
        assert len(result.metrics) > 0
