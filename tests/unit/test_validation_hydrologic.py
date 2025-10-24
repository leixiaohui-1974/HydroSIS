"""水文验证模块的单元测试

测试径流系数、水量平衡等水文验证功能
"""
import pytest
import numpy as np

from hydrosis.validation.hydrologic import (
    HydrologicCriteria,
    validate_runoff_coefficient,
    validate_water_balance,
)


class TestHydrologicCriteria:
    """测试 HydrologicCriteria 类"""

    def test_default_initialization(self):
        """测试默认初始化"""
        criteria = HydrologicCriteria()
        assert criteria.runoff_coefficient_min == 0.0
        assert criteria.runoff_coefficient_max == 1.0
        assert criteria.runoff_coefficient_warning_low == 0.05
        assert criteria.runoff_coefficient_warning_high == 0.9
        assert criteria.water_balance_max_error == 0.01
        assert criteria.mass_conservation_tolerance == 0.001

    def test_custom_initialization(self):
        """测试自定义初始化"""
        criteria = HydrologicCriteria(
            name="strict",
            runoff_coefficient_min=0.1,
            runoff_coefficient_max=0.8,
            water_balance_max_error=0.005,
            strict_mode=True
        )
        assert criteria.name == "strict"
        assert criteria.runoff_coefficient_min == 0.1
        assert criteria.runoff_coefficient_max == 0.8
        assert criteria.water_balance_max_error == 0.005
        assert criteria.strict_mode is True

    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            "name": "custom",
            "runoff_coefficient_min": 0.05,
            "runoff_coefficient_max": 0.95,
            "water_balance_max_error": 0.02,
            "strict_mode": False,
        }
        criteria = HydrologicCriteria.from_dict(data)
        assert criteria.name == "custom"
        assert criteria.runoff_coefficient_min == 0.05
        assert criteria.runoff_coefficient_max == 0.95
        assert criteria.water_balance_max_error == 0.02


class TestValidateRunoffCoefficient:
    """测试 validate_runoff_coefficient 函数"""

    def test_valid_coefficients_dict(self):
        """测试有效的径流系数（字典格式）"""
        rcs = {
            'zone_1': 0.3,
            'zone_2': 0.4,
            'zone_3': 0.5,
        }

        result = validate_runoff_coefficient(rcs)

        assert result.is_valid is True
        assert 'mean_runoff_coefficient' in result.metrics
        assert 'min_runoff_coefficient' in result.metrics
        assert 'max_runoff_coefficient' in result.metrics
        assert result.metrics['mean_runoff_coefficient'] == pytest.approx(0.4, abs=0.01)

    def test_valid_coefficients_sequence(self):
        """测试有效的径流系数（序列格式）"""
        rcs = [0.3, 0.4, 0.5]
        zone_ids = ['zone_1', 'zone_2', 'zone_3']

        result = validate_runoff_coefficient(rcs, zone_ids=zone_ids)

        assert result.is_valid is True
        assert result.metrics['mean_runoff_coefficient'] == pytest.approx(0.4, abs=0.01)

    def test_coefficient_below_minimum(self):
        """测试径流系数低于最小值"""
        rcs = {'zone_1': -0.1, 'zone_2': 0.5}
        
        result = validate_runoff_coefficient(rcs)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any('zone_1' in err for err in result.errors)

    def test_coefficient_above_maximum(self):
        """测试径流系数超过最大值"""
        rcs = {'zone_1': 1.5, 'zone_2': 0.5}
        
        result = validate_runoff_coefficient(rcs)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any('zone_1' in err for err in result.errors)

    def test_low_coefficient_warning(self):
        """测试低径流系数警告"""
        rcs = {'zone_1': 0.02, 'zone_2': 0.5}  # 0.02 < 0.05 (warning threshold)
        
        criteria = HydrologicCriteria(runoff_coefficient_warning_low=0.05)
        result = validate_runoff_coefficient(rcs, criteria=criteria)
        
        assert result.is_valid is True  # 仍然有效
        assert len(result.warnings) > 0
        assert any('zone_1' in warn for warn in result.warnings)

    def test_high_coefficient_warning(self):
        """测试高径流系数警告"""
        rcs = {'zone_1': 0.95, 'zone_2': 0.5}  # 0.95 > 0.9 (warning threshold)
        
        criteria = HydrologicCriteria(runoff_coefficient_warning_high=0.9)
        result = validate_runoff_coefficient(rcs, criteria=criteria)
        
        assert result.is_valid is True  # 仍然有效
        assert len(result.warnings) > 0
        assert any('zone_1' in warn for warn in result.warnings)

    def test_multiple_invalid_zones(self):
        """测试多个无效分区"""
        rcs = {
            'zone_1': -0.1,  # 小于0
            'zone_2': 1.2,   # 大于1
            'zone_3': 0.5,   # 有效
        }
        
        result = validate_runoff_coefficient(rcs)
        
        assert result.is_valid is False
        assert len(result.errors) >= 2

    def test_edge_cases(self):
        """测试边界情况"""
        rcs = {
            'zone_1': 0.0,  # 边界：最小值
            'zone_2': 1.0,  # 边界：最大值
            'zone_3': 0.5,  # 中间值
        }

        result = validate_runoff_coefficient(rcs)

        assert result.is_valid is True
        assert result.metrics['min_runoff_coefficient'] == 0.0
        assert result.metrics['max_runoff_coefficient'] == 1.0

    def test_custom_step_name(self):
        """测试自定义步骤名称"""
        rcs = {'zone_1': 0.5}
        
        result = validate_runoff_coefficient(
            rcs,
            step_name="自定义验证步骤"
        )
        
        assert result.step_name == "自定义验证步骤"


class TestValidateWaterBalance:
    """测试 validate_water_balance 函数"""

    def test_perfect_balance(self):
        """测试完美的水量平衡"""
        precipitation = 100.0
        runoff = 40.0
        evapotranspiration = 30.0
        storage_change = 30.0

        result = validate_water_balance(
            precipitation=precipitation,
            runoff=runoff,
            evapotranspiration=evapotranspiration,
            storage_change=storage_change
        )

        assert result.is_valid is True
        assert 'balance_error_mm' in result.metrics
        assert abs(result.metrics['balance_error_mm']) < 1e-10

    def test_small_balance_error(self):
        """测试小的水量平衡误差（在容差范围内）"""
        precipitation = 100.0
        runoff = 40.0
        evapotranspiration = 30.0
        storage_change = 30.5  # 0.5mm 误差
        
        criteria = HydrologicCriteria(water_balance_max_error=0.01)  # 1% 容差
        result = validate_water_balance(
            precipitation=precipitation,
            runoff=runoff,
            evapotranspiration=evapotranspiration,
            storage_change=storage_change,
            criteria=criteria
        )
        
        assert result.is_valid is True

    def test_large_balance_error(self):
        """测试大的水量平衡误差（超出容差）"""
        precipitation = 100.0
        runoff = 40.0
        evapotranspiration = 30.0
        storage_change = 35.0  # 5mm 误差，5%
        
        criteria = HydrologicCriteria(water_balance_max_error=0.01)  # 1% 容差
        result = validate_water_balance(
            precipitation=precipitation,
            runoff=runoff,
            evapotranspiration=evapotranspiration,
            storage_change=storage_change,
            criteria=criteria
        )
        
        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_without_evapotranspiration(self):
        """测试不包含蒸散发的水量平衡"""
        precipitation = 100.0
        runoff = 40.0
        storage_change = 60.0
        
        result = validate_water_balance(
            precipitation=precipitation,
            runoff=runoff,
            storage_change=storage_change
        )
        
        assert result.is_valid is True

    def test_metrics_calculation(self):
        """测试指标计算"""
        precipitation = 100.0
        runoff = 40.0
        evapotranspiration = 30.0
        storage_change = 30.0

        result = validate_water_balance(
            precipitation=precipitation,
            runoff=runoff,
            evapotranspiration=evapotranspiration,
            storage_change=storage_change
        )

        assert 'total_precipitation_mm' in result.metrics
        assert 'total_runoff_mm' in result.metrics
        assert 'total_et_mm' in result.metrics
        assert 'total_storage_change_mm' in result.metrics
        assert 'balance_error_mm' in result.metrics
        assert 'relative_error' in result.metrics

    def test_negative_values(self):
        """测试负值（应该被允许，如降雨损失）"""
        precipitation = 100.0
        runoff = 40.0
        evapotranspiration = 30.0
        storage_change = -10.0  # 负值表示土壤水分减少

        # 这应该导致不平衡
        result = validate_water_balance(
            precipitation=precipitation,
            runoff=runoff,
            evapotranspiration=evapotranspiration,
            storage_change=storage_change
        )

        # 检查是否正确处理负值
        assert 'balance_error_mm' in result.metrics
