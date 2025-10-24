"""VIC产流模型单元测试

测试Variable Infiltration Capacity (VIC)模型的核心功能。
"""
import pytest
import numpy as np

from hydrosis.runoff.vic import VICRunoff
from hydrosis.validation import ParameterValidationError


class MockSubbasin:
    """模拟Subbasin对象"""
    def __init__(self, area_km2: float):
        self.area_km2 = area_km2


@pytest.fixture
def default_vic_params():
    """默认VIC参数"""
    return {
        'infiltration_shape': 0.3,
        'max_soil_moisture': 150.0,
        'baseflow_coefficient': 0.005,
        'recession': 0.95,
        'initial_surface': 5.0,
        'initial_root': 50.0,
        'initial_deep': 20.0
    }


@pytest.fixture
def sample_subbasin():
    """示例子流域"""
    return MockSubbasin(area_km2=100.0)


class TestVICRunoffInitialization:
    """测试VIC模型初始化"""

    def test_initialization_with_defaults(self):
        """测试使用默认参数初始化"""
        model = VICRunoff({})

        assert model.infiltration_shape > 0
        assert model.max_soil_moisture > 0
        assert model.baseflow_coefficient >= 0
        assert 0 <= model.recession <= 1
        assert model.surface_storage >= 0
        assert model.root_storage >= 0
        assert model.deep_storage >= 0

    def test_initialization_with_custom_params(self, default_vic_params):
        """测试使用自定义参数初始化"""
        model = VICRunoff(default_vic_params)

        assert model.infiltration_shape == 0.3
        assert model.max_soil_moisture == 150.0
        assert model.baseflow_coefficient == 0.005
        assert model.recession == 0.95
        assert model.surface_storage == 5.0
        assert model.root_storage == 50.0
        assert model.deep_storage == 20.0

    def test_initialization_with_partial_params(self):
        """测试使用部分参数初始化"""
        params = {'infiltration_shape': 0.5, 'max_soil_moisture': 200.0}
        model = VICRunoff(params)

        assert model.infiltration_shape == 0.5
        assert model.max_soil_moisture == 200.0
        # 其他参数应使用默认值
        assert model.baseflow_coefficient > 0


class TestVICRunoffParameterValidation:
    """测试VIC模型参数验证"""

    def test_valid_parameters(self, default_vic_params):
        """测试有效参数"""
        model = VICRunoff(default_vic_params)
        # 应该不抛出异常
        model.validate_parameters()

    def test_invalid_infiltration_shape_zero(self):
        """测试无效的infiltration_shape（零）"""
        params = {'infiltration_shape': 0.0}

        with pytest.raises(ParameterValidationError, match="infiltration_shape"):
            model = VICRunoff(params)

    def test_invalid_infiltration_shape_negative(self):
        """测试无效的infiltration_shape（负数）"""
        params = {'infiltration_shape': -0.1}

        with pytest.raises(ParameterValidationError, match="infiltration_shape"):
            model = VICRunoff(params)

    def test_invalid_max_soil_moisture(self):
        """测试无效的max_soil_moisture"""
        params = {'max_soil_moisture': 0.0}

        with pytest.raises(ParameterValidationError, match="max_soil_moisture"):
            model = VICRunoff(params)

    def test_invalid_recession_above_one(self):
        """测试无效的recession（大于1）"""
        params = {'recession': 1.5}

        with pytest.raises(ParameterValidationError, match="recession"):
            model = VICRunoff(params)

    def test_invalid_recession_negative(self):
        """测试无效的recession（负数）"""
        params = {'recession': -0.1}

        with pytest.raises(ParameterValidationError, match="recession"):
            model = VICRunoff(params)


class TestVICRunoffSimulation:
    """测试VIC模型模拟"""

    def test_simulate_zero_rainfall(self, default_vic_params, sample_subbasin):
        """测试零降雨情况"""
        model = VICRunoff(default_vic_params)
        precipitation = [0.0] * 10

        runoff = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 10
        assert all(isinstance(q, (int, float)) for q in runoff)
        assert all(q >= 0 for q in runoff)  # 径流应为非负

    def test_simulate_constant_rainfall(self, default_vic_params, sample_subbasin):
        """测试恒定降雨"""
        model = VICRunoff(default_vic_params)
        precipitation = [5.0] * 20

        runoff = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 20
        assert all(q >= 0 for q in runoff)
        # 恒定降雨下，径流应趋于稳定
        assert np.std(runoff[-5:]) < np.std(runoff[:5])

    def test_simulate_varying_rainfall(self, default_vic_params, sample_subbasin):
        """测试变化降雨"""
        model = VICRunoff(default_vic_params)
        # 模拟降雨事件
        precipitation = [0] * 5 + [10, 20, 15, 10, 5] + [1] * 10

        runoff = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 20
        assert all(q >= 0 for q in runoff)
        # 峰值径流应在降雨峰值附近
        peak_precip_idx = 6  # 降雨峰值在第6个时间步
        peak_runoff_idx = np.argmax(runoff)
        assert abs(peak_runoff_idx - peak_precip_idx) <= 2

    def test_simulate_heavy_rainfall(self, default_vic_params, sample_subbasin):
        """测试强降雨事件"""
        model = VICRunoff(default_vic_params)
        precipitation = [0] * 5 + [50.0] + [0] * 10

        runoff = model.simulate(sample_subbasin, precipitation)

        # 强降雨应产生显著径流
        assert max(runoff) > 0
        # 峰值径流应大于平均径流
        assert max(runoff) > np.mean(runoff) * 2


class TestVICRunoffWaterBalance:
    """测试VIC模型水量平衡"""

    def test_storage_changes_with_precipitation(self, default_vic_params, sample_subbasin):
        """测试降雨时存储变化"""
        model = VICRunoff(default_vic_params)

        # 记录初始状态
        initial_storage = (model.surface_storage +
                          model.root_storage +
                          model.deep_storage)

        precipitation = [10.0] * 10
        runoff = model.simulate(sample_subbasin, precipitation)

        # 记录最终状态
        final_storage = (model.surface_storage +
                        model.root_storage +
                        model.deep_storage)

        # 降雨应产生径流
        assert sum(runoff) > 0
        # 存储应该变化（可能增加或减少，取决于降雨和损失的平衡）
        assert final_storage != initial_storage

    def test_storage_depletion_with_zero_rainfall(self, default_vic_params, sample_subbasin):
        """测试零降雨时的存储消耗"""
        model = VICRunoff(default_vic_params)

        initial_storage = (model.surface_storage +
                          model.root_storage +
                          model.deep_storage)

        precipitation = [0.0] * 20
        runoff = model.simulate(sample_subbasin, precipitation)

        final_storage = (model.surface_storage +
                        model.root_storage +
                        model.deep_storage)

        # 零降雨时，存储应该减少（基流和下渗损失）
        assert final_storage < initial_storage

        # 应该产生一些基流径流
        assert sum(runoff) > 0


class TestVICRunoffStorages:
    """测试VIC模型的存储机制"""

    def test_storage_limits(self, default_vic_params, sample_subbasin):
        """测试存储不会超过限制"""
        model = VICRunoff(default_vic_params)
        # 极端强降雨
        precipitation = [100.0] * 10

        model.simulate(sample_subbasin, precipitation)

        # 表层和根系存储不应超过max_soil_moisture
        assert model.surface_storage >= 0
        assert model.root_storage >= 0
        assert model.root_storage <= model.max_soil_moisture
        assert model.deep_storage >= 0

    def test_storage_depletion(self, default_vic_params, sample_subbasin):
        """测试存储消耗"""
        params = default_vic_params.copy()
        params['initial_root'] = 100.0
        params['initial_deep'] = 50.0
        model = VICRunoff(params)

        initial_deep = model.deep_storage

        # 长期无降雨
        precipitation = [0.0] * 50
        model.simulate(sample_subbasin, precipitation)

        # 深层存储应减少（基流消耗）
        assert model.deep_storage < initial_deep

    def test_infiltration_mechanism(self, default_vic_params, sample_subbasin):
        """测试入渗和下渗机制"""
        # 使用较低的初始存储
        params = default_vic_params.copy()
        params['initial_root'] = 10.0
        params['initial_deep'] = 5.0
        model = VICRunoff(params)

        initial_deep = model.deep_storage

        # 降雨应促进入渗，进而增加深层存储（通过下渗）
        precipitation = [5.0] * 10
        runoff = model.simulate(sample_subbasin, precipitation)

        # 深层存储应增加（接收根系层下渗的水）
        assert model.deep_storage > initial_deep
        # 应该产生一些径流
        assert sum(runoff) > 0


class TestVICRunoffEdgeCases:
    """测试VIC模型边界情况"""

    def test_empty_precipitation(self, default_vic_params, sample_subbasin):
        """测试空降雨序列"""
        model = VICRunoff(default_vic_params)
        precipitation = []

        runoff = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 0

    def test_single_timestep(self, default_vic_params, sample_subbasin):
        """测试单个时间步"""
        model = VICRunoff(default_vic_params)
        precipitation = [10.0]

        runoff = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 1
        assert runoff[0] >= 0

    def test_extreme_parameters(self, sample_subbasin):
        """测试极端参数值"""
        # 极小的入渗shape（快速产流）
        params = {
            'infiltration_shape': 0.01,
            'max_soil_moisture': 50.0,
            'baseflow_coefficient': 0.1,
            'recession': 0.5
        }
        model = VICRunoff(params)
        precipitation = [10.0] * 10

        runoff = model.simulate(sample_subbasin, precipitation)

        assert all(q >= 0 for q in runoff)
        # 快速产流应产生较大径流
        assert max(runoff) > 0

    def test_different_area_sizes(self, default_vic_params):
        """测试不同流域面积"""
        model = VICRunoff(default_vic_params)
        precipitation = [10.0] * 5

        # 小流域
        small_basin = MockSubbasin(10.0)
        runoff_small = model.simulate(small_basin, precipitation)

        # 重置模型状态
        model = VICRunoff(default_vic_params)

        # 大流域
        large_basin = MockSubbasin(1000.0)
        runoff_large = model.simulate(large_basin, precipitation)

        # 径流应与面积成正比
        ratio = large_basin.area_km2 / small_basin.area_km2
        assert abs(runoff_large[0] / runoff_small[0] - ratio) < 0.01


class TestVICRunoffPercolation:
    """测试VIC模型的下渗过程"""

    def test_percolation_to_deep_layer(self, default_vic_params, sample_subbasin):
        """测试向深层下渗"""
        params = default_vic_params.copy()
        params['initial_root'] = 100.0
        params['initial_deep'] = 10.0
        model = VICRunoff(params)

        initial_deep = model.deep_storage

        # 中等降雨应促进下渗
        precipitation = [5.0] * 20
        model.simulate(sample_subbasin, precipitation)

        # 深层存储应增加
        assert model.deep_storage > initial_deep

    def test_baseflow_generation(self, default_vic_params, sample_subbasin):
        """测试基流生成"""
        params = default_vic_params.copy()
        params['initial_deep'] = 100.0
        params['baseflow_coefficient'] = 0.02
        model = VICRunoff(params)

        # 无降雨，应产生基流
        precipitation = [0.0] * 10
        runoff = model.simulate(sample_subbasin, precipitation)

        # 应有持续的基流
        assert all(q > 0 for q in runoff[:5])
        # 基流应递减
        assert runoff[0] > runoff[-1]
