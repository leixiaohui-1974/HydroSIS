"""HBV产流模型单元测试

测试HBV (Hydrologiska Byråns Vattenbalansavdelning) 模型的核心功能。
HBV是经典的概念性水文模型，广泛应用于全球流域模拟。
"""
import pytest
import numpy as np

from hydrosis.runoff.hbv import HBVRunoff
from hydrosis.validation import ParameterValidationError


class MockSubbasin:
    """模拟Subbasin对象"""
    def __init__(self, area_km2: float):
        self.area_km2 = area_km2


@pytest.fixture
def default_hbv_params():
    """默认HBV参数"""
    return {
        'degree_day_factor': 3.0,
        'snow_threshold': 0.0,
        'field_capacity': 100.0,
        'beta': 1.0,
        'k0': 0.15,
        'k1': 0.05,
        'k2': 0.01,
        'percolation': 2.0,
        'initial_snow': 0.0,
        'initial_soil': 40.0,
        'initial_upper': 5.0,
        'initial_lower': 20.0
    }


@pytest.fixture
def sample_subbasin():
    """示例子流域"""
    return MockSubbasin(area_km2=100.0)


class TestHBVInitialization:
    """测试HBV模型初始化"""

    def test_initialization_with_defaults(self):
        """测试使用默认参数初始化"""
        model = HBVRunoff({})

        assert model.degree_day_factor >= 0
        assert model.field_capacity > 0
        assert model.beta > 0
        assert 0 <= model.k0 <= 1
        assert 0 <= model.k1 <= 1
        assert 0 <= model.k2 <= 1
        assert model.percolation >= 0
        assert model.snow >= 0
        assert model.soil >= 0
        assert model.upper >= 0
        assert model.lower >= 0

    def test_initialization_with_custom_params(self, default_hbv_params):
        """测试使用自定义参数初始化"""
        model = HBVRunoff(default_hbv_params)

        assert model.degree_day_factor == 3.0
        assert model.snow_threshold == 0.0
        assert model.field_capacity == 100.0
        assert model.beta == 1.0
        assert model.k0 == 0.15
        assert model.k1 == 0.05
        assert model.k2 == 0.01
        assert model.percolation == 2.0
        assert model.snow == 0.0
        assert model.soil == 40.0
        assert model.upper == 5.0
        assert model.lower == 20.0

    def test_initialization_with_partial_params(self):
        """测试使用部分参数初始化"""
        params = {'field_capacity': 150.0, 'beta': 2.0}
        model = HBVRunoff(params)

        assert model.field_capacity == 150.0
        assert model.beta == 2.0
        # 其他参数应使用默认值
        assert model.k0 > 0


class TestHBVParameterValidation:
    """测试HBV模型参数验证"""

    def test_valid_parameters(self, default_hbv_params):
        """测试有效参数"""
        model = HBVRunoff(default_hbv_params)
        # 应该不抛出异常
        model.validate_parameters()

    def test_invalid_field_capacity_zero(self):
        """测试无效的field_capacity（零）"""
        params = {'field_capacity': 0.0}

        with pytest.raises(ParameterValidationError, match="field_capacity"):
            model = HBVRunoff(params)

    def test_invalid_field_capacity_negative(self):
        """测试无效的field_capacity（负数）"""
        params = {'field_capacity': -10.0}

        with pytest.raises(ParameterValidationError, match="field_capacity"):
            model = HBVRunoff(params)

    def test_invalid_beta_zero(self):
        """测试无效的beta（零）"""
        params = {'beta': 0.0}

        with pytest.raises(ParameterValidationError, match="beta"):
            model = HBVRunoff(params)

    def test_invalid_k0_above_one(self):
        """测试无效的k0（大于1）"""
        params = {'k0': 1.5}

        with pytest.raises(ParameterValidationError, match="k0"):
            model = HBVRunoff(params)

    def test_invalid_k1_negative(self):
        """测试无效的k1（负数）"""
        params = {'k1': -0.1}

        with pytest.raises(ParameterValidationError, match="k1"):
            model = HBVRunoff(params)

    def test_valid_degree_day_factor_zero(self):
        """测试degree_day_factor=0是有效的（无融雪）"""
        params = {'degree_day_factor': 0.0}
        model = HBVRunoff(params)
        # 应该不抛出异常
        model.validate_parameters()


class TestHBVSimulation:
    """测试HBV模型模拟"""

    def test_simulate_zero_rainfall(self, default_hbv_params, sample_subbasin):
        """测试零降雨情况"""
        model = HBVRunoff(default_hbv_params)
        precipitation = [0.0] * 10

        runoff = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 10
        assert all(isinstance(q, (int, float)) for q in runoff)
        assert all(q >= 0 for q in runoff)

    def test_simulate_constant_rainfall(self, default_hbv_params, sample_subbasin):
        """测试恒定降雨"""
        model = HBVRunoff(default_hbv_params)
        precipitation = [5.0] * 30  # 更长时间以观察稳定

        runoff = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 30
        assert all(q >= 0 for q in runoff)
        # 恒定降雨下，径流应产生
        assert sum(runoff) > 0
        # 长期恒定降雨，径流应逐渐增加并趋于稳定
        # 检查后期径流的变化率降低
        late_diff = abs(runoff[-1] - runoff[-5])
        early_diff = abs(runoff[4] - runoff[0])
        assert late_diff < early_diff * 2  # 后期变化应小于早期

    def test_simulate_varying_rainfall(self, default_hbv_params, sample_subbasin):
        """测试变化降雨"""
        model = HBVRunoff(default_hbv_params)
        # 模拟降雨事件
        precipitation = [0] * 5 + [10, 20, 15, 10, 5] + [1] * 10

        runoff = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 20
        assert all(q >= 0 for q in runoff)
        # 降雨期径流应大于无雨期
        rain_period_mean = np.mean(runoff[5:10])
        no_rain_mean = np.mean(runoff[:5])
        assert rain_period_mean > no_rain_mean

    def test_simulate_heavy_rainfall(self, default_hbv_params, sample_subbasin):
        """测试强降雨事件"""
        model = HBVRunoff(default_hbv_params)
        precipitation = [0] * 5 + [50.0] + [0] * 10

        runoff = model.simulate(sample_subbasin, precipitation)

        # 强降雨应产生显著径流
        assert max(runoff) > 0
        # 峰值径流应大于平均径流
        assert max(runoff) > np.mean(runoff) * 2


class TestHBVSnowProcess:
    """测试HBV模型的积雪过程"""

    def test_snow_accumulation(self, default_hbv_params, sample_subbasin):
        """测试积雪累积"""
        params = default_hbv_params.copy()
        params['snow_threshold'] = 1.0  # 温度阈值
        params['degree_day_factor'] = 3.0
        params['initial_snow'] = 0.0
        model = HBVRunoff(params)

        # 低温降水（降雪）
        precipitation = [2.0] * 5  # 超过阈值部分为降雪
        model.simulate(sample_subbasin, precipitation)

        # 应有积雪累积
        assert model.snow > 0

    def test_snow_melt(self, default_hbv_params, sample_subbasin):
        """测试融雪"""
        params = default_hbv_params.copy()
        params['snow_threshold'] = 0.0
        params['degree_day_factor'] = 5.0  # 高融雪系数
        params['initial_snow'] = 50.0  # 初始积雪
        model = HBVRunoff(params)

        initial_snow = model.snow

        # 暖湿天气（降雨，促进融雪）
        precipitation = [3.0] * 10
        runoff = model.simulate(sample_subbasin, precipitation)

        # 积雪应减少
        assert model.snow < initial_snow
        # 应产生径流（融雪和降雨）
        assert sum(runoff) > 0

    def test_snow_cold_weather(self, default_hbv_params, sample_subbasin):
        """测试寒冷天气（无融雪）"""
        params = default_hbv_params.copy()
        params['degree_day_factor'] = 0.0  # 无融雪
        params['initial_snow'] = 20.0
        model = HBVRunoff(params)

        initial_snow = model.snow

        # 无降雨
        precipitation = [0.0] * 10
        model.simulate(sample_subbasin, precipitation)

        # 积雪应保持不变（无融雪无降雪）
        assert model.snow == initial_snow


class TestHBVSoilMoisture:
    """测试HBV模型的土壤水分过程"""

    def test_soil_moisture_increases(self, default_hbv_params, sample_subbasin):
        """测试土壤水分增加"""
        params = default_hbv_params.copy()
        params['initial_soil'] = 20.0  # 低初始土壤水
        model = HBVRunoff(params)

        initial_soil = model.soil

        # 降雨应增加土壤水分
        precipitation = [5.0] * 10
        model.simulate(sample_subbasin, precipitation)

        # 土壤水分应增加
        assert model.soil > initial_soil

    def test_beta_parameter_effect(self, sample_subbasin):
        """测试beta参数对土壤水分曲线的影响"""
        # beta=1: 线性响应
        model_beta1 = HBVRunoff({
            'beta': 1.0,
            'field_capacity': 100.0,
            'initial_soil': 50.0,
            'initial_upper': 0.0
        })

        # beta=3: 强非线性响应
        model_beta3 = HBVRunoff({
            'beta': 3.0,
            'field_capacity': 100.0,
            'initial_soil': 50.0,
            'initial_upper': 0.0
        })

        precipitation = [10.0] * 5

        runoff1 = model_beta1.simulate(sample_subbasin, precipitation)
        runoff3 = model_beta3.simulate(sample_subbasin, precipitation)

        # 不同beta值应产生不同的径流过程
        assert runoff1 != runoff3


class TestHBVReservoirs:
    """测试HBV模型的水库过程"""

    def test_upper_reservoir_quickflow(self, default_hbv_params, sample_subbasin):
        """测试上层水库快速流"""
        params = default_hbv_params.copy()
        params['initial_upper'] = 50.0  # 高上层水
        params['k0'] = 0.3  # 高快速流系数
        model = HBVRunoff(params)

        # 无降雨，测试快速流消退
        precipitation = [0.0] * 10
        runoff = model.simulate(sample_subbasin, precipitation)

        # 应产生快速流
        assert runoff[0] > 0
        # 快速流应递减
        assert runoff[0] > runoff[-1]

    def test_lower_reservoir_baseflow(self, default_hbv_params, sample_subbasin):
        """测试下层水库基流"""
        params = default_hbv_params.copy()
        params['initial_lower'] = 100.0  # 高下层水
        params['k2'] = 0.05  # 基流系数
        params['initial_upper'] = 0.0  # 无上层水，隔离下层效应
        model = HBVRunoff(params)

        # 无降雨
        precipitation = [0.0] * 20
        runoff = model.simulate(sample_subbasin, precipitation)

        # 应产生持续基流
        assert all(q > 0 for q in runoff[:10])
        # 基流应缓慢递减
        assert runoff[0] > runoff[-1]

    def test_percolation_to_lower_reservoir(self, default_hbv_params, sample_subbasin):
        """测试向下层水库的下渗"""
        params = default_hbv_params.copy()
        params['initial_upper'] = 20.0
        params['initial_lower'] = 5.0
        params['percolation'] = 5.0  # 高下渗率
        model = HBVRunoff(params)

        initial_lower = model.lower

        # 降雨补给上层，促进下渗
        precipitation = [5.0] * 10
        model.simulate(sample_subbasin, precipitation)

        # 下层水库应增加
        assert model.lower > initial_lower

    def test_reservoir_non_negative(self, default_hbv_params, sample_subbasin):
        """测试水库不会出现负值"""
        params = default_hbv_params.copy()
        params['initial_upper'] = 1.0  # 低水量
        params['initial_lower'] = 1.0
        params['k0'] = 0.9  # 高消退系数
        params['k2'] = 0.9
        model = HBVRunoff(params)

        # 长期无降雨，水库耗尽
        precipitation = [0.0] * 50
        model.simulate(sample_subbasin, precipitation)

        # 水库应保持非负
        assert model.upper >= 0
        assert model.lower >= 0


class TestHBVEdgeCases:
    """测试HBV模型边界情况"""

    def test_empty_precipitation(self, default_hbv_params, sample_subbasin):
        """测试空降雨序列"""
        model = HBVRunoff(default_hbv_params)
        precipitation = []

        runoff = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 0

    def test_single_timestep(self, default_hbv_params, sample_subbasin):
        """测试单个时间步"""
        model = HBVRunoff(default_hbv_params)
        precipitation = [10.0]

        runoff = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 1
        assert runoff[0] >= 0

    def test_extreme_parameters(self, sample_subbasin):
        """测试极端参数值"""
        # 极小田间持水量（快速饱和）
        params = {
            'field_capacity': 10.0,
            'beta': 0.5,
            'k0': 0.5,
            'k1': 0.1,
            'k2': 0.02,
            'percolation': 1.0
        }
        model = HBVRunoff(params)
        precipitation = [10.0] * 10

        runoff = model.simulate(sample_subbasin, precipitation)

        assert all(q >= 0 for q in runoff)
        # 小field_capacity应快速饱和，产生较大径流
        assert max(runoff) > 0

    def test_different_area_sizes(self, default_hbv_params):
        """测试不同流域面积"""
        model = HBVRunoff(default_hbv_params)
        precipitation = [10.0] * 5

        # 小流域
        small_basin = MockSubbasin(10.0)
        runoff_small = model.simulate(small_basin, precipitation)

        # 重置模型状态
        model = HBVRunoff(default_hbv_params)

        # 大流域
        large_basin = MockSubbasin(1000.0)
        runoff_large = model.simulate(large_basin, precipitation)

        # 径流应与面积成正比
        ratio = large_basin.area_km2 / small_basin.area_km2
        assert abs(runoff_large[0] / runoff_small[0] - ratio) < 0.01

    def test_zero_initial_storages(self, sample_subbasin):
        """测试零初始存储"""
        params = {
            'field_capacity': 100.0,
            'beta': 1.0,
            'initial_snow': 0.0,
            'initial_soil': 0.0,
            'initial_upper': 0.0,
            'initial_lower': 0.0
        }
        model = HBVRunoff(params)

        precipitation = [10.0] * 10
        runoff = model.simulate(sample_subbasin, precipitation)

        # 模型应正常运行
        assert all(q >= 0 for q in runoff)


class TestHBVStorageBalances:
    """测试HBV模型的存储平衡"""

    def test_total_storage_changes(self, default_hbv_params, sample_subbasin):
        """测试总存储变化"""
        model = HBVRunoff(default_hbv_params)

        initial_total = model.snow + model.soil + model.upper + model.lower

        precipitation = [10.0] * 10
        runoff = model.simulate(sample_subbasin, precipitation)

        final_total = model.snow + model.soil + model.upper + model.lower

        # 降雨应导致总存储变化
        assert final_total != initial_total
        # 应产生径流
        assert sum(runoff) > 0

    def test_storage_depletion_no_rain(self, default_hbv_params, sample_subbasin):
        """测试无降雨时存储消耗"""
        model = HBVRunoff(default_hbv_params)

        initial_total = model.snow + model.soil + model.upper + model.lower

        # 长期无降雨
        precipitation = [0.0] * 30
        runoff = model.simulate(sample_subbasin, precipitation)

        final_total = model.snow + model.soil + model.upper + model.lower

        # 总存储应减少（通过径流流出）
        assert final_total < initial_total

    def test_recession_behavior(self, default_hbv_params, sample_subbasin):
        """测试消退过程"""
        params = default_hbv_params.copy()
        params['initial_upper'] = 50.0
        params['initial_lower'] = 50.0
        model = HBVRunoff(params)

        # 无降雨，观察消退
        precipitation = [0.0] * 20
        runoff = model.simulate(sample_subbasin, precipitation)

        # 径流应递减
        assert runoff[0] > runoff[5]
        assert runoff[5] > runoff[15]


class TestHBVIntegratedProcesses:
    """测试HBV模型的集成过程"""

    def test_snow_rain_mix(self, default_hbv_params, sample_subbasin):
        """测试雨雪混合事件"""
        params = default_hbv_params.copy()
        params['snow_threshold'] = 2.0  # 阈值
        params['degree_day_factor'] = 3.0
        params['initial_snow'] = 10.0
        model = HBVRunoff(params)

        # 混合降水（部分降雪，部分降雨）
        precipitation = [3.0] * 10
        runoff = model.simulate(sample_subbasin, precipitation)

        # 应产生径流（降雨和部分融雪）
        assert sum(runoff) > 0
        # 积雪可能增加或减少（取决于降雪和融雪的平衡）
        assert model.snow >= 0

    def test_full_hydrological_cycle(self, default_hbv_params, sample_subbasin):
        """测试完整水文循环"""
        model = HBVRunoff(default_hbv_params)

        # 复杂降雨序列
        precipitation = (
            [0] * 5 +           # 无雨期
            [10, 20, 15] +      # 强降雨事件
            [5] * 5 +           # 中等降雨
            [1] * 10 +          # 小雨
            [0] * 15            # 消退期
        )

        runoff = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == len(precipitation)
        assert all(q >= 0 for q in runoff)

        # 应观察到清晰的峰值和消退过程
        peak_idx = np.argmax(runoff[:20])  # 前20步内应有峰值
        # 峰值后应有消退
        if peak_idx < len(runoff) - 5:
            assert runoff[peak_idx] > runoff[peak_idx + 5]
