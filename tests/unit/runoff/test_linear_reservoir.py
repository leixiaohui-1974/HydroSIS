"""LinearReservoir产流模型的单元测试"""
import pytest
import numpy as np
from hydrosis.runoff.linear_reservoir import LinearReservoirRunoff
from hydrosis.validation import ParameterValidationError


@pytest.fixture
def default_linear_params():
    """默认LinearReservoir参数"""
    return {
        "recession": 0.9,
        "conversion": 1.0,
        "initial_storage": 0.0
    }


@pytest.fixture
def sample_subbasin():
    """样本子流域（用于测试）"""
    class MockSubbasin:
        def __init__(self):
            self.area_km2 = 100.0
    return MockSubbasin()


# ==================== 初始化测试 ====================

class TestLinearReservoirInitialization:
    """测试LinearReservoir模型初始化"""

    def test_default_initialization(self):
        """测试默认参数初始化"""
        model = LinearReservoirRunoff({})
        assert model.recession == 0.9
        assert model.conversion == 1.0
        assert model.state == 0.0

    def test_custom_initialization(self, default_linear_params):
        """测试自定义参数初始化"""
        model = LinearReservoirRunoff(default_linear_params)
        assert model.recession == 0.9
        assert model.conversion == 1.0
        assert model.state == 0.0

    def test_partial_initialization(self):
        """测试部分参数初始化"""
        params = {"recession": 0.85}
        model = LinearReservoirRunoff(params)
        assert model.recession == 0.85
        assert model.conversion == 1.0  # 默认值
        assert model.state == 0.0  # 默认值

    def test_initialization_with_storage(self):
        """测试带初始存储的初始化"""
        params = {
            "recession": 0.8,
            "conversion": 0.9,
            "initial_storage": 50.0
        }
        model = LinearReservoirRunoff(params)
        assert model.recession == 0.8
        assert model.conversion == 0.9
        assert model.state == 50.0


# ==================== 参数验证测试 ====================

class TestLinearReservoirParameterValidation:
    """测试参数验证"""

    def test_validate_default_parameters(self):
        """测试默认参数验证"""
        model = LinearReservoirRunoff({})
        model.validate_parameters()  # 不应抛出异常

    def test_validate_custom_parameters(self, default_linear_params):
        """测试自定义参数验证"""
        model = LinearReservoirRunoff(default_linear_params)
        model.validate_parameters()  # 不应抛出异常

    def test_recession_out_of_range_low(self):
        """测试recession小于0"""
        params = {"recession": -0.1}
        with pytest.raises(ParameterValidationError, match="recession"):
            model = LinearReservoirRunoff(params)

    def test_recession_out_of_range_high(self):
        """测试recession大于1"""
        params = {"recession": 1.5}
        with pytest.raises(ParameterValidationError, match="recession"):
            model = LinearReservoirRunoff(params)

    def test_recession_boundary_values(self):
        """测试recession边界值"""
        # recession = 0 应该有效
        model_0 = LinearReservoirRunoff({"recession": 0.0})
        model_0.validate_parameters()
        assert model_0.recession == 0.0

        # recession = 1 应该有效
        model_1 = LinearReservoirRunoff({"recession": 1.0})
        model_1.validate_parameters()
        assert model_1.recession == 1.0

    def test_negative_conversion(self):
        """测试负转换系数"""
        params = {"conversion": -0.5}
        with pytest.raises(ParameterValidationError, match="conversion"):
            model = LinearReservoirRunoff(params)

    def test_zero_conversion_valid(self):
        """测试零转换系数有效"""
        params = {"conversion": 0.0}
        model = LinearReservoirRunoff(params)
        model.validate_parameters()  # 不应抛出异常

    def test_negative_initial_storage(self):
        """测试负初始存储"""
        params = {"initial_storage": -10.0}
        with pytest.raises(ParameterValidationError, match="initial_storage"):
            model = LinearReservoirRunoff(params)

    def test_zero_initial_storage_valid(self):
        """测试零初始存储有效"""
        params = {"initial_storage": 0.0}
        model = LinearReservoirRunoff(params)
        model.validate_parameters()  # 不应抛出异常


# ==================== 仿真测试 ====================

class TestLinearReservoirSimulation:
    """测试仿真功能"""

    def test_simulate_zero_precipitation(self, default_linear_params, sample_subbasin):
        """测试零降雨"""
        model = LinearReservoirRunoff(default_linear_params)
        precipitation = [0.0] * 10
        runoff = model.simulate(sample_subbasin, precipitation)

        # 零降雨且零初始存储应该产生零径流
        assert len(runoff) == 10
        assert all(r == 0.0 for r in runoff)

    def test_simulate_constant_precipitation(self, default_linear_params, sample_subbasin):
        """测试恒定降雨"""
        model = LinearReservoirRunoff(default_linear_params)
        precipitation = [10.0] * 20
        runoff = model.simulate(sample_subbasin, precipitation)

        # 径流应该随时间增加，然后趋于稳定
        assert len(runoff) == 20
        assert runoff[0] < runoff[10] < runoff[19]

    def test_simulate_varying_precipitation(self, default_linear_params, sample_subbasin):
        """测试变化降雨"""
        model = LinearReservoirRunoff(default_linear_params)
        precipitation = [5.0, 10.0, 15.0, 10.0, 5.0]
        runoff = model.simulate(sample_subbasin, precipitation)

        # 径流应该跟随降雨趋势（有延迟）
        assert len(runoff) == 5
        assert all(r > 0 for r in runoff)

    def test_simulate_heavy_precipitation(self, default_linear_params, sample_subbasin):
        """测试强降雨"""
        model = LinearReservoirRunoff(default_linear_params)
        precipitation = [100.0] * 5
        runoff = model.simulate(sample_subbasin, precipitation)

        # 强降雨应该产生大径流
        assert len(runoff) == 5
        assert all(r > 0 for r in runoff)
        assert max(runoff) > 100.0  # 考虑面积


# ==================== 存储机制测试 ====================

class TestLinearReservoirStorage:
    """测试存储机制"""

    def test_storage_increases_with_precipitation(self, default_linear_params, sample_subbasin):
        """测试降雨导致存储增加"""
        model = LinearReservoirRunoff(default_linear_params)
        initial_state = model.state

        precipitation = [10.0] * 5
        model.simulate(sample_subbasin, precipitation)

        assert model.state > initial_state

    def test_storage_decreases_without_precipitation(self, sample_subbasin):
        """测试无降雨时存储减少"""
        params = {
            "recession": 0.9,
            "conversion": 1.0,
            "initial_storage": 100.0
        }
        model = LinearReservoirRunoff(params)
        initial_state = model.state

        precipitation = [0.0] * 10
        model.simulate(sample_subbasin, precipitation)

        # 存储应该随recession衰减
        assert model.state < initial_state

    def test_storage_equilibrium(self, sample_subbasin):
        """测试存储趋向平衡"""
        params = {
            "recession": 0.9,
            "conversion": 1.0,
            "initial_storage": 0.0
        }
        model = LinearReservoirRunoff(params)

        # 恒定降雨应该导致存储趋向平衡
        precipitation = [10.0] * 100
        runoff = model.simulate(sample_subbasin, precipitation)

        # 后期径流变化应该很小（接近平衡）
        late_runoff_std = np.std(runoff[-10:])
        assert late_runoff_std < 10.0  # 相对稳定

    def test_storage_state_updates(self, default_linear_params, sample_subbasin):
        """测试存储状态更新"""
        model = LinearReservoirRunoff(default_linear_params)

        precipitation = [5.0]
        model.simulate(sample_subbasin, precipitation)

        # state应该等于 0 * 0.9 + 5.0 * 1.0 = 5.0
        assert abs(model.state - 5.0) < 1e-6


# ==================== 边界情况测试 ====================

class TestLinearReservoirEdgeCases:
    """测试边界情况"""

    def test_empty_precipitation(self, default_linear_params, sample_subbasin):
        """测试空降雨序列"""
        model = LinearReservoirRunoff(default_linear_params)
        precipitation = []
        runoff = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 0

    def test_single_timestep(self, default_linear_params, sample_subbasin):
        """测试单时间步"""
        model = LinearReservoirRunoff(default_linear_params)
        precipitation = [10.0]
        runoff = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 1
        assert runoff[0] > 0

    def test_recession_zero(self, sample_subbasin):
        """测试recession=0（无记忆）"""
        params = {
            "recession": 0.0,
            "conversion": 1.0,
            "initial_storage": 50.0
        }
        model = LinearReservoirRunoff(params)

        precipitation = [10.0]
        runoff = model.simulate(sample_subbasin, precipitation)

        # recession=0意味着state完全由当前降雨决定
        # state = 0 * 50.0 + 10.0 * 1.0 = 10.0
        # runoff = (1 - 0) * 10.0 * 100 = 1000.0
        expected_runoff = 10.0 * 100.0
        assert abs(runoff[0] - expected_runoff) < 1e-6

    def test_recession_one(self, sample_subbasin):
        """测试recession=1（完全记忆，无出流）"""
        params = {
            "recession": 1.0,
            "conversion": 1.0,
            "initial_storage": 0.0
        }
        model = LinearReservoirRunoff(params)

        precipitation = [10.0] * 5
        runoff = model.simulate(sample_subbasin, precipitation)

        # recession=1意味着runoff = (1-1) * state = 0
        assert all(r == 0.0 for r in runoff)

    def test_conversion_zero(self, sample_subbasin):
        """测试conversion=0（降雨不转换）"""
        params = {
            "recession": 0.9,
            "conversion": 0.0,
            "initial_storage": 100.0
        }
        model = LinearReservoirRunoff(params)

        precipitation = [10.0] * 5
        runoff = model.simulate(sample_subbasin, precipitation)

        # conversion=0意味着降雨不贡献到存储
        # 存储应该只衰减
        assert all(r > 0 for r in runoff)  # 仍有初始存储产生的径流
        # 径流应该递减
        for i in range(len(runoff) - 1):
            assert runoff[i] >= runoff[i + 1]

    def test_large_initial_storage(self, sample_subbasin):
        """测试大初始存储"""
        params = {
            "recession": 0.95,
            "conversion": 1.0,
            "initial_storage": 1000.0
        }
        model = LinearReservoirRunoff(params)

        precipitation = [5.0] * 10
        runoff = model.simulate(sample_subbasin, precipitation)

        # 大初始存储应该产生大径流
        assert all(r > 0 for r in runoff)
        assert runoff[0] > 1000.0  # 考虑面积


# ==================== 衰退行为测试 ====================

class TestLinearReservoirRecession:
    """测试衰退行为"""

    def test_higher_recession_slower_response(self, sample_subbasin):
        """测试更高的recession导致更慢的响应"""
        params_low = {"recession": 0.5, "conversion": 1.0, "initial_storage": 0.0}
        params_high = {"recession": 0.95, "conversion": 1.0, "initial_storage": 0.0}

        model_low = LinearReservoirRunoff(params_low)
        model_high = LinearReservoirRunoff(params_high)

        precipitation = [10.0] * 20

        runoff_low = model_low.simulate(sample_subbasin, precipitation)
        runoff_high = model_high.simulate(sample_subbasin, precipitation)

        # 低recession应该更快达到峰值
        # 前几个时间步，低recession径流应该更大
        assert runoff_low[2] > runoff_high[2]

    def test_recession_affects_depletion_rate(self, sample_subbasin):
        """测试recession影响衰减速率"""
        params_slow = {"recession": 0.95, "initial_storage": 100.0}
        params_fast = {"recession": 0.7, "initial_storage": 100.0}

        model_slow = LinearReservoirRunoff(params_slow)
        model_fast = LinearReservoirRunoff(params_fast)

        precipitation = [0.0] * 20

        runoff_slow = model_slow.simulate(sample_subbasin, precipitation)
        runoff_fast = model_fast.simulate(sample_subbasin, precipitation)

        # 高recession应该衰减更慢
        assert model_slow.state > model_fast.state

    def test_exponential_decay_without_precipitation(self, sample_subbasin):
        """测试无降雨时的指数衰减"""
        params = {"recession": 0.9, "initial_storage": 100.0}
        model = LinearReservoirRunoff(params)

        precipitation = [0.0] * 10
        runoff = model.simulate(sample_subbasin, precipitation)

        # 径流应该呈指数衰减
        # runoff[i+1] / runoff[i] 应该接近 recession
        for i in range(len(runoff) - 1):
            if runoff[i] > 0:
                ratio = runoff[i + 1] / runoff[i]
                # 考虑到计算精度，比率应该接近recession
                assert 0.8 < ratio < 1.0


# ==================== 转换系数测试 ====================

class TestLinearReservoirConversion:
    """测试转换系数效果"""

    def test_higher_conversion_more_runoff(self, sample_subbasin):
        """测试更高的转换系数产生更多径流"""
        params_low = {"recession": 0.9, "conversion": 0.5}
        params_high = {"recession": 0.9, "conversion": 1.5}

        model_low = LinearReservoirRunoff(params_low)
        model_high = LinearReservoirRunoff(params_high)

        precipitation = [10.0] * 10

        runoff_low = model_low.simulate(sample_subbasin, precipitation)
        runoff_high = model_high.simulate(sample_subbasin, precipitation)

        # 高转换系数应该产生更多径流
        assert sum(runoff_high) > sum(runoff_low)

    def test_conversion_scales_input(self, sample_subbasin):
        """测试转换系数缩放输入"""
        params = {"recession": 0.8, "conversion": 2.0}
        model = LinearReservoirRunoff(params)

        precipitation = [5.0]
        model.simulate(sample_subbasin, precipitation)

        # state应该是 0 * 0.8 + 5.0 * 2.0 = 10.0
        assert abs(model.state - 10.0) < 1e-6


# ==================== 流域面积效果测试 ====================

class TestLinearReservoirAreaEffect:
    """测试流域面积效果"""

    def test_larger_area_larger_runoff(self, default_linear_params):
        """测试更大的流域面积产生更大的径流"""
        class SmallBasin:
            area_km2 = 10.0

        class LargeBasin:
            area_km2 = 1000.0

        model = LinearReservoirRunoff(default_linear_params)
        precipitation = [10.0] * 5

        # 需要两个独立的模型实例，因为state会改变
        model1 = LinearReservoirRunoff(default_linear_params)
        model2 = LinearReservoirRunoff(default_linear_params)

        runoff_small = model1.simulate(SmallBasin(), precipitation)
        runoff_large = model2.simulate(LargeBasin(), precipitation)

        # 径流应该与面积成正比
        ratio = runoff_large[0] / runoff_small[0]
        expected_ratio = 1000.0 / 10.0
        assert abs(ratio - expected_ratio) < 1e-6

    def test_area_linearly_scales_output(self, default_linear_params):
        """测试面积线性缩放输出"""
        class Basin100:
            area_km2 = 100.0

        class Basin200:
            area_km2 = 200.0

        precipitation = [10.0] * 5

        model1 = LinearReservoirRunoff(default_linear_params)
        model2 = LinearReservoirRunoff(default_linear_params)

        runoff_100 = model1.simulate(Basin100(), precipitation)
        runoff_200 = model2.simulate(Basin200(), precipitation)

        # 200km²的径流应该是100km²的2倍
        for i in range(len(precipitation)):
            assert abs(runoff_200[i] / runoff_100[i] - 2.0) < 1e-6


# ==================== 长时间序列测试 ====================

class TestLinearReservoirLongSeries:
    """测试长时间序列"""

    def test_long_constant_precipitation(self, default_linear_params, sample_subbasin):
        """测试长恒定降雨"""
        model = LinearReservoirRunoff(default_linear_params)
        precipitation = [10.0] * 200
        runoff = model.simulate(sample_subbasin, precipitation)

        # 应该趋向稳定状态
        late_mean = np.mean(runoff[-20:])
        late_std = np.std(runoff[-20:])

        assert late_std < 1.0  # 变化很小

    def test_drought_recovery(self, sample_subbasin):
        """测试干旱后恢复"""
        params = {"recession": 0.9, "conversion": 1.0, "initial_storage": 50.0}
        model = LinearReservoirRunoff(params)

        # 先干旱（零降雨）
        drought = [0.0] * 20
        # 然后降雨
        rain = [15.0] * 20

        precipitation = drought + rain
        runoff = model.simulate(sample_subbasin, precipitation)

        # 干旱期径流应该递减
        for i in range(len(drought) - 1):
            assert runoff[i] >= runoff[i + 1]

        # 降雨期径流应该增加
        drought_end = len(drought)
        assert runoff[drought_end] < runoff[drought_end + 10]

    def test_multiple_rain_events(self, default_linear_params, sample_subbasin):
        """测试多次降雨事件"""
        model = LinearReservoirRunoff(default_linear_params)

        # 多次降雨-干旱循环
        precipitation = [20.0, 10.0, 5.0, 0.0, 0.0, 30.0, 15.0, 5.0, 0.0, 0.0]
        runoff = model.simulate(sample_subbasin, precipitation)

        # 每次降雨后应该有径流峰值
        assert len(runoff) == 10
        assert all(r >= 0 for r in runoff)


# ==================== 数值稳定性测试 ====================

class TestLinearReservoirNumericalStability:
    """测试数值稳定性"""

    def test_no_negative_runoff(self, default_linear_params, sample_subbasin):
        """测试径流始终非负"""
        model = LinearReservoirRunoff(default_linear_params)
        precipitation = [float(np.random.rand() * 20) for _ in range(100)]
        runoff = model.simulate(sample_subbasin, precipitation)

        assert all(r >= 0 for r in runoff)

    def test_very_long_series(self, default_linear_params, sample_subbasin):
        """测试非常长的时间序列"""
        model = LinearReservoirRunoff(default_linear_params)
        precipitation = [10.0] * 1000
        runoff = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 1000
        assert all(r >= 0 for r in runoff)
        # 应该数值稳定（不应该有NaN或Inf）
        assert all(np.isfinite(r) for r in runoff)
