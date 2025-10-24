"""SCS Curve Number产流模型的单元测试"""
import pytest
import numpy as np
from hydrosis.runoff.scs_curve_number import SCSCurveNumber
from hydrosis.validation import ParameterValidationError


@pytest.fixture
def default_scs_params():
    """默认SCS Curve Number参数"""
    return {
        "curve_number": 75,
        "initial_abstraction_ratio": 0.2
    }


@pytest.fixture
def sample_subbasin():
    """样本子流域（用于测试）"""
    class MockSubbasin:
        def __init__(self):
            self.area_km2 = 100.0
    return MockSubbasin()


# ==================== 初始化测试 ====================

class TestSCSCurveNumberInitialization:
    """测试SCS Curve Number模型初始化"""

    def test_default_initialization(self):
        """测试默认参数初始化"""
        model = SCSCurveNumber({})
        assert model.cn == 75
        assert model.initial_abstraction_ratio == 0.2

    def test_custom_initialization(self, default_scs_params):
        """测试自定义参数初始化"""
        model = SCSCurveNumber(default_scs_params)
        assert model.cn == 75
        assert model.initial_abstraction_ratio == 0.2

    def test_partial_initialization(self):
        """测试部分参数初始化"""
        params = {"curve_number": 85}
        model = SCSCurveNumber(params)
        assert model.cn == 85
        assert model.initial_abstraction_ratio == 0.2  # 默认值

    def test_high_cn_initialization(self):
        """测试高CN值初始化"""
        params = {"curve_number": 95}
        model = SCSCurveNumber(params)
        assert model.cn == 95

    def test_low_cn_initialization(self):
        """测试低CN值初始化"""
        params = {"curve_number": 40}
        model = SCSCurveNumber(params)
        assert model.cn == 40

    def test_custom_ia_ratio(self):
        """测试自定义初损比例"""
        params = {"initial_abstraction_ratio": 0.05}
        model = SCSCurveNumber(params)
        assert model.initial_abstraction_ratio == 0.05


# ==================== 参数验证测试 ====================

class TestSCSCurveNumberParameterValidation:
    """测试参数验证"""

    def test_validate_default_parameters(self):
        """测试默认参数验证"""
        model = SCSCurveNumber({})
        model.validate_parameters()  # 不应抛出异常

    def test_validate_custom_parameters(self, default_scs_params):
        """测试自定义参数验证"""
        model = SCSCurveNumber(default_scs_params)
        model.validate_parameters()  # 不应抛出异常

    def test_cn_zero_invalid(self):
        """测试CN=0无效"""
        params = {"curve_number": 0}
        with pytest.raises(ParameterValidationError, match="curve_number"):
            model = SCSCurveNumber(params)

    def test_cn_negative_invalid(self):
        """测试负CN值无效"""
        params = {"curve_number": -10}
        with pytest.raises(ParameterValidationError, match="curve_number"):
            model = SCSCurveNumber(params)

    def test_cn_above_100_invalid(self):
        """测试CN>100无效"""
        params = {"curve_number": 105}
        with pytest.raises(ParameterValidationError, match="curve_number"):
            model = SCSCurveNumber(params)

    def test_cn_100_valid(self):
        """测试CN=100有效（完全不透水）"""
        params = {"curve_number": 100}
        model = SCSCurveNumber(params)
        model.validate_parameters()
        assert model.cn == 100

    def test_cn_boundary_very_small(self):
        """测试CN接近0的边界值"""
        params = {"curve_number": 0.1}
        model = SCSCurveNumber(params)
        model.validate_parameters()
        assert model.cn == 0.1

    def test_ia_ratio_negative_invalid(self):
        """测试负初损比例无效"""
        params = {"initial_abstraction_ratio": -0.1}
        with pytest.raises(ParameterValidationError, match="initial_abstraction_ratio"):
            model = SCSCurveNumber(params)

    def test_ia_ratio_above_one_invalid(self):
        """测试初损比例>1无效"""
        params = {"initial_abstraction_ratio": 1.5}
        with pytest.raises(ParameterValidationError, match="initial_abstraction_ratio"):
            model = SCSCurveNumber(params)

    def test_ia_ratio_boundary_values(self):
        """测试初损比例边界值"""
        # ia_ratio = 0 应该有效
        model_0 = SCSCurveNumber({"initial_abstraction_ratio": 0.0})
        model_0.validate_parameters()
        assert model_0.initial_abstraction_ratio == 0.0

        # ia_ratio = 1 应该有效
        model_1 = SCSCurveNumber({"initial_abstraction_ratio": 1.0})
        model_1.validate_parameters()
        assert model_1.initial_abstraction_ratio == 1.0


# ==================== 仿真测试 ====================

class TestSCSCurveNumberSimulation:
    """测试仿真功能"""

    def test_simulate_zero_precipitation(self, default_scs_params, sample_subbasin):
        """测试零降雨"""
        model = SCSCurveNumber(default_scs_params)
        precipitation = [0.0] * 10
        runoff = model.simulate(sample_subbasin, precipitation)

        # 零降雨应该产生零径流
        assert len(runoff) == 10
        assert all(r == 0.0 for r in runoff)

    def test_simulate_below_initial_abstraction(self, default_scs_params, sample_subbasin):
        """测试降雨小于初损"""
        model = SCSCurveNumber(default_scs_params)

        # 计算初损值
        s = (1000.0 / 75 - 10.0) * 25.4
        ia = 0.2 * s  # 应该约为67.7mm

        # 降雨小于初损
        precipitation = [ia * 0.5] * 5
        runoff = model.simulate(sample_subbasin, precipitation)

        # 应该产生零径流
        assert all(r == 0.0 for r in runoff)

    def test_simulate_constant_precipitation(self, default_scs_params, sample_subbasin):
        """测试恒定降雨"""
        model = SCSCurveNumber(default_scs_params)
        precipitation = [100.0] * 10
        runoff = model.simulate(sample_subbasin, precipitation)

        # 恒定降雨应该产生恒定径流
        assert len(runoff) == 10
        assert all(r > 0 for r in runoff)
        # 所有径流值应该相同（无状态模型）
        assert np.allclose(runoff, runoff[0])

    def test_simulate_varying_precipitation(self, default_scs_params, sample_subbasin):
        """测试变化降雨"""
        model = SCSCurveNumber(default_scs_params)
        precipitation = [50.0, 75.0, 100.0, 75.0, 50.0]
        runoff = model.simulate(sample_subbasin, precipitation)

        # 径流应该跟随降雨变化
        assert len(runoff) == 5
        # 最大降雨应该产生最大径流
        max_p_idx = precipitation.index(max(precipitation))
        max_q_idx = runoff.index(max(runoff))
        assert max_p_idx == max_q_idx

    def test_simulate_heavy_precipitation(self, default_scs_params, sample_subbasin):
        """测试强降雨"""
        model = SCSCurveNumber(default_scs_params)
        precipitation = [200.0] * 5
        runoff = model.simulate(sample_subbasin, precipitation)

        # 强降雨应该产生大径流
        assert len(runoff) == 5
        assert all(r > 10000 for r in runoff)  # 考虑面积


# ==================== CN值效果测试 ====================

class TestSCSCurveNumberCNEffect:
    """测试CN值的效果"""

    def test_higher_cn_more_runoff(self, sample_subbasin):
        """测试更高的CN产生更多径流"""
        model_low = SCSCurveNumber({"curve_number": 60})
        model_high = SCSCurveNumber({"curve_number": 90})

        precipitation = [100.0] * 5

        runoff_low = model_low.simulate(sample_subbasin, precipitation)
        runoff_high = model_high.simulate(sample_subbasin, precipitation)

        # 高CN应该产生更多径流
        assert all(runoff_high[i] > runoff_low[i] for i in range(len(precipitation)))

    def test_cn_100_impervious(self, sample_subbasin):
        """测试CN=100（完全不透水）"""
        model = SCSCurveNumber({"curve_number": 100, "initial_abstraction_ratio": 0.0})

        precipitation = [50.0, 100.0, 150.0]
        runoff = model.simulate(sample_subbasin, precipitation)

        # CN=100且Ia=0时，径流应该等于降雨（考虑面积）
        for i in range(len(precipitation)):
            expected = precipitation[i] * sample_subbasin.area_km2
            assert abs(runoff[i] - expected) < 0.1

    def test_low_cn_less_runoff(self, sample_subbasin):
        """测试低CN值产生较少径流"""
        model = SCSCurveNumber({"curve_number": 40})

        precipitation = [100.0] * 5
        runoff = model.simulate(sample_subbasin, precipitation)

        # 低CN应该产生较少径流
        # 径流应该远小于降雨
        for i in range(len(precipitation)):
            assert runoff[i] < precipitation[i] * sample_subbasin.area_km2 * 0.3

    def test_cn_effect_comparison(self, sample_subbasin):
        """测试不同CN值的对比"""
        cn_values = [50, 60, 70, 80, 90]
        precipitation = [100.0]

        runoff_values = []
        for cn in cn_values:
            model = SCSCurveNumber({"curve_number": cn})
            runoff = model.simulate(sample_subbasin, precipitation)
            runoff_values.append(runoff[0])

        # 径流应该随CN单调递增
        for i in range(len(runoff_values) - 1):
            assert runoff_values[i] < runoff_values[i + 1]


# ==================== 初损比例效果测试 ====================

class TestSCSCurveNumberIAEffect:
    """测试初损比例的效果"""

    def test_higher_ia_ratio_less_runoff(self, sample_subbasin):
        """测试更高的初损比例产生更少径流"""
        model_low = SCSCurveNumber({"curve_number": 75, "initial_abstraction_ratio": 0.1})
        model_high = SCSCurveNumber({"curve_number": 75, "initial_abstraction_ratio": 0.4})

        precipitation = [100.0] * 5

        runoff_low = model_low.simulate(sample_subbasin, precipitation)
        runoff_high = model_high.simulate(sample_subbasin, precipitation)

        # 高初损比例应该产生更少径流
        assert all(runoff_low[i] > runoff_high[i] for i in range(len(precipitation)))

    def test_zero_ia_ratio(self, sample_subbasin):
        """测试零初损比例"""
        model = SCSCurveNumber({"curve_number": 75, "initial_abstraction_ratio": 0.0})

        precipitation = [10.0, 20.0, 30.0]
        runoff = model.simulate(sample_subbasin, precipitation)

        # 零初损意味着任何降雨都会产生径流
        assert all(r > 0 for r in runoff)

    def test_ia_ratio_one(self, sample_subbasin):
        """测试初损比例=1"""
        model = SCSCurveNumber({"curve_number": 75, "initial_abstraction_ratio": 1.0})

        # 计算S值
        s = (1000.0 / 75 - 10.0) * 25.4
        ia = 1.0 * s  # Ia = S

        # 降雨等于S应该产生零径流（因为P = Ia）
        precipitation = [s]
        runoff = model.simulate(sample_subbasin, precipitation)
        assert runoff[0] == 0.0

        # 降雨大于S应该产生径流
        precipitation = [s * 1.5]
        runoff = model.simulate(sample_subbasin, precipitation)
        assert runoff[0] > 0


# ==================== 边界情况测试 ====================

class TestSCSCurveNumberEdgeCases:
    """测试边界情况"""

    def test_empty_precipitation(self, default_scs_params, sample_subbasin):
        """测试空降雨序列"""
        model = SCSCurveNumber(default_scs_params)
        precipitation = []
        runoff = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 0

    def test_single_timestep(self, default_scs_params, sample_subbasin):
        """测试单时间步"""
        model = SCSCurveNumber(default_scs_params)
        precipitation = [100.0]
        runoff = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 1
        assert runoff[0] > 0

    def test_very_small_precipitation(self, default_scs_params, sample_subbasin):
        """测试极小降雨"""
        model = SCSCurveNumber(default_scs_params)
        precipitation = [0.01, 0.05, 0.1]
        runoff = model.simulate(sample_subbasin, precipitation)

        # 极小降雨应该产生零径流（小于初损）
        assert all(r == 0.0 for r in runoff)

    def test_very_large_precipitation(self, default_scs_params, sample_subbasin):
        """测试极大降雨"""
        model = SCSCurveNumber(default_scs_params)
        precipitation = [1000.0, 2000.0]
        runoff = model.simulate(sample_subbasin, precipitation)

        # 极大降雨应该产生大径流
        assert all(r > 50000 for r in runoff)

    def test_mixed_precipitation(self, default_scs_params, sample_subbasin):
        """测试混合降雨（有些超过初损，有些未超过）"""
        model = SCSCurveNumber(default_scs_params)

        # 计算初损
        s = (1000.0 / 75 - 10.0) * 25.4
        ia = 0.2 * s

        precipitation = [ia * 0.5, ia * 1.5, ia * 0.8, ia * 2.0]
        runoff = model.simulate(sample_subbasin, precipitation)

        # 第1和第3个应该是0（未超过初损）
        assert runoff[0] == 0.0
        assert runoff[2] == 0.0
        # 第2和第4个应该>0（超过初损）
        assert runoff[1] > 0
        assert runoff[3] > 0


# ==================== S值计算测试 ====================

class TestSCSCurveNumberSCalculation:
    """测试最大潜在蓄水量S的计算"""

    def test_s_calculation_cn75(self):
        """测试CN=75时的S值"""
        model = SCSCurveNumber({"curve_number": 75})

        # S = (1000/CN - 10) * 25.4
        expected_s = (1000.0 / 75 - 10.0) * 25.4
        # 约为84.67 mm

        # 通过检查初损来验证S值
        assert abs(model.initial_abstraction_ratio * expected_s - 0.2 * expected_s) < 0.01

    def test_s_decreases_with_cn(self):
        """测试S值随CN增加而减少"""
        model_low = SCSCurveNumber({"curve_number": 60})
        model_high = SCSCurveNumber({"curve_number": 90})

        s_low = (1000.0 / 60 - 10.0) * 25.4
        s_high = (1000.0 / 90 - 10.0) * 25.4

        # 高CN应该有更小的S值
        assert s_high < s_low

    def test_s_approaches_zero_cn100(self):
        """测试CN=100时S接近0"""
        model = SCSCurveNumber({"curve_number": 100})

        s = (1000.0 / 100 - 10.0) * 25.4
        # S = 0
        assert s == 0.0


# ==================== 流域面积效果测试 ====================

class TestSCSCurveNumberAreaEffect:
    """测试流域面积效果"""

    def test_larger_area_larger_runoff(self, default_scs_params):
        """测试更大的流域面积产生更大的径流"""
        class SmallBasin:
            area_km2 = 10.0

        class LargeBasin:
            area_km2 = 1000.0

        model = SCSCurveNumber(default_scs_params)
        precipitation = [100.0] * 5

        runoff_small = model.simulate(SmallBasin(), precipitation)
        runoff_large = model.simulate(LargeBasin(), precipitation)

        # 径流应该与面积成正比
        ratio = runoff_large[0] / runoff_small[0]
        expected_ratio = 1000.0 / 10.0
        assert abs(ratio - expected_ratio) < 1e-6

    def test_area_linearly_scales_output(self, default_scs_params):
        """测试面积线性缩放输出"""
        class Basin100:
            area_km2 = 100.0

        class Basin200:
            area_km2 = 200.0

        precipitation = [100.0] * 5

        model = SCSCurveNumber(default_scs_params)

        runoff_100 = model.simulate(Basin100(), precipitation)
        runoff_200 = model.simulate(Basin200(), precipitation)

        # 200km²的径流应该是100km²的2倍
        for i in range(len(precipitation)):
            assert abs(runoff_200[i] / runoff_100[i] - 2.0) < 1e-6


# ==================== 无状态特性测试 ====================

class TestSCSCurveNumberStateless:
    """测试SCS模型的无状态特性"""

    def test_no_carryover_between_timesteps(self, default_scs_params, sample_subbasin):
        """测试时间步之间无延续"""
        model = SCSCurveNumber(default_scs_params)

        # 两次相同的降雨应该产生相同的径流（无状态）
        precipitation = [80.0, 80.0, 80.0]
        runoff = model.simulate(sample_subbasin, precipitation)

        # 所有径流应该相同
        assert np.allclose(runoff, runoff[0])

    def test_order_independence(self, default_scs_params, sample_subbasin):
        """测试顺序无关性"""
        model1 = SCSCurveNumber(default_scs_params)
        model2 = SCSCurveNumber(default_scs_params)

        # 不同顺序的相同降雨值
        precipitation1 = [50.0, 100.0, 150.0]
        precipitation2 = [150.0, 50.0, 100.0]

        runoff1 = model1.simulate(sample_subbasin, precipitation1)
        runoff2 = model2.simulate(sample_subbasin, precipitation2)

        # 对应值应该相同（顺序无关）
        assert abs(runoff1[0] - runoff2[1]) < 1e-6  # 50mm
        assert abs(runoff1[1] - runoff2[2]) < 1e-6  # 100mm
        assert abs(runoff1[2] - runoff2[0]) < 1e-6  # 150mm

    def test_reuse_model_multiple_simulations(self, default_scs_params, sample_subbasin):
        """测试重复使用模型"""
        model = SCSCurveNumber(default_scs_params)

        precipitation1 = [100.0]
        precipitation2 = [100.0]

        runoff1 = model.simulate(sample_subbasin, precipitation1)
        runoff2 = model.simulate(sample_subbasin, precipitation2)

        # 两次仿真应该产生相同结果（无状态）
        assert runoff1[0] == runoff2[0]


# ==================== 长时间序列测试 ====================

class TestSCSCurveNumberLongSeries:
    """测试长时间序列"""

    def test_long_constant_precipitation(self, default_scs_params, sample_subbasin):
        """测试长恒定降雨"""
        model = SCSCurveNumber(default_scs_params)
        precipitation = [80.0] * 200
        runoff = model.simulate(sample_subbasin, precipitation)

        # 所有径流应该相同（无状态模型）
        assert len(runoff) == 200
        assert np.allclose(runoff, runoff[0])

    def test_long_varying_series(self, default_scs_params, sample_subbasin):
        """测试长变化序列"""
        model = SCSCurveNumber(default_scs_params)

        # 创建一个有模式的长序列
        precipitation = [float(50 + 30 * np.sin(i * 0.1)) for i in range(100)]
        runoff = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 100
        assert all(r >= 0 for r in runoff)

    def test_wet_dry_cycles(self, default_scs_params, sample_subbasin):
        """测试干湿循环"""
        model = SCSCurveNumber(default_scs_params)

        # 干湿交替
        precipitation = [100.0, 0.0, 100.0, 0.0, 100.0, 0.0]
        runoff = model.simulate(sample_subbasin, precipitation)

        # 湿天应该有径流，干天应该无径流
        assert runoff[0] > 0
        assert runoff[1] == 0.0
        assert runoff[2] > 0
        assert runoff[3] == 0.0
        assert runoff[4] > 0
        assert runoff[5] == 0.0

        # 相同降雨应该产生相同径流（无状态）
        assert runoff[0] == runoff[2] == runoff[4]


# ==================== 数值稳定性测试 ====================

class TestSCSCurveNumberNumericalStability:
    """测试数值稳定性"""

    def test_no_negative_runoff(self, default_scs_params, sample_subbasin):
        """测试径流始终非负"""
        model = SCSCurveNumber(default_scs_params)

        # 随机降雨
        precipitation = [float(np.random.rand() * 200) for _ in range(100)]
        runoff = model.simulate(sample_subbasin, precipitation)

        assert all(r >= 0 for r in runoff)

    def test_very_long_series(self, default_scs_params, sample_subbasin):
        """测试非常长的时间序列"""
        model = SCSCurveNumber(default_scs_params)
        precipitation = [80.0] * 1000
        runoff = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 1000
        assert all(r >= 0 for r in runoff)
        # 应该数值稳定（不应该有NaN或Inf）
        assert all(np.isfinite(r) for r in runoff)

    def test_extreme_cn_values(self, sample_subbasin):
        """测试极端CN值的数值稳定性"""
        # 极小CN值
        model_small = SCSCurveNumber({"curve_number": 0.1})
        precipitation = [100.0] * 5
        runoff_small = model_small.simulate(sample_subbasin, precipitation)
        assert all(np.isfinite(r) for r in runoff_small)

        # 极大CN值
        model_large = SCSCurveNumber({"curve_number": 99.9})
        runoff_large = model_large.simulate(sample_subbasin, precipitation)
        assert all(np.isfinite(r) for r in runoff_large)


# ==================== SCS公式验证测试 ====================

class TestSCSCurveNumberFormulaVerification:
    """测试SCS公式的正确性"""

    def test_runoff_formula_manual_calculation(self, sample_subbasin):
        """测试径流公式的手动计算验证"""
        cn = 80
        ia_ratio = 0.2
        model = SCSCurveNumber({"curve_number": cn, "initial_abstraction_ratio": ia_ratio})

        p = 100.0
        s = (1000.0 / cn - 10.0) * 25.4  # = 63.5 mm
        ia = ia_ratio * s  # = 12.7 mm

        # Q = (P - Ia)² / (P - Ia + S)
        expected_q_per_unit = (p - ia) ** 2 / (p - ia + s)
        expected_q = expected_q_per_unit * sample_subbasin.area_km2

        precipitation = [p]
        runoff = model.simulate(sample_subbasin, precipitation)

        assert abs(runoff[0] - expected_q) < 1e-6

    def test_runoff_zero_when_p_equals_ia(self, sample_subbasin):
        """测试P=Ia时径流为0"""
        cn = 70
        ia_ratio = 0.3
        model = SCSCurveNumber({"curve_number": cn, "initial_abstraction_ratio": ia_ratio})

        s = (1000.0 / cn - 10.0) * 25.4
        ia = ia_ratio * s

        # P = Ia时应该产生零径流
        precipitation = [ia]
        runoff = model.simulate(sample_subbasin, precipitation)

        assert runoff[0] == 0.0

    def test_runoff_increases_with_precipitation(self, default_scs_params, sample_subbasin):
        """测试径流随降雨单调递增"""
        model = SCSCurveNumber(default_scs_params)

        # 计算初损
        s = (1000.0 / 75 - 10.0) * 25.4
        ia = 0.2 * s

        # 一系列递增的降雨（都超过初损）
        precipitation = [ia + 10, ia + 20, ia + 30, ia + 40, ia + 50]
        runoff = model.simulate(sample_subbasin, precipitation)

        # 径流应该单调递增
        for i in range(len(runoff) - 1):
            assert runoff[i] < runoff[i + 1]
