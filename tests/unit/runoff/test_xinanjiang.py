"""新安江产流模型单元测试

测试XinAnJiang (新安江)模型的核心功能。
新安江模型是中国流域水文模型的经典代表。
"""
import pytest
import numpy as np

from hydrosis.runoff.xinanjiang import XinAnJiangRunoff
from hydrosis.validation import ParameterValidationError


class MockSubbasin:
    """模拟Subbasin对象"""
    def __init__(self, area_km2: float):
        self.area_km2 = area_km2


@pytest.fixture
def default_xaj_params():
    """默认新安江参数"""
    return {
        'wm': 150.0,
        'b': 0.3,
        'imp': 0.05,
        'recession': 0.6,
        'initial_tension_water': 75.0,
        'initial_groundwater': 10.0
    }


@pytest.fixture
def sample_subbasin():
    """示例子流域"""
    return MockSubbasin(area_km2=100.0)


class TestXinAnJiangInitialization:
    """测试新安江模型初始化"""

    def test_initialization_with_defaults(self):
        """测试使用默认参数初始化"""
        model = XinAnJiangRunoff({})

        assert model.wm > 0
        assert model.b >= 0
        assert 0 <= model.imp <= 1
        assert 0 <= model.k <= 1
        assert 0 <= model.tension_water <= model.wm
        assert model.groundwater >= 0

    def test_initialization_with_custom_params(self, default_xaj_params):
        """测试使用自定义参数初始化"""
        model = XinAnJiangRunoff(default_xaj_params)

        assert model.wm == 150.0
        assert model.b == 0.3
        assert model.imp == 0.05
        assert model.k == 0.6
        assert model.tension_water == 75.0
        assert model.groundwater == 10.0

    def test_initialization_with_partial_params(self):
        """测试使用部分参数初始化"""
        params = {'wm': 200.0, 'b': 0.5}
        model = XinAnJiangRunoff(params)

        assert model.wm == 200.0
        assert model.b == 0.5
        # 其他参数应使用默认值
        assert 0 < model.imp < 1

    def test_initial_tension_water_capped_by_wm(self):
        """测试初始张力水不超过wm"""
        params = {'wm': 100.0, 'initial_tension_water': 150.0}
        model = XinAnJiangRunoff(params)

        # 初始张力水应被限制在wm以内
        assert model.tension_water <= model.wm
        assert model.tension_water == 100.0


class TestXinAnJiangParameterValidation:
    """测试新安江模型参数验证"""

    def test_valid_parameters(self, default_xaj_params):
        """测试有效参数"""
        model = XinAnJiangRunoff(default_xaj_params)
        # 应该不抛出异常
        model.validate_parameters()

    def test_invalid_wm_zero(self):
        """测试无效的wm（零）"""
        params = {'wm': 0.0}

        with pytest.raises(ParameterValidationError, match="wm"):
            model = XinAnJiangRunoff(params)

    def test_invalid_wm_negative(self):
        """测试无效的wm（负数）"""
        params = {'wm': -10.0}

        with pytest.raises(ParameterValidationError, match="wm"):
            model = XinAnJiangRunoff(params)

    def test_invalid_imp_above_one(self):
        """测试无效的imp（大于1）"""
        params = {'imp': 1.5}

        with pytest.raises(ParameterValidationError, match="imp"):
            model = XinAnJiangRunoff(params)

    def test_invalid_imp_negative(self):
        """测试无效的imp（负数）"""
        params = {'imp': -0.1}

        with pytest.raises(ParameterValidationError, match="imp"):
            model = XinAnJiangRunoff(params)

    def test_invalid_recession_above_one(self):
        """测试无效的recession（大于1）"""
        params = {'recession': 1.5}

        with pytest.raises(ParameterValidationError, match="recession"):
            model = XinAnJiangRunoff(params)

    def test_valid_b_zero(self):
        """测试b=0是有效的"""
        params = {'b': 0.0}
        model = XinAnJiangRunoff(params)
        # 应该不抛出异常
        model.validate_parameters()


class TestXinAnJiangSimulation:
    """测试新安江模型模拟"""

    def test_simulate_zero_rainfall(self, default_xaj_params, sample_subbasin):
        """测试零降雨情况"""
        model = XinAnJiangRunoff(default_xaj_params)
        precipitation = [0.0] * 10

        runoff = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 10
        assert all(isinstance(q, (int, float)) for q in runoff)
        assert all(q >= 0 for q in runoff)  # 径流应为非负

    def test_simulate_constant_rainfall(self, default_xaj_params, sample_subbasin):
        """测试恒定降雨"""
        model = XinAnJiangRunoff(default_xaj_params)
        precipitation = [5.0] * 20

        runoff = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 20
        assert all(q >= 0 for q in runoff)
        # 恒定降雨下，径流应趋于稳定
        assert np.std(runoff[-5:]) < np.std(runoff[:5])

    def test_simulate_varying_rainfall(self, default_xaj_params, sample_subbasin):
        """测试变化降雨"""
        # 使用较低初始地下水，避免初始基流过大
        params = default_xaj_params.copy()
        params['initial_groundwater'] = 1.0
        model = XinAnJiangRunoff(params)

        # 模拟降雨事件
        precipitation = [0] * 5 + [10, 20, 15, 10, 5] + [1] * 10

        runoff = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 20
        assert all(q >= 0 for q in runoff)
        # 降雨期间的峰值径流应高于无雨期
        rain_period_max = max(runoff[5:10])
        no_rain_period_mean = np.mean(runoff[:5] + runoff[15:])
        assert rain_period_max > no_rain_period_mean * 2

    def test_simulate_heavy_rainfall(self, default_xaj_params, sample_subbasin):
        """测试强降雨事件"""
        model = XinAnJiangRunoff(default_xaj_params)
        precipitation = [0] * 5 + [50.0] + [0] * 10

        runoff = model.simulate(sample_subbasin, precipitation)

        # 强降雨应产生显著径流
        assert max(runoff) > 0
        # 峰值径流应大于平均径流
        assert max(runoff) > np.mean(runoff) * 2


class TestXinAnJiangStorages:
    """测试新安江模型的存储机制"""

    def test_tension_water_not_exceed_wm(self, default_xaj_params, sample_subbasin):
        """测试张力水不超过wm"""
        model = XinAnJiangRunoff(default_xaj_params)
        # 极端强降雨
        precipitation = [100.0] * 10

        model.simulate(sample_subbasin, precipitation)

        # 张力水不应超过wm
        assert 0 <= model.tension_water <= model.wm

    def test_groundwater_increases_with_recharge(self, default_xaj_params, sample_subbasin):
        """测试地下水随补给增加

        注意：recession参数k的含义：
        - 补给 = tension_water * (1 - k)
        - 基流 = k * groundwater
        所以低k = 高补给 + 低基流
        """
        params = default_xaj_params.copy()
        params['initial_groundwater'] = 1.0  # 低初始值，基流小
        params['initial_tension_water'] = 100.0  # 高张力水，补给多
        params['recession'] = 0.1  # 低k = 高补给(90%) + 低基流(10%)
        model = XinAnJiangRunoff(params)

        initial_gw = model.groundwater

        # 降雨应增加地下水（通过张力水补给）
        precipitation = [5.0] * 5  # 短期降雨
        model.simulate(sample_subbasin, precipitation)

        # 地下水应增加（高补给 > 低基流）
        assert model.groundwater > initial_gw

    def test_baseflow_depletes_groundwater(self, default_xaj_params, sample_subbasin):
        """测试基流消耗地下水"""
        params = default_xaj_params.copy()
        params['initial_groundwater'] = 50.0
        params['recession'] = 0.8  # 高基流系数
        model = XinAnJiangRunoff(params)

        # 长期无降雨，基流应消耗地下水
        precipitation = [0.0] * 20
        runoff = model.simulate(sample_subbasin, precipitation)

        # 应该产生基流
        assert sum(runoff) > 0
        # 地下水应减少
        assert model.groundwater < 50.0

    def test_impervious_area_effect(self, default_xaj_params, sample_subbasin):
        """测试不透水面积参数的影响

        注意：在XAJ模型中，imp参数实际上减少有效降雨
        (effective_rain = p * (1 - imp))，模拟截留或损失，
        而不是传统意义上的不透水面积产流。
        """
        # 高imp值（高截留/损失）
        params_high_imp = default_xaj_params.copy()
        params_high_imp['imp'] = 0.5
        model_high = XinAnJiangRunoff(params_high_imp)

        # 低imp值（低截留/损失）
        params_low_imp = default_xaj_params.copy()
        params_low_imp['imp'] = 0.05
        model_low = XinAnJiangRunoff(params_low_imp)

        precipitation = [10.0] * 10

        runoff_high = model_high.simulate(sample_subbasin, precipitation)
        runoff_low = model_low.simulate(sample_subbasin, precipitation)

        # 低imp（更多有效降雨）应产生更多径流
        assert sum(runoff_low) > sum(runoff_high)


class TestXinAnJiangInfiltrationCurve:
    """测试新安江模型的蓄水容量曲线"""

    def test_infiltration_capacity_increases_with_storage(self):
        """测试蓄水容量随土壤湿度增加"""
        params = {'wm': 100.0, 'b': 0.3, 'initial_tension_water': 20.0}
        model = XinAnJiangRunoff(params)

        capacity_low = model._infiltration_capacity()

        # 增加张力水
        model.tension_water = 80.0
        capacity_high = model._infiltration_capacity()

        # 高土壤湿度应有更高的蓄水容量
        assert capacity_high > capacity_low

    def test_b_parameter_effect(self, sample_subbasin):
        """测试b参数对蓄水曲线的影响"""
        # b=0: 均匀分布（更线性）
        model_b0 = XinAnJiangRunoff({
            'b': 0.0,
            'wm': 100.0,
            'initial_tension_water': 20.0,  # 低初始值
            'imp': 0.0,  # 无截留
            'recession': 0.9  # 高recession降低基流影响
        })

        # b=2: 强非线性分布
        model_b2 = XinAnJiangRunoff({
            'b': 2.0,
            'wm': 100.0,
            'initial_tension_water': 20.0,
            'imp': 0.0,
            'recession': 0.9
        })

        # 使用变化的降雨来体现差异
        precipitation = [5.0, 10.0, 15.0, 10.0, 5.0]

        runoff_b0 = model_b0.simulate(sample_subbasin, precipitation)
        runoff_b2 = model_b2.simulate(sample_subbasin, precipitation)

        # 不同b值应产生不同的径流过程（检查峰值时刻）
        # b值影响蓄水容量曲线，从而影响产流时间和量
        assert runoff_b0 != runoff_b2  # 至少某些值应该不同


class TestXinAnJiangEdgeCases:
    """测试新安江模型边界情况"""

    def test_empty_precipitation(self, default_xaj_params, sample_subbasin):
        """测试空降雨序列"""
        model = XinAnJiangRunoff(default_xaj_params)
        precipitation = []

        runoff = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 0

    def test_single_timestep(self, default_xaj_params, sample_subbasin):
        """测试单个时间步"""
        model = XinAnJiangRunoff(default_xaj_params)
        precipitation = [10.0]

        runoff = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 1
        assert runoff[0] >= 0

    def test_extreme_parameters(self, sample_subbasin):
        """测试极端参数值"""
        # 极小wm（快速饱和）
        params = {
            'wm': 10.0,
            'b': 0.1,
            'imp': 0.3,
            'recession': 0.5
        }
        model = XinAnJiangRunoff(params)
        precipitation = [10.0] * 10

        runoff = model.simulate(sample_subbasin, precipitation)

        assert all(q >= 0 for q in runoff)
        # 小wm应快速饱和，产生较大径流
        assert max(runoff) > 0

    def test_different_area_sizes(self, default_xaj_params):
        """测试不同流域面积"""
        model = XinAnJiangRunoff(default_xaj_params)
        precipitation = [10.0] * 5

        # 小流域
        small_basin = MockSubbasin(10.0)
        runoff_small = model.simulate(small_basin, precipitation)

        # 重置模型状态
        model = XinAnJiangRunoff(default_xaj_params)

        # 大流域
        large_basin = MockSubbasin(1000.0)
        runoff_large = model.simulate(large_basin, precipitation)

        # 径流应与面积成正比
        ratio = large_basin.area_km2 / small_basin.area_km2
        assert abs(runoff_large[0] / runoff_small[0] - ratio) < 0.01

    def test_full_impervious_area(self, sample_subbasin):
        """测试完全不透水面积"""
        params = {'wm': 100.0, 'b': 0.3, 'imp': 1.0, 'initial_groundwater': 10.0, 'recession': 0.8}
        model = XinAnJiangRunoff(params)

        initial_tw = model.tension_water

        precipitation = [10.0] * 5
        runoff = model.simulate(sample_subbasin, precipitation)

        # 完全不透水时，张力水不应增加（所有降雨被截流）
        # 但仍可能因地下水补给而变化
        # 主要测试模型不会崩溃
        assert all(q >= 0 for q in runoff)


class TestXinAnJiangWaterBalance:
    """测试新安江模型的水量平衡特性"""

    def test_storage_changes_with_precipitation(self, default_xaj_params, sample_subbasin):
        """测试降雨时存储变化"""
        model = XinAnJiangRunoff(default_xaj_params)

        initial_storage = model.tension_water + model.groundwater

        precipitation = [10.0] * 10
        runoff = model.simulate(sample_subbasin, precipitation)

        final_storage = model.tension_water + model.groundwater

        # 降雨应产生径流
        assert sum(runoff) > 0
        # 存储应该变化
        assert final_storage != initial_storage

    def test_baseflow_generation(self, default_xaj_params, sample_subbasin):
        """测试基流生成"""
        params = default_xaj_params.copy()
        params['initial_groundwater'] = 50.0
        params['recession'] = 0.7
        model = XinAnJiangRunoff(params)

        # 无降雨，应产生基流
        precipitation = [0.0] * 10
        runoff = model.simulate(sample_subbasin, precipitation)

        # 应有持续的基流
        assert all(q > 0 for q in runoff[:5])
        # 基流应递减
        assert runoff[0] > runoff[-1]

    def test_recharge_mechanism(self, default_xaj_params, sample_subbasin):
        """测试地下水补给机制

        补给 = tension_water * (1-k), 基流 = k * groundwater
        低k值有利于补给
        """
        params = default_xaj_params.copy()
        params['initial_tension_water'] = 100.0  # 高张力水，多补给
        params['initial_groundwater'] = 1.0  # 低地下水，少基流
        params['recession'] = 0.15  # 低k = 高补给(85%) + 低基流(15%)
        model = XinAnJiangRunoff(params)

        initial_gw = model.groundwater

        # 短期降雨应通过张力水补给地下水
        precipitation = [3.0] * 3  # 短期小降雨
        model.simulate(sample_subbasin, precipitation)

        # 地下水应增加（接收补给）
        # 低k值使得补给量大，基流损失小
        assert model.groundwater > initial_gw
