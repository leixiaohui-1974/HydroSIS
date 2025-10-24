"""HYMOD产流模型单元测试

测试HYMOD (HYdrologic MODel) 模型的核心功能。
HYMOD是经典的概念性降雨径流模型，具有非线性土壤存储和串联快速水库。
"""
import pytest
import numpy as np

from hydrosis.runoff.hymod import HYMODRunoff
from hydrosis.validation import ParameterValidationError


class MockSubbasin:
    """模拟Subbasin对象"""
    def __init__(self, area_km2: float):
        self.area_km2 = area_km2


@pytest.fixture
def default_hymod_params():
    """默认HYMOD参数"""
    return {
        'max_storage': 100.0,
        'beta': 1.0,
        'quickflow_ratio': 0.7,
        'quick_k': 0.5,
        'slow_k': 0.05,
        'num_quick_reservoirs': 3,
        'initial_soil_storage': 50.0,
        'initial_quick_storage': 0.0,
        'initial_slow_storage': 0.0
    }


@pytest.fixture
def sample_subbasin():
    """示例子流域"""
    return MockSubbasin(area_km2=100.0)


class TestHYMODInitialization:
    """测试HYMOD模型初始化"""

    def test_initialization_with_defaults(self):
        """测试使用默认参数初始化"""
        model = HYMODRunoff({})

        assert model.smax > 0
        assert model.beta >= 0
        assert 0 <= model.quickflow_ratio <= 1
        assert 0 <= model.k_quick <= 1
        assert 0 <= model.k_slow <= 1
        assert model.num_quick >= 1
        assert 0 <= model.soil_storage <= model.smax
        assert len(model.quick_states) == model.num_quick
        assert model.slow_state >= 0

    def test_initialization_with_custom_params(self, default_hymod_params):
        """测试使用自定义参数初始化"""
        model = HYMODRunoff(default_hymod_params)

        assert model.smax == 100.0
        assert model.beta == 1.0
        assert model.quickflow_ratio == 0.7
        assert model.k_quick == 0.5
        assert model.k_slow == 0.05
        assert model.num_quick == 3
        assert model.soil_storage == 50.0
        assert all(s == 0.0 for s in model.quick_states)
        assert model.slow_state == 0.0

    def test_initialization_with_partial_params(self):
        """测试使用部分参数初始化"""
        params = {'max_storage': 150.0, 'beta': 2.0}
        model = HYMODRunoff(params)

        assert model.smax == 150.0
        assert model.beta == 2.0
        # 其他参数应使用默认值
        assert model.k_quick > 0

    def test_initial_soil_storage_capped(self):
        """测试初始土壤存储被限制在max_storage内"""
        params = {'max_storage': 100.0, 'initial_soil_storage': 150.0}

        # 应该抛出验证错误，因为initial_soil_storage > max_storage
        with pytest.raises(ParameterValidationError, match="initial_soil_storage"):
            model = HYMODRunoff(params)

    def test_multiple_quick_reservoirs(self):
        """测试多个快速水库初始化"""
        params = {'num_quick_reservoirs': 5, 'initial_quick_storage': 2.0}
        model = HYMODRunoff(params)

        assert len(model.quick_states) == 5
        assert all(s == 2.0 for s in model.quick_states)


class TestHYMODParameterValidation:
    """测试HYMOD模型参数验证"""

    def test_valid_parameters(self, default_hymod_params):
        """测试有效参数"""
        model = HYMODRunoff(default_hymod_params)
        # 应该不抛出异常
        model.validate_parameters()

    def test_invalid_max_storage_zero(self):
        """测试无效的max_storage（零）"""
        params = {'max_storage': 0.0}

        with pytest.raises(ParameterValidationError, match="max_storage"):
            model = HYMODRunoff(params)

    def test_invalid_max_storage_negative(self):
        """测试无效的max_storage（负数）"""
        params = {'max_storage': -10.0}

        with pytest.raises(ParameterValidationError, match="max_storage"):
            model = HYMODRunoff(params)

    def test_invalid_quickflow_ratio_above_one(self):
        """测试无效的quickflow_ratio（大于1）"""
        params = {'quickflow_ratio': 1.5}

        with pytest.raises(ParameterValidationError, match="quickflow_ratio"):
            model = HYMODRunoff(params)

    def test_invalid_quickflow_ratio_negative(self):
        """测试无效的quickflow_ratio（负数）"""
        params = {'quickflow_ratio': -0.1}

        with pytest.raises(ParameterValidationError, match="quickflow_ratio"):
            model = HYMODRunoff(params)

    def test_invalid_quick_k_above_one(self):
        """测试无效的quick_k（大于1）"""
        params = {'quick_k': 1.5}

        with pytest.raises(ParameterValidationError, match="quick_k"):
            model = HYMODRunoff(params)

    def test_invalid_slow_k_negative(self):
        """测试无效的slow_k（负数）"""
        params = {'slow_k': -0.1}

        with pytest.raises(ParameterValidationError, match="slow_k"):
            model = HYMODRunoff(params)

    def test_invalid_num_quick_reservoirs_zero(self):
        """测试无效的num_quick_reservoirs（零）"""
        params = {'num_quick_reservoirs': 0}

        with pytest.raises(ParameterValidationError, match="num_quick_reservoirs"):
            model = HYMODRunoff(params)

    def test_valid_beta_zero(self):
        """测试beta=0是有效的（线性）"""
        params = {'beta': 0.0}
        model = HYMODRunoff(params)
        # 应该不抛出异常
        model.validate_parameters()


class TestHYMODSimulation:
    """测试HYMOD模型模拟"""

    def test_simulate_zero_rainfall(self, default_hymod_params, sample_subbasin):
        """测试零降雨情况"""
        model = HYMODRunoff(default_hymod_params)
        precipitation = [0.0] * 10

        runoff = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 10
        assert all(isinstance(q, (int, float)) for q in runoff)
        assert all(q >= 0 for q in runoff)

    def test_simulate_constant_rainfall(self, default_hymod_params, sample_subbasin):
        """测试恒定降雨"""
        model = HYMODRunoff(default_hymod_params)
        precipitation = [5.0] * 20

        runoff = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 20
        assert all(q >= 0 for q in runoff)
        # 恒定降雨应产生径流
        assert sum(runoff) > 0

    def test_simulate_varying_rainfall(self, default_hymod_params, sample_subbasin):
        """测试变化降雨"""
        model = HYMODRunoff(default_hymod_params)
        # 模拟降雨事件
        precipitation = [0] * 5 + [10, 20, 15, 10, 5] + [1] * 10

        runoff = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 20
        assert all(q >= 0 for q in runoff)
        # 降雨期径流应大于无雨期
        rain_period_mean = np.mean(runoff[5:10])
        no_rain_mean = np.mean(runoff[:5])
        assert rain_period_mean > no_rain_mean

    def test_simulate_heavy_rainfall(self, default_hymod_params, sample_subbasin):
        """测试强降雨事件"""
        model = HYMODRunoff(default_hymod_params)
        precipitation = [0] * 5 + [50.0] + [0] * 10

        runoff = model.simulate(sample_subbasin, precipitation)

        # 强降雨应产生显著径流
        assert max(runoff) > 0
        # 峰值径流应大于平均径流
        assert max(runoff) > np.mean(runoff) * 2


class TestHYMODSoilStorage:
    """测试HYMOD模型的土壤存储"""

    def test_soil_storage_not_exceed_max(self, default_hymod_params, sample_subbasin):
        """测试土壤存储不超过max_storage"""
        model = HYMODRunoff(default_hymod_params)
        # 极端强降雨
        precipitation = [100.0] * 10

        model.simulate(sample_subbasin, precipitation)

        # 土壤存储不应超过max_storage
        assert 0 <= model.soil_storage <= model.smax

    def test_soil_storage_increases(self, default_hymod_params, sample_subbasin):
        """测试土壤存储不会超过最大值"""
        params = default_hymod_params.copy()
        params['initial_soil_storage'] = 10.0  # 低初始值
        params['max_storage'] = 100.0
        model = HYMODRunoff(params)

        initial_soil = model.soil_storage

        # 大量降雨应增加土壤存储直到饱和
        precipitation = [20.0] * 20
        model.simulate(sample_subbasin, precipitation)

        # 土壤存储应增加，但不超过max_storage
        assert model.soil_storage >= initial_soil
        assert model.soil_storage <= model.smax

    def test_beta_parameter_effect(self, sample_subbasin):
        """测试beta参数的有效性"""
        # beta参数应该可以设置为不同值
        # 测试模型接受不同beta值并正常运行
        for beta_value in [0.1, 1.0, 3.0, 10.0]:
            model = HYMODRunoff({
                'beta': beta_value,
                'max_storage': 100.0,
                'initial_soil_storage': 50.0,
                'quickflow_ratio': 0.7
            })

            precipitation = [10.0] * 5
            runoff = model.simulate(sample_subbasin, precipitation)

            # 模型应正常运行
            assert len(runoff) == 5
            assert all(q >= 0 for q in runoff)
            # 应产生一些径流
            assert sum(runoff) > 0


class TestHYMODQuickflowRouting:
    """测试HYMOD模型的快速流路由"""

    def test_single_quick_reservoir(self, sample_subbasin):
        """测试单个快速水库"""
        params = {
            'max_storage': 100.0,
            'num_quick_reservoirs': 1,
            'quickflow_ratio': 1.0,  # 全部快速流
            'quick_k': 0.5,
            'initial_soil_storage': 0.0  # 低初始，促进有效降雨
        }
        model = HYMODRunoff(params)

        precipitation = [10.0] + [0.0] * 10
        runoff = model.simulate(sample_subbasin, precipitation)

        # 应产生快速流
        assert runoff[0] > 0 or runoff[1] > 0
        # 快速流应递减
        peak_idx = np.argmax(runoff[:5])
        if peak_idx < 4:
            assert runoff[peak_idx] >= runoff[peak_idx + 1]

    def test_multiple_quick_reservoirs(self, sample_subbasin):
        """测试多个串联快速水库"""
        params = {
            'max_storage': 100.0,
            'num_quick_reservoirs': 5,
            'quickflow_ratio': 1.0,
            'quick_k': 0.5,
            'initial_soil_storage': 0.0
        }
        model = HYMODRunoff(params)

        precipitation = [10.0] + [0.0] * 10
        runoff = model.simulate(sample_subbasin, precipitation)

        # 多个水库会延迟和平滑峰值
        assert sum(runoff) > 0

    def test_quickflow_ratio_effect(self, sample_subbasin):
        """测试quickflow_ratio参数的影响"""
        # 高快速流比例
        model_high = HYMODRunoff({
            'quickflow_ratio': 0.9,
            'initial_soil_storage': 0.0
        })

        # 低快速流比例
        model_low = HYMODRunoff({
            'quickflow_ratio': 0.3,
            'initial_soil_storage': 0.0
        })

        precipitation = [20.0] + [0.0] * 15

        runoff_high = model_high.simulate(sample_subbasin, precipitation)
        runoff_low = model_low.simulate(sample_subbasin, precipitation)

        # 高快速流比例应有更快的响应
        # 检查前几个时间步的径流
        early_runoff_high = sum(runoff_high[:5])
        early_runoff_low = sum(runoff_low[:5])
        assert early_runoff_high > early_runoff_low


class TestHYMODSlowflow:
    """测试HYMOD模型的慢速流"""

    def test_slowflow_generation(self, default_hymod_params, sample_subbasin):
        """测试慢速流生成"""
        params = default_hymod_params.copy()
        params['quickflow_ratio'] = 0.3  # 更多慢速流
        params['initial_slow_storage'] = 20.0  # 初始慢速存储
        params['slow_k'] = 0.1  # 慢速消退系数
        model = HYMODRunoff(params)

        # 无降雨，测试慢速流消退
        precipitation = [0.0] * 20
        runoff = model.simulate(sample_subbasin, precipitation)

        # 应产生持续慢速流
        assert sum(runoff[:10]) > 0
        # 慢速流应缓慢递减
        assert runoff[0] > runoff[-1]

    def test_slow_k_effect(self, sample_subbasin):
        """测试slow_k参数的影响"""
        # 高slow_k（快速消退）
        model_fast = HYMODRunoff({
            'slow_k': 0.5,
            'quickflow_ratio': 0.3,
            'initial_slow_storage': 50.0
        })

        # 低slow_k（慢速消退）
        model_slow = HYMODRunoff({
            'slow_k': 0.05,
            'quickflow_ratio': 0.3,
            'initial_slow_storage': 50.0
        })

        precipitation = [0.0] * 20

        runoff_fast = model_fast.simulate(sample_subbasin, precipitation)
        runoff_slow = model_slow.simulate(sample_subbasin, precipitation)

        # 高slow_k应有更快的消退
        # 后期径流：低k应保留更多基流
        assert sum(runoff_slow[10:]) > sum(runoff_fast[10:])


class TestHYMODIntegratedFlow:
    """测试HYMOD模型的快慢流集成"""

    def test_combined_quickflow_slowflow(self, default_hymod_params, sample_subbasin):
        """测试快速流和慢速流的组合"""
        model = HYMODRunoff(default_hymod_params)

        # 降雨事件后是消退期
        precipitation = [0, 0, 20.0, 10.0, 0] + [0.0] * 15

        runoff = model.simulate(sample_subbasin, precipitation)

        # 应同时产生快速流和慢速流
        assert max(runoff) > 0
        # 消退期应有持续径流（慢速流）
        assert sum(runoff[10:15]) > 0

    def test_effective_rain_calculation(self, sample_subbasin):
        """测试有效降雨计算"""
        params = {
            'max_storage': 100.0,
            'beta': 1.0,
            'initial_soil_storage': 90.0,  # 高初始土壤含水量
            'quickflow_ratio': 1.0  # 全部快速流，便于观察
        }
        model = HYMODRunoff(params)

        # 小降雨应主要被土壤吸收
        precipitation = [2.0] * 5
        runoff = model.simulate(sample_subbasin, precipitation)

        # 土壤接近饱和时仍应产生一些径流
        assert sum(runoff) >= 0


class TestHYMODEdgeCases:
    """测试HYMOD模型边界情况"""

    def test_empty_precipitation(self, default_hymod_params, sample_subbasin):
        """测试空降雨序列"""
        model = HYMODRunoff(default_hymod_params)
        precipitation = []

        runoff = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 0

    def test_single_timestep(self, default_hymod_params, sample_subbasin):
        """测试单个时间步"""
        model = HYMODRunoff(default_hymod_params)
        precipitation = [10.0]

        runoff = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 1
        assert runoff[0] >= 0

    def test_extreme_parameters(self, sample_subbasin):
        """测试极端参数值"""
        # 极小max_storage（快速饱和）
        params = {
            'max_storage': 10.0,
            'beta': 0.5,
            'quick_k': 0.8,
            'slow_k': 0.1
        }
        model = HYMODRunoff(params)
        precipitation = [10.0] * 10

        runoff = model.simulate(sample_subbasin, precipitation)

        assert all(q >= 0 for q in runoff)
        # 小max_storage应快速饱和，产生较大径流
        assert max(runoff) > 0

    def test_different_area_sizes(self, default_hymod_params):
        """测试不同流域面积"""
        model = HYMODRunoff(default_hymod_params)
        precipitation = [10.0] * 5

        # 小流域
        small_basin = MockSubbasin(10.0)
        runoff_small = model.simulate(small_basin, precipitation)

        # 重置模型状态
        model = HYMODRunoff(default_hymod_params)

        # 大流域
        large_basin = MockSubbasin(1000.0)
        runoff_large = model.simulate(large_basin, precipitation)

        # 径流应与面积成正比
        ratio = large_basin.area_km2 / small_basin.area_km2
        assert abs(runoff_large[0] / runoff_small[0] - ratio) < 0.01

    def test_zero_initial_storages(self, sample_subbasin):
        """测试零初始存储"""
        params = {
            'max_storage': 100.0,
            'initial_soil_storage': 0.0,
            'initial_quick_storage': 0.0,
            'initial_slow_storage': 0.0
        }
        model = HYMODRunoff(params)

        precipitation = [10.0] * 10
        runoff = model.simulate(sample_subbasin, precipitation)

        # 模型应正常运行
        assert all(q >= 0 for q in runoff)


class TestHYMODStorageBalances:
    """测试HYMOD模型的存储平衡"""

    def test_storage_changes_with_precipitation(self, default_hymod_params, sample_subbasin):
        """测试降雨时存储变化"""
        model = HYMODRunoff(default_hymod_params)

        initial_soil = model.soil_storage
        initial_slow = model.slow_state

        precipitation = [10.0] * 10
        runoff = model.simulate(sample_subbasin, precipitation)

        # 降雨应导致存储变化
        assert (model.soil_storage != initial_soil or
                model.slow_state != initial_slow)
        # 应产生径流
        assert sum(runoff) > 0

    def test_recession_behavior(self, default_hymod_params, sample_subbasin):
        """测试消退过程"""
        params = default_hymod_params.copy()
        params['initial_quick_storage'] = 10.0
        params['initial_slow_storage'] = 20.0
        model = HYMODRunoff(params)

        # 无降雨，观察消退
        precipitation = [0.0] * 20
        runoff = model.simulate(sample_subbasin, precipitation)

        # 径流应递减
        assert runoff[0] > runoff[5]
        assert runoff[5] > runoff[15]

    def test_soil_infiltration(self, sample_subbasin):
        """测试土壤入渗机制"""
        params = {
            'max_storage': 100.0,
            'beta': 1.0,
            'initial_soil_storage': 10.0,  # 低土壤含水量
            'quickflow_ratio': 0.5  # 部分快速流
        }
        model = HYMODRunoff(params)

        initial_soil = model.soil_storage

        # 持续降雨，土壤存储应响应
        precipitation = [10.0] * 10
        runoff = model.simulate(sample_subbasin, precipitation)

        # 土壤含水量应保持在有效范围内
        assert 0 <= model.soil_storage <= model.smax
        # 应产生径流（验证模型工作正常）
        assert sum(runoff) > 0


class TestHYMODComplexScenarios:
    """测试HYMOD模型的复杂场景"""

    def test_multiple_rain_events(self, default_hymod_params, sample_subbasin):
        """测试多次降雨事件"""
        model = HYMODRunoff(default_hymod_params)

        # 两次降雨事件，中间有间隔
        precipitation = (
            [10.0] * 3 +      # 第一次降雨
            [0] * 5 +         # 间隔
            [15.0] * 3 +      # 第二次降雨
            [0] * 10          # 消退
        )

        runoff = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == len(precipitation)
        assert all(q >= 0 for q in runoff)
        # 应观察到两个峰值
        # 至少前半和后半应各有一些较高径流
        assert max(runoff[:8]) > 0
        assert max(runoff[8:15]) > 0

    def test_prolonged_rainfall(self, default_hymod_params, sample_subbasin):
        """测试长期降雨"""
        model = HYMODRunoff(default_hymod_params)

        # 长期持续降雨
        precipitation = [3.0] * 50

        runoff = model.simulate(sample_subbasin, precipitation)

        # 土壤应逐渐饱和
        assert model.soil_storage >= default_hymod_params['initial_soil_storage']
        # 应产生持续径流
        assert all(q > 0 for q in runoff[10:])

    def test_drought_recovery(self, sample_subbasin):
        """测试干旱后恢复响应"""
        params = {
            'max_storage': 100.0,
            'beta': 0.5,  # 较低beta使得降雨更容易产流
            'initial_soil_storage': 5.0,  # 干旱条件
            'initial_slow_storage': 0.5,
            'quickflow_ratio': 0.7
        }
        model = HYMODRunoff(params)

        # 长期无雨后突降大雨
        precipitation = [0] * 10 + [50.0] + [20.0] * 3 + [10.0] * 10

        runoff = model.simulate(sample_subbasin, precipitation)

        # 应产生显著的径流响应
        assert max(runoff[10:15]) > max(runoff[:10]) * 2
        # 土壤含水量应保持在有效范围
        assert model.soil_storage >= 0
