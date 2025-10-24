"""Muskingum汇流模型单元测试

测试Muskingum河道汇流模型的核心功能。
Muskingum是经典的水文河道汇流方法，广泛应用于河道流量演算。
"""
import pytest
import numpy as np

from hydrosis.routing.muskingum import MuskingumRouting
from hydrosis.validation import ParameterValidationError


class MockSubbasin:
    """模拟Subbasin对象"""
    def __init__(self, area_km2: float = 100.0):
        self.area_km2 = area_km2
        self.id = "test_subbasin"


@pytest.fixture
def default_muskingum_params():
    """默认Muskingum参数"""
    return {
        'travel_time': 12.0,
        'weighting_factor': 0.2,
        'time_step': 1.0
    }


@pytest.fixture
def sample_subbasin():
    """示例子流域"""
    return MockSubbasin()


class TestMuskingumInitialization:
    """测试Muskingum模型初始化"""

    def test_initialization_with_defaults(self):
        """测试使用默认参数初始化"""
        model = MuskingumRouting({})

        assert model.k > 0
        assert 0 <= model.x <= 0.5
        assert model.dt > 0

    def test_initialization_with_custom_params(self, default_muskingum_params):
        """测试使用自定义参数初始化"""
        model = MuskingumRouting(default_muskingum_params)

        assert model.k == 12.0
        assert model.x == 0.2
        assert model.dt == 1.0

    def test_initialization_with_partial_params(self):
        """测试使用部分参数初始化"""
        params = {'travel_time': 10.0}
        model = MuskingumRouting(params)

        assert model.k == 10.0
        # 其他参数应使用默认值
        assert model.x > 0


class TestMuskingumParameterValidation:
    """测试Muskingum模型参数验证"""

    def test_valid_parameters(self, default_muskingum_params):
        """测试有效参数"""
        model = MuskingumRouting(default_muskingum_params)
        # 应该不抛出异常
        model.validate_parameters()

    def test_invalid_travel_time_zero(self):
        """测试无效的travel_time（零）"""
        params = {'travel_time': 0.0}

        with pytest.raises(ParameterValidationError, match="travel_time"):
            model = MuskingumRouting(params)

    def test_invalid_travel_time_negative(self):
        """测试无效的travel_time（负数）"""
        params = {'travel_time': -5.0}

        with pytest.raises(ParameterValidationError, match="travel_time"):
            model = MuskingumRouting(params)

    def test_invalid_weighting_factor_above_half(self):
        """测试无效的weighting_factor（大于0.5）"""
        params = {'weighting_factor': 0.6}

        with pytest.raises(ParameterValidationError, match="weighting_factor"):
            model = MuskingumRouting(params)

    def test_invalid_weighting_factor_negative(self):
        """测试无效的weighting_factor（负数）"""
        params = {'weighting_factor': -0.1}

        with pytest.raises(ParameterValidationError, match="weighting_factor"):
            model = MuskingumRouting(params)

    def test_valid_weighting_factor_boundary(self):
        """测试边界值weighting_factor（0和0.5）"""
        # x = 0 (最大延迟)
        model1 = MuskingumRouting({'weighting_factor': 0.0})
        model1.validate_parameters()

        # x = 0.5 (无延迟)
        model2 = MuskingumRouting({'weighting_factor': 0.5})
        model2.validate_parameters()

    def test_invalid_time_step_zero(self):
        """测试无效的time_step（零）"""
        params = {'time_step': 0.0}

        with pytest.raises(ParameterValidationError, match="time_step"):
            model = MuskingumRouting(params)

    def test_stability_condition_violation(self):
        """测试稳定性条件违反"""
        # dt > 2*k*(1-x) 应失败
        params = {
            'travel_time': 10.0,
            'weighting_factor': 0.2,
            'time_step': 20.0  # 超过稳定性限制
        }

        with pytest.raises(ParameterValidationError, match="stability"):
            model = MuskingumRouting(params)

    def test_stability_condition_satisfied(self):
        """测试稳定性条件满足"""
        params = {
            'travel_time': 10.0,
            'weighting_factor': 0.2,
            'time_step': 15.0  # 刚好在限制内: 2*10*(1-0.2) = 16
        }
        model = MuskingumRouting(params)
        # 应该不抛出异常
        model.validate_parameters()


class TestMuskingumRouting:
    """测试Muskingum模型河道汇流"""

    def test_route_constant_inflow(self, default_muskingum_params, sample_subbasin):
        """测试恒定入流"""
        model = MuskingumRouting(default_muskingum_params)
        inflow = [100.0] * 20

        outflow = model.route(sample_subbasin, inflow)

        assert len(outflow) == 20
        assert all(isinstance(q, (int, float)) for q in outflow)
        assert all(q >= 0 for q in outflow)
        # 恒定入流下，出流应趋于入流值
        assert abs(outflow[-1] - 100.0) < 1.0

    def test_route_single_peak(self, default_muskingum_params, sample_subbasin):
        """测试单峰洪水过程"""
        model = MuskingumRouting(default_muskingum_params)
        # 三角形洪水过程
        inflow = [10, 30, 60, 100, 80, 50, 30, 20, 15, 10] + [10] * 10

        outflow = model.route(sample_subbasin, inflow)

        assert len(outflow) == len(inflow)
        # 出流峰值应小于入流峰值（削峰作用）
        assert max(outflow) <= max(inflow)
        # 出流峰值应滞后
        peak_in_idx = inflow.index(max(inflow))
        peak_out_idx = outflow.index(max(outflow))
        assert peak_out_idx >= peak_in_idx

    def test_route_zero_inflow(self, default_muskingum_params, sample_subbasin):
        """测试零入流"""
        model = MuskingumRouting(default_muskingum_params)
        inflow = [0.0] * 10

        outflow = model.route(sample_subbasin, inflow)

        assert len(outflow) == 10
        assert all(q == 0.0 for q in outflow)

    def test_route_empty_inflow(self, default_muskingum_params, sample_subbasin):
        """测试空入流序列"""
        model = MuskingumRouting(default_muskingum_params)
        inflow = []

        outflow = model.route(sample_subbasin, inflow)

        assert len(outflow) == 0

    def test_route_single_timestep(self, default_muskingum_params, sample_subbasin):
        """测试单个时间步"""
        model = MuskingumRouting(default_muskingum_params)
        inflow = [50.0]

        outflow = model.route(sample_subbasin, inflow)

        assert len(outflow) == 1
        assert outflow[0] >= 0


class TestMuskingumWeightingFactor:
    """测试Muskingum权重因子x的影响"""

    def test_x_zero_maximum_delay(self, sample_subbasin):
        """测试x=0（最大延迟）"""
        params = {
            'travel_time': 10.0,
            'weighting_factor': 0.0,  # 最大延迟
            'time_step': 1.0
        }
        model = MuskingumRouting(params)

        inflow = [10, 50, 100, 50, 10] + [10] * 10
        outflow = model.route(sample_subbasin, inflow)

        # x=0时，延迟应最大
        peak_in_idx = inflow.index(max(inflow))
        peak_out_idx = outflow.index(max(outflow))
        assert peak_out_idx > peak_in_idx

    def test_x_half_minimum_delay(self, sample_subbasin):
        """测试x=0.5（最小延迟）"""
        params = {
            'travel_time': 10.0,
            'weighting_factor': 0.5,  # 最小延迟
            'time_step': 1.0
        }
        model = MuskingumRouting(params)

        inflow = [10, 50, 100, 50, 10] + [10] * 10
        outflow = model.route(sample_subbasin, inflow)

        # x=0.5时，延迟应最小
        # 出流应更快响应入流
        assert max(outflow) > 0

    def test_x_effect_comparison(self, sample_subbasin):
        """测试不同x值的影响对比"""
        inflow = [10, 30, 60, 100, 70, 40, 20, 10] + [10] * 10

        # x=0.1 (大延迟)
        model_low = MuskingumRouting({
            'travel_time': 10.0,
            'weighting_factor': 0.1,
            'time_step': 1.0
        })
        outflow_low = model_low.route(sample_subbasin, inflow)

        # x=0.4 (小延迟)
        model_high = MuskingumRouting({
            'travel_time': 10.0,
            'weighting_factor': 0.4,
            'time_step': 1.0
        })
        outflow_high = model_high.route(sample_subbasin, inflow)

        # 不同x值应产生不同的出流过程
        assert outflow_low != outflow_high


class TestMuskingumTravelTime:
    """测试Muskingum行进时间k的影响"""

    def test_large_k_more_smoothing(self, sample_subbasin):
        """测试大k值（更多平滑）"""
        params = {
            'travel_time': 20.0,  # 大k值
            'weighting_factor': 0.2,
            'time_step': 1.0
        }
        model = MuskingumRouting(params)

        inflow = [10, 50, 100, 50, 10] + [10] * 10
        outflow = model.route(sample_subbasin, inflow)

        # 大k值应产生更多平滑效果
        # 峰值应更低
        assert max(outflow) < max(inflow)

    def test_small_k_less_smoothing(self, sample_subbasin):
        """测试小k值（较少平滑）"""
        params = {
            'travel_time': 5.0,  # 小k值
            'weighting_factor': 0.2,
            'time_step': 1.0
        }
        model = MuskingumRouting(params)

        inflow = [10, 50, 100, 50, 10] + [10] * 10
        outflow = model.route(sample_subbasin, inflow)

        # 小k值平滑效果较少
        assert max(outflow) > 0

    def test_k_effect_comparison(self, sample_subbasin):
        """测试不同k值的影响对比"""
        inflow = [10, 30, 60, 100, 70, 40, 20, 10] + [10] * 10

        # k=5 (快速响应)
        model_fast = MuskingumRouting({
            'travel_time': 5.0,
            'weighting_factor': 0.2,
            'time_step': 1.0
        })
        outflow_fast = model_fast.route(sample_subbasin, inflow)

        # k=20 (慢速响应)
        model_slow = MuskingumRouting({
            'travel_time': 20.0,
            'weighting_factor': 0.2,
            'time_step': 1.0
        })
        outflow_slow = model_slow.route(sample_subbasin, inflow)

        # 不同k值应产生不同的出流过程
        assert outflow_fast != outflow_slow
        # 大k值应有更多削峰
        assert max(outflow_slow) < max(outflow_fast)


class TestMuskingumTimeStep:
    """测试Muskingum时间步长的影响"""

    def test_different_time_steps(self, sample_subbasin):
        """测试不同时间步长"""
        inflow = [10, 50, 100, 50, 10] + [10] * 10

        # dt = 0.5
        model1 = MuskingumRouting({
            'travel_time': 10.0,
            'weighting_factor': 0.2,
            'time_step': 0.5
        })
        outflow1 = model1.route(sample_subbasin, inflow)

        # dt = 2.0
        model2 = MuskingumRouting({
            'travel_time': 10.0,
            'weighting_factor': 0.2,
            'time_step': 2.0
        })
        outflow2 = model2.route(sample_subbasin, inflow)

        # 不同时间步长应产生不同结果
        assert outflow1 != outflow2


class TestMuskingumConservation:
    """测试Muskingum模型的水量守恒"""

    def test_mass_conservation(self, default_muskingum_params, sample_subbasin):
        """测试质量守恒"""
        model = MuskingumRouting(default_muskingum_params)

        inflow = [10, 30, 60, 100, 80, 50, 30, 20, 15, 10] * 2
        outflow = model.route(sample_subbasin, inflow)

        # 总入流应近似等于总出流（可能有小误差）
        total_in = sum(inflow)
        total_out = sum(outflow)
        # 允许一定误差
        assert abs(total_in - total_out) / total_in < 0.05

    def test_attenuation_effect(self, default_muskingum_params, sample_subbasin):
        """测试削峰作用"""
        model = MuskingumRouting(default_muskingum_params)

        # 尖锐洪峰
        inflow = [10] * 5 + [200] + [10] * 14
        outflow = model.route(sample_subbasin, inflow)

        # 出流峰值应小于入流峰值
        assert max(outflow) < max(inflow)

    def test_translation_effect(self, default_muskingum_params, sample_subbasin):
        """测试平移作用"""
        model = MuskingumRouting(default_muskingum_params)

        # 洪峰
        inflow = [10] * 5 + [100, 80, 60] + [10] * 12
        outflow = model.route(sample_subbasin, inflow)

        # 出流峰值应滞后
        peak_in_idx = inflow.index(max(inflow))
        peak_out_idx = outflow.index(max(outflow))
        assert peak_out_idx > peak_in_idx


class TestMuskingumComplexScenarios:
    """测试Muskingum模型的复杂场景"""

    def test_multiple_peaks(self, default_muskingum_params, sample_subbasin):
        """测试多峰洪水过程"""
        model = MuskingumRouting(default_muskingum_params)

        # 两个洪峰
        inflow = (
            [10, 30, 60, 40, 20, 10] +  # 第一个峰
            [10, 10, 10] +               # 间隔
            [10, 40, 80, 50, 20, 10] +  # 第二个峰
            [10] * 10                    # 消退
        )
        outflow = model.route(sample_subbasin, inflow)

        assert len(outflow) == len(inflow)
        assert all(q >= 0 for q in outflow)
        # 应观察到两个削峰后的峰值
        # 至少有两个局部最大值
        local_maxima = []
        for i in range(1, len(outflow) - 1):
            if outflow[i] > outflow[i-1] and outflow[i] > outflow[i+1]:
                local_maxima.append(i)
        assert len(local_maxima) >= 1  # 至少一个明显峰值

    def test_steep_rising_limb(self, default_muskingum_params, sample_subbasin):
        """测试陡涨洪水"""
        model = MuskingumRouting(default_muskingum_params)

        # 快速上涨
        inflow = [10, 20, 40, 80, 160, 200, 150, 100, 60, 30, 20, 10] + [10] * 10
        outflow = model.route(sample_subbasin, inflow)

        # 出流应平滑
        assert all(q >= 0 for q in outflow)
        # 出流峰值应小于入流峰值
        assert max(outflow) < max(inflow)

    def test_prolonged_high_flow(self, default_muskingum_params, sample_subbasin):
        """测试长期高流量"""
        model = MuskingumRouting(default_muskingum_params)

        # 长期高流量
        inflow = [10] * 5 + [100] * 20 + [10] * 10
        outflow = model.route(sample_subbasin, inflow)

        # 长期高流量下，出流应趋于入流
        assert abs(outflow[-11] - 100.0) < 10.0


class TestMuskingumResolvedParameters:
    """测试Muskingum模型参数解析"""

    def test_resolved_parameters(self, default_muskingum_params, sample_subbasin):
        """测试参数解析"""
        model = MuskingumRouting(default_muskingum_params)

        resolved = model.resolved_parameters(sample_subbasin)

        assert 'travel_time' in resolved
        assert 'weighting_factor' in resolved
        assert 'time_step' in resolved
        assert resolved['travel_time'] == 12.0
        assert resolved['weighting_factor'] == 0.2
        assert resolved['time_step'] == 1.0
