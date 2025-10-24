"""Unit tests for Simple routing model."""
import pytest
import numpy as np
from hydrosis.routing.simple import SimpleRouting


@pytest.fixture
def sample_subbasin():
    """样本子流域（用于测试）"""
    class MockSubbasin:
        def __init__(self):
            self.area_km2 = 100.0
    return MockSubbasin()


# ==================== 初始化测试 ====================

class TestSimpleRoutingInitialization:
    """测试Simple路由模型初始化"""

    def test_default_initialization(self):
        """测试默认参数初始化"""
        model = SimpleRouting({})
        assert hasattr(model, 'parameters')
        assert model.parameters == {}

    def test_initialization_with_parameters(self):
        """测试带参数初始化（参数会被忽略）"""
        params = {"some_param": 123, "another_param": "test"}
        model = SimpleRouting(params)
        # Simple routing不使用任何参数，但存储它们
        assert model.parameters == params

    def test_validate_parameters(self):
        """测试参数验证（应该总是通过）"""
        model = SimpleRouting({})
        model.validate_parameters()  # 不应抛出异常

        model_with_params = SimpleRouting({"param": "value"})
        model_with_params.validate_parameters()  # 也不应抛出异常


# ==================== 路由仿真测试 ====================

class TestSimpleRoutingSimulation:
    """测试路由仿真"""

    def test_inflow_equals_outflow(self, sample_subbasin):
        """测试入流等于出流"""
        model = SimpleRouting({})
        inflow = [10.0, 20.0, 30.0, 40.0, 50.0]
        outflow = model.route(sample_subbasin, inflow)

        assert outflow == inflow
        assert np.allclose(outflow, inflow)

    def test_constant_inflow(self, sample_subbasin):
        """测试恒定入流"""
        model = SimpleRouting({})
        inflow = [25.0] * 20
        outflow = model.route(sample_subbasin, inflow)

        assert len(outflow) == len(inflow)
        assert all(q == 25.0 for q in outflow)

    def test_varying_inflow(self, sample_subbasin):
        """测试变化入流"""
        model = SimpleRouting({})
        inflow = [10.0, 25.0, 40.0, 30.0, 15.0, 5.0]
        outflow = model.route(sample_subbasin, inflow)

        assert len(outflow) == len(inflow)
        for i in range(len(inflow)):
            assert outflow[i] == inflow[i]

    def test_peak_not_attenuated(self, sample_subbasin):
        """测试洪峰不被削减"""
        model = SimpleRouting({})
        inflow = [10.0, 20.0, 100.0, 30.0, 15.0, 10.0]
        outflow = model.route(sample_subbasin, inflow)

        # 洪峰值应该保持不变
        assert max(outflow) == max(inflow) == 100.0
        # 洪峰位置应该相同
        assert outflow.index(100.0) == inflow.index(100.0) == 2

    def test_no_delay(self, sample_subbasin):
        """测试没有延迟"""
        model = SimpleRouting({})
        inflow = [5.0, 10.0, 50.0, 30.0, 10.0, 5.0]
        outflow = model.route(sample_subbasin, inflow)

        # 每个时间步的值应该完全相同（无延迟）
        for i in range(len(inflow)):
            assert outflow[i] == inflow[i]


# ==================== 边界情况测试 ====================

class TestSimpleRoutingEdgeCases:
    """测试边界情况"""

    def test_empty_inflow(self, sample_subbasin):
        """测试空入流序列"""
        model = SimpleRouting({})
        inflow = []
        outflow = model.route(sample_subbasin, inflow)

        assert outflow == []
        assert len(outflow) == 0

    def test_single_value_inflow(self, sample_subbasin):
        """测试单值入流"""
        model = SimpleRouting({})
        inflow = [50.0]
        outflow = model.route(sample_subbasin, inflow)

        assert len(outflow) == 1
        assert outflow[0] == 50.0

    def test_zero_inflow(self, sample_subbasin):
        """测试全零入流"""
        model = SimpleRouting({})
        inflow = [0.0] * 10
        outflow = model.route(sample_subbasin, inflow)

        assert len(outflow) == 10
        assert all(q == 0.0 for q in outflow)

    def test_negative_inflow(self, sample_subbasin):
        """测试负值入流（虽然物理上不合理，但模型应该能处理）"""
        model = SimpleRouting({})
        inflow = [-10.0, -5.0, 0.0, 5.0, 10.0]
        outflow = model.route(sample_subbasin, inflow)

        # Simple routing应该原样返回，包括负值
        assert outflow == inflow

    def test_very_large_inflow(self, sample_subbasin):
        """测试极大入流值"""
        model = SimpleRouting({})
        inflow = [1e6, 1e7, 1e8]
        outflow = model.route(sample_subbasin, inflow)

        assert outflow == inflow


# ==================== 守恒性测试 ====================

class TestSimpleRoutingConservation:
    """测试质量守恒性"""

    def test_perfect_volume_conservation(self, sample_subbasin):
        """测试完美的体积守恒"""
        model = SimpleRouting({})
        inflow = [10.0, 20.0, 30.0, 40.0, 50.0]
        outflow = model.route(sample_subbasin, inflow)

        # Simple routing应该完美守恒
        assert sum(outflow) == sum(inflow)

    def test_no_attenuation(self, sample_subbasin):
        """测试没有衰减"""
        model = SimpleRouting({})
        inflow = [10.0, 25.0, 40.0, 30.0, 15.0, 5.0]
        outflow = model.route(sample_subbasin, inflow)

        # 每个值都应该完全相同
        assert np.allclose(outflow, inflow)

    def test_peak_value_unchanged(self, sample_subbasin):
        """测试洪峰值不变"""
        model = SimpleRouting({})
        inflow = [10.0, 20.0, 100.0, 30.0, 15.0, 10.0]
        outflow = model.route(sample_subbasin, inflow)

        # 洪峰值应该完全相同
        assert max(outflow) == max(inflow)

    def test_sum_conservation_long_series(self, sample_subbasin):
        """测试长序列的总和守恒"""
        model = SimpleRouting({})
        inflow = [float(i) for i in range(100)]
        outflow = model.route(sample_subbasin, inflow)

        assert sum(outflow) == sum(inflow)


# ==================== 流域面积无关性测试 ====================

class TestSimpleRoutingAreaIndependence:
    """测试Simple路由与流域面积无关"""

    def test_area_does_not_affect_routing(self):
        """测试流域面积不影响路由结果"""
        inflow = [10.0, 20.0, 30.0, 40.0, 50.0]

        class SmallBasin:
            area_km2 = 10.0

        class LargeBasin:
            area_km2 = 1000.0

        model = SimpleRouting({})

        outflow_small = model.route(SmallBasin(), inflow)
        outflow_large = model.route(LargeBasin(), inflow)

        # Simple routing是纯传递，不应受流域面积影响
        assert outflow_small == outflow_large
        assert np.allclose(outflow_small, outflow_large)


# ==================== 长时间序列测试 ====================

class TestSimpleRoutingLongSeries:
    """测试长时间序列路由"""

    def test_long_constant_series(self, sample_subbasin):
        """测试长恒定序列"""
        model = SimpleRouting({})
        inflow = [25.0] * 1000
        outflow = model.route(sample_subbasin, inflow)

        assert len(outflow) == 1000
        assert all(q == 25.0 for q in outflow)

    def test_long_varying_series(self, sample_subbasin):
        """测试长变化序列"""
        model = SimpleRouting({})
        # 创建一个有模式的长序列
        inflow = [float(i % 10) for i in range(500)]
        outflow = model.route(sample_subbasin, inflow)

        assert len(outflow) == 500
        assert outflow == inflow

    def test_multiple_peaks_unchanged(self, sample_subbasin):
        """测试多个洪峰保持不变"""
        model = SimpleRouting({})
        # 创建有多个洪峰的序列
        inflow = [5, 10, 50, 10, 5, 8, 15, 80, 15, 8, 5]
        outflow = model.route(sample_subbasin, inflow)

        # 找到所有峰值
        peaks_in = [i for i in range(1, len(inflow)-1)
                    if inflow[i] > inflow[i-1] and inflow[i] > inflow[i+1]]
        peaks_out = [i for i in range(1, len(outflow)-1)
                     if outflow[i] > outflow[i-1] and outflow[i] > outflow[i+1]]

        # 峰值位置应该相同
        assert peaks_in == peaks_out
        # 峰值大小应该相同
        for idx in peaks_in:
            assert outflow[idx] == inflow[idx]


# ==================== 数据类型测试 ====================

class TestSimpleRoutingDataTypes:
    """测试不同数据类型的处理"""

    def test_list_inflow(self, sample_subbasin):
        """测试列表类型入流"""
        model = SimpleRouting({})
        inflow = [10.0, 20.0, 30.0]
        outflow = model.route(sample_subbasin, inflow)

        assert isinstance(outflow, list)
        assert outflow == inflow

    def test_mixed_numeric_types(self, sample_subbasin):
        """测试混合数值类型入流"""
        model = SimpleRouting({})
        inflow = [10, 20.5, 30, 40.7, 50]  # 混合int和float
        outflow = model.route(sample_subbasin, inflow)

        # 应该保持原始类型
        assert outflow == inflow


# ==================== 对比测试 ====================

class TestSimpleRoutingComparison:
    """测试与其他路由方法对比"""

    def test_simpler_than_lag_routing(self, sample_subbasin):
        """测试比Lag路由更简单（无延迟）"""
        from hydrosis.routing.lag import LagRouting

        inflow = [10.0, 20.0, 30.0, 40.0, 50.0]

        simple_model = SimpleRouting({})
        lag_model = LagRouting({"lag_steps": 1})

        simple_out = simple_model.route(sample_subbasin, inflow)
        lag_out = lag_model.route(sample_subbasin, inflow)

        # Simple routing应该是原始入流
        assert simple_out == inflow
        # Lag routing应该有延迟（不同于入流）
        assert lag_out != inflow
        # 但长度应该相同
        assert len(simple_out) == len(lag_out)

    def test_simpler_than_muskingum_routing(self, sample_subbasin):
        """测试比Muskingum路由更简单（无削峰）"""
        from hydrosis.routing.muskingum import MuskingumRouting

        inflow = [10.0, 20.0, 50.0, 40.0, 30.0, 20.0, 10.0]

        simple_model = SimpleRouting({})
        muskingum_model = MuskingumRouting({
            "travel_time": 2.0,
            "weighting_factor": 0.2,
            "time_step": 1.0
        })

        simple_out = simple_model.route(sample_subbasin, inflow)
        muskingum_out = muskingum_model.route(sample_subbasin, inflow)

        # Simple routing保持洪峰不变
        assert max(simple_out) == max(inflow)
        # Muskingum应该削峰
        assert max(muskingum_out) < max(inflow)
