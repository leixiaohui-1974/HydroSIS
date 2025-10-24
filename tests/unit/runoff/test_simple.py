"""Simple产流模型的单元测试"""
import pytest
import numpy as np
from hydrosis.runoff.simple import SimpleRunoff


@pytest.fixture
def sample_subbasin():
    """样本子流域（用于测试）"""
    class MockSubbasin:
        def __init__(self):
            self.area_km2 = 100.0
    return MockSubbasin()


# ==================== 初始化测试 ====================

class TestSimpleRunoffInitialization:
    """测试Simple产流模型初始化"""

    def test_default_initialization(self):
        """测试默认参数初始化"""
        model = SimpleRunoff({})
        assert hasattr(model, 'parameters')
        assert model.parameters == {}

    def test_initialization_with_parameters(self):
        """测试带参数初始化（参数会被忽略）"""
        params = {"some_param": 123, "another_param": "test"}
        model = SimpleRunoff(params)
        # Simple runoff不使用任何参数，但存储它们
        assert model.parameters == params

    def test_validate_parameters(self):
        """测试参数验证（应该总是通过）"""
        model = SimpleRunoff({})
        model.validate_parameters()  # 不应抛出异常

        model_with_params = SimpleRunoff({"param": "value"})
        model_with_params.validate_parameters()  # 也不应抛出异常


# ==================== 仿真测试 ====================

class TestSimpleRunoffSimulation:
    """测试仿真功能"""

    def test_precipitation_equals_runoff(self, sample_subbasin):
        """测试降雨等于径流"""
        model = SimpleRunoff({})
        precipitation = [10.0, 20.0, 30.0, 40.0, 50.0]
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        assert runoff == precipitation
        assert storage == 0.0

    def test_zero_precipitation(self, sample_subbasin):
        """测试零降雨"""
        model = SimpleRunoff({})
        precipitation = [0.0] * 10
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 10
        assert all(r == 0.0 for r in runoff)
        assert storage == 0.0

    def test_constant_precipitation(self, sample_subbasin):
        """测试恒定降雨"""
        model = SimpleRunoff({})
        precipitation = [25.0] * 20
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 20
        assert all(r == 25.0 for r in runoff)
        assert storage == 0.0

    def test_varying_precipitation(self, sample_subbasin):
        """测试变化降雨"""
        model = SimpleRunoff({})
        precipitation = [10.0, 25.0, 40.0, 30.0, 15.0, 5.0]
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == len(precipitation)
        for i in range(len(precipitation)):
            assert runoff[i] == precipitation[i]
        assert storage == 0.0

    def test_heavy_precipitation(self, sample_subbasin):
        """测试强降雨"""
        model = SimpleRunoff({})
        precipitation = [200.0] * 5
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        assert all(r == 200.0 for r in runoff)
        assert storage == 0.0


# ==================== 边界情况测试 ====================

class TestSimpleRunoffEdgeCases:
    """测试边界情况"""

    def test_empty_precipitation(self, sample_subbasin):
        """测试空降雨序列"""
        model = SimpleRunoff({})
        precipitation = []
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        assert runoff == []
        assert len(runoff) == 0
        assert storage == 0.0

    def test_single_timestep(self, sample_subbasin):
        """测试单时间步"""
        model = SimpleRunoff({})
        precipitation = [50.0]
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 1
        assert runoff[0] == 50.0
        assert storage == 0.0

    def test_negative_precipitation(self, sample_subbasin):
        """测试负值降雨（虽然物理上不合理，但模型应该能处理）"""
        model = SimpleRunoff({})
        precipitation = [-10.0, -5.0, 0.0, 5.0, 10.0]
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        # Simple runoff应该原样返回，包括负值
        assert runoff == precipitation
        assert storage == 0.0

    def test_very_large_precipitation(self, sample_subbasin):
        """测试极大降雨值"""
        model = SimpleRunoff({})
        precipitation = [1e6, 1e7, 1e8]
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        assert runoff == precipitation
        assert storage == 0.0


# ==================== 存储测试 ====================

class TestSimpleRunoffStorage:
    """测试存储特性"""

    def test_storage_always_zero(self, sample_subbasin):
        """测试存储总是为0"""
        model = SimpleRunoff({})
        precipitation = [10.0, 20.0, 30.0, 40.0, 50.0]
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        assert storage == 0.0

    def test_storage_zero_with_initial_storage(self, sample_subbasin):
        """测试即使提供初始存储也返回0"""
        model = SimpleRunoff({})
        precipitation = [10.0, 20.0, 30.0]
        runoff, storage = model.simulate(sample_subbasin, precipitation, initial_storage=100.0)

        # 初始存储参数被忽略
        assert storage == 0.0

    def test_no_storage_accumulation(self, sample_subbasin):
        """测试无存储积累"""
        model = SimpleRunoff({})

        # 第一次仿真
        precipitation1 = [50.0] * 5
        runoff1, storage1 = model.simulate(sample_subbasin, precipitation1)

        # 第二次仿真
        precipitation2 = [50.0] * 5
        runoff2, storage2 = model.simulate(sample_subbasin, precipitation2)

        # 两次仿真应该产生相同结果（无状态）
        assert runoff1 == runoff2
        assert storage1 == storage2 == 0.0


# ==================== 无状态特性测试 ====================

class TestSimpleRunoffStateless:
    """测试Simple模型的无状态特性"""

    def test_no_carryover_between_simulations(self, sample_subbasin):
        """测试仿真之间无延续"""
        model = SimpleRunoff({})

        # 两次相同的仿真应该产生相同的结果
        precipitation = [80.0, 80.0, 80.0]

        runoff1, storage1 = model.simulate(sample_subbasin, precipitation)
        runoff2, storage2 = model.simulate(sample_subbasin, precipitation)

        assert runoff1 == runoff2
        assert storage1 == storage2 == 0.0

    def test_order_independence(self, sample_subbasin):
        """测试顺序无关性"""
        model = SimpleRunoff({})

        # 不同顺序的相同降雨值
        precipitation1 = [50.0, 100.0, 150.0]
        precipitation2 = [150.0, 50.0, 100.0]

        runoff1, storage1 = model.simulate(sample_subbasin, precipitation1)
        runoff2, storage2 = model.simulate(sample_subbasin, precipitation2)

        # 径流应该与输入顺序相同
        assert runoff1 == precipitation1
        assert runoff2 == precipitation2
        assert storage1 == storage2 == 0.0

    def test_reuse_model_multiple_simulations(self, sample_subbasin):
        """测试重复使用模型"""
        model = SimpleRunoff({})

        precipitation1 = [100.0]
        precipitation2 = [100.0]

        runoff1, storage1 = model.simulate(sample_subbasin, precipitation1)
        runoff2, storage2 = model.simulate(sample_subbasin, precipitation2)

        # 两次仿真应该产生相同结果（无状态）
        assert runoff1 == runoff2
        assert storage1 == storage2 == 0.0


# ==================== 流域面积无关性测试 ====================

class TestSimpleRunoffAreaIndependence:
    """测试Simple产流与流域面积无关"""

    def test_area_does_not_affect_runoff(self):
        """测试流域面积不影响径流结果"""
        precipitation = [10.0, 20.0, 30.0, 40.0, 50.0]

        class SmallBasin:
            area_km2 = 10.0

        class LargeBasin:
            area_km2 = 1000.0

        model = SimpleRunoff({})

        runoff_small, storage_small = model.simulate(SmallBasin(), precipitation)
        runoff_large, storage_large = model.simulate(LargeBasin(), precipitation)

        # Simple runoff是纯传递，不应受流域面积影响
        assert runoff_small == runoff_large
        assert runoff_small == precipitation
        assert storage_small == storage_large == 0.0


# ==================== 长时间序列测试 ====================

class TestSimpleRunoffLongSeries:
    """测试长时间序列"""

    def test_long_constant_series(self, sample_subbasin):
        """测试长恒定序列"""
        model = SimpleRunoff({})
        precipitation = [25.0] * 1000
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 1000
        assert all(r == 25.0 for r in runoff)
        assert storage == 0.0

    def test_long_varying_series(self, sample_subbasin):
        """测试长变化序列"""
        model = SimpleRunoff({})
        # 创建一个有模式的长序列
        precipitation = [float(i % 10) for i in range(500)]
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 500
        assert runoff == precipitation
        assert storage == 0.0

    def test_multiple_rain_events(self, sample_subbasin):
        """测试多次降雨事件"""
        model = SimpleRunoff({})
        # 多次降雨-干旱循环
        precipitation = [60.0, 40.0, 20.0, 0.0, 0.0, 80.0, 50.0, 20.0, 0.0, 0.0]
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        assert runoff == precipitation
        assert storage == 0.0


# ==================== 数据类型测试 ====================

class TestSimpleRunoffDataTypes:
    """测试不同数据类型的处理"""

    def test_list_precipitation(self, sample_subbasin):
        """测试列表类型降雨"""
        model = SimpleRunoff({})
        precipitation = [10.0, 20.0, 30.0]
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        assert isinstance(runoff, list)
        assert runoff == precipitation
        assert storage == 0.0

    def test_mixed_numeric_types(self, sample_subbasin):
        """测试混合数值类型降雨"""
        model = SimpleRunoff({})
        precipitation = [10, 20.5, 30, 40.7, 50]  # 混合int和float
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        # 应该保持原始类型
        assert runoff == precipitation
        assert storage == 0.0


# ==================== 返回值测试 ====================

class TestSimpleRunoffReturnValue:
    """测试返回值格式"""

    def test_returns_tuple(self, sample_subbasin):
        """测试返回元组"""
        model = SimpleRunoff({})
        precipitation = [10.0, 20.0, 30.0]
        result = model.simulate(sample_subbasin, precipitation)

        # 应该返回元组
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_tuple_first_element_is_runoff(self, sample_subbasin):
        """测试元组第一个元素是径流"""
        model = SimpleRunoff({})
        precipitation = [10.0, 20.0, 30.0]
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        assert runoff == precipitation
        assert isinstance(runoff, list)

    def test_tuple_second_element_is_storage(self, sample_subbasin):
        """测试元组第二个元素是存储"""
        model = SimpleRunoff({})
        precipitation = [10.0, 20.0, 30.0]
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        assert storage == 0.0
        assert isinstance(storage, float)


# ==================== 对比测试 ====================

class TestSimpleRunoffComparison:
    """测试与其他产流方法对比"""

    def test_simpler_than_linear_reservoir(self, sample_subbasin):
        """测试比LinearReservoir更简单（无衰减）"""
        from hydrosis.runoff.linear_reservoir import LinearReservoirRunoff

        precipitation = [50.0] * 5

        simple_model = SimpleRunoff({})
        linear_model = LinearReservoirRunoff({"recession": 0.9, "conversion": 1.0})

        simple_runoff, _ = simple_model.simulate(sample_subbasin, precipitation)
        linear_runoff = linear_model.simulate(sample_subbasin, precipitation)

        # Simple应该是原始降雨
        assert simple_runoff == precipitation
        # LinearReservoir会有不同的响应（需要考虑面积）
        # 但至少长度应该相同
        assert len(simple_runoff) == len(linear_runoff)

    def test_simpler_than_scs_curve_number(self, sample_subbasin):
        """测试比SCS Curve Number更简单（无初损）"""
        from hydrosis.runoff.scs_curve_number import SCSCurveNumber

        precipitation = [50.0, 75.0, 100.0]

        simple_model = SimpleRunoff({})
        scs_model = SCSCurveNumber({"curve_number": 75})

        simple_runoff, _ = simple_model.simulate(sample_subbasin, precipitation)
        scs_runoff = scs_model.simulate(sample_subbasin, precipitation)

        # Simple返回原始降雨
        assert simple_runoff == precipitation
        # SCS会有初损和非线性响应
        # Simple的径流应该更大（无初损）
        # 注意：SCS返回的是考虑面积后的值，所以不能直接比较大小


# ==================== 数值稳定性测试 ====================

class TestSimpleRunoffNumericalStability:
    """测试数值稳定性"""

    def test_handles_random_precipitation(self, sample_subbasin):
        """测试处理随机降雨"""
        model = SimpleRunoff({})

        # 随机降雨
        precipitation = [float(np.random.rand() * 200) for _ in range(100)]
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        assert runoff == precipitation
        assert storage == 0.0

    def test_very_long_series_stable(self, sample_subbasin):
        """测试非常长序列的稳定性"""
        model = SimpleRunoff({})
        precipitation = [50.0] * 10000
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 10000
        assert all(r == 50.0 for r in runoff)
        assert storage == 0.0


# ==================== 完美守恒测试 ====================

class TestSimpleRunoffConservation:
    """测试完美守恒性"""

    def test_perfect_mass_conservation(self, sample_subbasin):
        """测试完美的质量守恒"""
        model = SimpleRunoff({})
        precipitation = [10.0, 20.0, 30.0, 40.0, 50.0]
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        # Simple runoff应该完美守恒（径流=降雨，存储=0）
        assert sum(runoff) == sum(precipitation)
        assert storage == 0.0

    def test_no_loss_or_gain(self, sample_subbasin):
        """测试无损失或增益"""
        model = SimpleRunoff({})
        precipitation = [15.0, 25.0, 35.0]
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        # 每个值都应该完全相同
        for i in range(len(precipitation)):
            assert runoff[i] == precipitation[i]
        assert storage == 0.0

    def test_conservation_with_zeros(self, sample_subbasin):
        """测试包含零值的守恒"""
        model = SimpleRunoff({})
        precipitation = [0.0, 10.0, 0.0, 20.0, 0.0]
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        assert sum(runoff) == sum(precipitation)
        assert storage == 0.0
