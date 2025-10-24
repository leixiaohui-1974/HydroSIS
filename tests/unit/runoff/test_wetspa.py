"""WetSpa产流模型的单元测试"""
import pytest
import numpy as np
from hydrosis.runoff.wetspa import WETSPARunoff
from hydrosis.validation import ParameterValidationError


@pytest.fixture
def default_wetspa_params():
    """默认WetSpa参数"""
    return {
        "soil_storage_max": 200.0,
        "infiltration_coefficient": 0.6,
        "surface_runoff_coefficient": 0.4,
        "percolation_coefficient": 0.05,
        "baseflow_constant": 0.04,
        "initial_soil_moisture": 100.0,
        "initial_groundwater": 0.0
    }


@pytest.fixture
def sample_subbasin():
    """样本子流域（用于测试）"""
    class MockSubbasin:
        def __init__(self):
            self.area_km2 = 100.0
    return MockSubbasin()


# ==================== 初始化测试 ====================

class TestWETSPAInitialization:
    """测试WetSpa模型初始化"""

    def test_default_initialization(self):
        """测试默认参数初始化"""
        model = WETSPARunoff({})
        assert model.capacity == 200.0
        assert model.infiltration_coeff == 0.6
        assert model.surface_runoff_coeff == 0.4
        assert model.percolation_coeff == 0.05
        assert model.baseflow_constant == 0.04
        assert model.soil_moisture == 100.0  # 0.5 * 200.0
        assert model.groundwater == 0.0

    def test_custom_initialization(self, default_wetspa_params):
        """测试自定义参数初始化"""
        model = WETSPARunoff(default_wetspa_params)
        assert model.capacity == 200.0
        assert model.infiltration_coeff == 0.6
        assert model.surface_runoff_coeff == 0.4
        assert model.percolation_coeff == 0.05
        assert model.baseflow_constant == 0.04
        assert model.soil_moisture == 100.0
        assert model.groundwater == 0.0

    def test_partial_initialization(self):
        """测试部分参数初始化"""
        params = {"soil_storage_max": 150.0}
        model = WETSPARunoff(params)
        assert model.capacity == 150.0
        assert model.infiltration_coeff == 0.6  # 默认值
        assert model.soil_moisture == 75.0  # 0.5 * 150.0

    def test_initialization_with_groundwater(self):
        """测试带初始地下水的初始化"""
        params = {"initial_groundwater": 50.0}
        model = WETSPARunoff(params)
        assert model.groundwater == 50.0

    def test_initial_soil_moisture_capped_by_capacity(self):
        """测试初始土壤水分超过容量会在初始化时被限制"""
        # 在初始化代码中，soil_moisture会被限制在capacity内
        # 但参数验证会在之前执行，检查initial_soil_moisture不能超过soil_storage_max
        params = {
            "soil_storage_max": 100.0,
            "initial_soil_moisture": 100.0  # 等于容量是允许的
        }
        model = WETSPARunoff(params)
        # 应该等于容量
        assert model.soil_moisture == 100.0


# ==================== 参数验证测试 ====================

class TestWETSPAParameterValidation:
    """测试参数验证"""

    def test_validate_default_parameters(self):
        """测试默认参数验证"""
        model = WETSPARunoff({})
        model.validate_parameters()  # 不应抛出异常

    def test_validate_custom_parameters(self, default_wetspa_params):
        """测试自定义参数验证"""
        model = WETSPARunoff(default_wetspa_params)
        model.validate_parameters()  # 不应抛出异常

    def test_soil_storage_max_zero_invalid(self):
        """测试土壤蓄水量为0无效"""
        params = {"soil_storage_max": 0}
        with pytest.raises(ParameterValidationError, match="soil_storage_max"):
            model = WETSPARunoff(params)

    def test_soil_storage_max_negative_invalid(self):
        """测试负土壤蓄水量无效"""
        params = {"soil_storage_max": -10}
        with pytest.raises(ParameterValidationError, match="soil_storage_max"):
            model = WETSPARunoff(params)

    def test_infiltration_coefficient_out_of_range(self):
        """测试入渗系数超出范围"""
        # 测试<0
        params_low = {"infiltration_coefficient": -0.1}
        with pytest.raises(ParameterValidationError, match="infiltration_coefficient"):
            model = WETSPARunoff(params_low)

        # 测试>1
        params_high = {"infiltration_coefficient": 1.5}
        with pytest.raises(ParameterValidationError, match="infiltration_coefficient"):
            model = WETSPARunoff(params_high)

    def test_surface_runoff_coefficient_out_of_range(self):
        """测试地表径流系数超出范围"""
        params = {"surface_runoff_coefficient": -0.1}
        with pytest.raises(ParameterValidationError, match="surface_runoff_coefficient"):
            model = WETSPARunoff(params)

        params = {"surface_runoff_coefficient": 1.5}
        with pytest.raises(ParameterValidationError, match="surface_runoff_coefficient"):
            model = WETSPARunoff(params)

    def test_percolation_coefficient_out_of_range(self):
        """测试渗漏系数超出范围"""
        params = {"percolation_coefficient": -0.1}
        with pytest.raises(ParameterValidationError, match="percolation_coefficient"):
            model = WETSPARunoff(params)

        params = {"percolation_coefficient": 1.5}
        with pytest.raises(ParameterValidationError, match="percolation_coefficient"):
            model = WETSPARunoff(params)

    def test_baseflow_constant_out_of_range(self):
        """测试基流常数超出范围"""
        params = {"baseflow_constant": -0.1}
        with pytest.raises(ParameterValidationError, match="baseflow_constant"):
            model = WETSPARunoff(params)

        params = {"baseflow_constant": 1.5}
        with pytest.raises(ParameterValidationError, match="baseflow_constant"):
            model = WETSPARunoff(params)

    def test_initial_soil_moisture_out_of_range(self):
        """测试初始土壤水分超出范围"""
        # 负值
        params = {"initial_soil_moisture": -10}
        with pytest.raises(ParameterValidationError, match="initial_soil_moisture"):
            model = WETSPARunoff(params)

        # 超过容量（注意：初始化时会自动限制，但验证应该捕获）
        params = {"soil_storage_max": 100.0, "initial_soil_moisture": 150.0}
        with pytest.raises(ParameterValidationError, match="initial_soil_moisture"):
            model = WETSPARunoff(params)

    def test_initial_groundwater_negative_invalid(self):
        """测试负初始地下水无效"""
        params = {"initial_groundwater": -10}
        with pytest.raises(ParameterValidationError, match="initial_groundwater"):
            model = WETSPARunoff(params)

    def test_boundary_values_valid(self):
        """测试边界值有效"""
        params = {
            "infiltration_coefficient": 0.0,
            "surface_runoff_coefficient": 1.0,
            "percolation_coefficient": 0.0,
            "baseflow_constant": 1.0
        }
        model = WETSPARunoff(params)
        model.validate_parameters()


# ==================== 仿真测试 ====================

class TestWETSPASimulation:
    """测试仿真功能"""

    def test_simulate_zero_precipitation(self, sample_subbasin):
        """测试零降雨"""
        params = {
            "initial_soil_moisture": 0.0,
            "initial_groundwater": 0.0
        }
        model = WETSPARunoff(params)
        precipitation = [0.0] * 10
        runoff = model.simulate(sample_subbasin, precipitation)

        # 零降雨且零初始存储应该产生零径流
        assert len(runoff) == 10
        assert all(r == 0.0 for r in runoff)

    def test_simulate_constant_precipitation(self, default_wetspa_params, sample_subbasin):
        """测试恒定降雨"""
        model = WETSPARunoff(default_wetspa_params)
        precipitation = [50.0] * 20
        runoff = model.simulate(sample_subbasin, precipitation)

        # 径流应该存在
        assert len(runoff) == 20
        assert all(r >= 0 for r in runoff)

    def test_simulate_varying_precipitation(self, default_wetspa_params, sample_subbasin):
        """测试变化降雨"""
        model = WETSPARunoff(default_wetspa_params)
        precipitation = [30.0, 50.0, 70.0, 50.0, 30.0]
        runoff = model.simulate(sample_subbasin, precipitation)

        # 径流应该存在
        assert len(runoff) == 5
        assert all(r >= 0 for r in runoff)

    def test_simulate_heavy_precipitation(self, default_wetspa_params, sample_subbasin):
        """测试强降雨"""
        model = WETSPARunoff(default_wetspa_params)
        precipitation = [200.0] * 5
        runoff = model.simulate(sample_subbasin, precipitation)

        # 强降雨应该产生大径流
        assert len(runoff) == 5
        assert all(r > 1000 for r in runoff)


# ==================== 土壤水分机制测试 ====================

class TestWETSPASoilMoisture:
    """测试土壤水分机制"""

    def test_soil_moisture_increases_with_precipitation(self, default_wetspa_params, sample_subbasin):
        """测试降雨导致土壤水分增加"""
        model = WETSPARunoff(default_wetspa_params)
        initial_sm = model.soil_moisture

        precipitation = [30.0] * 5
        model.simulate(sample_subbasin, precipitation)

        # 土壤水分应该变化（可能增加或减少，取决于入渗和渗漏的平衡）
        assert model.soil_moisture != initial_sm

    def test_soil_moisture_not_exceed_capacity(self, default_wetspa_params, sample_subbasin):
        """测试土壤水分不超过容量"""
        model = WETSPARunoff(default_wetspa_params)

        # 大量降雨
        precipitation = [500.0] * 10
        model.simulate(sample_subbasin, precipitation)

        # 土壤水分不应超过容量
        assert model.soil_moisture <= model.capacity

    def test_soil_moisture_decreases_without_precipitation(self, sample_subbasin):
        """测试无降雨时土壤水分减少"""
        params = {
            "soil_storage_max": 200.0,
            "initial_soil_moisture": 150.0,
            "percolation_coefficient": 0.1  # 较高的渗漏系数
        }
        model = WETSPARunoff(params)
        initial_sm = model.soil_moisture

        precipitation = [0.0] * 10
        model.simulate(sample_subbasin, precipitation)

        # 土壤水分应该减少（由于渗漏）
        assert model.soil_moisture < initial_sm

    def test_infiltration_limited_by_capacity(self, sample_subbasin):
        """测试入渗受容量限制"""
        params = {
            "soil_storage_max": 100.0,
            "initial_soil_moisture": 95.0,  # 接近满
            "infiltration_coefficient": 1.0  # 100%入渗
        }
        model = WETSPARunoff(params)

        # 大降雨，但容量有限
        precipitation = [50.0]
        runoff = model.simulate(sample_subbasin, precipitation)

        # 土壤水分应该达到或接近容量
        assert model.soil_moisture <= model.capacity
        # 应该有地表径流（因为入渗能力不足）
        assert runoff[0] > 0


# ==================== 地下水机制测试 ====================

class TestWETSPAGroundwater:
    """测试地下水机制"""

    def test_groundwater_increases_with_percolation(self, default_wetspa_params, sample_subbasin):
        """测试渗漏导致地下水增加"""
        model = WETSPARunoff(default_wetspa_params)
        initial_gw = model.groundwater

        # 大量降雨会增加土壤水分，进而增加渗漏
        precipitation = [100.0] * 10
        model.simulate(sample_subbasin, precipitation)

        # 地下水应该增加
        assert model.groundwater > initial_gw

    def test_groundwater_decreases_with_baseflow(self, sample_subbasin):
        """测试基流导致地下水减少"""
        params = {
            "initial_groundwater": 100.0,
            "percolation_coefficient": 0.0,  # 无渗漏
            "baseflow_constant": 0.1  # 较高的基流常数
        }
        model = WETSPARunoff(params)
        initial_gw = model.groundwater

        # 无降雨，只有基流消耗地下水
        precipitation = [0.0] * 10
        model.simulate(sample_subbasin, precipitation)

        # 地下水应该减少
        assert model.groundwater < initial_gw

    def test_baseflow_proportional_to_groundwater(self, sample_subbasin):
        """测试基流与地下水成正比"""
        params_low = {"initial_groundwater": 50.0, "percolation_coefficient": 0.0}
        params_high = {"initial_groundwater": 150.0, "percolation_coefficient": 0.0}

        model_low = WETSPARunoff(params_low)
        model_high = WETSPARunoff(params_high)

        # 无降雨，只观察基流
        precipitation = [0.0]

        runoff_low = model_low.simulate(sample_subbasin, precipitation)
        runoff_high = model_high.simulate(sample_subbasin, precipitation)

        # 高地下水应该产生更多基流
        assert runoff_high[0] > runoff_low[0]


# ==================== 径流组成测试 ====================

class TestWETSPARunoffComponents:
    """测试径流组成"""

    def test_surface_runoff_from_excess(self, sample_subbasin):
        """测试超渗产生地表径流"""
        params = {
            "soil_storage_max": 100.0,
            "initial_soil_moisture": 99.0,  # 几乎满
            "infiltration_coefficient": 0.5,
            "percolation_coefficient": 0.0,  # 无渗漏
            "baseflow_constant": 0.0  # 无基流
        }
        model = WETSPARunoff(params)

        # 大降雨
        precipitation = [50.0]
        runoff = model.simulate(sample_subbasin, precipitation)

        # 应该有地表径流
        assert runoff[0] > 0

    def test_baseflow_from_groundwater(self, sample_subbasin):
        """测试地下水产生基流"""
        params = {
            "initial_soil_moisture": 0.0,
            "initial_groundwater": 100.0,
            "infiltration_coefficient": 0.0,  # 无入渗
            "percolation_coefficient": 0.0,  # 无渗漏
            "baseflow_constant": 0.05
        }
        model = WETSPARunoff(params)

        # 无降雨，只有基流
        precipitation = [0.0]
        runoff = model.simulate(sample_subbasin, precipitation)

        # 应该有基流
        expected_baseflow = 0.05 * 100.0 * sample_subbasin.area_km2
        assert abs(runoff[0] - expected_baseflow) < 1e-6

    def test_total_runoff_is_sum(self, default_wetspa_params, sample_subbasin):
        """测试总径流是地表径流和基流之和"""
        # 这个测试间接验证（无法直接访问组成部分）
        model = WETSPARunoff(default_wetspa_params)

        precipitation = [50.0]
        runoff = model.simulate(sample_subbasin, precipitation)

        # 总径流应该≥0
        assert runoff[0] >= 0


# ==================== 参数效果测试 ====================

class TestWETSPAParameterEffects:
    """测试参数效果"""

    def test_higher_infiltration_coefficient_less_surface_runoff(self, sample_subbasin):
        """测试更高的入渗系数产生更少地表径流"""
        params_low = {
            "infiltration_coefficient": 0.3,
            "initial_soil_moisture": 50.0,
            "percolation_coefficient": 0.0,
            "baseflow_constant": 0.0
        }
        params_high = {
            "infiltration_coefficient": 0.9,
            "initial_soil_moisture": 50.0,
            "percolation_coefficient": 0.0,
            "baseflow_constant": 0.0
        }

        model_low = WETSPARunoff(params_low)
        model_high = WETSPARunoff(params_high)

        precipitation = [100.0]

        runoff_low = model_low.simulate(sample_subbasin, precipitation)
        runoff_high = model_high.simulate(sample_subbasin, precipitation)

        # 高入渗系数应该产生更少地表径流（更多水入渗）
        assert runoff_low[0] > runoff_high[0]

    def test_higher_percolation_coefficient_more_baseflow(self, sample_subbasin):
        """测试更高的渗漏系数产生更多基流"""
        params_low = {
            "percolation_coefficient": 0.01,
            "baseflow_constant": 0.1
        }
        params_high = {
            "percolation_coefficient": 0.1,
            "baseflow_constant": 0.1
        }

        model_low = WETSPARunoff(params_low)
        model_high = WETSPARunoff(params_high)

        # 足够的降雨来增加土壤水分
        precipitation = [50.0] * 10

        runoff_low = model_low.simulate(sample_subbasin, precipitation)
        runoff_high = model_high.simulate(sample_subbasin, precipitation)

        # 高渗漏系数应该导致更多地下水，进而更多基流
        # 后期径流应该更高
        assert model_high.groundwater > model_low.groundwater

    def test_higher_baseflow_constant_more_baseflow(self, sample_subbasin):
        """测试更高的基流常数产生更多基流"""
        params_low = {"baseflow_constant": 0.02, "initial_groundwater": 100.0}
        params_high = {"baseflow_constant": 0.08, "initial_groundwater": 100.0}

        model_low = WETSPARunoff(params_low)
        model_high = WETSPARunoff(params_high)

        precipitation = [0.0]  # 无降雨，只看基流

        runoff_low = model_low.simulate(sample_subbasin, precipitation)
        runoff_high = model_high.simulate(sample_subbasin, precipitation)

        # 高基流常数应该产生更多基流
        assert runoff_high[0] > runoff_low[0]


# ==================== 边界情况测试 ====================

class TestWETSPAEdgeCases:
    """测试边界情况"""

    def test_empty_precipitation(self, default_wetspa_params, sample_subbasin):
        """测试空降雨序列"""
        model = WETSPARunoff(default_wetspa_params)
        precipitation = []
        runoff = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 0

    def test_single_timestep(self, default_wetspa_params, sample_subbasin):
        """测试单时间步"""
        model = WETSPARunoff(default_wetspa_params)
        precipitation = [50.0]
        runoff = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 1
        assert runoff[0] >= 0

    def test_very_small_capacity(self, sample_subbasin):
        """测试极小容量"""
        params = {"soil_storage_max": 1.0, "initial_soil_moisture": 0.5}
        model = WETSPARunoff(params)

        precipitation = [10.0] * 5
        runoff = model.simulate(sample_subbasin, precipitation)

        # 小容量应该快速饱和，产生大量地表径流
        assert all(r > 0 for r in runoff)

    def test_zero_coefficients(self, sample_subbasin):
        """测试零系数"""
        params = {
            "infiltration_coefficient": 0.0,
            "surface_runoff_coefficient": 0.0,
            "percolation_coefficient": 0.0,
            "baseflow_constant": 0.0
        }
        model = WETSPARunoff(params)

        precipitation = [50.0] * 5
        runoff = model.simulate(sample_subbasin, precipitation)

        # 所有系数为0，应该无径流
        assert all(r == 0.0 for r in runoff)

    def test_all_coefficients_one(self, sample_subbasin):
        """测试所有系数为1"""
        params = {
            "infiltration_coefficient": 1.0,
            "surface_runoff_coefficient": 1.0,
            "percolation_coefficient": 1.0,
            "baseflow_constant": 1.0
        }
        model = WETSPARunoff(params)

        precipitation = [50.0] * 5
        runoff = model.simulate(sample_subbasin, precipitation)

        # 应该产生径流
        assert all(r >= 0 for r in runoff)


# ==================== 流域面积效果测试 ====================

class TestWETSPAAreaEffect:
    """测试流域面积效果"""

    def test_larger_area_larger_runoff(self, default_wetspa_params):
        """测试更大的流域面积产生更大的径流"""
        class SmallBasin:
            area_km2 = 10.0

        class LargeBasin:
            area_km2 = 1000.0

        model1 = WETSPARunoff(default_wetspa_params)
        model2 = WETSPARunoff(default_wetspa_params)

        precipitation = [50.0] * 5

        runoff_small = model1.simulate(SmallBasin(), precipitation)
        runoff_large = model2.simulate(LargeBasin(), precipitation)

        # 径流应该与面积成正比
        ratio = runoff_large[0] / runoff_small[0]
        expected_ratio = 1000.0 / 10.0
        assert abs(ratio - expected_ratio) < 1e-6

    def test_area_linearly_scales_output(self, default_wetspa_params):
        """测试面积线性缩放输出"""
        class Basin100:
            area_km2 = 100.0

        class Basin200:
            area_km2 = 200.0

        precipitation = [50.0] * 5

        model1 = WETSPARunoff(default_wetspa_params)
        model2 = WETSPARunoff(default_wetspa_params)

        runoff_100 = model1.simulate(Basin100(), precipitation)
        runoff_200 = model2.simulate(Basin200(), precipitation)

        # 200km²的径流应该是100km²的2倍
        for i in range(len(precipitation)):
            assert abs(runoff_200[i] / runoff_100[i] - 2.0) < 1e-6


# ==================== 长时间序列测试 ====================

class TestWETSPALongSeries:
    """测试长时间序列"""

    def test_long_constant_precipitation(self, default_wetspa_params, sample_subbasin):
        """测试长恒定降雨"""
        model = WETSPARunoff(default_wetspa_params)
        precipitation = [40.0] * 100
        runoff = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 100
        assert all(r >= 0 for r in runoff)

    def test_drought_recovery(self, sample_subbasin):
        """测试干旱后恢复"""
        params = {
            "initial_soil_moisture": 100.0,
            "initial_groundwater": 50.0
        }
        model = WETSPARunoff(params)

        # 干旱期（无降雨）
        drought = [0.0] * 20
        # 降雨期
        rain = [60.0] * 20

        precipitation = drought + rain
        runoff = model.simulate(sample_subbasin, precipitation)

        # 降雨期径流应该比干旱期大
        drought_avg = np.mean(runoff[:20])
        rain_avg = np.mean(runoff[20:])
        assert rain_avg > drought_avg

    def test_multiple_rain_events(self, default_wetspa_params, sample_subbasin):
        """测试多次降雨事件"""
        model = WETSPARunoff(default_wetspa_params)

        # 多次降雨-干旱循环
        precipitation = [60.0, 40.0, 20.0, 0.0, 0.0, 80.0, 50.0, 20.0, 0.0, 0.0]
        runoff = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 10
        assert all(r >= 0 for r in runoff)


# ==================== 数值稳定性测试 ====================

class TestWETSPANumericalStability:
    """测试数值稳定性"""

    def test_no_negative_runoff(self, default_wetspa_params, sample_subbasin):
        """测试径流始终非负"""
        model = WETSPARunoff(default_wetspa_params)

        # 随机降雨
        precipitation = [float(np.random.rand() * 100) for _ in range(50)]
        runoff = model.simulate(sample_subbasin, precipitation)

        assert all(r >= 0 for r in runoff)

    def test_no_negative_storage(self, default_wetspa_params, sample_subbasin):
        """测试存储始终非负"""
        model = WETSPARunoff(default_wetspa_params)

        precipitation = [float(np.random.rand() * 100) for _ in range(50)]
        model.simulate(sample_subbasin, precipitation)

        assert model.soil_moisture >= 0
        assert model.groundwater >= 0

    def test_very_long_series(self, default_wetspa_params, sample_subbasin):
        """测试非常长的时间序列"""
        model = WETSPARunoff(default_wetspa_params)
        precipitation = [40.0] * 1000
        runoff = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 1000
        assert all(r >= 0 for r in runoff)
        # 应该数值稳定（不应该有NaN或Inf）
        assert all(np.isfinite(r) for r in runoff)
        assert np.isfinite(model.soil_moisture)
        assert np.isfinite(model.groundwater)
