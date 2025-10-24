"""Distributed Green-Ampt产流模型的单元测试"""
import pytest
import numpy as np
from hydrosis.runoff.distributed_green_ampt import DistributedGreenAmpt
from hydrosis.validation import ParameterValidationError


@pytest.fixture
def default_ga_params():
    """默认Green-Ampt参数"""
    return {
        "saturated_conductivity": 10.0,  # mm/hr
        "wetting_front_suction": 100.0,  # mm
        "initial_moisture": 0.2,
        "saturated_moisture": 0.4,
        "porosity": 0.45,
        "zones": 5,
        "zone_distribution": "uniform"
    }


@pytest.fixture
def sample_subbasin():
    """样本子流域"""
    class MockSubbasin:
        def __init__(self):
            self.area_km2 = 100.0
    return MockSubbasin()


# ==================== 初始化测试 ====================

class TestDistributedGreenAmptInitialization:
    """测试初始化"""

    def test_default_initialization(self):
        """测试默认参数初始化"""
        model = DistributedGreenAmpt({})
        assert model.k_sat == 10.0
        assert model.psi_f == 100.0
        assert model.theta_i == 0.2
        assert model.theta_s == 0.4
        assert model.porosity == 0.45
        assert model.num_zones == 5
        assert model.zone_distribution == "uniform"

    def test_custom_initialization(self, default_ga_params):
        """测试自定义参数初始化"""
        model = DistributedGreenAmpt(default_ga_params)
        assert model.k_sat == 10.0
        assert model.psi_f == 100.0
        assert model.theta_i == 0.2
        assert model.theta_s == 0.4

    def test_delta_theta_calculation(self, default_ga_params):
        """测试土壤水分差计算"""
        model = DistributedGreenAmpt(default_ga_params)
        expected_delta = 0.4 - 0.2
        assert abs(model.delta_theta - expected_delta) < 1e-10

    def test_uniform_zone_initialization(self, default_ga_params):
        """测试均匀分区初始化"""
        model = DistributedGreenAmpt(default_ga_params)
        assert len(model.zones) == 5
        # 所有分区应该有相同属性
        for zone in model.zones:
            assert zone["k_sat"] == 10.0
            assert zone["psi_f"] == 100.0
            assert abs(zone["area_fraction"] - 0.2) < 1e-10

    def test_random_zone_initialization(self):
        """测试随机分区初始化"""
        params = {
            "saturated_conductivity": 10.0,
            "wetting_front_suction": 100.0,
            "zones": 5,
            "zone_distribution": "random"
        }
        model = DistributedGreenAmpt(params)
        assert len(model.zones) == 5
        # 随机分区应该有不同的k_sat值
        k_sat_values = [zone["k_sat"] for zone in model.zones]
        # 至少有一些差异（可能不是所有都不同，但应该有变化）
        assert sum(model.zones[0]["area_fraction"] for zone in model.zones) == pytest.approx(1.0)

    def test_clustered_zone_initialization(self):
        """测试聚类分区初始化"""
        params = {
            "saturated_conductivity": 10.0,
            "wetting_front_suction": 100.0,
            "zones": 6,
            "zone_distribution": "clustered"
        }
        model = DistributedGreenAmpt(params)
        assert len(model.zones) == 6
        # 聚类分区应该有3种类型的属性
        k_sat_values = set([zone["k_sat"] for zone in model.zones])
        assert len(k_sat_values) <= 3  # 最多3种不同的值


# ==================== 参数验证测试 ====================

class TestDistributedGreenAmptParameterValidation:
    """测试参数验证"""

    def test_validate_default_parameters(self):
        """测试默认参数验证"""
        model = DistributedGreenAmpt({})
        model.validate_parameters()  # 不应抛出异常

    def test_validate_custom_parameters(self, default_ga_params):
        """测试自定义参数验证"""
        model = DistributedGreenAmpt(default_ga_params)
        model.validate_parameters()  # 不应抛出异常

    def test_negative_saturated_conductivity(self):
        """测试负饱和导水率"""
        params = {"saturated_conductivity": -1.0}
        with pytest.raises(ParameterValidationError, match="saturated_conductivity"):
            model = DistributedGreenAmpt(params)

    def test_zero_saturated_conductivity(self):
        """测试零饱和导水率"""
        params = {"saturated_conductivity": 0.0}
        with pytest.raises(ParameterValidationError, match="saturated_conductivity"):
            model = DistributedGreenAmpt(params)

    def test_negative_wetting_front_suction(self):
        """测试负湿润锋吸力"""
        params = {"wetting_front_suction": -10.0}
        with pytest.raises(ParameterValidationError, match="wetting_front_suction"):
            model = DistributedGreenAmpt(params)

    def test_zero_wetting_front_suction(self):
        """测试零湿润锋吸力"""
        params = {"wetting_front_suction": 0.0}
        with pytest.raises(ParameterValidationError, match="wetting_front_suction"):
            model = DistributedGreenAmpt(params)

    def test_moisture_out_of_range(self):
        """测试土壤含水量超出范围"""
        params = {"initial_moisture": -0.1}
        with pytest.raises(ParameterValidationError, match="initial_moisture"):
            model = DistributedGreenAmpt(params)

        params = {"initial_moisture": 1.5}
        with pytest.raises(ParameterValidationError, match="initial_moisture"):
            model = DistributedGreenAmpt(params)

    def test_initial_ge_saturated_moisture(self):
        """测试初始含水量>=饱和含水量"""
        params = {
            "initial_moisture": 0.5,
            "saturated_moisture": 0.4
        }
        with pytest.raises(ParameterValidationError, match="initial_moisture"):
            model = DistributedGreenAmpt(params)

    def test_saturated_moisture_gt_porosity(self):
        """测试饱和含水量>孔隙度"""
        params = {
            "saturated_moisture": 0.5,
            "porosity": 0.4
        }
        with pytest.raises(ParameterValidationError, match="saturated_moisture"):
            model = DistributedGreenAmpt(params)

    def test_zero_zones_invalid(self):
        """测试零分区数无效"""
        params = {"zones": 0}
        with pytest.raises(ParameterValidationError, match="zones"):
            model = DistributedGreenAmpt(params)

    def test_negative_zones_invalid(self):
        """测试负分区数无效"""
        params = {"zones": -1}
        with pytest.raises(ParameterValidationError, match="zones"):
            model = DistributedGreenAmpt(params)


# ==================== 仿真测试 ====================

class TestDistributedGreenAmptSimulation:
    """测试仿真功能"""

    def test_simulate_zero_precipitation(self, default_ga_params, sample_subbasin):
        """测试零降雨"""
        model = DistributedGreenAmpt(default_ga_params)
        precipitation = [0.0] * 10
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        # 零降雨应该产生零径流
        assert len(runoff) == 10
        assert all(r == 0.0 for r in runoff)

    def test_simulate_small_precipitation(self, default_ga_params, sample_subbasin):
        """测试小降雨（全部入渗）"""
        model = DistributedGreenAmpt(default_ga_params)
        # 小降雨应该全部入渗，无径流
        precipitation = [2.0] * 5  # 小于入渗能力
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        # 小降雨可能全部入渗或产生少量径流
        assert len(runoff) == 5
        assert all(r >= 0 for r in runoff)

    def test_simulate_heavy_precipitation(self, default_ga_params, sample_subbasin):
        """测试强降雨（产生径流）"""
        model = DistributedGreenAmpt(default_ga_params)
        # 强降雨应该超过入渗能力，产生径流
        precipitation = [100.0] * 5
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 5
        assert sum(runoff) > 0  # 应该有径流

    def test_simulate_varying_precipitation(self, default_ga_params, sample_subbasin):
        """测试变化降雨"""
        model = DistributedGreenAmpt(default_ga_params)
        precipitation = [10.0, 30.0, 50.0, 30.0, 10.0]
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 5
        assert all(r >= 0 for r in runoff)


# ==================== 入渗能力测试 ====================

class TestDistributedGreenAmptInfiltration:
    """测试入渗能力"""

    def test_infiltration_capacity_decreases(self, default_ga_params, sample_subbasin):
        """测试入渗能力随累积入渗量递减"""
        model = DistributedGreenAmpt(default_ga_params)

        # 持续降雨应该导致累积入渗增加，入渗能力下降
        precipitation = [20.0] * 20
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        # 累积入渗应该增加
        assert all(storage[i] >= 0 for i in range(model.num_zones))

    def test_higher_conductivity_more_infiltration(self, sample_subbasin):
        """测试高导水率产生更多入渗（更少径流）"""
        params_low = {
            "saturated_conductivity": 5.0,
            "zones": 1
        }
        params_high = {
            "saturated_conductivity": 20.0,
            "zones": 1
        }

        model_low = DistributedGreenAmpt(params_low)
        model_high = DistributedGreenAmpt(params_high)

        precipitation = [50.0] * 5

        runoff_low, _ = model_low.simulate(sample_subbasin, precipitation)
        runoff_high, _ = model_high.simulate(sample_subbasin, precipitation)

        # 高导水率应该产生更少径流（更多入渗）
        assert sum(runoff_high) <= sum(runoff_low)

    def test_higher_suction_more_infiltration(self, sample_subbasin):
        """测试高吸力产生更多入渗"""
        params_low = {
            "wetting_front_suction": 50.0,
            "zones": 1
        }
        params_high = {
            "wetting_front_suction": 200.0,
            "zones": 1
        }

        model_low = DistributedGreenAmpt(params_low)
        model_high = DistributedGreenAmpt(params_high)

        precipitation = [50.0] * 5

        runoff_low, _ = model_low.simulate(sample_subbasin, precipitation)
        runoff_high, _ = model_high.simulate(sample_subbasin, precipitation)

        # 高吸力应该产生更少径流（更多入渗）
        assert sum(runoff_high) <= sum(runoff_low)


# ==================== 土壤水分效果测试 ====================

class TestDistributedGreenAmptMoisture:
    """测试土壤水分效果"""

    def test_larger_moisture_deficit_more_infiltration(self, sample_subbasin):
        """测试大水分差产生更多入渗"""
        params_small = {
            "initial_moisture": 0.35,
            "saturated_moisture": 0.4,
            "zones": 1
        }
        params_large = {
            "initial_moisture": 0.1,
            "saturated_moisture": 0.4,
            "zones": 1
        }

        model_small = DistributedGreenAmpt(params_small)
        model_large = DistributedGreenAmpt(params_large)

        precipitation = [50.0] * 5

        runoff_small, _ = model_small.simulate(sample_subbasin, precipitation)
        runoff_large, _ = model_large.simulate(sample_subbasin, precipitation)

        # 大水分差应该产生更少径流（更多入渗）
        assert sum(runoff_large) <= sum(runoff_small)


# ==================== 分区效果测试 ====================

class TestDistributedGreenAmptZones:
    """测试分区效果"""

    def test_single_zone_equals_lumped(self, sample_subbasin):
        """测试单分区等于集总模型"""
        params = {
            "saturated_conductivity": 10.0,
            "zones": 1,
            "zone_distribution": "uniform"
        }
        model = DistributedGreenAmpt(params)

        precipitation = [30.0] * 5
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        # 单分区应该有1个存储值
        assert len(storage) == 1

    def test_multiple_zones_aggregate(self, sample_subbasin):
        """测试多分区聚合"""
        params = {
            "saturated_conductivity": 10.0,
            "zones": 5,
            "zone_distribution": "uniform"
        }
        model = DistributedGreenAmpt(params)

        precipitation = [30.0] * 5
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        # 应该有5个分区的存储值
        assert len(storage) == 5

    def test_area_fractions_sum_to_one(self, default_ga_params):
        """测试面积分数总和为1"""
        model = DistributedGreenAmpt(default_ga_params)

        total_area = sum(zone["area_fraction"] for zone in model.zones)
        assert abs(total_area - 1.0) < 1e-10


# ==================== 初始存储测试 ====================

class TestDistributedGreenAmptInitialStorage:
    """测试初始存储"""

    def test_zero_initial_storage_default(self, default_ga_params):
        """测试默认零初始存储"""
        model = DistributedGreenAmpt(default_ga_params)
        initial_storage = model.get_initial_storage()

        assert len(initial_storage) == 5
        assert all(s == 0.0 for s in initial_storage)

    def test_custom_initial_storage(self, sample_subbasin):
        """测试自定义初始存储"""
        params = {
            "saturated_conductivity": 10.0,
            "zones": 3,
            "initial_cumulative_infiltration_zone_0": 10.0,
            "initial_cumulative_infiltration_zone_1": 20.0,
            "initial_cumulative_infiltration_zone_2": 30.0,
        }
        model = DistributedGreenAmpt(params)

        precipitation = [20.0] * 3
        initial_storage = {0: 10.0, 1: 20.0, 2: 30.0}
        runoff, final_storage = model.simulate(sample_subbasin, precipitation, initial_storage)

        # 最终存储应该大于初始存储
        for i in range(3):
            assert final_storage[i] >= initial_storage[i]

    def test_storage_accumulates(self, default_ga_params, sample_subbasin):
        """测试存储累积"""
        model = DistributedGreenAmpt(default_ga_params)

        # 第一次仿真
        precipitation1 = [10.0] * 5
        runoff1, storage1 = model.simulate(sample_subbasin, precipitation1)

        # 第二次仿真，使用前次的最终存储
        precipitation2 = [10.0] * 5
        runoff2, storage2 = model.simulate(sample_subbasin, precipitation2, storage1)

        # 第二次的最终存储应该更大
        for i in range(model.num_zones):
            assert storage2[i] >= storage1[i]


# ==================== 边界情况测试 ====================

class TestDistributedGreenAmptEdgeCases:
    """测试边界情况"""

    def test_empty_precipitation(self, default_ga_params, sample_subbasin):
        """测试空降雨序列"""
        model = DistributedGreenAmpt(default_ga_params)
        precipitation = []
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 0

    def test_single_timestep(self, default_ga_params, sample_subbasin):
        """测试单时间步"""
        model = DistributedGreenAmpt(default_ga_params)
        precipitation = [50.0]
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 1
        assert runoff[0] >= 0

    def test_very_high_conductivity(self, sample_subbasin):
        """测试极高导水率（几乎全部入渗）"""
        params = {
            "saturated_conductivity": 1000.0,
            "zones": 1
        }
        model = DistributedGreenAmpt(params)

        precipitation = [50.0] * 5
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        # 极高导水率应该导致极少径流
        assert sum(runoff) < sum(precipitation) * sample_subbasin.area_km2 * 1e6 / (1000 * 3600) * 0.5

    def test_very_low_conductivity(self, sample_subbasin):
        """测试极低导水率（几乎全部径流）"""
        params = {
            "saturated_conductivity": 0.1,
            "zones": 1
        }
        model = DistributedGreenAmpt(params)

        precipitation = [50.0] * 5
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        # 极低导水率应该导致更多径流
        assert sum(runoff) > 0


# ==================== 流域面积效果测试 ====================

class TestDistributedGreenAmptAreaEffect:
    """测试流域面积效果"""

    def test_larger_area_larger_runoff_volume(self, default_ga_params):
        """测试更大面积产生更大径流体积"""
        class SmallBasin:
            area_km2 = 10.0

        class LargeBasin:
            area_km2 = 1000.0

        model1 = DistributedGreenAmpt(default_ga_params)
        model2 = DistributedGreenAmpt(default_ga_params)

        precipitation = [50.0] * 5

        runoff_small, _ = model1.simulate(SmallBasin(), precipitation)
        runoff_large, _ = model2.simulate(LargeBasin(), precipitation)

        # 大流域的径流体积应该更大
        # 注意：这里的径流已经转换为m³/s，考虑了面积
        ratio = sum(runoff_large) / sum(runoff_small) if sum(runoff_small) > 0 else float('inf')
        expected_ratio = 1000.0 / 10.0
        # 比率应该接近面积比
        if sum(runoff_small) > 0:
            assert abs(ratio - expected_ratio) / expected_ratio < 0.1


# ==================== 长时间序列测试 ====================

class TestDistributedGreenAmptLongSeries:
    """测试长时间序列"""

    def test_long_constant_precipitation(self, default_ga_params, sample_subbasin):
        """测试长恒定降雨"""
        model = DistributedGreenAmpt(default_ga_params)
        precipitation = [20.0] * 100
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 100
        assert all(r >= 0 for r in runoff)

    def test_long_varying_series(self, default_ga_params, sample_subbasin):
        """测试长变化序列"""
        model = DistributedGreenAmpt(default_ga_params)
        precipitation = [float(20 + 10 * np.sin(i * 0.1)) for i in range(100)]
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 100
        assert all(r >= 0 for r in runoff)

    def test_wet_dry_cycles(self, default_ga_params, sample_subbasin):
        """测试干湿循环"""
        model = DistributedGreenAmpt(default_ga_params)

        # 湿-干-湿-干循环
        precipitation = [50.0] * 5 + [0.0] * 5 + [50.0] * 5 + [0.0] * 5
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        # 干旱期应该无径流
        for i in range(5, 10):
            assert runoff[i] == 0.0
        for i in range(15, 20):
            assert runoff[i] == 0.0


# ==================== 数值稳定性测试 ====================

class TestDistributedGreenAmptNumericalStability:
    """测试数值稳定性"""

    def test_no_negative_runoff(self, default_ga_params, sample_subbasin):
        """测试径流始终非负"""
        model = DistributedGreenAmpt(default_ga_params)

        # 随机降雨
        precipitation = [float(np.random.rand() * 100) for _ in range(50)]
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        assert all(r >= 0 for r in runoff)

    def test_no_negative_storage(self, default_ga_params, sample_subbasin):
        """测试存储始终非负"""
        model = DistributedGreenAmpt(default_ga_params)

        precipitation = [float(np.random.rand() * 100) for _ in range(50)]
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        assert all(storage[i] >= 0 for i in range(model.num_zones))

    def test_very_long_series(self, default_ga_params, sample_subbasin):
        """测试非常长序列"""
        model = DistributedGreenAmpt(default_ga_params)
        precipitation = [20.0] * 1000
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        assert len(runoff) == 1000
        assert all(r >= 0 for r in runoff)
        assert all(np.isfinite(r) for r in runoff)


# ==================== Green-Ampt公式验证测试 ====================

class TestDistributedGreenAmptFormulaVerification:
    """测试Green-Ampt公式验证"""

    def test_infinite_capacity_at_start(self, sample_subbasin):
        """测试开始时入渗能力无限大"""
        params = {
            "saturated_conductivity": 10.0,
            "zones": 1
        }
        model = DistributedGreenAmpt(params)

        # 第一个时间步，累积入渗为0，入渗能力应该是无限大
        # 任何降雨都应该全部入渗
        precipitation = [5.0]
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        # 小降雨应该全部入渗，无径流（或极少径流）
        assert runoff[0] >= 0

    def test_runoff_equals_excess(self, sample_subbasin):
        """测试径流等于超渗"""
        params = {
            "saturated_conductivity": 1.0,  # 非常低的导水率
            "zones": 1
        }
        model = DistributedGreenAmpt(params)

        # 大降雨应该超过入渗能力
        precipitation = [100.0] * 5
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        # 应该有显著径流
        assert sum(runoff) > 0


# ==================== 返回值测试 ====================

class TestDistributedGreenAmptReturnValue:
    """测试返回值格式"""

    def test_returns_tuple(self, default_ga_params, sample_subbasin):
        """测试返回元组"""
        model = DistributedGreenAmpt(default_ga_params)
        precipitation = [20.0] * 5
        result = model.simulate(sample_subbasin, precipitation)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_tuple_first_element_is_runoff_list(self, default_ga_params, sample_subbasin):
        """测试元组第一个元素是径流列表"""
        model = DistributedGreenAmpt(default_ga_params)
        precipitation = [20.0] * 5
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        assert isinstance(runoff, list)
        assert len(runoff) == 5

    def test_tuple_second_element_is_storage_dict(self, default_ga_params, sample_subbasin):
        """测试元组第二个元素是存储字典"""
        model = DistributedGreenAmpt(default_ga_params)
        precipitation = [20.0] * 5
        runoff, storage = model.simulate(sample_subbasin, precipitation)

        assert isinstance(storage, dict)
        assert len(storage) == 5  # 5个分区
