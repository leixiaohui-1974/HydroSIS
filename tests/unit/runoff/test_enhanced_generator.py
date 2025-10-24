"""Enhanced Generator径流生成器的单元测试"""
import pytest
import numpy as np
from hydrosis.runoff.enhanced_generator import EnhancedRunoffGenerator


@pytest.fixture
def default_generator():
    """默认生成器"""
    return EnhancedRunoffGenerator()


@pytest.fixture
def sample_precipitation():
    """样本降雨数据"""
    return np.array([10.0, 20.0, 30.0, 20.0, 10.0])


# ==================== 初始化测试 ====================

class TestEnhancedGeneratorInitialization:
    """测试初始化"""

    def test_default_initialization(self):
        """测试默认参数初始化"""
        gen = EnhancedRunoffGenerator()
        assert gen.soil_capacity == 300.0
        assert gen.soil_beta == 2.0
        assert gen.fast_threshold == 0.7
        assert gen.fast_ratio + gen.inter_ratio + gen.base_ratio == pytest.approx(1.0)

    def test_custom_initialization(self):
        """测试自定义参数初始化"""
        gen = EnhancedRunoffGenerator(
            soil_capacity=400.0,
            soil_beta=3.0,
            fast_threshold=0.8
        )
        assert gen.soil_capacity == 400.0
        assert gen.soil_beta == 3.0
        assert gen.fast_threshold == 0.8

    def test_ratio_normalization(self):
        """测试分配比例自动归一化"""
        gen = EnhancedRunoffGenerator(
            fast_ratio=0.5,
            inter_ratio=0.3,
            base_ratio=0.3  # 总和>1
        )
        # 应该自动归一化到1
        assert gen.fast_ratio + gen.inter_ratio + gen.base_ratio == pytest.approx(1.0)

    def test_initial_state(self):
        """测试初始状态"""
        gen = EnhancedRunoffGenerator(
            initial_soil=150.0,
            initial_fast=5.0,
            initial_inter=10.0,
            initial_base=20.0
        )
        assert gen.soil_storage == 150.0
        assert gen.fast_storage == 5.0
        assert gen.inter_storage == 10.0
        assert gen.base_storage == 20.0


# ==================== 单步模拟测试 ====================

class TestEnhancedGeneratorStep:
    """测试单步模拟"""

    def test_step_zero_precipitation(self, default_generator):
        """测试零降雨"""
        runoff, components = default_generator.step(0.0)

        # 零降雨应该有低径流（来自水库衰减）
        assert runoff >= 0
        assert 'fast_runoff' in components
        assert 'inter_runoff' in components
        assert 'base_runoff' in components

    def test_step_positive_precipitation(self, default_generator):
        """测试正常降雨"""
        runoff, components = default_generator.step(10.0)

        # 应该产生径流
        assert runoff > 0
        assert components['fast_runoff'] >= 0
        assert components['inter_runoff'] >= 0
        assert components['base_runoff'] >= 0

    def test_step_components_sum(self, default_generator):
        """测试径流分量总和"""
        runoff, components = default_generator.step(10.0)

        # 总径流应该等于各分量之和
        total_components = (components['fast_runoff'] +
                          components['inter_runoff'] +
                          components['base_runoff'])
        assert abs(runoff - total_components) < 1e-6

    def test_step_soil_storage_updates(self, default_generator):
        """测试土壤储量更新"""
        initial_soil = default_generator.soil_storage
        runoff, components = default_generator.step(10.0)

        # 土壤储量应该变化
        assert components['soil_storage'] != initial_soil

    def test_step_soil_saturation_range(self, default_generator):
        """测试土壤饱和度范围"""
        runoff, components = default_generator.step(10.0)

        # 饱和度应该在[0, 1]范围内
        assert 0 <= components['soil_saturation'] <= 1.0

    def test_effective_precipitation_with_et(self):
        """测试蒸散发扣除"""
        gen = EnhancedRunoffGenerator(et_rate=5.0)
        runoff, components = gen.step(10.0)

        # 有效降雨应该是降雨减去蒸散发
        assert components['effective_precip'] == 5.0

    def test_zero_effective_precipitation(self):
        """测试有效降雨为零"""
        gen = EnhancedRunoffGenerator(et_rate=15.0)
        runoff, components = gen.step(10.0)

        # 当ET>降雨时，有效降雨应该为0
        assert components['effective_precip'] == 0.0


# ==================== 线性水库测试 ====================

class TestLinearReservoirStep:
    """测试线性水库"""

    def test_linear_reservoir_basic(self, default_generator):
        """测试线性水库基本功能"""
        storage = 10.0
        input_rate = 5.0
        k = 0.1
        dt = 1.0

        new_storage = default_generator._linear_reservoir_step(storage, input_rate, k, dt)

        # 新储量应该是正值
        assert new_storage > 0

    def test_linear_reservoir_zero_input(self, default_generator):
        """测试零输入衰减"""
        storage = 10.0
        input_rate = 0.0
        k = 0.1
        dt = 1.0

        new_storage = default_generator._linear_reservoir_step(storage, input_rate, k, dt)

        # 零输入时储量应该减少
        assert new_storage < storage

    def test_linear_reservoir_non_negative(self, default_generator):
        """测试储量非负"""
        storage = 1.0
        input_rate = 0.0
        k = 10.0  # 大衰减系数
        dt = 10.0

        new_storage = default_generator._linear_reservoir_step(storage, input_rate, k, dt)

        # 储量不应该为负
        assert new_storage >= 0


# ==================== 生成完整序列测试 ====================

class TestEnhancedGeneratorGenerate:
    """测试生成完整序列"""

    def test_generate_basic(self, default_generator, sample_precipitation):
        """测试基本生成"""
        area_km2 = 100.0
        runoff_m3s, stats = default_generator.generate(sample_precipitation, area_km2)

        # 径流序列长度应该等于降雨长度
        assert len(runoff_m3s) == len(sample_precipitation)

        # 统计信息应该包含必要字段
        assert 'total_precip_mm' in stats
        assert 'total_runoff_mm' in stats
        assert 'runoff_coefficient' in stats

    def test_generate_runoff_coefficient_range(self, default_generator, sample_precipitation):
        """测试径流系数范围"""
        area_km2 = 100.0
        runoff_m3s, stats = default_generator.generate(sample_precipitation, area_km2)

        # 径流系数通常在[0, 1]范围内
        # 但可能略大于1（初始存储释放）
        assert 0 <= stats['runoff_coefficient'] <= 1.5

    def test_generate_with_components(self, default_generator, sample_precipitation):
        """测试返回分量序列"""
        area_km2 = 100.0
        runoff_m3s, stats = default_generator.generate(
            sample_precipitation, area_km2, return_components=True
        )

        # 应该包含分量时间序列
        assert 'components' in stats
        assert 'fast_runoff' in stats['components']
        assert 'inter_runoff' in stats['components']
        assert 'base_runoff' in stats['components']

    def test_generate_total_equals_components(self, default_generator, sample_precipitation):
        """测试总径流等于分量之和"""
        area_km2 = 100.0
        runoff_m3s, stats = default_generator.generate(
            sample_precipitation, area_km2, return_components=True
        )

        # 计算分量总和
        components_total = (stats['components']['fast_runoff'] +
                          stats['components']['inter_runoff'] +
                          stats['components']['base_runoff'])

        # 转换为m³/s
        components_total_m3s = components_total * area_km2 / 3.6

        # 应该接近总径流
        assert np.allclose(runoff_m3s, components_total_m3s)

    def test_generate_longer_series(self, default_generator):
        """测试更长序列"""
        precip = np.random.rand(365) * 10  # 一年的随机降雨
        area_km2 = 100.0
        runoff_m3s, stats = default_generator.generate(precip, area_km2)

        # 应该成功生成
        assert len(runoff_m3s) == 365


# ==================== 重置功能测试 ====================

class TestEnhancedGeneratorReset:
    """测试重置功能"""

    def test_reset_soil_storage(self, default_generator, sample_precipitation):
        """测试重置土壤储量"""
        area_km2 = 100.0
        # 先运行一次改变状态
        default_generator.generate(sample_precipitation, area_km2)

        # 重置土壤储量
        default_generator.reset(initial_soil=150.0)
        assert default_generator.soil_storage == 150.0

    def test_reset_all_storages(self, default_generator):
        """测试重置所有储量"""
        default_generator.reset(
            initial_soil=100.0,
            initial_fast=2.0,
            initial_inter=5.0,
            initial_base=10.0
        )

        assert default_generator.soil_storage == 100.0
        assert default_generator.fast_storage == 2.0
        assert default_generator.inter_storage == 5.0
        assert default_generator.base_storage == 10.0

    def test_reset_partial(self, default_generator):
        """测试部分重置"""
        original_fast = default_generator.fast_storage
        default_generator.reset(initial_soil=200.0)

        # 只重置了土壤储量
        assert default_generator.soil_storage == 200.0
        # 其他储量不变
        assert default_generator.fast_storage == original_fast


# ==================== 参数获取测试 ====================

class TestEnhancedGeneratorGetParameters:
    """测试获取参数"""

    def test_get_parameters(self, default_generator):
        """测试获取所有参数"""
        params = default_generator.get_parameters()

        # 应该包含所有主要参数
        assert 'soil_capacity' in params
        assert 'soil_beta' in params
        assert 'fast_threshold' in params
        assert 'k_fast' in params
        assert 'k_inter' in params
        assert 'k_base' in params

    def test_parameters_match_attributes(self, default_generator):
        """测试参数与属性匹配"""
        params = default_generator.get_parameters()

        assert params['soil_capacity'] == default_generator.soil_capacity
        assert params['soil_beta'] == default_generator.soil_beta
        assert params['k_fast'] == default_generator.k_fast


# ==================== 参数效果测试 ====================

class TestParameterEffects:
    """测试参数效果"""

    def test_higher_soil_beta_more_nonlinear(self):
        """测试更高beta产生更非线性响应"""
        precip = np.array([20.0] * 10)
        area_km2 = 100.0

        gen_low_beta = EnhancedRunoffGenerator(soil_beta=1.0)
        gen_high_beta = EnhancedRunoffGenerator(soil_beta=3.0)

        runoff_low, _ = gen_low_beta.generate(precip, area_km2)
        runoff_high, _ = gen_high_beta.generate(precip, area_km2)

        # 两者都应该产生径流
        assert runoff_low.sum() > 0
        assert runoff_high.sum() > 0

    def test_higher_k_faster_recession(self):
        """测试更高k产生更快衰退"""
        gen_low_k = EnhancedRunoffGenerator(k_fast=0.1, initial_fast=20.0)
        gen_high_k = EnhancedRunoffGenerator(k_fast=0.5, initial_fast=20.0)

        # 无降雨，观察衰退
        precip = np.array([0.0] * 10)
        area_km2 = 100.0

        runoff_low, _ = gen_low_k.generate(precip, area_km2)
        runoff_high, _ = gen_high_k.generate(precip, area_km2)

        # 高k应该衰退更快，后期径流更低
        assert runoff_low[-1] > runoff_high[-1]

    def test_larger_soil_capacity_more_storage(self):
        """测试更大容量提供更多蓄水"""
        precip = np.array([30.0] * 10)
        area_km2 = 100.0

        gen_small = EnhancedRunoffGenerator(soil_capacity=200.0)
        gen_large = EnhancedRunoffGenerator(soil_capacity=500.0)

        runoff_small, _ = gen_small.generate(precip, area_km2)
        runoff_large, _ = gen_large.generate(precip, area_km2)

        # 大容量应该产生更少径流（更多蓄水）
        assert runoff_small.sum() >= runoff_large.sum()


# ==================== 物理过程测试 ====================

class TestPhysicalProcesses:
    """测试物理过程"""

    def test_soil_saturation_affects_runoff(self):
        """测试土壤饱和度影响径流"""
        # 干燥土壤
        gen_dry = EnhancedRunoffGenerator(initial_soil=50.0, soil_capacity=300.0)

        # 湿润土壤
        gen_wet = EnhancedRunoffGenerator(initial_soil=250.0, soil_capacity=300.0)

        # 相同降雨
        runoff_dry, comp_dry = gen_dry.step(20.0)
        runoff_wet, comp_wet = gen_wet.step(20.0)

        # 湿润土壤应该产生更多径流
        assert runoff_wet > runoff_dry

    def test_fast_threshold_effect(self):
        """测试快速径流阈值效果"""
        gen = EnhancedRunoffGenerator(
            fast_threshold=0.8,
            initial_soil=150.0,  # 50% saturation
            soil_capacity=300.0
        )

        runoff, components = gen.step(20.0)

        # 未超过阈值时，快速径流应该较少
        assert components['fast_runoff'] >= 0

    def test_evapotranspiration_depletes_soil(self):
        """测试蒸散发消耗土壤水分"""
        gen = EnhancedRunoffGenerator(et_rate=5.0, initial_soil=200.0)
        initial_soil = gen.soil_storage

        # 无降雨，只有蒸散发
        gen.step(0.0)

        # 土壤水分应该减少
        assert gen.soil_storage < initial_soil


# ==================== 数值稳定性测试 ====================

class TestNumericalStability:
    """测试数值稳定性"""

    def test_no_negative_runoff(self, default_generator):
        """测试径流非负"""
        precip = np.random.rand(100) * 20
        area_km2 = 100.0
        runoff_m3s, _ = default_generator.generate(precip, area_km2)

        # 所有径流值应该非负
        assert np.all(runoff_m3s >= 0)

    def test_no_nan_values(self, default_generator, sample_precipitation):
        """测试无NaN值"""
        area_km2 = 100.0
        runoff_m3s, stats = default_generator.generate(sample_precipitation, area_km2)

        # 径流序列不应该有NaN
        assert not np.any(np.isnan(runoff_m3s))

        # 统计值不应该是NaN
        assert not np.isnan(stats['runoff_coefficient'])

    def test_no_inf_values(self, default_generator, sample_precipitation):
        """测试无Inf值"""
        area_km2 = 100.0
        runoff_m3s, stats = default_generator.generate(sample_precipitation, area_km2)

        # 径流序列不应该有Inf
        assert not np.any(np.isinf(runoff_m3s))

    def test_very_long_series(self, default_generator):
        """测试超长序列"""
        precip = np.random.rand(10000) * 10
        area_km2 = 100.0
        runoff_m3s, stats = default_generator.generate(precip, area_km2)

        # 应该成功生成
        assert len(runoff_m3s) == 10000
        assert np.all(np.isfinite(runoff_m3s))


# ==================== 边界情况测试 ====================

class TestEdgeCases:
    """测试边界情况"""

    def test_zero_precipitation_series(self, default_generator):
        """测试全零降雨"""
        precip = np.array([0.0] * 10)
        area_km2 = 100.0
        runoff_m3s, stats = default_generator.generate(precip, area_km2)

        # 应该有低径流（来自初始存储衰减）
        assert np.all(runoff_m3s >= 0)

    def test_very_high_precipitation(self, default_generator):
        """测试极强降雨"""
        precip = np.array([1000.0] * 5)
        area_km2 = 100.0
        runoff_m3s, stats = default_generator.generate(precip, area_km2)

        # 应该产生大径流
        assert runoff_m3s.max() > 0

    def test_empty_precipitation_series(self, default_generator):
        """测试空降雨序列"""
        precip = np.array([])
        area_km2 = 100.0
        runoff_m3s, stats = default_generator.generate(precip, area_km2)

        # 应该返回空序列
        assert len(runoff_m3s) == 0

    def test_single_timestep(self, default_generator):
        """测试单时间步"""
        precip = np.array([20.0])
        area_km2 = 100.0
        runoff_m3s, stats = default_generator.generate(precip, area_km2)

        # 应该返回单个值
        assert len(runoff_m3s) == 1
        assert runoff_m3s[0] >= 0


# ==================== 流域面积效果测试 ====================

class TestAreaEffect:
    """测试流域面积效果"""

    def test_larger_area_larger_runoff_volume(self, default_generator, sample_precipitation):
        """测试更大面积产生更大径流体积"""
        runoff_small, _ = default_generator.generate(sample_precipitation, 50.0)

        # 重置状态
        default_generator.reset(initial_soil=150.0, initial_fast=5.0,
                               initial_inter=10.0, initial_base=20.0)

        runoff_large, _ = default_generator.generate(sample_precipitation, 500.0)

        # 大流域的径流体积应该更大（m³/s）
        assert runoff_large.sum() > runoff_small.sum()
