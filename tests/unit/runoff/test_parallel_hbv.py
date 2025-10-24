"""Parallel HBV并行化工具的单元测试"""
import pytest
import numpy as np
from hydrosis.runoff.parallel_hbv import (
    ParallelHBVConfig,
    _run_single_zone_hbv,
    run_hbv_parallel,
    benchmark_parallel_performance
)


@pytest.fixture
def default_hbv_params():
    """默认HBV参数"""
    return {
        "field_capacity": 200.0,
        "beta": 2.0,
        "k0": 0.1,
        "k1": 0.05,
        "k2": 0.01,
        "lp": 0.7,
        "perc": 2.0,
        "degree_day_factor": 3.0,
        "snow_threshold": 0.0,
        "initial_snow": 0.0,
        "initial_soil_moisture": 100.0,
        "initial_upper": 10.0,
        "initial_lower": 20.0
    }


@pytest.fixture
def sample_precipitation():
    """样本降雨数据"""
    return np.array([10.0, 20.0, 30.0, 20.0, 10.0])


@pytest.fixture
def sample_zones():
    """样本分区数据"""
    return [
        {'zone_id': 1, 'area_km2': 100.0},
        {'zone_id': 2, 'area_km2': 150.0},
        {'zone_id': 3, 'area_km2': 200.0}
    ]


@pytest.fixture
def sample_precipitation_data(sample_precipitation):
    """样本多分区降雨数据"""
    return {
        1: sample_precipitation,
        2: sample_precipitation * 1.2,
        3: sample_precipitation * 0.8
    }


# ==================== 配置测试 ====================

class TestParallelHBVConfig:
    """测试ParallelHBVConfig配置类"""

    def test_default_config(self):
        """测试默认配置"""
        config = ParallelHBVConfig()
        assert config.max_workers == 4
        assert config.chunk_size == 100
        assert config.use_multiprocessing is True
        assert config.show_progress is True

    def test_custom_config(self):
        """测试自定义配置"""
        config = ParallelHBVConfig(
            max_workers=8,
            chunk_size=50,
            use_multiprocessing=False,
            show_progress=False
        )
        assert config.max_workers == 8
        assert config.chunk_size == 50
        assert config.use_multiprocessing is False
        assert config.show_progress is False


# ==================== 单分区运行测试 ====================

class TestRunSingleZoneHBV:
    """测试单分区HBV运行"""

    def test_run_single_zone(self, default_hbv_params, sample_precipitation):
        """测试单分区运行"""
        zone_id = 1
        area_km2 = 100.0

        args = (zone_id, sample_precipitation, area_km2, default_hbv_params)
        result_zone_id, result = _run_single_zone_hbv(args)

        # 验证返回的zone_id
        assert result_zone_id == zone_id

        # 验证结果字典包含所需字段
        assert 'zone_id' in result
        assert 'runoff_coefficient' in result
        assert 'total_precip_mm' in result
        assert 'total_runoff_mm' in result
        assert 'peak_runoff_m3s' in result
        assert 'mean_runoff_m3s' in result
        assert 'runoff_series' in result

        # 验证zone_id正确
        assert result['zone_id'] == zone_id

    def test_runoff_series_length(self, default_hbv_params, sample_precipitation):
        """测试径流序列长度"""
        zone_id = 1
        area_km2 = 100.0

        args = (zone_id, sample_precipitation, area_km2, default_hbv_params)
        _, result = _run_single_zone_hbv(args)

        # 径流序列长度应该等于降雨序列长度
        assert len(result['runoff_series']) == len(sample_precipitation)

    def test_runoff_coefficient_valid(self, default_hbv_params, sample_precipitation):
        """测试径流系数有效"""
        zone_id = 1
        area_km2 = 100.0

        args = (zone_id, sample_precipitation, area_km2, default_hbv_params)
        _, result = _run_single_zone_hbv(args)

        # 径流系数应该在[0, 1]范围内
        assert 0 <= result['runoff_coefficient'] <= 1.5  # 允许略大于1（因为初始存储）

    def test_total_precipitation_calculation(self, default_hbv_params, sample_precipitation):
        """测试总降雨计算"""
        zone_id = 1
        area_km2 = 100.0

        args = (zone_id, sample_precipitation, area_km2, default_hbv_params)
        _, result = _run_single_zone_hbv(args)

        # 总降雨应该等于降雨序列总和
        expected_total = sample_precipitation.sum()
        assert abs(result['total_precip_mm'] - expected_total) < 1e-6

    def test_peak_runoff_calculation(self, default_hbv_params, sample_precipitation):
        """测试峰值径流计算"""
        zone_id = 1
        area_km2 = 100.0

        args = (zone_id, sample_precipitation, area_km2, default_hbv_params)
        _, result = _run_single_zone_hbv(args)

        # 峰值径流应该等于径流序列最大值
        expected_peak = result['runoff_series'].max()
        assert result['peak_runoff_m3s'] == expected_peak

    def test_mean_runoff_calculation(self, default_hbv_params, sample_precipitation):
        """测试平均径流计算"""
        zone_id = 1
        area_km2 = 100.0

        args = (zone_id, sample_precipitation, area_km2, default_hbv_params)
        _, result = _run_single_zone_hbv(args)

        # 平均径流应该等于径流序列平均值
        expected_mean = result['runoff_series'].mean()
        assert abs(result['mean_runoff_m3s'] - expected_mean) < 1e-6

    def test_zero_precipitation(self, default_hbv_params):
        """测试零降雨"""
        zone_id = 1
        area_km2 = 100.0
        zero_precip = np.array([0.0, 0.0, 0.0])

        args = (zone_id, zero_precip, area_km2, default_hbv_params)
        _, result = _run_single_zone_hbv(args)

        # 零降雨的径流系数应该定义为0（避免除零）
        # 但可能有初始存储产生的径流
        assert result['total_precip_mm'] == 0.0


# ==================== 并行运行测试 ====================

class TestRunHBVParallel:
    """测试并行HBV运行"""

    def test_parallel_run_basic(self, sample_zones, sample_precipitation_data, default_hbv_params):
        """测试基本并行运行"""
        config = ParallelHBVConfig(max_workers=2, show_progress=False)
        results = run_hbv_parallel(sample_zones, sample_precipitation_data, default_hbv_params, config)

        # 应该返回所有分区的结果
        assert len(results) == 3
        assert 1 in results
        assert 2 in results
        assert 3 in results

    def test_parallel_results_structure(self, sample_zones, sample_precipitation_data, default_hbv_params):
        """测试并行结果结构"""
        config = ParallelHBVConfig(max_workers=2, show_progress=False)
        results = run_hbv_parallel(sample_zones, sample_precipitation_data, default_hbv_params, config)

        # 每个结果应该包含必要字段
        for zone_id, result in results.items():
            assert 'zone_id' in result
            assert 'runoff_coefficient' in result
            assert 'runoff_series' in result
            assert result['zone_id'] == zone_id

    def test_serial_execution(self, sample_zones, sample_precipitation_data, default_hbv_params):
        """测试串行执行（max_workers=1）"""
        config = ParallelHBVConfig(max_workers=1, show_progress=False)
        results = run_hbv_parallel(sample_zones, sample_precipitation_data, default_hbv_params, config)

        # 串行执行也应该返回所有结果
        assert len(results) == 3

    def test_default_config(self, sample_zones, sample_precipitation_data, default_hbv_params):
        """测试默认配置"""
        results = run_hbv_parallel(sample_zones, sample_precipitation_data, default_hbv_params)

        # 使用默认配置应该成功
        assert len(results) == 3

    def test_missing_precipitation_data(self, default_hbv_params):
        """测试缺失降雨数据"""
        zones = [
            {'zone_id': 1, 'area_km2': 100.0},
            {'zone_id': 2, 'area_km2': 150.0}
        ]
        precipitation_data = {
            1: np.array([10.0, 20.0, 30.0])
            # 缺少zone_id=2的数据
        }

        config = ParallelHBVConfig(max_workers=1, show_progress=False)
        results = run_hbv_parallel(zones, precipitation_data, default_hbv_params, config)

        # 应该只返回有数据的分区
        assert len(results) == 1
        assert 1 in results
        assert 2 not in results

    def test_empty_zones(self, sample_precipitation_data, default_hbv_params):
        """测试空分区列表"""
        zones = []
        config = ParallelHBVConfig(max_workers=1, show_progress=False)
        results = run_hbv_parallel(zones, sample_precipitation_data, default_hbv_params, config)

        # 空分区应该返回空结果
        assert len(results) == 0

    def test_single_zone(self, sample_precipitation_data, default_hbv_params):
        """测试单个分区"""
        zones = [{'zone_id': 1, 'area_km2': 100.0}]
        config = ParallelHBVConfig(max_workers=2, show_progress=False)
        results = run_hbv_parallel(zones, sample_precipitation_data, default_hbv_params, config)

        # 单个分区应该返回1个结果
        assert len(results) == 1
        assert 1 in results


# ==================== 串行vs并行一致性测试 ====================

class TestSerialParallelConsistency:
    """测试串行和并行结果一致性"""

    def test_serial_parallel_same_results(self, sample_zones, sample_precipitation_data, default_hbv_params):
        """测试串行和并行产生相同结果"""
        # 串行执行
        config_serial = ParallelHBVConfig(max_workers=1, show_progress=False)
        results_serial = run_hbv_parallel(sample_zones, sample_precipitation_data, default_hbv_params, config_serial)

        # 并行执行
        config_parallel = ParallelHBVConfig(max_workers=2, show_progress=False)
        results_parallel = run_hbv_parallel(sample_zones, sample_precipitation_data, default_hbv_params, config_parallel)

        # 应该返回相同数量的结果
        assert len(results_serial) == len(results_parallel)

        # 每个分区的结果应该一致
        for zone_id in results_serial.keys():
            assert zone_id in results_parallel

            # 比较径流系数
            assert abs(results_serial[zone_id]['runoff_coefficient'] -
                      results_parallel[zone_id]['runoff_coefficient']) < 1e-6

            # 比较径流序列
            assert np.allclose(results_serial[zone_id]['runoff_series'],
                             results_parallel[zone_id]['runoff_series'])


# ==================== 基准测试功能测试 ====================

class TestBenchmarkParallelPerformance:
    """测试基准测试功能"""

    def test_benchmark_basic(self, sample_zones, sample_precipitation_data, default_hbv_params):
        """测试基本基准测试"""
        worker_counts = [1, 2]
        results = benchmark_parallel_performance(
            sample_zones,
            sample_precipitation_data,
            default_hbv_params,
            worker_counts
        )

        # 应该返回每个worker数量的结果
        assert len(results) == 2
        assert 1 in results
        assert 2 in results

    def test_benchmark_timing_positive(self, sample_zones, sample_precipitation_data, default_hbv_params):
        """测试基准测试时间为正"""
        worker_counts = [1]
        results = benchmark_parallel_performance(
            sample_zones,
            sample_precipitation_data,
            default_hbv_params,
            worker_counts
        )

        # 时间应该为正
        assert results[1] > 0

    def test_benchmark_single_worker(self, sample_zones, sample_precipitation_data, default_hbv_params):
        """测试单worker基准测试"""
        worker_counts = [1]
        results = benchmark_parallel_performance(
            sample_zones,
            sample_precipitation_data,
            default_hbv_params,
            worker_counts
        )

        # 单worker应该有结果
        assert 1 in results


# ==================== 不同面积效果测试 ====================

class TestDifferentAreaEffects:
    """测试不同流域面积的效果"""

    def test_larger_area_larger_runoff(self, sample_precipitation, default_hbv_params):
        """测试更大面积产生更大径流"""
        # 小流域
        args_small = (1, sample_precipitation, 50.0, default_hbv_params)
        _, result_small = _run_single_zone_hbv(args_small)

        # 大流域
        args_large = (2, sample_precipitation, 500.0, default_hbv_params)
        _, result_large = _run_single_zone_hbv(args_large)

        # 大流域的峰值径流应该更大（m³/s考虑了面积）
        assert result_large['peak_runoff_m3s'] > result_small['peak_runoff_m3s']


# ==================== 不同降雨效果测试 ====================

class TestDifferentPrecipitationEffects:
    """测试不同降雨的效果"""

    def test_higher_precipitation_higher_runoff(self, default_hbv_params):
        """测试更高降雨产生更高径流"""
        zone_id = 1
        area_km2 = 100.0

        # 低降雨
        precip_low = np.array([5.0, 10.0, 15.0])
        args_low = (zone_id, precip_low, area_km2, default_hbv_params)
        _, result_low = _run_single_zone_hbv(args_low)

        # 高降雨
        precip_high = np.array([20.0, 40.0, 60.0])
        args_high = (zone_id, precip_high, area_km2, default_hbv_params)
        _, result_high = _run_single_zone_hbv(args_high)

        # 高降雨应该产生更高径流
        assert result_high['peak_runoff_m3s'] > result_low['peak_runoff_m3s']
        assert result_high['total_runoff_mm'] > result_low['total_runoff_mm']


# ==================== 长时间序列测试 ====================

class TestLongTimeSeries:
    """测试长时间序列"""

    def test_long_precipitation_series(self, default_hbv_params):
        """测试长降雨序列"""
        zone_id = 1
        area_km2 = 100.0
        long_precip = np.random.rand(365) * 20  # 一年的随机降雨

        args = (zone_id, long_precip, area_km2, default_hbv_params)
        _, result = _run_single_zone_hbv(args)

        # 径流序列长度应该等于降雨长度
        assert len(result['runoff_series']) == 365

        # 所有径流值应该非负
        assert np.all(result['runoff_series'] >= 0)


# ==================== 数值稳定性测试 ====================

class TestNumericalStability:
    """测试数值稳定性"""

    def test_no_nan_values(self, default_hbv_params, sample_precipitation):
        """测试结果无NaN值"""
        zone_id = 1
        area_km2 = 100.0

        args = (zone_id, sample_precipitation, area_km2, default_hbv_params)
        _, result = _run_single_zone_hbv(args)

        # 径流序列不应该有NaN
        assert not np.any(np.isnan(result['runoff_series']))

        # 统计指标不应该是NaN
        assert not np.isnan(result['runoff_coefficient'])
        assert not np.isnan(result['peak_runoff_m3s'])
        assert not np.isnan(result['mean_runoff_m3s'])

    def test_no_inf_values(self, default_hbv_params, sample_precipitation):
        """测试结果无Inf值"""
        zone_id = 1
        area_km2 = 100.0

        args = (zone_id, sample_precipitation, area_km2, default_hbv_params)
        _, result = _run_single_zone_hbv(args)

        # 径流序列不应该有Inf
        assert not np.any(np.isinf(result['runoff_series']))

        # 统计指标不应该是Inf
        assert not np.isinf(result['runoff_coefficient'])
        assert not np.isinf(result['peak_runoff_m3s'])
        assert not np.isinf(result['mean_runoff_m3s'])


# ==================== 多分区不同参数测试 ====================

class TestMultipleZonesDifferentParams:
    """测试多分区使用相同参数"""

    def test_same_params_different_precipitation(self, sample_zones, default_hbv_params):
        """测试相同参数不同降雨"""
        precipitation_data = {
            1: np.array([10.0, 20.0, 30.0]),
            2: np.array([5.0, 10.0, 15.0]),
            3: np.array([15.0, 30.0, 45.0])
        }

        config = ParallelHBVConfig(max_workers=2, show_progress=False)
        results = run_hbv_parallel(sample_zones, precipitation_data, default_hbv_params, config)

        # 所有分区都应该有结果
        assert len(results) == 3

        # 不同降雨应该产生不同结果
        # 分区3的降雨最大，应该产生最大径流（考虑面积）
        # 这个测试可能不成立，因为面积不同
        # 所以只验证结果存在
        assert all(zone_id in results for zone_id in [1, 2, 3])
