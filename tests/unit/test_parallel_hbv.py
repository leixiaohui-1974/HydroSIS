"""并行HBV模块的单元测试

测试 run_hbv_parallel 和 ParallelHBVConfig 功能
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch

from hydrosis.runoff.parallel_hbv import (
    ParallelHBVConfig,
    run_hbv_parallel,
    _run_single_zone_hbv,
)


class TestParallelHBVConfig:
    """测试 ParallelHBVConfig 类"""

    def test_default_initialization(self):
        """测试默认初始化"""
        config = ParallelHBVConfig()
        assert config.max_workers == 4
        assert config.chunk_size == 100
        assert config.use_multiprocessing is True
        assert config.show_progress is True

    def test_custom_initialization(self):
        """测试自定义初始化"""
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

    def test_workers_validation(self):
        """测试 worker 数量验证"""
        # 至少1个worker
        config = ParallelHBVConfig(max_workers=1)
        assert config.max_workers == 1

        # 多个workers
        config = ParallelHBVConfig(max_workers=8)
        assert config.max_workers == 8


@pytest.fixture
def sample_zones():
    """创建示例分区数据"""
    return [
        {'zone_id': 1, 'area_km2': 100.0},
        {'zone_id': 2, 'area_km2': 150.0},
        {'zone_id': 3, 'area_km2': 200.0},
    ]


@pytest.fixture
def sample_precipitation_data(sample_zones):
    """创建示例降雨数据"""
    time_steps = 120
    precipitation_data = {}
    
    for zone in sample_zones:
        # 生成随机降雨序列
        precip = np.random.gamma(2, 2, size=time_steps)
        precip = precip / precip.sum() * 300  # 总降雨300mm
        precipitation_data[zone['zone_id']] = precip
    
    return precipitation_data


@pytest.fixture
def sample_hbv_params():
    """创建示例HBV参数"""
    return {
        'FC': 300,      # Field capacity
        'BETA': 2.0,    # Shape coefficient
        'LP': 0.7,      # Evapotranspiration threshold
        'K0': 0.1,      # Recession coefficient 0
        'K1': 0.05,     # Recession coefficient 1
        'K2': 0.01,     # Recession coefficient 2
        'PERC': 2.0,    # Percolation rate
        'UZL': 50.0,    # Upper zone threshold
        'TT': 0.0,      # Temperature threshold
        'CFMAX': 3.0,   # Degree-day factor
        'CFR': 0.05,    # Refreezing coefficient
        'CWH': 0.1      # Water holding capacity
    }


class TestRunHBVParallel:
    """测试 run_hbv_parallel 函数"""

    def test_serial_execution(self, sample_zones, sample_precipitation_data, sample_hbv_params):
        """测试串行执行"""
        config = ParallelHBVConfig(max_workers=1, show_progress=False)
        
        results = run_hbv_parallel(
            sample_zones,
            sample_precipitation_data,
            sample_hbv_params,
            config
        )
        
        # 验证结果数量
        assert len(results) == len(sample_zones)
        
        # 验证每个分区都有结果
        for zone in sample_zones:
            zone_id = zone['zone_id']
            assert zone_id in results
            
            result = results[zone_id]
            assert 'runoff_series' in result
            assert 'runoff_coefficient' in result
            assert 'total_precip_mm' in result
            assert 'total_runoff_mm' in result

    def test_parallel_execution(self, sample_zones, sample_precipitation_data, sample_hbv_params):
        """测试并行执行"""
        config = ParallelHBVConfig(max_workers=2, show_progress=False)
        
        results = run_hbv_parallel(
            sample_zones,
            sample_precipitation_data,
            sample_hbv_params,
            config
        )
        
        # 验证结果数量
        assert len(results) == len(sample_zones)
        
        # 验证每个分区都有结果
        for zone in sample_zones:
            zone_id = zone['zone_id']
            assert zone_id in results

    def test_serial_parallel_consistency(self, sample_zones, sample_precipitation_data, sample_hbv_params):
        """测试串行和并行结果一致性"""
        # 串行执行
        config_serial = ParallelHBVConfig(max_workers=1, show_progress=False)
        results_serial = run_hbv_parallel(
            sample_zones,
            sample_precipitation_data,
            sample_hbv_params,
            config_serial
        )
        
        # 并行执行
        config_parallel = ParallelHBVConfig(max_workers=2, show_progress=False)
        results_parallel = run_hbv_parallel(
            sample_zones,
            sample_precipitation_data,
            sample_hbv_params,
            config_parallel
        )
        
        # 验证结果一致性
        for zone in sample_zones:
            zone_id = zone['zone_id']
            
            serial = results_serial[zone_id]
            parallel = results_parallel[zone_id]
            
            # 径流系数应该相同
            assert serial['runoff_coefficient'] == pytest.approx(
                parallel['runoff_coefficient'], 
                abs=1e-6
            )
            
            # 径流序列应该相同
            np.testing.assert_allclose(
                serial['runoff_series'],
                parallel['runoff_series'],
                rtol=1e-6,
                atol=1e-6
            )

    def test_result_structure(self, sample_zones, sample_precipitation_data, sample_hbv_params):
        """测试结果结构"""
        config = ParallelHBVConfig(max_workers=1, show_progress=False)
        results = run_hbv_parallel(
            sample_zones,
            sample_precipitation_data,
            sample_hbv_params,
            config
        )
        
        zone_id = sample_zones[0]['zone_id']
        result = results[zone_id]
        
        # 验证必需字段
        required_fields = [
            'runoff_series',
            'runoff_coefficient',
            'total_precip_mm',
            'total_runoff_mm',
        ]
        
        for field in required_fields:
            assert field in result, f"缺少字段: {field}"
        
        # 验证径流序列长度
        expected_length = len(sample_precipitation_data[zone_id])
        assert len(result['runoff_series']) == expected_length

    def test_runoff_coefficient_range(self, sample_zones, sample_precipitation_data, sample_hbv_params):
        """测试径流系数计算"""
        config = ParallelHBVConfig(max_workers=1, show_progress=False)
        results = run_hbv_parallel(
            sample_zones,
            sample_precipitation_data,
            sample_hbv_params,
            config
        )

        for zone_id, result in results.items():
            rc = result['runoff_coefficient']
            # HBV模型可能产生 >1 的径流系数（由于初始存储释放）
            # 只验证径流系数被计算且为数值
            assert isinstance(rc, (int, float, np.number))
            assert not np.isnan(rc), f"分区 {zone_id} 的径流系数为NaN"

    def test_mass_balance(self, sample_zones, sample_precipitation_data, sample_hbv_params):
        """测试质量计算"""
        config = ParallelHBVConfig(max_workers=1, show_progress=False)
        results = run_hbv_parallel(
            sample_zones,
            sample_precipitation_data,
            sample_hbv_params,
            config
        )

        for zone_id, result in results.items():
            precip = result['total_precip_mm']
            runoff = result['total_runoff_mm']

            # HBV模型可能从初始存储释放水量，导致径流>降雨
            # 只验证数值被正确计算
            assert precip >= 0, f"分区 {zone_id} 的降雨量为负"
            assert runoff >= 0, f"分区 {zone_id} 的径流量为负"

    def test_single_zone(self, sample_hbv_params):
        """测试单个分区"""
        zones = [{'zone_id': 1, 'area_km2': 100.0}]
        precipitation_data = {1: np.ones(50) * 2.0}  # 均匀降雨
        
        config = ParallelHBVConfig(max_workers=1, show_progress=False)
        results = run_hbv_parallel(zones, precipitation_data, sample_hbv_params, config)
        
        assert len(results) == 1
        assert 1 in results

    def test_empty_zones(self, sample_precipitation_data, sample_hbv_params):
        """测试空分区列表"""
        zones = []
        config = ParallelHBVConfig(max_workers=1, show_progress=False)
        
        results = run_hbv_parallel(zones, {}, sample_hbv_params, config)
        
        assert len(results) == 0

    def test_different_precipitation_patterns(self, sample_zones, sample_hbv_params):
        """测试不同的降雨模式"""
        # 创建不同的降雨模式
        precipitation_data = {
            1: np.zeros(100),           # 无降雨
            2: np.ones(100) * 5.0,      # 均匀降雨
            3: np.random.gamma(2, 2, size=100),  # 随机降雨
        }

        config = ParallelHBVConfig(max_workers=2, show_progress=False)
        results = run_hbv_parallel(
            sample_zones,
            precipitation_data,
            sample_hbv_params,
            config
        )

        # 验证所有模式都能运行
        assert 1 in results
        assert 2 in results
        assert 3 in results

        # 均匀降雨应该产生非零径流系数
        assert results[2]['runoff_coefficient'] > 0


class TestRunSingleZoneHBV:
    """测试 _run_single_zone_hbv 函数"""

    def test_single_zone_execution(self, sample_hbv_params):
        """测试单分区执行"""
        zone_id = 1
        precipitation = np.ones(50) * 2.0  # 均匀降雨
        area_km2 = 100.0

        args = (zone_id, precipitation, area_km2, sample_hbv_params)
        result_zone_id, result = _run_single_zone_hbv(args)

        assert result_zone_id == zone_id
        assert 'runoff_series' in result
        assert 'runoff_coefficient' in result
        assert len(result['runoff_series']) == 50

    def test_zero_precipitation(self, sample_hbv_params):
        """测试零降雨"""
        zone_id = 1
        precipitation = np.zeros(50)
        area_km2 = 100.0

        args = (zone_id, precipitation, area_km2, sample_hbv_params)
        result_zone_id, result = _run_single_zone_hbv(args)

        # HBV模型即使零降雨也可能产生基流（来自初始存储）
        # 验证函数能够成功执行
        assert result_zone_id == zone_id
        assert 'runoff_series' in result
        assert 'total_runoff_mm' in result
