"""Pytest配置文件

提供共享fixtures和测试配置
"""
import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_precipitation_df():
    """示例降雨数据DataFrame"""
    dates = pd.date_range('2024-01-01', periods=120, freq='h')
    
    # 创建5个站点的降雨数据
    data = {
        'station_1': np.random.gamma(2, 2, size=120),
        'station_2': np.random.gamma(2, 2, size=120),
        'station_3': np.random.gamma(2, 2, size=120),
        'station_4': np.random.gamma(2, 2, size=120),
        'station_5': np.random.gamma(2, 2, size=120),
    }
    
    df = pd.DataFrame(data, index=dates)
    
    # 标准化总降雨量
    total_precip = 500  # mm
    df = df / df.sum().mean() * total_precip
    
    return df


@pytest.fixture
def sample_zones():
    """示例参数分区"""
    return [
        {'zone_id': 1, 'area_km2': 100.5},
        {'zone_id': 2, 'area_km2': 150.2},
        {'zone_id': 3, 'area_km2': 200.8},
        {'zone_id': 4, 'area_km2': 120.3},
    ]


@pytest.fixture
def sample_hbv_params():
    """示例HBV参数"""
    return {
        'FC': 300,
        'BETA': 2.0,
        'LP': 0.7,
        'K0': 0.1,
        'K1': 0.05,
        'K2': 0.01,
        'PERC': 2.0,
        'UZL': 50.0,
        'TT': 0.0,
        'CFMAX': 3.0,
        'CFR': 0.05,
        'CWH': 0.1
    }
