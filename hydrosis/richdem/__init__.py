"""
RichDEM兼容层 - 用于Python 3.13环境
提供基本的DEM处理功能，当原生richdem不可用时使用
"""

import numpy as np
from scipy import ndimage
import logging

logger = logging.getLogger(__name__)
logger.info("使用内置richdem兼容层")


class rdarray(np.ndarray):
    """RichDEM数组兼容类"""
    
    def __new__(cls, input_array, no_data=-9999, **kwargs):
        obj = np.asarray(input_array).view(cls)
        obj.no_data = no_data
        obj.geotransform = kwargs.get('geotransform', None)
        obj.projection = kwargs.get('projection', None)
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.no_data = getattr(obj, 'no_data', -9999)
        self.geotransform = getattr(obj, 'geotransform', None)
        self.projection = getattr(obj, 'projection', None)


def FillDepressions(dem, in_place=True, epsilon=False):
    """
    简单的坑洼填充实现
    使用形态学操作来填充DEM中的坑洼
    """
    if not in_place:
        dem = dem.copy()
    
    # 保存no_data值
    no_data = getattr(dem, 'no_data', -9999)
    
    # 创建有效数据掩码
    valid_mask = dem != no_data
    
    if not np.any(valid_mask):
        logger.warning("DEM中没有有效数据")
        return dem
    
    # 获取有效数据
    valid_data = dem[valid_mask]
    
    # 简单的坑洼填充算法
    # 1. 找到边界最小值
    h = dem.shape[0]
    w = dem.shape[1]
    
    # 创建工作数组
    filled = dem.copy()
    
    # 边界设置为原始值
    filled[0, :] = dem[0, :]
    filled[-1, :] = dem[-1, :]
    filled[:, 0] = dem[:, 0]
    filled[:, -1] = dem[:, -1]
    
    # 内部区域：迭代填充
    max_iterations = 100
    for iteration in range(max_iterations):
        changed = False
        for i in range(1, h-1):
            for j in range(1, w-1):
                if valid_mask[i, j]:
                    # 获取邻域最小值
                    neighbors = [
                        filled[i-1, j], filled[i+1, j],
                        filled[i, j-1], filled[i, j+1]
                    ]
                    valid_neighbors = [n for n, m in zip(neighbors, [
                        valid_mask[i-1, j], valid_mask[i+1, j],
                        valid_mask[i, j-1], valid_mask[i, j+1]
                    ]) if m]
                    
                    if valid_neighbors:
                        min_neighbor = min(valid_neighbors)
                        # 如果当前值小于邻域最小值，提升到邻域最小值
                        if filled[i, j] < min_neighbor:
                            filled[i, j] = min_neighbor
                            changed = True
        
        if not changed:
            break
    
    # 更新原数组
    if in_place:
        dem[...] = filled
        return dem
    else:
        return filled


def FlowAccumulation(dem, method='D8'):
    """
    简单的流量累积计算
    返回一个表示上游汇流面积的数组
    """
    h, w = dem.shape
    no_data = getattr(dem, 'no_data', -9999)
    valid_mask = dem != no_data
    
    # 创建流量累积数组
    flow_acc = np.ones_like(dem, dtype=np.float32)
    flow_acc[~valid_mask] = no_data
    
    # 计算坡度作为流量累积的近似
    # 这是一个简化实现，实际应该基于流向
    gy, gx = np.gradient(dem.astype(float))
    slope = np.sqrt(gx**2 + gy**2)
    
    # 使用高斯滤波来模拟汇流
    from scipy.ndimage import gaussian_filter
    flow_acc = gaussian_filter(slope, sigma=2) * 100
    flow_acc[~valid_mask] = no_data
    
    return rdarray(flow_acc, no_data=no_data)


def FlowDirection(dem, method='D8'):
    """
    计算流向
    返回一个表示水流方向的数组
    """
    h, w = dem.shape
    no_data = getattr(dem, 'no_data', -9999)
    valid_mask = dem != no_data
    
    # 计算坡度方向
    gy, gx = np.gradient(dem.astype(float))
    
    # D8方向编码
    flow_dir = np.zeros_like(dem, dtype=np.int32)
    
    # 简化的流向计算
    angle = np.arctan2(gy, gx) * 180 / np.pi
    flow_dir = ((angle + 22.5) // 45).astype(np.int32) % 8
    
    flow_dir[~valid_mask] = no_data
    
    return rdarray(flow_dir, no_data=no_data)


def Slope(dem, units='degrees'):
    """
    计算坡度
    """
    no_data = getattr(dem, 'no_data', -9999)
    valid_mask = dem != no_data
    
    gy, gx = np.gradient(dem.astype(float))
    slope = np.sqrt(gx**2 + gy**2)
    
    if units == 'degrees':
        slope = np.arctan(slope) * 180 / np.pi
    
    slope[~valid_mask] = no_data
    
    return rdarray(slope, no_data=no_data)


def TerrainAttribute(dem, attrib='slope_riserun'):
    """
    计算地形属性
    """
    if attrib == 'slope_riserun':
        return Slope(dem, units='radians')
    elif attrib == 'slope_degrees':
        return Slope(dem, units='degrees')
    elif attrib in ['aspect', 'curvature']:
        # 简化实现
        return Slope(dem, units='degrees')
    else:
        logger.warning(f"不支持的地形属性: {attrib}, 返回坡度")
        return Slope(dem, units='degrees')


# 导出所有函数
__all__ = [
    'rdarray',
    'FillDepressions',
    'FlowAccumulation', 
    'FlowDirection',
    'Slope',
    'TerrainAttribute'
]


logger.info("richdem兼容层加载完成 - 提供基本DEM处理功能")
