"""
Pfafstetter编码系统实现

基于Otto Pfafstetter (1989)的河流编码系统
编码格式: ZZZ-PPPPPPPP
  - ZZZ: 3位参数分区编码 (001-999)
  - PPPPPPPP: Pfafstetter子流域编码
"""

import numpy as np
from typing import Dict, List, Tuple
import geopandas as gpd
from pathlib import Path

class PfafstetterCoding:
    """Pfafstetter编码系统"""
    
    def __init__(self, parameter_zone_digits=3):
        """
        初始化Pfafstetter编码系统
        
        Args:
            parameter_zone_digits: 参数分区位数（默认3位）
        """
        self.param_zone_digits = parameter_zone_digits
        
    def assign_parameter_zones(self, watersheds: gpd.GeoDataFrame, 
                               num_zones: int = None) -> gpd.GeoDataFrame:
        """
        分配参数分区编码
        
        基于流域面积、地理位置等特征自动分区
        
        Args:
            watersheds: 流域GeoDataFrame
            num_zones: 分区数量（None则自动确定）
            
        Returns:
            添加了parameter_zone字段的GeoDataFrame
        """
        if num_zones is None:
            # 根据流域数量自动确定分区数
            num_watersheds = len(watersheds)
            if num_watersheds < 10:
                num_zones = 1
            elif num_watersheds < 50:
                num_zones = 3
            elif num_watersheds < 100:
                num_zones = 5
            else:
                num_zones = 10
        
        # 基于空间位置聚类分区（简化版：基于y坐标）
        centroids = watersheds.geometry.centroid
        y_coords = centroids.y.values
        
        # 分位数分区
        zones = np.zeros(len(watersheds), dtype=int)
        for i in range(num_zones):
            lower = np.percentile(y_coords, i * 100 / num_zones)
            upper = np.percentile(y_coords, (i + 1) * 100 / num_zones)
            mask = (y_coords >= lower) & (y_coords <= upper)
            zones[mask] = i + 1
        
        # 生成3位编码
        watersheds['parameter_zone'] = [f"{z:03d}" for z in zones]
        
        return watersheds
    
    def pfafstetter_encode(self, 
                          watersheds: gpd.GeoDataFrame,
                          topology: Dict,
                          main_outlet_id: str = None) -> gpd.GeoDataFrame:
        """
        生成Pfafstetter子流域编码
        
        Pfafstetter规则:
        - 主河道支流编号为奇数（1,3,5,7,9）从上游到下游
        - 主河道之间的区域编号为偶数（2,4,6,8）
        - 干流本身编号为0
        
        Args:
            watersheds: 流域GeoDataFrame
            topology: 流域拓扑关系
            main_outlet_id: 主出口ID（None则自动识别）
            
        Returns:
            添加了pfafstetter_code字段的GeoDataFrame
        """
        if main_outlet_id is None:
            # 查找主出口（没有下游的流域）
            outlet_candidates = []
            for wid, topo in topology.items():
                if topo.get('downstream') is None:
                    outlet_candidates.append(wid)
            
            if len(outlet_candidates) == 1:
                main_outlet_id = outlet_candidates[0]
            elif len(outlet_candidates) > 1:
                # 选择面积最大的作为主出口
                max_area = 0
                for candidate in outlet_candidates:
                    area = watersheds[watersheds.index == candidate]['area_km2'].iloc[0]
                    if area > max_area:
                        max_area = area
                        main_outlet_id = candidate
            else:
                # 默认第一个
                main_outlet_id = watersheds.index[0]
        
        # 追踪主河道
        main_stem = self._trace_main_stem(topology, main_outlet_id)
        
        # 生成编码
        codes = {}
        
        # 主河道编码为0
        for ws_id in main_stem:
            codes[ws_id] = "0"
        
        # 识别主要支流（按流域面积排序）
        tributaries = []
        for ws_id in main_stem:
            upstream_ids = topology.get(ws_id, {}).get('upstream', [])
            for up_id in upstream_ids:
                if up_id not in main_stem:
                    # 计算支流总面积
                    trib_area = self._calculate_tributary_area(
                        up_id, watersheds, topology
                    )
                    tributaries.append((up_id, trib_area))
        
        # 按面积排序，选择前4-5个主要支流
        tributaries.sort(key=lambda x: x[1], reverse=True)
        main_tributaries = tributaries[:min(5, len(tributaries))]
        
        # 分配奇数编号给主要支流
        for i, (trib_id, area) in enumerate(main_tributaries):
            code = str(2*i + 1)  # 1, 3, 5, 7, 9
            self._assign_tributary_code(trib_id, code, codes, topology)
        
        # 剩余流域分配偶数编号
        remaining_code = 2
        for ws_id in watersheds.index:
            if ws_id not in codes:
                codes[ws_id] = str(remaining_code)
                remaining_code += 2
        
        # 添加到GeoDataFrame
        watersheds['pfafstetter_code'] = watersheds.index.map(lambda x: codes.get(x, "99"))
        
        return watersheds
    
    def _trace_main_stem(self, topology: Dict, outlet_id: str) -> List[str]:
        """追踪主河道（从出口向上游）"""
        main_stem = [outlet_id]
        current = outlet_id
        
        while True:
            upstream_ids = topology.get(current, {}).get('upstream', [])
            if not upstream_ids:
                break
            
            # 选择上游中最大的一个作为主河道
            # （实际应该基于流量或面积）
            current = upstream_ids[0]  # 简化版：选第一个
            main_stem.append(current)
            
        return main_stem
    
    def _calculate_tributary_area(self, 
                                   trib_id: str,
                                   watersheds: gpd.GeoDataFrame,
                                   topology: Dict) -> float:
        """计算支流总面积（包括所有上游）"""
        total_area = 0
        
        # 递归计算
        def add_upstream_area(ws_id):
            nonlocal total_area
            if ws_id in watersheds.index:
                total_area += watersheds.loc[ws_id, 'area_km2']
            
            upstream_ids = topology.get(ws_id, {}).get('upstream', [])
            for up_id in upstream_ids:
                add_upstream_area(up_id)
        
        add_upstream_area(trib_id)
        return total_area
    
    def _assign_tributary_code(self,
                               trib_id: str,
                               code: str,
                               codes: Dict,
                               topology: Dict):
        """为支流及其上游分配编码"""
        codes[trib_id] = code
        
        # 递归分配上游
        upstream_ids = topology.get(trib_id, {}).get('upstream', [])
        for up_id in upstream_ids:
            codes[up_id] = code
            self._assign_tributary_code(up_id, code, codes, topology)
    
    def generate_full_code(self, 
                          watersheds: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        生成完整的Pfafstetter编码
        
        格式: ZZZ-PPPPPPPP
        
        Args:
            watersheds: 必须包含parameter_zone和pfafstetter_code字段
            
        Returns:
            添加了full_code字段的GeoDataFrame
        """
        if 'parameter_zone' not in watersheds.columns:
            raise ValueError("缺少parameter_zone字段，请先运行assign_parameter_zones")
        
        if 'pfafstetter_code' not in watersheds.columns:
            raise ValueError("缺少pfafstetter_code字段，请先运行pfafstetter_encode")
        
        # 生成完整编码
        watersheds['full_pfafstetter_code'] = (
            watersheds['parameter_zone'] + '-' + 
            watersheds['pfafstetter_code'].str.zfill(8)
        )
        
        return watersheds
    
    def encode_watersheds(self,
                         watersheds_path: str,
                         output_path: str = None) -> gpd.GeoDataFrame:
        """
        完整的流域编码流程
        
        Args:
            watersheds_path: 流域GeoJSON文件路径
            output_path: 输出路径（None则不保存）
            
        Returns:
            编码后的GeoDataFrame
        """
        # 读取流域
        watersheds = gpd.read_file(watersheds_path)
        
        # 简单的拓扑构建（基于空间邻接）
        topology = {}
        for idx in watersheds.index:
            topology[idx] = {'upstream': [], 'downstream': None}
        
        # 分配参数分区
        watersheds = self.assign_parameter_zones(watersheds)
        
        # Pfafstetter编码
        watersheds = self.pfafstetter_encode(watersheds, topology)
        
        # 生成完整编码
        watersheds = self.generate_full_code(watersheds)
        
        # 保存
        if output_path:
            watersheds.to_file(output_path, driver='GeoJSON')
            print(f"✅ 编码后的流域已保存: {output_path}")
        
        return watersheds


def demo():
    """演示Pfafstetter编码"""
    print("="*80)
    print("Pfafstetter编码系统演示")
    print("="*80)
    
    # 使用测试结果
    possible_paths = [
        "results/workflow_tests/03_delineation/watersheds/watersheds.geojson",
        "results/workflow_tests/08_complete/03_watersheds/watersheds.geojson",
        "results/enhanced_workflow_tests/08_完整十一步/outputs/step03_watershed/watersheds.geojson",
    ]
    
    watersheds_path = None
    for path in possible_paths:
        if Path(path).exists():
            watersheds_path = path
            break
    
    if not Path(watersheds_path).exists():
        print(f"错误: 未找到流域文件")
        return
    
    print(f"\n读取流域: {watersheds_path}")
    
    # 创建编码器
    coder = PfafstetterCoding(parameter_zone_digits=3)
    
    # 编码
    output_dir = Path("results/pfafstetter_coding")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    watersheds = coder.encode_watersheds(
        watersheds_path,
        output_path=output_dir / "watersheds_coded.geojson"
    )
    
    print(f"\n编码结果示例（前10个）:")
    print(watersheds[['parameter_zone', 'pfafstetter_code', 'full_pfafstetter_code', 'area_km2']].head(10))
    
    # 统计
    print(f"\n编码统计:")
    print(f"  参数分区数: {watersheds['parameter_zone'].nunique()}")
    print(f"  流域总数: {len(watersheds)}")
    print(f"  编码类型分布:")
    for code in sorted(watersheds['pfafstetter_code'].unique()):
        count = np.sum(watersheds['pfafstetter_code'] == code)
        print(f"    代码{code}: {count}个流域")
    
    print("\n" + "="*80)
    print("编码完成")
    print("="*80)

if __name__ == "__main__":
    demo()
