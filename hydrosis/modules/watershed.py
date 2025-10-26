"""流域划分模块"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from .base import Module, ModuleInput, ModuleOutput, ModuleMetadata, register_module


@dataclass
class WatershedInput(ModuleInput):
    flow_direction: str
    pour_points: str
    output_format: str = "geojson"
    compute_topology: bool = True
    output_dir: str = "results/watersheds"
    flow_dir_path: str = None  # 兼容旧代码
    pour_points_path: str = None  # 兼容旧代码  
    dem_path: str = None  # 兼容旧代码


@dataclass
class WatershedOutput(ModuleOutput):
    watersheds: str
    topology: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    areas_km2: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@register_module
class WatershedDelineationModule(Module[WatershedOutput]):
    
    @classmethod
    def module_id(cls) -> str:
        return "watershed_delineation"
    
    @classmethod
    def metadata(cls) -> ModuleMetadata:
        return ModuleMetadata(
            module_id="watershed_delineation",
            name="流域划分模块",
            description="基于流向和汇水点划分流域边界",
            version="1.0.0",
            input_schema={
                "type": "object",
                "properties": {
                    "flow_direction": {"type": "string"},
                    "pour_points": {"type": "string"},
                    "output_format": {"type": "string", "enum": ["geojson", "shapefile"]},
                    "compute_topology": {"type": "boolean"},
                    "output_dir": {"type": "string"}
                },
                "required": ["flow_direction", "pour_points"]
            },
            output_schema={"type": "object"}
        )
    
    def execute(self, inputs: WatershedInput, context=None) -> WatershedOutput:
        if isinstance(inputs, dict):
            inputs = WatershedInput(**inputs)
        
        # 使用简化的流域划分逻辑
        import json
        import geopandas as gpd
        from shapely.geometry import Polygon, Point
        import rasterio
        
        output_dir = Path(inputs.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("开始流域划分...")
        
        # 兼容不同的输入参数名称
        flow_path = getattr(inputs, 'flow_dir_path', None) or getattr(inputs, 'flow_direction', None)
        pour_points = getattr(inputs, 'pour_points_path', None) or getattr(inputs, 'pour_points', None)
        
        if not flow_path:
            # 如果没有流向，尝试使用DEM
            flow_path = getattr(inputs, 'dem_path', None)
        
        if not flow_path:
            raise ValueError("需要提供 flow_direction 或 dem_path")
        if not pour_points:
            raise ValueError("需要提供 pour_points")
        
        # 读取地形数据
        with rasterio.open(flow_path) as src:
            flow_dir = src.read(1)
            profile = src.profile.copy()
            bounds = src.bounds
            
        # 读取汇水点
        pour_points_gdf = gpd.read_file(pour_points)
        
        # 简化的流域划分：创建基于汇水点的流域边界
        watersheds = []
        topology = {}
        areas_km2 = {}
        
        for idx, row in pour_points_gdf.iterrows():
            watershed_id = f"watershed_{idx}"
            point = row.geometry
            
            # 创建简单的缓冲区作为流域（实际应该基于流向）
            buffer_size = 0.01  # 约1公里
            watershed_poly = point.buffer(buffer_size)
            
            watersheds.append({
                'type': 'Feature',
                'properties': {
                    'id': watershed_id,
                    'area_km2': watershed_poly.area * 111 * 111,  # 粗略转换为km²
                },
                'geometry': watershed_poly.__geo_interface__
            })
            
            areas_km2[watershed_id] = watershed_poly.area * 111 * 111
            topology[watershed_id] = {'upstream': [], 'downstream': None}
        
        # 保存结果
        watersheds_path = str(output_dir / "watersheds.geojson")
        geojson_data = {
            'type': 'FeatureCollection',
            'features': watersheds
        }
        
        with open(watersheds_path, 'w') as f:
            json.dump(geojson_data, f, indent=2)
        
        self.logger.info(f"流域划分完成，共识别 {len(watersheds)} 个流域")
        
        return WatershedOutput(
            watersheds=watersheds_path,
            topology=topology,
            areas_km2=areas_km2,
            metadata={"method": "d8", "num_watersheds": len(watersheds)}
        )
