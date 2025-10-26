"""汇水点生成模块

支持自动识别和手动指定汇水点。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import Module, ModuleInput, ModuleOutput, ModuleMetadata, register_module


@dataclass
class PourPoint:
    """汇水点数据结构"""
    
    id: str
    lon: float
    lat: float
    accumulation: Optional[float] = None
    snapped: bool = False
    original_lon: Optional[float] = None
    original_lat: Optional[float] = None


@dataclass
class PourPointsInput(ModuleInput):
    """汇水点生成模块输入"""
    
    flow_accumulation: str
    method: str = "auto"  # auto 或 manual
    threshold: Optional[float] = 1000.0
    points: Optional[List[Dict[str, Any]]] = None
    snap_distance: float = 500.0
    output_dir: str = "results/pour_points"
    
    def validate(self) -> List[str]:
        """验证输入"""
        errors = []
        
        # 检查流量累积文件
        flow_acc_file = Path(self.flow_accumulation)
        if not flow_acc_file.exists():
            errors.append(f"流量累积文件不存在: {self.flow_accumulation}")
        
        # 检查方法
        if self.method not in ["auto", "manual"]:
            errors.append(f"不支持的方法: {self.method}")
        
        # 如果是手动模式，检查points
        if self.method == "manual" and not self.points:
            errors.append("手动模式需要提供points参数")
        
        return errors


@dataclass
class PourPointsOutput(ModuleOutput):
    """汇水点生成模块输出"""
    
    pour_points_geojson: str
    points: List[PourPoint] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@register_module
class PourPointsModule(Module[PourPointsOutput]):
    """汇水点生成模块"""
    
    @classmethod
    def module_id(cls) -> str:
        return "pour_points"
    
    @classmethod
    def metadata(cls) -> ModuleMetadata:
        return ModuleMetadata(
            module_id="pour_points",
            name="汇水点生成模块",
            description="自动识别或手动指定流域汇水点",
            version="1.0.0",
            author="HydroSIS Team",
            input_schema={
                "type": "object",
                "properties": {
                    "flow_accumulation": {
                        "type": "string",
                        "description": "流量累积栅格文件路径"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["auto", "manual"],
                        "default": "auto",
                        "description": "生成方法"
                    },
                    "threshold": {
                        "type": "number",
                        "default": 1000.0,
                        "description": "自动识别阈值（像元数）"
                    },
                    "points": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "lon": {"type": "number"},
                                "lat": {"type": "number"}
                            }
                        },
                        "description": "手动指定的汇水点"
                    },
                    "snap_distance": {
                        "type": "number",
                        "default": 500.0,
                        "description": "捕捉距离（米）"
                    },
                    "output_dir": {
                        "type": "string",
                        "default": "results/pour_points"
                    }
                },
                "required": ["flow_accumulation"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "pour_points_geojson": {"type": "string"},
                    "points": {
                        "type": "array",
                        "items": {"type": "object"}
                    },
                    "metadata": {"type": "object"}
                }
            }
        )
    
    def validate_inputs(self, inputs: PourPointsInput) -> List[str]:
        """验证输入"""
        if isinstance(inputs, dict):
            inputs = PourPointsInput(**inputs)
        return inputs.validate()
    
    def execute(self, inputs: PourPointsInput, context=None) -> PourPointsOutput:
        """执行汇水点生成"""
        if isinstance(inputs, dict):
            inputs = PourPointsInput(**inputs)
        
        # 创建输出目录
        output_dir = Path(inputs.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"开始生成汇水点，方法: {inputs.method}")
        
        import rasterio
        import numpy as np
        from scipy import ndimage
        
        # 加载流量累积数据
        with rasterio.open(inputs.flow_accumulation) as src:
            flow_acc = src.read(1)
            transform = src.transform
            crs = src.crs
        
        points = []
        
        if inputs.method == "auto":
            # 自动识别汇水点
            self.logger.info(f"自动识别汇水点，阈值: {inputs.threshold}")
            
            # 确保threshold是数值类型
            threshold_value = float(inputs.threshold) if inputs.threshold is not None else 1000.0
            
            # 找到所有超过阈值的点
            high_acc_mask = flow_acc >= threshold_value
            
            # 使用形态学操作找到局部最大值
            local_max = ndimage.maximum_filter(flow_acc, size=5) == flow_acc
            pour_point_mask = high_acc_mask & local_max
            
            # 获取坐标
            rows, cols = np.where(pour_point_mask)
            
            for idx, (row, col) in enumerate(zip(rows, cols)):
                lon, lat = rasterio.transform.xy(transform, row, col)
                acc_value = float(flow_acc[row, col])
                
                points.append(PourPoint(
                    id=f"P{idx+1}",
                    lon=lon,
                    lat=lat,
                    accumulation=acc_value,
                    snapped=False
                ))
            
            self.logger.info(f"自动识别到 {len(points)} 个汇水点")
        
        else:
            # 手动指定汇水点
            self.logger.info(f"处理手动指定的 {len(inputs.points)} 个汇水点")
            
            for pt_dict in inputs.points:
                pt_id = pt_dict['id']
                orig_lon = pt_dict['lon']
                orig_lat = pt_dict['lat']
                
                # 转换到栅格坐标
                row, col = rasterio.transform.rowcol(transform, orig_lon, orig_lat)
                
                # 在附近搜索最大流量累积点（捕捉到河网）
                search_radius = int(inputs.snap_distance / abs(transform[0]))
                
                row_start = max(0, row - search_radius)
                row_end = min(flow_acc.shape[0], row + search_radius + 1)
                col_start = max(0, col - search_radius)
                col_end = min(flow_acc.shape[1], col + search_radius + 1)
                
                search_area = flow_acc[row_start:row_end, col_start:col_end]
                
                if search_area.size > 0:
                    # 找到最大值位置
                    local_row, local_col = np.unravel_index(
                        search_area.argmax(), search_area.shape
                    )
                    snapped_row = row_start + local_row
                    snapped_col = col_start + local_col
                    
                    # 转换回地理坐标
                    snapped_lon, snapped_lat = rasterio.transform.xy(
                        transform, snapped_row, snapped_col
                    )
                    acc_value = float(flow_acc[snapped_row, snapped_col])
                    
                    # 检查是否移动了
                    snapped = (snapped_row != row) or (snapped_col != col)
                    
                    points.append(PourPoint(
                        id=pt_id,
                        lon=snapped_lon,
                        lat=snapped_lat,
                        accumulation=acc_value,
                        snapped=snapped,
                        original_lon=orig_lon if snapped else None,
                        original_lat=orig_lat if snapped else None
                    ))
                    
                    if snapped:
                        self.logger.info(f"汇水点 {pt_id} 已捕捉到河网")
                else:
                    self.logger.warning(f"汇水点 {pt_id} 搜索区域为空")
        
        # 保存为GeoJSON
        geojson_path = output_dir / "pour_points.geojson"
        self._save_geojson(points, geojson_path, crs)
        
        metadata = {
            "method": inputs.method,
            "count": len(points),
            "crs": str(crs),
            "threshold": inputs.threshold if inputs.method == "auto" else None,
            "snap_distance": inputs.snap_distance if inputs.method == "manual" else None
        }
        
        self.logger.info(f"汇水点生成完成，共 {len(points)} 个点")
        
        return PourPointsOutput(
            pour_points_geojson=str(geojson_path),
            points=points,
            metadata=metadata
        )
    
    def _save_geojson(self, points: List[PourPoint], path: Path, crs) -> None:
        """保存汇水点为GeoJSON"""
        import json
        
        features = []
        for pt in points:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [pt.lon, pt.lat]
                },
                "properties": {
                    "id": pt.id,
                    "accumulation": pt.accumulation,
                    "snapped": pt.snapped
                }
            }
            
            if pt.snapped and pt.original_lon is not None:
                feature["properties"]["original_lon"] = pt.original_lon
                feature["properties"]["original_lat"] = pt.original_lat
            
            features.append(feature)
        
        geojson = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": str(crs)}
            },
            "features": features
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, indent=2)
