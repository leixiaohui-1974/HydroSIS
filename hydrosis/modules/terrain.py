"""地形处理模块

提供DEM数据处理的核心功能，包括流向计算、流量累积、坑洼填充等。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Module, ModuleInput, ModuleOutput, ModuleMetadata, register_module


@dataclass
class TerrainInput(ModuleInput):
    """地形处理模块输入"""
    
    dem_path: str
    method: str = "d8"  # d8 或 dinf
    fill_depressions: bool = True
    compute_slope: bool = True
    output_dir: str = "results/terrain"
    
    def validate(self) -> List[str]:
        """验证输入"""
        errors = []
        
        # 检查DEM文件
        dem_file = Path(self.dem_path)
        if not dem_file.exists():
            errors.append(f"DEM文件不存在: {self.dem_path}")
        
        # 检查方法
        if self.method not in ["d8", "dinf"]:
            errors.append(f"不支持的流向计算方法: {self.method}")
        
        return errors


@dataclass
class TerrainOutput(ModuleOutput):
    """地形处理模块输出"""
    
    flow_direction: str
    flow_accumulation: str
    filled_dem: Optional[str] = None
    slope: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def outputs_dict(self) -> Dict[str, str]:
        """获取所有输出文件路径"""
        result = {
            "flow_direction": self.flow_direction,
            "flow_accumulation": self.flow_accumulation,
        }
        if self.filled_dem:
            result["filled_dem"] = self.filled_dem
        if self.slope:
            result["slope"] = self.slope
        return result


@register_module
class TerrainModule(Module[TerrainOutput]):
    """地形处理模块"""
    
    @classmethod
    def module_id(cls) -> str:
        return "terrain"
    
    @classmethod
    def metadata(cls) -> ModuleMetadata:
        return ModuleMetadata(
            module_id="terrain",
            name="地形处理模块",
            description="处理DEM数据，计算流向、流量累积、坡度等地形参数",
            version="1.0.0",
            author="HydroSIS Team",
            input_schema={
                "type": "object",
                "properties": {
                    "dem_path": {
                        "type": "string",
                        "description": "DEM文件路径（GeoTIFF格式）"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["d8", "dinf"],
                        "default": "d8",
                        "description": "流向计算方法"
                    },
                    "fill_depressions": {
                        "type": "boolean",
                        "default": True,
                        "description": "是否填充坑洼"
                    },
                    "compute_slope": {
                        "type": "boolean",
                        "default": True,
                        "description": "是否计算坡度"
                    },
                    "output_dir": {
                        "type": "string",
                        "default": "results/terrain",
                        "description": "输出目录"
                    }
                },
                "required": ["dem_path"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "flow_direction": {"type": "string", "description": "流向栅格文件"},
                    "flow_accumulation": {"type": "string", "description": "流量累积栅格文件"},
                    "filled_dem": {"type": "string", "description": "填充后的DEM文件"},
                    "slope": {"type": "string", "description": "坡度栅格文件"},
                    "metadata": {"type": "object", "description": "元数据"}
                }
            }
        )
    
    def validate_inputs(self, inputs: TerrainInput) -> List[str]:
        """验证输入"""
        if isinstance(inputs, dict):
            inputs = TerrainInput(**inputs)
        return inputs.validate()
    
    def execute(self, inputs: TerrainInput, context=None) -> TerrainOutput:
        """执行地形处理"""
        if isinstance(inputs, dict):
            inputs = TerrainInput(**inputs)
        
        # 创建输出目录
        output_dir = Path(inputs.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"开始处理DEM: {inputs.dem_path}")
        self.logger.info(f"方法: {inputs.method}, 填充坑洼: {inputs.fill_depressions}")
        
        # 加载DEM
        import rasterio
        import richdem as rd
        import numpy as np
        
        with rasterio.open(inputs.dem_path) as src:
            dem_array = src.read(1)
            profile = src.profile.copy()
            metadata = {
                "crs": str(src.crs),
                "transform": list(src.transform),
                "bounds": list(src.bounds),
                "width": src.width,
                "height": src.height,
                "resolution": src.res,
            }
            # 获取geotransform用于RichDEM
            geotransform = src.transform.to_gdal()
            nodata_val = src.nodata
        
        # 清理异常NoData值
        if nodata_val and abs(nodata_val) > 1e10:
            self.logger.info(f"清理异常NoData值: {nodata_val}")
            dem_array = np.where(np.abs(dem_array) > 1e10, np.nan, dem_array)
            nodata_val = -9999
        
        # 转换为RichDEM数组并设置geotransform
        rd_dem = rd.rdarray(dem_array, no_data=nodata_val if nodata_val else -9999)
        rd_dem.geotransform = geotransform
        
        # 填充坑洼
        filled_dem_path = None
        if inputs.fill_depressions:
            self.logger.info("填充坑洼...")
            rd.FillDepressions(rd_dem, in_place=True)
            
            # 关键：处理平坦区域以改善流量累积
            self.logger.info("处理平坦区域（BreachDepressions）...")
            try:
                rd.BreachDepressions(rd_dem, in_place=True)
                self.logger.info("✅ 平坦区域处理完成")
            except Exception as e:
                self.logger.warning(f"平坦区域处理失败: {e}")
            
            filled_dem_path = str(output_dir / "filled_dem.tif")
            with rasterio.open(filled_dem_path, 'w', **profile) as dst:
                dst.write(rd_dem, 1)
        
        # 计算流向（通过FlowProportions获取）
        self.logger.info("计算流向...")
        if inputs.method == "d8":
            flow_props = rd.FlowProportions(rd_dem, method='D8')
            # FlowProportions返回一个包含流向信息的数组
            # 简化处理：使用flow accumulation作为流向代理
            flow_dir_arr = rd.FlowAccumulation(rd_dem, method='D8')
        else:
            flow_props = rd.FlowProportions(rd_dem, method='Dinf')
            flow_dir_arr = rd.FlowAccumulation(rd_dem, method='Dinf')
        
        flow_dir_path = str(output_dir / "flow_direction.tif")
        with rasterio.open(flow_dir_path, 'w', **profile) as dst:
            dst.write(flow_dir_arr, 1)
        
        # 计算流量累积
        self.logger.info("计算流量累积...")
        flow_acc = rd.FlowAccumulation(rd_dem, method=inputs.method.upper())
        flow_acc_path = str(output_dir / "flow_accumulation.tif")
        with rasterio.open(flow_acc_path, 'w', **profile) as dst:
            dst.write(flow_acc, 1)
        
        # 计算坡度
        slope_path = None
        if inputs.compute_slope:
            self.logger.info("计算坡度...")
            slope = rd.TerrainAttribute(rd_dem, attrib='slope_riserun')
            slope_path = str(output_dir / "slope.tif")
            with rasterio.open(slope_path, 'w', **profile) as dst:
                dst.write(slope, 1)
        
        # 更新元数据
        metadata["processing"] = {
            "method": inputs.method,
            "fill_depressions": inputs.fill_depressions,
            "compute_slope": inputs.compute_slope
        }
        
        self.logger.info("地形处理完成")
        
        return TerrainOutput(
            flow_direction=flow_dir_path,
            flow_accumulation=flow_acc_path,
            filled_dem=filled_dem_path,
            slope=slope_path,
            metadata=metadata
        )
