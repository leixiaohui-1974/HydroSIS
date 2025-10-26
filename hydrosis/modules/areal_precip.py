"""面雨量计算模块 - 占位实现"""
from __future__ import annotations
from dataclasses import dataclass
from .base import Module, ModuleInput, ModuleOutput, ModuleMetadata, register_module

@dataclass
class ArealPrecipInput(ModuleInput):
    precipitation_timeseries: str
    thiessen_polygons: str
    watersheds: str
    method: str = "thiessen"
    output_dir: str = "results/areal_precip"

@dataclass
class ArealPrecipOutput(ModuleOutput):
    areal_precipitation: str

@register_module
class ArealPrecipitationModule(Module[ArealPrecipOutput]):
    @classmethod
    def module_id(cls) -> str:
        return "areal_precipitation"
    
    @classmethod
    def metadata(cls) -> ModuleMetadata:
        return ModuleMetadata(
            module_id="areal_precipitation",
            name="面雨量计算模块",
            description="计算流域面雨量",
            version="1.0.0",
            input_schema={"type": "object"},
            output_schema={"type": "object"}
        )
    
    def execute(self, inputs: ArealPrecipInput, context=None) -> ArealPrecipOutput:
        if isinstance(inputs, dict):
            inputs = ArealPrecipInput(**inputs)
        from pathlib import Path
        output_dir = Path(inputs.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return ArealPrecipOutput(
            areal_precipitation=str(output_dir / "areal_precip.csv")
        )
