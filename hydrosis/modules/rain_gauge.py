"""雨量站布局模块 - 占位实现"""
from __future__ import annotations
from dataclasses import dataclass
from .base import Module, ModuleInput, ModuleOutput, ModuleMetadata, register_module

@dataclass
class RainGaugeInput(ModuleInput):
    watershed: str
    target_density: float = 0.01
    output_dir: str = "results/rain_gauges"

@dataclass
class RainGaugeOutput(ModuleOutput):
    gauge_layout: str
    thiessen_polygons: str

@register_module
class RainGaugeLayoutModule(Module[RainGaugeOutput]):
    @classmethod
    def module_id(cls) -> str:
        return "rain_gauge_layout"
    
    @classmethod
    def metadata(cls) -> ModuleMetadata:
        return ModuleMetadata(
            module_id="rain_gauge_layout",
            name="雨量站布局模块",
            description="优化雨量站布局",
            version="1.0.0",
            input_schema={"type": "object"},
            output_schema={"type": "object"}
        )
    
    def execute(self, inputs: RainGaugeInput, context=None) -> RainGaugeOutput:
        if isinstance(inputs, dict):
            inputs = RainGaugeInput(**inputs)
        from pathlib import Path
        output_dir = Path(inputs.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return RainGaugeOutput(
            gauge_layout=str(output_dir / "gauges.geojson"),
            thiessen_polygons=str(output_dir / "thiessen.geojson")
        )
