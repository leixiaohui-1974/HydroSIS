"""降雨序列生成模块 - 占位实现"""
from __future__ import annotations
from dataclasses import dataclass
from .base import Module, ModuleInput, ModuleOutput, ModuleMetadata, register_module

@dataclass
class PrecipitationInput(ModuleInput):
    method: str = "uniform"
    duration_hours: int = 168
    intensity_mm_h: float = 10.0
    output_dir: str = "results/precipitation"

@dataclass
class PrecipitationOutput(ModuleOutput):
    precipitation_timeseries: str

@register_module
class PrecipitationGenerationModule(Module[PrecipitationOutput]):
    @classmethod
    def module_id(cls) -> str:
        return "precipitation_generation"
    
    @classmethod
    def metadata(cls) -> ModuleMetadata:
        return ModuleMetadata(
            module_id="precipitation_generation",
            name="降雨序列生成模块",
            description="生成降雨时间序列",
            version="1.0.0",
            input_schema={"type": "object"},
            output_schema={"type": "object"}
        )
    
    def execute(self, inputs: PrecipitationInput, context=None) -> PrecipitationOutput:
        if isinstance(inputs, dict):
            inputs = PrecipitationInput(**inputs)
        from pathlib import Path
        output_dir = Path(inputs.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return PrecipitationOutput(
            precipitation_timeseries=str(output_dir / "precip.csv")
        )
