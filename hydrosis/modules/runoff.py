"""产流模拟模块 - 占位实现"""
from __future__ import annotations
from dataclasses import dataclass
from .base import Module, ModuleInput, ModuleOutput, ModuleMetadata, register_module

@dataclass
class RunoffInput(ModuleInput):
    precipitation: str
    watersheds: str
    model: str = "hbv"
    parameters: dict = None
    output_dir: str = "results/runoff"

@dataclass
class RunoffOutput(ModuleOutput):
    runoff_timeseries: str

@register_module
class RunoffGenerationModule(Module[RunoffOutput]):
    @classmethod
    def module_id(cls) -> str:
        return "runoff_generation"
    
    @classmethod
    def metadata(cls) -> ModuleMetadata:
        return ModuleMetadata(
            module_id="runoff_generation",
            name="产流模拟模块",
            description="使用水文模型模拟产流过程",
            version="1.0.0",
            input_schema={"type": "object"},
            output_schema={"type": "object"}
        )
    
    def execute(self, inputs: RunoffInput, context=None) -> RunoffOutput:
        if isinstance(inputs, dict):
            inputs = RunoffInput(**inputs)
        from pathlib import Path
        output_dir = Path(inputs.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return RunoffOutput(
            runoff_timeseries=str(output_dir / "runoff.csv")
        )
