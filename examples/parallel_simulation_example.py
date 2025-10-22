"""Example demonstrating parallel simulation capabilities of HydroSIS.

This example shows how to use the ParallelHydroSISModel to accelerate
simulations for large-scale watershed modeling. It compares the performance
of sequential and parallel execution for the same model configuration.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

# Add the parent directory to the path to import hydrosis
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hydrosis import (
    HydroSISModel,
    ModelConfig,
    ParallelConfig,
    ParallelHydroSISModel,
    run_workflow,
)
from hydrosis.io.inputs import load_forcing


def _load_model_config(config_path: Path) -> ModelConfig:
    """Load the model configuration with a JSON fallback when PyYAML is absent."""
    try:
        return ModelConfig.from_yaml(config_path)
    except ImportError:
        json_path = config_path.with_suffix(".json")
        print(
            "PyYAML 未安装，改用 JSON 配置加载示例 (", json_path.as_posix(), ")",
            sep="",
        )
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return ModelConfig.from_dict(data)


def _generate_large_forcing(subbasin_ids: List[str], time_steps: int = 1000) -> Dict[str, List[float]]:
    """Generate synthetic forcing data for a large number of time steps.
    
    Args:
        subbasin_ids: List of subbasin identifiers
        time_steps: Number of time steps to generate
        
    Returns:
        Dictionary mapping subbasin IDs to precipitation time series
    """
    import random

    forcing = {}
    for sub_id in subbasin_ids:
        # Generate synthetic precipitation with some random variation
        precipitation = []
        for _ in range(time_steps):
            # Base precipitation with random variation
            base = random.uniform(0, 20)
            # Add some seasonal pattern
            seasonal = 10 * (0.5 - abs((_ % 365) / 365 - 0.5))
            # Add some random events
            if random.random() < 0.05:  # 5% chance of storm
                storm = random.uniform(30, 80)
            else:
                storm = 0
            
            precipitation.append(max(0, base + seasonal + storm))
        
        forcing[sub_id] = precipitation
    
    return forcing


def main() -> None:
    """Run the parallel simulation example."""
    print("HydroSIS 并行计算示例")
    print("=" * 50)
    
    # Load the model configuration
    repo_root = REPO_ROOT
    config_path = repo_root / "config" / "example_model.yaml"
    config = _load_model_config(config_path)
    
    # Get subbasin IDs from the configuration
    subbasins = config.delineation.to_subbasins()
    subbasin_ids = [sub.id for sub in subbasins]
    
    print(f"模型包含 {len(subbasin_ids)} 个子流域: {', '.join(subbasin_ids)}")
    
    # Generate synthetic forcing data
    time_steps = 1000
    forcing = _generate_large_forcing(subbasin_ids, time_steps)
    print(f"生成了 {time_steps} 个时间步的合成降雨数据")
    
    # Create synthetic observations
    baseline_model = HydroSISModel.from_config(config)
    baseline_local = baseline_model.run(forcing)
    observations = baseline_model.accumulate_discharge(baseline_local)
    
    # Test sequential execution
    print("\n1. 顺序执行模拟...")
    sequential_model = HydroSISModel.from_config(config)
    start_time = time.time()
    sequential_result = sequential_model.run(forcing)
    sequential_time = time.time() - start_time
    print(f"顺序执行耗时: {sequential_time:.2f} 秒")
    
    # Test parallel execution with different configurations
    parallel_configs = [
        ParallelConfig(use_processes=True, max_workers=2, chunk_size=1),
        ParallelConfig(use_processes=True, max_workers=4, chunk_size=1),
        ParallelConfig(use_processes=True, max_workers=8, chunk_size=1),
        ParallelConfig(use_processes=True, max_workers=4, chunk_size=2),
        ParallelConfig(use_processes=False, max_workers=4, chunk_size=1),
    ]
    
    print("\n2. 并行执行模拟...")
    for i, parallel_config in enumerate(parallel_configs):
        worker_type = "进程" if parallel_config.use_processes else "线程"
        print(f"\n配置 {i+1}: {worker_type}并行, 工作单元={parallel_config.max_workers}, 块大小={parallel_config.chunk_size}")
        
        parallel_model = ParallelHydroSISModel.from_config(config, parallel_config)
        start_time = time.time()
        parallel_result = parallel_model.run(forcing)
        parallel_time = time.time() - start_time
        
        # Verify results are the same
        max_diff = max(
            max(abs(a - b) for a, b in zip(sequential_result[sub_id], parallel_result[sub_id]))
            for sub_id in subbasin_ids
        )
        
        speedup = sequential_time / parallel_time if parallel_time > 0 else float('inf')
        print(f"并行执行耗时: {parallel_time:.2f} 秒")
        print(f"加速比: {speedup:.2f}x")
        print(f"最大结果差异: {max_diff:.6f}")
    
    # Test parallel workflow execution
    print("\n3. 并行工作流执行...")
    parallel_config = ParallelConfig(use_processes=True, max_workers=4)
    parallel_model = ParallelHydroSISModel.from_config(config, parallel_config)
    
    start_time = time.time()
    workflow_result = run_workflow(
        config,
        forcing,
        observations=observations,
        scenario_ids=["alternate_routing"],
        persist_outputs=False,
        generate_report=False,
    )
    workflow_time = time.time() - start_time
    
    print(f"工作流执行耗时: {workflow_time:.2f} 秒")
    print(f"基准情景NSE: {workflow_result.overall_scores[0].aggregated['nse']:.4f}")
    print(f"情景NSE: {workflow_result.overall_scores[1].aggregated['nse']:.4f}")
    
    print("\n示例完成！")


if __name__ == "__main__":
    main()
