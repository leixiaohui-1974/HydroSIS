"""Example demonstrating the distributed Green-Ampt runoff model.

This example shows how to use the distributed Green-Ampt model for
spatially variable infiltration and runoff calculations. It compares
the results with the SCS Curve Number method for the same precipitation
series.
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
    run_workflow,
)
from hydrosis.io.inputs import load_forcing
from hydrosis.runoff.base import RunoffModelConfig
from hydrosis.routing.base import RoutingModelConfig


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


def _create_green_ampt_config() -> ModelConfig:
    """Create a model configuration using the distributed Green-Ampt model."""
    from hydrosis.delineation.dem_delineator import DelineationConfig
    from hydrosis.parameters.zone import ParameterZoneConfig
    from hydrosis.config import IOConfig, ScenarioConfig, EvaluationConfig

    # Simple delineation with predefined subbasins
    delineation = DelineationConfig(
        dem_path=Path("dem.tif"),
        pour_points_path=Path("pour_points.geojson"),
        precomputed_subbasins=[
            {"id": "S1", "area_km2": 10.0, "downstream": "S3", "parameters": {}},
            {"id": "S2", "area_km2": 12.0, "downstream": "S3", "parameters": {}},
            {"id": "S3", "area_km2": 20.0, "downstream": None, "parameters": {}},
        ],
    )

    # Use distributed Green-Ampt for runoff
    runoff_models = [
        RunoffModelConfig(
            id="green_ampt_uniform",
            model_type="distributed_green_ampt",
            parameters={
                "saturated_conductivity": 10.0,  # mm/hr
                "wetting_front_suction": 100.0,  # mm
                "initial_moisture": 0.2,  # fraction
                "saturated_moisture": 0.4,  # fraction
                "porosity": 0.45,  # fraction
                "zones": 5,
                "zone_distribution": "uniform",
            },
        ),
        RunoffModelConfig(
            id="green_ampt_random",
            model_type="distributed_green_ampt",
            parameters={
                "saturated_conductivity": 10.0,  # mm/hr
                "wetting_front_suction": 100.0,  # mm
                "initial_moisture": 0.2,  # fraction
                "saturated_moisture": 0.4,  # fraction
                "porosity": 0.45,  # fraction
                "zones": 5,
                "zone_distribution": "random",
            },
        ),
        RunoffModelConfig(
            id="green_ampt_clustered",
            model_type="distributed_green_ampt",
            parameters={
                "saturated_conductivity": 10.0,  # mm/hr
                "wetting_front_suction": 100.0,  # mm
                "initial_moisture": 0.2,  # fraction
                "saturated_moisture": 0.4,  # fraction
                "porosity": 0.45,  # fraction
                "zones": 6,
                "zone_distribution": "clustered",
            },
        ),
    ]

    # Simple lag routing
    routing_models = [
        RoutingModelConfig(
            id="lag",
            model_type="lag",
            parameters={"lag_steps": 1},
        ),
    ]

    # Parameter zones
    parameter_zones = [
        ParameterZoneConfig(
            id="Z1",
            description="Headwater zone",
            control_points=["S1"],
            parameters={"runoff_model": "green_ampt_uniform", "routing_model": "lag"},
        ),
        ParameterZoneConfig(
            id="Z2",
            description="Middle zone",
            control_points=["S2"],
            parameters={"runoff_model": "green_ampt_random", "routing_model": "lag"},
        ),
        ParameterZoneConfig(
            id="Z3",
            description="Outlet zone",
            control_points=["S3"],
            parameters={"runoff_model": "green_ampt_clustered", "routing_model": "lag"},
        ),
    ]

    # IO configuration
    io_config = IOConfig(
        precipitation=Path("data/forcing/precipitation"),
        results_directory=Path("results/green_ampt_example"),
        figures_directory=Path("results/green_ampt_example/figures"),
        reports_directory=Path("results/green_ampt_example/reports"),
    )

    # Scenario with different initial moisture
    scenarios = [
        ScenarioConfig(
            id="dry_conditions",
            description="Dry initial conditions",
            modifications={
                "S1": {"initial_moisture": 0.1},
                "S2": {"initial_moisture": 0.1},
                "S3": {"initial_moisture": 0.1},
            },
        ),
        ScenarioConfig(
            id="wet_conditions",
            description="Wet initial conditions",
            modifications={
                "S1": {"initial_moisture": 0.35},
                "S2": {"initial_moisture": 0.35},
                "S3": {"initial_moisture": 0.35},
            },
        ),
    ]

    return ModelConfig(
        delineation=delineation,
        runoff_models=runoff_models,
        routing_models=routing_models,
        parameter_zones=parameter_zones,
        io=io_config,
        scenarios=scenarios,
    )


def _generate_storm_precipitation(time_steps: int = 48) -> Dict[str, List[float]]:
    """Generate a synthetic storm event with high-intensity rainfall.
    
    Args:
        time_steps: Number of time steps (hours)
        
    Returns:
        Dictionary mapping subbasin IDs to precipitation time series (mm/hr)
    """
    import random

    # Create a storm with a peak in the middle
    precipitation = []
    for i in range(time_steps):
        # Base precipitation
        base = 0.5
        
        # Storm peak in the middle of the event
        if 12 <= i <= 24:
            # Peak intensity with some variation
            peak = 30.0 * (1.0 - 0.5 * abs(i - 18) / 6.0)
            # Add random variation
            variation = random.uniform(0.8, 1.2)
            p = base + peak * variation
        else:
            # Light rain outside the peak period
            p = base + random.uniform(0, 2.0)
        
        precipitation.append(max(0, p))
    
    # Apply the same precipitation to all subbasins
    return {
        "S1": precipitation.copy(),
        "S2": precipitation.copy(),
        "S3": precipitation.copy(),
    }


def main() -> None:
    """Run the distributed Green-Ampt example."""
    print("分布式 Green-Ampt 模型示例")
    print("=" * 50)
    
    # Create model configuration
    config = _create_green_ampt_config()
    
    # Generate storm precipitation
    precipitation = _generate_storm_precipitation(48)  # 48 hours
    print(f"生成了 48 小时的暴雨事件，最大降雨强度: {max(precipitation['S1']):.1f} mm/hr")
    
    # Create model and run simulation
    model = HydroSISModel.from_config(config)
    
    print("\n运行基准模拟...")
    start_time = time.time()
    local_results = model.run(precipitation)
    aggregated_results = model.accumulate_discharge(local_results)
    simulation_time = time.time() - start_time
    print(f"模拟完成，耗时: {simulation_time:.2f} 秒")
    
    # Calculate some statistics
    outlet_flow = aggregated_results.get("S3", [])
    if outlet_flow:
        peak_flow = max(outlet_flow)
        peak_time = outlet_flow.index(peak_flow)
        total_volume = sum(outlet_flow)  # m³/s * hours
        
        print(f"\n出口子流域 (S3) 统计:")
        print(f"  峰值流量: {peak_flow:.2f} m³/s")
        print(f"  峰现时间: 第 {peak_time} 小时")
        print(f"  总径流量: {total_volume:.2f} m³/s·h")
    
    # Run scenarios
    print("\n运行情景模拟...")
    scenario_results = {}
    
    for scenario_id in ["dry_conditions", "wet_conditions"]:
        print(f"\n情景: {scenario_id}")
        scenario_config = _create_green_ampt_config()
        scenario_config.apply_scenario(scenario_id, model.subbasins.values())
        
        scenario_model = HydroSISModel.from_config(scenario_config)
        scenario_local = scenario_model.run(precipitation)
        scenario_aggregated = scenario_model.accumulate_discharge(scenario_local)
        scenario_results[scenario_id] = scenario_aggregated
        
        # Calculate statistics
        scenario_outlet = scenario_aggregated.get("S3", [])
        if scenario_outlet:
            scenario_peak = max(scenario_outlet)
            scenario_peak_time = scenario_outlet.index(scenario_peak)
            scenario_volume = sum(scenario_outlet)
            
            print(f"  峰值流量: {scenario_peak:.2f} m³/s")
            print(f"  峰现时间: 第 {scenario_peak_time} 小时")
            print(f"  总径流量: {scenario_volume:.2f} m³/s·h")
            
            # Compare with baseline
            if outlet_flow:
                peak_change = (scenario_peak - peak_flow) / peak_flow * 100
                volume_change = (scenario_volume - total_volume) / total_volume * 100
                print(f"  峰值变化: {peak_change:+.1f}%")
                print(f"  径流量变化: {volume_change:+.1f}%")
    
    # Create a simple visualization if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        print("\n生成可视化图表...")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot precipitation
        ax2 = ax.twinx()
        ax2.plot(precipitation["S1"], 'b-', alpha=0.3, label='降雨强度')
        ax2.set_ylabel('降雨强度 (mm/hr)', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        
        # Plot runoff
        ax.plot(outlet_flow, 'k-', linewidth=2, label='基准')
        for scenario_id, scenario_data in scenario_results.items():
            scenario_outlet = scenario_data.get("S3", [])
            ax.plot(scenario_outlet, '--', linewidth=1.5, label=scenario_id)
        
        ax.set_xlabel('时间 (小时)')
        ax.set_ylabel('流量 (m³/s)')
        ax.set_title('分布式 Green-Ampt 模型 - 情景对比')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Save figure
        output_dir = REPO_ROOT / "results" / "green_ampt_example"
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "green_ampt_comparison.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"图表已保存到: {output_dir / 'green_ampt_comparison.png'}")
    except ImportError:
        print("\nMatplotlib 不可用，跳过可视化生成")
    
    print("\n示例完成！")


if __name__ == "__main__":
    main()
