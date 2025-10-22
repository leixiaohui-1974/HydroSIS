"""Example demonstrating visualization capabilities of HydroSIS.

This example shows how to use the visualization module to create charts
and dashboards for HydroSIS simulation results.
"""
from __future__ import annotations

import json
import sys
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
from hydrosis.visualization import (
    create_hydrograph_chart,
    create_comparison_chart,
    create_metrics_chart,
    create_dashboard_html,
)


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


def main() -> None:
    """Run the visualization example."""
    print("HydroSIS 可视化示例")
    print("=" * 50)
    
    # Load the model configuration
    repo_root = REPO_ROOT
    config_path = repo_root / "config" / "example_model.yaml"
    config = _load_model_config(config_path)
    
    # Load forcing data
    forcing = load_forcing(config.io.precipitation)
    print(f"加载了 {len(forcing)} 个子流域的降雨数据")
    
    # Create synthetic observations
    baseline_model = HydroSISModel.from_config(config)
    baseline_local = baseline_model.run(forcing)
    observations = baseline_model.accumulate_discharge(baseline_local)
    
    # Run workflow with scenarios
    print("\n运行模拟...")
    workflow_result = run_workflow(
        config,
        forcing,
        observations=observations,
        scenario_ids=["alternate_routing"],
        persist_outputs=False,
        generate_report=False,
    )
    
    print("模拟完成！")
    
    # Prepare data for visualization
    baseline_data = workflow_result.baseline.aggregated
    scenarios_data = {
        scenario_id: scenario.aggregated 
        for scenario_id, scenario in workflow_result.scenarios.items()
    }
    
    # Create hydrograph chart
    print("\n生成流量过程线图表...")
    hydrograph_html = create_hydrograph_chart(
        {**{"基准": baseline_data.get("S3", [])}, **{
            f"情景_{sid}": data.get("S3", []) 
            for sid, data in scenarios_data.items()
        }},
        title="出口子流域流量过程线",
    )
    
    # Create comparison chart
    print("生成情景对比图表...")
    comparison_html = create_comparison_chart(
        baseline_data,
        scenarios_data,
        title="情景对比",
        subbasin_id="S3",
    )
    
    # Create metrics chart
    print("生成评价指标图表...")
    metrics_data = {}
    if workflow_result.overall_scores:
        for score in workflow_result.overall_scores:
            metrics_data[score.model_id] = score.aggregated
    
    metrics_html = create_metrics_chart(
        metrics_data,
        title="模型评价指标",
    )
    
    # Create dashboard
    print("生成完整仪表板...")
    dashboard_html = create_dashboard_html(
        hydrograph_html,
        comparison_html,
        metrics_html,
        title="HydroSIS 模拟结果仪表板",
    )
    
    # Save visualizations
    output_dir = repo_root / "results" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual charts
    with open(output_dir / "hydrograph.html", "w", encoding="utf-8") as f:
        f.write(hydrograph_html)
    
    with open(output_dir / "comparison.html", "w", encoding="utf-8") as f:
        f.write(comparison_html)
    
    with open(output_dir / "metrics.html", "w", encoding="utf-8") as f:
        f.write(metrics_html)
    
    # Save dashboard
    with open(output_dir / "dashboard.html", "w", encoding="utf-8") as f:
        f.write(dashboard_html)
    
    print(f"\n可视化文件已保存到: {output_dir}")
    print("  - hydrograph.html: 流量过程线图表")
    print("  - comparison.html: 情景对比图表")
    print("  - metrics.html: 评价指标图表")
    print("  - dashboard.html: 完整仪表板")
    
    # Print some metrics
    if workflow_result.overall_scores:
        print("\n模型评价指标:")
        for score in workflow_result.overall_scores:
            print(f"  {score.model_id}:")
            for metric, value in score.aggregated.items():
                print(f"    {metric}: {value:.4f}")
    
    print("\n示例完成！")


if __name__ == "__main__":
    main()
