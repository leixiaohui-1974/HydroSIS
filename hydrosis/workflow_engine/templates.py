"""预定义工作流模板"""
from typing import Dict, Any


class WorkflowTemplates:
    """工作流模板库"""
    
    @staticmethod
    def pour_points_only() -> Dict[str, Any]:
        """汇水点生成工作流"""
        return {
            "workflow": {
                "id": "pour_points_only",
                "name": "汇水点生成工作流",
                "version": "1.0",
                "description": "仅生成流域汇水点",
                "parameters": {
                    "dem_path": "data/dem.tif",
                    "output_dir": "results/pour_points_workflow"
                },
                "steps": [
                    {
                        "id": "terrain_processing",
                        "module": "terrain",
                        "inputs": {
                            "dem_path": "${parameters.dem_path}",
                            "method": "d8",
                            "output_dir": "${parameters.output_dir}/terrain"
                        },
                        "outputs": {
                            "flow_accumulation": "${parameters.output_dir}/terrain/flow_accumulation.tif"
                        }
                    },
                    {
                        "id": "pour_point_generation",
                        "module": "pour_points",
                        "depends_on": ["terrain_processing"],
                        "inputs": {
                            "flow_accumulation": "${steps.terrain_processing.outputs.flow_accumulation}",
                            "method": "auto",
                            "threshold": 1000,
                            "output_dir": "${parameters.output_dir}/pour_points"
                        }
                    }
                ],
                "outputs": {
                    "pour_points": "${steps.pour_point_generation.outputs.pour_points_geojson}"
                }
            }
        }
    
    @staticmethod
    def complete_simulation() -> Dict[str, Any]:
        """完整水文模拟工作流"""
        return {
            "workflow": {
                "id": "complete_simulation",
                "name": "完整水文模拟工作流",
                "version": "1.0",
                "description": "从DEM到流量模拟的完整流程",
                "parameters": {
                    "dem_path": "data/dem.tif",
                    "output_dir": "results/complete_workflow"
                },
                "steps": [
                    {
                        "id": "terrain",
                        "module": "terrain",
                        "inputs": {
                            "dem_path": "${parameters.dem_path}",
                            "output_dir": "${parameters.output_dir}/terrain"
                        }
                    },
                    {
                        "id": "pour_points",
                        "module": "pour_points",
                        "depends_on": ["terrain"],
                        "inputs": {
                            "flow_accumulation": "${steps.terrain.outputs.flow_accumulation}",
                            "output_dir": "${parameters.output_dir}/pour_points"
                        }
                    },
                    {
                        "id": "watershed",
                        "module": "watershed_delineation",
                        "depends_on": ["terrain", "pour_points"],
                        "inputs": {
                            "flow_direction": "${steps.terrain.outputs.flow_direction}",
                            "pour_points": "${steps.pour_points.outputs.pour_points_geojson}",
                            "output_dir": "${parameters.output_dir}/watersheds"
                        }
                    },
                    {
                        "id": "precipitation",
                        "module": "precipitation_generation",
                        "inputs": {
                            "duration_hours": 168,
                            "intensity_mm_h": 10.0,
                            "output_dir": "${parameters.output_dir}/precipitation"
                        }
                    },
                    {
                        "id": "runoff",
                        "module": "runoff_generation",
                        "depends_on": ["watershed", "precipitation"],
                        "inputs": {
                            "precipitation": "${steps.precipitation.outputs.precipitation_timeseries}",
                            "watersheds": "${steps.watershed.outputs.watersheds}",
                            "model": "hbv",
                            "output_dir": "${parameters.output_dir}/runoff"
                        }
                    },
                    {
                        "id": "routing",
                        "module": "routing",
                        "depends_on": ["runoff", "watershed"],
                        "inputs": {
                            "runoff": "${steps.runoff.outputs.runoff_timeseries}",
                            "watersheds": "${steps.watershed.outputs.watersheds}",
                            "method": "muskingum",
                            "output_dir": "${parameters.output_dir}/routing"
                        }
                    }
                ],
                "outputs": {
                    "discharge": "${steps.routing.outputs.discharge_timeseries}"
                }
            }
        }
    
    @staticmethod
    def get_template(template_id: str) -> Dict[str, Any]:
        """获取模板
        
        Args:
            template_id: 模板ID
        
        Returns:
            工作流定义字典
        """
        templates = {
            "pour_points_only": WorkflowTemplates.pour_points_only,
            "complete_simulation": WorkflowTemplates.complete_simulation,
        }
        
        template_func = templates.get(template_id)
        if template_func is None:
            raise KeyError(f"模板不存在: {template_id}")
        
        return template_func()
    
    @staticmethod
    def list_templates() -> list[str]:
        """列出所有可用模板"""
        return [
            "pour_points_only",
            "complete_simulation",
        ]
