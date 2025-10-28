"""
Config Converter - 配置格式转换器

将HydroMind生成的JSON配置转换为HydroSIS的ModelConfig格式
"""

import copy
from typing import Dict, Any, List, Tuple
from pathlib import Path


class ConfigConverter:
    """配置格式转换器"""
    
    def __init__(self):
        """初始化转换器"""
        self.default_config = self._get_default_config()
    
    def hydromind_to_hydrosis(
        self,
        hydromind_config: Dict[str, Any],
        project_path: str = None
    ) -> Dict[str, Any]:
        """
        将HydroMind配置转换为HydroSIS ModelConfig
        
        Args:
            hydromind_config: HydroMind生成的配置
            project_path: 项目路径（用于数据路径）
            
        Returns:
            HydroSIS ModelConfig字典
        """
        # 深拷贝避免修改原配置
        config = copy.deepcopy(hydromind_config)
        
        # 补充默认值
        config = self.enrich_with_defaults(config)
        
        # 验证配置
        is_valid, errors = self.validate_config(config)
        if not is_valid:
            raise ValueError(f"配置验证失败: {errors}")
        
        # 转换为HydroSIS格式
        hydrosis_config = self._convert_format(config, project_path)
        
        return hydrosis_config
    
    def _convert_format(
        self,
        config: Dict[str, Any],
        project_path: str = None
    ) -> Dict[str, Any]:
        """转换配置格式"""
        
        # 基础路径
        if project_path:
            base_path = Path(project_path)
        else:
            base_path = Path(".")
        
        # 构建HydroSIS配置
        hydrosis_config = {
            # 流域划分配置
            "delineation": self._convert_delineation(
                config.get("delineation", {})
            ),
            
            # 产流模型配置
            "runoff": self._convert_runoff(
                config.get("runoff", {})
            ),
            
            # 汇流配置
            "routing": self._convert_routing(
                config.get("routing", {})
            ),
            
            # 参数分区
            "parameter_zones": self._convert_parameter_zones(
                config.get("parameter_zones", [])
            ),
            
            # IO配置
            "io": self._convert_io(
                config.get("io", {}),
                base_path
            ),
            
            # 情景配置
            "scenarios": config.get("scenarios", []),
            
            # 评价配置
            "evaluation": config.get("evaluation", {
                "metrics": ["nse", "rmse", "mae"]
            })
        }
        
        return hydrosis_config
    
    def _convert_delineation(self, delineation: Dict) -> Dict:
        """转换流域划分配置"""
        return {
            "method": delineation.get("method", "automatic"),
            "pour_points": delineation.get("pour_points", []),
            "dem_path": delineation.get("dem_path"),
            "burn_streams": delineation.get("burn_streams", False)
        }
    
    def _convert_runoff(self, runoff: Dict) -> Dict:
        """转换产流配置"""
        model_type = runoff.get("model_type", "HBV")
        parameters = runoff.get("parameters", {})
        
        # 确保参数类型正确
        typed_params = {}
        for key, value in parameters.items():
            if isinstance(value, (int, float)):
                typed_params[key] = float(value)
            else:
                typed_params[key] = value
        
        return {
            "model_type": model_type,
            "parameters": typed_params
        }
    
    def _convert_routing(self, routing: Dict) -> Dict:
        """转换汇流配置"""
        model_type = routing.get("model_type", "Muskingum")
        parameters = routing.get("parameters", {})
        
        # 类型转换
        typed_params = {}
        for key, value in parameters.items():
            if isinstance(value, (int, float)):
                typed_params[key] = float(value)
            else:
                typed_params[key] = value
        
        return {
            "model_type": model_type,
            "parameters": typed_params
        }
    
    def _convert_parameter_zones(self, zones: List[Dict]) -> List[Dict]:
        """转换参数分区配置"""
        converted_zones = []
        
        for zone in zones:
            converted_zones.append({
                "id": zone.get("id", f"zone_{len(converted_zones)}"),
                "controllers": zone.get("controllers", []),
                "parameters": zone.get("parameters", {})
            })
        
        return converted_zones
    
    def _convert_io(self, io: Dict, base_path: Path) -> Dict:
        """转换IO配置"""
        return {
            "precipitation": str(base_path / io.get("precipitation", "data/precipitation")),
            "dem": io.get("dem"),
            "discharge_observations": io.get("discharge_observations"),
            "results_directory": str(base_path / io.get("results_directory", "results"))
        }
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        验证配置完整性
        
        Returns:
            (是否有效, 错误列表)
        """
        errors = []
        
        # 检查必需字段
        if "runoff" not in config:
            errors.append("缺少runoff配置")
        elif "model_type" not in config["runoff"]:
            errors.append("runoff配置缺少model_type")
        
        if "routing" not in config:
            errors.append("缺少routing配置")
        elif "model_type" not in config["routing"]:
            errors.append("routing配置缺少model_type")
        
        # 检查参数
        if "runoff" in config:
            runoff_type = config["runoff"].get("model_type")
            params = config["runoff"].get("parameters", {})
            
            # 检查参数是否合理
            param_errors = self._validate_runoff_parameters(runoff_type, params)
            errors.extend(param_errors)
        
        return len(errors) == 0, errors
    
    def _validate_runoff_parameters(
        self,
        model_type: str,
        parameters: Dict[str, Any]
    ) -> List[str]:
        """验证产流模型参数"""
        errors = []
        
        # 参数范围定义
        param_ranges = {
            "HBV": {
                "fc": (50, 500),
                "beta": (0.5, 5.0),
                "lp": (0.1, 1.0)
            },
            "SCS": {
                "curve_number": (30, 98)
            },
            "XinAnJiang": {
                "wm": (50, 300),
                "b": (0.05, 0.6),
                "imp": (0.0, 0.2)
            }
        }
        
        if model_type in param_ranges:
            for param, (min_val, max_val) in param_ranges[model_type].items():
                if param in parameters:
                    value = parameters[param]
                    if not isinstance(value, (int, float)):
                        errors.append(f"参数{param}应为数值")
                    elif value < min_val or value > max_val:
                        errors.append(
                            f"参数{param}={value}超出合理范围[{min_val}, {max_val}]"
                        )
        
        return errors
    
    def enrich_with_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        补充默认值
        
        Args:
            config: 部分配置
            
        Returns:
            补充后的完整配置
        """
        # 深拷贝避免修改原配置
        enriched = copy.deepcopy(config)
        
        # 补充流域划分默认值
        if "delineation" not in enriched:
            enriched["delineation"] = {}
        if "method" not in enriched["delineation"]:
            enriched["delineation"]["method"] = "automatic"
        if "pour_points" not in enriched["delineation"]:
            enriched["delineation"]["pour_points"] = []
        
        # 补充产流默认值
        if "runoff" not in enriched:
            enriched["runoff"] = {"model_type": "HBV"}
        
        runoff_type = enriched["runoff"].get("model_type", "HBV")
        if "parameters" not in enriched["runoff"]:
            enriched["runoff"]["parameters"] = self._get_default_runoff_params(runoff_type)
        else:
            # 补充缺失的参数
            default_params = self._get_default_runoff_params(runoff_type)
            for key, value in default_params.items():
                if key not in enriched["runoff"]["parameters"]:
                    enriched["runoff"]["parameters"][key] = value
        
        # 补充汇流默认值
        if "routing" not in enriched:
            enriched["routing"] = {"model_type": "Muskingum"}
        
        routing_type = enriched["routing"].get("model_type", "Muskingum")
        if "parameters" not in enriched["routing"]:
            enriched["routing"]["parameters"] = self._get_default_routing_params(routing_type)
        
        # 补充IO默认值
        if "io" not in enriched:
            enriched["io"] = {}
        if "precipitation" not in enriched["io"]:
            enriched["io"]["precipitation"] = "data/precipitation"
        if "results_directory" not in enriched["io"]:
            enriched["io"]["results_directory"] = "results"
        
        # 补充参数分区
        if "parameter_zones" not in enriched:
            enriched["parameter_zones"] = []
        
        return enriched
    
    def _get_default_runoff_params(self, model_type: str) -> Dict[str, float]:
        """获取产流模型默认参数"""
        defaults = {
            "HBV": {
                "fc": 200.0,
                "beta": 2.0,
                "lp": 0.7,
                "k0": 0.1,
                "k1": 0.05,
                "k2": 0.01
            },
            "SCS": {
                "curve_number": 75.0
            },
            "XinAnJiang": {
                "wm": 180.0,
                "b": 0.3,
                "imp": 0.02,
                "recession": 0.95
            }
        }
        
        return defaults.get(model_type, {})
    
    def _get_default_routing_params(self, routing_type: str) -> Dict[str, float]:
        """获取汇流方法默认参数"""
        defaults = {
            "Muskingum": {
                "k": 2.0,
                "x": 0.2,
                "time_step": 1.0
            },
            "Lag": {
                "lag_steps": 2
            },
            "DynamicWave": {
                "time_step": 0.5,
                "wave_celerity": 1.5,
                "diffusivity": 100.0
            }
        }
        
        return defaults.get(routing_type, {})
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取完整默认配置"""
        return {
            "delineation": {
                "method": "automatic",
                "pour_points": []
            },
            "runoff": {
                "model_type": "HBV",
                "parameters": self._get_default_runoff_params("HBV")
            },
            "routing": {
                "model_type": "Muskingum",
                "parameters": self._get_default_routing_params("Muskingum")
            },
            "parameter_zones": [],
            "io": {
                "precipitation": "data/precipitation",
                "results_directory": "results"
            }
        }


if __name__ == "__main__":
    # 测试代码
    converter = ConfigConverter()
    
    print("=== 配置转换器测试 ===\n")
    
    # 测试1: 简单配置
    print("1. 测试简单配置...")
    simple_config = {
        "runoff": {
            "model_type": "HBV",
            "parameters": {"fc": 200}
        },
        "routing": {
            "model_type": "Muskingum"
        }
    }
    
    enriched = converter.enrich_with_defaults(simple_config)
    print(f"   补充后有 {len(enriched['runoff']['parameters'])} 个产流参数")
    print(f"   补充后有 {len(enriched['routing']['parameters'])} 个汇流参数")
    
    # 测试2: 验证
    print("\n2. 测试配置验证...")
    is_valid, errors = converter.validate_config(enriched)
    print(f"   验证结果: {'✅ 通过' if is_valid else '❌ 失败'}")
    if errors:
        for error in errors:
            print(f"   - {error}")
    
    # 测试3: 完整转换
    print("\n3. 测试完整转换...")
    hydrosis_config = converter.hydromind_to_hydrosis(simple_config)
    print(f"   ✅ 转换成功")
    print(f"   产流模型: {hydrosis_config['runoff']['model_type']}")
    print(f"   汇流方法: {hydrosis_config['routing']['model_type']}")
    
    print("\n配置转换器已就绪")
