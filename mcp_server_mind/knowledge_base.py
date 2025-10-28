"""
Knowledge Base - 水文建模知识库

存储和检索水文建模相关的知识，包括：
1. 模型选择规则
2. 参数经验范围
3. 诊断规则库
4. 文献参考
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ModelKnowledge:
    """模型知识"""
    model_type: str
    description: str
    suitable_for: List[str]
    parameters: Dict[str, Dict[str, Any]]
    references: List[str]


@dataclass
class ParameterKnowledge:
    """参数知识"""
    param_name: str
    model_type: str
    typical_range: tuple
    units: str
    physical_meaning: str
    calibration_priority: int  # 1-5, 1最高


@dataclass
class DiagnosticRule:
    """诊断规则"""
    rule_id: str
    symptoms: List[str]
    probable_causes: List[Dict[str, Any]]
    recommended_actions: List[Dict[str, Any]]


class KnowledgeBase:
    """知识库管理器"""
    
    def __init__(self, kb_dir: Optional[Path] = None):
        """
        初始化知识库
        
        Args:
            kb_dir: 知识库目录，默认为 knowledge_base/
        """
        if kb_dir is None:
            current_dir = Path(__file__).parent
            kb_dir = current_dir / "knowledge_base"
        
        self.kb_dir = Path(kb_dir)
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化各类知识
        self.models = self._load_model_knowledge()
        self.parameters = self._load_parameter_knowledge()
        self.diagnostic_rules = self._load_diagnostic_rules()
    
    def _load_model_knowledge(self) -> Dict[str, ModelKnowledge]:
        """加载模型知识"""
        # 内置模型知识
        models_data = {
            "HBV": {
                "model_type": "HBV",
                "description": "HBV模型是概念性水文模型，包含雪蓄、土壤、地下水多仓结构",
                "suitable_for": [
                    "有雪融过程的流域",
                    "中高纬度地区",
                    "需要考虑土壤蓄水的流域",
                    "山地流域"
                ],
                "parameters": {
                    "fc": {
                        "description": "田间持水量",
                        "range": [100, 400],
                        "unit": "mm",
                        "typical": 200
                    },
                    "beta": {
                        "description": "非线性系数",
                        "range": [1.0, 4.0],
                        "unit": "-",
                        "typical": 2.0
                    },
                    "lp": {
                        "description": "蒸发阈值",
                        "range": [0.3, 0.9],
                        "unit": "-",
                        "typical": 0.7
                    }
                },
                "references": [
                    "Bergström, S. (1992). The HBV model - its structure and applications.",
                    "Seibert, J. & Vis, M. (2012). Teaching hydrological modeling."
                ]
            },
            "SCS": {
                "model_type": "SCS",
                "description": "SCS曲线数法，基于土壤类型和土地利用的经验产流模型",
                "suitable_for": [
                    "设计洪水计算",
                    "数据缺乏地区",
                    "小流域快速评估",
                    "城市流域"
                ],
                "parameters": {
                    "curve_number": {
                        "description": "曲线数",
                        "range": [30, 98],
                        "unit": "-",
                        "typical": 75
                    }
                },
                "references": [
                    "USDA-NRCS (2004). Hydrology National Engineering Handbook."
                ]
            },
            "XinAnJiang": {
                "model_type": "XinAnJiang",
                "description": "新安江模型，中国经典的分布式水文模型",
                "suitable_for": [
                    "湿润半湿润地区",
                    "中国南方流域",
                    "需要考虑蓄满产流的流域"
                ],
                "parameters": {
                    "wm": {
                        "description": "流域平均蓄水容量",
                        "range": [100, 300],
                        "unit": "mm",
                        "typical": 180
                    },
                    "b": {
                        "description": "蓄水容量曲线指数",
                        "range": [0.1, 0.5],
                        "unit": "-",
                        "typical": 0.3
                    },
                    "imp": {
                        "description": "不透水面积占比",
                        "range": [0.0, 0.1],
                        "unit": "-",
                        "typical": 0.02
                    }
                },
                "references": [
                    "Zhao, R.J. (1992). The Xinanjiang model applied in China.",
                    "赵人俊 (1984). 新安江模型."
                ]
            }
        }
        
        # 转换为ModelKnowledge对象
        models = {}
        for model_type, data in models_data.items():
            models[model_type] = ModelKnowledge(**data)
        
        # 尝试从文件加载更多
        kb_file = self.kb_dir / "models.json"
        if kb_file.exists():
            with open(kb_file, 'r', encoding='utf-8') as f:
                extra_data = json.load(f)
                for model_type, data in extra_data.items():
                    if model_type not in models:
                        models[model_type] = ModelKnowledge(**data)
        
        return models
    
    def _load_parameter_knowledge(self) -> List[ParameterKnowledge]:
        """加载参数知识"""
        params = []
        
        # HBV参数
        params.extend([
            ParameterKnowledge(
                param_name="fc",
                model_type="HBV",
                typical_range=(100, 400),
                units="mm",
                physical_meaning="田间持水量，代表土壤能够持有的最大水量",
                calibration_priority=1
            ),
            ParameterKnowledge(
                param_name="beta",
                model_type="HBV",
                typical_range=(1.0, 4.0),
                units="-",
                physical_meaning="土壤蓄水非线性系数，控制产流速率",
                calibration_priority=2
            ),
            ParameterKnowledge(
                param_name="lp",
                model_type="HBV",
                typical_range=(0.3, 0.9),
                units="-",
                physical_meaning="蒸发阈值，低于此值蒸发受限",
                calibration_priority=3
            )
        ])
        
        return params
    
    def _load_diagnostic_rules(self) -> List[DiagnosticRule]:
        """加载诊断规则"""
        rules = [
            DiagnosticRule(
                rule_id="peak_underestimation",
                symptoms=[
                    "模拟峰值流量明显小于实测",
                    "NSE较低但水量平衡尚可",
                    "峰值相对误差 > 15%"
                ],
                probable_causes=[
                    {
                        "cause": "产流参数过大（如HBV的FC过高）",
                        "probability": 0.7,
                        "evidence": "FC参数控制产流量，过大导致产流不足"
                    },
                    {
                        "cause": "降雨插值误差",
                        "probability": 0.5,
                        "evidence": "雨量站稀疏导致峰值降雨被低估"
                    },
                    {
                        "cause": "时间步长过粗",
                        "probability": 0.3,
                        "evidence": "日步长可能无法捕捉短历时强降雨"
                    }
                ],
                recommended_actions=[
                    {
                        "action": "减小产流参数（FC、WM等）",
                        "priority": 1,
                        "expected_improvement": "峰值流量增加10-20%",
                        "implementation": "使用calibrate_parameters工具重新率定"
                    },
                    {
                        "action": "增加雨量站密度或使用格点降雨",
                        "priority": 2,
                        "expected_improvement": "降雨输入更准确",
                        "implementation": "补充雨量站或使用雷达/卫星降雨"
                    }
                ]
            ),
            DiagnosticRule(
                rule_id="timing_error",
                symptoms=[
                    "峰现时间偏差 > 6小时",
                    "洪峰幅值可能准确但时间不对"
                ],
                probable_causes=[
                    {
                        "cause": "汇流参数不当",
                        "probability": 0.8,
                        "evidence": "Muskingum的K值控制传播时间"
                    },
                    {
                        "cause": "河网结构简化",
                        "probability": 0.4,
                        "evidence": "子流域划分过粗导致汇流时间误差"
                    }
                ],
                recommended_actions=[
                    {
                        "action": "调整汇流参数K",
                        "priority": 1,
                        "expected_improvement": "峰现时间校正",
                        "implementation": "如果峰现偏早，增大K值；偏晚则减小K值"
                    }
                ]
            ),
            DiagnosticRule(
                rule_id="water_balance_error",
                symptoms=[
                    "总径流量与实测偏差 > 10%",
                    "PBIAS指标异常"
                ],
                probable_causes=[
                    {
                        "cause": "蒸发参数不准确",
                        "probability": 0.6,
                        "evidence": "蒸发是水量平衡的重要组成"
                    },
                    {
                        "cause": "产流参数系统性偏差",
                        "probability": 0.7,
                        "evidence": "参数导致产流总量偏多或偏少"
                    }
                ],
                recommended_actions=[
                    {
                        "action": "检查蒸发数据和参数",
                        "priority": 1,
                        "expected_improvement": "水量平衡改善",
                        "implementation": "验证潜在蒸发数据，调整蒸发系数"
                    }
                ]
            )
        ]
        
        return rules
    
    # ============ 查询接口 ============
    
    def get_model_info(self, model_type: str) -> Optional[ModelKnowledge]:
        """获取模型信息"""
        return self.models.get(model_type)
    
    def suggest_model(self, basin_features: Dict[str, Any]) -> List[str]:
        """
        根据流域特征推荐模型
        
        Args:
            basin_features: 流域特征字典
                - climate: 气候类型 (humid/semi-humid/arid)
                - has_snow: 是否有雪融
                - area_km2: 流域面积
                - data_availability: 数据丰富度 (rich/moderate/poor)
                
        Returns:
            推荐的模型列表（按优先级排序）
        """
        climate = basin_features.get("climate", "humid")
        has_snow = basin_features.get("has_snow", False)
        area = basin_features.get("area_km2", 1000)
        data = basin_features.get("data_availability", "moderate")
        
        suggestions = []
        
        # HBV适合有雪融、数据较丰富的流域
        if has_snow or data == "rich":
            suggestions.append("HBV")
        
        # SCS适合数据缺乏、小流域
        if data == "poor" or area < 100:
            suggestions.append("SCS")
        
        # 新安江适合湿润地区
        if climate in ["humid", "semi-humid"] and not has_snow:
            suggestions.append("XinAnJiang")
        
        # 如果没有匹配，默认推荐HBV
        if not suggestions:
            suggestions = ["HBV", "SCS"]
        
        return suggestions
    
    def get_parameter_range(
        self,
        model_type: str,
        param_name: str
    ) -> Optional[tuple]:
        """获取参数推荐范围"""
        for param in self.parameters:
            if param.model_type == model_type and param.param_name == param_name:
                return param.typical_range
        return None
    
    def get_all_parameters(self, model_type: str) -> List[ParameterKnowledge]:
        """获取某模型的所有参数"""
        return [p for p in self.parameters if p.model_type == model_type]
    
    def diagnose(self, symptoms: List[str]) -> List[DiagnosticRule]:
        """
        根据症状诊断问题
        
        Args:
            symptoms: 症状列表
            
        Returns:
            匹配的诊断规则（按匹配度排序）
        """
        matched_rules = []
        
        for rule in self.diagnostic_rules:
            # 计算匹配度
            match_count = sum(
                1 for symptom in symptoms
                if any(s in symptom for s in rule.symptoms)
            )
            
            if match_count > 0:
                matched_rules.append((match_count, rule))
        
        # 按匹配度排序
        matched_rules.sort(key=lambda x: x[0], reverse=True)
        
        return [rule for _, rule in matched_rules]
    
    def search(self, query: str) -> Dict[str, Any]:
        """
        搜索知识库
        
        Args:
            query: 搜索关键词
            
        Returns:
            搜索结果
        """
        results = {
            "models": [],
            "parameters": [],
            "rules": []
        }
        
        query_lower = query.lower()
        
        # 搜索模型
        for model_type, model in self.models.items():
            if (query_lower in model_type.lower() or
                query_lower in model.description.lower()):
                results["models"].append(model_type)
        
        # 搜索参数
        for param in self.parameters:
            if (query_lower in param.param_name.lower() or
                query_lower in param.physical_meaning.lower()):
                results["parameters"].append(param.param_name)
        
        # 搜索规则
        for rule in self.diagnostic_rules:
            if any(query_lower in s.lower() for s in rule.symptoms):
                results["rules"].append(rule.rule_id)
        
        return results


if __name__ == "__main__":
    # 测试代码
    kb = KnowledgeBase()
    
    print("=== 模型推荐 ===")
    basin = {
        "climate": "humid",
        "has_snow": True,
        "area_km2": 5000,
        "data_availability": "rich"
    }
    models = kb.suggest_model(basin)
    print(f"推荐模型: {models}")
    
    print("\n=== 模型信息 ===")
    hbv_info = kb.get_model_info("HBV")
    if hbv_info:
        print(f"模型: {hbv_info.model_type}")
        print(f"描述: {hbv_info.description}")
        print(f"适用: {hbv_info.suitable_for}")
    
    print("\n=== 参数查询 ===")
    params = kb.get_all_parameters("HBV")
    for param in params:
        print(f"  {param.param_name}: {param.typical_range} {param.units}")
        print(f"    {param.physical_meaning}")
    
    print("\n=== 问题诊断 ===")
    symptoms = ["模拟峰值流量明显小于实测", "NSE较低"]
    rules = kb.diagnose(symptoms)
    for rule in rules:
        print(f"规则: {rule.rule_id}")
        print(f"  可能原因: {[c['cause'] for c in rule.probable_causes]}")
