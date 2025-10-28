"""
Cognitive Tools - HydroMind 认知工具集

12个基于LLM的认知工具：
- 理解层 (3个): 意图识别、实体抽取、需求验证
- 配置层 (3个): 配置生成、参数推荐、情景设计
- 分析层 (3个): 结果解读、问题诊断、模型对比
- 报告层 (3个): 叙述生成、执行报告、智能问答
"""

import json
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

try:
    from .llm_backend import LLMBackend, LLMMessage, create_default_backend
    from .prompt_templates import PromptTemplateManager
    from .knowledge_base import KnowledgeBase
except ImportError:
    from llm_backend import LLMBackend, LLMMessage, create_default_backend
    from prompt_templates import PromptTemplateManager
    from knowledge_base import KnowledgeBase


class CognitiveTools:
    """认知工具集 - HydroMind Agent的核心"""
    
    def __init__(
        self,
        llm_backend: Optional[LLMBackend] = None,
        prompt_manager: Optional[PromptTemplateManager] = None,
        knowledge_base: Optional[KnowledgeBase] = None
    ):
        """
        初始化认知工具
        
        Args:
            llm_backend: LLM后端（如果不提供，自动创建）
            prompt_manager: 提示词管理器
            knowledge_base: 知识库
        """
        self.llm = llm_backend or create_default_backend()
        self.prompts = prompt_manager or PromptTemplateManager()
        self.kb = knowledge_base or KnowledgeBase()
        
        print(f"✅ HydroMind认知工具初始化完成")
        print(f"   LLM后端: {self.llm.__class__.__name__}")
        print(f"   可用性: {self.llm.is_available()}")
    
    # ============ 理解层工具 (Understanding Layer) ============
    
    async def parse_user_intent(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        工具1: 解析用户意图
        
        识别用户想要做什么，分解为子任务
        """
        user_input = args.get("user_input", "")
        conversation_history = args.get("conversation_history", [])
        
        # 准备提示词
        available_actions = [
            "create_project", "configure_model", "upload_data",
            "run_simulation", "calibrate_params", "generate_report",
            "analyze_results", "compare_scenarios"
        ]
        
        prompt_text = self.prompts.render("intent_parsing", {
            "user_input": user_input,
            "conversation_history": json.dumps(conversation_history, ensure_ascii=False),
            "available_actions": json.dumps(available_actions, ensure_ascii=False)
        })
        
        # 调用LLM
        messages = [
            LLMMessage(role="system", content="你是HydroSIS意图理解专家"),
            LLMMessage(role="user", content=prompt_text)
        ]
        
        response = await self.llm.complete(messages, temperature=0.3)
        
        # 解析响应
        try:
            result = json.loads(response.content)
        except json.JSONDecodeError:
            # 如果解析失败，返回默认结果
            result = {
                "action": "general_query",
                "sub_intents": [],
                "confidence": 0.5,
                "missing_info": [],
                "clarification_needed": False
            }
        
        return result
    
    async def extract_entities(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        工具2: 提取建模实体
        
        从用户输入中抽取流域、模型、时间等关键信息
        """
        user_input = args.get("user_input", "")
        
        # 实体Schema
        entity_schema = {
            "basin": {"name": "str", "area_km2": "float", "location": "dict"},
            "model": {"runoff_type": "str", "routing_type": "str"},
            "time_period": {"start": "date", "end": "date"},
            "objectives": ["list"],
            "data_requirements": ["list"]
        }
        
        prompt_text = self.prompts.render("entity_extraction", {
            "text": user_input,
            "entity_schema": json.dumps(entity_schema, ensure_ascii=False, indent=2)
        })
        
        messages = [
            LLMMessage(role="system", content="你是水文建模实体抽取专家"),
            LLMMessage(role="user", content=prompt_text)
        ]
        
        response = await self.llm.complete(messages, temperature=0.2)
        
        try:
            entities = json.loads(response.content)
        except json.JSONDecodeError:
            entities = {
                "basin": None,
                "model": None,
                "time_period": None,
                "objectives": [],
                "data_requirements": []
            }
        
        return entities
    
    async def validate_requirements(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        工具3: 验证需求完整性和合理性
        
        检查需求是否可行，是否缺少信息
        """
        requirements = args.get("requirements", {})
        
        # 基于规则的基本验证
        issues = []
        completeness_score = 1.0
        
        # 检查必需字段
        if not requirements.get("basin"):
            issues.append({
                "type": "missing_data",
                "severity": "error",
                "message": "缺少流域信息",
                "suggestion": "请提供流域名称或位置"
            })
            completeness_score -= 0.3
        
        if not requirements.get("model"):
            issues.append({
                "type": "missing_data",
                "severity": "warning",
                "message": "未指定模型类型",
                "suggestion": "将根据流域特征自动推荐模型"
            })
            completeness_score -= 0.1
        
        # LLM深度验证
        if issues or completeness_score < 1.0:
            validation_rules = {
                "required_fields": ["basin", "time_period"],
                "optional_fields": ["model", "data"],
                "constraints": {
                    "area_km2": "> 0",
                    "time_period": "至少30天"
                }
            }
            
            prompt_text = self.prompts.render("requirement_validation", {
                "requirements": json.dumps(requirements, ensure_ascii=False),
                "validation_rules": json.dumps(validation_rules, ensure_ascii=False)
            })
            
            messages = [
                LLMMessage(role="system", content="你是水文建模需求验证专家"),
                LLMMessage(role="user", content=prompt_text)
            ]
            
            response = await self.llm.complete(messages, temperature=0.3)
            
            try:
                llm_validation = json.loads(response.content)
                # 合并规则验证和LLM验证
                issues.extend(llm_validation.get("issues", []))
                completeness_score = min(completeness_score, llm_validation.get("completeness_score", 1.0))
            except:
                pass
        
        return {
            "is_valid": len([i for i in issues if i["severity"] == "error"]) == 0,
            "completeness_score": completeness_score,
            "issues": issues,
            "feasibility": {
                "data_availability": 0.8,
                "computational_cost": "medium",
                "expected_accuracy": "good"
            }
        }
    
    # ============ 配置层工具 (Configuration Layer) ============
    
    async def generate_model_config(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        工具4: 生成模型配置 ⭐ 核心工具
        
        基于意图和实体生成完整的ModelConfig
        """
        intent = args.get("intent", {})
        entities = args.get("entities", {})
        
        # 首先尝试基于知识库生成
        basin = entities.get("basin", {})
        model_pref = entities.get("model", {})
        
        # 如果没有指定模型，推荐模型
        if not model_pref or not model_pref.get("runoff_type"):
            basin_features = {
                "climate": "humid",  # 可从位置推断
                "has_snow": False,
                "area_km2": basin.get("area_km2", 1000),
                "data_availability": "moderate"
            }
            suggested_models = self.kb.suggest_model(basin_features)
            runoff_type = suggested_models[0]
        else:
            runoff_type = model_pref.get("runoff_type", "HBV")
        
        # 获取模型信息
        model_info = self.kb.get_model_info(runoff_type)
        
        # 构建基础配置
        base_config = {
            "delineation": {
                "method": "automatic",
                "pour_points": []
            },
            "runoff": {
                "model_type": runoff_type,
                "parameters": {}
            },
            "routing": {
                "model_type": model_pref.get("routing_type", "Muskingum"),
                "parameters": {
                    "k": 2.0,
                    "x": 0.2
                }
            },
            "parameter_zones": [],
            "io": {
                "precipitation": "data/precipitation",
                "results_directory": "results"
            }
        }
        
        # 添加推荐参数
        if model_info:
            for param_name, param_info in model_info.parameters.items():
                base_config["runoff"]["parameters"][param_name] = param_info.get("typical", param_info["range"][0])
        
        # 使用LLM优化配置
        config_examples = json.dumps({
            "example": base_config
        }, ensure_ascii=False, indent=2)
        
        prompt_text = self.prompts.render("config_generation", {
            "intent": json.dumps(intent, ensure_ascii=False),
            "entities": json.dumps(entities, ensure_ascii=False),
            "config_examples": config_examples,
            "validation_schema": "{}"
        })
        
        messages = [
            LLMMessage(role="system", content="你是HydroSIS配置生成专家"),
            LLMMessage(role="user", content=prompt_text)
        ]
        
        response = await self.llm.complete(messages, temperature=0.4)
        
        try:
            config = json.loads(response.content)
        except:
            config = base_config
        
        return {
            "config": config,
            "config_summary": f"基于{runoff_type}模型的{basin.get('name', '流域')}模拟配置",
            "rationale": {
                "model_selection": f"选择{runoff_type}模型",
                "parameter_defaults": "参数基于区域经验值",
                "data_strategy": "使用默认数据路径"
            },
            "warnings": ["建议进行参数率定以提高精度"] if runoff_type != "SCS" else []
        }
    
    async def suggest_parameters(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        工具5: 参数推荐
        
        基于流域特征和模型类型推荐参数
        """
        basin_features = args.get("basin_features", {})
        model_type = args.get("model_type", "HBV")
        
        # 从知识库获取参数范围
        params_knowledge = self.kb.get_all_parameters(model_type)
        
        suggested_parameters = {}
        rationale = {}
        
        for param in params_knowledge:
            # 使用经验值
            typical_range = param.typical_range
            typical_value = (typical_range[0] + typical_range[1]) / 2
            
            suggested_parameters[param.param_name] = {
                "value": typical_value,
                "range": list(typical_range),
                "confidence": 0.7,
                "unit": param.units
            }
            
            rationale[param.param_name] = param.physical_meaning
        
        # 获取率定优先级
        calibration_priority = [
            p.param_name for p in sorted(params_knowledge, key=lambda x: x.calibration_priority)
        ]
        
        return {
            "suggested_parameters": suggested_parameters,
            "rationale": rationale,
            "references": [
                f"{model_type}模型参数经验值",
                "基于知识库推荐"
            ],
            "calibration_priority": calibration_priority
        }
    
    async def design_scenarios(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        工具6: 情景设计
        
        根据分析目标设计对比情景
        """
        objective = args.get("analysis_objective", "")
        baseline = args.get("baseline_config", {})
        
        # 基于目标生成情景
        scenarios = []
        
        if "土地利用" in objective or "城镇化" in objective:
            scenarios.extend([
                {
                    "id": "baseline",
                    "name": "当前状态",
                    "description": "现状土地利用",
                    "modifications": {}
                },
                {
                    "id": "urbanization",
                    "name": "城镇化情景",
                    "description": "城镇化率提高20%",
                    "modifications": {
                        "runoff.parameters.curve_number": "+10"
                    },
                    "rationale": "城镇化增加不透水面，CN值升高"
                }
            ])
        elif "气候" in objective:
            scenarios.extend([
                {
                    "id": "baseline",
                    "name": "历史气候",
                    "description": "历史气候条件",
                    "modifications": {}
                },
                {
                    "id": "climate_change",
                    "name": "气候变化情景",
                    "description": "降雨增加10%",
                    "modifications": {
                        "precipitation_scaling": 1.1
                    },
                    "rationale": "气候变化导致降雨增加"
                }
            ])
        else:
            # 默认参数敏感性情景
            scenarios = [
                {
                    "id": "baseline",
                    "name": "基准情景",
                    "description": "默认参数",
                    "modifications": {}
                }
            ]
        
        return {
            "scenarios": scenarios,
            "comparison_plan": {
                "reference": "baseline",
                "metrics": ["nse", "rmse", "peak_flow_change"],
                "visualization": ["hydrograph_overlay", "metric_comparison"]
            }
        }
    
    # ============ 分析层工具 (Analysis Layer) ============
    
    async def interpret_results(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        工具7: 结果解读
        
        智能解读模拟结果
        """
        results = args.get("simulation_results", {})
        config = args.get("model_config", {})
        objective = args.get("user_objective", "")
        
        # 提取关键指标
        metrics = results.get("metrics", {})
        nse = metrics.get("nse", 0.0)
        rmse = metrics.get("rmse", 0.0)
        
        # 评价等级
        if nse > 0.8:
            performance_level = "excellent"
            key_message = "模型精度优秀"
        elif nse > 0.6:
            performance_level = "good"
            key_message = "模型精度良好"
        elif nse > 0.4:
            performance_level = "fair"
            key_message = "模型精度尚可"
        else:
            performance_level = "poor"
            key_message = "模型精度较差，需要改进"
        
        # 使用LLM生成详细解读
        prompt_text = self.prompts.render("result_interpretation", {
            "results": json.dumps(results, ensure_ascii=False)[:1000],
            "config": json.dumps(config, ensure_ascii=False)[:1000],
            "objective": objective,
            "evaluation_criteria": json.dumps({
                "excellent": "NSE > 0.8",
                "good": "NSE 0.6-0.8",
                "fair": "NSE 0.4-0.6",
                "poor": "NSE < 0.4"
            })
        })
        
        messages = [
            LLMMessage(role="system", content="你是水文模拟结果解读专家"),
            LLMMessage(role="user", content=prompt_text)
        ]
        
        response = await self.llm.complete(messages, temperature=0.5)
        
        try:
            llm_interpretation = json.loads(response.content)
        except:
            llm_interpretation = {}
        
        return {
            "overall_assessment": {
                "performance_level": performance_level,
                "key_message": key_message,
                "confidence": 0.8
            },
            "detailed_findings": llm_interpretation.get("detailed_findings", []),
            "strengths": llm_interpretation.get("strengths", []),
            "weaknesses": llm_interpretation.get("weaknesses", []),
            "root_causes": llm_interpretation.get("root_causes", {})
        }
    
    async def diagnose_issues(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        工具8: 问题诊断
        
        诊断模型表现不佳的原因
        """
        results = args.get("poor_results", {})
        config = args.get("model_config", {})
        
        # 识别症状
        symptoms = []
        metrics = results.get("metrics", {})
        
        if metrics.get("nse", 1.0) < 0.6:
            symptoms.append("NSE较低")
        
        if metrics.get("peak_error", 0) > 0.15:
            symptoms.append("峰值误差较大")
        
        # 使用知识库诊断
        diagnostic_rules = self.kb.diagnose(symptoms)
        
        identified_issues = []
        for rule in diagnostic_rules[:2]:  # 取前2个最相关的
            identified_issues.append({
                "issue_type": rule.rule_id,
                "severity": "high" if symptoms else "medium",
                "symptoms": rule.symptoms,
                "probable_causes": rule.probable_causes,
                "recommended_actions": rule.recommended_actions
            })
        
        return {
            "identified_issues": identified_issues,
            "diagnostic_confidence": 0.75,
            "knowledge_sources": ["规则库", "经验知识"]
        }
    
    async def compare_models(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        工具9: 模型对比
        
        对比多个模型的表现
        """
        models_results = args.get("models_results", {})
        
        # 简单排序
        ranking = []
        for model_name, result in models_results.items():
            nse = result.get("metrics", {}).get("nse", 0)
            ranking.append({
                "model": model_name,
                "score": nse,
                "rank": 0
            })
        
        # 排序
        ranking.sort(key=lambda x: x["score"], reverse=True)
        for i, item in enumerate(ranking):
            item["rank"] = i + 1
        
        return {
            "comparison_summary": f"{ranking[0]['model']}模型表现最优",
            "ranking": ranking,
            "comparative_analysis": {},
            "selection_recommendation": {
                "recommended_model": ranking[0]["model"] if ranking else "HBV",
                "rationale": "基于NSE指标评价",
                "trade_offs": ""
            }
        }
    
    # ============ 报告层工具 (Reporting Layer) ============
    
    async def generate_narrative(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        工具10: 生成叙述
        
        生成特定章节的自然语言叙述
        """
        section = args.get("section", "executive_summary")
        context = args.get("context", {})
        
        # 选择模板
        template_name = f"narrative_{section}"
        if template_name not in self.prompts.get_template_names():
            template_name = "narrative_executive_summary"
        
        prompt_text = self.prompts.render(template_name, context)
        
        messages = [
            LLMMessage(role="system", content="你是资深水文专家"),
            LLMMessage(role="user", content=prompt_text)
        ]
        
        response = await self.llm.complete(messages, temperature=0.7)
        
        return {
            "narrative": response.content,
            "key_points": self._extract_key_points(response.content),
            "supporting_data": {}
        }
    
    async def create_executive_report(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        工具11: 生成执行报告 ⭐ 核心工具
        
        生成完整的自然语言分析报告
        """
        workflow_result = args.get("workflow_result", {})
        model_config = args.get("model_config", {})
        report_type = args.get("report_type", "executive")
        
        # 构建报告上下文
        context = {
            "basin_name": "测试流域",
            "start_date": "2020-01-01",
            "end_date": "2020-12-31",
            "runoff_model": model_config.get("runoff", {}).get("model_type", "HBV"),
            "routing_model": model_config.get("routing", {}).get("model_type", "Muskingum"),
            "metrics": json.dumps(workflow_result.get("metrics", {}), ensure_ascii=False),
            "user_objective": args.get("user_objective", "水文模拟")
        }
        
        # 生成执行摘要
        summary_result = await self.generate_narrative({
            "section": "executive_summary",
            "context": context
        })
        
        # 组装报告
        report_content = f"""# {context['basin_name']} 水文模拟分析报告

## 执行摘要

{summary_result['narrative']}

## 模型配置

- 产流模型: {context['runoff_model']}
- 汇流方法: {context['routing_model']}
- 模拟时段: {context['start_date']} 至 {context['end_date']}

## 模拟结果

{context['metrics']}

## 建议

基于当前模拟结果，建议：
1. 进行参数率定以提高精度
2. 补充数据以改善输入质量
3. 验证关键子流域的模拟效果

---
*报告由HydroMind智能生成*
"""
        
        return {
            "report": {
                "format": "markdown",
                "content": report_content,
                "sections": {
                    "executive_summary": summary_result['narrative']
                }
            },
            "metadata": {
                "word_count": len(report_content),
                "generated_at": "2025-10-28"
            }
        }
    
    async def answer_questions(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        工具12: 智能问答
        
        回答用户关于结果的问题
        """
        question = args.get("question", "")
        context = args.get("context", {})
        
        # 构建对话上下文
        messages = [
            LLMMessage(
                role="system",
                content="你是水文模拟专家，基于模拟结果回答用户问题。"
            ),
            LLMMessage(
                role="user",
                content=f"问题: {question}\n\n上下文: {json.dumps(context, ensure_ascii=False)[:500]}"
            )
        ]
        
        response = await self.llm.complete(messages, temperature=0.6)
        
        return {
            "answer": response.content,
            "confidence": 0.8,
            "supporting_evidence": [],
            "related_questions": [
                "模型精度如何改进？",
                "参数如何调整？"
            ],
            "suggested_actions": []
        }
    
    # ============ 辅助方法 ============
    
    def _extract_key_points(self, text: str) -> List[str]:
        """从文本中提取关键点"""
        # 简单实现：按句子分割
        sentences = [s.strip() for s in text.split('。') if s.strip()]
        return sentences[:5]  # 返回前5句
    
    def get_tool_list(self) -> List[Dict[str, str]]:
        """获取所有工具列表"""
        return [
            {"name": "parse_user_intent", "category": "understanding", "description": "解析用户意图"},
            {"name": "extract_entities", "category": "understanding", "description": "提取建模实体"},
            {"name": "validate_requirements", "category": "understanding", "description": "验证需求"},
            {"name": "generate_model_config", "category": "configuration", "description": "生成模型配置"},
            {"name": "suggest_parameters", "category": "configuration", "description": "推荐参数"},
            {"name": "design_scenarios", "category": "configuration", "description": "设计情景"},
            {"name": "interpret_results", "category": "analysis", "description": "解读结果"},
            {"name": "diagnose_issues", "category": "analysis", "description": "诊断问题"},
            {"name": "compare_models", "category": "analysis", "description": "对比模型"},
            {"name": "generate_narrative", "category": "reporting", "description": "生成叙述"},
            {"name": "create_executive_report", "category": "reporting", "description": "生成执行报告"},
            {"name": "answer_questions", "category": "reporting", "description": "智能问答"},
        ]


# 测试入口
if __name__ == "__main__":
    async def test():
        print("=== HydroMind 认知工具测试 ===\n")
        
        tools = CognitiveTools()
        
        print(f"\n可用工具({len(tools.get_tool_list())}个):")
        for tool in tools.get_tool_list():
            print(f"  [{tool['category']}] {tool['name']}: {tool['description']}")
        
        # 测试工具1: 意图识别
        print("\n=== 测试: 意图识别 ===")
        result = await tools.parse_user_intent({
            "user_input": "我想建立长江上游的HBV模型并运行模拟",
            "conversation_history": []
        })
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        # 测试工具2: 实体抽取
        print("\n=== 测试: 实体抽取 ===")
        result = await tools.extract_entities({
            "user_input": "流域面积5000平方公里，使用2020年数据"
        })
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    asyncio.run(test())
