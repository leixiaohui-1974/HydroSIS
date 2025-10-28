"""
Prompt Template Manager - 提示词模板管理系统

管理所有认知工具的提示词模板，支持：
1. 模板加载和渲染
2. 变量替换
3. 模板版本管理
4. 多语言支持
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
from string import Template


class PromptTemplateManager:
    """提示词模板管理器"""
    
    def __init__(self, templates_dir: Optional[Path] = None):
        """
        初始化模板管理器
        
        Args:
            templates_dir: 模板目录路径，默认为 prompt_templates/
        """
        if templates_dir is None:
            current_dir = Path(__file__).parent
            templates_dir = current_dir / "prompt_templates"
        
        self.templates_dir = Path(templates_dir)
        self._templates_cache: Dict[str, str] = {}
        self._load_builtin_templates()
    
    def _load_builtin_templates(self):
        """加载内置模板"""
        self._templates_cache = {
            # ============ 理解层模板 ============
            "intent_parsing": """你是HydroSIS水文建模系统的意图理解专家。

【任务】
分析用户输入，识别用户的建模意图。

【用户输入】
$user_input

【对话历史】
$conversation_history

【可用操作】
$available_actions

【输出要求】
返回JSON格式，包含以下字段：
{
  "action": "主要操作类型",
  "sub_intents": ["子操作1", "子操作2"],
  "confidence": 0.0-1.0的置信度,
  "missing_info": ["缺失信息1", "缺失信息2"],
  "clarification_needed": true/false,
  "suggested_question": "如需澄清，这里是建议的追问"
}

【操作类型】
- create_project: 创建新项目
- configure_model: 配置模型
- upload_data: 上传数据
- run_simulation: 运行模拟
- calibrate_params: 参数率定
- generate_report: 生成报告
- analyze_results: 分析结果
- compare_scenarios: 对比情景

请分析并返回JSON：""",

            "entity_extraction": """你是水文建模实体抽取专家。

【任务】
从用户输入中抽取水文建模相关的关键实体。

【用户输入】
$text

【实体Schema】
$entity_schema

【输出要求】
返回JSON格式，包含以下实体类型：
{
  "basin": {
    "name": "流域名称",
    "area_km2": 面积（数值）,
    "location": {"lon": 经度, "lat": 纬度}
  },
  "model": {
    "runoff_type": "产流模型类型（HBV/SCS/XinAnJiang等）",
    "routing_type": "汇流方法（Muskingum/Lag等）"
  },
  "time_period": {
    "start": "开始日期 YYYY-MM-DD",
    "end": "结束日期 YYYY-MM-DD"
  },
  "objectives": ["建模目标1", "目标2"],
  "data_requirements": ["所需数据类型"]
}

注意：
1. 如果某个字段在输入中未提及，设为null
2. 数值型字段请提取数字
3. 日期格式统一为 YYYY-MM-DD

请抽取实体：""",

            "requirement_validation": """你是水文建模需求验证专家。

【任务】
验证用户建模需求的完整性、合理性和可行性。

【建模需求】
$requirements

【验证规则】
$validation_rules

【输出要求】
返回JSON格式：
{
  "is_valid": true/false,
  "completeness_score": 0.0-1.0,
  "issues": [
    {
      "type": "missing_data/invalid_param/infeasible",
      "severity": "error/warning/info",
      "message": "问题描述",
      "suggestion": "建议"
    }
  ],
  "feasibility": {
    "data_availability": 0.0-1.0,
    "computational_cost": "low/medium/high",
    "expected_accuracy": "excellent/good/fair/poor"
  }
}

【常见问题】
1. 缺少DEM数据 → 使用预设流域或建议上传
2. 时间范围过短 → 警告可能影响精度
3. 流域过大 → 警告计算成本
4. 参数范围不合理 → 错误

请验证需求：""",

            # ============ 配置层模板 ============
            "config_generation": """你是HydroSIS模型配置生成专家。

【任务】
基于用户需求生成完整的HydroSIS模型配置（JSON格式）。

【用户需求】
意图: $intent
实体: $entities

【配置示例】
$config_examples

【配置Schema】
$validation_schema

【输出要求】
返回完整的ModelConfig JSON，包含：
{
  "delineation": {
    "method": "automatic/manual",
    "pour_points": [
      {"id": "站点ID", "lon": 经度, "lat": 纬度}
    ]
  },
  "runoff": {
    "model_type": "HBV/SCS/XinAnJiang/...",
    "parameters": {
      // 模型特定参数
    }
  },
  "routing": {
    "model_type": "Muskingum/Lag/DynamicWave",
    "parameters": {
      // 汇流参数
    }
  },
  "parameter_zones": [
    {
      "id": "zone1",
      "controllers": ["控制站点"],
      "parameters": {}
    }
  ],
  "io": {
    "precipitation": "数据路径",
    "results_directory": "结果目录"
  }
}

【参数设置原则】
1. HBV模型: fc=150-300, beta=1-3, lp=0.5-0.9
2. SCS模型: CN值60-95（根据土地利用）
3. Muskingum: K=1-10小时, X=0.1-0.3
4. 参数应符合物理意义和区域经验

请生成配置：""",

            "parameter_suggestion": """你是水文模型参数推荐专家。

【任务】
基于流域特征和模型类型，推荐合理的参数值和范围。

【流域特征】
$basin_features

【模型类型】
$model_type

【参数Schema】
$parameter_schema

【相关知识】
$knowledge

【输出要求】
返回JSON格式：
{
  "suggested_parameters": {
    "param_name": {
      "value": 推荐值,
      "range": [最小值, 最大值],
      "confidence": 0.0-1.0,
      "unit": "单位"
    }
  },
  "rationale": {
    "param_name": "选择该值的理由"
  },
  "references": ["相关文献或经验"],
  "calibration_priority": ["优先率定的参数"]
}

【参数推荐原则】
1. 考虑气候类型（湿润/半湿润/干旱）
2. 考虑地形（平原/丘陵/山地）
3. 考虑土地利用
4. 参考相似流域经验
5. 符合物理意义

请推荐参数：""",

            "scenario_design": """你是水文情景设计专家。

【任务】
根据分析目标，设计合理的对比情景。

【分析目标】
$objective

【基准配置】
$baseline

【情景库】
$scenario_library

【约束条件】
$constraints

【输出要求】
返回JSON格式：
{
  "scenarios": [
    {
      "id": "情景ID",
      "name": "情景名称",
      "description": "简要描述",
      "modifications": {
        "参数路径": 修改值
      },
      "rationale": "设计理由"
    }
  ],
  "comparison_plan": {
    "reference": "参考情景ID",
    "metrics": ["对比指标"],
    "visualization": ["可视化类型"]
  }
}

【常见情景类型】
1. 土地利用变化: 城镇化、退耕还林
2. 气候变化: 降雨增减、极端事件
3. 工程措施: 水库调度、生态流量
4. 参数敏感性: 关键参数扰动

请设计情景：""",

            # ============ 分析层模板 ============
            "result_interpretation": """你是水文模拟结果解读专家。

【任务】
解读模拟结果，提供专业的分析和诊断。

【模拟结果】
$results

【模型配置】
$config

【分析目标】
$objective

【评价标准】
$evaluation_criteria

【输出要求】
返回JSON格式：
{
  "overall_assessment": {
    "performance_level": "excellent/good/fair/poor",
    "key_message": "一句话总结",
    "confidence": 0.0-1.0
  },
  "detailed_findings": [
    {
      "aspect": "分析方面（如洪峰模拟）",
      "finding": "发现的现象",
      "interpretation": "解释",
      "severity": "critical/moderate/minor"
    }
  ],
  "strengths": ["优势1", "优势2"],
  "weaknesses": ["不足1", "不足2"],
  "root_causes": {
    "likely": ["可能原因"],
    "possible": ["待验证原因"]
  }
}

【评价准则】
- NSE > 0.8: excellent
- NSE 0.6-0.8: good  
- NSE 0.4-0.6: fair
- NSE < 0.4: poor

请解读结果：""",

            "issue_diagnosis": """你是水文模型问题诊断专家。

【任务】
诊断模型表现不佳的原因，提供改进建议。

【模拟结果】
$results

【模型配置】
$config

【规则诊断】
$rule_based

【相似案例】
$similar_cases

【诊断知识库】
$diagnostic_knowledge

【输出要求】
返回JSON格式：
{
  "identified_issues": [
    {
      "issue_type": "问题类型",
      "severity": "high/medium/low",
      "symptoms": ["症状"],
      "probable_causes": [
        {
          "cause": "原因",
          "probability": 0.0-1.0,
          "evidence": "支持证据"
        }
      ],
      "recommended_actions": [
        {
          "action": "建议行动",
          "priority": 1-5,
          "expected_improvement": "预期改善",
          "implementation": "实施方法"
        }
      ]
    }
  ],
  "diagnostic_confidence": 0.0-1.0
}

【常见问题】
1. 洪峰偏小 → FC参数过大、降雨插值误差
2. 退水过快 → 地下水衰减系数过大
3. 水量不平衡 → 蒸发参数、产流参数问题

请诊断问题：""",

            # ============ 报告层模板 ============
            "narrative_executive_summary": """你是资深水文专家，请撰写**执行摘要**。

【项目信息】
- 流域：$basin_name
- 时间：$start_date 至 $end_date  
- 模型：$runoff_model + $routing_model

【关键指标】
$metrics

【用户目标】
$user_objective

【要求】
1. 3-5句话，简明扼要
2. 突出模型表现（优秀/良好/需改进）
3. 说明能否满足用户目标
4. 使用专业但易懂的语言
5. 中文输出

请撰写执行摘要：""",

            "narrative_performance_analysis": """你是水文模型专家，请分析**性能表现**。

【精度指标】
$metrics_table

【洪峰分析】
$peak_analysis

【问题子流域】
$problematic_subbasins

【要求】
1. 客观分析指标含义
2. 说明模型在哪些方面表现好/差
3. 分析可能原因（参数、数据、模型结构）
4. 300-500字
5. 中文输出，专业术语准确

请分析性能：""",

            "narrative_recommendations": """基于模拟结果，提供专业**建议**。

【当前状态】
- 模型精度: $accuracy_level
- 主要问题: $main_issues

【数据情况】
$data_quality

【要求】
1. 针对性建议（优先级排序）
2. 涵盖：参数率定、数据补充、模型改进
3. 每条建议说明原因和预期效果
4. 5-8条建议，每条2-3句话
5. 中文输出

请提供建议：""",
        }
    
    def render(self, template_name: str, variables: Dict[str, Any]) -> str:
        """
        渲染模板
        
        Args:
            template_name: 模板名称
            variables: 变量字典
            
        Returns:
            渲染后的文本
        """
        # 尝试从缓存获取
        if template_name in self._templates_cache:
            template_str = self._templates_cache[template_name]
        else:
            # 尝试从文件加载
            template_file = self.templates_dir / f"{template_name}.txt"
            if template_file.exists():
                with open(template_file, 'r', encoding='utf-8') as f:
                    template_str = f.read()
                self._templates_cache[template_name] = template_str
            else:
                raise ValueError(f"模板不存在: {template_name}")
        
        # 渲染模板
        template = Template(template_str)
        
        # 处理变量（转为字符串）
        str_variables = {}
        for key, value in variables.items():
            if isinstance(value, (dict, list)):
                str_variables[key] = json.dumps(value, ensure_ascii=False, indent=2)
            else:
                str_variables[key] = str(value)
        
        try:
            return template.safe_substitute(str_variables)
        except Exception as e:
            raise ValueError(f"模板渲染失败 {template_name}: {e}")
    
    def get_template_names(self) -> list:
        """获取所有可用模板名称"""
        return list(self._templates_cache.keys())
    
    def save_template(self, name: str, content: str):
        """保存模板到文件"""
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        template_file = self.templates_dir / f"{name}.txt"
        
        with open(template_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # 更新缓存
        self._templates_cache[name] = content


if __name__ == "__main__":
    # 测试代码
    manager = PromptTemplateManager()
    
    print("=== 可用模板 ===")
    for name in manager.get_template_names():
        print(f"  - {name}")
    
    print("\n=== 测试模板渲染 ===")
    prompt = manager.render("intent_parsing", {
        "user_input": "我想建立HBV模型",
        "conversation_history": "[]",
        "available_actions": json.dumps([
            "create_project",
            "configure_model"
        ], ensure_ascii=False)
    })
    
    print(prompt[:500] + "...")
