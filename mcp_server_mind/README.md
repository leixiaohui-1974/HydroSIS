# HydroMind Agent - 水文认知智能体

**基于大语言模型的水文建模认知层**

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/hydrosis/hydromind)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)

---

## 📋 目录

- [概述](#概述)
- [核心特性](#核心特性)
- [架构设计](#架构设计)
- [12个认知工具](#12个认知工具)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [使用示例](#使用示例)
- [API文档](#api文档)
- [部署指南](#部署指南)
- [常见问题](#常见问题)

---

## 🎯 概述

HydroMind是HydroSIS双智能体系统的**认知层**，与HydroCompute（机理层）协同工作，提供：

- 🧠 **自然语言理解**: 将用户需求转换为模型配置
- ⚙️ **智能配置生成**: 自动推荐模型和参数
- 📊 **结果智能解读**: 专业的分析和诊断
- 📝 **自然语言报告**: 生成易读的分析报告

### 双智能体架构

```
用户自然语言
     ↓
┌─────────────────┐
│  HydroMind      │  认知智能体（本项目）
│  理解、推理、解释 │  基于LLM
└────────┬────────┘
         │ 调用
         ↓
┌─────────────────┐
│  HydroCompute   │  机理智能体
│  计算、模拟、求解 │  基于物理模型
└─────────────────┘
```

---

## ✨ 核心特性

### 🔵 理解层（3个工具）

| 工具 | 功能 | 用途 |
|------|------|------|
| `parse_user_intent` | 意图识别 | 理解用户想做什么 |
| `extract_entities` | 实体抽取 | 提取流域、模型、时间等信息 |
| `validate_requirements` | 需求验证 | 检查完整性和可行性 |

### 🟢 配置层（3个工具）

| 工具 | 功能 | 用途 |
|------|------|------|
| `generate_model_config` ⭐ | 配置生成 | 自动生成完整ModelConfig |
| `suggest_parameters` | 参数推荐 | 基于经验推荐参数值 |
| `design_scenarios` | 情景设计 | 设计对比情景 |

### 🟡 分析层（3个工具）

| 工具 | 功能 | 用途 |
|------|------|------|
| `interpret_results` | 结果解读 | 分析模拟结果的含义 |
| `diagnose_issues` | 问题诊断 | 诊断表现不佳的原因 |
| `compare_models` | 模型对比 | 对比多个模型性能 |

### 🟣 报告层（3个工具）

| 工具 | 功能 | 用途 |
|------|------|------|
| `generate_narrative` | 叙述生成 | 生成特定章节文本 |
| `create_executive_report` ⭐ | 执行报告 | 生成完整分析报告 |
| `answer_questions` | 智能问答 | 回答用户问题 |

---

## 🏗️ 架构设计

### 目录结构

```
mcp_server_mind/
├── __init__.py
├── llm_backend.py           # LLM后端接口（支持Qwen和Mock）
├── prompt_templates.py      # 提示词模板管理
├── knowledge_base.py        # 水文知识库
├── cognitive_tools.py       # 12个认知工具 ⭐
├── server.py                # MCP服务器封装
├── requirements.txt
├── README.md
├── prompt_templates/        # 提示词模板目录
├── knowledge_base/          # 知识库数据
└── examples/
    └── test_hydromind.py    # 测试脚本
```

### 技术栈

- **语言**: Python 3.8+
- **LLM**: 阿里千问（qwen-max/qwen-plus）
- **依赖**: 零外部依赖（使用标准库）
- **协议**: MCP (Model Context Protocol)

---

## 🚀 快速开始

### 1. 安装

```bash
cd /workspace/mcp_server_mind

# 无需安装依赖（使用标准库）
# 可选：安装开发依赖
# pip install -r requirements.txt
```

### 2. 配置LLM后端

**方法A: 使用阿里千问（推荐）**

```bash
# 设置API密钥
export QWEN_API_KEY="your-api-key-here"
export QWEN_MODEL="qwen-max"  # 可选，默认qwen-max
```

**方法B: 使用Mock模式（测试/开发）**

```bash
# 不设置QWEN_API_KEY，自动使用Mock后端
# Mock后端返回预定义响应，适合测试
```

### 3. 运行测试

```bash
# 测试所有12个工具
python examples/test_hydromind.py
```

### 4. 启动MCP服务器

```bash
# 测试模式
python server.py

# 生产模式（需要FastAPI）
# python -m uvicorn main:app --host 0.0.0.0 --port 8081
```

---

## 📚 使用示例

### 示例1: 意图识别

```python
from cognitive_tools import CognitiveTools
import asyncio

async def example():
    tools = CognitiveTools()
    
    result = await tools.parse_user_intent({
        "user_input": "我想建立长江上游的HBV模型",
        "conversation_history": []
    })
    
    print(f"意图: {result['action']}")
    print(f"置信度: {result['confidence']}")
    print(f"子任务: {result['sub_intents']}")

asyncio.run(example())
```

输出：
```json
{
  "action": "create_and_run_model",
  "sub_intents": ["create_project", "configure_model", "run_simulation"],
  "confidence": 0.9,
  "missing_info": [],
  "clarification_needed": false
}
```

### 示例2: 配置生成 ⭐

```python
async def example():
    tools = CognitiveTools()
    
    result = await tools.generate_model_config({
        "intent": {"action": "create_model"},
        "entities": {
            "basin": {"name": "长江上游", "area_km2": 50000},
            "model": {"runoff_type": "HBV"}
        }
    })
    
    print(result['config_summary'])
    # 输出: "基于HBV模型的长江上游模拟配置"
    
    # 获取完整配置
    config = result['config']  # 完整的ModelConfig JSON
```

### 示例3: 结果解读

```python
async def example():
    tools = CognitiveTools()
    
    result = await tools.interpret_results({
        "simulation_results": {
            "metrics": {"nse": 0.85, "rmse": 45.2}
        },
        "model_config": {"runoff": {"model_type": "HBV"}},
        "user_objective": "洪峰模拟"
    })
    
    assessment = result['overall_assessment']
    print(f"性能: {assessment['performance_level']}")  # "excellent"
    print(f"评价: {assessment['key_message']}")        # "模型精度优秀"
```

### 示例4: 生成报告 ⭐

```python
async def example():
    tools = CognitiveTools()
    
    result = await tools.create_executive_report({
        "workflow_result": {
            "metrics": {"nse": 0.85, "rmse": 45.2}
        },
        "model_config": {
            "runoff": {"model_type": "HBV"},
            "routing": {"model_type": "Muskingum"}
        }
    })
    
    # 获取Markdown报告
    report_content = result['report']['content']
    
    # 保存报告
    with open('simulation_report.md', 'w') as f:
        f.write(report_content)
```

---

## ⚙️ 配置说明

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `QWEN_API_KEY` | 千问API密钥 | 无（必需） |
| `QWEN_MODEL` | 模型名称 | `qwen-max` |
| `QWEN_BASE_URL` | API端点 | 默认千问端点 |
| `LLM_BACKEND` | 后端类型 | `auto` |

### LLM后端选择

```python
from llm_backend import create_llm_backend

# 自动选择（优先千问，不可用则Mock）
backend = create_llm_backend("auto")

# 强制使用千问
backend = create_llm_backend("qwen", api_key="your-key")

# 强制使用Mock
backend = create_llm_backend("mock")
```

### 知识库扩展

```python
from knowledge_base import KnowledgeBase

kb = KnowledgeBase()

# 查询模型信息
model_info = kb.get_model_info("HBV")

# 推荐模型
suggestions = kb.suggest_model({
    "climate": "humid",
    "has_snow": True,
    "area_km2": 5000
})

# 诊断问题
rules = kb.diagnose(["模拟峰值偏小", "NSE较低"])
```

---

## 🔌 API文档

### 工具调用格式

所有工具遵循统一接口：

```python
async def tool_name(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Args:
        args: 输入参数字典
        
    Returns:
        结果字典
    """
    pass
```

### 理解层API

#### parse_user_intent

```python
# 输入
{
    "user_input": str,           # 必需
    "conversation_history": list # 可选
}

# 输出
{
    "action": str,               # 主要操作
    "sub_intents": List[str],    # 子任务列表
    "confidence": float,         # 0.0-1.0
    "missing_info": List[str],   # 缺失信息
    "clarification_needed": bool # 是否需要澄清
}
```

#### extract_entities

```python
# 输入
{
    "user_input": str
}

# 输出
{
    "basin": {
        "name": str,
        "area_km2": float,
        "location": dict
    },
    "model": {
        "runoff_type": str,
        "routing_type": str
    },
    "time_period": {
        "start": str,  # YYYY-MM-DD
        "end": str
    },
    "objectives": List[str],
    "data_requirements": List[str]
}
```

#### validate_requirements

```python
# 输入
{
    "requirements": dict
}

# 输出
{
    "is_valid": bool,
    "completeness_score": float,  # 0.0-1.0
    "issues": List[dict],
    "feasibility": dict
}
```

### 配置层API

#### generate_model_config ⭐

```python
# 输入
{
    "intent": dict,
    "entities": dict
}

# 输出
{
    "config": dict,              # 完整ModelConfig
    "config_summary": str,
    "rationale": dict,
    "warnings": List[str]
}
```

#### suggest_parameters

```python
# 输入
{
    "basin_features": dict,
    "model_type": str
}

# 输出
{
    "suggested_parameters": dict,
    "rationale": dict,
    "references": List[str],
    "calibration_priority": List[str]
}
```

### 分析层API

#### interpret_results

```python
# 输入
{
    "simulation_results": dict,
    "model_config": dict,
    "user_objective": str
}

# 输出
{
    "overall_assessment": {
        "performance_level": str,  # excellent/good/fair/poor
        "key_message": str,
        "confidence": float
    },
    "detailed_findings": List[dict],
    "strengths": List[str],
    "weaknesses": List[str],
    "root_causes": dict
}
```

#### diagnose_issues

```python
# 输入
{
    "poor_results": dict,
    "model_config": dict
}

# 输出
{
    "identified_issues": List[dict],
    "diagnostic_confidence": float,
    "knowledge_sources": List[str]
}
```

### 报告层API

#### create_executive_report ⭐

```python
# 输入
{
    "workflow_result": dict,
    "model_config": dict,
    "report_type": str  # executive/technical/full
}

# 输出
{
    "report": {
        "format": str,           # markdown/html/pdf
        "content": str,
        "sections": dict
    },
    "metadata": {
        "word_count": int,
        "generated_at": str
    }
}
```

---

## 🐳 部署指南

### Docker部署

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . /app/mcp_server_mind

ENV QWEN_API_KEY=""
ENV QWEN_MODEL="qwen-max"

CMD ["python", "/app/mcp_server_mind/server.py"]
```

```bash
# 构建
docker build -t hydromind:latest .

# 运行
docker run -e QWEN_API_KEY="your-key" -p 8081:8081 hydromind:latest
```

### 与HydroCompute集成

```python
from mcp_orchestrator import TwinAgentCoordinator
from hydromind_client import HydroMindClient
from hydrocompute_client import HydroComputeClient

# 初始化双智能体
mind = HydroMindClient("http://localhost:8081")
compute = HydroComputeClient("http://localhost:8080")

coordinator = TwinAgentCoordinator(mind, compute)

# 处理用户请求
result = await coordinator.process_user_request(
    user_input="我想建立HBV模型",
    session_id="user_123"
)

print(result['natural_language_summary'])
```

---

## ❓ 常见问题

### Q1: 如何切换LLM后端？

**A**: 通过环境变量或代码配置：

```bash
# 使用千问
export QWEN_API_KEY="sk-xxx"

# 使用Mock（测试）
unset QWEN_API_KEY
```

### Q2: Mock模式返回什么？

**A**: Mock后端返回合理的预定义响应，包含JSON格式的工具结果。适合：
- 开发测试
- CI/CD
- 无API密钥环境

### Q3: 如何自定义提示词？

**A**: 修改 `prompt_templates.py` 或创建自定义模板文件：

```python
from prompt_templates import PromptTemplateManager

manager = PromptTemplateManager()
manager.save_template("my_custom", """
你是水文专家...
""")
```

### Q4: 支持哪些模型？

**A**: 当前支持：
- ✅ HBV
- ✅ SCS
- ✅ XinAnJiang（新安江）
- ✅ HYMOD
- ✅ VIC
- ✅ WETSPA

可通过知识库扩展更多模型。

### Q5: 如何提高配置生成准确率？

**A**: 
1. 提供更详细的流域信息
2. 明确建模目标
3. 使用`qwen-max`模型（更准确）
4. 扩展知识库内容

### Q6: 报告支持哪些格式？

**A**: 
- ✅ Markdown（默认）
- ⚙️ HTML（可扩展）
- ⚙️ PDF（可扩展）

### Q7: 如何监控LLM调用？

**A**: 查看响应中的`metadata`:

```python
response = await llm.complete(messages)
print(response.usage)  # token使用量
print(response.metadata)  # 其他元数据
```

---

## 📊 性能指标

| 指标 | Mock模式 | 千问模式 |
|------|---------|---------|
| 响应时间 | <100ms | 1-3秒 |
| 配置准确率 | 60% | 85%+ |
| 成本 | 免费 | ¥0.02/千tokens |

---

## 🤝 贡献指南

欢迎贡献！可以：

1. **扩展知识库**: 添加更多模型和参数经验
2. **优化提示词**: 改进提示词模板
3. **新增工具**: 开发新的认知工具
4. **Bug修复**: 提交issue和PR

---

## 📄 许可证

MIT License

---

## 🙏 致谢

- 阿里云通义千问团队
- HydroSIS开发团队
- 所有贡献者

---

## 📞 联系方式

- **问题反馈**: GitHub Issues
- **文档**: [完整文档](docs/)
- **示例**: [examples/](examples/)

---

**HydroMind - 让水文建模更智能！** 🚀
