# HydroMind Agent - 5分钟快速开始

**从零到第一次成功调用！**

---

## 🚀 第一步：验证环境（30秒）

```bash
cd /workspace/mcp_server_mind

# 检查Python版本（需要3.8+）
python3 --version

# 查看项目结构
ls -la
```

应该看到：
- ✅ `cognitive_tools.py` - 12个认知工具
- ✅ `llm_backend.py` - LLM后端
- ✅ `server.py` - MCP服务器
- ✅ `examples/test_hydromind.py` - 测试脚本

---

## 🧪 第二步：运行测试（1分钟）

### Mock模式（无需API密钥）

```bash
# 直接运行测试
python3 examples/test_hydromind.py
```

你将看到：
```
============================================================
HydroMind Agent - 认知智能体测试
============================================================

提示: 当前使用Mock LLM后端
设置环境变量 QWEN_API_KEY 可使用真实的千问模型

[工具1] 意图识别
{
  "action": "create_and_run_model",
  "sub_intents": ["create_project", "configure_model"],
  ...
}
```

✅ 如果看到上面的输出，说明系统工作正常！

---

## 🔑 第三步：接入千问（1分钟）

**当您获得API密钥后：**

```bash
# 设置环境变量
export QWEN_API_KEY="sk-你的密钥"
export QWEN_MODEL="qwen-max"  # 可选，默认qwen-max

# 再次运行测试
python3 examples/test_hydromind.py
```

现在您将看到**真实的LLM响应**！

---

## 💻 第四步：编写第一个程序（2分钟）

创建 `my_first_hydromind.py`:

```python
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from cognitive_tools import CognitiveTools

async def main():
    # 1. 初始化工具
    tools = CognitiveTools()
    print("✅ HydroMind初始化完成\n")
    
    # 2. 理解用户需求
    print("📝 理解用户需求...")
    intent = await tools.parse_user_intent({
        "user_input": "我想建立长江上游的HBV模型，流域面积5000平方公里"
    })
    print(f"   意图: {intent['action']}")
    print(f"   置信度: {intent['confidence']}\n")
    
    # 3. 抽取实体
    print("🔍 抽取关键信息...")
    entities = await tools.extract_entities({
        "user_input": "我想建立长江上游的HBV模型，流域面积5000平方公里"
    })
    print(f"   流域: {entities.get('basin', {}).get('name')}")
    print(f"   模型: {entities.get('model', {}).get('runoff_type')}")
    print(f"   面积: {entities.get('basin', {}).get('area_km2')} km²\n")
    
    # 4. 生成配置
    print("⚙️  生成模型配置...")
    config = await tools.generate_model_config({
        "intent": intent,
        "entities": entities
    })
    print(f"   配置: {config['config_summary']}\n")
    
    # 5. 推荐参数
    print("🎯 推荐模型参数...")
    params = await tools.suggest_parameters({
        "basin_features": {"area_km2": 5000, "climate": "humid"},
        "model_type": "HBV"
    })
    for param, info in params['suggested_parameters'].items():
        print(f"   {param}: {info['value']} (范围: {info['range']})")
    
    print("\n🎉 完成！这就是HydroMind的能力！")

# 运行
asyncio.run(main())
```

运行：
```bash
python3 my_first_hydromind.py
```

---

## 🌟 第五步：常用操作速查

### 操作1: 意图识别

```python
result = await tools.parse_user_intent({
    "user_input": "你的自然语言输入"
})
```

### 操作2: 生成配置 ⭐

```python
config = await tools.generate_model_config({
    "intent": {"action": "create_model"},
    "entities": {
        "basin": {"name": "长江上游", "area_km2": 50000},
        "model": {"runoff_type": "HBV"}
    }
})
# 获取完整配置
model_config = config['config']
```

### 操作3: 解读结果

```python
interpretation = await tools.interpret_results({
    "simulation_results": {"metrics": {"nse": 0.85}},
    "model_config": {"runoff": {"model_type": "HBV"}}
})
print(interpretation['overall_assessment']['key_message'])
```

### 操作4: 生成报告 ⭐

```python
report = await tools.create_executive_report({
    "workflow_result": simulation_results,
    "model_config": model_config
})
# 保存报告
with open('report.md', 'w') as f:
    f.write(report['report']['content'])
```

---

## 🔧 常见问题快速解决

### Q: 如何知道当前使用哪个后端？

```python
tools = CognitiveTools()
print(f"后端: {tools.llm.__class__.__name__}")
print(f"可用: {tools.llm.is_available()}")
```

### Q: 如何切换到Mock模式？

```bash
# 取消API密钥
unset QWEN_API_KEY

# 或者代码中强制指定
from llm_backend import MockLLMBackend
tools = CognitiveTools(llm_backend=MockLLMBackend())
```

### Q: 返回的JSON格式是什么？

每个工具都返回Dict格式，包含：
- 核心结果字段（如`action`, `config`, `narrative`等）
- 元数据（如`confidence`, `rationale`等）

详见 [README.md API文档部分](README.md#api文档)

---

## 📚 下一步学习

1. **完整测试**: 运行 `examples/test_hydromind.py` 查看所有工具
2. **阅读文档**: [README.md](README.md) 有详细的API文档
3. **查看源码**: `cognitive_tools.py` 了解实现细节
4. **扩展知识库**: `knowledge_base.py` 添加更多模型和规则

---

## 🎯 典型使用流程

```
用户需求
   ↓
[工具1] parse_user_intent     → 理解意图
   ↓
[工具2] extract_entities      → 提取信息
   ↓
[工具3] validate_requirements → 验证需求
   ↓
[工具4] generate_model_config → 生成配置 ⭐
   ↓
【HydroCompute执行模拟】
   ↓
[工具7] interpret_results     → 解读结果
   ↓
[工具11] create_executive_report → 生成报告 ⭐
   ↓
自然语言报告
```

---

## ✅ 检查清单

开始前确认：
- [ ] Python 3.8+
- [ ] 代码已下载到 `/workspace/mcp_server_mind`
- [ ] （可选）设置了 `QWEN_API_KEY`

测试通过后：
- [ ] Mock模式测试成功
- [ ] （可选）千问模式测试成功
- [ ] 能够运行自己的第一个程序

---

**恭喜！您已经掌握了HydroMind的基本使用！** 🎉

现在您可以：
- 让系统理解您的自然语言需求
- 自动生成模型配置
- 智能解读模拟结果
- 生成专业的分析报告

**准备好与HydroCompute协同，开始智能水文建模吧！** 🚀
