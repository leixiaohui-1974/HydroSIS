# HydroSIS 双智能体系统 - 部署和使用指南

**版本**: 2.0  
**更新**: 2025-10-28  
**状态**: ✅ 核心功能完成，可部署测试

---

## 📋 目录

1. [系统架构](#系统架构)
2. [环境准备](#环境准备)
3. [部署方式](#部署方式)
4. [使用指南](#使用指南)
5. [测试验证](#测试验证)
6. [故障排查](#故障排查)

---

## 🏗️ 系统架构

### 双智能体协同

```
┌─────────────────────────────────────────────┐
│           用户自然语言输入                    │
│  "我想建立长江上游的HBV模型"                 │
└──────────────┬──────────────────────────────┘
               ↓
    ┌──────────────────────┐
    │  Twin-Agent          │  双智能体协调器
    │  Coordinator         │  (可选独立部署)
    └──────┬───────────┬───┘
           ↓           ↓
    ┌──────────┐  ┌──────────┐
    │HydroMind │  │HydroComp │
    │认知智能体 │  │机理智能体 │
    │:8081     │  │:8080     │
    └──────────┘  └──────────┘
         ↓               ↓
    理解、配置       计算、模拟
    解读、报告       求解、优化
```

### 端口分配

| 服务 | 端口 | 说明 |
|------|------|------|
| HydroCompute | 8080 | 机理智能体（已有） |
| HydroMind | 8081 | 认知智能体（新增） |
| Coordinator | 8082 | 协调器（可选） |

---

## 🔧 环境准备

### 方式1: Docker（推荐）

**要求**:
- Docker 20.10+
- Docker Compose 1.29+

**检查**:
```bash
docker --version
docker-compose --version
```

### 方式2: 本地Python

**要求**:
- Python 3.8+
- pip

**依赖**:
```bash
# 安装依赖
pip install fastapi uvicorn

# 可选：测试依赖
pip install pytest pytest-asyncio
```

### 环境变量

创建 `.env` 文件：
```bash
# 复制示例文件
cp mcp_server_mind/.env.example .env

# 编辑填写API密钥
nano .env
```

必需变量：
```bash
QWEN_API_KEY=sk-你的千问密钥  # 必需（或使用Mock模式）
QWEN_MODEL=qwen-max          # 可选，默认qwen-max
```

---

## 🚀 部署方式

### 方式A: Docker Compose一键部署 ⭐推荐

```bash
# 1. 设置环境变量
export QWEN_API_KEY="sk-你的密钥"

# 2. 启动双智能体系统
cd /workspace
docker-compose -f docker-compose.twin-agent.yml up -d

# 3. 查看日志
docker-compose -f docker-compose.twin-agent.yml logs -f

# 4. 检查状态
docker-compose -f docker-compose.twin-agent.yml ps
```

**预期输出**:
```
NAME                  STATUS              PORTS
hydrocompute-agent    Up (healthy)        0.0.0.0:8080->8080/tcp
hydromind-agent       Up (healthy)        0.0.0.0:8081->8081/tcp
```

### 方式B: 本地开发模式

#### 终端1: 启动HydroMind
```bash
cd /workspace
export QWEN_API_KEY="sk-你的密钥"
python3 -m mcp_server_mind.main
```

预期输出：
```
✅ HydroMind认知工具初始化完成
   LLM后端: QwenBackend (或 MockLLMBackend)
   可用性: True

INFO: Started server process
INFO: Uvicorn running on http://0.0.0.0:8081
```

#### 终端2: 启动HydroCompute
```bash
cd /workspace
python3 -m mcp_server.main
```

#### 终端3: 测试
```bash
# 测试HydroMind
curl http://localhost:8081/health

# 测试HydroCompute  
curl http://localhost:8080/health

# 运行集成测试
python3 tests/integration/test_end_to_end.py
```

---

## 📖 使用指南

### 1. 快速验证

```bash
# 健康检查
curl http://localhost:8081/health
curl http://localhost:8080/health

# 列出HydroMind工具
curl http://localhost:8081/mcp/tools | jq '.tools[].name'

# 列出HydroCompute工具
curl http://localhost:8080/mcp/tools | jq '.tools[].name'
```

### 2. 使用Python客户端

```python
import asyncio
from mcp_orchestrator import TwinAgentCoordinator

async def main():
    # 创建协调器
    coordinator = TwinAgentCoordinator(
        hydromind_url="http://localhost:8081",
        hydrocompute_url="http://localhost:8080"
    )
    
    # 处理自然语言请求
    result = await coordinator.process_user_request(
        user_input="我想建立长江上游的HBV模型，流域面积50000平方公里",
        session_id="demo_user"
    )
    
    # 查看结果
    print("状态:", result['status'])
    print("配置:", result['configuration']['config_summary'])
    print("自然语言摘要:", result.get('natural_language_summary'))

asyncio.run(main())
```

### 3. 直接调用工具

#### 调用HydroMind工具

```bash
# 意图识别
curl -X POST http://localhost:8081/mcp/tools/parse_user_intent \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "我想建立HBV模型"
  }'

# 生成配置
curl -X POST http://localhost:8081/mcp/tools/generate_model_config \
  -H "Content-Type: application/json" \
  -d '{
    "intent": {"action": "create_model"},
    "entities": {
      "basin": {"name": "长江上游", "area_km2": 50000},
      "model": {"runoff_type": "HBV"}
    }
  }'
```

#### 调用HydroCompute工具

```bash
# 创建项目
curl -X POST http://localhost:8080/mcp/tools/create_project \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "demo_user",
    "project_name": "测试项目"
  }'
```

### 4. 使用便捷端点

```bash
# 快速理解（组合调用：意图+实体+验证）
curl -X POST "http://localhost:8081/understand?user_input=我想建立HBV模型"

# 快速配置生成
curl -X POST http://localhost:8081/generate_config \
  -H "Content-Type: application/json" \
  -d '{
    "intent": {"action": "create_model"},
    "entities": {"basin": {"name": "长江上游"}}
  }'
```

---

## 🧪 测试验证

### 测试1: 离线测试（无需服务器）

```bash
# 配置转换器测试
cd /workspace
python3 -c "
from mcp_orchestrator.config_converter import ConfigConverter
converter = ConfigConverter()
config = converter.hydromind_to_hydrosis({
    'runoff': {'model_type': 'HBV'},
    'routing': {'model_type': 'Muskingum'}
})
print('✅ 配置转换测试通过')
print(f'参数数: {len(config[\"runoff\"][\"parameters\"])}')
"

# 协调器初始化测试
python3 -c "
from mcp_orchestrator import TwinAgentCoordinator
coordinator = TwinAgentCoordinator()
print('✅ 协调器初始化测试通过')
"
```

### 测试2: 在线测试（需要服务器）

```bash
# 启动服务器后运行
cd /workspace
python3 tests/integration/test_end_to_end.py
```

预期输出：
```
============================================================
测试总结
============================================================
  ✅ 通过  基础理解功能
  ✅ 通过  配置生成
  ✅ 通过  配置转换
  ✅ 通过  协调器初始化
  ✅ 通过  快速理解模式
  ✅ 通过  完整工作流（Mock）
  ✅ 通过  报告生成

总计: 7/7 通过
```

### 测试3: Mock模式验证

```bash
# 不设置API密钥
unset QWEN_API_KEY

# 运行HydroMind测试
cd /workspace/mcp_server_mind
python3 examples/test_hydromind.py
```

应该看到：
```
ℹ️  千问后端不可用，使用Mock后端
✅ HydroMind认知工具初始化完成
   LLM后端: MockLLMBackend
   可用性: True
```

---

## 🔍 故障排查

### 问题1: 服务器启动失败

**症状**: `ModuleNotFoundError: No module named 'fastapi'`

**解决**:
```bash
pip install fastapi uvicorn
```

### 问题2: 端口已被占用

**症状**: `Address already in use`

**解决**:
```bash
# 查找占用端口的进程
lsof -i :8081
lsof -i :8080

# 杀死进程
kill -9 <PID>

# 或使用其他端口
export PORT=8091
python3 -m mcp_server_mind.main
```

### 问题3: 客户端连接失败

**症状**: `ConnectionError: 无法连接到 http://localhost:8081`

**检查清单**:
1. 服务器是否启动？
   ```bash
   curl http://localhost:8081/health
   ```

2. 防火墙是否阻止？
   ```bash
   sudo ufw allow 8081
   ```

3. Docker网络是否正常？
   ```bash
   docker network ls
   docker network inspect hydrosis-network
   ```

### 问题4: 千问API调用失败

**症状**: `千问API错误: Invalid API key`

**检查**:
1. API密钥是否正确？
   ```bash
   echo $QWEN_API_KEY
   ```

2. 切换到Mock模式测试：
   ```bash
   export LLM_BACKEND=mock
   ```

### 问题5: 配置转换失败

**症状**: `配置验证失败: 参数fc超出范围`

**解决**:
```python
# 检查参数范围
from mcp_orchestrator.config_converter import ConfigConverter
converter = ConfigConverter()

# 查看默认参数
defaults = converter._get_default_runoff_params("HBV")
print(defaults)

# 使用合理范围
config = {
    "runoff": {
        "model_type": "HBV",
        "parameters": {"fc": 200}  # 在[50,500]范围内
    }
}
```

---

## 📊 性能指标

### 响应时间

| 操作 | Mock模式 | 千问模式 | 说明 |
|------|---------|---------|------|
| 意图识别 | <100ms | 1-2秒 | 单次调用 |
| 配置生成 | <200ms | 2-3秒 | 包含LLM推理 |
| 结果解读 | <100ms | 1-2秒 | 分析和解释 |
| 报告生成 | <300ms | 3-5秒 | 完整报告 |
| 完整流程 | <1秒 | 10-15秒 | 端到端 |

### 资源占用

| 服务 | 内存 | CPU | 说明 |
|------|------|-----|------|
| HydroMind | ~200MB | <10% | 轻量级 |
| HydroCompute | ~500MB | 变化 | 取决于模拟规模 |

---

## 🎯 典型使用场景

### 场景1: 快速建模

```python
# 1分钟完成建模配置
from mcp_orchestrator.clients import HydroMindClient

mind = HydroMindClient()

# 理解需求
intent = await mind.parse_user_intent(
    "我想建立HBV模型，流域1000平方公里"
)

# 生成配置
config = await mind.generate_model_config(
    intent=intent,
    entities={"basin": {"area_km2": 1000}}
)

# 配置已就绪！
print(config['config'])
```

### 场景2: 完整工作流

```python
# 端到端自动化
from mcp_orchestrator import TwinAgentCoordinator

coordinator = TwinAgentCoordinator()

result = await coordinator.process_user_request(
    user_input="建立长江上游HBV模型并运行模拟",
    session_id="user_123"
)

# 自动完成：理解→配置→计算→解读→报告
print(result['natural_language_summary'])
```

### 场景3: 结果分析

```python
# 智能解读模拟结果
mind = HydroMindClient()

interpretation = await mind.interpret_results(
    simulation_results={"metrics": {"nse": 0.85}},
    model_config={"runoff": {"model_type": "HBV"}}
)

print(interpretation['overall_assessment']['key_message'])
# 输出: "模型精度优秀"
```

---

## 📚 API文档

### HydroMind API (端口8081)

**基础端点**:
```
GET  /health                    # 健康检查
GET  /mcp/tools                 # 列出12个工具
POST /mcp/tools/{tool_name}     # 调用工具
```

**便捷端点**:
```
POST /understand                # 快速理解（意图+实体+验证）
POST /generate_config           # 快速配置生成
POST /analyze_results           # 快速结果分析
```

**工具列表**:
- `parse_user_intent` - 意图识别
- `extract_entities` - 实体抽取
- `validate_requirements` - 需求验证
- `generate_model_config` ⭐ - 配置生成
- `suggest_parameters` - 参数推荐
- `design_scenarios` - 情景设计
- `interpret_results` - 结果解读
- `diagnose_issues` - 问题诊断
- `compare_models` - 模型对比
- `generate_narrative` - 叙述生成
- `create_executive_report` ⭐ - 执行报告
- `answer_questions` - 智能问答

### HydroCompute API (端口8080)

详见: `mcp_server/README.md`

---

## 🛠️ 运维命令

### Docker模式

```bash
# 启动
docker-compose -f docker-compose.twin-agent.yml up -d

# 停止
docker-compose -f docker-compose.twin-agent.yml down

# 重启某个服务
docker-compose -f docker-compose.twin-agent.yml restart hydromind

# 查看日志
docker-compose -f docker-compose.twin-agent.yml logs -f hydromind

# 进入容器
docker exec -it hydromind-agent bash

# 更新镜像
docker-compose -f docker-compose.twin-agent.yml build
docker-compose -f docker-compose.twin-agent.yml up -d
```

### 本地模式

```bash
# 启动HydroMind
cd /workspace && python3 -m mcp_server_mind.main &
echo $! > /tmp/hydromind.pid

# 启动HydroCompute
cd /workspace && python3 -m mcp_server.main &
echo $! > /tmp/hydrocompute.pid

# 停止
kill $(cat /tmp/hydromind.pid)
kill $(cat /tmp/hydrocompute.pid)

# 查看日志
tail -f mcp_server.log
```

---

## ✅ 验收检查表

### 部署前检查
- [ ] Docker已安装（如果使用Docker）
- [ ] Python 3.8+已安装
- [ ] 环境变量已配置
- [ ] 端口8080/8081未被占用

### 部署后验证
- [ ] HydroMind健康检查通过
- [ ] HydroCompute健康检查通过
- [ ] 可以列出工具
- [ ] 可以调用工具
- [ ] 集成测试通过

### 功能验证
- [ ] 可以理解自然语言
- [ ] 可以生成配置
- [ ] 配置可以转换
- [ ] 可以执行模拟（需要HydroCompute）
- [ ] 可以生成报告

---

## 📞 获取帮助

### 文档
- HydroMind文档: `mcp_server_mind/README.md`
- HydroCompute文档: `mcp_server/README.md`
- 开发路线图: `DEVELOPMENT_ROADMAP.md`
- Phase 2A报告: `PHASE2A_COMPLETION_REPORT.md`

### 测试
- Mock测试: `python3 mcp_server_mind/examples/test_hydromind.py`
- 集成测试: `python3 tests/integration/test_end_to_end.py`
- 单元测试: `python3 mcp_server_mind/tests/test_llm_backend.py`

### 日志
- HydroMind日志: 控制台输出
- HydroCompute日志: `mcp_server.log`
- Docker日志: `docker-compose logs`

---

## 🎉 快速开始总结

### 3步启动系统

```bash
# 步骤1: 设置API密钥
export QWEN_API_KEY="sk-你的密钥"

# 步骤2: 启动服务器
docker-compose -f docker-compose.twin-agent.yml up -d

# 步骤3: 验证
curl http://localhost:8081/health
curl http://localhost:8080/health
```

### 第一次使用

```python
from mcp_orchestrator.clients import HydroMindClient

mind = HydroMindClient()

# 生成配置
config = await mind.generate_model_config(
    intent={"action": "create_model"},
    entities={"basin": {"name": "测试流域"}}
)

print(config['config_summary'])
```

**恭喜！您已成功部署HydroSIS双智能体系统！** 🎉

---

**文档版本**: 2.0  
**最后更新**: 2025-10-28  
**维护者**: HydroSIS开发团队
