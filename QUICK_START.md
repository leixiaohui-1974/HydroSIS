# HydroSIS 双智能体系统 - 快速开始

## 🚀 3分钟快速启动

### 方式1: Mock模式（无需API密钥）

```bash
# 1. 安装依赖
pip install fastapi uvicorn

# 2. 启动HydroMind（Mock模式）
cd /workspace
export LLM_BACKEND=mock
python3 -m mcp_server_mind.main &

# 3. 测试
curl http://localhost:8081/health
python3 mcp_server_mind/examples/test_hydromind.py
```

### 方式2: 真实LLM模式

```bash
# 1. 设置API密钥
export QWEN_API_KEY="sk-你的千问密钥"

# 2. 启动服务
python3 -m mcp_server_mind.main

# 3. 测试
curl http://localhost:8081/health
```

### 方式3: Docker模式

```bash
# 1. 构建镜像
docker build -t hydromind:latest -f mcp_server_mind/Dockerfile .

# 2. 运行
docker run -p 8081:8081 -e QWEN_API_KEY="your-key" hydromind:latest

# 3. 测试
curl http://localhost:8081/health
```

## 📝 第一次使用

```python
import asyncio
from mcp_orchestrator.clients import HydroMindClient

async def main():
    mind = HydroMindClient("http://localhost:8081")
    
    # 生成配置
    config = await mind.generate_model_config(
        intent={"action": "create_model"},
        entities={"basin": {"name": "测试流域"}}
    )
    
    print(config['config_summary'])

asyncio.run(main())
```

## 🧪 运行测试

```bash
# 离线测试
python3 -c "from mcp_orchestrator.config_converter import ConfigConverter; print('✅ OK')"

# 在线测试（需要服务器运行）
python3 tests/integration/test_end_to_end.py
```

## 📚 详细文档

- 完整部署指南: `TWIN_AGENT_DEPLOYMENT_GUIDE.md`
- Phase 2A报告: `PHASE2A_COMPLETION_REPORT.md`
- 开发完成报告: `DEVELOPMENT_COMPLETION_REPORT.md`
