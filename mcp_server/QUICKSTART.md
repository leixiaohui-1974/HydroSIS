# HydroSIS MCP服务器 - 快速开始指南

## 5分钟快速体验

### 1. 安装依赖（1分钟）

```bash
cd /workspace
pip install -r mcp_server/requirements.txt
pip install -e .
```

### 2. 启动服务器（1分钟）

```bash
# 设置环境变量（可选）
export DATA_ROOT=/tmp/hydrosis-data
export PORT=8080

# 启动服务器
python -m mcp_server.main
```

看到以下输出表示启动成功：
```
============================================================
初始化HydroSIS MCP服务器
============================================================
数据根目录: /tmp/hydrosis-data
已注册 18 个MCP工具
MCP服务器初始化完成
============================================================
启动MCP服务器: http://0.0.0.0:8080
```

### 3. 测试连接（1分钟）

打开新终端，测试服务器：

```bash
# 健康检查
curl http://localhost:8080/health

# 预期输出：
# {
#   "status": "healthy",
#   "timestamp": "2025-10-26T12:00:00",
#   "tools_count": 18
# }
```

### 4. 列出所有工具（1分钟）

```bash
curl http://localhost:8080/mcp/tools | jq .

# 或使用Python
python -c "
import requests
tools = requests.get('http://localhost:8080/mcp/tools').json()['tools']
for tool in tools:
    print(f\"- {tool['name']}: {tool['description']}\")
"
```

### 5. 创建第一个项目（1分钟）

```bash
curl -X POST http://localhost:8080/mcp/tools/create_project \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "demo_user",
    "project_name": "我的第一个水文项目",
    "description": "快速开始示例",
    "template": "basic"
  }' | jq .
```

成功响应示例：
```json
{
  "content": [
    {
      "type": "text",
      "text": "{
        \"project_id\": \"abc-123-def-456\",
        \"name\": \"我的第一个水文项目\",
        \"path\": \"/tmp/hydrosis-data/users/demo_user/projects/abc-123-def-456\",
        \"status\": \"created\",
        \"message\": \"项目 '我的第一个水文项目' 创建成功\"
      }"
    }
  ],
  "isError": false
}
```

## 使用Python客户端

### 安装示例

将 `example_client.py` 复制到你的项目：

```bash
cp mcp_server/examples/example_client.py ./my_client.py
```

### 运行示例

```python
from my_client import HydroSISMCPClient

# 创建客户端
client = HydroSISMCPClient("http://localhost:8080")

# 健康检查
health = client.health_check()
print(f"服务器状态: {health['status']}")

# 创建项目
project = client.call_tool("create_project", {
    "user_id": "demo_user",
    "project_name": "Python客户端示例",
    "template": "advanced"
})

print(f"项目ID: {project['project_id']}")

# 列出用户项目
projects = client.call_tool("list_projects", {
    "user_id": "demo_user"
})

print(f"项目总数: {projects['count']}")
for p in projects['projects']:
    print(f"  - {p['name']}")
```

## 常见工具使用示例

### 1. 流域划分

```python
client.call_tool("delineate_watershed", {
    "project_id": "your-project-id",
    "dem_path": "/path/to/dem.tif",
    "pour_points": [
        {"id": "outlet", "lon": 120.5, "lat": 30.5}
    ],
    "burn_streams": False
})
```

### 2. 配置产流模型

```python
# SCS模型
client.call_tool("configure_runoff_model", {
    "project_id": "your-project-id",
    "model_type": "scs",
    "parameters": {
        "cn": 75  # 曲线数
    }
})

# HBV模型
client.call_tool("configure_runoff_model", {
    "project_id": "your-project-id",
    "model_type": "hbv",
    "parameters": {
        "fc": 200.0,
        "beta": 2.0,
        "lp": 0.7
    }
})

# 新安江模型
client.call_tool("configure_runoff_model", {
    "project_id": "your-project-id",
    "model_type": "xinanjiang",
    "parameters": {
        "wm": 120.0,
        "b": 0.3,
        "im": 0.01
    }
})
```

### 3. 上传驱动数据

```python
import pandas as pd
from datetime import datetime, timedelta

# 生成示例数据
start_date = datetime(2020, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(365)]
precip = [5.2, 10.3, 0.0, 15.6, 8.9, ...]  # 365个值

client.call_tool("upload_forcing_data", {
    "project_id": "your-project-id",
    "data_type": "precipitation",
    "data": {
        "zone1": precip
    },
    "timestamps": [d.isoformat() for d in dates],
    "time_step": "daily"
})
```

### 4. 运行模拟（异步）

```python
# 提交任务
task_id = client.submit_task(
    tool_name="run_simulation",
    arguments={
        "project_id": "your-project-id",
        "start_date": "2020-01-01",
        "end_date": "2020-12-31",
        "generate_report": True
    },
    user_id="demo_user"
)

print(f"任务ID: {task_id}")

# 等待完成
result = client.wait_for_task(task_id, timeout=600)

if result['status'] == 'completed':
    print("模拟完成!")
    print(f"运行ID: {result['result']['run_id']}")
else:
    print(f"失败: {result['error']}")
```

### 5. 参数校准

```python
# 准备观测数据
observed = {
    "outlet": [12.5, 15.3, 18.9, 22.1, 25.6, ...]  # 流量观测
}

# 提交校准任务
task_id = client.submit_task(
    tool_name="calibrate_parameters",
    arguments={
        "project_id": "your-project-id",
        "observed_data": observed,
        "parameter_ranges": {
            "cn": [60, 90]  # SCS曲线数范围
        },
        "optimization_metric": "nse",
        "max_iterations": 100
    },
    user_id="demo_user"
)

# 等待校准完成
result = client.wait_for_task(task_id)
best_params = result['result']['best_parameters']
print(f"最优参数: {best_params}")
```

## Docker快速部署

### 单机部署

```bash
cd mcp_server
docker-compose up -d
```

这将启动：
- ✅ MCP服务器 (http://localhost:8080)
- ✅ Redis缓存
- ✅ PostgreSQL数据库
- ✅ MinIO对象存储 (http://localhost:9001)
- ✅ Celery任务队列

### 查看日志

```bash
docker-compose logs -f mcp-server
```

### 停止服务

```bash
docker-compose down
```

## 与大模型集成

### 通义千问示例

```python
from dashscope import Generation
import requests

# 1. 获取MCP工具列表
mcp_tools = requests.get("http://localhost:8080/mcp/tools").json()['tools']

# 2. 转换为通义千问格式
qwen_tools = [
    {
        "type": "function",
        "function": {
            "name": tool['name'],
            "description": tool['description'],
            "parameters": tool['inputSchema']
        }
    }
    for tool in mcp_tools
]

# 3. 调用通义千问
messages = [
    {"role": "user", "content": "帮我创建一个水文模拟项目，名称是'长江流域模拟'"}
]

response = Generation.call(
    model='qwen-max',
    messages=messages,
    tools=qwen_tools,
    result_format='message'
)

# 4. 处理工具调用
if response.output.get('tool_calls'):
    for tool_call in response.output['tool_calls']:
        # 调用MCP工具
        result = requests.post(
            f"http://localhost:8080/mcp/tools/{tool_call['function']['name']}",
            json=tool_call['function']['arguments']
        ).json()
        
        print(f"执行结果: {result}")
```

### Claude示例

```python
import anthropic

client = anthropic.Anthropic(api_key="your-api-key")

# 定义工具
tools = [
    {
        "name": "create_project",
        "description": "创建新的水文模拟项目",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "project_name": {"type": "string"}
            },
            "required": ["user_id", "project_name"]
        }
    }
]

# 调用Claude
response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    tools=tools,
    messages=[
        {"role": "user", "content": "帮我创建一个项目"}
    ]
)

# 处理工具调用
for block in response.content:
    if block.type == "tool_use":
        # 调用MCP工具
        result = requests.post(
            f"http://localhost:8080/mcp/tools/{block.name}",
            json=block.input
        ).json()
        
        print(result)
```

## 下一步

- 📖 阅读[完整文档](README.md)
- 🔧 查看[API文档](#api文档)
- 💻 运行[完整示例](examples/example_client.py)
- 🚀 [生产部署指南](README.md#部署指南)

## 问题排查

### 服务器无法启动

1. 检查端口是否被占用：
```bash
lsof -i :8080
```

2. 查看日志：
```bash
tail -f mcp_server.log
```

### 工具调用失败

1. 检查参数是否符合schema：
```bash
curl http://localhost:8080/mcp/tools/tool_name/schema
```

2. 查看错误详情：
```python
result = client.call_tool("tool_name", {...})
if result.get('isError'):
    print(result['content'][0]['text'])
```

### 连接数据库失败

检查环境变量：
```bash
echo $POSTGRES_HOST
echo $POSTGRES_PASSWORD
```

## 获取帮助

- 📧 support@hydrosis.example.com
- 🐛 [问题反馈](https://github.com/hydrosis/issues)
- 💬 [社区论坛](https://community.hydrosis.example.com)

祝你使用愉快！🎉
