# HydroSIS MCP服务器

HydroSIS分布式水文模拟框架的MCP（Model Context Protocol）服务封装。

## 目录

- [概述](#概述)
- [架构设计](#架构设计)
- [快速开始](#快速开始)
- [API文档](#api文档)
- [工具列表](#工具列表)
- [部署指南](#部署指南)
- [开发指南](#开发指南)
- [常见问题](#常见问题)

## 概述

MCP服务器将HydroSIS的核心功能封装为标准化的MCP工具，使其可以：

- ✅ 与大语言模型（如通义千问、Claude等）无缝集成
- ✅ 提供RESTful API接口
- ✅ 支持异步任务处理
- ✅ 提供进度跟踪和实时反馈
- ✅ 支持多用户隔离和权限管理
- ✅ 云原生部署（Docker + Kubernetes）

### 核心特性

1. **MCP协议兼容**：完全符合Model Context Protocol标准
2. **工具化封装**：18个核心工具覆盖完整水文建模流程
3. **异步处理**：长时间运行的任务支持异步执行和进度查询
4. **权限管理**：基于RBAC的细粒度权限控制
5. **高可用性**：支持水平扩展和自动故障恢复

## 架构设计

```
┌─────────────────────────────────────────────────────┐
│              大模型服务（阿里云/OpenAI）              │
│                  通义千问 / Claude                   │
└────────────────────┬────────────────────────────────┘
                     │ MCP over HTTPS
                     ▼
┌─────────────────────────────────────────────────────┐
│              HydroSIS MCP Server                    │
│  ┌─────────────────────────────────────────────┐   │
│  │         MCP协议层 (FastAPI)                  │   │
│  │  - 工具注册  - 参数验证  - 结果序列化        │   │
│  └─────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────┐   │
│  │         工具封装层 (HydroSISTools)           │   │
│  │  - 项目管理  - 流域划分  - 模型配置          │   │
│  │  - 模拟运行  - 参数校准  - 结果分析          │   │
│  └─────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────┐   │
│  │         HydroSIS核心引擎                     │   │
│  │  - 产流模型  - 汇流计算  - GIS处理           │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## 快速开始

### 前置要求

- Python 3.10+
- Docker & Docker Compose (可选)
- Kubernetes 1.20+ (用于生产部署)

### 本地开发

1. **安装依赖**

```bash
cd /workspace
pip install -r requirements.txt
pip install -e .
```

2. **启动服务器**

```bash
python -m mcp_server.main
```

服务器将在 `http://localhost:8080` 启动。

3. **验证服务**

```bash
# 健康检查
curl http://localhost:8080/health

# 列出所有工具
curl http://localhost:8080/mcp/tools
```

### Docker部署

1. **使用Docker Compose**

```bash
cd mcp_server
docker-compose up -d
```

这将启动：
- MCP服务器 (端口8080)
- Redis (端口6379)
- PostgreSQL (端口5432)
- MinIO (端口9000, 9001)
- Celery Worker

2. **查看日志**

```bash
docker-compose logs -f mcp-server
```

3. **停止服务**

```bash
docker-compose down
```

### Kubernetes部署

1. **应用配置**

```bash
kubectl apply -f mcp_server/k8s-deployment.yaml
```

2. **查看部署状态**

```bash
kubectl get pods -n hydrosis
kubectl get services -n hydrosis
```

3. **查看日志**

```bash
kubectl logs -f deployment/hydrosis-mcp-server -n hydrosis
```

## API文档

### 核心端点

#### 1. 健康检查

```http
GET /health
```

响应：
```json
{
  "status": "healthy",
  "timestamp": "2025-10-26T12:00:00",
  "tools_count": 18
}
```

#### 2. 列出所有工具

```http
GET /mcp/tools
```

响应：
```json
{
  "tools": [
    {
      "name": "create_project",
      "description": "创建新的水文模拟项目",
      "category": "project_management",
      "inputSchema": {...}
    },
    ...
  ]
}
```

#### 3. 调用工具

```http
POST /mcp/tools/{tool_name}
Content-Type: application/json

{
  "user_id": "user123",
  "project_name": "我的水文项目",
  "description": "测试项目"
}
```

响应：
```json
{
  "content": [
    {
      "type": "text",
      "text": "{\"project_id\": \"...\", \"status\": \"created\"}"
    }
  ],
  "isError": false,
  "metadata": {
    "tool_name": "create_project",
    "timestamp": "2025-10-26T12:00:00"
  }
}
```

#### 4. 提交异步任务

```http
POST /tasks/submit
Content-Type: application/json

{
  "tool_name": "run_simulation",
  "arguments": {
    "project_id": "abc-123",
    "generate_report": true
  },
  "user_id": "user123",
  "callback_url": "https://your-server.com/callback"
}
```

响应：
```json
{
  "task_id": "task-uuid",
  "status": "submitted",
  "message": "任务已提交"
}
```

#### 5. 查询任务状态

```http
GET /tasks/{task_id}
```

响应：
```json
{
  "task_id": "task-uuid",
  "tool_name": "run_simulation",
  "status": "running",
  "progress": 45.5,
  "created_at": "2025-10-26T12:00:00",
  "started_at": "2025-10-26T12:00:05",
  "metadata": {
    "progress_messages": [
      {
        "timestamp": "2025-10-26T12:00:10",
        "progress": 20,
        "message": "流域划分中..."
      }
    ]
  }
}
```

## 工具列表

### 项目管理

| 工具名称 | 描述 | 分类 |
|---------|------|------|
| `create_project` | 创建新项目 | project_management |
| `list_projects` | 列出用户项目 | project_management |
| `get_project` | 获取项目详情 | project_management |

### GIS处理

| 工具名称 | 描述 | 分类 |
|---------|------|------|
| `delineate_watershed` | 流域划分 | gis_processing |
| `generate_gis_report` | 生成GIS报告 | visualization |

### 模型配置

| 工具名称 | 描述 | 分类 |
|---------|------|------|
| `configure_runoff_model` | 配置产流模型 | model_configuration |
| `configure_routing` | 配置汇流方法 | model_configuration |

### 数据管理

| 工具名称 | 描述 | 分类 |
|---------|------|------|
| `upload_forcing_data` | 上传驱动数据 | data_management |

### 模拟与分析

| 工具名称 | 描述 | 分类 |
|---------|------|------|
| `run_simulation` | 运行模拟 | simulation |
| `calibrate_parameters` | 参数校准 | calibration |
| `analyze_results` | 结果分析 | analysis |

### 详细工具说明

#### create_project

创建新的水文模拟项目。

**参数：**
```json
{
  "user_id": "string (必需)",
  "project_name": "string (必需)",
  "description": "string (可选)",
  "template": "basic|advanced|custom (可选，默认: basic)"
}
```

**返回：**
```json
{
  "project_id": "uuid",
  "name": "项目名称",
  "path": "/data/users/user123/projects/uuid",
  "status": "created",
  "message": "项目创建成功"
}
```

**示例：**
```python
import requests

response = requests.post(
    "http://localhost:8080/mcp/tools/create_project",
    json={
        "user_id": "user123",
        "project_name": "长江流域模拟",
        "description": "基于SCS方法的产流模拟",
        "template": "advanced"
    }
)

result = response.json()
print(f"项目ID: {result['content'][0]['text']}")
```

#### delineate_watershed

根据DEM和汇水点进行流域划分。

**参数：**
```json
{
  "project_id": "string (必需)",
  "dem_path": "string (必需)",
  "pour_points": [
    {
      "id": "string",
      "lon": "number",
      "lat": "number"
    }
  ],
  "burn_streams": "boolean (可选，默认: false)"
}
```

#### run_simulation

运行水文模拟。

**参数：**
```json
{
  "project_id": "string (必需)",
  "scenario_ids": ["string"] (可选),
  "start_date": "YYYY-MM-DD (可选)",
  "end_date": "YYYY-MM-DD (可选)",
  "generate_report": "boolean (可选，默认: true)"
}
```

**注意：** 这是一个耗时操作，建议使用异步任务方式调用。

## 部署指南

### 环境变量配置

创建 `.env` 文件：

```bash
# 基本配置
DATA_ROOT=/data
HOST=0.0.0.0
PORT=8080
WORKERS=4

# 安全配置
JWT_SECRET=your-super-secret-jwt-key
REQUEST_SECRET_KEY=your-request-signature-key

# 数据库配置
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=hydrosis
POSTGRES_USER=hydrosis
POSTGRES_PASSWORD=your-secure-password

# Redis配置
REDIS_HOST=redis
REDIS_PORT=6379

# MinIO配置
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

# IP白名单（可选）
IP_WHITELIST=192.168.1.0/24,10.0.0.0/8

# 日志级别
LOG_LEVEL=INFO
```

### 生产部署建议

1. **使用HTTPS**
   - 配置SSL证书
   - 使用Nginx反向代理

2. **数据持久化**
   - 挂载持久化卷（PV/PVC）
   - 定期备份数据库

3. **监控告警**
   - Prometheus + Grafana
   - 配置资源告警

4. **负载均衡**
   - 使用Kubernetes Service
   - 配置HPA自动伸缩

5. **日志管理**
   - 集中式日志收集（ELK/Loki）
   - 日志轮转和归档

## 开发指南

### 添加新工具

1. **在 `hydrosis_tools.py` 中定义工具方法**

```python
async def my_new_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
    """我的新工具"""
    # 实现逻辑
    return {"status": "success"}
```

2. **在 `_register_all_tools()` 中注册工具**

```python
self.mcp.register_tool(
    name="my_new_tool",
    func=self.my_new_tool,
    description="这是一个新工具",
    category="custom",
    schema={
        "type": "object",
        "properties": {
            "param1": {"type": "string"}
        },
        "required": ["param1"]
    }
)
```

3. **重启服务器并测试**

```bash
curl http://localhost:8080/mcp/tools/my_new_tool \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"param1": "value"}'
```

### 运行测试

```bash
# 单元测试
pytest tests/test_mcp_server.py

# 集成测试
pytest tests/test_integration.py

# 覆盖率报告
pytest --cov=mcp_server --cov-report=html
```

### 代码规范

- 遵循PEP 8编码规范
- 使用类型注解
- 编写文档字符串
- 添加单元测试

## 使用示例

### Python客户端

```python
import requests

class MCPClient:
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url
    
    def list_tools(self):
        """列出所有工具"""
        response = requests.get(f"{self.base_url}/mcp/tools")
        return response.json()
    
    def call_tool(self, tool_name, arguments):
        """调用工具"""
        response = requests.post(
            f"{self.base_url}/mcp/tools/{tool_name}",
            json=arguments
        )
        return response.json()
    
    def submit_task(self, tool_name, arguments, user_id):
        """提交异步任务"""
        response = requests.post(
            f"{self.base_url}/tasks/submit",
            json={
                "tool_name": tool_name,
                "arguments": arguments,
                "user_id": user_id
            }
        )
        return response.json()
    
    def get_task_status(self, task_id):
        """查询任务状态"""
        response = requests.get(f"{self.base_url}/tasks/{task_id}")
        return response.json()

# 使用示例
client = MCPClient()

# 创建项目
result = client.call_tool("create_project", {
    "user_id": "user123",
    "project_name": "我的项目"
})
print(result)

# 提交模拟任务
task = client.submit_task("run_simulation", {
    "project_id": "project-id-here"
}, "user123")
print(f"任务ID: {task['task_id']}")

# 查询任务状态
status = client.get_task_status(task['task_id'])
print(f"进度: {status['progress']}%")
```

### 与通义千问集成

```python
from dashscope import Generation

# 获取MCP工具列表
mcp_tools = requests.get("http://localhost:8080/mcp/tools").json()['tools']

# 转换为通义千问工具格式
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

# 调用通义千问
messages = [
    {"role": "user", "content": "帮我创建一个水文模拟项目"}
]

response = Generation.call(
    model='qwen-max',
    messages=messages,
    tools=qwen_tools,
    result_format='message'
)

# 处理工具调用
if response.output.get('tool_calls'):
    for tool_call in response.output['tool_calls']:
        tool_name = tool_call['function']['name']
        tool_args = tool_call['function']['arguments']
        
        # 调用MCP工具
        result = requests.post(
            f"http://localhost:8080/mcp/tools/{tool_name}",
            json=tool_args
        ).json()
        
        print(f"工具执行结果: {result}")
```

## 常见问题

### Q1: 如何配置认证？

A: 在环境变量中设置JWT密钥，并在请求头中添加Bearer Token：

```python
headers = {
    "Authorization": f"Bearer {your_jwt_token}"
}
requests.get("http://localhost:8080/mcp/tools", headers=headers)
```

### Q2: 任务执行失败怎么办？

A: 查询任务状态获取错误信息：

```python
status = client.get_task_status(task_id)
if status['status'] == 'failed':
    print(f"错误: {status['error']}")
```

### Q3: 如何设置数据存储路径？

A: 通过环境变量 `DATA_ROOT` 配置：

```bash
export DATA_ROOT=/path/to/your/data
python -m mcp_server.main
```

### Q4: 支持哪些产流模型？

A: 目前支持：
- SCS (Soil Conservation Service)
- 新安江模型 (XinAnJiang)
- HBV模型
- HYMOD模型
- VIC模型
- WETSPA模型

### Q5: 如何扩展更多工具？

A: 参考[开发指南](#开发指南)中的"添加新工具"部分。

## 技术支持

- 📧 邮箱: support@hydrosis.example.com
- 🐛 问题反馈: https://github.com/hydrosis/issues
- 📚 文档: https://docs.hydrosis.example.com
- 💬 社区: https://community.hydrosis.example.com

## 许可证

MIT License

## 更新日志

### v1.0.0 (2025-10-26)

- ✨ 初始版本发布
- ✅ 实现18个核心MCP工具
- ✅ 支持异步任务处理
- ✅ 完整的Docker和K8s部署配置
- ✅ RBAC权限管理
- ✅ 进度跟踪和实时反馈

## 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 致谢

感谢HydroSIS开发团队和所有贡献者！
