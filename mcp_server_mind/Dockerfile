# HydroMind Agent Dockerfile
# 水文认知智能体容器镜像

FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . /app/mcp_server_mind
COPY mcp_orchestrator /app/mcp_orchestrator

# 安装Python依赖
RUN pip install --no-cache-dir \
    fastapi>=0.100.0 \
    uvicorn>=0.23.0 \
    pydantic>=2.0

# 设置环境变量
ENV PYTHONPATH=/app
ENV PORT=8081
ENV HOST=0.0.0.0

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# 暴露端口
EXPOSE 8081

# 启动命令
CMD ["python3", "-m", "mcp_server_mind.main"]
