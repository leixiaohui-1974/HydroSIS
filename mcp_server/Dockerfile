# HydroSIS MCP Server Dockerfile
# 基于Python 3.10的官方镜像

FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libgdal-dev \
    gdal-bin \
    python3-gdal \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt /app/

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY hydrosis/ /app/hydrosis/
COPY mcp_server/ /app/mcp_server/
COPY config/ /app/config/
COPY setup.py /app/
COPY README.md /app/

# 安装HydroSIS包
RUN pip install -e .

# 创建数据目录
RUN mkdir -p /data

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV DATA_ROOT=/data
ENV HOST=0.0.0.0
ENV PORT=8080
ENV WORKERS=4

# 暴露端口
EXPOSE 8080

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')"

# 启动命令
CMD ["python", "-m", "mcp_server.main"]
