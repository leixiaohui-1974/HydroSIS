# HydroSIS Docker 部署指南

本文档介绍如何使用Docker容器化部署HydroSIS分布式水文模拟框架。

## 目录

- [先决条件](#先决条件)
- [快速开始](#快速开始)
- [配置选项](#配置选项)
- [数据持久化](#数据持久化)
- [生产部署](#生产部署)
- [故障排除](#故障排除)

## 先决条件

在开始之前，请确保您的系统已安装以下软件：

- [Docker](https://docs.docker.com/get-docker/)（版本 20.10 或更高）
- [Docker Compose](https://docs.docker.com/compose/install/)（版本 1.29 或更高）

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/your-org/HydroSIS.git
cd HydroSIS
```

### 2. 构建并运行

使用提供的Makefile可以快速构建和运行HydroSIS：

```bash
# 构建Docker镜像
make build

# 启动所有服务（包括数据库）
make run
```

或者使用Docker Compose直接：

```bash
# 构建并启动服务
docker-compose up -d --build
```

### 3. 访问应用

- **Web界面**: <http://localhost:8000>
- **API文档**: <http://localhost:8000/docs>
- **数据库**: localhost:5432（用户: hydrosis，密码： password）

### 4. 验证安装

```bash
# 检查服务状态
docker-compose ps

# 查看日志
make logs

# 运行健康检查
curl http://localhost:8000/health
```

## 配置选项

### 环境变量

您可以通过环境变量配置HydroSIS：

```bash
# 数据库连接
HYDROSIS_PORTAL_DB_URL=postgresql://user:password@host:port/database

# 配置文件路径
HYDROSIS_PORTAL_CONFIG=/app/config/portal_config.json

# 日志级别
LOG_LEVEL=INFO
```

### 配置文件

将配置文件放在`config/`目录中：

```bash
# 创建配置目录
mkdir -p config

# 复制示例配置
cp config/example_model.yaml config/model.yaml
cp config/portal_config.json.example config/portal_config.json
```

### 数据目录

确保以下目录存在并包含必要的数据：

```bash
# 创建数据目录
mkdir -p data/forcing data/observations data/gis

# 复制示例数据
cp -r data/sample/* data/
```

## 数据持久化

### 数据库数据

PostgreSQL数据持久化在Docker卷中：

```bash
# 备份数据库
make backup

# 恢复数据库
make restore BACKUP_FILE=backup_20231201_120000.sql
```

### 应用数据

应用数据（输入文件、结果等）通过主机卷挂载：

```yaml
volumes:
  - ./data:/app/data
  - ./results:/app/results
  - ./config:/app/config
```

### Redis缓存（可选）

如果启用Redis，缓存数据也会持久化：

```yaml
redis:
  volumes:
    - redis_data:/data
```

## 生产部署

### 1. 生产配置

创建生产环境的配置文件：

```bash
# docker-compose.prod.yml
version: '3.8'

services:
  hydrosis-app:
    image: hydrosis:latest
    environment:
      - HYDROSIS_PORTAL_DB_URL=${DATABASE_URL}
      - LOG_LEVEL=WARNING
    restart: always
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

### 2. 使用Nginx反向代理

```bash
# 启动Nginx
docker-compose -f docker-compose.yml -f docker-compose.nginx.yml up -d
```

### 3. SSL/TLS配置

将SSL证书放在`nginx/ssl/`目录中：

```bash
mkdir -p nginx/ssl
cp your-cert.pem nginx/ssl/
cp your-key.pem nginx/ssl/
```

### 4. 监控和日志

```bash
# 查看资源使用情况
make monitor

# 查看应用日志
docker-compose logs -f hydrosis-app

# 查看数据库日志
docker-compose logs -f postgres
```

## 常用命令

### 开发命令

```bash
# 开发环境（构建并运行）
make dev

# 运行测试
make test

# 代码检查
make lint

# 代码格式化
make format

# 进入容器shell
make shell
```

### 维护命令

```bash
# 更新依赖
make update

# 安全扫描
make security

# 性能分析
make profile

# 生成文档
make docs
```

### 数据库命令

```bash
# 数据库迁移
make migrate

# 创建数据库模式
make schema
```

### 清理命令

```bash
# 停止服务
make stop

# 清理所有数据（谨慎使用）
make clean
```

## 故障排除

### 常见问题

#### 1. 容器启动失败

```bash
# 查看详细错误信息
docker-compose logs hydrosis-app

# 检查容器状态
docker-compose ps
```

#### 2. 数据库连接失败

```bash
# 检查数据库是否运行
docker-compose logs postgres

# 测试数据库连接
docker-compose exec postgres psql -U hydrosis -d hydrosis -c "SELECT 1;"
```

#### 3. 权限问题

```bash
# 检查文件权限
ls -la data/ results/ config/

# 修复权限（如果需要）
sudo chown -R $USER:$USER data/ results/ config/
```

#### 4. 内存不足

```bash
# 检查内存使用情况
docker stats

# 增加Docker内存限制
# 在docker-compose.yml中添加：
services:
  hydrosis-app:
    deploy:
      resources:
        limits:
          memory: 8G
```

#### 5. 端口冲突

```bash
# 检查端口占用
netstat -tulpn | grep :8000

# 修改端口映射
# 在docker-compose.yml中修改：
ports:
  - "8080:8000"  # 使用8080端口
```

### 调试技巧

#### 1. 进入容器调试

```bash
# 进入运行中的容器
make shell

# 或者临时启动调试容器
docker-compose run --rm hydrosis-app /bin/bash
```

#### 2. 查看容器内部状态

```bash
# 查看环境变量
docker-compose exec hydrosis-app env

# 查看安装的包
docker-compose exec hydrosis-app pip list

# 查看Python版本
docker-compose exec hydrosis-app python --version
```

#### 3. 重建镜像

```bash
# 强制重建镜像
docker-compose build --no-cache

# 删除旧镜像
docker rmi hydrosis:latest
```

## 性能优化

### 1. 资源限制

在`docker-compose.yml`中设置合适的资源限制：

```yaml
services:
  hydrosis-app:
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G
```

### 2. 并行处理

HydroSIS支持并行计算，可以通过环境变量启用：

```yaml
environment:
  - HYDROSIS_PARALLEL_WORKERS=4
  - HYDROSIS_PARALLEL_USE_PROCESSES=true
```

### 3. 缓存优化

启用Redis缓存以提高性能：

```yaml
services:
  hydrosis-app:
    environment:
      - HYDROSIS_CACHE_URL=redis://redis:6379/0
  redis:
    image: redis:6-alpine
    command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
```

## 安全考虑

### 1. 网络安全

- 使用自定义网络而不是默认网络
- 限制容器间通信
- 使用防火墙规则

### 2. 数据安全

- 定期备份数据
- 使用强密码
- 启用数据库SSL连接

### 3. 容器安全

- 使用非root用户运行容器
- 定期更新基础镜像
- 扫描容器漏洞

```bash
# 安全扫描
make security

# 查看容器用户
docker-compose exec hydrosis-app whoami
```

## 备份和恢复

### 自动备份

设置定时备份任务：

```bash
# 创建备份脚本
cat > backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
docker-compose exec postgres pg_dump -U hydrosis hydrosis > backup_$DATE.sql
gzip backup_$DATE.sql
EOF

chmod +x backup.sh

# 添加到crontab
# 0 2 * * * /path/to/backup.sh
```

### 灾难恢复

```bash
# 1. 停止服务
make stop

# 2. 恢复数据
make restore BACKUP_FILE=backup_20231201_120000.sql.gz

# 3. 重启服务
make run
```

## 更新和维护

### 更新应用

```bash
# 1. 拉取最新代码
git pull origin main

# 2. 重新构建镜像
make build

# 3. 重启服务
make stop
make run
```

### 更新依赖

```bash
# 更新Python依赖
make update

# 重新构建镜像
docker-compose build --no-cache
```

## 支持

如果您在使用Docker部署HydroSIS时遇到问题，请：

1. 查看本文档的故障排除部分
2. 检查[GitHub Issues](https://github.com/your-org/HydroSIS/issues)
3. 创建新的Issue并提供详细信息

## 最后更新

2023年12月
