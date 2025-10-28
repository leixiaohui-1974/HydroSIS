#!/bin/bash
# 启动双智能体服务器脚本

echo "========================================"
echo "HydroSIS 双智能体系统启动脚本"
echo "========================================"

# 检查环境变量
if [ -z "$QWEN_API_KEY" ]; then
    echo "⚠️  警告: 未设置QWEN_API_KEY"
    echo "   将使用Mock模式"
    echo "   设置方法: export QWEN_API_KEY='sk-your-key'"
    echo ""
fi

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到python3"
    exit 1
fi

# 检查依赖
echo "检查依赖..."
python3 -c "import fastapi, uvicorn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  FastAPI未安装，正在安装..."
    pip install fastapi uvicorn
fi

# 创建日志目录
mkdir -p /workspace/logs

echo ""
echo "启动服务器..."
echo ""

# 选项1: Docker模式
echo "选项1: 使用Docker Compose启动（推荐）"
echo "  cd /workspace && docker-compose -f docker-compose.twin-agent.yml up -d"
echo ""

# 选项2: 本地模式
echo "选项2: 本地启动"
echo ""

echo "2.1 启动HydroMind (认知智能体)..."
echo "  cd /workspace && python3 -m mcp_server_mind.main &"
echo "  # 监听端口: 8081"
echo ""

echo "2.2 启动HydroCompute (机理智能体)..."
echo "  cd /workspace && python3 -m mcp_server.main &"
echo "  # 监听端口: 8080"
echo ""

echo "2.3 测试连接..."
echo "  sleep 5"
echo "  curl http://localhost:8081/health  # HydroMind"
echo "  curl http://localhost:8080/health  # HydroCompute"
echo ""

echo "2.4 运行集成测试..."
echo "  python3 tests/integration/test_end_to_end.py"
echo ""

echo "========================================"
echo "请选择启动方式并执行相应命令"
echo "========================================"
