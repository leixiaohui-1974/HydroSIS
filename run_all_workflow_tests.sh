#!/bin/bash
# HydroSIS 工作流测试运行脚本

echo "################################################################################"
echo "HydroSIS 模块化API - 多工作流测试套件"
echo "################################################################################"
echo ""

# 设置颜色
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 创建输出目录
OUTPUT_DIR="results/workflow_tests"
mkdir -p "$OUTPUT_DIR"

echo "📁 测试输出目录: $OUTPUT_DIR"
echo ""

# 测试场景列表
declare -a scenarios=(
    "01_minimal_terrain"
    "02_two_step_basic"
    "03_three_step_delineation"
    "04_precipitation_analysis"
    "05_hydrologic_simulation"
    "06_calibration_workflow"
    "07_parallel_analysis"
    "08_complete_eleven_steps"
)

# 统计变量
total=0
passed=0
failed=0
skipped=0

# 运行每个测试场景
for scenario in "${scenarios[@]}"; do
    total=$((total + 1))
    config_file="config/workflows/test_scenarios/${scenario}.yaml"
    
    echo "--------------------------------------------------------------------------------"
    echo "测试场景 ${total}/8: ${scenario}"
    echo "配置文件: ${config_file}"
    echo "--------------------------------------------------------------------------------"
    
    # 检查配置文件是否存在
    if [ ! -f "$config_file" ]; then
        echo -e "${YELLOW}⚠ 配置文件不存在，跳过${NC}"
        skipped=$((skipped + 1))
        echo ""
        continue
    fi
    
    # 运行测试
    python3 tests/test_multiple_workflows.py --test "$config_file" --output "$OUTPUT_DIR"
    
    # 检查退出码
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ 测试通过${NC}"
        passed=$((passed + 1))
    else
        echo -e "${RED}❌ 测试失败${NC}"
        failed=$((failed + 1))
    fi
    
    echo ""
done

# 打印总结
echo "################################################################################"
echo "测试总结"
echo "################################################################################"
echo ""
echo "测试总数: $total"
echo -e "${GREEN}通过: $passed${NC}"
echo -e "${RED}失败: $failed${NC}"
echo -e "${YELLOW}跳过: $skipped${NC}"

if [ $total -gt 0 ]; then
    pass_rate=$((passed * 100 / total))
    echo "通过率: ${pass_rate}%"
fi

echo ""
echo "📊 详细报告:"
echo "  - JSON: $OUTPUT_DIR/workflow_test_report.json"
echo "  - Markdown: $OUTPUT_DIR/workflow_test_report.md"
echo ""

# 返回退出码
if [ $failed -eq 0 ]; then
    echo -e "${GREEN}🎉 所有测试通过！${NC}"
    exit 0
else
    echo -e "${RED}❌ 有 $failed 个测试失败${NC}"
    exit 1
fi
