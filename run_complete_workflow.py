#!/usr/bin/env python3
"""运行完整的10步工作流并生成所有结果图、表和报告"""
from __future__ import annotations

import sys
from pathlib import Path

# 确保可以导入HydroSIS模块
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def main():
    """运行完整的HydroSIS工作流示例"""
    print("=" * 80)
    print("HydroSIS 完整工作流演示")
    print("包含：流域划分、雨量站点生成、面雨量计算、河道断面提取、产汇流计算")
    print("=" * 80)
    print()

    # 运行示例工作流
    print("开始运行完整工作流...")
    print("-" * 80)

    from examples.run_sample_workflow import main as run_sample

    try:
        run_sample()
        print()
        print("=" * 80)
        print("工作流运行完成！")
        print("=" * 80)
        print()
        print("生成的结果包括：")
        print("  - 流域划分图和统计表")
        print("  - 子流域面积分布图")
        print("  - 参数区划分")
        print("  - 情景流量对比图")
        print("  - 流量过程线")
        print("  - 评估指标报告")
        print("  - GIS交互式地图报告")
        print()
        print("结果保存在: results/example_run/")
        print()
        return 0
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
