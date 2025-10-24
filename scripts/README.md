# HydroSIS 脚本目录

本目录包含HydroSIS的各类实用脚本，按功能分类组织。

---

## 📂 目录结构

```
scripts/
├── calibration/        # 参数率定脚本
├── diagnostics/        # 诊断分析脚本
├── workflows/          # 完整工作流脚本
├── analysis/           # 数据分析和对比脚本
└── README.md          # 本文件
```

---

## 🔧 脚本分类

### 1. 参数率定脚本 (`calibration/`)

HBV模型参数校准和优化相关脚本。

| 脚本 | 功能 | 使用场景 |
|-----|------|---------|
| `calibrate_hbv_all_zones.py` | 全流域多分区HBV参数率定 | ⭐ 主要率定脚本 |
| `calibrate_zone1_hbv_parameters.py` | Zone 1单独率定 | 单分区调试 |
| `calibrate_zone1_with_sensitivity.py` | 带敏感性分析的率定 | 参数敏感性研究 |
| `calibrate_watershed_cascading.py` | 级联率定 | 多分区逐级率定 |
| `calibrate_watershed_hybrid.py` | 混合率定策略 | 高级率定 |

**示例用法**:
```bash
# 率定所有分区的HBV参数
cd /path/to/HydroSIS
python scripts/calibration/calibrate_hbv_all_zones.py \
    --config config/workflow_config.yaml \
    --validation-config config/validation_criteria.yaml
```

---

### 2. 诊断分析脚本 (`diagnostics/`)

数据质量检查和问题诊断工具。

| 脚本 | 功能 | 使用场景 |
|-----|------|---------|
| `diagnose_zone2_precipitation.py` | Zone 2降雨数据诊断 | ⭐ 降雨质量检查 |
| `diagnose_hbv_configuration.py` | HBV配置诊断 | HBV参数检查 |
| `diagnose_water_balance.py` | 水量平衡诊断 | 水量平衡检查 |

**示例用法**:
```bash
# 诊断Zone 2降雨数据质量
python scripts/diagnostics/diagnose_zone2_precipitation.py
```

**输出**:
- 降雨空间分布统计
- 空间变异系数
- 诊断报告和可视化图表

---

### 3. 完整工作流脚本 (`workflows/`)

端到端的水文模拟工作流。

| 脚本 | 功能 | 使用场景 |
|-----|------|---------|
| `run_upper_truckee_complete_11steps.py` | 完整11步工作流 | ⭐ 主要工作流 |
| `run_upper_truckee_complete.py` | 简化工作流 | 快速模拟 |
| `run_complete_workflow.py` | 通用工作流模板 | 新项目模板 |

**示例用法**:
```bash
# 运行完整11步工作流
python scripts/workflows/run_upper_truckee_complete_11steps.py
```

**工作流步骤**:
1. DEM处理和地形分析
2. 汇水点生成
3. 参数分区和子流域划分
4. 河道断面提取
5. 雨量站分布
6-8. 雨量处理
9-10. 水文水动力模拟
11. 结果报告生成

---

### 4. 分析和对比脚本 (`analysis/`)

模型对比、结果分析和测试脚本。

| 脚本 | 功能 | 使用场景 |
|-----|------|---------|
| `analyze_runoff_coefficients.py` | 径流系数分析 | 结果质量检查 |
| `compare_calibration_results.py` | 对比不同率定结果 | 率定效果评估 |
| `compare_models_comprehensive.py` | 多模型综合对比 | 模型选择 |
| `test_parallel_hbv.py` | HBV并行化测试 | 性能验证 |
| `test_validation.py` | 验证框架测试 | 功能测试 |

**示例用法**:
```bash
# 分析径流系数
python scripts/analysis/analyze_runoff_coefficients.py

# 测试HBV并行化
python scripts/analysis/test_parallel_hbv.py
```

---

## 🚀 快速开始

### 新项目工作流

```bash
# 1. 检查配置
python tools/check_config.py

# 2. 运行完整工作流
python scripts/workflows/run_upper_truckee_complete_11steps.py

# 3. 批量验证结果
python tools/batch_validate.py --report

# 4. 分析径流系数
python scripts/analysis/analyze_runoff_coefficients.py
```

---

## 📝 脚本开发规范

### 命名规范

- **动词开头**: `run_`, `calibrate_`, `diagnose_`, `analyze_`, `compare_`
- **描述性**: 清晰说明脚本功能
- **snake_case**: 使用下划线分隔

### 代码规范

```python
#!/usr/bin/env python3
"""脚本简短描述

详细功能说明

用法:
    python scripts/category/script_name.py [options]

示例:
    python scripts/calibration/calibrate_hbv_all_zones.py \
        --config config/workflow_config.yaml
"""
import argparse
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='脚本描述')
    # 添加参数...
    args = parser.parse_args()

    # 实现逻辑...

    print("✅ 完成")

if __name__ == '__main__':
    main()
```

### 必需元素

- [ ] Shebang行 (`#!/usr/bin/env python3`)
- [ ] 完整的docstring
- [ ] 使用argparse处理参数
- [ ] 路径处理（添加项目根目录到sys.path）
- [ ] 清晰的输出信息
- [ ] 适当的错误处理

---

## 🔄 迁移说明

### 从根目录迁移

之前在项目根目录的脚本已迁移到此目录：

```
根目录 (旧)                          → scripts/ (新)
├── calibrate_*.py                  → calibration/
├── diagnose_*.py                   → diagnostics/
├── run_*.py                        → workflows/
└── analyze_*, compare_*, test_*.py → analysis/
```

### 路径更新

使用脚本时需要更新路径：

```bash
# 旧方式
python calibrate_hbv_all_zones.py

# 新方式
python scripts/calibration/calibrate_hbv_all_zones.py
```

或者从项目根目录运行：
```bash
cd /path/to/HydroSIS
python scripts/workflows/run_upper_truckee_complete_11steps.py
```

---

## 📚 相关文档

- [用户手册](../docs/用户手册.md) - 完整使用指南
- [开发指南](../docs/开发指南.md) - 开发规范
- [工具脚本](../tools/README.md) - 实用工具
- [代码分析报告](../docs/code_analysis_report.md) - 项目分析

---

## 🤝 贡献脚本

欢迎贡献新的脚本！请遵循：

1. 选择合适的分类目录
2. 遵循命名和代码规范
3. 添加完整的文档字符串
4. 更新本README
5. 提交PR

---

## ⚙️ 维护日志

### 2025-01-24
- ✅ 初始化scripts目录结构
- ✅ 从根目录迁移所有脚本
- ✅ 创建README文档
- ✅ 按功能分类组织

---

**维护者**: HydroSIS开发团队
**最后更新**: 2025-01-24
