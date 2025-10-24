# HydroSIS 工具集

实用工具脚本，帮助您更高效地使用HydroSIS。

---

## 📋 工具列表

### 1. 批量验证工具 (`batch_validate.py`)

**功能**: 对项目中的所有关键数据进行批量质量检查

**用法**:
```bash
# 基本用法
python tools/batch_validate.py

# 指定配置文件
python tools/batch_validate.py --config config/workflow_config.yaml

# 生成JSON验证报告
python tools/batch_validate.py --report

# 自定义报告输出路径
python tools/batch_validate.py --report --output my_validation_report.json
```

**验证内容**:
- ✅ 降雨数据质量（空间CV、站点相关性、缺测率）
- ✅ 流域几何数据（几何有效性、面积范围）
- ✅ 河网拓扑一致性（下游引用、环路检测、出口点）
- ✅ 径流时间序列（缺失值、数值范围、变化率、趋势）

**输出示例**:
```
================================================================================
批量验证工具
================================================================================
项目目录: results/upper_truckee_complete_11steps

⚙ 验证降雨数据...
✓ 降雨数据验证通过
⚠ 降雨数据警告:
  - 空间变异系数偏高: 0.47 > 0.40

⚙ 验证流域几何...
✓ 流域几何验证通过

⚙ 验证河网拓扑...
✓ 河网拓扑验证通过

⚙ 验证径流时间序列...
✓ 径流时序验证完成: 6/6 个分区通过

✅ 批量验证完成
总体通过率: 92.3%
```

**报告格式** (`validation_report.json`):
```json
{
  "overall": {
    "total_checks": 9,
    "passed_checks": 8,
    "pass_rate": "88.9%"
  },
  "validation_summary": {
    "precipitation": {
      "is_valid": true,
      "errors": 0,
      "warnings": 1
    },
    "runoff_timeseries": {
      "total": 6,
      "passed": 6,
      "pass_rate": "100.0%"
    }
  },
  "details": {
    "precipitation": {
      "errors": [],
      "warnings": ["空间变异系数偏高: 0.47 > 0.40"],
      "metrics": {
        "spatial_cv": 0.47,
        "mean_correlation": 0.62
      }
    }
  }
}
```

---

### 2. 配置检查工具 (`check_config.py`)

**功能**: 检查workflow_config.yaml的完整性和合理性

**用法**:
```bash
# 基本用法
python tools/check_config.py

# 指定配置文件
python tools/check_config.py --config config/workflow_config.yaml

# 严格模式（警告也算失败）
python tools/check_config.py --strict
```

**检查内容**:
- ✅ 文件存在性和语法正确性
- ✅ 必需字段完整性（project_name、directories、hbv_parameters等）
- ✅ HBV参数合理性（范围检查）
- ✅ 雨量站配置合理性（密度等级）
- ✅ 并行配置合理性（worker数量）
- ✅ 目录配置完整性

**输出示例**:
```
================================================================================
配置文件检查工具
================================================================================
配置文件: config/workflow_config.yaml
严格模式: 否

⚙ 检查必需字段...
✓ 找到字段: project_name
✓ 找到字段: directories
✓ 找到字段: hbv_parameters

⚙ 检查HBV参数...
✓ FC = 300
✓ BETA = 2.0
✓ LP = 0.7
✓ K0 = 0.1
✓ K1 = 0.05
✓ K2 = 0.01
✓ PERC = 2.0
✓ UZL = 50.0

⚙ 检查雨量站配置...
✓ target_density = 0.01
✓ min_distance_m = 1000
✓ random_seed = 42
雨量站密度良好: 0.01 >= 0.01

⚙ 检查并行配置...
✓ max_workers = 4

⚙ 检查目录配置...
✓ base_results = results/${project_name}
  展开为: results/upper_truckee_river

================================================================================
检查结果汇总
================================================================================

✓ 信息 (15条):
  ✓ 配置文件加载成功
  ✓ 找到字段: project_name
  ...

⚠ 警告 (0条):

✗ 错误 (0条):

================================================================================
✅ 配置检查通过（无警告无错误）
================================================================================
```

**退出码**:
- `0`: 检查通过
- `1`: 检查失败

**脚本集成示例**:
```bash
#!/bin/bash
# 运行前先检查配置
python tools/check_config.py --config config/workflow_config.yaml || exit 1

# 配置检查通过，继续运行工作流
python run_upper_truckee_complete_11steps.py
```

---

## 🚀 推荐工作流

### 新项目启动检查清单

```bash
# 1. 检查配置文件
python tools/check_config.py

# 2. 运行工作流
python run_upper_truckee_complete_11steps.py

# 3. 批量验证结果
python tools/batch_validate.py --report

# 4. 查看验证报告
cat validation_report.json | python -m json.tool
```

### 持续集成 (CI) 示例

```yaml
# .github/workflows/validation.yml
name: Data Validation

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Check configuration
        run: python tools/check_config.py --strict

      - name: Run batch validation
        run: python tools/batch_validate.py --report

      - name: Upload validation report
        uses: actions/upload-artifact@v2
        with:
          name: validation-report
          path: validation_report.json
```

---

## 📝 添加自己的工具

### 工具模板

```python
#!/usr/bin/env python3
"""工具名称

工具描述

用法:
    python tools/my_tool.py [选项]
"""
import argparse
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hydrosis.config import load_workflow_config
from hydrosis.validation import ValidationResult


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='工具描述')
    parser.add_argument('--config', type=Path, default=Path('config/workflow_config.yaml'))

    args = parser.parse_args()

    # 实现工具逻辑
    print("工具执行中...")

    # 返回退出码
    sys.exit(0)


if __name__ == '__main__':
    main()
```

### 工具开发指南

1. **命名**: 使用snake_case，描述性名称
2. **文档**: 添加完整的docstring和用法说明
3. **参数**: 使用argparse提供清晰的命令行接口
4. **退出码**: 0表示成功，非0表示失败
5. **输出**: 使用清晰的格式化输出（✓/✗/⚠符号）
6. **错误处理**: 捕获异常并提供有用的错误信息

---

## 🔧 故障排查

### 工具运行失败

**症状**: `ModuleNotFoundError: No module named 'hydrosis'`

**解决方案**:
```bash
# 确保在项目根目录运行
cd /path/to/HydroSIS
python tools/batch_validate.py
```

### 找不到数据文件

**症状**: `降雨数据文件不存在`

**解决方案**:
```bash
# 检查项目是否已运行工作流
ls results/upper_truckee_complete_11steps/

# 如果没有结果，先运行工作流
python run_upper_truckee_complete_11steps.py
```

### 配置文件格式错误

**症状**: `配置文件加载失败: yaml.scanner.ScannerError`

**解决方案**:
```bash
# 使用工具检查配置
python tools/check_config.py

# 或手动验证YAML语法
python -c "import yaml; yaml.safe_load(open('config/workflow_config.yaml'))"
```

---

## 📚 相关文档

- [用户手册](../docs/用户手册.md) - 完整使用指南
- [开发指南](../docs/开发指南.md) - 开发规范
- [验证框架API](../docs/api/validation_framework.md) - API参考

---

## 🤝 贡献工具

欢迎贡献新的实用工具！请遵循以下步骤：

1. 在`tools/`目录创建新工具脚本
2. 添加完整的文档字符串
3. 更新本README添加工具说明
4. 提交PR

---

**工具版本**: v1.0.0
**最后更新**: 2025-01-24
