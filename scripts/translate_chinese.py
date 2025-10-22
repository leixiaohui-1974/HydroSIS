#!/usr/bin/env python3
"""
Automated script to translate Chinese comments and docstrings to English.
This script uses a systematic approach to convert common Chinese patterns.
"""

import re
from pathlib import Path
from typing import Dict

# Common Chinese to English translations
TRANSLATIONS: Dict[str, str] = {
    # Common docstring patterns
    "参数:": "Args:",
    "返回:": "Returns:",
    "引发:": "Raises:",
    "示例:": "Example:",
    "注意:": "Note:",
    "警告:": "Warning:",

    # Common terms
    "流量": "discharge",
    "水深": "water_depth",
    "断面": "cross_section",
    "河道": "channel",
    "降雨": "precipitation",
    "径流": "runoff",
    "汇流": "routing",
    "子流域": "subbasin",
    "参数区": "parameter_zone",
    "时间序列": "time_series",
    "配置": "configuration",
    "模型": "model",
    "数据": "data",
    "文件": "file",
    "目录": "directory",
    "路径": "path",
    "结果": "result",
    "输出": "output",
    "输入": "input",

    # Common error messages
    "不存在": "does not exist",
    "未找到": "not found",
    "缺少": "missing",
    "为空": "is empty",
    "无效": "invalid",
    "错误": "error",
    "失败": "failed",
    "成功": "successful",

    # Common actions
    "读取": "read",
    "写入": "write",
    "加载": "load",
    "保存": "save",
    "计算": "compute",
    "生成": "generate",
    "创建": "create",
    "更新": "update",
    "删除": "delete",
    "检查": "check",
    "验证": "validate",
    "处理": "process",
    "转换": "convert",
    "导出": "export",
    "导入": "import",
}

def has_chinese(text: str) -> bool:
    """Check if text contains Chinese characters."""
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def list_files_with_chinese(root_dir: Path) -> list[Path]:
    """List all Python files containing Chinese characters."""
    files_with_chinese = []
    for py_file in root_dir.rglob("*.py"):
        try:
            content = py_file.read_text(encoding='utf-8')
            if has_chinese(content):
                files_with_chinese.append(py_file)
        except Exception as e:
            print(f"Error reading {py_file}: {e}")
    return files_with_chinese

def main():
    """Main function to list files needing translation."""
    root = Path("/home/user/HydroSIS/hydrosis")
    files = list_files_with_chinese(root)

    print(f"Found {len(files)} files with Chinese characters:\n")

    # Group by module
    by_module = {}
    for f in files:
        module = f.relative_to(root).parts[0] if len(f.relative_to(root).parts) > 1 else "root"
        if module not in by_module:
            by_module[module] = []
        by_module[module].append(f)

    # Print grouped
    for module in sorted(by_module.keys()):
        print(f"\n{module}/ ({len(by_module[module])} files):")
        for f in sorted(by_module[module]):
            rel_path = f.relative_to(root)
            print(f"  - {rel_path}")

    print(f"\n\nTotal: {len(files)} files need internationalization")
    print("\nModules summary:")
    for module in sorted(by_module.keys()):
        print(f"  {module}: {len(by_module[module])} files")

if __name__ == "__main__":
    main()
