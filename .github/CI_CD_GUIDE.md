# CI/CD 配置指南

## 概述

本项目使用 GitHub Actions 实现持续集成和持续部署（CI/CD）。

## Workflows 说明

### 1. CI Workflow (`.github/workflows/ci.yml`)

**触发条件**:
- 推送到 `main`, `develop`, `claude/*` 分支
- Pull Request 到 `main`, `develop` 分支

**执行内容**:
- **测试作业** (`test`):
  - 在 Python 3.8, 3.9, 3.10, 3.11 上运行测试
  - 安装 GDAL 等地理空间依赖
  - 运行 pytest 并生成覆盖率报告
  - 上传覆盖率到 Codecov（仅 Python 3.10）

- **代码质量检查** (`lint`):
  - flake8: 检查语法错误和代码风格
  - black: 检查代码格式
  - isort: 检查导入排序
  - mypy: 类型检查（可选，不影响构建）

- **文档构建** (`docs`):
  - 尝试构建 Sphinx 文档

### 2. Scheduled Tests (`.github/workflows/scheduled.yml`)

**触发条件**:
- 每周日 UTC 00:00 自动运行
- 可手动触发

**执行内容**:
- 在所有支持的 Python 版本上运行完整测试
- 生成详细的覆盖率报告（HTML格式）
- 检查依赖安全性（pip-audit）
- 检查过时的依赖

### 3. Release Workflow (`.github/workflows/release.yml`)

**触发条件**:
- 推送版本标签（如 `v1.0.0`）

**执行内容**:
- 构建 Python 包
- 生成变更日志
- 创建 GitHub Release
- （可选）发布到 PyPI

## 使用指南

### 运行本地测试

在推送代码前，建议先在本地运行测试：

```bash
# 运行所有测试
pytest tests/ -v

# 运行测试并生成覆盖率报告
pytest tests/ -v --cov=hydrosis --cov-report=html

# 代码质量检查
flake8 hydrosis/
black --check hydrosis/ tests/
isort --check-only hydrosis/ tests/
```

### 查看 CI 结果

1. 访问仓库的 **Actions** 标签页
2. 查看最近的 workflow 运行
3. 点击具体的运行查看详细日志

### 修复 CI 失败

如果 CI 失败：

1. **测试失败**:
   - 查看测试日志找到失败的测试
   - 在本地运行该测试：`pytest tests/test_xxx.py::test_function -v`
   - 修复问题后重新推送

2. **Linting 失败**:
   - 运行 `black hydrosis/ tests/` 自动格式化
   - 运行 `isort hydrosis/ tests/` 自动排序导入
   - 手动修复 flake8 报告的问题

3. **依赖安装失败**:
   - 检查 `requirements.txt` 是否有问题
   - 确保 GDAL 版本兼容

### 创建 Release

创建新版本的步骤：

```bash
# 1. 确保所有测试通过
pytest tests/

# 2. 更新版本号（如果使用 setup.py）
# 编辑 setup.py 或 __init__.py 中的版本号

# 3. 创建并推送标签
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# 4. GitHub Actions 会自动创建 Release
```

## 配置 Secrets

某些功能需要配置 GitHub Secrets：

### Codecov（可选）

如果使用 Codecov：

1. 访问 https://codecov.io/
2. 连接 GitHub 仓库
3. 获取 token
4. 在 GitHub 仓库设置中添加 Secret：`CODECOV_TOKEN`

### PyPI（可选）

如果要自动发布到 PyPI：

1. 在 PyPI 创建 API token
2. 在 GitHub 仓库设置中添加 Secret：`PYPI_API_TOKEN`
3. 在 `release.yml` 中设置 `if: true` 启用 PyPI 发布

## 徽章

添加状态徽章到 README.md：

```markdown
[![CI](https://github.com/leixiaohui-1974/HydroSIS/actions/workflows/ci.yml/badge.svg)](https://github.com/leixiaohui-1974/HydroSIS/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/leixiaohui-1974/HydroSIS/branch/main/graph/badge.svg)](https://codecov.io/gh/leixiaohui-1974/HydroSIS)
```

## 故障排除

### GDAL 安装问题

如果 GDAL 安装失败：

1. 检查系统 GDAL 版本：`gdal-config --version`
2. 确保 Python GDAL 版本匹配系统版本
3. 考虑使用 conda 环境（更可靠）

### 内存不足

对于大型测试，可能需要：

1. 减少并行测试进程
2. 使用 `pytest -n auto` 控制并行度
3. 增加 GitHub Actions runner 的内存（付费功能）

### 超时

如果测试超时：

1. 增加超时设置：`timeout-minutes: 30`
2. 分割测试套件
3. 优化慢速测试

## 最佳实践

1. **频繁提交**: 小步提交，快速反馈
2. **本地测试**: 推送前先在本地运行测试
3. **保持绿色**: 及时修复失败的 CI
4. **代码审查**: 合并前检查 CI 状态
5. **文档更新**: 代码变更同时更新文档

## 性能优化

### 缓存依赖

GitHub Actions 已配置 pip 缓存：

```yaml
- uses: actions/setup-python@v4
  with:
    cache: 'pip'
```

### 矩阵策略

使用矩阵策略并行测试多个 Python 版本：

```yaml
strategy:
  matrix:
    python-version: ['3.8', '3.9', '3.10', '3.11']
```

### 条件执行

某些步骤仅在特定条件下执行：

```yaml
- name: Upload coverage
  if: matrix.python-version == '3.10'
```

## 扩展配置

### 添加新的 Workflow

1. 在 `.github/workflows/` 创建新的 YAML 文件
2. 定义触发条件和作业
3. 测试 workflow
4. 添加到文档

### 自定义测试环境

如果需要特殊的测试环境：

```yaml
- name: Set up custom environment
  run: |
    # 安装特殊依赖
    # 配置环境变量
    # 准备测试数据
```

## 参考资料

- [GitHub Actions 文档](https://docs.github.com/en/actions)
- [pytest 文档](https://docs.pytest.org/)
- [Codecov 文档](https://docs.codecov.com/)
- [Python 打包指南](https://packaging.python.org/)

---

**维护者**: HydroSIS 开发团队
**最后更新**: 2025-01-24
