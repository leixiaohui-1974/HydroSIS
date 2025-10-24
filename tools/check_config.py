#!/usr/bin/env python3
"""配置文件检查工具

检查workflow_config.yaml的完整性和合理性。

用法:
    python tools/check_config.py
    python tools/check_config.py --config config/workflow_config.yaml
    python tools/check_config.py --strict  # 严格模式
"""
import argparse
from pathlib import Path
import sys
from typing import List, Dict, Any

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ConfigChecker:
    """配置检查器"""

    def __init__(self, strict: bool = False):
        """初始化

        Args:
            strict: 严格模式（警告也算失败）
        """
        self.strict = strict
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []

    def check_config(self, config_path: Path) -> bool:
        """检查配置文件

        Args:
            config_path: 配置文件路径

        Returns:
            是否通过检查
        """
        print("=" * 80)
        print("配置文件检查工具")
        print("=" * 80)
        print(f"配置文件: {config_path}")
        print(f"严格模式: {'是' if self.strict else '否'}")
        print()

        # 1. 文件存在性检查
        if not config_path.exists():
            self.errors.append(f"配置文件不存在: {config_path}")
            return False

        # 2. 加载配置
        try:
            import yaml
            with open(config_path, encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            self.errors.append(f"配置文件加载失败: {e}")
            return False

        self.info.append(f"✓ 配置文件加载成功")

        # 3. 检查必需字段
        self._check_required_fields(config)

        # 4. 检查HBV参数
        self._check_hbv_parameters(config)

        # 5. 检查雨量站配置
        self._check_rain_gauge_config(config)

        # 6. 检查并行配置
        self._check_parallel_config(config)

        # 7. 检查目录配置
        self._check_directories(config)

        # 8. 输出结果
        self._print_results()

        # 9. 返回检查结果
        is_valid = len(self.errors) == 0
        if self.strict:
            is_valid = is_valid and len(self.warnings) == 0

        return is_valid

    def _check_required_fields(self, config: Dict[str, Any]) -> None:
        """检查必需字段"""
        print("⚙ 检查必需字段...")

        required_fields = [
            'project_name',
            'directories',
            'hbv_parameters',
        ]

        for field in required_fields:
            if field not in config:
                self.errors.append(f"缺少必需字段: {field}")
            else:
                self.info.append(f"✓ 找到字段: {field}")

    def _check_hbv_parameters(self, config: Dict[str, Any]) -> None:
        """检查HBV参数"""
        print("\n⚙ 检查HBV参数...")

        if 'hbv_parameters' not in config:
            return

        hbv_params = config['hbv_parameters']

        # 必需参数
        required_params = ['FC', 'BETA', 'LP', 'K0', 'K1', 'K2', 'PERC', 'UZL']

        for param in required_params:
            if param not in hbv_params:
                self.errors.append(f"HBV参数缺失: {param}")
            else:
                value = hbv_params[param]
                self.info.append(f"✓ {param} = {value}")

        # 参数范围检查
        param_ranges = {
            'FC': (10, 1000),      # Field capacity
            'BETA': (1.0, 5.0),    # Shape coefficient
            'LP': (0.0, 1.0),      # ET threshold
            'K0': (0.0, 1.0),      # Recession coef
            'K1': (0.0, 1.0),
            'K2': (0.0, 0.1),
            'PERC': (0.0, 10.0),   # Percolation
            'UZL': (0.0, 200.0),   # Upper zone threshold
        }

        for param, (min_val, max_val) in param_ranges.items():
            if param in hbv_params:
                value = hbv_params[param]
                if not (min_val <= value <= max_val):
                    self.warnings.append(
                        f"HBV参数 {param}={value} 超出建议范围 [{min_val}, {max_val}]"
                    )

    def _check_rain_gauge_config(self, config: Dict[str, Any]) -> None:
        """检查雨量站配置"""
        print("\n⚙ 检查雨量站配置...")

        if 'rain_gauge' not in config:
            self.warnings.append("缺少rain_gauge配置节（可选）")
            return

        rg_config = config['rain_gauge']

        # 建议字段
        recommended_fields = {
            'target_density': (0.001, 0.1),
            'min_distance_m': (100, 10000),
            'random_seed': (0, 100000),
        }

        for field, (min_val, max_val) in recommended_fields.items():
            if field not in rg_config:
                self.warnings.append(f"雨量站配置缺少建议字段: {field}")
            else:
                value = rg_config[field]
                self.info.append(f"✓ {field} = {value}")

                if isinstance(value, (int, float)):
                    if not (min_val <= value <= max_val):
                        self.warnings.append(
                            f"雨量站参数 {field}={value} 超出建议范围 [{min_val}, {max_val}]"
                        )

        # 密度等级检查
        if 'target_density' in rg_config:
            density = rg_config['target_density']
            if density < 0.002:
                self.warnings.append(f"雨量站密度过低: {density} < 0.002 (较差)")
            elif density >= 0.02:
                self.info.append(f"雨量站密度优秀: {density} >= 0.02")
            elif density >= 0.01:
                self.info.append(f"雨量站密度良好: {density} >= 0.01")

    def _check_parallel_config(self, config: Dict[str, Any]) -> None:
        """检查并行配置"""
        print("\n⚙ 检查并行配置...")

        if 'parallel' not in config:
            self.info.append("未配置parallel（将使用默认值）")
            return

        parallel_config = config['parallel']

        # 检查max_workers
        if 'max_workers' in parallel_config:
            max_workers = parallel_config['max_workers']
            self.info.append(f"✓ max_workers = {max_workers}")

            import multiprocessing
            cpu_count = multiprocessing.cpu_count()

            if max_workers > cpu_count:
                self.warnings.append(
                    f"max_workers ({max_workers}) > CPU核心数 ({cpu_count})"
                )
            elif max_workers < 1:
                self.errors.append(f"max_workers 必须 >= 1, 当前值: {max_workers}")

    def _check_directories(self, config: Dict[str, Any]) -> None:
        """检查目录配置"""
        print("\n⚙ 检查目录配置...")

        if 'directories' not in config:
            return

        dirs = config['directories']

        # 检查base_results
        if 'base_results' in dirs:
            base_results = Path(dirs['base_results'])
            self.info.append(f"✓ base_results = {base_results}")

            # 展开环境变量
            if '${project_name}' in str(base_results):
                if 'project_name' in config:
                    expanded = str(base_results).replace(
                        '${project_name}', config['project_name']
                    )
                    self.info.append(f"  展开为: {expanded}")
                else:
                    self.errors.append("base_results使用${project_name}但未定义project_name")

    def _print_results(self) -> None:
        """打印检查结果"""
        print("\n" + "=" * 80)
        print("检查结果汇总")
        print("=" * 80)

        if self.info:
            print(f"\n✓ 信息 ({len(self.info)}条):")
            for msg in self.info:
                print(f"  {msg}")

        if self.warnings:
            print(f"\n⚠ 警告 ({len(self.warnings)}条):")
            for msg in self.warnings:
                print(f"  {msg}")

        if self.errors:
            print(f"\n✗ 错误 ({len(self.errors)}条):")
            for msg in self.errors:
                print(f"  {msg}")

        print("\n" + "=" * 80)

        if len(self.errors) == 0:
            if len(self.warnings) == 0:
                print("✅ 配置检查通过（无警告无错误）")
            else:
                if self.strict:
                    print("⚠️ 配置检查失败（严格模式，存在警告）")
                else:
                    print(f"✅ 配置检查通过（{len(self.warnings)}个警告）")
        else:
            print(f"❌ 配置检查失败（{len(self.errors)}个错误）")

        print("=" * 80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='配置文件检查工具')
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('config/workflow_config.yaml'),
        help='配置文件路径'
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='严格模式（警告也算失败）'
    )

    args = parser.parse_args()

    # 创建检查器
    checker = ConfigChecker(strict=args.strict)

    # 执行检查
    is_valid = checker.check_config(args.config)

    # 返回退出码
    sys.exit(0 if is_valid else 1)


if __name__ == '__main__':
    main()
