"""时间序列验证模块

提供水文时间序列的验证功能，包括：
- 连续性检查
- 趋势分析
- 异常值检测
- 物理合理性检查
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd

from .base import ValidationCriteria, ValidationResult


@dataclass
class TimeSeriesCriteria(ValidationCriteria):
    """时间序列验证标准"""
    # 数据完整性
    max_missing_ratio: float = 0.1
    max_consecutive_missing: int = 24

    # 数值范围
    min_value: float = 0.0
    max_value: float = 1e6

    # 变化率
    max_hourly_change_ratio: float = 0.5  # 最大小时变化率
    max_daily_change_ratio: float = 2.0   # 最大日变化率

    # 趋势检查
    allow_negative_trend: bool = True
    max_trend_slope: float = 1e6

    @classmethod
    def from_dict(cls, data: Dict) -> "TimeSeriesCriteria":
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


def validate_time_series(
    series: pd.Series,
    criteria: Optional[TimeSeriesCriteria] = None,
    series_name: str = "时间序列",
    step_name: str = "时间序列验证"
) -> ValidationResult:
    """验证单个时间序列

    Args:
        series: pandas Series
        criteria: 验证标准
        series_name: 序列名称
        step_name: 步骤名称

    Returns:
        验证结果
    """
    if criteria is None:
        criteria = TimeSeriesCriteria()

    result = ValidationResult(step_name=f"{step_name} - {series_name}")

    # 1. 缺失值检查
    missing_count = series.isna().sum()
    missing_ratio = missing_count / len(series)

    result.metrics['missing_count'] = int(missing_count)
    result.metrics['missing_ratio'] = missing_ratio

    if missing_ratio > criteria.max_missing_ratio:
        result.add_error(
            f"缺失率过高: {missing_ratio:.2%} > {criteria.max_missing_ratio:.2%}"
        )

    # 连续缺失
    if missing_count > 0:
        is_missing = series.isna().astype(int)
        consecutive_missing = is_missing.groupby(
            (is_missing != is_missing.shift()).cumsum()
        ).sum()
        max_consecutive = consecutive_missing.max()

        if max_consecutive > criteria.max_consecutive_missing:
            result.add_warning(
                f"连续缺失过多: {max_consecutive} > {criteria.max_consecutive_missing}"
            )

    # 2. 数值范围检查
    valid_series = series.dropna()
    if len(valid_series) > 0:
        min_val = valid_series.min()
        max_val = valid_series.max()

        result.metrics['min_value'] = float(min_val)
        result.metrics['max_value'] = float(max_val)
        result.metrics['mean_value'] = float(valid_series.mean())
        result.metrics['std_value'] = float(valid_series.std())

        if min_val < criteria.min_value:
            result.add_error(f"存在负值: {min_val}")

        if max_val > criteria.max_value:
            result.add_warning(f"存在异常高值: {max_val}")

    # 3. 变化率检查
    if len(valid_series) > 1:
        # 小时变化率
        hourly_change = valid_series.diff().abs()
        max_hourly_change = hourly_change.max()

        if max_val > 0:
            relative_change = max_hourly_change / max_val
            if relative_change > criteria.max_hourly_change_ratio:
                result.add_warning(
                    f"小时变化率过大: {relative_change:.2%} > "
                    f"{criteria.max_hourly_change_ratio:.2%}"
                )

    # 4. 单调性检查(简化的趋势检测)
    if len(valid_series) > 10:
        # 使用线性回归检测趋势
        x = np.arange(len(valid_series))
        y = valid_series.values
        slope = np.polyfit(x, y, 1)[0]

        result.metrics['trend_slope'] = float(slope)

        if not criteria.allow_negative_trend and slope < -abs(slope) * 0.1:
            result.add_warning(f"检测到负趋势: 斜率={slope:.4f}")

    return result


def validate_multiple_series(
    df: pd.DataFrame,
    criteria: Optional[TimeSeriesCriteria] = None,
    step_name: str = "多序列验证"
) -> Dict[str, ValidationResult]:
    """验证多个时间序列

    Args:
        df: DataFrame (时间×变量)
        criteria: 验证标准
        step_name: 步骤名称

    Returns:
        每个序列的验证结果字典
    """
    results = {}

    for col in df.columns:
        result = validate_time_series(
            df[col],
            criteria=criteria,
            series_name=str(col),
            step_name=step_name
        )
        results[str(col)] = result

    return results
