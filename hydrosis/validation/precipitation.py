"""降雨数据验证模块

提供降雨数据质量检查功能，包括：
- 空间一致性检查
- 时间连续性检查
- 数值合理性检查
- 站点间相关性检查
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd

from .base import ValidationCriteria, ValidationResult, BaseValidator


@dataclass
class PrecipitationCriteria(ValidationCriteria):
    """降雨验证标准"""
    # 数值范围
    min_value: float = 0.0  # mm/h
    max_value: float = 100.0  # mm/h
    max_daily_value: float = 500.0  # mm/day

    # 空间一致性
    max_spatial_cv: float = 0.5  # 最大空间变异系数
    min_spatial_correlation: float = 0.3  # 最小站点间相关系数

    # 时间连续性
    max_missing_ratio: float = 0.1  # 最大缺测率
    max_consecutive_zeros: int = 24  # 最大连续零值小时数

    # 质量等级阈值
    excellent_cv: float = 0.2
    good_cv: float = 0.35
    fair_cv: float = 0.5

    @classmethod
    def from_dict(cls, data: Dict) -> "PrecipitationCriteria":
        """从字典创建"""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


def validate_precipitation_data(
    precipitation_df: pd.DataFrame,
    criteria: Optional[PrecipitationCriteria] = None,
    step_name: str = "降雨数据验证"
) -> ValidationResult:
    """验证降雨数据质量

    Args:
        precipitation_df: 降雨数据DataFrame (时间×站点/子流域)
        criteria: 验证标准
        step_name: 步骤名称

    Returns:
        验证结果
    """
    if criteria is None:
        criteria = PrecipitationCriteria()

    result = ValidationResult(step_name=step_name)

    # 1. 数值范围检查
    min_val = precipitation_df.min().min()
    max_val = precipitation_df.max().max()

    if min_val < criteria.min_value:
        result.add_error(f"存在负降雨值: {min_val:.2f} mm/h")

    if max_val > criteria.max_value:
        result.add_warning(f"存在异常高降雨强度: {max_val:.2f} mm/h (阈值: {criteria.max_value})")

    result.metrics['min_value'] = min_val
    result.metrics['max_value'] = max_val

    # 2. 累积降雨检查
    total_precip = precipitation_df.sum(axis=0)
    mean_total = total_precip.mean()
    std_total = total_precip.std()

    if max_val > criteria.max_daily_value:
        result.add_warning(f"累积降雨过高: {max_val:.2f} mm (阈值: {criteria.max_daily_value})")

    result.metrics['mean_total_precip'] = mean_total
    result.metrics['std_total_precip'] = std_total

    # 3. 空间一致性检查
    spatial_cv = std_total / mean_total if mean_total > 0 else 0
    result.metrics['spatial_cv'] = spatial_cv

    if spatial_cv > criteria.max_spatial_cv:
        result.add_error(
            f"空间变异系数过大: {spatial_cv:.4f} (阈值: {criteria.max_spatial_cv}), "
            f"降雨分布极不均匀"
        )
    elif spatial_cv > criteria.good_cv:
        result.add_warning(f"空间变异系数偏高: {spatial_cv:.4f}")

    # 空间变异等级
    if spatial_cv <= criteria.excellent_cv:
        result.metrics['spatial_quality'] = "优秀"
    elif spatial_cv <= criteria.good_cv:
        result.metrics['spatial_quality'] = "良好"
    elif spatial_cv <= criteria.fair_cv:
        result.metrics['spatial_quality'] = "一般"
    else:
        result.metrics['spatial_quality'] = "差"

    # 4. 缺测检查
    missing_ratio = precipitation_df.isna().sum().sum() / (len(precipitation_df) * len(precipitation_df.columns))
    result.metrics['missing_ratio'] = missing_ratio

    if missing_ratio > criteria.max_missing_ratio:
        result.add_warning(f"缺测率过高: {missing_ratio*100:.2f}% (阈值: {criteria.max_missing_ratio*100}%)")

    # 5. 连续零值检查
    for col in precipitation_df.columns:
        series = precipitation_df[col]
        # 找到连续零值
        is_zero = (series == 0).astype(int)
        consecutive_zeros = is_zero.groupby((is_zero != is_zero.shift()).cumsum()).sum()
        max_consecutive = consecutive_zeros.max()

        if max_consecutive > criteria.max_consecutive_zeros:
            result.add_warning(
                f"站点/子流域 {col}: 连续{max_consecutive}小时无降雨 "
                f"(阈值: {criteria.max_consecutive_zeros}小时)"
            )

    # 6. 站点间相关性 (如果站点数>=2)
    if len(precipitation_df.columns) >= 2:
        corr_matrix = precipitation_df.corr()
        # 提取下三角(不包括对角线)
        mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
        correlations = corr_matrix.where(mask).stack().values

        mean_corr = np.mean(correlations)
        min_corr = np.min(correlations)

        result.metrics['mean_correlation'] = mean_corr
        result.metrics['min_correlation'] = min_corr

        if min_corr < criteria.min_spatial_correlation:
            result.add_warning(
                f"存在低相关站点: 最小相关系数{min_corr:.3f} "
                f"(阈值: {criteria.min_spatial_correlation})"
            )

    return result


def identify_precipitation_outliers(
    precipitation_df: pd.DataFrame,
    method: str = "zscore",
    threshold: float = 3.0
) -> Dict[str, List]:
    """识别降雨异常值

    Args:
        precipitation_df: 降雨数据
        method: 方法 ('zscore' 或 'iqr')
        threshold: 阈值 (zscore方法用3.0, iqr方法用1.5)

    Returns:
        异常值字典 {站点: [(时间索引, 值), ...]}
    """
    outliers = {}

    for col in precipitation_df.columns:
        series = precipitation_df[col]
        col_outliers = []

        if method == "zscore":
            # Z-score方法
            mean = series.mean()
            std = series.std()
            if std > 0:
                z_scores = np.abs((series - mean) / std)
                outlier_mask = z_scores > threshold
                outlier_indices = np.where(outlier_mask)[0]

                for idx in outlier_indices:
                    col_outliers.append((precipitation_df.index[idx], series.iloc[idx]))

        elif method == "iqr":
            # IQR方法
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            outlier_mask = (series < lower_bound) | (series > upper_bound)
            outlier_indices = np.where(outlier_mask)[0]

            for idx in outlier_indices:
                col_outliers.append((precipitation_df.index[idx], series.iloc[idx]))

        if col_outliers:
            outliers[col] = col_outliers

    return outliers


def suggest_precipitation_fixes(
    validation_result: ValidationResult,
    precipitation_df: pd.DataFrame
) -> Dict[str, str]:
    """基于验证结果建议修复措施

    Args:
        validation_result: 验证结果
        precipitation_df: 降雨数据

    Returns:
        修复建议字典
    """
    suggestions = {}

    spatial_cv = validation_result.metrics.get('spatial_cv', 0)
    spatial_quality = validation_result.metrics.get('spatial_quality', 'unknown')

    if spatial_cv > 0.5:
        suggestions['spatial_uniformity'] = (
            f"空间变异系数过大({spatial_cv:.4f}), 建议措施:\n"
            f"  1. 检查雨量站分布是否合理\n"
            f"  2. 使用更高级的空间插值方法(IDW, Kriging)\n"
            f"  3. 增加雨量站密度\n"
            f"  4. 使用雷达降雨数据辅助"
        )

    missing_ratio = validation_result.metrics.get('missing_ratio', 0)
    if missing_ratio > 0.1:
        suggestions['missing_data'] = (
            f"缺测率过高({missing_ratio*100:.1f}%), 建议措施:\n"
            f"  1. 使用临近站点数据插补\n"
            f"  2. 使用历史数据回归填补\n"
            f"  3. 使用网格化降雨产品"
        )

    min_corr = validation_result.metrics.get('min_correlation')
    if min_corr is not None and min_corr < 0.3:
        suggestions['low_correlation'] = (
            f"站点间相关性低({min_corr:.3f}), 建议措施:\n"
            f"  1. 检查数据质量\n"
            f"  2. 考虑流域地形影响\n"
            f"  3. 分区域进行插值"
        )

    return suggestions
