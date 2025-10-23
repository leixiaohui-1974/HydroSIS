"""
水量平衡分析工具

提供水文模型水量平衡验证和诊断功能。
"""
from typing import Sequence, Dict
from dataclasses import dataclass
import math


@dataclass
class WaterBalanceResult:
    """水量平衡分析结果"""

    # 输入
    total_precipitation_mm: float
    basin_area_km2: float
    timestep_hours: float

    # 输出
    total_runoff_mm: float
    total_runoff_volume_m3: float

    # 平衡
    runoff_coefficient: float
    storage_change_mm: float

    # 评估
    is_balanced: bool
    balance_quality: str  # "good", "acceptable", "poor", "critical"
    warnings: list[str]

    def __str__(self) -> str:
        """格式化输出"""
        lines = [
            "=" * 60,
            "Water Balance Analysis Result",
            "=" * 60,
            f"Basin Area:        {self.basin_area_km2:>10.2f} km²",
            f"Timestep:          {self.timestep_hours:>10.2f} hours",
            "",
            "Input:",
            f"  Total Precip:    {self.total_precipitation_mm:>10.2f} mm",
            "",
            "Output:",
            f"  Total Runoff:    {self.total_runoff_mm:>10.2f} mm",
            f"  Runoff Volume:   {self.total_runoff_volume_m3:>10.2f} m³",
            "",
            "Balance:",
            f"  Runoff Coeff:    {self.runoff_coefficient:>10.4f}",
            f"  Storage Change:  {self.storage_change_mm:>10.2f} mm",
            f"  Balance Quality: {self.balance_quality}",
            "",
        ]

        if self.warnings:
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")
        else:
            lines.append("No warnings - balance looks good!")

        lines.append("=" * 60)
        return "\n".join(lines)


def calculate_water_balance(
    precipitation_mm: Sequence[float],
    runoff_m3s: Sequence[float],
    basin_area_km2: float,
    timestep_hours: float = 1.0,
) -> WaterBalanceResult:
    """
    计算水量平衡

    Parameters
    ----------
    precipitation_mm : Sequence[float]
        降雨序列，单位: mm
    runoff_m3s : Sequence[float]
        径流序列，单位: m³/s
    basin_area_km2 : float
        流域面积，单位: km²
    timestep_hours : float, optional
        时间步长，单位: 小时，默认1.0

    Returns
    -------
    WaterBalanceResult
        水量平衡分析结果

    Examples
    --------
    >>> precip = [10.0, 8.0, 5.0, 2.0]  # mm
    >>> runoff = [50.0, 60.0, 45.0, 30.0]  # m³/s
    >>> result = calculate_water_balance(precip, runoff, basin_area_km2=100.0)
    >>> print(result.runoff_coefficient)
    0.456
    """
    # 验证输入
    if len(precipitation_mm) != len(runoff_m3s):
        raise ValueError("Precipitation and runoff series must have same length")

    if basin_area_km2 <= 0:
        raise ValueError("Basin area must be positive")

    # 计算总降雨量
    total_precip_mm = sum(precipitation_mm) * timestep_hours
    total_precip_volume_m3 = total_precip_mm * basin_area_km2 * 1000  # m³

    # 计算总径流量
    total_runoff_volume_m3 = sum(runoff_m3s) * timestep_hours * 3600  # m³
    total_runoff_mm = total_runoff_volume_m3 / (basin_area_km2 * 1000)  # mm

    # 径流系数
    runoff_coefficient = total_runoff_mm / total_precip_mm if total_precip_mm > 0 else 0.0

    # 蓄水量变化
    storage_change_mm = total_precip_mm - total_runoff_mm

    # 评估水量平衡质量
    warnings = []
    is_balanced = True

    # 检查径流系数
    if runoff_coefficient > 1.0:
        warnings.append(f"Runoff coefficient > 1.0 ({runoff_coefficient:.3f}) - possible unit conversion error")
        is_balanced = False
        balance_quality = "critical"
    elif runoff_coefficient > 0.9:
        warnings.append(f"Very high runoff coefficient ({runoff_coefficient:.3f}) - check for saturated soil or urban area")
        balance_quality = "poor"
    elif runoff_coefficient < 0.1:
        warnings.append(f"Very low runoff coefficient ({runoff_coefficient:.3f}) - check for high infiltration or ET")
        balance_quality = "poor"
    elif 0.3 <= runoff_coefficient <= 0.7:
        balance_quality = "good"
    else:
        balance_quality = "acceptable"

    # 检查蓄水量变化
    if abs(storage_change_mm) > total_precip_mm * 0.5:
        warnings.append(f"Large storage change ({storage_change_mm:.1f} mm) relative to precipitation")

    return WaterBalanceResult(
        total_precipitation_mm=total_precip_mm,
        basin_area_km2=basin_area_km2,
        timestep_hours=timestep_hours,
        total_runoff_mm=total_runoff_mm,
        total_runoff_volume_m3=total_runoff_volume_m3,
        runoff_coefficient=runoff_coefficient,
        storage_change_mm=storage_change_mm,
        is_balanced=is_balanced,
        balance_quality=balance_quality,
        warnings=warnings,
    )


def compare_water_balance(
    balance_results: Dict[str, WaterBalanceResult],
) -> str:
    """
    对比多个模型的水量平衡

    Parameters
    ----------
    balance_results : Dict[str, WaterBalanceResult]
        模型名称到水量平衡结果的映射

    Returns
    -------
    str
        对比报告

    Examples
    --------
    >>> results = {
    ...     "Observed": calc_balance(precip, obs_runoff, area),
    ...     "HBV": calc_balance(precip, hbv_runoff, area),
    ... }
    >>> report = compare_water_balance(results)
    >>> print(report)
    """
    lines = [
        "=" * 80,
        "Water Balance Comparison",
        "=" * 80,
    ]

    if not balance_results:
        lines.append("No results to compare")
        return "\n".join(lines)

    # 表头
    lines.append(f"\n{'Model':<20s} {'Runoff (mm)':<15s} {'Coeff':<10s} {'ΔS (mm)':<12s} {'Quality':<15s}")
    lines.append("-" * 80)

    # 每个模型的结果
    for model_name, result in balance_results.items():
        lines.append(
            f"{model_name:<20s} "
            f"{result.total_runoff_mm:<15.2f} "
            f"{result.runoff_coefficient:<10.4f} "
            f"{result.storage_change_mm:<12.2f} "
            f"{result.balance_quality:<15s}"
        )

    # 对比分析
    if len(balance_results) >= 2:
        lines.append("\n" + "=" * 80)
        lines.append("Comparison Analysis")
        lines.append("=" * 80)

        coeffs = {name: r.runoff_coefficient for name, r in balance_results.items()}
        max_name = max(coeffs, key=coeffs.get)
        min_name = min(coeffs, key=coeffs.get)

        lines.append(f"\nHighest runoff coefficient: {max_name} ({coeffs[max_name]:.4f})")
        lines.append(f"Lowest runoff coefficient:  {min_name} ({coeffs[min_name]:.4f})")
        lines.append(f"Coefficient range:          {coeffs[max_name] - coeffs[min_name]:.4f}")

        # 一致性检查
        coeff_range = coeffs[max_name] - coeffs[min_name]
        if coeff_range < 0.1:
            lines.append("\n✓ Models show good agreement in runoff coefficient (Δ < 0.1)")
        elif coeff_range < 0.2:
            lines.append("\n⚠ Models show moderate differences in runoff coefficient (0.1 < Δ < 0.2)")
        else:
            lines.append("\n✗ Models show large differences in runoff coefficient (Δ > 0.2)")
            lines.append("  → Check model configurations and parameter settings")

    lines.append("=" * 80)
    return "\n".join(lines)


def precip_mmh_to_m3s(
    precip_mmh: float | Sequence[float],
    area_km2: float,
) -> float | list[float]:
    """
    转换降雨从 mm/h 到 m³/s

    Parameters
    ----------
    precip_mmh : float or Sequence[float]
        降雨强度，单位: mm/h
    area_km2 : float
        流域面积，单位: km²

    Returns
    -------
    float or list[float]
        流量，单位: m³/s

    Examples
    --------
    >>> flow = precip_mmh_to_m3s(10.0, 100.0)  # 10 mm/h over 100 km²
    >>> print(f"{flow:.2f} m³/s")
    277.78 m³/s

    Notes
    -----
    转换公式:
        1 mm/h over 1 km² = 1000 m³/h = 1000/3600 m³/s ≈ 0.2778 m³/s
    """
    conversion_factor = area_km2 * 1000.0 / 3600.0  # km² * (m³/km²/mm) / (s/h)

    if isinstance(precip_mmh, (int, float)):
        return precip_mmh * conversion_factor
    else:
        return [p * conversion_factor for p in precip_mmh]


def runoff_m3s_to_mm(
    runoff_m3s: float | Sequence[float],
    area_km2: float,
    timestep_hours: float = 1.0,
) -> float | list[float]:
    """
    转换径流从 m³/s 到 mm

    Parameters
    ----------
    runoff_m3s : float or Sequence[float]
        流量，单位: m³/s
    area_km2 : float
        流域面积，单位: km²
    timestep_hours : float, optional
        时间步长，单位: 小时，默认1.0

    Returns
    -------
    float or list[float]
        径流深度，单位: mm

    Examples
    --------
    >>> depth = runoff_m3s_to_mm(277.78, 100.0, timestep_hours=1.0)
    >>> print(f"{depth:.2f} mm")
    10.00 mm
    """
    # m³/s * 3600 s/h * timestep_h / (km² * 1000 m³/km²/mm) = mm
    conversion_factor = 3600.0 * timestep_hours / (area_km2 * 1000.0)

    if isinstance(runoff_m3s, (int, float)):
        return runoff_m3s * conversion_factor
    else:
        return [q * conversion_factor for q in runoff_m3s]


__all__ = [
    "WaterBalanceResult",
    "calculate_water_balance",
    "compare_water_balance",
    "precip_mmh_to_m3s",
    "runoff_m3s_to_mm",
]
