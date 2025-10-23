"""Estimated runoff generator for parameter calibration and sensitivity analysis.

This module generates synthetic runoff time series from precipitation using the
runoff coefficient method with flood routing corrections (lag and attenuation).
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EstimatedRunoffGenerator:
    """Generate estimated runoff from precipitation using empirical methods.

    Uses:
    1. Runoff coefficient method for volume estimation
    2. Lag time for flood wave propagation delay
    3. Attenuation for peak reduction (routing effect)
    4. Baseflow for sustained low flow
    """

    def __init__(
        self,
        runoff_coefficient: float = 0.345,
        lag_hours: float = 2.0,
        attenuation_factor: float = 0.75,
        baseflow_ratio: float = 0.10,
        time_step_hours: float = 1.0
    ):
        """Initialize the estimated runoff generator.

        Args:
            runoff_coefficient: Ratio of runoff to precipitation (0-1), default 0.345
            lag_hours: Time delay from rainfall to runoff peak (hours)
            attenuation_factor: Peak reduction factor (0-1), higher = less attenuation
            baseflow_ratio: Baseflow as fraction of peak flow (0-1)
            time_step_hours: Time step in hours
        """
        if not 0 < runoff_coefficient <= 1.0:
            raise ValueError(f"Runoff coefficient must be in (0, 1], got {runoff_coefficient}")
        if lag_hours < 0:
            raise ValueError(f"Lag hours must be >= 0, got {lag_hours}")
        if not 0 < attenuation_factor <= 1.0:
            raise ValueError(f"Attenuation factor must be in (0, 1], got {attenuation_factor}")
        if not 0 <= baseflow_ratio < 1.0:
            raise ValueError(f"Baseflow ratio must be in [0, 1), got {baseflow_ratio}")

        self.runoff_coefficient = runoff_coefficient
        self.lag_hours = lag_hours
        self.attenuation_factor = attenuation_factor
        self.baseflow_ratio = baseflow_ratio
        self.time_step_hours = time_step_hours

        logger.info(f"EstimatedRunoffGenerator initialized: Rc={runoff_coefficient:.3f}, "
                   f"lag={lag_hours:.1f}h, attn={attenuation_factor:.2f}")

    def generate(
        self,
        precipitation_series: np.ndarray,
        area_km2: float,
        apply_lag: bool = True,
        apply_attenuation: bool = True,
        apply_baseflow: bool = True
    ) -> Tuple[np.ndarray, dict]:
        """Generate estimated runoff from precipitation.

        Args:
            precipitation_series: Precipitation time series (mm/hr)
            area_km2: Catchment area (km²)
            apply_lag: Whether to apply time lag
            apply_attenuation: Whether to apply peak attenuation
            apply_baseflow: Whether to add baseflow component

        Returns:
            Tuple of:
            - runoff_series: Estimated runoff (m³/s)
            - stats: Dictionary with generation statistics
        """
        if area_km2 <= 0:
            raise ValueError(f"Area must be positive, got {area_km2}")

        precip = np.asarray(precipitation_series)
        if len(precip) == 0:
            raise ValueError("Precipitation series is empty")

        # Step 1: Runoff coefficient method
        # Q = Rc × P × A / 3.6
        # where P is in mm/hr, A in km², Q in m³/s
        # 1 mm/hr × 1 km² = 1000 m³/hr = 1000/3600 m³/s ≈ 0.278 m³/s
        runoff_direct = self.runoff_coefficient * precip * area_km2 / 3.6

        logger.debug(f"Direct runoff: max={runoff_direct.max():.2f} m³/s")

        # Step 2: Apply time lag (flood wave propagation delay)
        if apply_lag and self.lag_hours > 0:
            runoff_lagged = self._apply_lag(runoff_direct)
            logger.debug(f"After lag: max={runoff_lagged.max():.2f} m³/s")
        else:
            runoff_lagged = runoff_direct

        # Step 3: Apply attenuation (peak reduction during routing)
        if apply_attenuation and self.attenuation_factor < 1.0:
            runoff_attenuated = self._apply_attenuation(runoff_lagged)
            logger.debug(f"After attenuation: max={runoff_attenuated.max():.2f} m³/s")
        else:
            runoff_attenuated = runoff_lagged

        # Step 4: Add baseflow component
        if apply_baseflow and self.baseflow_ratio > 0:
            runoff_total = self._add_baseflow(runoff_attenuated)
            logger.debug(f"After baseflow: max={runoff_total.max():.2f} m³/s, "
                        f"min={runoff_total.min():.2f} m³/s")
        else:
            runoff_total = runoff_attenuated

        # Calculate statistics
        stats = self._calculate_statistics(
            precip, runoff_total, area_km2,
            runoff_direct, runoff_lagged, runoff_attenuated
        )

        return runoff_total, stats

    def _apply_lag(self, runoff: np.ndarray) -> np.ndarray:
        """Apply time lag by shifting the series.

        Args:
            runoff: Input runoff series

        Returns:
            Lagged runoff series
        """
        lag_steps = int(np.round(self.lag_hours / self.time_step_hours))

        if lag_steps == 0:
            return runoff

        # Shift series by lag_steps, padding with zeros at the beginning
        lagged = np.zeros_like(runoff)
        lagged[lag_steps:] = runoff[:-lag_steps]

        return lagged

    def _apply_attenuation(self, runoff: np.ndarray) -> np.ndarray:
        """Apply peak attenuation using exponential smoothing.

        Simulates the peak reduction effect of channel routing.

        Args:
            runoff: Input runoff series

        Returns:
            Attenuated runoff series
        """
        # Exponential moving average
        # Q_smooth(t) = α × Q(t) + (1-α) × Q_smooth(t-1)
        alpha = self.attenuation_factor

        attenuated = np.zeros_like(runoff)
        attenuated[0] = runoff[0]

        for i in range(1, len(runoff)):
            attenuated[i] = alpha * runoff[i] + (1 - alpha) * attenuated[i-1]

        return attenuated

    def _add_baseflow(self, runoff: np.ndarray) -> np.ndarray:
        """Add baseflow component to surface runoff.

        Baseflow is modeled as a minimum sustained flow.

        Args:
            runoff: Surface runoff series

        Returns:
            Total runoff (surface + baseflow)
        """
        # Baseflow as a fraction of peak surface runoff
        peak_runoff = np.max(runoff)
        baseflow = self.baseflow_ratio * peak_runoff

        # Add constant baseflow
        total_runoff = runoff + baseflow

        return total_runoff

    def _calculate_statistics(
        self,
        precip: np.ndarray,
        runoff_final: np.ndarray,
        area_km2: float,
        runoff_direct: np.ndarray,
        runoff_lagged: np.ndarray,
        runoff_attenuated: np.ndarray
    ) -> dict:
        """Calculate generation statistics.

        Args:
            precip: Precipitation series (mm/hr)
            runoff_final: Final runoff series (m³/s)
            area_km2: Catchment area (km²)
            runoff_direct: Direct runoff before routing (m³/s)
            runoff_lagged: After lag (m³/s)
            runoff_attenuated: After attenuation (m³/s)

        Returns:
            Dictionary with statistics
        """
        # Total precipitation depth (mm)
        total_precip_depth = precip.sum() * self.time_step_hours

        # Total runoff volume (m³)
        total_runoff_volume = runoff_final.sum() * self.time_step_hours * 3600

        # Total runoff depth (mm)
        total_runoff_depth = (total_runoff_volume / (area_km2 * 1e6)) * 1000

        # Actual runoff coefficient
        actual_rc = total_runoff_depth / total_precip_depth if total_precip_depth > 0 else 0

        # Peak values
        peak_precip = precip.max()
        peak_runoff_direct = runoff_direct.max()
        peak_runoff_final = runoff_final.max()
        peak_reduction_pct = (1 - peak_runoff_final / peak_runoff_direct) * 100 if peak_runoff_direct > 0 else 0

        # Peak times
        peak_precip_time = np.argmax(precip) * self.time_step_hours
        peak_runoff_time = np.argmax(runoff_final) * self.time_step_hours
        actual_lag = peak_runoff_time - peak_precip_time

        # Specific discharge (m³/s/km²)
        specific_peak = peak_runoff_final / area_km2

        stats = {
            'area_km2': area_km2,
            'total_precip_mm': total_precip_depth,
            'total_runoff_mm': total_runoff_depth,
            'target_rc': self.runoff_coefficient,
            'actual_rc': actual_rc,
            'peak_precip_mmhr': peak_precip,
            'peak_runoff_direct_m3s': peak_runoff_direct,
            'peak_runoff_final_m3s': peak_runoff_final,
            'peak_reduction_pct': peak_reduction_pct,
            'peak_precip_time_hr': peak_precip_time,
            'peak_runoff_time_hr': peak_runoff_time,
            'target_lag_hr': self.lag_hours,
            'actual_lag_hr': actual_lag,
            'specific_peak_m3s_per_km2': specific_peak,
            'min_runoff_m3s': runoff_final.min(),
            'mean_runoff_m3s': runoff_final.mean(),
        }

        return stats

    def validate_parameters(self) -> list:
        """Validate generator parameters.

        Returns:
            List of validation warnings (empty if all good)
        """
        warnings = []

        # Runoff coefficient range check
        if self.runoff_coefficient < 0.2:
            warnings.append(f"Runoff coefficient {self.runoff_coefficient:.3f} is quite low (< 0.2)")
        elif self.runoff_coefficient > 0.6:
            warnings.append(f"Runoff coefficient {self.runoff_coefficient:.3f} is quite high (> 0.6)")

        # Lag time reasonableness
        if self.lag_hours > 24:
            warnings.append(f"Lag time {self.lag_hours:.1f} hours seems very long (> 24 hr)")

        # Attenuation factor
        if self.attenuation_factor < 0.5:
            warnings.append(f"Attenuation factor {self.attenuation_factor:.2f} is very low (strong smoothing)")

        # Baseflow ratio
        if self.baseflow_ratio > 0.3:
            warnings.append(f"Baseflow ratio {self.baseflow_ratio:.2f} is quite high (> 0.3)")

        return warnings

    def __repr__(self):
        return (f"EstimatedRunoffGenerator(Rc={self.runoff_coefficient:.3f}, "
                f"lag={self.lag_hours:.1f}h, attn={self.attenuation_factor:.2f}, "
                f"baseflow={self.baseflow_ratio:.2f})")


__all__ = ['EstimatedRunoffGenerator']
