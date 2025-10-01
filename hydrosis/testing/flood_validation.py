"""Reusable synthetic flood validation scenario for runoff and routing models."""
from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from typing import Dict, List

from hydrosis.evaluation import ModelComparator, SimulationEvaluator
from hydrosis.model import Subbasin
from hydrosis.runoff.hymod import HYMODRunoff
from hydrosis.runoff.scs_curve_number import SCSCurveNumber
from hydrosis.runoff.xinanjiang import XinAnJiangRunoff
from hydrosis.routing.dynamic_wave import DynamicWaveRouting
from hydrosis.routing.lag import LagRouting
from hydrosis.routing.muskingum import MuskingumRouting


@dataclass(frozen=True)
class FloodValidationResult:
    """Container describing the synthetic flood validation experiment."""

    rainfall: List[float]
    subbasin: Subbasin
    observations: Dict[str, List[float]]
    candidate_runoff: Dict[str, List[float]]
    candidate_discharge: Dict[str, List[float]]
    hydro_stats: Dict[str, Dict[str, float]]
    aggregated_metrics: Dict[str, Dict[str, float]]
    ranking: List[str]
    observed_summary: Dict[str, float]
    rainfall_total: float
    rainfall_volume: float


def generate_flood_validation_case() -> FloodValidationResult:
    """Simulate a full flood hydrograph across multiple runoff-routing pairings."""

    rainfall = [
        0.0, 0.0, 0.0,
        5.0, 12.0, 25.0, 40.0, 65.0, 90.0, 110.0,
        95.0, 80.0, 60.0, 40.0, 25.0, 15.0,
        8.0, 5.0, 3.0, 1.0,
        0.0, 0.0,
    ]
    subbasin = Subbasin(id="B1", area_km2=42.0, downstream=None)

    rainfall_total = fsum(rainfall)
    rainfall_volume = rainfall_total * subbasin.area_km2

    # Reference runoff and routing combination used as pseudo-observations.
    reference_runoff = HYMODRunoff(
        {
            "max_storage": 125.0,
            "beta": 1.05,
            "quickflow_ratio": 0.62,
            "quick_k": 0.42,
            "slow_k": 0.06,
            "num_quick_reservoirs": 3,
        }
    ).simulate(subbasin, rainfall)

    dynamic_wave_params = {
        "time_step": 1.0,
        "reach_length": 10.0,
        "wave_celerity": 1.3,
        "diffusivity": 0.08,
    }
    reference_discharge = DynamicWaveRouting(dynamic_wave_params).route(
        subbasin, reference_runoff
    )
    observations = {"B1": list(reference_discharge)}

    candidate_runoff: Dict[str, List[float]] = {}
    candidate_discharge: Dict[str, List[float]] = {}
    hydro_stats: Dict[str, Dict[str, float]] = {}

    def _record(name: str, runoff: List[float], discharge: List[float]) -> None:
        runoff_series = list(runoff)
        discharge_series = list(discharge)
        candidate_runoff[name] = runoff_series
        candidate_discharge[name] = discharge_series
        if discharge_series:
            discharge_peak = max(discharge_series)
            time_to_peak = discharge_series.index(discharge_peak)
        else:
            discharge_peak = 0.0
            time_to_peak = 0
        hydro_stats[name] = {
            "runoff_peak": max(runoff_series) if runoff_series else 0.0,
            "runoff_volume": fsum(runoff_series),
            "discharge_peak": discharge_peak,
            "discharge_time_to_peak": time_to_peak,
            "discharge_volume": fsum(discharge_series),
        }

    _record("reference_hymod_dynamic", reference_runoff, reference_discharge)

    # Correct runoff with slower Muskingum routing.
    muskingum_discharge = MuskingumRouting(
        {"travel_time": 8.0, "weighting_factor": 0.3, "time_step": 1.0}
    ).route(subbasin, reference_runoff)
    _record("hymod_muskingum", reference_runoff, muskingum_discharge)

    # Alternate runoff representations paired with the same dynamic wave routing.
    scs_runoff = SCSCurveNumber(
        {"curve_number": 78, "initial_abstraction_ratio": 0.18}
    ).simulate(subbasin, rainfall)
    scs_dynamic = DynamicWaveRouting(dynamic_wave_params).route(subbasin, scs_runoff)
    _record("scs_dynamic", scs_runoff, scs_dynamic)

    xin_an_runoff = XinAnJiangRunoff(
        {"wm": 200.0, "b": 0.5, "imp": 0.04, "recession": 0.5}
    ).simulate(subbasin, rainfall)
    xin_an_dynamic = DynamicWaveRouting(dynamic_wave_params).route(
        subbasin, xin_an_runoff
    )
    _record("xinan_dynamic", xin_an_runoff, xin_an_dynamic)

    # Fast translation routing to highlight timing sensitivity.
    scs_lag = LagRouting({"lag_steps": 2}).route(subbasin, scs_runoff)
    _record("scs_lag", scs_runoff, scs_lag)

    simulations = {name: {"B1": discharge} for name, discharge in candidate_discharge.items()}
    evaluator = SimulationEvaluator()
    comparator = ModelComparator(evaluator)
    scores = comparator.compare(simulations, observations)
    aggregated_metrics = {
        score.model_id: dict(score.aggregated) for score in scores
    }
    ranking = [score.model_id for score in comparator.rank(scores, metric="nse")]

    observed_summary = {
        "peak": max(reference_discharge) if reference_discharge else 0.0,
        "time_to_peak": reference_discharge.index(max(reference_discharge)) if reference_discharge else 0,
        "volume": fsum(reference_discharge),
    }

    return FloodValidationResult(
        rainfall=list(rainfall),
        subbasin=subbasin,
        observations=observations,
        candidate_runoff=candidate_runoff,
        candidate_discharge=candidate_discharge,
        hydro_stats=hydro_stats,
        aggregated_metrics=aggregated_metrics,
        ranking=ranking,
        observed_summary=observed_summary,
        rainfall_total=rainfall_total,
        rainfall_volume=rainfall_volume,
    )
