"""Unit tests for workflow stage helpers."""
from __future__ import annotations

import pathlib
from typing import Dict

import pandas as pd
import pytest
from shapely.geometry import Polygon

from hydrosis.model import Subbasin
from hydrosis.workflow.stages import (
    generate_precipitation_for_parameters,
    run_channel_diagnostics,
)
from examples.run_upper_truckee_project import (
    _basin_average_precipitation,
    _build_forcing,
)


def _build_base_series(length: int = 4) -> pd.Series:
    index = pd.date_range("2023-01-01", periods=length, freq="h")
    data = pd.Series(range(1, length + 1), index=index, dtype=float, name="precip")
    data.index.name = "Timestamp"
    return data


def test_generate_precipitation_with_synthetic_gauges() -> None:
    base_series = _build_base_series()
    parameter_geometries: Dict[str, Polygon] = {
        "P1": Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        "P2": Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
    }
    parameter_to_zone = {"P1": "H1", "P2": "H2"}
    parameter_areas = {"P1": 5.0, "P2": 7.0}
    subbasins = [
        Subbasin(id="H1", area_km2=5.0, downstream="H2"),
        Subbasin(id="H2", area_km2=7.0, downstream=None),
    ]

    result_a = generate_precipitation_for_parameters(
        base_precipitation=base_series,
        parameter_geometries=parameter_geometries,
        parameter_to_zone=parameter_to_zone,
        parameter_areas=parameter_areas,
        subbasins=subbasins,
        rainfall_options={"station_count": 3, "rng_seed": 11},
    )
    result_b = generate_precipitation_for_parameters(
        base_precipitation=base_series,
        parameter_geometries=parameter_geometries,
        parameter_to_zone=parameter_to_zone,
        parameter_areas=parameter_areas,
        subbasins=subbasins,
        rainfall_options={"station_count": 3, "rng_seed": 11},
    )

    assert list(result_a.parameter_series.columns) == ["P1", "P2"]
    assert list(result_a.subbasin_series.columns) == ["H1", "H2"]
    assert result_a.parameter_series.index.equals(base_series.index)
    assert result_a.subbasin_series.index.equals(base_series.index)
    assert result_a.rain_inputs is not None
    assert result_a.station_series.shape[1] == 3

    pd.testing.assert_frame_equal(result_a.station_series, result_b.station_series)
    pd.testing.assert_frame_equal(result_a.subbasin_series, result_b.subbasin_series)
    pd.testing.assert_frame_equal(result_a.parameter_series, result_b.parameter_series)
    pd.testing.assert_series_equal(
        result_a.subbasin_series["H1"],
        result_a.parameter_series["P1"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        result_a.subbasin_series["H2"],
        result_a.parameter_series["P2"],
        check_names=False,
    )


def test_generate_precipitation_via_interpolation() -> None:
    base_series = _build_base_series(length=3)
    station_index = base_series.index
    station_series = pd.DataFrame(
        {
            "S1": [10.0, 11.0, 12.0],
            "S2": [5.0, 6.0, 7.0],
        },
        index=station_index,
    )
    parameter_geometries: Dict[str, Polygon] = {
        "P1": Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        "P2": Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
    }
    parameter_to_zone = {"P1": "H1", "P2": "H2"}
    parameter_areas = {"P1": 3.0, "P2": 4.0}
    thiessen_polygons = {
        "S1": Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        "S2": Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
    }
    subbasins = [
        Subbasin(id="H1", area_km2=3.0, downstream="H2"),
        Subbasin(id="H2", area_km2=4.0, downstream=None),
    ]

    result = generate_precipitation_for_parameters(
        base_precipitation=base_series,
        parameter_geometries=parameter_geometries,
        parameter_to_zone=parameter_to_zone,
        parameter_areas=parameter_areas,
        subbasins=subbasins,
        station_series=station_series,
        thiessen_polygons=thiessen_polygons,
    )

    assert result.rain_inputs is None
    pd.testing.assert_series_equal(
        result.parameter_series["P1"],
        station_series["S1"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        result.parameter_series["P2"],
        station_series["S2"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        result.subbasin_series["H1"],
        station_series["S1"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        result.subbasin_series["H2"],
        station_series["S2"],
        check_names=False,
    )
    assert pytest.approx(1.0) == result.station_weights["P1"]["S1"]
    assert pytest.approx(1.0) == result.station_weights["P2"]["S2"]


def test_run_channel_diagnostics_invokes_dependencies(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parameter_dir = tmp_path / "parameters"
    parameter_dir.mkdir()
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    intermediate_dir = tmp_path / "intermediate"
    intermediate_dir.mkdir()
    local_dir = tmp_path / "baseline_local"
    local_dir.mkdir()
    precipitation_path = tmp_path / "rain.csv"
    precipitation_path.write_text("Timestamp,precip\n2023-01-01 00:00:00,1.0\n", encoding="utf-8")

    flows_path = intermediate_dir / "channel_flow_timeseries.csv"
    comparison_path = intermediate_dir / "channel_flow_comparison.png"
    coefficients_path = intermediate_dir / "zone_runoff_coefficients.csv"

    call_log: Dict[str, Dict[str, object]] = {}

    def fake_compute_channel_flows(
        parameter_dir: pathlib.Path,
        baseline_dir: pathlib.Path,
        intermediate_dir: pathlib.Path,
        local_dir: pathlib.Path | None = None,
    ) -> pathlib.Path:
        call_log["compute_channel_flows"] = {
            "parameter_dir": parameter_dir,
            "baseline_dir": baseline_dir,
            "intermediate_dir": intermediate_dir,
            "local_dir": local_dir,
        }
        flows_path.write_text("flows", encoding="utf-8")
        return flows_path

    def fake_compare_channel_flows(
        parameter_dir: pathlib.Path,
        baseline_dir: pathlib.Path,
        aggregated_dir: pathlib.Path,
        intermediate_dir: pathlib.Path,
        local_dir: pathlib.Path | None = None,
        zones=None,
        output_filename: str = "channel_flow_comparison.png",
    ) -> pathlib.Path:
        call_log["compare_channel_flows"] = {
            "parameter_dir": parameter_dir,
            "baseline_dir": baseline_dir,
            "aggregated_dir": aggregated_dir,
            "intermediate_dir": intermediate_dir,
            "local_dir": local_dir,
            "zones": zones,
            "output_filename": output_filename,
        }
        comparison_path.write_text("comparison", encoding="utf-8")
        return comparison_path

    def fake_compute_zone_runoff_coefficients(
        parameter_dir: pathlib.Path,
        aggregated_dir: pathlib.Path,
        local_dir: pathlib.Path,
        precipitation_path: pathlib.Path,
        output_path: pathlib.Path,
    ) -> pathlib.Path:
        call_log["compute_zone_runoff_coefficients"] = {
            "parameter_dir": parameter_dir,
            "aggregated_dir": aggregated_dir,
            "local_dir": local_dir,
            "precipitation_path": precipitation_path,
            "output_path": output_path,
        }
        output_path.write_text("coefficients", encoding="utf-8")
        return output_path

    monkeypatch.setattr(
        "hydrosis.analysis.channel_flow.compute_channel_flows",
        fake_compute_channel_flows,
    )
    monkeypatch.setattr(
        "hydrosis.analysis.channel_flow.compare_channel_flows",
        fake_compare_channel_flows,
    )
    monkeypatch.setattr(
        "hydrosis.analysis.runoff_coefficients.compute_zone_runoff_coefficients",
        fake_compute_zone_runoff_coefficients,
    )

    diagnostics = run_channel_diagnostics(
        parameter_dir=parameter_dir,
        baseline_dir=baseline_dir,
        intermediate_dir=intermediate_dir,
        local_dir=local_dir,
        precipitation_path=precipitation_path,
    )

    assert diagnostics.timeseries_path == flows_path
    assert diagnostics.comparison_path == comparison_path
    assert diagnostics.runoff_coefficients_path == coefficients_path

    assert call_log["compute_channel_flows"]["parameter_dir"] == parameter_dir
    assert call_log["compare_channel_flows"]["aggregated_dir"] == baseline_dir
    assert call_log["compute_zone_runoff_coefficients"]["precipitation_path"] == precipitation_path


def test_build_forcing_returns_complete_mapping() -> None:
    index = pd.date_range("2023-01-01", periods=3, freq="h")
    subbasin_series = pd.DataFrame(
        {
            "H1": [1.0, 2.0, 3.0],
            "H2": [0.5, 1.5, 2.5],
        },
        index=index,
    )
    subbasins = [
        Subbasin(id="H1", area_km2=1.0, downstream="H2"),
        Subbasin(id="H2", area_km2=2.0, downstream=None),
    ]

    forcing = _build_forcing(subbasin_series, subbasins)

    assert set(forcing) == {"H1", "H2"}
    assert forcing["H1"] == [1.0, 2.0, 3.0]
    assert forcing["H2"] == [0.5, 1.5, 2.5]


def test_build_forcing_missing_series_raises() -> None:
    index = pd.date_range("2023-01-01", periods=2, freq="h")
    subbasin_series = pd.DataFrame({"H1": [1.0, 2.0]}, index=index)
    subbasins = [
        Subbasin(id="H1", area_km2=1.0, downstream=None),
        Subbasin(id="H2", area_km2=2.0, downstream=None),
    ]

    with pytest.raises(KeyError):
        _build_forcing(subbasin_series, subbasins)


def test_basin_average_precipitation_matches_area_weights() -> None:
    index = pd.date_range("2023-01-01", periods=2, freq="h")
    subbasin_series = pd.DataFrame(
        {
            "H1": [10.0, 20.0],
            "H2": [5.0, 15.0],
        },
        index=index,
    )
    subbasins = [
        Subbasin(id="H1", area_km2=1.0, downstream=None),
        Subbasin(id="H2", area_km2=3.0, downstream=None),
    ]

    averaged = _basin_average_precipitation(subbasin_series, subbasins)

    expected = pd.Series(
        [6.25, 16.25],
        index=index,
        name="precipitation_mm_per_hr",
    )
    pd.testing.assert_series_equal(averaged, expected)
