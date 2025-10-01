"""Standalone example verifying runoff and routing model behaviour on a synthetic flood event."""
from __future__ import annotations

from hydrosis.testing.flood_validation import generate_flood_validation_case


def _format_number(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    case = generate_flood_validation_case()

    print("Synthetic rainfall hyetograph (mm per step):")
    print("  " + ", ".join(_format_number(val, 1) for val in case.rainfall))

    print(f"\nCatchment area: {case.subbasin.area_km2:.1f} km^2")
    print(f"Total rainfall: {case.rainfall_total:.1f} mm (area-weighted volume {case.rainfall_volume:.1f})")

    print("\nReference discharge statistics (HYMOD + dynamic wave):")
    for key, value in case.observed_summary.items():
        if isinstance(value, float):
            print(f"  {key}: {_format_number(value)}")
        else:
            print(f"  {key}: {value}")

    print("\nScenario comparison (ordered by NSE ranking):")
    header = "    {name:<24} peak  time_to_peak  volume    NSE     RMSE    PBIAS"
    print(header.format(name="scenario"))
    print("    " + "-" * 66)
    for name in case.ranking:
        stats = case.hydro_stats[name]
        metrics = case.aggregated_metrics[name]
        peak = _format_number(stats["discharge_peak"])
        t_peak = f"{int(stats['discharge_time_to_peak']):>5d}"
        volume = _format_number(stats["discharge_volume"])
        nse = _format_number(metrics.get("nse", float("nan")))
        rmse = _format_number(metrics.get("rmse", float("nan")))
        pbias = f"{metrics.get('pbias', float('nan')):>7.2f}"
        print(f"    {name:<24} {peak:>6}  {t_peak}     {volume:>6}  {nse:>6}  {rmse:>6}  {pbias}")

    print("\nRanking by NSE (best -> worst):")
    print("  " + " > ".join(case.ranking))


if __name__ == "__main__":
    main()
