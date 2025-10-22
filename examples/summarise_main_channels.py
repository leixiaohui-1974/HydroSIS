"""Summarise main channel subbasins and cross-sections for selected zones."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


def _parse_upstream(value: object, valid_segments: set[str]) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value)
    if not text:
        return []
    candidates = [item.strip() for item in text.split(";") if item.strip()]
    return [item for item in candidates if item in valid_segments]


def _resolve_main_path(zone_df: pd.DataFrame) -> List[str]:
    segments = set(zone_df["segment_id"])
    length_map = {row["segment_id"]: float(row["length_m"]) for _, row in zone_df.iterrows()}
    downstream_map = {
        row["segment_id"]: row["downstream_id"] if row["downstream_id"] in segments else None
        for _, row in zone_df.iterrows()
    }
    upstream_map = {
        row["segment_id"]: _parse_upstream(row.get("upstream_ids", ""), segments)
        for _, row in zone_df.iterrows()
    }

    outlet_candidates = [seg for seg, downstream in downstream_map.items() if downstream is None]
    if not outlet_candidates:
        # fall back to segments whose downstream leaves the zone prefix
        outlet_candidates = [row["segment_id"] for _, row in zone_df.iterrows() if not row["downstream_id"]]
    if not outlet_candidates:
        raise RuntimeError("Unable to identify outlet segment for zone.")
    outlet = outlet_candidates[0]

    memo: Dict[str, Tuple[float, List[str]]] = {}

    def longest_path(seg: str) -> Tuple[float, List[str]]:
        if seg in memo:
            return memo[seg]
        ups = upstream_map.get(seg, [])
        if not ups:
            result = (length_map.get(seg, 0.0), [seg])
        else:
            best_length = -1.0
            best_path: List[str] = []
            for upstream in ups:
                total, path = longest_path(upstream)
                if total > best_length:
                    best_length = total
                    best_path = path
            result = (best_length + length_map.get(seg, 0.0), best_path + [seg])
        memo[seg] = result
        return result

    _, path = longest_path(outlet)
    return path


def summarise_zones(
    zones: Iterable[str],
    channel_csv: Path,
    cross_section_dir: Path,
    output_dir: Path,
) -> Tuple[Path, Path, Dict[str, float]]:
    channel_df = pd.read_csv(channel_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []
    cross_section_frames: List[pd.DataFrame] = []
    zone_lengths: Dict[str, float] = {}

    for zone in zones:
        zone_df = channel_df[channel_df["zone_id"] == zone].copy()
        if zone_df.empty:
            continue
        path_segments = _resolve_main_path(zone_df)
        cumulative = 0.0
        for seg in path_segments:
            row = zone_df[zone_df["segment_id"] == seg].iloc[0]
            length = float(row["length_m"])
            start_offset = cumulative
            cumulative += length
            summary_rows.append(
                {
                    "zone_id": zone,
                    "segment_id": seg,
                    "subzone_id": row["subzone_id"],
                    "length_m": length,
                    "slope": float(row["slope"]),
                    "drop_m": float(row["drop_m"]),
                    "downstream_id": row.get("downstream_id"),
                    "upstream_ids": row.get("upstream_ids"),
                    "segment_start_m": start_offset,
                    "cumulative_length_m": cumulative,
                }
            )

            cross_path = cross_section_dir / f"{seg}_cross_sections.csv"
            if cross_path.exists():
                section_df = pd.read_csv(cross_path)
                if "zone_id" not in section_df.columns:
                    section_df.insert(0, "zone_id", zone)
                else:
                    section_df["zone_id"] = zone
                if "segment_id" not in section_df.columns:
                    section_df.insert(1, "segment_id", seg)
                else:
                    section_df["segment_id"] = seg
                if "station_m" in section_df.columns:
                    section_df["station_global_m"] = section_df["station_m"].astype(float) + start_offset
                else:
                    section_df["station_global_m"] = start_offset
                cross_section_frames.append(section_df)

        zone_lengths[zone] = cumulative

    summary_path = output_dir / "main_channel_segments.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    cross_sections_path = output_dir / "main_channel_cross_sections.csv"
    if cross_section_frames:
        pd.concat(cross_section_frames, ignore_index=True).to_csv(cross_sections_path, index=False)
    else:
        pd.DataFrame(
            columns=[
                "zone_id",
                "segment_id",
                "station_m",
                "station_global_m",
                "distance_from_center_m",
                "elevation_m",
                "x",
                "y",
            ]
        ).to_csv(cross_sections_path, index=False)

    return summary_path, cross_sections_path, zone_lengths


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise main channel subbasins for specified zones.")
    parser.add_argument("--zones", nargs="*", default=["P3", "P4"], help="Zone identifiers to process.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/upper_truckee_channel_demo"),
        help="Base results directory.",
    )
    parser.add_argument(
        "--channel-csv",
        type=Path,
        default=Path("results/upper_truckee_channel_demo/parameters/parameter_channels.csv"),
        help="Parameter channels CSV path.",
    )
    parser.add_argument(
        "--cross-section-dir",
        type=Path,
        default=Path("results/upper_truckee_channel_demo/intermediate/channel_cross_sections"),
        help="Directory containing cross-section CSV extracts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write summary CSV files (defaults to results_root/intermediate/analysis).",
    )
    args = parser.parse_args()

    output_dir = (
        args.output_dir if args.output_dir is not None else args.results_root / "intermediate" / "analysis"
    )

    summary_path, cross_sections_path, zone_lengths = summarise_zones(
        args.zones,
        args.channel_csv,
        args.cross_section_dir,
        output_dir,
    )
    print("Main channel segment summary:", summary_path)
    print("Cross-section aggregation:", cross_sections_path)
    print("Zone total lengths:")
    for zone, total in zone_lengths.items():
        print(f"  {zone}: {total:.2f} m")


if __name__ == "__main__":
    main()
