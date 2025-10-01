"""Lightweight spatial+attribute database utilities (SQLite + GeoJSON).

This module provides an optional storage backend for HydroSIS to persist
geospatial layers (as GeoJSON) and model metadata (subbasins, zones, etc.).

Design goals:
- Keep existing config-file I/O intact; this is an additional capability.
- Avoid heavy GIS dependencies; rely on the Python standard library only.
- Store geometries as GeoJSON text to ensure portability.

Schema (SQLite):
- table `layers(name TEXT PRIMARY KEY, geojson TEXT NOT NULL)`
- table `subbasins(id TEXT PRIMARY KEY, area_km2 REAL, downstream TEXT)`
- table `zones(id TEXT PRIMARY KEY, description TEXT)`
- table `zone_subbasin(zone_id TEXT, subbasin_id TEXT, PRIMARY KEY(zone_id, subbasin_id))`

This is intentionally simple; users can migrate to SpatiaLite/PostGIS later.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Mapping, Sequence

try:  # pragma: no cover - optional import
    from ..model import Subbasin
    from ..parameters.zone import ParameterZone
except Exception:  # pragma: no cover - type-only fallback
    Subbasin = object  # type: ignore
    ParameterZone = object  # type: ignore


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(db_path: Path) -> None:
    """Create tables if they do not already exist."""

    with _connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS layers (
              name TEXT PRIMARY KEY,
              geojson TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS subbasins (
              id TEXT PRIMARY KEY,
              area_km2 REAL,
              downstream TEXT
            );
            CREATE TABLE IF NOT EXISTS zones (
              id TEXT PRIMARY KEY,
              description TEXT
            );
            CREATE TABLE IF NOT EXISTS zone_subbasin (
              zone_id TEXT,
              subbasin_id TEXT,
              PRIMARY KEY(zone_id, subbasin_id)
            );
            """
        )


def write_layers(db_path: Path, layers: Mapping[str, Mapping[str, object]]) -> None:
    """Upsert a set of named GeoJSON layers into the database."""

    with _connect(db_path) as conn:
        for name, geojson in layers.items():
            conn.execute(
                "INSERT INTO layers(name, geojson) VALUES(?, ?)\n"
                "ON CONFLICT(name) DO UPDATE SET geojson=excluded.geojson",
                (name, json.dumps(geojson, ensure_ascii=False)),
            )


def write_subbasins(
    db_path: Path,
    subbasins: Sequence[Subbasin],
    subbasin_geojson: Mapping[str, object] | None = None,
) -> None:
    """Persist subbasin attributes and optional geometry layer."""

    with _connect(db_path) as conn:
        for sb in subbasins:
            conn.execute(
                "INSERT INTO subbasins(id, area_km2, downstream) VALUES(?, ?, ?)\n"
                "ON CONFLICT(id) DO UPDATE SET area_km2=excluded.area_km2, downstream=excluded.downstream",
                (getattr(sb, "id", str(sb)), float(getattr(sb, "area_km2", 0.0)), getattr(sb, "downstream", None)),
            )
        if subbasin_geojson is not None:
            conn.execute(
                "INSERT INTO layers(name, geojson) VALUES(?, ?)\n"
                "ON CONFLICT(name) DO UPDATE SET geojson=excluded.geojson",
                ("subbasins", json.dumps(subbasin_geojson, ensure_ascii=False)),
            )


def write_zones(
    db_path: Path,
    zones: Sequence[ParameterZone],
    zone_geojson: Mapping[str, object] | None = None,
) -> None:
    """Persist parameter zones and their coverage relations."""

    with _connect(db_path) as conn:
        for zone in zones:
            conn.execute(
                "INSERT INTO zones(id, description) VALUES(?, ?)\n"
                "ON CONFLICT(id) DO UPDATE SET description=excluded.description",
                (getattr(zone, "id", str(zone)), getattr(zone, "description", "")),
            )
            for sid in getattr(zone, "controlled_subbasins", []):
                conn.execute(
                    "INSERT INTO zone_subbasin(zone_id, subbasin_id) VALUES(?, ?)\n"
                    "ON CONFLICT(zone_id, subbasin_id) DO NOTHING",
                    (getattr(zone, "id", str(zone)), sid),
                )
        if zone_geojson is not None:
            conn.execute(
                "INSERT INTO layers(name, geojson) VALUES(?, ?)\n"
                "ON CONFLICT(name) DO UPDATE SET geojson=excluded.geojson",
                ("parameter_zones", json.dumps(zone_geojson, ensure_ascii=False)),
            )


def load_layers(db_path: Path) -> Mapping[str, Mapping[str, object]]:
    """Return all stored GeoJSON layers by name."""

    with _connect(db_path) as conn:
        cursor = conn.execute("SELECT name, geojson FROM layers")
        result: dict[str, Mapping[str, object]] = {}
        for name, geojson in cursor.fetchall():
            try:
                result[name] = json.loads(geojson)
            except Exception:
                result[name] = {"type": "FeatureCollection", "features": []}
        return result


__all__ = [
    "init_db",
    "write_layers",
    "write_subbasins",
    "write_zones",
    "load_layers",
]