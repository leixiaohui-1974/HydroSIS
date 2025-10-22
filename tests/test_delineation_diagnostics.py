"""Tests for DEM-based subbasin delineation diagnostics."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from hydrosis.delineation.dem_delineator import DelineationConfig


class DelineationDiagnosticsTests(unittest.TestCase):
    """Ensure delineation outputs diagnostic artefacts with sane metadata."""

    def test_json_delineation_produces_diagnostics_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dem_src = Path("data/sample/dem/typical_watershed_dem.json")
            pour_src = Path("data/sample/gis/typical_watershed_pour_points.geojson")
            dem_path = tmp_path / "dem.json"
            pour_path = tmp_path / "pour_points.geojson"
            shutil.copy2(dem_src, dem_path)
            shutil.copy2(pour_src, pour_path)

            config = DelineationConfig(
                dem_path=dem_path,
                pour_points_path=pour_path,
                accumulation_threshold=500.0,
            )

            subbasins = config.to_subbasins()
            self.assertGreater(len(subbasins), 0)
            for sub in subbasins:
                self.assertGreater(sub.area_km2, 0.0)

            derived_dir = dem_path.parent / "derived"
            diagnostics_path = derived_dir / "delineation_diagnostics.json"
            self.assertTrue(
                diagnostics_path.exists(),
                "Delineation diagnostics JSON should be exported alongside derived artefacts.",
            )

            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            self.assertIn("pour_points", diagnostics)
            self.assertIsInstance(diagnostics["pour_points"], list)
            self.assertEqual(len(diagnostics["pour_points"]), len(subbasins))
            self.assertIn("total_area_km2", diagnostics)

            for entry in diagnostics["pour_points"]:
                self.assertIn("id", entry)
                self.assertIn("area_km2", entry)
                self.assertGreater(entry["area_km2"], 0.0)
                self.assertIn("downstream_id", entry)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
