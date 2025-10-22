"""Integration tests ensuring delineation exposes channel network metadata."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hydrosis.delineation.dem_delineator import DelineationConfig
from hydrosis.testing.synthetic_datasets import write_synthetic_delineation_inputs


class DelineationChannelNetworkTests(unittest.TestCase):
    """Validate that channel extraction runs during delineation."""

    def test_channel_network_available_after_delineation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dem_path, pour_points_path = write_synthetic_delineation_inputs(tmp_path)

            config = DelineationConfig(
                dem_path=dem_path,
                pour_points_path=pour_points_path,
                accumulation_threshold=2.0,
                channel_threshold=2.0,
            )

            subbasins = config.to_subbasins()
            self.assertGreater(len(subbasins), 0)
            for sub in subbasins:
                self.assertIsNotNone(sub.channel_id)
                self.assertGreater(sub.channel_length_m or 0.0, 0.0)

            network = config.to_channel_network()
            self.assertIsNotNone(network)

            summary = network.summary() if network is not None else {}
            self.assertGreater(summary.get("segment_count", 0), 0)
            self.assertGreater(summary.get("total_length_m", 0.0), 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
