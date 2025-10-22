"""Tests for routing models making use of channel metadata."""
from __future__ import annotations

import unittest

from hydrosis.model import Subbasin
from hydrosis.routing.dynamic_wave import DynamicWaveRouting
from hydrosis.routing.muskingum import MuskingumRouting


class RoutingChannelIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.subbasin = Subbasin(
            id="SB1",
            area_km2=5.0,
            downstream=None,
            channel_id="S1",
            channel_length_m=2_000.0,
            channel_slope=0.0025,
            channel_drop_m=5.0,
        )

    def test_muskingum_uses_channel_length(self) -> None:
        routing = MuskingumRouting(parameters={})
        resolved = routing.resolved_parameters(self.subbasin)
        self.assertLess(resolved["travel_time"], 2.0)  # hours
        self.assertGreater(resolved["travel_time"], 0.0)

    def test_dynamic_wave_derives_reach_length(self) -> None:
        routing = DynamicWaveRouting(parameters={})
        dt, segments, dx, wave_celerity, diffusivity = routing._resolve_channel_properties(self.subbasin)
        self.assertEqual(segments, max(2, int(routing.parameters.get("segments", 5))))
        self.assertAlmostEqual(dx * segments, 2_000.0, delta=1.0)
        self.assertGreater(wave_celerity, 0.5)
        self.assertGreater(diffusivity, 0.0)
        self.assertGreater(dt, 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
