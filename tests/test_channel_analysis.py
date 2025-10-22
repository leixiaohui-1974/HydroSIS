"""Tests for channel network extraction utilities."""
from __future__ import annotations

import math
import unittest

import numpy as np
from affine import Affine

from hydrosis.delineation.channel_analysis import (
    compute_channel_mask,
    trace_channel_segments,
)
from hydrosis.delineation.utils import D8_OFFSETS
from hydrosis.model import ChannelNetwork


def _build_upstream(flowdir: np.ndarray) -> list[list[tuple[int, int]]]:
    rows, cols = flowdir.shape
    upstream: list[list[tuple[int, int]]] = [[] for _ in range(rows * cols)]
    for row in range(rows):
        for col in range(cols):
            code = int(flowdir[row, col])
            if code not in D8_OFFSETS:
                continue
            dr, dc = D8_OFFSETS[code]
            nr, nc = row + dr, col + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                upstream[nr * cols + nc].append((row, col))
    return upstream


class ChannelAnalysisTests(unittest.TestCase):
    """Validate channel extraction on a small synthetic network."""

    def test_channel_tracing_with_branching_network(self) -> None:
        accumulation = np.zeros((4, 4), dtype=float)
        channel_cells = [(3, 0), (2, 1), (2, 2), (2, 3), (1, 0), (1, 1)]
        for row, col in channel_cells:
            accumulation[row, col] = 5.0

        flowdir = np.zeros((4, 4), dtype=int)
        flowdir[3, 0] = 2  # NE to (2, 1)
        flowdir[2, 1] = 1  # E to (2, 2)
        flowdir[2, 2] = 1  # E to (2, 3)
        flowdir[1, 0] = 1  # E to (1, 1)
        flowdir[1, 1] = 8  # SE to (2, 2)

        dem = np.zeros((4, 4), dtype=float)
        dem[3, 0] = 110.0
        dem[2, 1] = 100.0
        dem[2, 2] = 90.0
        dem[2, 3] = 80.0
        dem[1, 0] = 105.0
        dem[1, 1] = 95.0

        channel_mask = compute_channel_mask(accumulation, threshold=1.0)
        self.assertTrue(channel_mask.dtype == bool)
        upstream = _build_upstream(flowdir)
        transform = Affine(1, 0, 0, 0, -1, 0)

        segments = trace_channel_segments(flowdir, upstream, channel_mask, transform, dem)
        self.assertEqual(len(segments), 3)

        segments_by_start = {tuple(seg.cells[0]): seg for seg in segments}
        self.assertCountEqual(
            list(segments_by_start.keys()),
            [(3, 0), (1, 0), (2, 2)],
        )

        seg_a = segments_by_start[(3, 0)]
        seg_b = segments_by_start[(1, 0)]
        seg_downstream = segments_by_start[(2, 2)]

        self.assertTrue(math.isclose(seg_a.length_m, math.sqrt(2) + 1.0, rel_tol=1e-6))
        self.assertTrue(math.isclose(seg_b.length_m, 1.0 + math.sqrt(2), rel_tol=1e-6))
        self.assertTrue(math.isclose(seg_downstream.length_m, 1.0, rel_tol=1e-6))

        self.assertTrue(math.isclose(seg_a.drop_m, 20.0))
        self.assertTrue(math.isclose(seg_b.drop_m, 15.0))
        self.assertTrue(math.isclose(seg_downstream.drop_m, 10.0))

        expected_downstream_id = seg_downstream.id
        self.assertEqual(seg_a.downstream, expected_downstream_id)
        self.assertEqual(seg_b.downstream, expected_downstream_id)
        self.assertSetEqual(
            set(seg_downstream.upstream_ids),
            {seg_a.id, seg_b.id},
        )

        network = ChannelNetwork()
        for segment in segments:
            network.add_segment(segment)

        summary = network.summary()
        self.assertEqual(summary["segment_count"], 3)
        self.assertTrue(
            math.isclose(
                summary["total_length_m"],
                seg_a.length_m + seg_b.length_m + seg_downstream.length_m,
                rel_tol=1e-6,
            )
        )
        self.assertIsNotNone(summary["mean_slope"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
