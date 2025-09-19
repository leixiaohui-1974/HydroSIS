"""Network utilities for accumulating subbasin flows."""
from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Dict, Mapping, MutableMapping, Sequence, List

if TYPE_CHECKING:  # pragma: no cover - avoid runtime import cycles
    from ..model import Subbasin


def _ensure_lengths(local_flows: Mapping[str, Sequence[float]]) -> int:
    """Validate that all series share the same temporal dimension."""

    unique_lengths = {len(series) for series in local_flows.values()}
    if not unique_lengths:
        return 0
    if len(unique_lengths) != 1:
        raise ValueError("Inconsistent time series lengths across subbasins")
    return unique_lengths.pop()


def accumulate_subbasin_flows(
    subbasins: Mapping[str, "Subbasin"],
    local_flows: Mapping[str, Sequence[float]],
) -> Dict[str, List[float]]:
    """Accumulate routed flows along the basin network.

    Parameters
    ----------
    subbasins:
        Mapping of subbasin identifiers to the delineated units forming the
        drainage network.
    local_flows:
        Mapping of subbasin identifiers to routed discharge produced locally by
        each subbasin (i.e. without upstream contributions).

    Returns
    -------
    dict
        Dictionary keyed by subbasin identifiers with discharge series that
        include upstream contributions.

    Raises
    ------
    KeyError
        If local flows are missing for a delineated subbasin or if a subbasin
        references a downstream unit that is not present in the delineation.
    ValueError
        When the network contains cycles or the time dimensions differ across
        local flow series.
    """

    missing = set(subbasins) - set(local_flows)
    if missing:
        raise KeyError(
            "Local routed flows missing for subbasins: " + ", ".join(sorted(missing))
        )

    steps = _ensure_lengths(local_flows)

    accumulated: Dict[str, MutableMapping[int, float]] = {
        sub_id: {idx: value for idx, value in enumerate(series)}
        for sub_id, series in local_flows.items()
    }

    upstream_counts: Dict[str, int] = {sub_id: 0 for sub_id in subbasins}
    for sub in subbasins.values():
        downstream = sub.downstream
        if downstream is None:
            continue
        if downstream not in subbasins:
            raise KeyError(
                f"Subbasin {sub.id} references unknown downstream {downstream}"
            )
        upstream_counts[downstream] += 1

    frontier: deque[str] = deque(
        [sub_id for sub_id, count in upstream_counts.items() if count == 0]
    )
    visited = 0

    while frontier:
        current = frontier.popleft()
        visited += 1
        downstream = subbasins[current].downstream
        if downstream is None:
            continue

        target = accumulated.setdefault(downstream, {idx: 0.0 for idx in range(steps)})
        if len(target) != steps:
            raise ValueError("Downstream series length does not match upstream flows")

        for idx in range(steps):
            target[idx] = target[idx] + accumulated[current][idx]

        upstream_counts[downstream] -= 1
        if upstream_counts[downstream] == 0:
            frontier.append(downstream)

    if visited != len(subbasins):
        raise ValueError("Network contains cycles or inaccessible subbasins")

    return {
        sub_id: [series[idx] for idx in range(steps)]
        for sub_id, series in accumulated.items()
    }


__all__ = ["accumulate_subbasin_flows"]
