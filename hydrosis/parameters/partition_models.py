"""Parameter partitioning utilities - Data Models

Data classes for zone summaries and partition outputs.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Set

import numpy as np

GridPath = Tuple[int, int]


@dataclass
class ZoneSummary:
    """Metadata describing a derived parameter zone."""

    id: str
    downstream_id: Optional[str]
    area_km2: float
    runoff_method: str
    routing_method: str


@dataclass
class SubzoneSummary:
    """Metadata describing a derived parameter subzone."""

    zone_id: str
    subzone_id: str
    area_km2: float
    downstream_subzone_id: Optional[str]
    mean_elevation: Optional[float]
    max_accumulation: Optional[float]
    pour_row: int
    pour_col: int


@dataclass
class ChannelSummary:
    """Metadata describing a derived channel segment."""

    segment_id: str
    zone_id: str
    subzone_id: str
    downstream_id: Optional[str]
    length_m: float
    slope: float
    drop_m: float


@dataclass
class PartitionOutputs:
    """Collection of derived artefacts for parameter partitioning."""

    parameter_zones: List[ParameterZoneConfig]
    zone_definitions: Dict[str, Dict[str, object]]
    zone_features: Dict[str, object]
    subzone_features: Dict[str, object]
    channel_features: Dict[str, object]
    pour_point_features: Dict[str, object]
    zone_table: List[Dict[str, object]]
    subzone_table: List[Dict[str, object]]
    channel_table: List[Dict[str, object]]
    zone_summaries: List[ZoneSummary]
    subzone_summaries: List[SubzoneSummary]
    channel_summaries: List[ChannelSummary]
    channel_network: ChannelNetwork

