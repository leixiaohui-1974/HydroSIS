"""High-level orchestration helpers wired around :class:`HydroProjectConfig`."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ..config import HydroProjectConfig, ModelConfig
from ..parameters.partition import ChannelSummary, PartitionOutputs, partition_parameter_zones
from ..workflow import run_workflow


def build_project_model(project_config: HydroProjectConfig) -> Tuple[PartitionOutputs, ModelConfig]:
    """Derive parameter partitions and construct a :class:`ModelConfig`."""

    partition_outputs = partition_parameter_zones(
        project_config.delineation,
        project_config.partition,
        project_config.model,
        project_config.outputs,
    )

    def _build_precomputed_subbasins(
        outputs: PartitionOutputs,
    ) -> Tuple[
        List[Dict[str, object]],
        Optional[object],
        Dict[str, List[str]],
        Dict[str, str],
    ]:
        if not outputs.subzone_summaries:
            return [], None, {}, {}
        channel_by_subzone: Dict[str, ChannelSummary] = {
            channel.subzone_id: channel for channel in outputs.channel_summaries
        }
        subbasins_payload: List[Dict[str, object]] = []
        zone_to_subzones: Dict[str, List[str]] = defaultdict(list)
        zone_outlet: Dict[str, str] = {}
        for summary in outputs.subzone_summaries:
            entry: Dict[str, object] = {
                "id": summary.subzone_id,
                "area_km2": summary.area_km2,
                "downstream": summary.downstream_subzone_id,
                "parameters": {},
            }
            zone_to_subzones[summary.zone_id].append(summary.subzone_id)
            channel = channel_by_subzone.get(summary.subzone_id)
            if channel is not None:
                entry.update(
                    {
                        "channel_id": channel.segment_id,
                        "channel_length_m": channel.length_m,
                        "channel_slope": channel.slope,
                        "channel_drop_m": channel.drop_m,
                    }
                )
            subbasins_payload.append(entry)
        for summary in outputs.subzone_summaries:
            downstream = summary.downstream_subzone_id
            if (
                downstream is None
                or not downstream.startswith(summary.zone_id)
                or downstream not in zone_to_subzones[summary.zone_id]
            ):
                zone_outlet.setdefault(summary.zone_id, summary.subzone_id)
        for zone_id, subzones in zone_to_subzones.items():
            if zone_id not in zone_outlet and subzones:
                zone_outlet[zone_id] = subzones[0]
        return subbasins_payload, outputs.channel_network, zone_to_subzones, zone_outlet

    (
        precomputed_subbasins,
        channel_network,
        zone_to_subzones,
        zone_outlet,
    ) = _build_precomputed_subbasins(partition_outputs)
    if precomputed_subbasins:
        project_config.delineation.precomputed_subbasins = precomputed_subbasins
        project_config.delineation._channel_network = channel_network
        updated_zone_configs: List = []
        for cfg in partition_outputs.parameter_zones:
            sub_list = zone_to_subzones.get(cfg.id, list(cfg.explicit_subbasins or []))
            control_point = zone_outlet.get(cfg.id)
            control_points = [control_point] if control_point else list(cfg.control_points)
            updated_zone_configs.append(
                type(cfg)(
                    id=cfg.id,
                    description=cfg.description,
                    control_points=control_points,
                    parameters=dict(cfg.parameters),
                    explicit_subbasins=sub_list or None,
                )
            )
        partition_outputs.parameter_zones = updated_zone_configs
        for zone_id, definition in partition_outputs.zone_definitions.items():
            if zone_id in zone_to_subzones:
                definition["subbasins"] = sorted(zone_to_subzones[zone_id])
            if zone_id in zone_outlet:
                definition["control_points"] = [zone_outlet[zone_id]]

    model_config = project_config.build_model_config(partition_outputs.parameter_zones)
    return partition_outputs, model_config


def run_project(
    project_config: HydroProjectConfig,
    forcing: Mapping[str, Sequence[float]],
    observations: Optional[Mapping[str, Sequence[float]]] = None,
    scenario_ids: Optional[Sequence[str]] = None,
    persist_outputs: bool = True,
    generate_report: bool = False,
):
    """Prepare and execute a HydroSIS project using its unified configuration."""

    partition_outputs, model_config = build_project_model(project_config)
    result = run_workflow(
        model_config,
        forcing,
        observations=observations,
        scenario_ids=scenario_ids,
        persist_outputs=persist_outputs,
        generate_report=generate_report,
    )
    return partition_outputs, result
