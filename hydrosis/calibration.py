"""Parameter calibration and adjustment module for HydroSIS.

This module provides functionality to:
1. Load parameter adjustments from YAML configuration files
2. Apply adjustments using multiply, add, or set methods
3. Manage parameter priorities (global defaults -> zone adjustments -> subbasin overrides)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import yaml

logger = logging.getLogger(__name__)


class ParameterAdjustment:
    """Represents a single parameter adjustment."""

    def __init__(self, method: str, value: float):
        """Initialize parameter adjustment.

        Args:
            method: Adjustment method - 'multiply', 'add', or 'set'
            value: Adjustment value
        """
        if method not in ['multiply', 'add', 'set']:
            raise ValueError(f"Invalid adjustment method: {method}. Must be 'multiply', 'add', or 'set'")

        self.method = method
        self.value = float(value)

    def apply(self, base_value: float) -> float:
        """Apply adjustment to a base value.

        Args:
            base_value: Original parameter value

        Returns:
            Adjusted parameter value
        """
        if self.method == 'multiply':
            result = base_value * self.value
        elif self.method == 'add':
            result = base_value + self.value
        elif self.method == 'set':
            result = self.value
        else:
            raise ValueError(f"Unknown method: {self.method}")

        logger.debug(f"Applied {self.method}({self.value}) to {base_value} -> {result}")
        return result

    def __repr__(self):
        return f"ParameterAdjustment(method='{self.method}', value={self.value})"


class ZoneAdjustment:
    """Represents parameter adjustments for a specific zone."""

    def __init__(self, zone_id: int, description: str = ""):
        self.zone_id = zone_id
        self.description = description
        self.runoff_adjustments: Dict[str, Dict[str, ParameterAdjustment]] = {}
        self.routing_adjustments: Dict[str, Dict[str, ParameterAdjustment]] = {}

    def add_runoff_adjustment(self, model: str, param: str, method: str, value: float):
        """Add a runoff parameter adjustment.

        Args:
            model: Model name (e.g., 'hbv')
            param: Parameter name (e.g., 'FC')
            method: Adjustment method
            value: Adjustment value
        """
        if model not in self.runoff_adjustments:
            self.runoff_adjustments[model] = {}
        self.runoff_adjustments[model][param] = ParameterAdjustment(method, value)

    def add_routing_adjustment(self, model: str, param: str, method: str, value: float):
        """Add a routing parameter adjustment.

        Args:
            model: Model name (e.g., 'muskingum')
            param: Parameter name (e.g., 'K')
            method: Adjustment method
            value: Adjustment value
        """
        if model not in self.routing_adjustments:
            self.routing_adjustments[model] = {}
        self.routing_adjustments[model][param] = ParameterAdjustment(method, value)

    def apply_runoff_adjustments(self, model: str, parameters: Dict[str, float]) -> Dict[str, float]:
        """Apply runoff adjustments to a parameter dict.

        Args:
            model: Model name
            parameters: Current parameters

        Returns:
            Adjusted parameters
        """
        if model not in self.runoff_adjustments:
            return parameters

        adjusted = parameters.copy()
        for param, adjustment in self.runoff_adjustments[model].items():
            if param in adjusted:
                adjusted[param] = adjustment.apply(adjusted[param])
            else:
                logger.warning(f"Parameter {param} not found in base parameters for zone {self.zone_id}")

        return adjusted

    def apply_routing_adjustments(self, model: str, parameters: Dict[str, float]) -> Dict[str, float]:
        """Apply routing adjustments to a parameter dict.

        Args:
            model: Model name
            parameters: Current parameters

        Returns:
            Adjusted parameters
        """
        if model not in self.routing_adjustments:
            return parameters

        adjusted = parameters.copy()
        for param, adjustment in self.routing_adjustments[model].items():
            if param in adjusted:
                adjusted[param] = adjustment.apply(adjusted[param])
            else:
                logger.warning(f"Parameter {param} not found in base parameters for zone {self.zone_id}")

        return adjusted


class SubbasinOverride:
    """Represents parameter overrides for a specific subbasin."""

    def __init__(self, subbasin_id: Union[int, str], description: str = ""):
        self.subbasin_id = str(subbasin_id)
        self.description = description
        self.runoff_adjustments: Dict[str, Dict[str, ParameterAdjustment]] = {}
        self.routing_adjustments: Dict[str, Dict[str, ParameterAdjustment]] = {}

    def add_runoff_adjustment(self, model: str, param: str, method: str, value: float):
        """Add a runoff parameter adjustment."""
        if model not in self.runoff_adjustments:
            self.runoff_adjustments[model] = {}
        self.runoff_adjustments[model][param] = ParameterAdjustment(method, value)

    def add_routing_adjustment(self, model: str, param: str, method: str, value: float):
        """Add a routing parameter adjustment."""
        if model not in self.routing_adjustments:
            self.routing_adjustments[model] = {}
        self.routing_adjustments[model][param] = ParameterAdjustment(method, value)

    def apply_runoff_adjustments(self, model: str, parameters: Dict[str, float]) -> Dict[str, float]:
        """Apply runoff adjustments."""
        if model not in self.runoff_adjustments:
            return parameters

        adjusted = parameters.copy()
        for param, adjustment in self.runoff_adjustments[model].items():
            if param in adjusted:
                adjusted[param] = adjustment.apply(adjusted[param])

        return adjusted

    def apply_routing_adjustments(self, model: str, parameters: Dict[str, float]) -> Dict[str, float]:
        """Apply routing adjustments."""
        if model not in self.routing_adjustments:
            return parameters

        adjusted = parameters.copy()
        for param, adjustment in self.routing_adjustments[model].items():
            if param in adjusted:
                adjusted[param] = adjustment.apply(adjusted[param])

        return adjusted


class ParameterCalibration:
    """Main class for parameter calibration management."""

    def __init__(self, yaml_path: Optional[Union[str, Path]] = None):
        """Initialize parameter calibration.

        Args:
            yaml_path: Path to YAML calibration file (optional)
        """
        self.global_defaults: Dict[str, Dict[str, float]] = {}
        self.zone_adjustments: Dict[int, ZoneAdjustment] = {}
        self.subbasin_overrides: Dict[str, SubbasinOverride] = {}

        if yaml_path:
            self.load_from_yaml(yaml_path)

    def load_from_yaml(self, yaml_path: Union[str, Path]):
        """Load calibration configuration from YAML file.

        Args:
            yaml_path: Path to YAML file
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {yaml_path}")

        logger.info(f"Loading parameter calibration from: {yaml_path}")

        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        # Load global defaults
        if 'global_defaults' in config:
            self.global_defaults = config['global_defaults']
            logger.info(f"Loaded global defaults for models: {list(self.global_defaults.keys())}")

        # Load zone adjustments
        if 'zone_adjustments' in config:
            for zone_config in config['zone_adjustments']:
                zone_id = int(zone_config['zone_id'])
                description = zone_config.get('description', '')

                zone_adj = ZoneAdjustment(zone_id, description)

                # Runoff parameters
                if 'runoff_parameters' in zone_config:
                    for model, params in zone_config['runoff_parameters'].items():
                        for param, adj_config in params.items():
                            zone_adj.add_runoff_adjustment(
                                model, param,
                                adj_config['method'],
                                adj_config['value']
                            )

                # Routing parameters
                if 'routing_parameters' in zone_config:
                    for model, params in zone_config['routing_parameters'].items():
                        for param, adj_config in params.items():
                            zone_adj.add_routing_adjustment(
                                model, param,
                                adj_config['method'],
                                adj_config['value']
                            )

                self.zone_adjustments[zone_id] = zone_adj
                logger.info(f"Loaded adjustments for zone {zone_id}: {description}")

        # Load subbasin overrides
        if 'subbasin_overrides' in config and config['subbasin_overrides']:
            for sub_config in config['subbasin_overrides']:
                sub_id = str(sub_config['subbasin_id'])
                description = sub_config.get('description', '')

                sub_override = SubbasinOverride(sub_id, description)

                # Runoff parameters
                if 'runoff_parameters' in sub_config:
                    for model, params in sub_config['runoff_parameters'].items():
                        for param, adj_config in params.items():
                            sub_override.add_runoff_adjustment(
                                model, param,
                                adj_config['method'],
                                adj_config['value']
                            )

                # Routing parameters
                if 'routing_parameters' in sub_config:
                    for model, params in sub_config['routing_parameters'].items():
                        for param, adj_config in params.items():
                            sub_override.add_routing_adjustment(
                                model, param,
                                adj_config['method'],
                                adj_config['value']
                            )

                self.subbasin_overrides[sub_id] = sub_override
                logger.info(f"Loaded overrides for subbasin {sub_id}: {description}")

        logger.info(f"Calibration loaded: {len(self.zone_adjustments)} zones, "
                   f"{len(self.subbasin_overrides)} subbasin overrides")

    def get_runoff_parameters(
        self,
        model: str,
        zone_id: int,
        subbasin_id: Optional[Union[int, str]] = None,
        base_parameters: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Get adjusted runoff parameters for a subbasin.

        Priority: global_defaults -> base_parameters -> zone_adjustments -> subbasin_overrides

        Args:
            model: Model name (e.g., 'hbv')
            zone_id: Zone ID
            subbasin_id: Subbasin ID (optional)
            base_parameters: Base parameters (optional, uses global defaults if not provided)

        Returns:
            Adjusted parameters
        """
        # Start with global defaults
        params = self.global_defaults.get(model, {}).copy()

        # Override with base parameters if provided
        if base_parameters:
            params.update(base_parameters)

        # Apply zone adjustments
        if zone_id in self.zone_adjustments:
            params = self.zone_adjustments[zone_id].apply_runoff_adjustments(model, params)

        # Apply subbasin overrides
        if subbasin_id is not None:
            sub_id_str = str(subbasin_id)
            if sub_id_str in self.subbasin_overrides:
                params = self.subbasin_overrides[sub_id_str].apply_runoff_adjustments(model, params)

        return params

    def get_routing_parameters(
        self,
        model: str,
        zone_id: int,
        channel_id: Optional[Union[int, str]] = None,
        base_parameters: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Get adjusted routing parameters for a channel.

        Priority: global_defaults -> base_parameters -> zone_adjustments -> channel_overrides

        Args:
            model: Model name (e.g., 'muskingum')
            zone_id: Zone ID
            channel_id: Channel ID (optional)
            base_parameters: Base parameters (optional)

        Returns:
            Adjusted parameters
        """
        # Start with global defaults
        params = self.global_defaults.get(model, {}).copy()

        # Override with base parameters if provided
        if base_parameters:
            params.update(base_parameters)

        # Apply zone adjustments
        if zone_id in self.zone_adjustments:
            params = self.zone_adjustments[zone_id].apply_routing_adjustments(model, params)

        # Apply channel overrides (if using subbasin_overrides for channels)
        if channel_id is not None:
            ch_id_str = str(channel_id)
            if ch_id_str in self.subbasin_overrides:
                params = self.subbasin_overrides[ch_id_str].apply_routing_adjustments(model, params)

        return params

    def get_summary(self) -> str:
        """Get a summary of the calibration configuration.

        Returns:
            Summary string
        """
        lines = ["=" * 80]
        lines.append("Parameter Calibration Summary")
        lines.append("=" * 80)

        # Global defaults
        lines.append("\nGlobal Defaults:")
        for model, params in self.global_defaults.items():
            lines.append(f"  {model}: {len(params)} parameters")

        # Zone adjustments
        lines.append(f"\nZone Adjustments: {len(self.zone_adjustments)} zones")
        for zone_id, zone_adj in sorted(self.zone_adjustments.items()):
            runoff_count = sum(len(p) for p in zone_adj.runoff_adjustments.values())
            routing_count = sum(len(p) for p in zone_adj.routing_adjustments.values())
            lines.append(f"  Zone {zone_id}: {runoff_count} runoff + {routing_count} routing adjustments")
            if zone_adj.description:
                lines.append(f"    Description: {zone_adj.description}")

        # Subbasin overrides
        if self.subbasin_overrides:
            lines.append(f"\nSubbasin Overrides: {len(self.subbasin_overrides)} subbasins")
            for sub_id, sub_override in self.subbasin_overrides.items():
                lines.append(f"  Subbasin {sub_id}")

        lines.append("=" * 80)
        return "\n".join(lines)


__all__ = ['ParameterCalibration', 'ParameterAdjustment', 'ZoneAdjustment', 'SubbasinOverride']
