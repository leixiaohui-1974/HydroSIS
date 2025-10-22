"""Distributed Green-Ampt infiltration and runoff model.

This module implements a spatially distributed version of the Green-Ampt
infiltration equation, which is a physically-based model for computing
infiltration and runoff from precipitation. The model accounts for spatial
variability in soil properties and uses a time-varying infiltration capacity
that depends on cumulative infiltration.

The Green-Ampt equation is:
    f(t) = K * (1 + (ψ * Δθ) / F(t))
    
where:
    - f(t) is the infiltration rate at time t
    - K is the saturated hydraulic conductivity
    - ψ is the wetting front suction head
    - Δθ is the soil moisture deficit
    - F(t) is the cumulative infiltration at time t

This distributed implementation allows for different soil properties in
different parts of the watershed, making it suitable for heterogeneous
catchments with varying land use and soil types.

References:
    - Green, W. H., & Ampt, G. A. (1911). Studies on soil phyics.
      The Journal of Agricultural Science, 4(1), 1-24.
    - Mein, R. G., & Larson, C. L. (1973). Modeling infiltration
      during a steady rain. Water Resources Research, 9(2), 384-394.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Mapping, Dict

from .base import RunoffModel, RunoffModelConfig
from ..validation import validate_positive, validate_probability, validate_integer, ParameterValidationError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..model import Subbasin


class DistributedGreenAmpt(RunoffModel):
    """Distributed Green-Ampt infiltration and runoff model.
    
    This model implements a physically-based approach to computing infiltration
    and runoff using the Green-Ampt equation. It accounts for spatial
    variability in soil properties and uses a time-varying infiltration
    capacity that depends on cumulative infiltration.
    
    The model divides the subbasin into multiple zones with different
    soil properties, computes infiltration for each zone, and then aggregates
    the results to get the total runoff from the subbasin.
    
    The model uses the following parameters:
        - saturated_conductivity: Saturated hydraulic conductivity (mm/hr)
        - wetting_front_suction: Wetting front suction head (mm)
        - initial_moisture: Initial soil moisture content (fraction)
        - saturated_moisture: Saturated soil moisture content (fraction)
        - porosity: Soil porosity (fraction)
        - zones: Number of zones to divide the subbasin into
        - zone_distribution: Distribution of zones (default: uniform)
    
    The model is particularly suitable for:
        - Watersheds with heterogeneous soil properties
        - Events with high-intensity rainfall
        - Studies where physical realism is important
    """

    def __init__(self, parameters: Mapping[str, float]) -> None:
        """Initialize the distributed Green-Ampt model.
        
        Args:
            parameters: Dictionary containing model parameters:
                - saturated_conductivity: Saturated hydraulic conductivity (mm/hr)
                - wetting_front_suction: Wetting front suction head (mm)
                - initial_moisture: Initial soil moisture content (fraction, 0-1)
                - saturated_moisture: Saturated soil moisture content (fraction, 0-1)
                - porosity: Soil porosity (fraction, 0-1)
                - zones: Number of zones to divide the subbasin into (default: 5)
                - zone_distribution: Type of zone distribution (uniform/random/clustered)
        """
        super().__init__(parameters)
        
        # Physical parameters
        self.k_sat = float(self.parameters.get("saturated_conductivity", 10.0))  # mm/hr
        self.psi_f = float(self.parameters.get("wetting_front_suction", 100.0))  # mm
        self.theta_i = float(self.parameters.get("initial_moisture", 0.2))  # fraction
        self.theta_s = float(self.parameters.get("saturated_moisture", 0.4))  # fraction
        self.porosity = float(self.parameters.get("porosity", 0.45))  # fraction
        
        # Ensure valid parameter ranges
        self.theta_i = max(0.0, min(1.0, self.theta_i))
        self.theta_s = max(0.0, min(1.0, self.theta_s))
        self.porosity = max(0.0, min(1.0, self.porosity))
        
        # Ensure theta_s <= porosity
        self.theta_s = min(self.theta_s, self.porosity)
        
        # Zone parameters
        self.num_zones = int(self.parameters.get("zones", 5))
        self.zone_distribution = self.parameters.get("zone_distribution", "uniform")
        
        # Calculate soil moisture deficit
        self.delta_theta = self.theta_s - self.theta_i
        
        # Initialize zones
        self.zones = self._initialize_zones()

    def validate_parameters(self) -> None:
        """Validate Distributed Green-Ampt model parameters.

        Validates:
            - saturated_conductivity: Must be > 0
            - wetting_front_suction: Must be > 0
            - initial_moisture: Must be in [0, 1]
            - saturated_moisture: Must be in [0, 1]
            - porosity: Must be in [0, 1]
            - zones: Must be >= 1
            - Cross-parameter check: initial_moisture < saturated_moisture <= porosity
        """
        k_sat = float(self.parameters.get("saturated_conductivity", 10.0))
        validate_positive("saturated_conductivity", k_sat, strict=True)

        psi_f = float(self.parameters.get("wetting_front_suction", 100.0))
        validate_positive("wetting_front_suction", psi_f, strict=True)

        theta_i = float(self.parameters.get("initial_moisture", 0.2))
        validate_probability("initial_moisture", theta_i)

        theta_s = float(self.parameters.get("saturated_moisture", 0.4))
        validate_probability("saturated_moisture", theta_s)

        porosity = float(self.parameters.get("porosity", 0.45))
        validate_probability("porosity", porosity)

        zones = int(self.parameters.get("zones", 5))
        validate_integer("zones", zones, min_value=1)

        # Cross-parameter validation
        if theta_i >= theta_s:
            raise ParameterValidationError(
                "initial_moisture",
                theta_i,
                f"must be less than saturated_moisture ({theta_s})"
            )

        if theta_s > porosity:
            raise ParameterValidationError(
                "saturated_moisture",
                theta_s,
                f"must be less than or equal to porosity ({porosity})"
            )

    def get_initial_storage(self) -> List[float]:
        """Get initial cumulative infiltration for each zone from parameters.

        Returns:
            List of initial cumulative infiltration values for each zone (mm).
        """
        return [
            float(self.parameters.get(f"initial_cumulative_infiltration_zone_{i}", 0.0))
            for i in range(self.num_zones)
        ]

    def simulate(
        self, subbasin: "Subbasin", precipitation: List[float], initial_storage: dict | None = None
    ) -> tuple[List[float], dict]:
        """Simulate runoff using the distributed Green-Ampt model.

        Args:
            subbasin: The subbasin object, not used in this model.
            precipitation: A list of precipitation values for each time step.
            initial_storage: A dictionary of initial cumulative infiltration for each zone.

        Returns:
            A tuple containing:
                - A list of runoff values for each time step.
                - A dictionary of final cumulative infiltration for each zone.
        """
        storage = initial_storage if initial_storage is not None else self.get_initial_storage()
        cumulative_infiltration = {
            i: storage[i] for i in range(self.num_zones)
        }

        total_runoff: List[float] = [0.0] * len(precipitation)

        for zone_idx, props in enumerate(self.zones):
            k = props["k_sat"]
            psi_delta_theta = props["psi_f"] * props["delta_theta"]
            area_fraction = props["area_fraction"]

            for i, p in enumerate(precipitation):
                if p <= 0:
                    continue

                # Calculate infiltration capacity based on cumulative infiltration
                f_cumulative = cumulative_infiltration[zone_idx]
                if f_cumulative > 0:
                    infiltration_capacity = k * (1 + psi_delta_theta / f_cumulative)
                else:
                    infiltration_capacity = float("inf")  # Effectively infinite at t=0

                # Infiltration is the minimum of precipitation and capacity
                infiltration = min(p, infiltration_capacity)

                # Update cumulative infiltration
                cumulative_infiltration[zone_idx] += infiltration

                # Runoff is the excess precipitation
                runoff = p - infiltration
                total_runoff[i] += runoff * area_fraction

        # Convert runoff from mm to m³/s
        # (runoff_mm / 1000) * (subbasin.area_km2 * 1e6) / 3600
        conversion_factor = subbasin.area_km2 * 1e6 / (1000 * 3600)
        flows = [q * conversion_factor for q in total_runoff]

        return flows, cumulative_infiltration

    def _initialize_zones(self) -> List[Dict[str, float]]:
        """Initialize zones with spatially variable soil properties.
        
        Returns:
            List of zone dictionaries with soil properties
        """
        zones = []
        
        if self.zone_distribution == "uniform":
            # All zones have the same properties
            zone_area = 1.0 / self.num_zones
            for _ in range(self.num_zones):
                zones.append({
                    "k_sat": self.k_sat,
                    "psi_f": self.psi_f,
                    "delta_theta": self.delta_theta,
                    "area_fraction": zone_area,
                })
        
        elif self.zone_distribution == "random":
            # Random variation in properties (±30%)
            import random
            zone_area = 1.0 / self.num_zones
            for _ in range(self.num_zones):
                k_sat_var = self.k_sat * (0.7 + 0.6 * random.random())
                psi_f_var = self.psi_f * (0.7 + 0.6 * random.random())
                
                zones.append({
                    "k_sat": k_sat_var,
                    "psi_f": psi_f_var,
                    "delta_theta": self.delta_theta,
                    "area_fraction": zone_area,
                })
        
        elif self.zone_distribution == "clustered":
            # Create clusters with different properties
            import random
            zone_area = 1.0 / self.num_zones
            
            # Define 3 cluster types with different properties
            cluster_types = [
                {"k_sat": self.k_sat * 0.5, "psi_f": self.psi_f * 1.5},  # Low conductivity, high suction
                {"k_sat": self.k_sat, "psi_f": self.psi_f},           # Medium properties
                {"k_sat": self.k_sat * 1.5, "psi_f": self.psi_f * 0.5},  # High conductivity, low suction
            ]
            
            # Assign zones to clusters
            for i in range(self.num_zones):
                cluster_idx = i % 3
                cluster = cluster_types[cluster_idx]
                
                zones.append({
                    "k_sat": cluster["k_sat"],
                    "psi_f": cluster["psi_f"],
                    "delta_theta": self.delta_theta,
                    "area_fraction": zone_area,
                })
        
        else:
            # Default to uniform distribution
            zone_area = 1.0 / self.num_zones
            for _ in range(self.num_zones):
                zones.append({
                    "k_sat": self.k_sat,
                    "psi_f": self.psi_f,
                    "delta_theta": self.delta_theta,
                    "area_fraction": zone_area,
                })
        
        return zones


# Register the model with the configuration system
RunoffModelConfig.register("distributed_green_ampt", DistributedGreenAmpt)
