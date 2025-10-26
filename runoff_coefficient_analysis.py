#!/usr/bin/env python3
"""Runoff Coefficient Analysis and Calibration

This script performs comprehensive analysis of runoff coefficients:
1. Calculate observed runoff coefficient from generated precipitation-runoff data
2. Simulate runoff using multiple hydrologic models
3. Compare observed vs simulated runoff coefficients by zone
4. Parameter sensitivity analysis
5. Automatic calibration to match observed coefficients
"""
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RunoffCoefficientResult:
    """Runoff coefficient result for a zone"""
    zone_id: str
    observed_rc: float
    simulated_rc: Dict[str, float]  # model_name -> rc
    precipitation_mm: float
    observed_runoff_mm: float
    simulated_runoff_mm: Dict[str, float]
    area_km2: float
    model_parameters: Dict[str, Dict[str, float]] = field(default_factory=dict)


class HydrologicModel:
    """Base class for hydrologic models"""
    
    def __init__(self, name: str):
        self.name = name
        self.parameters = {}
    
    def run(self, precipitation: np.ndarray, **params) -> np.ndarray:
        """Run the model
        
        Args:
            precipitation: Precipitation time series (mm/h)
            **params: Model parameters
            
        Returns:
            Runoff time series (mm/h)
        """
        raise NotImplementedError


class HBVModel(HydrologicModel):
    """HBV Hydrologic Model"""
    
    def __init__(self):
        super().__init__("HBV")
        self.default_params = {
            'field_capacity': 200.0,  # FC (mm)
            'beta': 2.0,               # Beta (-)
            'k0': 0.05,                # Fast reservoir coefficient (1/h)
            'k1': 0.01,                # Medium reservoir coefficient (1/h)
            'k2': 0.001,               # Slow reservoir coefficient (1/h)
            'percolation': 2.0,        # Percolation rate (mm/h)
            'maxbas': 3.0              # Routing parameter
        }
    
    def run(self, precipitation: np.ndarray, **params) -> np.ndarray:
        """Run HBV model"""
        # Merge default and provided parameters
        p = {**self.default_params, **params}
        
        n = len(precipitation)
        runoff = np.zeros(n)
        
        # State variables
        soil_moisture = p['field_capacity'] * 0.5
        upper_zone = 0.0
        lower_zone = 0.0
        
        for t in range(n):
            precip = precipitation[t]
            
            # Soil moisture accounting
            recharge = precip * (soil_moisture / p['field_capacity']) ** p['beta']
            
            # Update soil moisture
            soil_moisture = min(p['field_capacity'], soil_moisture + precip - recharge)
            
            # Upper zone response
            upper_zone += recharge
            
            # Percolation to lower zone
            perc = min(p['percolation'], upper_zone)
            upper_zone -= perc
            lower_zone += perc
            
            # Runoff generation
            q0 = p['k0'] * upper_zone
            q1 = p['k1'] * upper_zone
            q2 = p['k2'] * lower_zone
            
            runoff[t] = q0 + q1 + q2
            
            # Update zones
            upper_zone -= (q0 + q1)
            lower_zone -= q2
        
        return runoff


class SCSCNModel(HydrologicModel):
    """SCS Curve Number Model"""
    
    def __init__(self):
        super().__init__("SCS-CN")
        self.default_params = {
            'curve_number': 75.0,      # CN (-)
            'initial_abstraction_ratio': 0.2  # Ia/S ratio
        }
    
    def run(self, precipitation: np.ndarray, **params) -> np.ndarray:
        """Run SCS-CN model"""
        p = {**self.default_params, **params}
        
        # Convert CN to S (maximum retention)
        S = 25400.0 / p['curve_number'] - 254.0  # mm
        Ia = p['initial_abstraction_ratio'] * S
        
        n = len(precipitation)
        runoff = np.zeros(n)
        
        for t in range(n):
            P = precipitation[t]
            
            if P > Ia:
                # Direct runoff (mm/h)
                Q = (P - Ia) ** 2 / (P - Ia + S)
                runoff[t] = Q
        
        return runoff


class GreenAmptModel(HydrologicModel):
    """Green-Ampt Infiltration Model"""
    
    def __init__(self):
        super().__init__("Green-Ampt")
        self.default_params = {
            'hydraulic_conductivity': 10.0,  # mm/h
            'suction_head': 100.0,           # mm
            'porosity': 0.45,                # -
            'initial_moisture': 0.2          # -
        }
    
    def run(self, precipitation: np.ndarray, **params) -> np.ndarray:
        """Run Green-Ampt model"""
        p = {**self.default_params, **params}
        
        n = len(precipitation)
        runoff = np.zeros(n)
        
        # Initial conditions
        deficit = p['porosity'] - p['initial_moisture']
        cumulative_infiltration = 0.0
        
        for t in range(n):
            P = precipitation[t]
            
            # Calculate infiltration capacity
            if cumulative_infiltration > 0:
                f = p['hydraulic_conductivity'] * (1 + p['suction_head'] * deficit / cumulative_infiltration)
            else:
                f = p['hydraulic_conductivity'] * 100  # Very high initially
            
            # Actual infiltration
            actual_infiltration = min(P, f)
            cumulative_infiltration += actual_infiltration
            
            # Runoff is excess precipitation
            runoff[t] = max(0, P - actual_infiltration)
        
        return runoff


class MuskingumRouting:
    """Muskingum Routing Model"""
    
    def __init__(self):
        self.name = "Muskingum"
        self.default_params = {
            'K': 2.0,  # Travel time (hours)
            'x': 0.2   # Weighting factor
        }
    
    def route(self, inflow: np.ndarray, dt: float = 1.0, **params) -> np.ndarray:
        """Route hydrograph using Muskingum method"""
        p = {**self.default_params, **params}
        
        K = p['K']
        x = p['x']
        
        # Muskingum coefficients
        C0 = (-K * x + 0.5 * dt) / (K - K * x + 0.5 * dt)
        C1 = (K * x + 0.5 * dt) / (K - K * x + 0.5 * dt)
        C2 = (K - K * x - 0.5 * dt) / (K - K * x + 0.5 * dt)
        
        n = len(inflow)
        outflow = np.zeros(n)
        outflow[0] = inflow[0]
        
        for t in range(1, n):
            outflow[t] = C0 * inflow[t] + C1 * inflow[t-1] + C2 * outflow[t-1]
        
        return outflow


class KinematicWaveRouting:
    """Kinematic Wave Routing Model"""
    
    def __init__(self):
        self.name = "Kinematic Wave"
        self.default_params = {
            'alpha': 1.0,  # Channel coefficient
            'beta': 0.6    # Channel exponent
        }
    
    def route(self, inflow: np.ndarray, dt: float = 1.0, **params) -> np.ndarray:
        """Route hydrograph using Kinematic Wave"""
        p = {**self.default_params, **params}
        
        # Simplified kinematic wave routing
        n = len(inflow)
        outflow = np.zeros(n)
        
        storage = 0.0
        
        for t in range(n):
            storage += inflow[t] * dt
            
            # Discharge calculation
            if storage > 0:
                Q = p['alpha'] * storage ** p['beta']
                outflow[t] = Q
                storage = max(0, storage - Q * dt)
            else:
                outflow[t] = 0
        
        return outflow


class RunoffCoefficientAnalyzer:
    """Comprehensive runoff coefficient analysis and calibration"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.runoff_models = {
            'HBV': HBVModel(),
            'SCS-CN': SCSCNModel(),
            'Green-Ampt': GreenAmptModel()
        }
        
        self.routing_models = {
            'Muskingum': MuskingumRouting(),
            'Kinematic': KinematicWaveRouting()
        }
        
        self.results = {}
        
        # Setup matplotlib
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            self.plt = plt
        except ImportError:
            logger.warning("matplotlib not available")
            self.plt = None
    
    def load_data(
        self,
        precipitation_file: Path,
        discharge_file: Path,
        watershed_file: Path
    ):
        """Load precipitation, discharge, and watershed data"""
        logger.info(f"Loading data files...")
        
        # Load precipitation
        self.precip_df = pd.read_csv(precipitation_file)
        logger.info(f"  Precipitation: {len(self.precip_df)} time steps, {len(self.precip_df.columns)-1} zones")
        
        # Load discharge
        self.discharge_df = pd.read_csv(discharge_file)
        logger.info(f"  Discharge: {len(self.discharge_df)} time steps, {len(self.discharge_df.columns)-1} points")
        
        # Load watershed info
        try:
            import geopandas as gpd
            self.watersheds = gpd.read_file(watershed_file)
            logger.info(f"  Watersheds: {len(self.watersheds)} polygons")
        except Exception as e:
            logger.warning(f"Could not load watershed file: {e}")
            self.watersheds = None
    
    def calculate_observed_coefficients(self) -> Dict[str, float]:
        """Calculate observed runoff coefficients from data"""
        logger.info("\nCalculating observed runoff coefficients...")
        
        observed_rc = {}
        
        # Get column names
        precip_cols = [col for col in self.precip_df.columns if col != self.precip_df.columns[0]]
        discharge_cols = [col for col in self.discharge_df.columns if col != self.discharge_df.columns[0]]
        
        for i, (pcol, dcol) in enumerate(zip(precip_cols, discharge_cols)):
            # Total precipitation (mm) - hourly sum
            total_precip = self.precip_df[pcol].sum()
            
            # Total discharge: sum of m³/s values, need to convert to volume
            # Each value is for 1 hour, so volume = sum(m³/s) * 3600 s
            total_discharge_m3s_sum = self.discharge_df[dcol].sum()
            total_discharge_m3 = total_discharge_m3s_sum * 3600  # m³
            
            # Get watershed area
            if self.watersheds is not None and i < len(self.watersheds):
                area_m2 = self.watersheds.iloc[i].geometry.area
                area_km2 = area_m2 / 1e6
            else:
                area_km2 = 100.0  # Default
                area_m2 = area_km2 * 1e6
            
            # Convert discharge to depth
            if area_m2 > 0:
                runoff_depth_mm = (total_discharge_m3 / area_m2) * 1000
            else:
                runoff_depth_mm = 0
            
            # Runoff coefficient
            if total_precip > 0:
                rc = runoff_depth_mm / total_precip
            else:
                rc = 0.0
            
            observed_rc[dcol] = rc
            
            logger.info(f"  {dcol}: RC = {rc:.4f} (P={total_precip:.1f}mm, R={runoff_depth_mm:.1f}mm)")
        
        return observed_rc
    
    def run_model_comparison(
        self,
        zone_id: str,
        precipitation: np.ndarray,
        area_km2: float
    ) -> Dict[str, np.ndarray]:
        """Run all models for a specific zone"""
        
        simulated_runoff = {}
        
        # Run each runoff generation model
        for model_name, model in self.runoff_models.items():
            try:
                runoff = model.run(precipitation)
                simulated_runoff[model_name] = runoff
            except Exception as e:
                logger.warning(f"Model {model_name} failed for {zone_id}: {e}")
                simulated_runoff[model_name] = np.zeros_like(precipitation)
        
        # Apply routing to each
        routed_runoff = {}
        for runoff_model, runoff in simulated_runoff.items():
            for routing_model_name, routing_model in self.routing_models.items():
                combined_name = f"{runoff_model}+{routing_model_name}"
                try:
                    routed = routing_model.route(runoff)
                    routed_runoff[combined_name] = routed
                except Exception as e:
                    logger.warning(f"Routing {combined_name} failed: {e}")
        
        # Combine all results
        all_results = {**simulated_runoff, **routed_runoff}
        
        return all_results
    
    def analyze_all_zones(self) -> List[RunoffCoefficientResult]:
        """Analyze all zones with all model combinations"""
        logger.info("\nAnalyzing all zones with multiple models...")
        
        results = []
        
        # Get observed coefficients
        observed_rc = self.calculate_observed_coefficients()
        
        # Get column names
        precip_cols = [col for col in self.precip_df.columns if col != self.precip_df.columns[0]]
        discharge_cols = [col for col in self.discharge_df.columns if col != self.discharge_df.columns[0]]
        
        for i, (pcol, dcol) in enumerate(zip(precip_cols, discharge_cols)):
            logger.info(f"\n  Processing zone {dcol}...")
            
            # Get data
            precip = self.precip_df[pcol].values
            total_precip = precip.sum()
            
            # Get area
            if self.watersheds is not None and i < len(self.watersheds):
                area_m2 = self.watersheds.iloc[i].geometry.area
                area_km2 = area_m2 / 1e6
            else:
                area_m2 = 100e6  # 100 km²
                area_km2 = 100.0
            
            # Run all model combinations
            simulated = self.run_model_comparison(dcol, precip, area_km2)
            
            # Calculate simulated runoff coefficients
            simulated_rc = {}
            simulated_runoff_mm = {}
            
            for model_name, runoff in simulated.items():
                total_runoff_mm = runoff.sum()
                simulated_runoff_mm[model_name] = total_runoff_mm
                
                if total_precip > 0:
                    rc = total_runoff_mm / total_precip
                else:
                    rc = 0.0
                
                simulated_rc[model_name] = rc
                logger.info(f"    {model_name}: RC = {rc:.4f}")
            
            # Get observed runoff
            observed_runoff_m3s_sum = self.discharge_df[dcol].sum()
            observed_runoff_m3 = observed_runoff_m3s_sum * 3600  # Convert to volume
            observed_runoff_mm = (observed_runoff_m3 / area_m2) * 1000
            
            # Create result
            result = RunoffCoefficientResult(
                zone_id=dcol,
                observed_rc=observed_rc.get(dcol, 0.0),
                simulated_rc=simulated_rc,
                precipitation_mm=total_precip,
                observed_runoff_mm=observed_runoff_mm,
                simulated_runoff_mm=simulated_runoff_mm,
                area_km2=area_km2
            )
            
            results.append(result)
        
        self.results = {r.zone_id: r for r in results}
        return results
    
    def sensitivity_analysis(
        self,
        zone_id: str,
        model_name: str = 'HBV',
        param_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> pd.DataFrame:
        """Perform parameter sensitivity analysis"""
        logger.info(f"\nPerforming sensitivity analysis for {model_name} on {zone_id}...")
        
        if param_ranges is None:
            # Default HBV parameter ranges
            param_ranges = {
                'field_capacity': (50.0, 500.0),
                'beta': (1.0, 5.0),
                'k0': (0.01, 0.2),
                'k1': (0.001, 0.05),
                'k2': (0.0001, 0.01)
            }
        
        # Get data
        precip_cols = [col for col in self.precip_df.columns if col != self.precip_df.columns[0]]
        col_idx = [col for col in self.discharge_df.columns if col != self.discharge_df.columns[0]].index(zone_id)
        pcol = precip_cols[col_idx]
        precip = self.precip_df[pcol].values
        total_precip = precip.sum()
        
        # Get model
        model = self.runoff_models[model_name]
        
        # Sensitivity analysis results
        sensitivity_results = []
        
        # Test each parameter
        for param_name, (min_val, max_val) in param_ranges.items():
            logger.info(f"  Testing parameter: {param_name}")
            
            # Test 10 values across the range
            param_values = np.linspace(min_val, max_val, 10)
            
            for param_val in param_values:
                # Set parameter
                params = {param_name: param_val}
                
                # Run model
                runoff = model.run(precip, **params)
                total_runoff = runoff.sum()
                
                # Calculate RC
                rc = total_runoff / total_precip if total_precip > 0 else 0
                
                sensitivity_results.append({
                    'parameter': param_name,
                    'value': param_val,
                    'runoff_coefficient': rc,
                    'total_runoff_mm': total_runoff
                })
        
        df = pd.DataFrame(sensitivity_results)
        
        # Save results
        output_file = self.output_dir / f"sensitivity_{model_name}_{zone_id}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"  Sensitivity results saved: {output_file}")
        
        return df
    
    def calibrate_model(
        self,
        zone_id: str,
        model_name: str = 'HBV',
        target_rc: Optional[float] = None
    ) -> Dict[str, float]:
        """Calibrate model parameters to match observed RC"""
        logger.info(f"\nCalibrating {model_name} for {zone_id}...")
        
        # Get target RC
        if target_rc is None:
            if zone_id in self.results:
                target_rc = self.results[zone_id].observed_rc
            else:
                logger.error(f"No observed RC for {zone_id}")
                return {}
        
        logger.info(f"  Target RC: {target_rc:.4f}")
        
        # Get data
        precip_cols = [col for col in self.precip_df.columns if col != self.precip_df.columns[0]]
        discharge_cols = [col for col in self.discharge_df.columns if col != self.discharge_df.columns[0]]
        col_idx = discharge_cols.index(zone_id)
        pcol = precip_cols[col_idx]
        precip = self.precip_df[pcol].values
        total_precip = precip.sum()
        
        # Get model
        model = self.runoff_models[model_name]
        
        # Define objective function
        def objective(params_array):
            if model_name == 'HBV':
                params = {
                    'field_capacity': params_array[0],
                    'beta': params_array[1],
                    'k0': params_array[2],
                    'k1': params_array[3],
                    'k2': params_array[4]
                }
            elif model_name == 'SCS-CN':
                params = {
                    'curve_number': params_array[0]
                }
            else:
                params = {}
            
            # Run model
            try:
                runoff = model.run(precip, **params)
                total_runoff = runoff.sum()
                simulated_rc = total_runoff / total_precip if total_precip > 0 else 0
                
                # Objective: minimize difference
                error = abs(simulated_rc - target_rc)
                return error
            except:
                return 1e10  # Large penalty for invalid parameters
        
        # Parameter bounds
        if model_name == 'HBV':
            bounds = [
                (50, 500),    # field_capacity
                (1, 5),       # beta
                (0.01, 0.2),  # k0
                (0.001, 0.05), # k1
                (0.0001, 0.01) # k2
            ]
            x0 = [200, 2, 0.05, 0.01, 0.001]
        elif model_name == 'SCS-CN':
            bounds = [(40, 98)]
            x0 = [75]
        else:
            return {}
        
        # Optimize
        try:
            from scipy.optimize import differential_evolution
            
            logger.info("  Running optimization...")
            result = differential_evolution(
                objective,
                bounds,
                maxiter=100,
                popsize=15,
                tol=0.0001,
                seed=42
            )
            
            optimal_params = result.x
            optimal_error = result.fun
            
            logger.info(f"  Optimization complete!")
            logger.info(f"  Final error: {optimal_error:.6f}")
            
            # Create parameter dict
            if model_name == 'HBV':
                calibrated_params = {
                    'field_capacity': optimal_params[0],
                    'beta': optimal_params[1],
                    'k0': optimal_params[2],
                    'k1': optimal_params[3],
                    'k2': optimal_params[4]
                }
            elif model_name == 'SCS-CN':
                calibrated_params = {
                    'curve_number': optimal_params[0]
                }
            else:
                calibrated_params = {}
            
            # Verify
            runoff = model.run(precip, **calibrated_params)
            total_runoff = runoff.sum()
            final_rc = total_runoff / total_precip if total_precip > 0 else 0
            
            logger.info(f"  Calibrated RC: {final_rc:.4f} (Target: {target_rc:.4f})")
            logger.info(f"  Calibrated parameters:")
            for k, v in calibrated_params.items():
                logger.info(f"    {k} = {v:.4f}")
            
            return calibrated_params
            
        except ImportError:
            logger.error("scipy not available for optimization")
            return {}
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            return {}
    
    def generate_comparison_plots(self):
        """Generate comprehensive comparison plots"""
        if not self.plt or not self.results:
            return
        
        logger.info("\nGenerating comparison plots...")
        
        # 1. Overall RC comparison
        self._plot_rc_comparison()
        
        # 2. Model comparison by zone
        self._plot_model_comparison_by_zone()
        
        # 3. Error analysis
        self._plot_error_analysis()
        
        # 4. Best model selection
        self._plot_best_model_selection()
    
    def _plot_rc_comparison(self):
        """Plot observed vs simulated RC for all zones"""
        fig, ax = self.plt.subplots(figsize=(14, 8))
        
        zones = list(self.results.keys())
        x = np.arange(len(zones))
        width = 0.15
        
        # Observed RC
        observed_rc = [self.results[z].observed_rc for z in zones]
        ax.bar(x, observed_rc, width, label='Observed', color='darkblue', alpha=0.8)
        
        # Get all model names
        first_result = list(self.results.values())[0]
        model_names = list(first_result.simulated_rc.keys())[:5]  # First 5 models
        
        # Plot each model
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        for i, model_name in enumerate(model_names):
            simulated_rc = [self.results[z].simulated_rc.get(model_name, 0) for z in zones]
            ax.bar(x + (i+1)*width, simulated_rc, width, label=model_name, 
                   color=colors[i], alpha=0.7)
        
        ax.set_xlabel('Zone', fontsize=12)
        ax.set_ylabel('Runoff Coefficient', fontsize=12)
        ax.set_title('Observed vs Simulated Runoff Coefficients', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 2.5)
        ax.set_xticklabels(zones, rotation=45, ha='right')
        ax.legend(loc='upper right', ncol=2)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.3, label='Reference (0.5)')
        
        output_file = self.output_dir / 'rc_comparison_all_zones.png'
        self.plt.tight_layout()
        self.plt.savefig(output_file, dpi=150, bbox_inches='tight')
        self.plt.close()
        
        logger.info(f"  RC comparison plot saved: {output_file}")
    
    def _plot_model_comparison_by_zone(self):
        """Plot detailed comparison for each zone"""
        for zone_id, result in self.results.items():
            fig, (ax1, ax2) = self.plt.subplots(1, 2, figsize=(16, 6))
            
            # Left: RC comparison
            models = list(result.simulated_rc.keys())
            rc_values = [result.simulated_rc[m] for m in models]
            observed = result.observed_rc
            
            x = np.arange(len(models))
            bars = ax1.bar(x, rc_values, alpha=0.7, color='steelblue')
            ax1.axhline(y=observed, color='red', linestyle='--', linewidth=2, label=f'Observed ({observed:.4f})')
            
            # Color bars by error
            for i, (bar, rc) in enumerate(zip(bars, rc_values)):
                error = abs(rc - observed)
                if error < 0.05:
                    bar.set_color('green')
                elif error < 0.1:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
            
            ax1.set_xticks(x)
            ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
            ax1.set_ylabel('Runoff Coefficient', fontsize=11)
            ax1.set_title(f'{zone_id}: Model Comparison', fontsize=12, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Right: Error magnitude
            errors = [abs(rc - observed) for rc in rc_values]
            bars2 = ax2.bar(x, errors, alpha=0.7, color='coral')
            
            ax2.set_xticks(x)
            ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
            ax2.set_ylabel('Absolute Error', fontsize=11)
            ax2.set_title(f'{zone_id}: Model Errors', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.axhline(y=0.05, color='green', linestyle='--', alpha=0.5, label='Good (<0.05)')
            ax2.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Fair (<0.1)')
            ax2.legend()
            
            output_file = self.output_dir / f'model_comparison_{zone_id}.png'
            self.plt.tight_layout()
            self.plt.savefig(output_file, dpi=150, bbox_inches='tight')
            self.plt.close()
    
    def _plot_error_analysis(self):
        """Plot error analysis across all zones and models"""
        fig, (ax1, ax2) = self.plt.subplots(2, 1, figsize=(14, 12))
        
        # Collect all errors
        zones = list(self.results.keys())
        first_result = list(self.results.values())[0]
        models = list(first_result.simulated_rc.keys())
        
        # Heatmap data
        error_matrix = []
        for zone in zones:
            observed = self.results[zone].observed_rc
            errors = [abs(self.results[zone].simulated_rc.get(m, 0) - observed) for m in models]
            error_matrix.append(errors)
        
        error_matrix = np.array(error_matrix)
        
        # Plot heatmap
        im = ax1.imshow(error_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.2)
        ax1.set_xticks(np.arange(len(models)))
        ax1.set_yticks(np.arange(len(zones)))
        ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
        ax1.set_yticklabels(zones)
        ax1.set_title('Error Heatmap: Absolute RC Difference', fontsize=13, fontweight='bold')
        self.plt.colorbar(im, ax=ax1, label='Absolute Error')
        
        # Plot box plot of errors by model
        error_by_model = []
        for model in models:
            errors = [abs(self.results[z].simulated_rc.get(model, 0) - self.results[z].observed_rc) 
                     for z in zones]
            error_by_model.append(errors)
        
        bp = ax2.boxplot(error_by_model, labels=models, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.set_ylabel('Absolute Error', fontsize=11)
        ax2.set_title('Error Distribution by Model', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0.05, color='green', linestyle='--', alpha=0.5)
        ax2.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5)
        
        output_file = self.output_dir / 'error_analysis.png'
        self.plt.tight_layout()
        self.plt.savefig(output_file, dpi=150, bbox_inches='tight')
        self.plt.close()
        
        logger.info(f"  Error analysis plot saved: {output_file}")
    
    def _plot_best_model_selection(self):
        """Plot best model for each zone"""
        fig, ax = self.plt.subplots(figsize=(14, 8))
        
        zones = list(self.results.keys())
        best_models = []
        best_errors = []
        
        for zone in zones:
            observed = self.results[zone].observed_rc
            simulated = self.results[zone].simulated_rc
            
            # Find best model
            errors = {m: abs(rc - observed) for m, rc in simulated.items()}
            best_model = min(errors, key=errors.get)
            best_error = errors[best_model]
            
            best_models.append(best_model)
            best_errors.append(best_error)
        
        # Plot
        x = np.arange(len(zones))
        colors = ['green' if e < 0.05 else 'orange' if e < 0.1 else 'red' for e in best_errors]
        bars = ax.bar(x, best_errors, color=colors, alpha=0.7, edgecolor='black')
        
        # Add model names on bars
        for i, (bar, model) in enumerate(zip(bars, best_models)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   model.split('+')[0][:10],  # Truncate long names
                   ha='center', va='bottom', fontsize=8, rotation=45)
        
        ax.set_xticks(x)
        ax.set_xticklabels(zones, rotation=45, ha='right')
        ax.set_ylabel('Best Model Error', fontsize=12)
        ax.set_title('Best Model Selection by Zone', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0.05, color='green', linestyle='--', alpha=0.5, label='Excellent (<0.05)')
        ax.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Good (<0.1)')
        ax.legend()
        
        output_file = self.output_dir / 'best_model_selection.png'
        self.plt.tight_layout()
        self.plt.savefig(output_file, dpi=150, bbox_inches='tight')
        self.plt.close()
        
        logger.info(f"  Best model selection plot saved: {output_file}")
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        logger.info("\n生成分析报告...")
        
        report_path = self.output_dir / 'runoff_coefficient_analysis_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 径流系数对比分析报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 分析概要\n\n")
            f.write(f"- 分析分区数量: {len(self.results)}\n")
            f.write(f"- 产流模型: {', '.join(self.runoff_models.keys())}\n")
            f.write(f"- 汇流模型: {', '.join(self.routing_models.keys())}\n\n")
            
            f.write("## 观测径流系数\n\n")
            f.write("| 分区 | 观测RC | 降雨量(mm) | 径流量(mm) | 面积(km²) |\n")
            f.write("|------|--------|-----------|-----------|----------|\n")
            
            for zone_id, result in self.results.items():
                f.write(f"| {zone_id} | {result.observed_rc:.4f} | ")
                f.write(f"{result.precipitation_mm:.2f} | {result.observed_runoff_mm:.2f} | ")
                f.write(f"{result.area_km2:.2f} |\n")
            
            f.write("\n## 各分区模型性能\n\n")
            
            for zone_id, result in self.results.items():
                f.write(f"### {zone_id}\n\n")
                f.write(f"**观测径流系数**: {result.observed_rc:.4f}\n\n")
                f.write("| 模型 | 模拟RC | 误差 | 状态 |\n")
                f.write("|------|--------|------|------|\n")
                
                for model_name, sim_rc in sorted(result.simulated_rc.items(), 
                                                 key=lambda x: abs(x[1] - result.observed_rc)):
                    error = abs(sim_rc - result.observed_rc)
                    if error < 0.05:
                        status = "✅ 优秀"
                    elif error < 0.1:
                        status = "✓ 良好"
                    elif error < 0.2:
                        status = "⚠ 一般"
                    else:
                        status = "❌ 较差"
                    
                    f.write(f"| {model_name} | {sim_rc:.4f} | {error:.4f} | {status} |\n")
                
                f.write("\n")
            
            f.write("## 模型选择建议\n\n")
            
            # Find best performing models overall
            model_errors = {}
            for result in self.results.values():
                for model_name, sim_rc in result.simulated_rc.items():
                    if model_name not in model_errors:
                        model_errors[model_name] = []
                    error = abs(sim_rc - result.observed_rc)
                    model_errors[model_name].append(error)
            
            # Calculate mean errors
            mean_errors = {m: np.mean(errors) for m, errors in model_errors.items()}
            best_models = sorted(mean_errors.items(), key=lambda x: x[1])[:3]
            
            f.write("### 总体最佳模型\n\n")
            for i, (model, error) in enumerate(best_models, 1):
                f.write(f"{i}. **{model}**: 平均误差 = {error:.4f}\n")
            
            f.write("\n### 分区专用建议\n\n")
            for zone_id, result in self.results.items():
                best_model = min(result.simulated_rc.items(), 
                               key=lambda x: abs(x[1] - result.observed_rc))
                f.write(f"- **{zone_id}**: 推荐使用 {best_model[0]} (误差 = {abs(best_model[1] - result.observed_rc):.4f})\n")
            
            f.write("\n## 问题分析\n\n")
            
            # 分析常见问题
            high_error_zones = [z for z, r in self.results.items() 
                               if min(abs(rc - r.observed_rc) for rc in r.simulated_rc.values()) > 0.1]
            
            if high_error_zones:
                f.write("### 高误差分区\n\n")
                f.write(f"以下分区所有模型误差均较大（>0.1）：\n\n")
                for zone in high_error_zones:
                    result = self.results[zone]
                    f.write(f"- **{zone}**: 观测RC={result.observed_rc:.4f}\n")
                    best_sim = min(result.simulated_rc.items(), key=lambda x: abs(x[1] - result.observed_rc))
                    f.write(f"  - 最佳模型: {best_sim[0]}, RC={best_sim[1]:.4f}, 误差={abs(best_sim[1]-result.observed_rc):.4f}\n")
                
                f.write("\n**可能原因**：\n")
                f.write("1. 观测数据存在问题（面积计算、单位转换等）\n")
                f.write("2. 该分区特殊水文过程未被模型捕捉\n")
                f.write("3. 需要进一步参数率定\n")
                f.write("4. 可能需要考虑其他损失项（如蒸发、渗漏）\n\n")
            
            # 分析模型系统性偏差
            f.write("### 模型偏差分析\n\n")
            for model_name in list(self.runoff_models.keys())[:3]:  # 主要产流模型
                errors = [result.simulated_rc.get(model_name, 0) - result.observed_rc 
                         for result in self.results.values()]
                mean_bias = np.mean(errors)
                
                f.write(f"**{model_name}模型**:\n")
                if abs(mean_bias) < 0.05:
                    f.write(f"- 平均偏差: {mean_bias:+.4f} (无明显系统性偏差)\n")
                elif mean_bias > 0:
                    f.write(f"- 平均偏差: {mean_bias:+.4f} (系统性高估)\n")
                    f.write(f"- 建议: 减小产流参数或增大土壤蓄水容量\n")
                else:
                    f.write(f"- 平均偏差: {mean_bias:+.4f} (系统性低估)\n")
                    f.write(f"- 建议: 增大产流参数或减小土壤蓄水容量\n")
                f.write("\n")
        
        logger.info(f"  Report saved: {report_path}")
        
        # Save JSON results
        json_path = self.output_dir / 'analysis_results.json'
        json_data = {}
        for zone_id, result in self.results.items():
            json_data[zone_id] = {
                'observed_rc': result.observed_rc,
                'simulated_rc': result.simulated_rc,
                'precipitation_mm': result.precipitation_mm,
                'observed_runoff_mm': result.observed_runoff_mm,
                'simulated_runoff_mm': result.simulated_runoff_mm,
                'area_km2': result.area_km2
            }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
        
        logger.info(f"  JSON results saved: {json_path}")


def main():
    """Main function"""
    logger.info("=" * 80)
    logger.info("Runoff Coefficient Analysis and Calibration")
    logger.info("=" * 80)
    
    # Setup paths
    output_dir = Path("results/runoff_coefficient_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Example data paths (adjust as needed)
    test_data_dir = Path("results/test_data_for_rc_analysis")
    precip_file = test_data_dir / "precipitation_timeseries.csv"
    discharge_file = test_data_dir / "discharge_timeseries.csv"
    watershed_file = test_data_dir / "watersheds.geojson"
    
    # Check if files exist
    if not precip_file.exists():
        logger.error(f"Precipitation file not found: {precip_file}")
        logger.info("Please run the comprehensive test scenarios first:")
        logger.info("  python3 run_comprehensive_test_scenarios.py")
        return 1
    
    # Create analyzer
    analyzer = RunoffCoefficientAnalyzer(output_dir)
    
    # Load data
    analyzer.load_data(precip_file, discharge_file, watershed_file)
    
    # Analyze all zones with all models
    results = analyzer.analyze_all_zones()
    
    # Perform sensitivity analysis for first zone
    if results:
        first_zone = results[0].zone_id
        analyzer.sensitivity_analysis(first_zone, 'HBV')
    
    # Calibrate HBV for each zone
    for result in results[:3]:  # First 3 zones
        analyzer.calibrate_model(result.zone_id, 'HBV')
    
    # Generate comparison plots
    analyzer.generate_comparison_plots()
    
    # Generate report
    analyzer.generate_report()
    
    logger.info("\n" + "=" * 80)
    logger.info("Analysis complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
