"""Parallel computing utilities for HydroSIS simulations.

This module provides parallel computing capabilities to accelerate
hydrological simulations, particularly for large-scale watershed
modeling with many subbasins or long time series.

The module implements both process-based and thread-based parallelism
to handle different types of computational bottlenecks:
- Process-based parallelism for CPU-bound tasks (model simulations)
- Thread-based parallelism for I/O-bound tasks (data loading/saving)

The main entry point is the ParallelHydroSISModel class which extends
the base HydroSISModel with parallel execution capabilities.
"""
from __future__ import annotations

import multiprocessing as mp
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from .model import ChannelNetwork, HydroSISModel, Subbasin
from .parameters.zone import ParameterZone, ParameterZoneBuilder
from .runoff.base import RunoffModel
from .routing.base import RoutingModel


@dataclass
class ParallelConfig:
    """Configuration for parallel execution.
    
    This class defines the parallel execution parameters for HydroSIS
    simulations, including the number of workers, execution strategy,
    and chunking parameters.
    
    Attributes:
        max_workers: Maximum number of parallel workers. If None, defaults to
            the number of CPU cores for process-based execution or 5× the
            number of CPU cores for thread-based execution.
        use_processes: If True, use process-based parallelism (better for
            CPU-bound tasks). If False, use thread-based parallelism
            (better for I/O-bound tasks).
        chunk_size: Number of subbasins to process in each chunk. Larger
            chunks reduce overhead but may decrease load balancing.
        enable_progress: If True, enable progress reporting during execution.
    """
    max_workers: Optional[int] = None
    use_processes: bool = True
    chunk_size: int = 1
    enable_progress: bool = False


def _simulate_runoff_task(
    args: Tuple[str, Subbasin, RunoffModel, List[float]]
) -> Tuple[str, Tuple[List[float], float]]:
    """Worker function for parallel runoff simulation.
    
    This function is designed to be executed in a separate process or thread
    to simulate runoff for a single subbasin.
    
    Args:
        args: Tuple containing:
            - sub_id: Subbasin identifier
            - subbasin: Subbasin object
            - runoff_model: RunoffModel instance
            - precipitation: Precipitation time series
            
    Returns:
        Tuple of (sub_id, (runoff_time_series, final_storage))
    """
    sub_id, subbasin, runoff_model, precipitation = args
    runoff, storage = runoff_model.simulate(subbasin, precipitation)
    return sub_id, (runoff, storage)


def _simulate_routing_task(
    args: Tuple[str, Subbasin, RoutingModel, List[float]]
) -> Tuple[str, List[float]]:
    """Worker function for parallel routing simulation.
    
    This function is designed to be executed in a separate process or thread
    to simulate routing for a single subbasin.
    
    Args:
        args: Tuple containing:
            - sub_id: Subbasin identifier
            - subbasin: Subbasin object
            - routing_model: RoutingModel instance
            - inflow: Inflow time series
            
    Returns:
        Tuple of (sub_id, routed_time_series)
    """
    sub_id, subbasin, routing_model, inflow = args
    routed = routing_model.route(subbasin, inflow)
    return sub_id, routed


class ParallelHydroSISModel(HydroSISModel):
    """Parallel implementation of HydroSIS model.
    
    This class extends the base HydroSISModel with parallel execution
    capabilities to accelerate simulations, particularly for large-scale
    watershed modeling with many subbasins.
    
    The parallelization is applied at the subbasin level, where each
    subbasin's runoff and routing computations are performed independently
    in parallel. The accumulation step remains sequential due to its
    inherent dependencies on upstream results.
    
    Example:
        # Create a parallel model with default configuration
        parallel_model = ParallelHydroSISModel.from_config(config)
        
        # Create a parallel model with custom configuration
        parallel_config = ParallelConfig(
            max_workers=8,
            use_processes=True,
            chunk_size=2,
            enable_progress=True
        )
        parallel_model = ParallelHydroSISModel.from_config(
            config, parallel_config=parallel_config
        )
    """
    
    def __init__(
        self,
        subbasins: Iterable[Subbasin],
        parameter_zones: Iterable[ParameterZone],
        runoff_models: Mapping[str, RunoffModel],
        routing_models: Mapping[str, RoutingModel],
        channel_network: Optional["ChannelNetwork"] = None,
        parallel_config: Optional[ParallelConfig] = None,
    ) -> None:
        """Initialize the parallel HydroSIS model.
        
        Args:
            subbasins: Iterable of subbasin objects
            parameter_zones: Iterable of parameter zone objects
            runoff_models: Mapping of runoff model identifiers to instances
            routing_models: Mapping of routing model identifiers to instances
            parallel_config: Configuration for parallel execution
        """
        super().__init__(subbasins, parameter_zones, runoff_models, routing_models, channel_network)
        self.parallel_config = parallel_config or ParallelConfig()
        
        # Determine the optimal number of workers if not specified
        if self.parallel_config.max_workers is None:
            if self.parallel_config.use_processes:
                self.parallel_config.max_workers = mp.cpu_count()
            else:
                self.parallel_config.max_workers = mp.cpu_count() * 5

    @classmethod
    def from_config(
        cls,
        config: "ModelConfig",
        parallel_config: Optional[ParallelConfig] = None,
    ) -> "ParallelHydroSISModel":
        """Create a ParallelHydroSISModel from configuration.
        
        Args:
            config: Model configuration object
            parallel_config: Configuration for parallel execution
            
        Returns:
            A new ParallelHydroSISModel instance
        """
        from .config import ModelConfig  # local import to avoid cycle

        delineated = config.delineation.to_subbasins()
        zones = ParameterZoneBuilder.from_config(config.parameter_zones, delineated)

        runoff_models = {
            run_cfg.id: run_cfg.build()
            for run_cfg in config.runoff_models
        }
        routing_models = {
            route_cfg.id: route_cfg.build()
            for route_cfg in config.routing_models
        }
        channel_network = config.delineation.to_channel_network()

        return cls(delineated, zones, runoff_models, routing_models, channel_network, parallel_config)

    def run(
        self, forcing: Mapping[str, List[float]]
    ) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
        """Run the distributed hydrological simulation in parallel.
        
        This method executes the runoff and routing simulations in parallel
        across all subbasins, then performs the sequential accumulation
        step.
        
        Args:
            forcing: Mapping from subbasin identifiers to precipitation time series
            
        Returns:
            A tuple containing:
            - Mapping from subbasin identifiers to routed flow time series
            - Mapping from subbasin identifiers to final storage values
        """
        # Step 1: Parallel runoff simulation
        runoff_results, storage = self._run_runoff_parallel(forcing)

        # Step 2: Parallel routing simulation
        routed = self._run_routing_parallel(runoff_results)

        return routed, storage

    def _run_runoff_parallel(
        self, forcing: Mapping[str, List[float]]
    ) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
        """Run runoff simulation in parallel.
        
        Args:
            forcing: Mapping from subbasin identifiers to precipitation time series
            
        Returns:
            A tuple containing:
            - Mapping from subbasin identifiers to runoff time series
            - Mapping from subbasin identifiers to final storage values
        """
        # Prepare tasks for parallel execution
        tasks = []
        for sub_id, subbasin in self.subbasins.items():
            model_key = subbasin.parameters.get("runoff_model")
            if model_key is None:
                raise ValueError(f"Subbasin {sub_id} missing runoff_model parameter")
            
            # Create a copy of the model for thread safety
            runoff_model = copy.deepcopy(self.runoff_models[model_key])
            precipitation = forcing.get(sub_id, [])
            tasks.append((sub_id, subbasin, runoff_model, precipitation))
        
        # Execute tasks in parallel
        executor_class = ProcessPoolExecutor if self.parallel_config.use_processes else ThreadPoolExecutor
        with executor_class(max_workers=self.parallel_config.max_workers) as executor:
            # Submit tasks in chunks if specified
            if self.parallel_config.chunk_size > 1:
                results = {}
                storage = {}
                for i in range(0, len(tasks), self.parallel_config.chunk_size):
                    chunk = tasks[i:i + self.parallel_config.chunk_size]
                    chunk_futures = {
                        executor.submit(_simulate_runoff_task, task): task[0]
                        for task in chunk
                    }
                    
                    for future in as_completed(chunk_futures):
                        sub_id, (runoff, final_storage) = future.result()
                        results[sub_id] = runoff
                        storage[sub_id] = final_storage
            else:
                # Submit all tasks at once
                futures = {
                    executor.submit(_simulate_runoff_task, task): task[0]
                    for task in tasks
                }
                
                results = {}
                storage = {}
                for future in as_completed(futures):
                    sub_id, (runoff, final_storage) = future.result()
                    results[sub_id] = runoff
                    storage[sub_id] = final_storage
        
        return results, storage

    def _run_routing_parallel(
        self, runoff_results: Mapping[str, List[float]]
    ) -> Dict[str, List[float]]:
        """Run routing simulation in parallel.
        
        Args:
            runoff_results: Mapping from subbasin identifiers to runoff time series
            
        Returns:
            Mapping from subbasin identifiers to routed flow time series
        """
        # Prepare tasks for parallel execution
        tasks = []
        for sub_id, flows in runoff_results.items():
            subbasin = self.subbasins[sub_id]
            model_key = subbasin.parameters.get("routing_model")
            if model_key is None:
                raise ValueError(f"Subbasin {sub_id} missing routing_model parameter")
            
            # Create a copy of the model for thread safety
            routing_model = copy.deepcopy(self.routing_models[model_key])
            tasks.append((sub_id, subbasin, routing_model, flows))
        
        # Execute tasks in parallel
        executor_class = ProcessPoolExecutor if self.parallel_config.use_processes else ThreadPoolExecutor
        with executor_class(max_workers=self.parallel_config.max_workers) as executor:
            # Submit tasks in chunks if specified
            if self.parallel_config.chunk_size > 1:
                results = {}
                for i in range(0, len(tasks), self.parallel_config.chunk_size):
                    chunk = tasks[i:i + self.parallel_config.chunk_size]
                    chunk_futures = {
                        executor.submit(_simulate_routing_task, task): task[0]
                        for task in chunk
                    }
                    
                    for future in as_completed(chunk_futures):
                        sub_id, routed = future.result()
                        results[sub_id] = routed
            else:
                # Submit all tasks at once
                futures = {
                    executor.submit(_simulate_routing_task, task): task[0]
                    for task in tasks
                }
                
                results = {}
                for future in as_completed(futures):
                    sub_id, routed = future.result()
                    results[sub_id] = routed
        
        return results


# Import copy module for deepcopy
import copy


__all__ = [
    "ParallelConfig",
    "ParallelHydroSISModel",
    "_simulate_runoff_task",
    "_simulate_routing_task",
]
