"""Optimization subsystem package for quantum-inspired and heuristic fleet optimization.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Exports QPSOOptimizer, PSOOptimizer, and SwarmOptimizationEngine.
"""

from src.optimization.base import SwarmOptimizationEngine
from src.optimization.capacity_optimizer import VesselCapacityOptimizer
from src.optimization.fleet_composition_optimizer import FleetCompositionOptimizer
from src.optimization.fleet_strategy_optimizer import FleetStrategyOptimizer
from src.optimization.nsga2_pareto import (
    ParetoFleetOptimizer,
    calculate_crowding_distance,
    non_dominated_sort,
    run_nsga2_pareto,
)
from src.optimization.pso_baseline import PSOOptimizer
from src.optimization.qpso import QPSOOptimizer
from src.optimization.speed_optimizer import EcoSpeedOptimizer

__all__ = [
    "SwarmOptimizationEngine",
    "QPSOOptimizer",
    "PSOOptimizer",
    "ParetoFleetOptimizer",
    "calculate_crowding_distance",
    "non_dominated_sort",
    "run_nsga2_pareto",
    "FleetCompositionOptimizer",
    "VesselCapacityOptimizer",
    "EcoSpeedOptimizer",
    "FleetStrategyOptimizer",
]

