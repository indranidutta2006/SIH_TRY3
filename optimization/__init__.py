"""Root package bridge for optimization modules.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Re-exports optimization models and engines from src.optimization.
"""

from src.optimization.capacity_optimizer import VesselCapacityOptimizer
from src.optimization.fleet_composition_optimizer import FleetCompositionOptimizer
from src.optimization.fleet_strategy_optimizer import FleetStrategyOptimizer
from src.optimization.speed_optimizer import EcoSpeedOptimizer

__all__ = [
    "FleetCompositionOptimizer",
    "VesselCapacityOptimizer",
    "EcoSpeedOptimizer",
    "FleetStrategyOptimizer",
]
