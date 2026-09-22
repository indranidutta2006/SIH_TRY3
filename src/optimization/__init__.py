"""Optimization subsystem package for quantum-inspired and heuristic fleet optimization.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Exports QPSOOptimizer, PSOOptimizer, and SwarmOptimizationEngine.
"""

from src.optimization.base import SwarmOptimizationEngine
from src.optimization.pso_baseline import PSOOptimizer
from src.optimization.qpso import QPSOOptimizer

__all__ = [
    "SwarmOptimizationEngine",
    "QPSOOptimizer",
    "PSOOptimizer",
]
