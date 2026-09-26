"""Operations root shim module.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Redirects to src.operations to support top-level imports.
"""

from src.operations import (
    CargoDemandSatisfactionEngine,
    ScheduleReliabilityEngine,
)

__all__ = [
    "CargoDemandSatisfactionEngine",
    "ScheduleReliabilityEngine",
]
