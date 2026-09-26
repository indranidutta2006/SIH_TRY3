"""Operations root shim module.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Redirects to src.operations to support top-level imports.
"""

from src.operations import (
    CargoDemandSatisfactionEngine,
    DemandSatisfactionEngine,
    ScheduleReliabilityEngine,
)

__all__ = [
    "CargoDemandSatisfactionEngine",
    "DemandSatisfactionEngine",
    "ScheduleReliabilityEngine",
]
