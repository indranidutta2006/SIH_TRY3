"""Multi-Scenario Analysis and Macro-Economic Stress Testing package.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 4 Deliverable: Evaluates fleet operational resilience across macro scenarios.
"""

from src.scenarios.scenario_comparison import (
    DEFAULT_SCENARIOS,
    MultiScenarioAnalyzer,
)

__all__ = [
    "DEFAULT_SCENARIOS",
    "MultiScenarioAnalyzer",
]
