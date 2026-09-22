"""Compliance subsystem package for IMO CII and EU FuelEU Maritime regulations.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Exports MaritimeComplianceEngine.
"""

from src.compliance.compliance_engine import (
    FUELEU_REFERENCE_GHG_INTENSITY,
    FUELEU_STATUTORY_PENALTY_RATE_EUR,
    VLSFO_ENERGY_DENSITY_MJ_PER_TON,
    MaritimeComplianceEngine,
)

__all__ = [
    "MaritimeComplianceEngine",
    "FUELEU_REFERENCE_GHG_INTENSITY",
    "VLSFO_ENERGY_DENSITY_MJ_PER_TON",
    "FUELEU_STATUTORY_PENALTY_RATE_EUR",
]
