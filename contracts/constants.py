"""Central constants for the maritime fuel prediction and fleet optimization system.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
All categorical tokens, supported fuel types, optimizers, and model architectures
are centrally defined here to avoid hardcoding strings across downstream modules.
"""

from enum import StrEnum
from typing import Final


class FuelType(StrEnum):
    """Enumeration of supported marine fuel types and alternative power sources."""

    DIESEL = "Diesel"
    LNG = "LNG"
    METHANOL = "Methanol"
    HYDROGEN = "Hydrogen"
    AMMONIA = "Ammonia"
    SHORE_POWER = "ShorePower"


class OptimizerType(StrEnum):
    """Enumeration of supported fleet assignment and route optimization algorithms."""

    PSO = "PSO"
    QPSO = "QPSO"
    NSGAII = "NSGAII"


class ModelType(StrEnum):
    """Enumeration of supported maritime fuel consumption prediction models."""

    LINEAR_REGRESSION = "LinearRegression"
    XGBOOST = "XGBoost"
    QIFCP = "QIFCP"


SUPPORTED_FUELS: Final[tuple[FuelType, ...]] = (
    FuelType.DIESEL,
    FuelType.LNG,
    FuelType.METHANOL,
    FuelType.HYDROGEN,
    FuelType.AMMONIA,
    FuelType.SHORE_POWER,
)

SUPPORTED_OPTIMIZERS: Final[tuple[OptimizerType, ...]] = (
    OptimizerType.PSO,
    OptimizerType.QPSO,
    OptimizerType.NSGAII,
)

SUPPORTED_PREDICTION_MODELS: Final[tuple[ModelType, ...]] = (
    ModelType.LINEAR_REGRESSION,
    ModelType.XGBOOST,
    ModelType.QIFCP,
)
