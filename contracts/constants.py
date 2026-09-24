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
    RANDOM_FOREST = "RandomForest"
    HIST_GRADIENT_BOOSTING = "HistGradientBoosting"
    QIFCP = "QIFCP"
    XGBOOST = "XGBoost"  # Contract alias for tree-based GBDT (standardized on HistGradientBoosting)


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
    ModelType.RANDOM_FOREST,
    ModelType.HIST_GRADIENT_BOOSTING,
    ModelType.QIFCP,
)

# Financial & Economic Baseline Assumptions:
# Statutory FuelEU penalties are denominated in EUR (€) under Regulation (EU) 2023/1805 Article 23.
# Bunker fuel spot/contract prices are denominated in USD ($) per metric ton.
# Configurable baseline foreign exchange rate (1 EUR = 1.08 USD).
DEFAULT_EUR_TO_USD_FX_RATE: Final[float] = 1.08

# Standard commercial bunker fuel prices (USD per metric ton)
FUEL_PRICES_USD_PER_TON: Final[dict[str, float]] = {
    FuelType.DIESEL.value: 650.0,
    FuelType.LNG.value: 800.0,
    FuelType.METHANOL.value: 950.0,
    FuelType.HYDROGEN.value: 2500.0,
    FuelType.AMMONIA.value: 1200.0,
}

# Cold-ironing shore power grid electricity tariff (USD per megawatt-hour)
SHORE_POWER_PRICE_USD_PER_MWH: Final[float] = 300.0

# Legacy compatibility mapping
STANDARD_FUEL_PRICES_USD: Final[dict[str, float]] = {
    **FUEL_PRICES_USD_PER_TON,
    FuelType.SHORE_POWER.value: SHORE_POWER_PRICE_USD_PER_MWH,
}


