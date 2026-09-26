"""Central constants for the maritime fuel prediction and fleet optimization system.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
All categorical tokens, supported fuel types, optimizers, and model architectures
are centrally defined here to avoid hardcoding strings across downstream modules.
"""

from enum import StrEnum
from typing import Any, Final


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

# Standard naval architectural baseline specifications by vessel class
VESSEL_CLASS_SPECS: Final[dict[str, dict[str, Any]]] = {
    "HANDYMAX": {
        "vessel_type": "Bulk carrier",
        "default_dwt": 35_000.0,
        "min_dwt": 20_000.0,
        "max_dwt": 40_000.0,
        "default_annual_voyages": 25,
    },
    "PANAMAX": {
        "vessel_type": "Bulk carrier",
        "default_dwt": 45_000.0,
        "min_dwt": 25_000.0,
        "max_dwt": 55_000.0,
        "default_annual_voyages": 20,
    },
    "CAPESIZE": {
        "vessel_type": "Bulk carrier",
        "default_dwt": 120_000.0,
        "min_dwt": 100_000.0,
        "max_dwt": 200_000.0,
        "default_annual_voyages": 10,
    },
    "FEEDER": {
        "vessel_type": "Containership",
        "default_dwt": 12_000.0,
        "min_dwt": 5_000.0,
        "max_dwt": 15_000.0,
        "default_annual_voyages": 35,
    },
    "POST_PANAMAX": {
        "vessel_type": "Containership",
        "default_dwt": 80_000.0,
        "min_dwt": 60_000.0,
        "max_dwt": 100_000.0,
        "default_annual_voyages": 15,
    },
}


