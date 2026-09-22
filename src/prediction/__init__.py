"""Baseline fuel consumption prediction and green fleet optimization subsystem.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 2A & 2B components establishing benchmark regressors, metrics evaluation,
model training, explainability, carbon emissions accounting, and optimization recommendations.
"""

from src.prediction.emission_engine import (
    CH4_FACTORS,
    CO2_FACTORS,
    FUEL_EMISSION_FACTORS,
    N2O_FACTORS,
    WTT_CO2E_FACTORS,
    CarbonEmissionEngine,
    CarbonEmissionResult,
    EmissionResult,
    MaritimeEmissionEngine,
    normalize_fuel_type,
)
from src.prediction.explainability import PredictionExplainer
from src.prediction.metrics import (
    ModelComparisonResult,
    ModelMetrics,
    compute_mae,
    compute_mape,
    compute_r2,
    compute_rmse,
    evaluate_predictions,
    select_best_model,
)
from src.prediction.model_manager import ProductionModelManager
from src.prediction.model_registry import (
    DEFAULT_RANDOM_STATE,
    ModelRegistry,
    normalize_model_name,
)
from src.prediction.predictor import PredictionInferenceEngine
from src.prediction.qifcp import QIFCPRegressor
from src.prediction.recommendation_engine import (
    OptimizationRecommendation,
    VoyageOptimizationEngine,
)
from src.prediction.trainer import PredictionTrainer, train_all_models

__all__ = [
    "QIFCPRegressor",
    "ModelRegistry",
    "PredictionTrainer",
    "PredictionInferenceEngine",
    "ModelMetrics",
    "ModelComparisonResult",
    "train_all_models",
    "compute_mae",
    "compute_rmse",
    "compute_mape",
    "compute_r2",
    "evaluate_predictions",
    "select_best_model",
    "DEFAULT_RANDOM_STATE",
    "normalize_model_name",
    "MaritimeEmissionEngine",
    "CarbonEmissionEngine",
    "EmissionResult",
    "CarbonEmissionResult",
    "CO2_FACTORS",
    "CH4_FACTORS",
    "N2O_FACTORS",
    "WTT_CO2E_FACTORS",
    "FUEL_EMISSION_FACTORS",
    "normalize_fuel_type",
    "PredictionExplainer",
    "VoyageOptimizationEngine",
    "OptimizationRecommendation",
    "ProductionModelManager",
]
