"""Data ingestion, validation, and feature engineering subsystem.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 1B operational pipeline converting raw voyage telemetry into model-ready datasets.
"""

from src.ingestion.dataset_loader import CSVDatasetLoader
from src.ingestion.feature_pipeline import FeatureEngineeringPipeline
from src.ingestion.validators import ValidationEngine, ValidationResult

__all__ = [
    "CSVDatasetLoader",
    "ValidationEngine",
    "ValidationResult",
    "FeatureEngineeringPipeline",
]
