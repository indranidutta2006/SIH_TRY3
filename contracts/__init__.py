"""Contracts package defining all project schemas, interfaces, constants, and exceptions.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Serves as the single source of truth for architecture contracts, preventing circular dependencies.
"""

from contracts.version import CONTRACT_VERSION
from contracts.constants import (
    FuelType,
    ModelType,
    OptimizerType,
    SUPPORTED_FUELS,
    SUPPORTED_OPTIMIZERS,
    SUPPORTED_PREDICTION_MODELS,
)
from contracts.exceptions import (
    ComplianceError,
    DataValidationError,
    IntegrationError,
    MaritimeSystemError,
    OptimizationError,
    PredictionError,
    SchedulerError,
)
from contracts.interfaces import (
    ComplianceEngine,
    DatasetLoader,
    EmissionEngine,
    FuelPhysicsEngine,
    OptimizationEngine,
    PredictionEngine,
    ScenarioEngine,
    SchedulerEngine,
)
from contracts.schemas import (
    CIIResult,
    ComplianceAssessment,
    ComplianceResult,
    EmissionResult,
    FleetAssignment,
    FuelEUResult,
    OptimizationResult,
    PredictionResult,
    ScenarioResult,
    VoyageRecord,
)

__all__ = [
    # Versioning
    "CONTRACT_VERSION",
    # Constants
    "FuelType",
    "OptimizerType",
    "ModelType",
    "SUPPORTED_FUELS",
    "SUPPORTED_OPTIMIZERS",
    "SUPPORTED_PREDICTION_MODELS",
    # Exceptions
    "MaritimeSystemError",
    "DataValidationError",
    "PredictionError",
    "OptimizationError",
    "SchedulerError",
    "ComplianceError",
    "IntegrationError",
    # Interfaces
    "DatasetLoader",
    "PredictionEngine",
    "FuelPhysicsEngine",
    "EmissionEngine",
    "ComplianceEngine",
    "SchedulerEngine",
    "OptimizationEngine",
    "ScenarioEngine",
    # Schemas
    "VoyageRecord",
    "PredictionResult",
    "EmissionResult",
    "ComplianceResult",
    "CIIResult",
    "FuelEUResult",
    "ComplianceAssessment",
    "FleetAssignment",
    "OptimizationResult",
    "ScenarioResult",
]
