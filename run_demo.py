"""Phase 0 repository verification script.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Verifies foundational repository integrity by asserting contract imports directly from contracts/,
configuration loading, and logging subsystem initialization without invoking business logic.
"""

import sys

# 1. System Configuration & Logging
import config
import contracts
from contracts import (
    CONTRACT_VERSION,
    ComplianceEngine,
    ComplianceError,
    ComplianceResult,
    DatasetLoader,
    DataValidationError,
    EmissionEngine,
    EmissionResult,
    FleetAssignment,
    FuelPhysicsEngine,
    MaritimeSystemError,
    OptimizationEngine,
    OptimizationResult,
    PredictionEngine,
    PredictionResult,
    ScenarioEngine,
    ScenarioResult,
    SchedulerEngine,
    VoyageRecord,
    constants,
    exceptions,
    interfaces,
    schemas,
)
import logging_config


def main() -> int:
    """Execute Phase 0 architectural health check.

    Returns:
        Integer exit code (0 on success).
    """
    # 2. Subsystem Initialization
    logger = logging_config.configure_logging(log_file_name="phase0_verification.log")
    logger.info("Initializing Phase 0 Repository Health Verification")

    # 3. Assert schemas contract presence
    assert VoyageRecord is not None
    assert PredictionResult is not None
    assert EmissionResult is not None
    assert ComplianceResult is not None
    assert FleetAssignment is not None
    assert OptimizationResult is not None
    assert ScenarioResult is not None

    # 4. Assert interfaces contract presence
    assert DatasetLoader is not None
    assert PredictionEngine is not None
    assert FuelPhysicsEngine is not None
    assert EmissionEngine is not None
    assert ComplianceEngine is not None
    assert SchedulerEngine is not None
    assert OptimizationEngine is not None
    assert ScenarioEngine is not None

    # 5. Assert config, constants & version contracts
    assert config.SystemConfig is not None
    assert contracts.CONTRACT_VERSION == "1.0.0"
    assert len(constants.SUPPORTED_FUELS) == 6
    assert len(constants.SUPPORTED_OPTIMIZERS) == 3
    assert len(constants.SUPPORTED_PREDICTION_MODELS) == 3

    # 6. Assert exceptions hierarchy contract
    assert MaritimeSystemError is not None
    assert DataValidationError is not None
    assert exceptions.PredictionError is not None

    logger.info("All architectural interfaces and schemas successfully verified via contracts/.")

    # 7. Standard verification confirmation
    print("System initialized successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
