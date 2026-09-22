"""Contract validation tests for Phase 0 architecture.

Ensures that all schemas, abstract interfaces, configuration containers,
and constants conform to contract specifications, serialize cleanly to JSON,
and have no circular dependencies.
"""

from dataclasses import asdict, is_dataclass
import inspect
import json
import logging
from pathlib import Path
import pytest

import config
from contracts import (
    ComplianceEngine,
    ComplianceError,
    ComplianceResult,
    DatasetLoader,
    DataValidationError,
    EmissionEngine,
    EmissionResult,
    FleetAssignment,
    FuelPhysicsEngine,
    FuelType,
    IntegrationError,
    MaritimeSystemError,
    ModelType,
    OptimizationEngine,
    OptimizationError,
    OptimizationResult,
    OptimizerType,
    PredictionEngine,
    PredictionError,
    PredictionResult,
    ScenarioEngine,
    ScenarioResult,
    SchedulerEngine,
    SchedulerError,
    VoyageRecord,
    constants,
    exceptions,
    interfaces,
    schemas,
)
import logging_config


def test_schemas_are_dataclasses() -> None:
    """Verify all domain models are defined as Python dataclasses."""
    schema_classes = [
        VoyageRecord,
        PredictionResult,
        EmissionResult,
        ComplianceResult,
        FleetAssignment,
        OptimizationResult,
        ScenarioResult,
    ]
    for cls in schema_classes:
        assert is_dataclass(cls), f"{cls.__name__} must be a dataclass."


def test_dataclasses_json_serialization() -> None:
    """Verify that all schema dataclasses cleanly serialize to and from JSON."""
    voyage = VoyageRecord(
        voyage_id="VY-001",
        vessel_id="VSL-Alpha",
        vessel_type="Container",
        vessel_dwt=65000.0,
        cargo_tons=45000.0,
        distance_nm=1200.0,
        speed_knots=14.5,
        hours_at_sea=82.75,
        fuel_type=FuelType.DIESEL.value,
        weather_factor=1.15,
        sea_state=4,
        data_source="fuelcast",
        is_synthetic=False,
        fuel_consumption=128.5,
        co2_emissions=404.7,
    )
    prediction = PredictionResult(
        model_name="QIFCP",
        predicted_fuel_consumption=126.8,
        confidence_score=0.94,
        runtime_seconds=0.012,
    )
    emission = EmissionResult(
        fuel_type=FuelType.LNG.value,
        co2=350.0,
        ch4=1.2,
        n2o=0.04,
        co2e=385.0,
    )
    compliance = ComplianceResult(
        cii_rating="B",
        fueleu_pass=True,
        compliance_score=0.88,
    )
    assignment = FleetAssignment(
        vessel_id="VSL-Alpha",
        cargo_id="CRG-101",
        assigned=True,
        estimated_cost=24000.0,
        estimated_fuel=120.0,
    )
    optimization = OptimizationResult(
        optimizer_name=OptimizerType.QPSO.value,
        best_score=118.2,
        runtime_seconds=2.45,
        iterations=150,
        converged=True,
    )
    scenario = ScenarioResult(
        scenario_name="Bio-Methanol Transition",
        fuel_type=FuelType.METHANOL.value,
        total_cost=450000.0,
        total_emissions=185.5,
        fuel_consumption=210.0,
    )

    instances = [voyage, prediction, emission, compliance, assignment, optimization, scenario]

    for item in instances:
        # Test custom to_json method
        json_str = item.to_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

        # Test standard asdict serialization
        dict_repr = asdict(item)
        serialized = json.dumps(dict_repr)
        assert json.loads(serialized) == parsed

    # Test from_dict and from_json roundtrips specifically on VoyageRecord (ingestion contract)
    deserialized_from_dict = VoyageRecord.from_dict(voyage.to_dict())
    deserialized_from_json = VoyageRecord.from_json(voyage.to_json())
    assert deserialized_from_dict == voyage
    assert deserialized_from_json == voyage


def test_interfaces_are_abstract() -> None:
    """Verify that architecture interfaces cannot be instantiated directly."""
    interface_classes = [
        DatasetLoader,
        PredictionEngine,
        FuelPhysicsEngine,
        EmissionEngine,
        ComplianceEngine,
        SchedulerEngine,
        OptimizationEngine,
        ScenarioEngine,
    ]
    for iface in interface_classes:
        assert inspect.isabstract(iface), f"{iface.__name__} must be an abstract base class."
        with pytest.raises(TypeError):
            iface()  # type: ignore[abstract]


def test_configuration_defaults() -> None:
    """Verify system configuration factory and default parameters."""
    cfg = config.get_default_config()
    assert cfg.random_seed == 42
    assert cfg.cross_validation_folds == 5
    assert cfg.logging_level == "INFO"
    assert cfg.data_paths.base_dir.name == "data"
    assert cfg.output_paths.base_dir.name == "outputs"


def test_constants_definitions() -> None:
    """Verify defined fuel types, optimizers, and model types."""
    assert FuelType.DIESEL == "Diesel"
    assert FuelType.AMMONIA == "Ammonia"
    assert OptimizerType.QPSO == "QPSO"
    assert ModelType.QIFCP == "QIFCP"


def test_custom_exceptions_hierarchy() -> None:
    """Verify that domain exceptions inherit from MaritimeSystemError."""
    domain_exceptions = [
        DataValidationError,
        PredictionError,
        OptimizationError,
        SchedulerError,
        ComplianceError,
        IntegrationError,
    ]
    for exc in domain_exceptions:
        assert issubclass(exc, MaritimeSystemError)


def test_logger_initialization(tmp_path: Path) -> None:
    """Verify that logger initializes console and rotating file handlers correctly."""
    logger = logging_config.configure_logging(
        log_file_name="test_run.log",
        level="DEBUG",
    )
    assert logger.name == "maritime_system"
    root_logger = logging.getLogger()
    assert len(root_logger.handlers) >= 2
    has_stream = any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers)
    has_file = any(
        isinstance(h, logging.handlers.RotatingFileHandler) for h in root_logger.handlers
    )
    assert has_stream
    assert has_file
