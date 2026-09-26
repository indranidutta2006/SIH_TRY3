"""Import consistency and identity tests for root compatibility shims.

Verifies that:
1. contracts/ is the single source of truth.
2. Root files (schemas.py, interfaces.py, constants.py, exceptions.py) are pure shims.
3. Importing from 'contracts' or from legacy root modules resolves to the exact same object in memory.
4. All contract modules can be imported independently without circular dependencies.
"""

import importlib
import sys


def test_schema_import_consistency() -> None:
    """Verify schemas imported via contracts and root shim refer to identical objects."""
    import contracts.schemas as direct_schemas
    from contracts import (
        ComplianceResult,
        EmissionResult,
        FleetAssignment,
        OptimizationResult,
        PredictionResult,
        ScenarioResult,
        VoyageRecord,
    )
    from schemas import (
        ComplianceResult as LegacyComplianceResult,
        EmissionResult as LegacyEmissionResult,
        FleetAssignment as LegacyFleetAssignment,
        OptimizationResult as LegacyOptimizationResult,
        PredictionResult as LegacyPredictionResult,
        ScenarioResult as LegacyScenarioResult,
        VoyageRecord as LegacyVoyageRecord,
    )

    # Assert module identity sanity
    assert direct_schemas.__name__ == "contracts.schemas"

    # Assert object identity (is) between contracts package and legacy root
    assert VoyageRecord is LegacyVoyageRecord
    assert PredictionResult is LegacyPredictionResult
    assert EmissionResult is LegacyEmissionResult
    assert ComplianceResult is LegacyComplianceResult
    assert FleetAssignment is LegacyFleetAssignment
    assert OptimizationResult is LegacyOptimizationResult
    assert ScenarioResult is LegacyScenarioResult

    # Assert object identity with contracts.schemas module directly
    assert VoyageRecord is direct_schemas.VoyageRecord
    assert PredictionResult is direct_schemas.PredictionResult
    assert EmissionResult is direct_schemas.EmissionResult
    assert ComplianceResult is direct_schemas.ComplianceResult
    assert FleetAssignment is direct_schemas.FleetAssignment
    assert OptimizationResult is direct_schemas.OptimizationResult
    assert ScenarioResult is direct_schemas.ScenarioResult


def test_interface_import_consistency() -> None:
    """Verify interfaces imported via contracts and root shim refer to identical objects."""
    import contracts.interfaces as direct_interfaces
    from contracts import (
        ComplianceEngine,
        DatasetLoader,
        EmissionEngine,
        FuelPhysicsEngine,
        OptimizationEngine,
        PredictionEngine,
        ScenarioEngine,
        SchedulerEngine,
    )
    from interfaces import (
        ComplianceEngine as LegacyComplianceEngine,
        DatasetLoader as LegacyDatasetLoader,
        EmissionEngine as LegacyEmissionEngine,
        FuelPhysicsEngine as LegacyFuelPhysicsEngine,
        OptimizationEngine as LegacyOptimizationEngine,
        PredictionEngine as LegacyPredictionEngine,
        ScenarioEngine as LegacyScenarioEngine,
        SchedulerEngine as LegacySchedulerEngine,
    )

    # Assert module identity sanity
    assert direct_interfaces.__name__ == "contracts.interfaces"

    assert DatasetLoader is LegacyDatasetLoader
    assert PredictionEngine is LegacyPredictionEngine
    assert FuelPhysicsEngine is LegacyFuelPhysicsEngine
    assert EmissionEngine is LegacyEmissionEngine
    assert ComplianceEngine is LegacyComplianceEngine
    assert SchedulerEngine is LegacySchedulerEngine
    assert OptimizationEngine is LegacyOptimizationEngine
    assert ScenarioEngine is LegacyScenarioEngine

    assert DatasetLoader is direct_interfaces.DatasetLoader
    assert PredictionEngine is direct_interfaces.PredictionEngine
    assert FuelPhysicsEngine is direct_interfaces.FuelPhysicsEngine
    assert EmissionEngine is direct_interfaces.EmissionEngine
    assert ComplianceEngine is direct_interfaces.ComplianceEngine
    assert SchedulerEngine is direct_interfaces.SchedulerEngine
    assert OptimizationEngine is direct_interfaces.OptimizationEngine
    assert ScenarioEngine is direct_interfaces.ScenarioEngine


def test_constant_import_consistency() -> None:
    """Verify constants imported via contracts and root shim refer to identical objects."""
    import contracts.constants as direct_constants
    from constants import (
        DEFAULT_EUR_TO_USD_FX_RATE as Legacy_FX_RATE,
        FUEL_PRICES_USD_PER_TON as Legacy_FUEL_PRICES_USD_PER_TON,
        SHORE_POWER_PRICE_USD_PER_MWH as Legacy_SHORE_POWER_PRICE,
        STANDARD_FUEL_PRICES_USD as Legacy_STANDARD_FUEL_PRICES_USD,
        SUPPORTED_FUELS as Legacy_SUPPORTED_FUELS,
        SUPPORTED_OPTIMIZERS as Legacy_SUPPORTED_OPTIMIZERS,
        SUPPORTED_PREDICTION_MODELS as Legacy_SUPPORTED_PREDICTION_MODELS,
        FuelType as LegacyFuelType,
        ModelType as LegacyModelType,
        OptimizerType as LegacyOptimizerType,
    )
    from contracts import (
        DEFAULT_EUR_TO_USD_FX_RATE,
        FUEL_PRICES_USD_PER_TON,
        FuelType,
        ModelType,
        OptimizerType,
        SHORE_POWER_PRICE_USD_PER_MWH,
        STANDARD_FUEL_PRICES_USD,
        SUPPORTED_FUELS,
        SUPPORTED_OPTIMIZERS,
        SUPPORTED_PREDICTION_MODELS,
    )

    # Assert module identity sanity
    assert direct_constants.__name__ == "contracts.constants"

    assert FuelType is LegacyFuelType
    assert OptimizerType is LegacyOptimizerType
    assert ModelType is LegacyModelType
    assert SUPPORTED_FUELS is Legacy_SUPPORTED_FUELS
    assert SUPPORTED_OPTIMIZERS is Legacy_SUPPORTED_OPTIMIZERS
    assert SUPPORTED_PREDICTION_MODELS is Legacy_SUPPORTED_PREDICTION_MODELS
    assert DEFAULT_EUR_TO_USD_FX_RATE == Legacy_FX_RATE
    assert FUEL_PRICES_USD_PER_TON is Legacy_FUEL_PRICES_USD_PER_TON
    assert SHORE_POWER_PRICE_USD_PER_MWH == Legacy_SHORE_POWER_PRICE
    assert STANDARD_FUEL_PRICES_USD is Legacy_STANDARD_FUEL_PRICES_USD

    assert FuelType is direct_constants.FuelType
    assert OptimizerType is direct_constants.OptimizerType
    assert ModelType is direct_constants.ModelType
    assert SUPPORTED_FUELS is direct_constants.SUPPORTED_FUELS
    assert FUEL_PRICES_USD_PER_TON is direct_constants.FUEL_PRICES_USD_PER_TON
    assert SHORE_POWER_PRICE_USD_PER_MWH == direct_constants.SHORE_POWER_PRICE_USD_PER_MWH
    assert STANDARD_FUEL_PRICES_USD is direct_constants.STANDARD_FUEL_PRICES_USD


def test_exception_import_consistency() -> None:
    """Verify exceptions imported via contracts and root shim refer to identical objects."""
    import contracts.exceptions as direct_exceptions
    from contracts import (
        ComplianceError,
        DataValidationError,
        IntegrationError,
        MaritimeSystemError,
        OptimizationError,
        PredictionError,
        SchedulerError,
    )
    from exceptions import (
        ComplianceError as LegacyComplianceError,
        DataValidationError as LegacyDataValidationError,
        IntegrationError as LegacyIntegrationError,
        MaritimeSystemError as LegacyMaritimeSystemError,
        OptimizationError as LegacyOptimizationError,
        PredictionError as LegacyPredictionError,
        SchedulerError as LegacySchedulerError,
    )

    # Assert module identity sanity
    assert direct_exceptions.__name__ == "contracts.exceptions"

    assert MaritimeSystemError is LegacyMaritimeSystemError
    assert DataValidationError is LegacyDataValidationError
    assert PredictionError is LegacyPredictionError
    assert OptimizationError is LegacyOptimizationError
    assert SchedulerError is LegacySchedulerError
    assert ComplianceError is LegacyComplianceError
    assert IntegrationError is LegacyIntegrationError

    assert MaritimeSystemError is direct_exceptions.MaritimeSystemError
    assert DataValidationError is direct_exceptions.DataValidationError


def test_independent_module_importability() -> None:
    """Verify that each contract module can be imported independently in a clean namespace."""
    module_names = [
        "contracts.version",
        "contracts.constants",
        "contracts.exceptions",
        "contracts.schemas",
        "contracts.interfaces",
        "contracts",
        "schemas",
        "interfaces",
        "constants",
        "exceptions",
    ]

    for mod_name in module_names:
        # Evict from sys.modules to simulate cold import
        sys.modules.pop(mod_name, None)
        imported = importlib.import_module(mod_name)
        assert imported is not None


def test_all_schema_classes_identity() -> None:
    """Verify all 34 classes in contracts.schemas match schemas by object identity."""
    import contracts.schemas as direct_schemas
    import schemas as root_schemas

    classes_tested = 0
    for attr in dir(direct_schemas):
        if not attr.startswith("_"):
            obj = getattr(direct_schemas, attr)
            if isinstance(obj, type) and obj.__module__ == "contracts.schemas":
                assert getattr(root_schemas, attr, None) is obj
                classes_tested += 1
    assert classes_tested == 34


def test_package_bridge_import_consistency() -> None:
    """Verify root compatibility packages (optimization, operations, benchmarking) re-export identical src objects."""
    import optimization as root_opt
    import src.optimization as src_opt
    for name in root_opt.__all__:
        assert getattr(root_opt, name) is getattr(src_opt, name)

    import operations as root_ops
    import src.operations as src_ops
    for name in ["CargoDemandSatisfactionEngine", "ScheduleReliabilityEngine"]:
        assert getattr(root_ops, name) is getattr(src_ops, name)

    import benchmarking as root_bmk
    import src.benchmarking as src_bmk
    for name in root_bmk.__all__:
        assert getattr(root_bmk, name) is getattr(src_bmk, name)


def test_transition_milestone_serialization() -> None:
    """Verify TransitionMilestone serialization and deserialization symmetry."""
    from contracts.schemas import TransitionMilestone

    milestone = TransitionMilestone(
        year=2030,
        fleet_mix={"diesel": 6, "lng": 2},
        fuel_shares={"diesel": 0.75, "lng": 0.25},
        avg_speed_knots=13.5,
        capex_usd=12_000_000.0,
        opex_usd=3_500_000.0,
        annual_emissions_tons=28_500.0,
        cii_rating="B",
        fueleu_penalty_usd=0.0,
        metadata={"source": "test"},
    )

    d = milestone.to_dict()
    assert d["year"] == 2030
    assert d["cii_rating"] == "B"

    j = milestone.to_json()
    assert '"year": 2030' in j

    m_from_dict = TransitionMilestone.from_dict(d)
    assert m_from_dict == milestone

    m_from_json = TransitionMilestone.from_json(j)
    assert m_from_json == milestone

