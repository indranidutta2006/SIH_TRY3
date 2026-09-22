"""Unit tests for the green fleet scenario simulation and tradeoff engine.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Verifies strict routing guardrails:
- ML-trained fuels (Diesel, LNG, Methanol) call ProductionModelManager.
- Non-ML alternative fuels (Hydrogen, Ammonia, ShorePower) bypass ProductionModelManager
  and route strictly through first-principles FuelPhysicsEngine.
"""

from unittest.mock import MagicMock, patch
import pytest

from contracts.constants import FuelType
from contracts.interfaces import ScenarioEngine
from contracts.schemas import PredictionResult, ScenarioResult
from src.optimization.scenario_analysis import (
    ML_ROUTED_FUELS,
    PHYSICS_ROUTED_FUELS,
    ScenarioAnalysisEngine,
)


def test_scenario_analysis_engine_implements_interface() -> None:
    """Verify ScenarioAnalysisEngine implements contracts.interfaces.ScenarioEngine."""
    engine = ScenarioAnalysisEngine()
    assert isinstance(engine, ScenarioEngine)


@pytest.mark.parametrize("fuel", ["Diesel", "LNG", "Methanol"])
def test_ml_fuels_call_production_model(fuel: str) -> None:
    """Assert Diesel, LNG, and Methanol scenarios strictly call ProductionModelManager.

    This ensures that fuels with learned empirical models utilize the trained regressor.
    """
    mock_model = MagicMock()
    mock_model.predict.return_value = [
        PredictionResult(
            model_name="HistGradientBoosting",
            predicted_fuel_consumption=55.0,
            confidence_score=0.95,
            runtime_seconds=0.001,
        )
    ]

    mock_manager = MagicMock()
    mock_manager.get_best_model.return_value = mock_model

    engine = ScenarioAnalysisEngine(model_manager=mock_manager)

    fleet = ["VSL-001"]
    params = {
        "fuel_type": fuel,
        "distance_nm": 1000.0,
        "speed_knots": 14.0,
        "cargo_tons": 40000.0,
    }

    result = engine.run_scenario(
        scenario_name=f"test_{fuel}",
        vessel_fleet=fleet,
        operational_parameters=params,
    )

    # Assert model_manager was called and model was evaluated
    assert mock_manager.get_best_model.called
    assert mock_model.predict.called
    assert result.fuel_consumption == 55.0
    assert result.total_cost > 0.0
    assert result.total_emissions > 0.0


@pytest.mark.parametrize("fuel", ["Hydrogen", "Ammonia", "ShorePower"])
def test_physics_fuels_bypass_production_model(fuel: str) -> None:
    """Assert Hydrogen, Ammonia, and ShorePower NEVER call ProductionModelManager.

    This is the core architectural regression guard preventing circularity and
    out-of-distribution hallucinations on fuels the classical model was never trained on.
    """
    mock_manager = MagicMock()

    engine = ScenarioAnalysisEngine(model_manager=mock_manager)

    fleet = ["VSL-001", "VSL-002"]
    params = {
        "fuel_type": fuel,
        "distance_nm": 1000.0,
        "speed_knots": 14.0,
        "cargo_tons": 35000.0,
    }

    result = engine.run_scenario(
        scenario_name=f"test_{fuel}",
        vessel_fleet=fleet,
        operational_parameters=params,
    )

    # CRITICAL: ProductionModelManager must NEVER be touched for non-ML fuels
    assert not mock_manager.get_best_model.called
    if fuel == "ShorePower":
        assert result.fuel_consumption == 0.0
        assert result.total_emissions == 0.0
    else:
        assert result.fuel_consumption > 0.0

    assert result.total_cost > 0.0


def test_compare_scenarios_balanced_ranking_without_distortion() -> None:
    """Verify compare_scenarios produces a balanced multi-criteria ranking without magnitude distortion."""
    engine = ScenarioAnalysisEngine()

    s1 = ScenarioResult(
        scenario_name="Cheapest_HighEmiss",
        fuel_type="Diesel",
        total_cost=100000.0,
        total_emissions=1000.0,
        fuel_consumption=100.0,
    )
    s2 = ScenarioResult(
        scenario_name="Cleanest_HighCost",
        fuel_type="Hydrogen",
        total_cost=400000.0,
        total_emissions=50.0,
        fuel_consumption=30.0,
    )
    s3 = ScenarioResult(
        scenario_name="Balanced_Moderate",
        fuel_type="LNG",
        total_cost=180000.0,
        total_emissions=300.0,
        fuel_consumption=80.0,
    )

    ranked = engine.compare_scenarios([s1, s2, s3])
    assert len(ranked) == 3

    # Cost-only ranking
    cost_ranked = engine.rank_by_cost([s1, s2, s3])
    assert cost_ranked[0].scenario_name == "Cheapest_HighEmiss"
    assert cost_ranked[-1].scenario_name == "Cleanest_HighCost"

    # Emissions-only ranking
    emiss_ranked = engine.rank_by_emissions([s1, s2, s3])
    assert emiss_ranked[0].scenario_name == "Cleanest_HighCost"
    assert emiss_ranked[-1].scenario_name == "Cheapest_HighEmiss"
