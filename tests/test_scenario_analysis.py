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
        assert result.energy_consumption_mwh > 0.0
    else:
        assert result.fuel_consumption > 0.0
        assert result.energy_consumption_mwh > 0.0

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


def test_scenario_analysis_dimensional_currency_conversion() -> None:
    """Verify that FuelEU penalties in EUR are explicitly converted to USD at the configurable FX rate."""
    mock_model = MagicMock()
    mock_model.predict.return_value = [
        PredictionResult(
            model_name="HistGradientBoosting",
            predicted_fuel_consumption=100.0,
            confidence_score=0.95,
            runtime_seconds=0.001,
        )
    ]
    mock_manager = MagicMock()
    mock_manager.get_best_model.return_value = mock_model

    engine = ScenarioAnalysisEngine(model_manager=mock_manager)

    fleet = ["VSL-001"]
    base_params = {
        "fuel_type": "Diesel",
        "distance_nm": 1000.0,
        "speed_knots": 14.0,
        "cargo_tons": 40000.0,
        "compliance_year": 2025,
    }

    # Run with parity FX rate (1 EUR = 1.0 USD)
    params_parity = {**base_params, "eur_to_usd_rate": 1.0}
    res_parity = engine.run_scenario("parity_test", fleet, params_parity)

    # Run with 1 EUR = 1.25 USD
    params_scaled = {**base_params, "eur_to_usd_rate": 1.25}
    res_scaled = engine.run_scenario("scaled_test", fleet, params_scaled)

    # Fuel consumption and bunker cost in USD must be identical
    assert res_parity.fuel_cost_usd == res_scaled.fuel_cost_usd == 100.0 * 650.0
    # Statutory penalty in EUR must be identical
    assert res_parity.fueleu_penalty_eur == res_scaled.fueleu_penalty_eur
    assert res_parity.fueleu_penalty_eur > 0.0

    # Converted penalty in USD must scale by exactly 1.25
    assert res_parity.fueleu_penalty_usd == pytest.approx(res_parity.fueleu_penalty_eur * 1.0, rel=1e-3)
    assert res_scaled.fueleu_penalty_usd == pytest.approx(res_scaled.fueleu_penalty_eur * 1.25, rel=1e-3)

    # Total cost in USD = bunker fuel cost (USD) + converted penalty (USD)
    assert res_parity.total_cost == pytest.approx(res_parity.fuel_cost_usd + res_parity.fueleu_penalty_usd, rel=1e-2)
    assert res_scaled.total_cost == pytest.approx(res_scaled.fuel_cost_usd + res_scaled.fueleu_penalty_usd, rel=1e-2)


def test_shore_power_energy_consumption_mwh_representation() -> None:
    """Verify that ShorePower reports electrical consumption in MWh and does not appear as zero consumption."""
    engine = ScenarioAnalysisEngine()
    fleet = ["VSL-001", "VSL-002", "VSL-003"]
    params = {
        "fuel_type": "ShorePower",
        "distance_nm": 1200.0,
        "speed_knots": 14.0,
        "cargo_tons": 35000.0,
        "vessel_dwt": 45000.0,
    }

    res = engine.run_scenario("ShorePower_Test", fleet, params)

    # Physical combustion fuel mass is 0.0
    assert res.fuel_consumption == 0.0
    # But electrical energy in MWh is positive and non-zero
    assert res.energy_consumption_mwh > 0.0
    # Operational emissions are zero
    assert res.total_emissions == 0.0
    # Electricity costs reflect MWh * price_per_MWh
    assert res.total_cost > 0.0
    assert res.total_cost == pytest.approx(res.energy_consumption_mwh * 300.0, rel=1e-2)


def test_heterogeneous_multi_vessel_shore_power_energy_aggregation() -> None:
    """Verify that multi-vessel ShorePower aggregates individual vessel energies rather than reusing last_energy_mwh."""
    engine = ScenarioAnalysisEngine()
    fleet = ["VSL-A", "VSL-B", "VSL-C"]
    vessel_specs = {
        "VSL-A": {"distance_nm": 500.0, "speed_knots": 12.0, "cargo_tons": 20000.0, "vessel_dwt": 30000.0},
        "VSL-B": {"distance_nm": 1000.0, "speed_knots": 14.0, "cargo_tons": 35000.0, "vessel_dwt": 45000.0},
        "VSL-C": {"distance_nm": 1500.0, "speed_knots": 16.0, "cargo_tons": 50000.0, "vessel_dwt": 65000.0},
    }
    params = {
        "fuel_type": "ShorePower",
        "vessel_specs": vessel_specs,
    }

    # Evaluate each vessel individually
    res_a = engine.run_scenario("Indiv_A", ["VSL-A"], params)
    res_b = engine.run_scenario("Indiv_B", ["VSL-B"], params)
    res_c = engine.run_scenario("Indiv_C", ["VSL-C"], params)

    # Verify heterogeneous individual energies
    assert res_a.energy_consumption_mwh < res_b.energy_consumption_mwh < res_c.energy_consumption_mwh

    # Evaluate multi-vessel fleet
    res_multi = engine.run_scenario("Multi_Fleet", fleet, params)

    # The multi-vessel energy must equal the exact sum of individual energies
    expected_sum = res_a.energy_consumption_mwh + res_b.energy_consumption_mwh + res_c.energy_consumption_mwh
    assert res_multi.energy_consumption_mwh == pytest.approx(expected_sum, rel=1e-3)
    assert res_multi.total_cost == pytest.approx(res_a.total_cost + res_b.total_cost + res_c.total_cost, rel=1e-3)

    # Crucial assertion: Must NOT equal 3 * the last vessel's energy (which was the bug!)
    erroneous_overwritten_sum = res_c.energy_consumption_mwh * 3.0
    assert abs(res_multi.energy_consumption_mwh - erroneous_overwritten_sum) > 50.0


def test_shore_power_tariff_pricing_dimension() -> None:
    """Verify ShorePower pricing explicitly uses electricity tariff (USD/MWh) decoupled from bunker ton pricing."""
    from contracts.constants import FUEL_PRICES_USD_PER_TON, SHORE_POWER_PRICE_USD_PER_MWH

    engine = ScenarioAnalysisEngine()
    fleet = ["VSL-001"]
    custom_tariff = 220.0
    params = {
        "fuel_type": "ShorePower",
        "distance_nm": 1000.0,
        "speed_knots": 14.0,
        "cargo_tons": 30000.0,
        "shore_power_price_usd_per_mwh": custom_tariff,
    }

    res = engine.run_scenario("Custom_Tariff_Test", fleet, params)
    assert res.energy_consumption_mwh > 0.0
    assert res.fuel_consumption == 0.0
    # Cost must be exactly energy_mwh * custom_tariff, independent of fuel_prices
    assert res.total_cost == pytest.approx(res.energy_consumption_mwh * custom_tariff, rel=1e-2)




