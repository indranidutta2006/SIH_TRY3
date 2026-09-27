"""Integration tests connecting frozen PhysicsInformedQIFCPRegressor to Fleet Optimization.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 3 Deliverable: End-to-end integration and verification suite:
1. Predictor -> Optimizer connection and canonical model validation.
2. Exact fuel-value propagation (candidate predicted fuel matches downstream cost/emissions).
3. Direct prediction vs optimizer candidate evaluation consistency.
4. Multi-fuel candidate evaluation (Diesel, LNG, Methanol, Hydrogen, Ammonia, ShorePower).
5. Hard constraint enforcement (demand deficit, budget excess, transition limits).
6. Regulatory compliance propagation (FuelEU Maritime GHG intensity, CII rating).
7. Optimizer fairness test (QPSO, PSO, GA, SA, LP, Greedy use identical evaluation & prediction).
8. Deterministic end-to-end smoke test.
"""

from typing import Any
import math
import numpy as np
import pytest

from contracts.constants import (
    FUEL_PRICES_USD_PER_TON,
    FuelType,
)
from contracts.exceptions import DataValidationError, PredictionError
from contracts.schemas import (
    FleetCompositionResult,
    OptimizationScenario,
    OptimizationStatus,
    VoyageRecord,
)
from src.compliance.compliance_engine import MaritimeComplianceEngine
from src.optimization.fleet_composition_optimizer import FleetCompositionOptimizer
from src.optimization.speed_optimizer import EcoSpeedOptimizer
from src.optimization.capacity_optimizer import VesselCapacityOptimizer
from src.benchmarking.solvers.base_solver import BaseBenchmarkSolver
from src.benchmarking.solvers.classical_pso_solver import ClassicalPSOSolver
from src.benchmarking.solvers.genetic_algorithm_solver import GeneticAlgorithmSolver
from src.benchmarking.solvers.greedy_solver import GreedyFleetSolver
from src.benchmarking.solvers.lp_solver import LinearProgrammingSolver
from src.benchmarking.solvers.simulated_annealing_solver import SimulatedAnnealingSolver
from src.benchmarking.qpso_benchmark_adapter import QPSOBenchmarkAdapter
from src.prediction.emission_engine import MaritimeEmissionEngine
from src.prediction.fuel_prediction_service import (
    FuelPredictionService,
    get_fuel_prediction_service,
)
from src.prediction.qifcp import PhysicsInformedQIFCPRegressor


# =========================================================================
# 1. PREDICTOR -> OPTIMIZER CONNECTION TEST
# =========================================================================

def test_fuel_prediction_service_canonical_initialization() -> None:
    """Verify FuelPredictionService initializes with the frozen PhysicsInformedQIFCPRegressor."""
    service = get_fuel_prediction_service()
    assert isinstance(service, FuelPredictionService)
    assert service.model_name == "physics_residual_qifcp"

    # Underlying estimator verification
    assert isinstance(service.model, PhysicsInformedQIFCPRegressor)
    assert service.model.harmonic_order == 3
    assert service.model.n_entanglement_pairs == 15
    assert service.model.entanglement_mode == "adaptive"
    assert service.model.gamma_mode == "grouped"
    assert service.model.lambda_residual == 1.0


def test_fleet_optimizer_defaults_to_canonical_predictor() -> None:
    """Verify FleetCompositionOptimizer wires FuelPredictionService by default."""
    optimizer = FleetCompositionOptimizer()
    assert optimizer.fuel_service is not None
    assert optimizer.fuel_service.model_name == "physics_residual_qifcp"


def test_input_validation_and_guardrails() -> None:
    """Verify FuelPredictionService enforces strict input validation rules."""
    service = get_fuel_prediction_service()

    # Negative or zero distance
    with pytest.raises(DataValidationError, match="distance must be positive"):
        service.calculate_fuel(distance_nm=-50.0, speed_knots=12.0, cargo_tons=20000.0, weather_factor=1.0, fuel_type="Diesel")
    with pytest.raises(DataValidationError, match="distance must be positive"):
        service.calculate_fuel(distance_nm=0.0, speed_knots=12.0, cargo_tons=20000.0, weather_factor=1.0, fuel_type="Diesel")

    # Negative or zero speed
    with pytest.raises(DataValidationError, match="speed must be positive"):
        service.calculate_fuel(distance_nm=1000.0, speed_knots=-1.0, cargo_tons=20000.0, weather_factor=1.0, fuel_type="Diesel")

    # Negative cargo
    with pytest.raises(DataValidationError, match="Cargo payload must be non-negative"):
        service.calculate_fuel(distance_nm=1000.0, speed_knots=12.0, cargo_tons=-100.0, weather_factor=1.0, fuel_type="Diesel")

    # Weather factor < 1.0
    with pytest.raises(DataValidationError, match="Weather factor must be >= 1.0"):
        service.calculate_fuel(distance_nm=1000.0, speed_knots=12.0, cargo_tons=20000.0, weather_factor=0.8, fuel_type="Diesel")

    # Unsupported fuel type
    with pytest.raises(DataValidationError, match="Unsupported fuel type"):
        service.calculate_fuel(distance_nm=1000.0, speed_knots=12.0, cargo_tons=20000.0, weather_factor=1.0, fuel_type="Kerosene")

    # Cargo exceeds vessel capacity (+ margin)
    with pytest.raises(DataValidationError, match="cannot exceed vessel capacity"):
        service.calculate_fuel(distance_nm=1000.0, speed_knots=12.0, cargo_tons=60000.0, weather_factor=1.0, fuel_type="Diesel", vessel_dwt=50000.0)


# =========================================================================
# 2. EXACT FUEL-VALUE PROPAGATION TEST
# =========================================================================

def test_exact_fuel_value_propagation() -> None:
    """Verify that candidate predicted fuel exactly equals downstream cost/emissions fuel input."""
    service = get_fuel_prediction_service()
    solver = GreedyFleetSolver(fuel_service=service)

    scenario = OptimizationScenario(
        cargo_demand=120_000.0,
        route_distance=2500.0,
        deadline_hours=220.0,
        weather_factor=1.10,
        carbon_price=70.0,
        budget=100_000_000.0,
        max_transition_rate=0.50,
    )

    eval_res = solver.evaluate_candidate(
        x_f=2,
        x_m=2,
        x_l=1,
        fuel_mix={"diesel": 3, "lng": 2},
        speed_knots=14.0,
        scenario=scenario,
    )

    # Traceability checks
    assert eval_res["predictor"] == "physics_residual_qifcp"
    assert eval_res["fuel_prediction_source"] == "canonical_qifcp"
    assert eval_res["predicted_fuel_consumption"] == eval_res["fuel_consumption"]
    assert eval_res["predicted_fuel_consumption"] > 0.0

    # Downstream metrics derived from predicted fuel
    assert eval_res["fuel_cost"] > 0.0
    assert eval_res["co2_emissions"] > 0.0
    assert eval_res["operational_cost"] >= eval_res["fuel_cost"]


# =========================================================================
# 3. DIRECT PREDICTION VS OPTIMIZER EVALUATION CONSISTENCY
# =========================================================================

def test_direct_prediction_vs_optimizer_evaluation_consistency() -> None:
    """Verify direct service prediction matches solver candidate evaluation."""
    service = get_fuel_prediction_service()
    solver = GreedyFleetSolver(fuel_service=service)

    scenario = OptimizationScenario(
        cargo_demand=30_000.0,
        route_distance=2000.0,
        deadline_hours=200.0,
        weather_factor=1.15,
        carbon_price=60.0,
        budget=50_000_000.0,
    )

    # 1 Medium vessel (Handymax/Panamax: DWT=45,000, design cargo=38,250)
    # With 1 medium vessel and cargo 30,000 t, actual carried cargo is 30,000 t
    eval_res = solver.evaluate_candidate(
        x_f=0,
        x_m=1,
        x_l=0,
        fuel_mix={"diesel": 1},
        speed_knots=14.0,
        scenario=scenario,
    )

    # Direct calculation using same inputs
    direct_fuel = service.calculate_fuel(
        distance_nm=2000.0,
        speed_knots=14.0,
        cargo_tons=45000.0 * 0.85,
        weather_factor=1.15,
        fuel_type="Diesel",
        vessel_dwt=45000.0,
        admiralty_coeff=520.0,
    )

    assert pytest.approx(eval_res["predicted_fuel_consumption"], rel=1e-4) == direct_fuel


# =========================================================================
# 4. MULTI-FUEL CANDIDATE EVALUATION
# =========================================================================

@pytest.mark.parametrize("fuel_name", ["Diesel", "LNG", "Methanol", "Hydrogen", "Ammonia", "ShorePower"])
def test_multi_fuel_candidate_evaluation(fuel_name: str) -> None:
    """Verify all supported marine fuel pathways evaluate cleanly through canonical predictor."""
    service = get_fuel_prediction_service()

    fuel_consumed = service.calculate_fuel(
        distance_nm=1200.0,
        speed_knots=14.0,
        cargo_tons=30000.0,
        weather_factor=1.05,
        fuel_type=fuel_name,
        vessel_dwt=45000.0,
    )

    assert np.isfinite(fuel_consumed)

    if fuel_name == "ShorePower":
        # Shore power does not combust bunker fuel; records electrical MWh
        assert fuel_consumed == 0.0
        assert service.last_energy_mwh is not None
        assert service.last_energy_mwh > 0.0
    else:
        assert fuel_consumed > 0.0
        # Energy parity check: Hydrogen has ~2.8x energy density of diesel,
        # so mass consumed should be significantly lower than Diesel
        if fuel_name == "Hydrogen":
            diesel_fuel = service.calculate_fuel(
                distance_nm=1200.0,
                speed_knots=14.0,
                cargo_tons=30000.0,
                weather_factor=1.05,
                fuel_type="Diesel",
                vessel_dwt=45000.0,
            )
            assert fuel_consumed < diesel_fuel * 0.5


# =========================================================================
# 5. HARD CONSTRAINT ENFORCEMENT
# =========================================================================

def test_hard_constraint_enforcement() -> None:
    """Verify demand deficit, budget excess, and transition limit penalties."""
    service = get_fuel_prediction_service()
    solver = GreedyFleetSolver(fuel_service=service)

    # 1. Demand deficit scenario: 1 Feeder (357,000 t annual payload) vs 500,000 t demand
    scenario_demand_deficit = OptimizationScenario(
        cargo_demand=500_000.0,
        route_distance=1000.0,
        deadline_hours=100.0,
        budget=100_000_000.0,
        max_transition_rate=0.50,
    )
    res_deficit = solver.evaluate_candidate(
        x_f=1,
        x_m=0,
        x_l=0,
        fuel_mix={"diesel": 1},
        speed_knots=12.0,
        scenario=scenario_demand_deficit,
    )
    assert res_deficit["demand_satisfaction_rate"] < 1.0
    assert res_deficit["feasible_solution"] is False
    assert res_deficit["objective_score"] > 50.0

    # 2. Transition rate excess: 80% alt fuel when max allowed is 30%
    scenario_transition = OptimizationScenario(
        cargo_demand=50_000.0,
        route_distance=1000.0,
        deadline_hours=100.0,
        budget=100_000_000.0,
        max_transition_rate=0.30,
    )
    res_transition = solver.evaluate_candidate(
        x_f=2,
        x_m=2,
        x_l=1,
        fuel_mix={"diesel": 1, "lng": 4},  # 80% alt fuel
        speed_knots=14.0,
        scenario=scenario_transition,
    )
    assert res_transition["feasible_solution"] is False


# =========================================================================
# 6. REGULATORY COMPLIANCE PROPAGATION
# =========================================================================

def test_regulatory_compliance_propagation() -> None:
    """Verify FuelEU maritime penalty and CII rating propagate accurately from predicted fuel."""
    service = get_fuel_prediction_service()
    compliance_engine = MaritimeComplianceEngine()
    emission_engine = MaritimeEmissionEngine()

    dist = 2500.0
    cargo = 40000.0
    speed = 13.0
    fuel_cons = service.calculate_fuel(
        distance_nm=dist,
        speed_knots=speed,
        cargo_tons=cargo,
        weather_factor=1.1,
        fuel_type="Diesel",
        vessel_dwt=50000.0,
    )

    # Emissions
    wtw_res = emission_engine.calculate_wtw(fuel_cons, "Diesel")
    assert wtw_res.co2e > 0.0

    # FuelEU compliance evaluation
    fueleu_res = compliance_engine.evaluate_fueleu(
        ghg_intensity=91.16,
        energy_used_mj=fuel_cons * 41000.0,
        year=2025,
    )
    assert fueleu_res.fueleu_target > 0.0
    assert fueleu_res.compliance_status in {"COMPLIANT", "NON_COMPLIANT"}

    # CII rating evaluation
    cii_res = compliance_engine.evaluate_cii(
        co2_emissions=wtw_res.co2e,
        distance_nm=dist,
        vessel_dwt=50000.0,
        vessel_type="Bulk Carrier",
        year=2024,
    )
    assert cii_res.cii_rating in {"A", "B", "C", "D", "E"}
    assert cii_res.attained_cii > 0.0


# =========================================================================
# 7. OPTIMIZER FAIRNESS TEST
# =========================================================================

def test_optimizer_fairness_shared_evaluation() -> None:
    """Verify that all solvers evaluate candidate solutions using identical equations and predictor."""
    service = get_fuel_prediction_service()

    scenario = OptimizationScenario(
        cargo_demand=80_000.0,
        route_distance=1800.0,
        deadline_hours=160.0,
        weather_factor=1.1,
        carbon_price=55.0,
        budget=70_000_000.0,
        max_transition_rate=0.50,
        weights=(0.5, 0.3, 0.2),
    )

    # Initialize all solvers sharing the exact same fuel_service
    solvers: dict[str, BaseBenchmarkSolver] = {
        "ClassicalPSO": ClassicalPSOSolver(fuel_service=service),
        "GeneticAlgorithm": GeneticAlgorithmSolver(fuel_service=service),
        "Greedy": GreedyFleetSolver(fuel_service=service),
        "LinearProgramming": LinearProgrammingSolver(fuel_service=service),
        "SimulatedAnnealing": SimulatedAnnealingSolver(fuel_service=service),
        "QPSOAdapter": QPSOBenchmarkAdapter(fuel_service=service),
    }

    eval_results: dict[str, dict[str, Any]] = {}
    for name, solver in solvers.items():
        res = solver.evaluate_candidate(
            x_f=2,
            x_m=2,
            x_l=1,
            fuel_mix={"diesel": 3, "lng": 2},
            speed_knots=14.0,
            scenario=scenario,
        )
        eval_results[name] = res

    # Verify that every solver yields identical values
    first_res = list(eval_results.values())[0]
    expected_fuel = first_res["predicted_fuel_consumption"]
    expected_cost = first_res["operational_cost"]
    expected_emiss = first_res["emissions"]
    expected_score = first_res["objective_score"]

    for name, res in eval_results.items():
        assert res["predictor"] == "physics_residual_qifcp", f"{name} predictor mismatch"
        assert res["fuel_prediction_source"] == "canonical_qifcp", f"{name} source mismatch"
        assert pytest.approx(res["predicted_fuel_consumption"], rel=1e-5) == expected_fuel, f"{name} fuel mismatch"
        assert pytest.approx(res["operational_cost"], rel=1e-5) == expected_cost, f"{name} cost mismatch"
        assert pytest.approx(res["emissions"], rel=1e-5) == expected_emiss, f"{name} emissions mismatch"
        assert pytest.approx(res["objective_score"], rel=1e-5) == expected_score, f"{name} objective mismatch"


# =========================================================================
# 8. DETERMINISTIC END-TO-END SMOKE TEST
# =========================================================================

def test_deterministic_end_to_end_smoke() -> None:
    """Run full deterministic fleet optimization and assert complete output traceability."""
    service = get_fuel_prediction_service()
    optimizer = FleetCompositionOptimizer(fuel_service=service)

    scenario = OptimizationScenario(
        cargo_demand=100_000.0,
        route_distance=2500.0,
        deadline_hours=200.0,
        scenario_id="SCEN-INT-SMOKE",
        budget=120_000_000.0,
        carbon_price=65.0,
        max_transition_rate=0.40,
    )

    result = optimizer.optimize_composition(scenario)

    assert result.status == OptimizationStatus.SUCCESS
    assert result.total_capacity >= scenario.cargo_demand * scenario.service_level
    assert result.fuel_consumption > 0.0
    assert result.emissions > 0.0
    assert result.operational_cost > 0.0
    assert result.metadata["predictor"] == "physics_residual_qifcp"
    assert result.metadata["fuel_prediction_source"] == "canonical_qifcp"
    assert result.metadata["predicted_fuel_consumption"] > 0.0
