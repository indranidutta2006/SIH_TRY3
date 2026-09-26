"""Comprehensive unit and integration tests for Phase 1 Strategic Fleet Optimization.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Tests:
1. OptimizationScenario schema, validation, and serialization.
2. FleetCompositionOptimizer: vessel classes, green fuel mix, budget limits, transition rates.
3. VesselCapacityOptimizer: class-dependent sizing (DWT/TEU), port limits, utilization.
4. EcoSpeedOptimizer: cubic power curve, deadline enforcement, delay demurrage, weather scaling.
5. FleetStrategyOptimizer: end-to-end orchestration, baseline comparisons, route deployment plan,
   and scenario persistence (save/load).
6. Robustness against edge cases (zero demand, excessive demand, infeasible constraints).
"""

import json
from pathlib import Path
import pytest

from contracts.schemas import (
    CapacityOptimizationResult,
    FleetCompositionResult,
    FleetStrategyRecommendation,
    OptimizationScenario,
    OptimizationStatus,
    SpeedOptimizationResult,
)
from src.optimization.capacity_optimizer import VesselCapacityOptimizer
from src.optimization.fleet_composition_optimizer import FleetCompositionOptimizer
from src.optimization.fleet_strategy_optimizer import FleetStrategyOptimizer
from src.optimization.speed_optimizer import EcoSpeedOptimizer


# =============================================================================
# 1. OPTIMIZATION SCENARIO TESTS
# =============================================================================

def test_optimization_scenario_initialization_and_serialization() -> None:
    """Verify OptimizationScenario handles defaults, custom fields, and roundtrip serialization."""
    scenario = OptimizationScenario(
        cargo_demand=150_000.0,
        route_distance=4000.0,
        deadline_hours=300.0,
        scenario_id="SCEN-TEST-001",
        carbon_price=100.0,
        budget=80_000_000.0,
        weather_factor=1.10,
        vessel_class="PANAMAX",
        service_level=0.95,
        max_transition_rate=0.30,
        weights=(1.0, 1.0, 1.0),
    )

    d = scenario.to_dict()
    assert d["scenario_id"] == "SCEN-TEST-001"
    assert d["cargo_demand"] == 150_000.0
    assert d["carbon_price"] == 100.0
    assert d["vessel_class"] == "PANAMAX"
    assert d["max_transition_rate"] == 0.30

    json_str = scenario.to_json()
    assert isinstance(json_str, str)
    assert "SCEN-TEST-001" in json_str

    reconstructed = OptimizationScenario.from_dict(json.loads(json_str))
    assert reconstructed.scenario_id == scenario.scenario_id
    assert reconstructed.cargo_demand == scenario.cargo_demand
    assert reconstructed.budget == scenario.budget


# =============================================================================
# 2. FLEET COMPOSITION OPTIMIZER TESTS
# =============================================================================

def test_fleet_composition_standard_optimization() -> None:
    """Verify FleetCompositionOptimizer determines feasible fleet mix and standardized metadata."""
    optimizer = FleetCompositionOptimizer()
    scenario = OptimizationScenario(
        cargo_demand=200_000.0,
        route_distance=3000.0,
        deadline_hours=240.0,
        scenario_id="SCEN-COMP-01",
        budget=150_000_000.0,
        carbon_price=80.0,
        max_transition_rate=0.50,
    )

    result = optimizer.optimize_composition(scenario)
    assert isinstance(result, FleetCompositionResult)
    assert result.status == OptimizationStatus.SUCCESS
    assert result.total_capacity >= scenario.cargo_demand * scenario.service_level
    assert result.fuel_consumption > 0.0
    assert result.emissions > 0.0
    assert result.operational_cost > 0.0
    assert result.carbon_cost > 0.0

    mix = result.fleet_mix
    assert "feeder" in mix and "medium" in mix and "large" in mix
    assert "diesel" in mix and "lng" in mix and "methanol" in mix
    total_ships = mix["feeder"] + mix["medium"] + mix["large"]
    assert total_ships > 0

    # Verify standardized solver metadata hooks
    meta = result.metadata
    assert meta["solver_name"] == "deterministic_combinatorial_mip"
    assert meta["runtime_ms"] >= 0.0
    assert meta["iterations"] > 0
    assert meta["convergence_score"] == 1.0
    assert "optimization_trace" in meta
    assert "regulatory_breakdown" in meta
    assert "cii_score" in meta["regulatory_breakdown"]


def test_fleet_composition_zero_demand_edge_case() -> None:
    """Verify zero cargo demand gracefully returns zero vessels and status SUCCESS."""
    optimizer = FleetCompositionOptimizer()
    scenario = OptimizationScenario(
        cargo_demand=0.0,
        route_distance=3000.0,
        deadline_hours=200.0,
    )

    result = optimizer.optimize_composition(scenario)
    assert result.status == OptimizationStatus.SUCCESS
    assert result.total_capacity == 0.0
    assert result.operational_cost == 0.0
    assert result.fuel_consumption == 0.0
    total_ships = sum(v for k, v in result.fleet_mix.items() if k in {"feeder", "medium", "large"})
    assert total_ships == 0


def test_fleet_composition_budget_exceeded_status() -> None:
    """Verify extremely tight budget returns BUDGET_EXCEEDED without unhandled exception."""
    optimizer = FleetCompositionOptimizer()
    scenario = OptimizationScenario(
        cargo_demand=300_000.0,
        route_distance=3000.0,
        deadline_hours=200.0,
        budget=100.0,  # $100 budget is impossible for a fleet
    )

    result = optimizer.optimize_composition(scenario)
    assert result.status == OptimizationStatus.BUDGET_EXCEEDED
    assert result.optimization_score == float("inf")


def test_fleet_composition_demand_unsatisfiable_status() -> None:
    """Verify impossible cargo demand exceeding max fleet availability returns DEMAND_UNSATISFIABLE."""
    optimizer = FleetCompositionOptimizer()
    scenario = OptimizationScenario(
        cargo_demand=100_000_000.0,  # 100M tons is impossible with max 25 vessels
        route_distance=5000.0,
        deadline_hours=200.0,
        budget=10_000_000_000.0,
    )

    result = optimizer.optimize_composition(scenario, max_fleet_size=10)
    assert result.status == OptimizationStatus.DEMAND_UNSATISFIABLE


def test_fleet_composition_transition_rate_constraint() -> None:
    """Verify max_transition_rate restricts alternative fuel adoption to specified fraction."""
    optimizer = FleetCompositionOptimizer()
    # 0.0 transition rate means strictly 100% diesel
    scenario_zero_trans = OptimizationScenario(
        cargo_demand=100_000.0,
        route_distance=2000.0,
        deadline_hours=180.0,
        max_transition_rate=0.0,
        budget=100_000_000.0,
    )
    res_zero = optimizer.optimize_composition(scenario_zero_trans)
    assert res_zero.status == OptimizationStatus.SUCCESS
    mix = res_zero.fleet_mix
    assert mix["lng"] == 0
    assert mix["methanol"] == 0
    assert mix["hydrogen"] == 0
    assert mix["ammonia"] == 0
    assert mix["diesel"] > 0


# =============================================================================
# 3. VESSEL CAPACITY OPTIMIZER TESTS
# =============================================================================

def test_vessel_capacity_optimization_panamax() -> None:
    """Verify VesselCapacityOptimizer sizing bounds for Panamax vessel class."""
    optimizer = VesselCapacityOptimizer()
    scenario = OptimizationScenario(
        cargo_demand=120_000.0,
        route_distance=3500.0,
        deadline_hours=250.0,
        vessel_class="PANAMAX",
    )

    result = optimizer.optimize_capacity(scenario)
    assert isinstance(result, CapacityOptimizationResult)
    assert result.status == OptimizationStatus.SUCCESS
    assert result.vessel_class == "PANAMAX"
    # Panamax DWT range: [25,000, 55,000]
    assert 25000.0 <= result.recommended_capacity <= 55000.0
    assert result.capacity_teu > 0.0
    assert 0.0 <= result.utilization_rate <= 1.0
    assert result.fuel_consumption > 0.0
    assert result.optimal_trips >= 1
    assert result.metadata["solver_name"] == "naval_architecture_golden_section"


def test_vessel_capacity_class_scaling() -> None:
    """Verify Feeder capacity is strictly smaller than Capesize capacity."""
    optimizer = VesselCapacityOptimizer()
    scenario_feeder = OptimizationScenario(
        cargo_demand=40_000.0,
        route_distance=1500.0,
        deadline_hours=150.0,
        vessel_class="FEEDER",
    )
    scenario_capesize = OptimizationScenario(
        cargo_demand=250_000.0,
        route_distance=6000.0,
        deadline_hours=400.0,
        vessel_class="CAPESIZE",
    )

    res_feeder = optimizer.optimize_capacity(scenario_feeder)
    res_cape = optimizer.optimize_capacity(scenario_capesize)

    assert res_feeder.vessel_class == "FEEDER"
    assert res_cape.vessel_class == "CAPESIZE"
    assert res_feeder.recommended_capacity <= 15000.0
    assert res_cape.recommended_capacity >= 100000.0
    assert res_feeder.capacity_teu < res_cape.capacity_teu


def test_vessel_capacity_infeasible_port_limit() -> None:
    """Verify port capacity limit below minimum vessel class DWT returns INFEASIBLE."""
    optimizer = VesselCapacityOptimizer()
    scenario = OptimizationScenario(
        cargo_demand=100_000.0,
        route_distance=2000.0,
        deadline_hours=150.0,
        vessel_class="PANAMAX",  # Minimum DWT 25,000
    )
    # Port limit of 10,000 is below Panamax minimum of 25,000
    result = optimizer.optimize_capacity(scenario, port_capacity_limit=10000.0)
    assert result.status == OptimizationStatus.INFEASIBLE


def test_vessel_capacity_zero_demand() -> None:
    """Verify zero cargo demand returns capacity 0 and status SUCCESS."""
    optimizer = VesselCapacityOptimizer()
    scenario = OptimizationScenario(cargo_demand=0.0, route_distance=2000.0, deadline_hours=150.0)
    result = optimizer.optimize_capacity(scenario)
    assert result.status == OptimizationStatus.SUCCESS
    assert result.recommended_capacity == 0.0
    assert result.capacity_teu == 0.0


# =============================================================================
# 4. ECO-SPEED OPTIMIZER TESTS
# =============================================================================

def test_eco_speed_optimization_within_bounds() -> None:
    """Verify EcoSpeedOptimizer returns optimal speed within operational hydrodynamic bounds."""
    optimizer = EcoSpeedOptimizer()
    scenario = OptimizationScenario(
        cargo_demand=50_000.0,
        route_distance=3000.0,
        deadline_hours=260.0,
    )

    result = optimizer.optimize_speed(scenario, vessel_type="Bulk Carrier")
    assert isinstance(result, SpeedOptimizationResult)
    assert result.status == OptimizationStatus.SUCCESS
    assert 9.0 <= result.optimal_speed <= 17.0
    assert result.estimated_eta <= scenario.deadline_hours
    assert result.fuel_consumption > 0.0
    assert result.cost > 0.0
    assert result.delay_hours == 0.0
    assert result.metadata["solver_name"] == "bounded_scalar_brent"


def test_eco_speed_tight_vs_relaxed_deadline() -> None:
    """Verify tight deadline forces faster speed while relaxed deadline enables slower eco-steaming."""
    optimizer = EcoSpeedOptimizer()
    dist = 2000.0
    # Relaxed deadline: 250 hours (gives freedom to slow-steam at ~10-12 kts)
    scenario_relaxed = OptimizationScenario(
        cargo_demand=40_000.0,
        route_distance=dist,
        deadline_hours=250.0,
    )
    # Tight deadline: 135 hours (requires speed >= 2000/135 = ~14.8 kts)
    scenario_tight = OptimizationScenario(
        cargo_demand=40_000.0,
        route_distance=dist,
        deadline_hours=135.0,
    )

    res_relaxed = optimizer.optimize_speed(scenario_relaxed, vessel_type="Bulk Carrier")
    res_tight = optimizer.optimize_speed(scenario_tight, vessel_type="Bulk Carrier")

    assert res_tight.optimal_speed > res_relaxed.optimal_speed
    assert res_tight.fuel_consumption > res_relaxed.fuel_consumption


def test_eco_speed_deadline_violated_edge_case() -> None:
    """Verify physically impossible deadline returns DEADLINE_VIOLATED with maximum speed."""
    optimizer = EcoSpeedOptimizer()
    # 3000 nm in 10 hours requires 300 knots — impossible for marine cargo ship
    scenario_impossible = OptimizationScenario(
        cargo_demand=40_000.0,
        route_distance=3000.0,
        deadline_hours=10.0,
    )

    result = optimizer.optimize_speed(scenario_impossible, vessel_type="Bulk Carrier")
    assert result.status == OptimizationStatus.DEADLINE_VIOLATED
    assert result.optimal_speed == 17.0  # Max speed for Bulk Carrier
    assert result.delay_hours > 0.0


# =============================================================================
# 5. FLEET STRATEGY OPTIMIZER INTEGRATION & PERSISTENCE TESTS
# =============================================================================

def test_fleet_strategy_optimization_end_to_end(tmp_path: Path) -> None:
    """Verify FleetStrategyOptimizer orchestrates all 3 tiers, baseline comparison, and JSON persistence."""
    optimizer = FleetStrategyOptimizer()
    scenario = OptimizationScenario(
        cargo_demand=180_000.0,
        route_distance=3200.0,
        deadline_hours=260.0,
        scenario_id="SCEN-STRAT-E2E",
        carbon_price=80.0,
        budget=120_000_000.0,
        vessel_class="PANAMAX",
        max_transition_rate=0.40,
    )

    rec = optimizer.optimize_strategy(scenario)
    assert isinstance(rec, FleetStrategyRecommendation)
    assert rec.status == OptimizationStatus.SUCCESS
    assert rec.fuel_estimate > 0.0
    assert rec.cost_estimate > 0.0
    assert rec.emissions_estimate > 0.0
    assert rec.service_reliability is not None
    assert 0.0 <= rec.service_reliability <= 1.0

    # 1. Verify Deployment Plan
    assert rec.deployment_plan
    for route_id, assigned_vessels in rec.deployment_plan.items():
        assert len(assigned_vessels) >= 1
        v = assigned_vessels[0]
        assert "vessel_id" in v
        assert "vessel_class" in v
        assert "fuel_type" in v
        assert "cruising_speed_knots" in v

    # 2. Verify Baseline vs. Optimized Comparison
    b_data = rec.baseline_comparison
    assert "baseline" in b_data and "optimized" in b_data and "deltas" in b_data
    deltas = b_data["deltas"]
    # Green strategy must demonstrate positive savings relative to unoptimized baseline
    assert deltas["fuel_reduction_tons"] >= 0.0
    assert deltas["fuel_reduction_pct"] >= 0.0
    assert deltas["cost_savings_usd"] >= 0.0
    assert deltas["emissions_abated_tons"] >= 0.0

    # 3. Verify Scenario Persistence (Save & Load Roundtrip)
    json_path = tmp_path / "saved_scenario.json"
    saved_path = FleetStrategyOptimizer.save_scenario(rec, json_path)
    assert saved_path.exists()

    loaded_rec = FleetStrategyOptimizer.load_scenario(saved_path)
    assert isinstance(loaded_rec, FleetStrategyRecommendation)
    assert loaded_rec.scenario.scenario_id == "SCEN-STRAT-E2E"
    assert loaded_rec.status == rec.status
    assert loaded_rec.fuel_estimate == rec.fuel_estimate
    assert loaded_rec.cost_estimate == rec.cost_estimate
    assert loaded_rec.capacity_recommendation.recommended_capacity == rec.capacity_recommendation.recommended_capacity
    assert loaded_rec.speed_recommendation.optimal_speed == rec.speed_recommendation.optimal_speed
    assert loaded_rec.service_reliability == rec.service_reliability


def test_deployment_plan_exact_fuel_and_size_count_consumption() -> None:
    """Verify that deployment assignment consumes fleet mix fuel and size counts exactly once (Remaining Issue 4)."""
    optimizer = FleetStrategyOptimizer()
    scenario = OptimizationScenario(
        cargo_demand=150_000.0,
        route_distance=3500.0,
        deadline_hours=300.0,
        scenario_id="SCEN-EXACT-FUEL-DEPLOY",
    )

    # 1. Test decomposed counts: Diesel=4, LNG=3, Methanol=2, Medium=9
    comp = FleetCompositionResult(
        status=OptimizationStatus.SUCCESS,
        fleet_mix={"medium": 9, "diesel": 4, "lng": 3, "methanol": 2},
        total_capacity=405_000.0,
        fuel_consumption=12000.0,
        emissions=20000.0,
        operational_cost=15_000_000.0,
        carbon_cost=400_000.0,
        optimization_score=0.92,
        service_level_achieved=1.0,
        metadata={},
    )
    cap = CapacityOptimizationResult(
        status=OptimizationStatus.SUCCESS,
        vessel_class="PANAMAX",
        recommended_capacity=45_000.0,
        capacity_teu=3200.0,
        utilization_rate=0.85,
        fuel_consumption=12000.0,
        cost=15_000_000.0,
        emissions=20000.0,
        optimal_trips=4,
        metadata={},
    )
    speed = SpeedOptimizationResult(
        status=OptimizationStatus.SUCCESS,
        optimal_speed=14.0,
        estimated_eta=250.0,
        fuel_consumption=12000.0,
        cost=15_000_000.0,
        emissions=20000.0,
        delay_hours=0.0,
        metadata={},
    )

    plan = optimizer._generate_deployment_plan(scenario, comp, cap, speed)

    # Collect all deployed vessels across all routes
    deployed_vessels: list[dict[str, Any]] = [
        v for route_vessels in plan.values() for v in route_vessels
    ]

    assert len(deployed_vessels) == 9, f"Expected 9 deployed vessels, got {len(deployed_vessels)}"

    # Count fuel types across deployed vessels
    fuel_counts: dict[str, int] = {}
    for v in deployed_vessels:
        ft = v["fuel_type"]
        fuel_counts[ft] = fuel_counts.get(ft, 0) + 1

    assert fuel_counts.get("Diesel", 0) == 4, f"Expected 4 Diesel vessels, got {fuel_counts.get('Diesel')}"
    assert fuel_counts.get("LNG", 0) == 3, f"Expected 3 LNG vessels, got {fuel_counts.get('LNG')}"
    assert fuel_counts.get("Methanol", 0) == 2, f"Expected 2 Methanol vessels, got {fuel_counts.get('Methanol')}"

    # Verify all vessel IDs are distinct
    vessel_ids = [v["vessel_id"] for v in deployed_vessels]
    assert len(set(vessel_ids)) == 9, "All deployed vessel IDs must be distinct"

    # 2. Test explicit composite keys: feeder_diesel=2, medium_lng=1, large_methanol=1
    comp_composite = FleetCompositionResult(
        status=OptimizationStatus.SUCCESS,
        fleet_mix={"feeder_diesel": 2, "medium_lng": 1, "large_methanol": 1},
        total_capacity=210_000.0,
        fuel_consumption=8000.0,
        emissions=14000.0,
        operational_cost=10_000_000.0,
        carbon_cost=280_000.0,
        optimization_score=0.90,
        service_level_achieved=1.0,
        metadata={},
    )
    plan_comp = optimizer._generate_deployment_plan(scenario, comp_composite, cap, speed)
    deployed_comp = [v for r_v in plan_comp.values() for v in r_v]

    assert len(deployed_comp) == 4
    composite_pairs = [(v["vessel_class"], v["fuel_type"]) for v in deployed_comp]
    assert composite_pairs.count(("Feeder", "Diesel")) == 2
    assert composite_pairs.count(("Medium", "LNG")) == 1
    assert composite_pairs.count(("Large", "Methanol")) == 1


def test_fleet_composition_full_5_fuel_decision_space_enumeration() -> None:
    """Verify FleetCompositionOptimizer systematically enumerates the full 5-fuel decision space.

    Problem Finding 2:
    The candidate generator must systematically enumerate all integer allocations:
        y_D + y_L + y_M + y_H + y_A = N
    subject to:
        sum_{alt} y_alt <= N * r_max
    including complex multi-fuel blends (e.g., Diesel+LNG+Hydrogen, Diesel+Methanol+Ammonia,
    LNG+Methanol+Hydrogen) rather than just single-fuel alternatives or a single 2-fuel blend.
    """
    optimizer = FleetCompositionOptimizer()

    # 1. Exact mathematical combinatorial enumeration check:
    # For N=3, max_alt=3: total allocations = C(3 + 5 - 1, 5 - 1) = C(7, 4) = 35
    candidates_3 = optimizer._generate_fuel_mix_candidates(total_vessels=3, max_alt=3)
    assert len(candidates_3) == 35, f"Expected 35 allocations, got {len(candidates_3)}"

    # Verify each candidate satisfies the governing constraints
    seen = set()
    for c in candidates_3:
        y_D = c["diesel"]
        y_L = c["lng"]
        y_M = c["methanol"]
        y_H = c["hydrogen"]
        y_A = c["ammonia"]
        assert all(v >= 0 for v in (y_D, y_L, y_M, y_H, y_A))
        assert y_D + y_L + y_M + y_H + y_A == 3
        key = (y_D, y_L, y_M, y_H, y_A)
        assert key not in seen, f"Duplicate candidate found: {key}"
        seen.add(key)

    # Verify multi-fuel blends beyond 2 fuels are present
    assert {"diesel": 1, "lng": 1, "methanol": 0, "hydrogen": 1, "ammonia": 0} in candidates_3  # Diesel + LNG + Hydrogen
    assert {"diesel": 1, "lng": 0, "methanol": 1, "hydrogen": 0, "ammonia": 1} in candidates_3  # Diesel + Methanol + Ammonia
    assert {"diesel": 0, "lng": 1, "methanol": 1, "hydrogen": 1, "ammonia": 0} in candidates_3  # LNG + Methanol + Hydrogen
    assert {"diesel": 0, "lng": 0, "methanol": 1, "hydrogen": 1, "ammonia": 1} in candidates_3  # Methanol + Hydrogen + Ammonia

    # 2. Transition rate constraint enforcement:
    # For N=4, max_alt=2: sum(alt) <= 2
    candidates_4_2 = optimizer._generate_fuel_mix_candidates(total_vessels=4, max_alt=2)
    # n_alt=0: 1 (4 diesel)
    # n_alt=1: C(1+3, 3) = 4
    # n_alt=2: C(2+3, 3) = 10
    # Total = 1 + 4 + 10 = 15 candidates
    assert len(candidates_4_2) == 15, f"Expected 15 allocations for N=4, max_alt=2, got {len(candidates_4_2)}"
    for c in candidates_4_2:
        alt_sum = c["lng"] + c["methanol"] + c["hydrogen"] + c["ammonia"]
        assert alt_sum <= 2
        assert c["diesel"] + alt_sum == 4

    # 3. End-to-end optimization execution across full decision space
    scenario = OptimizationScenario(
        cargo_demand=120_000.0,
        route_distance=3000.0,
        deadline_hours=240.0,
        scenario_id="SCEN-5FUEL-TEST",
        budget=150_000_000.0,
        carbon_price=80.0,
        max_transition_rate=0.60,
    )
    result = optimizer.optimize_composition(scenario)
    assert result.status == OptimizationStatus.SUCCESS
    assert result.metadata["iterations"] > 0
    # Verify fleet mix contains all 5 fuel keys and size keys
    for k in ("feeder", "medium", "large", "diesel", "lng", "methanol", "hydrogen", "ammonia"):
        assert k in result.fleet_mix
    total_fuel_vessels = sum(result.fleet_mix[f] for f in ("diesel", "lng", "methanol", "hydrogen", "ammonia"))
    total_size_vessels = sum(result.fleet_mix[s] for s in ("feeder", "medium", "large"))
    assert total_fuel_vessels == total_size_vessels


def test_fleet_composition_qpso_solver() -> None:
    """Verify FleetCompositionOptimizer successfully executes with solver='qpso'.

    Tests:
    1. Direct invocation with solver='qpso'.
    2. Verification of QPSO metadata (solver_name, iterations, n_evaluations, convergence_score).
    3. Structural validity of output: all 8 keys present, size and fuel vessel counts match.
    4. Compliance with cargo demand and max transition rate constraints.
    """
    optimizer = FleetCompositionOptimizer()
    scenario = OptimizationScenario(
        cargo_demand=120_000.0,
        route_distance=3000.0,
        deadline_hours=240.0,
        scenario_id="SCEN-QPSO-COMP",
        budget=150_000_000.0,
        carbon_price=80.0,
        max_transition_rate=0.50,
    )

    result = optimizer.optimize_composition(scenario, solver="qpso", population_size=20, max_iterations=30, seed=42)
    assert isinstance(result, FleetCompositionResult)
    assert result.status == OptimizationStatus.SUCCESS
    assert result.metadata["solver_name"] == "qpso"
    assert result.metadata["iterations"] == 30
    assert result.metadata["n_evaluations"] == 20 * 30
    assert "optimization_trace" in result.metadata
    assert len(result.metadata["optimization_trace"]) == 30

    # Verify keys
    for k in ("feeder", "medium", "large", "diesel", "lng", "methanol", "hydrogen", "ammonia"):
        assert k in result.fleet_mix

    total_size = result.fleet_mix["feeder"] + result.fleet_mix["medium"] + result.fleet_mix["large"]
    total_fuel = sum(result.fleet_mix[f] for f in ("diesel", "lng", "methanol", "hydrogen", "ammonia"))
    assert total_size == total_fuel
    assert total_size > 0

    # Verify transition rate constraint
    alt_count = total_fuel - result.fleet_mix["diesel"]
    assert alt_count <= int(total_size * scenario.max_transition_rate)


def test_fleet_strategy_qpso_integrated_solver() -> None:
    """Verify FleetStrategyOptimizer orchestrates end-to-end with solver='qpso'.

    Tests:
    1. FleetStrategyOptimizer.optimize_strategy(scenario, solver='qpso').
    2. Recommendation metadata accurately tags 'qpso' composition solver.
    3. All downstream tiers (Capacity, Speed, Deployment, Reliability, Demand) succeed seamlessly.
    """
    optimizer = FleetStrategyOptimizer()
    scenario = OptimizationScenario(
        cargo_demand=150_000.0,
        route_distance=3500.0,
        deadline_hours=260.0,
        scenario_id="SCEN-QPSO-STRAT",
        budget=180_000_000.0,
        carbon_price=80.0,
        vessel_class="PANAMAX",
        max_transition_rate=0.40,
    )

    rec = optimizer.optimize_strategy(scenario, solver="qpso", population_size=20, max_iterations=25, seed=101)
    assert isinstance(rec, FleetStrategyRecommendation)
    assert rec.status == OptimizationStatus.SUCCESS
    assert rec.summary["composition_solver"] == "qpso"
    assert rec.fleet_mix.metadata["solver_name"] == "qpso"
    assert rec.cost_estimate > 0.0
    assert rec.fuel_estimate > 0.0
    assert rec.emissions_estimate > 0.0
    assert rec.speed_recommendation.optimal_speed > 0.0
    assert rec.demand_metrics.satisfaction_percentage >= 95.0


def test_deployment_plan_fails_validation_on_size_fuel_count_mismatch() -> None:
    """Verify _generate_deployment_plan strictly raises ValueError when size and fuel counts mismatch.

    Problem Finding 4:
    Silent padding (e.g. extending size_pool with default class or fuel_pool with Diesel)
    distorts the optimized fuel distribution. The deployment planner must fail validation
    with an informative error whenever sum(vessel counts) != sum(fuel counts).
    """
    optimizer = FleetStrategyOptimizer()
    scenario = OptimizationScenario(
        cargo_demand=150_000.0,
        route_distance=3500.0,
        deadline_hours=260.0,
        scenario_id="SCEN-MISMATCH-TEST",
    )
    cap = CapacityOptimizationResult(
        status=OptimizationStatus.SUCCESS,
        vessel_class="PANAMAX",
        recommended_capacity=45_000.0,
        capacity_teu=3200.0,
        utilization_rate=0.85,
        fuel_consumption=12000.0,
        cost=15_000_000.0,
        emissions=20000.0,
        optimal_trips=4,
        metadata={},
    )
    speed = SpeedOptimizationResult(
        status=OptimizationStatus.SUCCESS,
        optimal_speed=14.0,
        estimated_eta=250.0,
        fuel_consumption=12000.0,
        cost=15_000_000.0,
        emissions=20000.0,
        delay_hours=0.0,
        metadata={},
    )

    # Case 1: More size vessels (5) than fuel vessels (3)
    comp_mismatch_sizes = FleetCompositionResult(
        status=OptimizationStatus.SUCCESS,
        fleet_mix={"feeder": 3, "medium": 2, "diesel": 2, "lng": 1},  # 5 sizes vs 3 fuels
        total_capacity=100_000.0,
        fuel_consumption=5000.0,
        emissions=8000.0,
        operational_cost=5_000_000.0,
        carbon_cost=100_000.0,
        optimization_score=0.90,
        service_level_achieved=1.0,
        metadata={},
    )

    with pytest.raises(ValueError, match="Fleet mix validation failed.*total vessel size counts.*does not match total fuel counts"):
        optimizer._generate_deployment_plan(scenario, comp_mismatch_sizes, cap, speed)

    # Case 2: More fuel vessels (6) than size vessels (4)
    comp_mismatch_fuels = FleetCompositionResult(
        status=OptimizationStatus.SUCCESS,
        fleet_mix={"medium": 4, "diesel": 2, "lng": 2, "methanol": 2},  # 4 sizes vs 6 fuels
        total_capacity=100_000.0,
        fuel_consumption=5000.0,
        emissions=8000.0,
        operational_cost=5_000_000.0,
        carbon_cost=100_000.0,
        optimization_score=0.90,
        service_level_achieved=1.0,
        metadata={},
    )

    with pytest.raises(ValueError, match="Fleet mix validation failed.*total vessel size counts.*does not match total fuel counts"):
        optimizer._generate_deployment_plan(scenario, comp_mismatch_fuels, cap, speed)


# =============================================================================
# 6. ADVANCED REQUIREMENT & SENSITIVITY VERIFICATION TESTS
# =============================================================================

def test_high_demand_scaling() -> None:
    """Verify fleet composition scales up vessel numbers to satisfy high demand feasibly."""
    optimizer = FleetCompositionOptimizer()
    scenario = OptimizationScenario(
        cargo_demand=3_000_000.0,
        route_distance=4000.0,
        deadline_hours=300.0,
        scenario_id="SCEN-HIGH-DEMAND",
        budget=400_000_000.0,
        carbon_price=80.0,
        max_transition_rate=0.50,
    )

    result = optimizer.optimize_composition(scenario, max_fleet_size=25)
    assert result.status == OptimizationStatus.SUCCESS
    total_ships = sum(result.fleet_mix[k] for k in ("feeder", "medium", "large"))
    assert total_ships >= 3
    assert result.total_capacity >= scenario.cargo_demand * scenario.service_level


def test_fleet_availability_limits() -> None:
    """Verify fleet composition respects maximum available vessel counts per size category."""
    optimizer = FleetCompositionOptimizer()
    scenario = OptimizationScenario(
        cargo_demand=400_000.0,
        route_distance=3000.0,
        deadline_hours=250.0,
        budget=500_000_000.0,
    )
    result = optimizer.optimize_composition(scenario, max_fleet_size=30)
    assert result.status == OptimizationStatus.SUCCESS
    # From VESSEL_SPECS: feeder max 20, medium max 15, large max 10
    assert result.fleet_mix["feeder"] <= 20
    assert result.fleet_mix["medium"] <= 15
    assert result.fleet_mix["large"] <= 10


def test_vessel_class_variations_and_monotonicity() -> None:
    """Verify capacity sizing across all naval architecture classes maintains physical monotonicity."""
    optimizer = VesselCapacityOptimizer()
    classes = ["FEEDER", "HANDYMAX", "PANAMAX", "POST_PANAMAX", "CAPESIZE"]
    results = {}

    for vc in classes:
        scenario = OptimizationScenario(
            cargo_demand=100_000.0,
            route_distance=3000.0,
            deadline_hours=250.0,
            vessel_class=vc,
        )
        res = optimizer.optimize_capacity(scenario)
        assert res.status == OptimizationStatus.SUCCESS
        assert res.vessel_class == vc
        results[vc] = res.recommended_capacity

    # Verify monotonic capacity growth across naval classes
    assert results["FEEDER"] < results["HANDYMAX"]
    assert results["HANDYMAX"] <= results["PANAMAX"]
    assert results["PANAMAX"] < results["POST_PANAMAX"]
    assert results["POST_PANAMAX"] < results["CAPESIZE"]


def test_speed_deadline_and_weather_penalty() -> None:
    """Verify weather severity factor increases hydrodynamic drag, fuel consumption, and total cost."""
    optimizer = EcoSpeedOptimizer()
    scen_calm = OptimizationScenario(
        cargo_demand=50_000.0,
        route_distance=3000.0,
        deadline_hours=260.0,
        weather_factor=1.00,
    )
    scen_rough = OptimizationScenario(
        cargo_demand=50_000.0,
        route_distance=3000.0,
        deadline_hours=260.0,
        weather_factor=1.25,
    )

    res_calm = optimizer.optimize_speed(scen_calm, vessel_type="Bulk Carrier")
    res_rough = optimizer.optimize_speed(scen_rough, vessel_type="Bulk Carrier")

    assert res_calm.status == OptimizationStatus.SUCCESS
    assert res_rough.status == OptimizationStatus.SUCCESS
    assert res_rough.fuel_consumption > res_calm.fuel_consumption
    assert res_rough.cost > res_calm.cost


def test_deterministic_reproducibility() -> None:
    """Verify deterministic MIP solver produces 100% identical results on repeated runs."""
    optimizer = FleetCompositionOptimizer()
    scenario = OptimizationScenario(
        cargo_demand=150_000.0,
        route_distance=3500.0,
        deadline_hours=280.0,
        scenario_id="SCEN-REPRODUCIBLE",
        budget=100_000_000.0,
        carbon_price=80.0,
        max_transition_rate=0.40,
    )

    run_1 = optimizer.optimize_composition(scenario, solver="deterministic")
    run_2 = optimizer.optimize_composition(scenario, solver="deterministic")

    assert run_1.status == run_2.status
    assert run_1.fleet_mix == run_2.fleet_mix
    assert run_1.total_capacity == run_2.total_capacity
    assert run_1.fuel_consumption == run_2.fuel_consumption
    assert run_1.operational_cost == run_2.operational_cost
    assert run_1.carbon_cost == run_2.carbon_cost
    assert run_1.optimization_score == run_2.optimization_score


def test_port_limits_constraint() -> None:
    """Verify scenario.port_limits constrains recommended vessel deadweight capacity."""
    optimizer = VesselCapacityOptimizer()
    # Panamax normally sizes up to 55,000 DWT
    scenario_restricted = OptimizationScenario(
        cargo_demand=200_000.0,
        route_distance=3500.0,
        deadline_hours=300.0,
        vessel_class="PANAMAX",
        port_limits={"max_dwt": 32_000.0},
    )

    result = optimizer.optimize_capacity(scenario_restricted)
    assert result.status == OptimizationStatus.SUCCESS
    assert result.recommended_capacity <= 32_000.0


def test_speed_optimizer_metadata_cost_decomposition() -> None:
    """Verify SpeedOptimizationResult metadata provides full economic traceability."""
    optimizer = EcoSpeedOptimizer()
    scenario = OptimizationScenario(
        cargo_demand=50_000.0,
        route_distance=3000.0,
        deadline_hours=260.0,
        carbon_price=80.0,
    )

    res = optimizer.optimize_speed(scenario, vessel_type="Bulk Carrier")
    assert res.status == OptimizationStatus.SUCCESS
    meta = res.metadata
    assert "bunker_cost" in meta
    assert "carbon_cost" in meta
    assert "charter_cost" in meta
    assert "delay_cost" in meta
    assert "total_cost" in meta
    assert meta["bunker_cost"] > 0.0
    assert meta["carbon_cost"] > 0.0
    assert meta["charter_cost"] > 0.0
    # Bunker + Carbon + Charter + Delay matches total_cost
    summed = meta["bunker_cost"] + meta["carbon_cost"] + meta["charter_cost"] + meta["delay_cost"]
    assert abs(summed - meta["total_cost"]) < 0.05





