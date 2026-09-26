"""Comprehensive test suite for Operational Reliability and Cargo Demand Satisfaction.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Coverage:
1. Demand Satisfaction Engine (zero demand, normal demand, demand surge, insufficient capacity, cap at 100%).
2. Schedule Reliability Engine (all on-time, mixed delays, missed voyages, observation cardinality mismatch).
3. Hard Constraints and Optimization Status (DEMAND_NOT_MET, RELIABILITY_TOO_LOW, DEADLINE_VIOLATED).
4. Environmental Disruptions (severe weather degradation, port congestion degradation).
5. Reliability-Aware Route Deployment Planning (buffer capacity, redundancy factor, corridor scores).
6. End-to-End Orchestrator Integration and Scenario Persistence Roundtrip.
"""

from pathlib import Path
import pytest

from contracts.schemas import (
    DemandSatisfactionMetrics,
    FleetStrategyRecommendation,
    OptimizationScenario,
    OptimizationStatus,
    ReliabilityMetrics,
)
from src.operations.demand_satisfaction_engine import (
    CargoDemandSatisfactionEngine,
    DemandSatisfactionEngine,
)
from src.operations.reliability_engine import ScheduleReliabilityEngine
from src.optimization.fleet_strategy_optimizer import FleetStrategyOptimizer


# =============================================================================
# 1. CARGO DEMAND SATISFACTION ENGINE TESTS (Mandatory Tests 1-5)
# =============================================================================

def test_zero_demand() -> None:
    """Mandatory Test 1: Zero demand -> satisfaction=1.0, unserved=0."""
    engine = DemandSatisfactionEngine()
    scenario = OptimizationScenario(
        cargo_demand=0.0,
        route_distance=1000.0,
        deadline_hours=100.0,
    )
    metrics = engine.evaluate_demand_satisfaction(scenario)
    assert isinstance(metrics, DemandSatisfactionMetrics)
    assert metrics.demand_satisfaction_rate == 1.0
    assert metrics.satisfaction_percentage == 100.0
    assert metrics.unserved_cargo == 0.0
    assert metrics.service_gap == 0.0
    assert metrics.is_satisfied is True
    assert metrics.status == "SATISFIED"


def test_normal_demand() -> None:
    """Mandatory Test 2: Normal demand -> exact ratio delivered/required."""
    engine = DemandSatisfactionEngine()
    scenario = OptimizationScenario(
        cargo_demand=200_000.0,
        route_distance=3000.0,
        deadline_hours=200.0,
        service_level=0.90,
    )

    metrics = engine.evaluate_demand_satisfaction(scenario, delivered_cargo=150_000.0)
    assert metrics.required_demand == 200_000.0
    assert metrics.delivered_cargo == 150_000.0
    assert metrics.demand_satisfaction_rate == 0.75
    assert metrics.satisfaction_percentage == 75.0
    assert metrics.unserved_cargo == 50_000.0
    assert round(metrics.service_level_gap, 2) == 0.15
    assert metrics.service_gap == 0.25  # max(0, 1 - 0.75)
    assert metrics.is_satisfied is False
    assert metrics.status == "UNSATISFIED"


def test_demand_surge() -> None:
    """Mandatory Test 3: Forecasted demand surge -> unserved volume and satisfaction."""
    engine = DemandSatisfactionEngine()
    scenario = OptimizationScenario(
        cargo_demand=100_000.0,
        route_distance=2000.0,
        deadline_hours=150.0,
        service_level=0.95,
        forecasted_demand=150_000.0,
    )

    # Delivered 120,000 -> 120,000 / 150,000 = 0.80 (below 0.95 target service level)
    metrics = engine.evaluate_demand_satisfaction(scenario, delivered_cargo=120_000.0)
    assert metrics.required_demand == 150_000.0
    assert metrics.demand_satisfaction_rate == 0.80
    assert metrics.unserved_cargo == 30_000.0
    assert metrics.service_gap == 0.20
    assert metrics.is_satisfied is False


def test_insufficient_capacity() -> None:
    """Mandatory Test 4: Capacity below requirement -> satisfaction < 1.0."""
    engine = DemandSatisfactionEngine()
    scenario = OptimizationScenario(
        cargo_demand=300_000.0,
        route_distance=3000.0,
        deadline_hours=200.0,
        service_level=0.95,
    )
    # 2 vessels: 20k DWT each, 4 voyages = 160k DWT * 0.85 standard utilization = 136k tons delivered
    deployment = {
        "ROUTE-1": [
            {"vessel_id": "V1", "allocated_capacity_dwt": 20_000.0, "annual_voyages": 4},
            {"vessel_id": "V2", "allocated_capacity_dwt": 20_000.0, "annual_voyages": 4},
        ]
    }
    metrics = engine.evaluate_demand_satisfaction(scenario, deployment_plan=deployment)
    assert metrics.delivered_cargo == 136_000.0
    assert metrics.demand_satisfaction_rate < 1.0
    assert metrics.unserved_cargo == 164_000.0
    assert metrics.is_satisfied is False


def test_satisfaction_capped_at_100() -> None:
    """Mandatory Test 5: Delivered > required -> strictly capped at 1.0 (100%)."""
    engine = DemandSatisfactionEngine()
    scenario = OptimizationScenario(
        cargo_demand=100_000.0,
        route_distance=3000.0,
        deadline_hours=200.0,
        service_level=0.95,
    )

    metrics = engine.evaluate_demand_satisfaction(scenario, delivered_cargo=250_000.0)
    assert metrics.required_demand == 100_000.0
    assert metrics.delivered_cargo == 250_000.0
    assert metrics.demand_satisfaction_rate == 1.0  # Must be strictly capped at 1.0
    assert metrics.satisfaction_percentage == 100.0
    assert metrics.unserved_cargo == 0.0
    assert metrics.service_level_gap == 0.0
    assert metrics.service_gap == 0.0
    assert metrics.is_satisfied is True
    assert metrics.status == "SATISFIED"


# =============================================================================
# 2. SCHEDULE RELIABILITY ENGINE TESTS (Mandatory Tests 6-9)
# =============================================================================

def test_all_on_time_voyages() -> None:
    """Mandatory Test 6: All delays 0 -> on_time_rate=1.0, reliability=100."""
    engine = ScheduleReliabilityEngine()
    scenario = OptimizationScenario(
        cargo_demand=100_000.0,
        route_distance=3000.0,
        deadline_hours=250.0,
        weather_factor=1.0,
        port_delay_factor=1.0,
    )

    delays = [0.0] * 8
    metrics = engine.evaluate_schedule_reliability(
        scenario=scenario,
        simulated_delays=delays,
        missed_voyages=0,
        total_voyages=8,
    )

    assert isinstance(metrics, ReliabilityMetrics)
    assert metrics.reliability_score == 100.0
    assert metrics.on_time_arrival_rate == 1.0
    assert metrics.on_time_rate == 1.0
    assert metrics.delay_rate == 0.0
    assert metrics.average_delay_hours == 0.0
    assert metrics.missed_voyages == 0
    assert metrics.on_time_voyages == 8
    assert metrics.delayed_voyages == 0
    assert metrics.score_breakdown["on_time_component"] == 100.0
    assert metrics.score_breakdown["delay_penalty"] == 0.0
    assert metrics.score_breakdown["missed_voyage_penalty"] == 0.0


def test_mixed_delay_voyages() -> None:
    """Mandatory Test 7: Delays > 0 -> proportional delay penalty."""
    engine = ScheduleReliabilityEngine()
    scenario = OptimizationScenario(
        cargo_demand=100_000.0,
        route_distance=3000.0,
        deadline_hours=200.0,
    )
    # 5 voyages: 3 on time (0h), 2 delayed (20h and 40h) -> avg delay = 12.0h
    delays = [0.0, 0.0, 0.0, 20.0, 40.0]
    metrics = engine.evaluate_schedule_reliability(
        scenario=scenario,
        simulated_delays=delays,
        missed_voyages=0,
        total_voyages=5,
    )
    assert metrics.total_voyages == 5
    assert metrics.on_time_voyages == 3
    assert metrics.delayed_voyages == 2
    assert metrics.on_time_rate == 0.60
    assert metrics.delay_rate == 0.40
    assert metrics.average_delay_hours == 12.0
    assert 0.0 < metrics.reliability_score < 100.0
    assert metrics.score_breakdown["delay_penalty"] > 0.0


def test_missed_voyages() -> None:
    """Mandatory Test 8: Missed voyages > 0 -> normalized penalty scale-invariant."""
    engine = ScheduleReliabilityEngine()
    # 1 missed out of 5 voyages (20%) vs 2 missed out of 10 voyages (20%)
    m1 = engine.evaluate_from_observations(
        total_voyages=5,
        on_time_voyages=4,
        missed_voyages=1,
        average_delay_hours=0.0,
        deadline_hours=200.0,
    )
    m2 = engine.evaluate_from_observations(
        total_voyages=10,
        on_time_voyages=8,
        missed_voyages=2,
        average_delay_hours=0.0,
        deadline_hours=200.0,
    )
    assert m1.score_breakdown["missed_voyage_penalty"] == m2.score_breakdown["missed_voyage_penalty"]
    assert m1.reliability_score == m2.reliability_score


def test_observation_cardinality_mismatch() -> None:
    """Mandatory Test 9: len(delays) != total_voyages raises ValueError."""
    engine = ScheduleReliabilityEngine()
    scenario = OptimizationScenario(
        cargo_demand=100_000.0,
        route_distance=3000.0,
        deadline_hours=200.0,
    )

    # len(delays)=4, total_voyages=16 -> must raise ValueError
    with pytest.raises(ValueError, match="Observation cardinality mismatch"):
        engine.evaluate_schedule_reliability(
            scenario=scenario,
            simulated_delays=[0.0, 5.0, 0.0, 2.0],
            total_voyages=16,
        )

    # Inconsistent observations in evaluate_from_observations: on_time + missed > total
    with pytest.raises(ValueError, match="Inconsistent observation counts"):
        engine.evaluate_from_observations(
            total_voyages=10,
            on_time_voyages=8,
            missed_voyages=5,
            average_delay_hours=10.0,
            deadline_hours=200.0,
        )


# =============================================================================
# 3. OPTIMIZER HARD CONSTRAINTS & STATUS (Mandatory Tests 10-11, 14)
# =============================================================================

def test_infeasible_demand_status() -> None:
    """Mandatory Test 10: Demand satisfaction below service level -> DEMAND_UNSATISFIABLE / DEMAND_NOT_MET."""
    optimizer = FleetStrategyOptimizer()
    scenario = OptimizationScenario(
        cargo_demand=50_000_000.0,  # Far exceeds maximum possible fleet capacity
        route_distance=4000.0,
        deadline_hours=500.0,
        service_level=0.95,
    )
    rec = optimizer.optimize_strategy(scenario)
    assert rec.status == OptimizationStatus.DEMAND_UNSATISFIABLE
    assert rec.summary.get("failure_reason") == "DEMAND_NOT_MET"


def test_infeasible_reliability_status() -> None:
    """Mandatory Test 11: Reliability below target -> INFEASIBLE / RELIABILITY_TOO_LOW."""
    optimizer = FleetStrategyOptimizer()
    scenario = OptimizationScenario(
        cargo_demand=100_000.0,
        route_distance=4000.0,
        deadline_hours=400.0,
        target_reliability=99.9,  # Unattainable under adverse weather and congestion
        weather_factor=1.35,
        port_delay_factor=1.6,
    )
    rec = optimizer.optimize_strategy(scenario)
    assert rec.status == OptimizationStatus.INFEASIBLE
    assert rec.summary.get("failure_reason") == "RELIABILITY_TOO_LOW"


def test_impossible_deadline_status() -> None:
    """Mandatory Test 14: Impossible deadline -> DEADLINE_VIOLATED."""
    optimizer = FleetStrategyOptimizer()
    scenario = OptimizationScenario(
        cargo_demand=100_000.0,
        route_distance=5000.0,
        deadline_hours=20.0,  # 5000 nm in 20h is physically impossible
    )
    rec = optimizer.optimize_strategy(scenario)
    assert rec.status in {OptimizationStatus.INFEASIBLE, OptimizationStatus.DEADLINE_VIOLATED}
    assert rec.summary.get("failure_reason") == "DEADLINE_VIOLATED"


# =============================================================================
# 4. ENVIRONMENTAL DEGRADATION SENSITIVITY (Mandatory Tests 12-13)
# =============================================================================

def test_severe_weather_degradation() -> None:
    """Mandatory Test 12: High weather factor reduces reliability score."""
    engine = ScheduleReliabilityEngine()
    scenario_calm = OptimizationScenario(
        cargo_demand=100_000.0,
        route_distance=3000.0,
        deadline_hours=220.0,
        weather_factor=1.0,
    )
    scenario_storm = OptimizationScenario(
        cargo_demand=100_000.0,
        route_distance=3000.0,
        deadline_hours=220.0,
        weather_factor=1.35,
    )
    calm = engine.evaluate_schedule_reliability(scenario_calm)
    storm = engine.evaluate_schedule_reliability(scenario_storm)
    assert storm.reliability_score < calm.reliability_score
    assert storm.average_delay_hours > calm.average_delay_hours


def test_port_congestion_degradation() -> None:
    """Mandatory Test 13: High port delay factor increases average delay."""
    engine = ScheduleReliabilityEngine()
    scenario_free = OptimizationScenario(
        cargo_demand=80_000.0,
        route_distance=2500.0,
        deadline_hours=180.0,
        port_delay_factor=1.0,
    )
    scenario_congested = OptimizationScenario(
        cargo_demand=80_000.0,
        route_distance=2500.0,
        deadline_hours=180.0,
        port_delay_factor=1.8,
    )
    free = engine.evaluate_schedule_reliability(scenario_free)
    congested = engine.evaluate_schedule_reliability(scenario_congested)
    assert congested.average_delay_hours > free.average_delay_hours
    assert congested.reliability_score <= free.reliability_score


# =============================================================================
# 5. RELIABILITY-AWARE DEPLOYMENT & BUFFER CONFIGURATION
# =============================================================================

def test_configurable_buffer_percentage() -> None:
    """Verify configurable buffer_percentage scales spare capacity and buffers."""
    optimizer = FleetStrategyOptimizer()
    scenario = OptimizationScenario(
        cargo_demand=100_000.0,
        route_distance=3000.0,
        deadline_hours=250.0,
        buffer_percentage=0.25,
    )
    rec = optimizer.optimize_strategy(scenario)
    assert rec.status == OptimizationStatus.SUCCESS
    for r_id, vessels in rec.deployment_plan.items():
        for v in vessels:
            assert v.get("buffer_percentage") == 25.0
            assert v.get("redundancy_factor") == 1.25
            assert "port_turnaround_buffer_hours" in v
            assert "weather_buffer_hours" in v
            assert "deadline_buffer_hours" in v


def test_route_level_metrics_dictionary_access() -> None:
    """Verify route-level metrics dictionary structure and property aliases."""
    engine = ScheduleReliabilityEngine()
    scenario = OptimizationScenario(
        cargo_demand=200_000.0,
        route_distance=3500.0,
        deadline_hours=260.0,
        routes=(
            {"route_id": "ROUTE-ASIA-EU", "distance_nm": 4000.0},
            {"route_id": "ROUTE-TRANS-PACIFIC", "distance_nm": 5500.0},
        ),
    )
    metrics = engine.evaluate_schedule_reliability(scenario)
    assert "ROUTE-ASIA-EU" in metrics.route_reliability
    assert "ROUTE-TRANS-PACIFIC" in metrics.route_reliability
    assert "ROUTE-ASIA-EU" in metrics.route_delays
    assert "ROUTE-TRANS-PACIFIC" in metrics.route_delays
    assert "ROUTE-ASIA-EU" in metrics.route_demands
    assert "ROUTE-TRANS-PACIFIC" in metrics.route_demands
    assert "ROUTE-ASIA-EU" in metrics.route_utilizations
    assert "ROUTE-TRANS-PACIFIC" in metrics.route_utilizations

    # Property aliases
    assert metrics.route_delay == metrics.route_delays
    assert metrics.route_demand == metrics.route_demands
    assert metrics.route_utilization == metrics.route_utilizations
    assert metrics.on_time_rate == metrics.on_time_arrival_rate
    assert metrics.delayed_voyages >= 0
    assert metrics.on_time_voyages <= metrics.total_voyages


# =============================================================================
# 6. SCENARIO PERSISTENCE & DATA CONTRACT INTEGRITY
# =============================================================================

def test_strategy_recommendation_serialization_roundtrip_with_phase2(tmp_path: Path) -> None:
    """Verify full save and load roundtrip with ReliabilityMetrics and DemandSatisfactionMetrics."""
    optimizer = FleetStrategyOptimizer()
    scenario = OptimizationScenario(
        cargo_demand=150_000.0,
        route_distance=4000.0,
        deadline_hours=320.0,
        scenario_id="SCEN-PHASE2-ROUNDTRIP",
        target_reliability=85.0,
        port_delay_factor=1.05,
        buffer_percentage=0.18,
    )

    rec = optimizer.optimize_strategy(scenario)
    assert rec.reliability_metrics is not None
    assert rec.demand_metrics is not None

    file_path = tmp_path / "phase2_strategy.json"
    saved_path = FleetStrategyOptimizer.save_scenario(rec, file_path)
    assert saved_path.exists()

    loaded = FleetStrategyOptimizer.load_scenario(saved_path)
    assert loaded.status == rec.status
    assert loaded.scenario.scenario_id == "SCEN-PHASE2-ROUNDTRIP"
    assert loaded.scenario.target_reliability == 85.0
    assert loaded.scenario.port_delay_factor == 1.05
    assert loaded.scenario.buffer_percentage == 0.18

    # Check metrics fidelity
    assert loaded.reliability_metrics is not None
    assert loaded.reliability_metrics.reliability_score == rec.reliability_metrics.reliability_score
    assert loaded.reliability_metrics.score_breakdown == rec.reliability_metrics.score_breakdown
    assert loaded.reliability_metrics.route_delays == rec.reliability_metrics.route_delays
    assert loaded.reliability_metrics.route_demands == rec.reliability_metrics.route_demands
    assert loaded.reliability_metrics.route_utilizations == rec.reliability_metrics.route_utilizations

    assert loaded.demand_metrics is not None
    assert loaded.demand_metrics.required_demand == rec.demand_metrics.required_demand
    assert loaded.demand_metrics.demand_satisfaction_rate == rec.demand_metrics.demand_satisfaction_rate
    assert loaded.demand_metrics.is_satisfied == rec.demand_metrics.is_satisfied
    assert loaded.demand_metrics.service_gap == rec.demand_metrics.service_gap
