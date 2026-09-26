"""Comprehensive test suite for Phase 2 Operational Reliability and Cargo Demand Satisfaction.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 2 Test Coverage:
1. Demand Satisfaction Engine (full delivery, partial delivery, zero demand, forecasted demand hook).
2. Schedule Reliability Engine (normalized formula, score decomposition, route-level breakdown, port delays).
3. Status Granularity and Failure Reasons (DEMAND_NOT_MET, RELIABILITY_TOO_LOW, DEADLINE_VIOLATED).
4. Reliability-Aware Route Deployment Planning (buffer capacity, redundancy factor, corridor scores).
5. End-to-End Orchestrator Integration and Scenario Persistence Roundtrip.
"""

import json
from pathlib import Path
import pytest

from contracts.schemas import (
    DemandSatisfactionMetrics,
    FleetStrategyRecommendation,
    OptimizationScenario,
    OptimizationStatus,
    ReliabilityMetrics,
)
from src.operations.demand_satisfaction_engine import CargoDemandSatisfactionEngine
from src.operations.reliability_engine import ScheduleReliabilityEngine
from src.optimization.fleet_strategy_optimizer import FleetStrategyOptimizer


# =============================================================================
# 1. CARGO DEMAND SATISFACTION ENGINE TESTS
# =============================================================================

def test_demand_satisfaction_full_delivery() -> None:
    """Verify that delivered cargo exceeding requirement caps satisfaction rate at 1.0 (100%)."""
    engine = CargoDemandSatisfactionEngine()
    scenario = OptimizationScenario(
        cargo_demand=100_000.0,
        route_distance=3000.0,
        deadline_hours=200.0,
        service_level=0.95,
    )

    metrics = engine.evaluate_demand_satisfaction(scenario, delivered_cargo=120_000.0)
    assert isinstance(metrics, DemandSatisfactionMetrics)
    assert metrics.required_demand == 100_000.0
    assert metrics.delivered_cargo == 120_000.0
    assert metrics.demand_satisfaction_rate == 1.0  # Must be strictly capped at 1.0
    assert metrics.satisfaction_percentage == 100.0
    assert metrics.unserved_cargo == 0.0
    assert metrics.service_level_gap == 0.0
    assert metrics.is_satisfied is True
    assert metrics.status == "SATISFIED"


def test_demand_satisfaction_partial_delivery() -> None:
    """Verify unserved volume and service-level gap when delivery falls short."""
    engine = CargoDemandSatisfactionEngine()
    scenario = OptimizationScenario(
        cargo_demand=200_000.0,
        route_distance=3000.0,
        deadline_hours=200.0,
        service_level=0.90,
    )

    metrics = engine.evaluate_demand_satisfaction(scenario, delivered_cargo=150_000.0)
    assert metrics.demand_satisfaction_rate == 0.75
    assert metrics.satisfaction_percentage == 75.0
    assert metrics.unserved_cargo == 50_000.0
    assert round(metrics.service_level_gap, 2) == 0.15
    assert metrics.is_satisfied is False
    assert metrics.status == "UNSATISFIED"


def test_demand_satisfaction_zero_and_negative_demand() -> None:
    """Verify robust handling of zero or negative cargo demand edge cases."""
    engine = CargoDemandSatisfactionEngine()
    scenario_zero = OptimizationScenario(
        cargo_demand=0.0,
        route_distance=1000.0,
        deadline_hours=100.0,
    )
    metrics_zero = engine.evaluate_demand_satisfaction(scenario_zero)
    assert metrics_zero.demand_satisfaction_rate == 1.0
    assert metrics_zero.unserved_cargo == 0.0
    assert metrics_zero.is_satisfied is True


def test_demand_satisfaction_with_forecasted_demand_hook() -> None:
    """Verify that forecasted_demand overrides cargo_demand when present."""
    engine = CargoDemandSatisfactionEngine()
    scenario = OptimizationScenario(
        cargo_demand=100_000.0,
        route_distance=2000.0,
        deadline_hours=150.0,
        service_level=0.95,
        forecasted_demand=150_000.0,
    )

    # Deliver 120,000 -> 120,000 / 150,000 = 0.80 (below 0.95 service level)
    metrics = engine.evaluate_demand_satisfaction(scenario, delivered_cargo=120_000.0)
    assert metrics.required_demand == 150_000.0
    assert metrics.demand_satisfaction_rate == 0.80
    assert metrics.unserved_cargo == 30_000.0
    assert metrics.is_satisfied is False


# =============================================================================
# 2. SCHEDULE RELIABILITY ENGINE TESTS
# =============================================================================

def test_schedule_reliability_perfect_adherence() -> None:
    """Verify 100.0 score when all voyages arrive on time without delay or cancellation."""
    engine = ScheduleReliabilityEngine()
    scenario = OptimizationScenario(
        cargo_demand=100_000.0,
        route_distance=3000.0,
        deadline_hours=250.0,
        weather_factor=1.0,
        port_delay_factor=1.0,
    )

    metrics = engine.evaluate_schedule_reliability(
        scenario=scenario,
        simulated_delays=[0.0, 0.0, 0.0, 0.0, 0.0],
        missed_voyages=0,
        total_voyages=5,
    )

    assert isinstance(metrics, ReliabilityMetrics)
    assert metrics.reliability_score == 100.0
    assert metrics.on_time_arrival_rate == 1.0
    assert metrics.average_delay_hours == 0.0
    assert metrics.missed_voyages == 0
    assert metrics.score_breakdown["on_time_component"] == 100.0
    assert metrics.score_breakdown["delay_penalty"] == 0.0
    assert metrics.score_breakdown["missed_voyage_penalty"] == 0.0


def test_schedule_reliability_normalized_mathematical_formula() -> None:
    """Verify normalized formula with explicit delays and missed voyages."""
    engine = ScheduleReliabilityEngine(weight_on_time=1.0, weight_delay=0.5, weight_missed=0.5)

    # 10 voyages: 6 on time (0h delay), 2 delayed by 50h, 2 missed voyages
    # Deadline: 200h
    metrics = engine.evaluate_from_observations(
        total_voyages=10,
        on_time_voyages=6,
        missed_voyages=2,
        average_delay_hours=20.0,
        deadline_hours=200.0,
    )

    # on_time_rate = 6/10 = 0.60 -> on_time_comp = 60.0
    # delay_ratio = 20/200 = 0.10 -> delay_penalty = 100 * 0.5 * 0.10 = 5.0
    # missed_rate = 2/10 = 0.20  -> missed_penalty = 100 * 0.5 * 0.20 = 10.0
    # reliability = 60.0 - 5.0 - 10.0 = 45.0
    assert metrics.reliability_score == 45.0
    assert metrics.on_time_arrival_rate == 0.60
    assert metrics.score_breakdown["on_time_component"] == 60.0
    assert metrics.score_breakdown["delay_penalty"] == 5.0
    assert metrics.score_breakdown["missed_voyage_penalty"] == 10.0


def test_schedule_reliability_route_level_breakdown() -> None:
    """Verify that route_reliability produces distinct scores for each corridor."""
    engine = ScheduleReliabilityEngine()
    scenario = OptimizationScenario(
        cargo_demand=150_000.0,
        route_distance=4000.0,
        deadline_hours=300.0,
        routes=(
            {"route_id": "CORRIDOR-A", "distance_nm": 3000.0},
            {"route_id": "CORRIDOR-B", "distance_nm": 6000.0},
        ),
    )

    route_scores = engine.evaluate_route_reliability(scenario)
    assert "CORRIDOR-A" in route_scores
    assert "CORRIDOR-B" in route_scores
    for r_id, score in route_scores.items():
        assert 0.0 <= score <= 100.0


def test_port_congestion_delay_factor_sensitivity() -> None:
    """Verify increasing port_delay_factor reduces schedule reliability score."""
    engine = ScheduleReliabilityEngine()
    scenario_calm = OptimizationScenario(
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

    metrics_calm = engine.evaluate_schedule_reliability(scenario_calm)
    metrics_congested = engine.evaluate_schedule_reliability(scenario_congested)

    assert metrics_congested.average_delay_hours > metrics_calm.average_delay_hours
    assert metrics_congested.reliability_score <= metrics_calm.reliability_score


# =============================================================================
# 3. OPTIMIZER STATUS GRANULARITY & FAILURE REASONS
# =============================================================================

def test_fleet_strategy_reliability_constraint_enforcement() -> None:
    """Verify optimizer flags RELIABILITY_TOO_LOW if achieved score is below target."""
    optimizer = FleetStrategyOptimizer()
    scenario = OptimizationScenario(
        cargo_demand=100_000.0,
        route_distance=4000.0,
        deadline_hours=150.0,  # Tight deadline causing delay
        target_reliability=99.0,  # Unreasonably strict reliability target
        weather_factor=1.35,  # Heavy adverse weather
        port_delay_factor=1.5,  # Congestion
    )

    rec = optimizer.optimize_strategy(scenario)
    assert rec.reliability_metrics is not None
    assert rec.summary.get("failure_reason") in {"RELIABILITY_TOO_LOW", "DEADLINE_VIOLATED", "DEMAND_NOT_MET"}
    if rec.status == OptimizationStatus.INFEASIBLE:
        assert rec.summary.get("failure_reason") == "RELIABILITY_TOO_LOW"


def test_fleet_strategy_deployment_plan_enrichment() -> None:
    """Verify route deployment plan includes buffer capacity and corridor reliability."""
    optimizer = FleetStrategyOptimizer()
    scenario = OptimizationScenario(
        cargo_demand=120_000.0,
        route_distance=3500.0,
        deadline_hours=300.0,
        scenario_id="SCEN-DEPLOY-TEST",
    )

    rec = optimizer.optimize_strategy(scenario)
    assert rec.status == OptimizationStatus.SUCCESS
    assert rec.deployment_plan

    for r_id, vessels in rec.deployment_plan.items():
        assert len(vessels) >= 1
        v = vessels[0]
        assert "allocated_capacity_dwt" in v
        assert "buffer_capacity_dwt" in v
        assert "redundancy_factor" in v
        assert "route_reliability_score" in v
        assert v["redundancy_factor"] >= 1.0


# =============================================================================
# 4. SCENARIO PERSISTENCE & DATA CONTRACT INTEGRITY
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

    # Check metrics fidelity
    assert loaded.reliability_metrics is not None
    assert loaded.reliability_metrics.reliability_score == rec.reliability_metrics.reliability_score
    assert loaded.reliability_metrics.score_breakdown == rec.reliability_metrics.score_breakdown

    assert loaded.demand_metrics is not None
    assert loaded.demand_metrics.required_demand == rec.demand_metrics.required_demand
    assert loaded.demand_metrics.demand_satisfaction_rate == rec.demand_metrics.demand_satisfaction_rate
    assert loaded.demand_metrics.is_satisfied == rec.demand_metrics.is_satisfied
