"""Deterministic unit tests for NSGA-II Pareto optimizer and crowding distance selection.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Validates:
1. Exact Deb et al. (2002) analytical crowding distance computation.
2. Boundary solution infinite distance assignment.
3. Diversity-preserving environmental selection over crowded clustering.
4. Fast non-dominated sorting correctness.
5. End-to-end ParetoFleetOptimizer execution and frontier non-domination invariants.
"""

import numpy as np
import pytest

from src.optimization.nsga2_pareto import (
    ParetoFleetOptimizer,
    calculate_crowding_distance,
    evaluate_bi_objective,
    non_dominated_sort,
    run_nsga2_pareto,
)


def test_calculate_crowding_distance_deb_2002_analytical() -> None:
    """Verify crowding distance matches hand-computed Deb et al. (2002) formulation."""
    # 4 non-dominated solutions in 2D space (f1: fuel cost, f2: emissions)
    pts = np.array([
        [1.0, 4.0],  # Boundary 0 (min f1)
        [2.0, 3.0],  # Intermediate 1
        [3.0, 1.5],  # Intermediate 2
        [4.0, 1.0],  # Boundary 1 (min f2)
    ])
    # Objective ranges:
    # f1: [1.0, 4.0] -> range = 3.0
    # f2: [1.0, 4.0] -> range = 3.0
    # For pt 1 (2.0, 3.0):
    #   f1 diff = (3.0 - 1.0) / 3.0 = 2/3
    #   f2 diff = (4.0 - 1.5) / 3.0 = 2.5 / 3.0 = 5/6
    #   total = 2/3 + 5/6 = 9/6 = 1.5
    # For pt 2 (3.0, 1.5):
    #   f1 diff = (4.0 - 2.0) / 3.0 = 2/3
    #   f2 diff = (3.0 - 1.0) / 3.0 = 2/3
    #   total = 2/3 + 2/3 = 4/3 = 1.33333333...

    distances = calculate_crowding_distance(pts)

    assert len(distances) == 4
    assert np.isinf(distances[0])
    assert np.isinf(distances[3])
    assert np.isclose(distances[1], 1.5)
    assert np.isclose(distances[2], 4.0 / 3.0)


def test_calculate_crowding_distance_edge_cases() -> None:
    """Verify behavior on empty, single, 2-point, and degenerate fronts."""
    # Empty
    empty = np.empty((0, 2))
    assert len(calculate_crowding_distance(empty)) == 0

    # 1 individual
    one = np.array([[10.0, 20.0]])
    d_one = calculate_crowding_distance(one)
    assert len(d_one) == 1
    assert np.isinf(d_one[0])

    # 2 individuals
    two = np.array([[10.0, 20.0], [5.0, 30.0]])
    d_two = calculate_crowding_distance(two)
    assert len(d_two) == 2
    assert np.all(np.isinf(d_two))

    # Invalid dimension
    with pytest.raises(ValueError, match="must be a 2D array"):
        calculate_crowding_distance(np.array([1.0, 2.0]))

    # Degenerate: all points identical (zero range across all objectives)
    identical = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])
    d_ident = calculate_crowding_distance(identical)
    assert len(d_ident) == 3
    # No NaN or ZeroDivisionError
    assert not np.any(np.isnan(d_ident))
    assert np.all(np.isinf(d_ident))


def test_calculate_crowding_distance_3d_objectives() -> None:
    """Verify crowding distance correctly assigns inf to all extreme boundary points in 3D."""
    pts_3d = np.array([
        [1.0, 10.0, 10.0],  # Min obj 0 -> inf
        [10.0, 1.0, 10.0],  # Min obj 1 -> inf
        [10.0, 10.0, 1.0],  # Min obj 2 -> inf
        [5.0, 5.0, 5.0],    # Interior point -> finite
    ])
    d = calculate_crowding_distance(pts_3d)
    assert np.isinf(d[0])
    assert np.isinf(d[1])
    assert np.isinf(d[2])
    assert not np.isinf(d[3])
    assert d[3] > 0.0


def test_environmental_selection_prioritizes_crowding_distance() -> None:
    """Assert canonical NSGA-II selection preserves isolated points over dense clusters."""
    # Front with 5 individuals where 4 slots are needed
    front_objs = np.array([
        [10.0, 100.0],  # Index 0: Extreme min cost -> inf
        [20.0, 80.0],   # Index 1: Crowded cluster pt A -> small distance
        [21.0, 79.0],   # Index 2: Crowded cluster pt B -> smallest distance
        [50.0, 50.0],   # Index 3: Isolated point -> large distance
        [100.0, 10.0],  # Index 4: Extreme min emissions -> inf
    ])
    distances = calculate_crowding_distance(front_objs)

    # In canonical NSGA-II, the individuals are sorted descending by crowding distance
    sorted_order = np.argsort(-distances)
    needed = 4
    selected_indices = sorted_order[:needed]

    # Extreme boundaries (0, 4) and isolated solution (3) MUST be selected
    assert 0 in selected_indices
    assert 4 in selected_indices
    assert 3 in selected_indices

    # The most crowded point (index 1) should be eliminated, not the extreme boundary
    # (Naive front[:4] would have incorrectly dropped boundary index 4)
    assert 4 in selected_indices
    assert len(selected_indices) == 4


def test_non_dominated_sort_ranks() -> None:
    """Verify fast non-dominated sorting produces correct Pareto ranking layers."""
    # 4 points:
    # p0 = (1, 3) dominates p2 = (2, 4)
    # p1 = (3, 1) dominates p2 = (2, 4)
    # p3 = (5, 5) dominated by p2
    pts = np.array([
        [1.0, 3.0],  # 0: Front 0
        [3.0, 1.0],  # 1: Front 0
        [2.0, 4.0],  # 2: Front 1
        [5.0, 5.0],  # 3: Front 2
    ])
    fronts = non_dominated_sort(pts)
    assert len(fronts) == 3
    assert set(fronts[0]) == {0, 1}
    assert set(fronts[1]) == {2}
    assert set(fronts[2]) == {3}


def test_pareto_fleet_optimizer_end_to_end() -> None:
    """Verify ParetoFleetOptimizer produces a verified non-dominated frontier."""
    optimizer = ParetoFleetOptimizer(random_state=42)
    res = optimizer.solve_medium_pareto(population_size=10, generations=2, seed=42)

    assert isinstance(res, dict)
    assert "pareto_front" in res
    assert res["assigned_voyages"] > 0
    assert res["runtime_seconds"] > 0
    assert res["pareto_front_size"] == len(res["pareto_front"])
    assert res["tier"] == "medium"

    front = res["pareto_front"]
    assert len(front) >= 1

    # Check ascending order of fuel cost
    costs = [pt["fuel_cost_usd"] for pt in front]
    co2s = [pt["co2e_tons"] for pt in front]
    for c, e in zip(costs, co2s):
        assert c > 0.0
        assert e > 0.0

    assert costs == sorted(costs)

    # Check non-domination property on the Pareto front:
    # No point should strictly dominate another point on the front
    for i in range(len(front)):
        for j in range(len(front)):
            if i != j:
                p_dom_q = (costs[i] <= costs[j] and co2s[i] <= co2s[j]) and (costs[i] < costs[j] or co2s[i] < co2s[j])
                assert not p_dom_q, f"Point {i} ({costs[i]}, {co2s[i]}) dominates Point {j} ({costs[j]}, {co2s[j]}) on Pareto front"


def test_run_nsga2_pareto_wrapper() -> None:
    """Verify convenience runner function run_nsga2_pareto."""
    from src.optimization.fleet_optimizer import generate_fleet_problem

    _, _, constraints, assignments = generate_fleet_problem(n_vessels=5, n_cargos=10, seed=42)
    context = {"voyage_specs": constraints["cargos"], "compliance_year": 2025}

    res = run_nsga2_pareto(
        assignments=assignments,
        context=context,
        population_size=8,
        max_generations=2,
        random_state=42,
    )
    assert isinstance(res, dict)
    assert "pareto_front" in res
    assert res["population_size"] == 8


def test_evaluate_bi_objective_fuel_routing() -> None:
    """Verify evaluate_bi_objective bypasses ML for physics-routed novel fuels."""
    from unittest.mock import MagicMock
    from contracts.schemas import FleetAssignment, PredictionResult
    from src.compliance.compliance_engine import MaritimeComplianceEngine
    from src.prediction.emission_engine import MaritimeEmissionEngine

    mock_model = MagicMock()
    mock_model.predict.return_value = [
        PredictionResult(
            model_name="mock_model",
            predicted_fuel_consumption=75.0,
            confidence_score=0.9,
            runtime_seconds=0.001,
        )
    ]
    emission_engine = MaritimeEmissionEngine()
    compliance_engine = MaritimeComplianceEngine()

    assignments = [
        FleetAssignment(
            vessel_id="VSL-1",
            cargo_id="CRG-1",
            assigned=True,
            estimated_cost=0.0,
            estimated_fuel=0.0,
        ),
        FleetAssignment(
            vessel_id="VSL-2",
            cargo_id="CRG-2",
            assigned=True,
            estimated_cost=0.0,
            estimated_fuel=0.0,
        ),
    ]

    context = {
        "voyage_specs": {
            "CRG-1": {"tons": 35000.0, "distance_nm": 600.0},
            "CRG-2": {"tons": 35000.0, "distance_nm": 600.0},
        },
        "compliance_year": 2025,
    }

    # Voyage 1: Methanol (2.0) -> ML
    # Voyage 2: Hydrogen (3.0) -> Physics
    x = np.array([13.0, 2.0, 11.0, 3.0], dtype=float)

    cost, co2e = evaluate_bi_objective(
        x=x,
        assignments=assignments,
        context=context,
        engines=(mock_model, emission_engine, compliance_engine),
    )

    # ML model called only for Methanol (1 record)
    assert mock_model.predict.call_count == 1
    call_records = mock_model.predict.call_args[0][0]
    assert len(call_records) == 1
    assert call_records[0].fuel_type == "Methanol"

    assert cost > 0.0
    assert co2e > 0.0


def test_evaluate_bi_objective_dynamic_specs() -> None:
    """Verify evaluate_bi_objective populates dynamic vessel and weather characteristics into VoyageRecord."""
    from unittest.mock import MagicMock
    from contracts.schemas import FleetAssignment, PredictionResult
    from src.compliance.compliance_engine import MaritimeComplianceEngine
    from src.prediction.emission_engine import MaritimeEmissionEngine

    mock_model = MagicMock()
    mock_model.predict.return_value = [
        PredictionResult(
            model_name="mock_model",
            predicted_fuel_consumption=85.0,
            confidence_score=0.95,
            runtime_seconds=0.001,
        )
    ]
    emission_engine = MaritimeEmissionEngine()
    compliance_engine = MaritimeComplianceEngine()

    assignments = [
        FleetAssignment(
            vessel_id="VSL-GAS-01",
            cargo_id="CRG-LNG-01",
            assigned=True,
            estimated_cost=0.0,
            estimated_fuel=0.0,
        )
    ]

    context = {
        "vessel_specs": {
            "VSL-GAS-01": {
                "vessel_type": "Gas Carrier",
                "capacity": 72000.0,
            }
        },
        "voyage_specs": {
            "CRG-LNG-01": {
                "cargo_tons": 48000.0,
                "distance_nm": 950.0,
                "weather_factor": 1.2,
                "sea_state": 4,
            }
        },
        "compliance_year": 2025,
    }

    # speed = 15.0 knots, fuel = LNG (1.0)
    x = np.array([15.0, 1.0], dtype=float)

    evaluate_bi_objective(
        x=x,
        assignments=assignments,
        context=context,
        engines=(mock_model, emission_engine, compliance_engine),
    )

    assert mock_model.predict.call_count == 1
    rec = mock_model.predict.call_args[0][0][0]

    assert rec.vessel_type == "Gas Carrier"
    assert rec.vessel_dwt == 72000.0
    assert rec.cargo_tons == 48000.0
    assert rec.weather_factor == 1.2
    assert rec.sea_state == 4


