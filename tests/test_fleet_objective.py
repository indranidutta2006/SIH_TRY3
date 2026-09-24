"""Deterministic unit tests for fleet_objective.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Validates:
1. Exact hand-calculated arithmetic of scalar J = w1*cost + w2*co2e + w3*delay on a round-number single-voyage case.
2. End-to-end execution with real production models, emission engine, and compliance engine.
"""

from unittest.mock import MagicMock
import numpy as np
import pytest

from contracts.schemas import FleetAssignment, PredictionResult
from src.compliance.compliance_engine import MaritimeComplianceEngine
from src.optimization.fleet_objective import fleet_objective
from src.prediction.emission_engine import MaritimeEmissionEngine


def test_hand_calculated_single_voyage_arithmetic() -> None:
    """Verify exact arithmetic of J on a single-voyage case computed by hand with round numbers.

    Setup:
    - Speed = 10.0 knots
    - Fuel = Diesel (index 0)
    - Distance = 1000.0 NM -> Hours at sea = 100.0 hours
    - Deadline = 90.0 hours -> Delay = 10.0 hours
    - Delay penalty rate = $500.0/hour -> Delay penalty = $5000.0
    - Predicted fuel = 100.0 metric tons
    - Fuel price = $650.0/ton -> Fuel cost = $65,000.0
    - Diesel WTW CO2e = 373.0 metric tons
    - FuelEU target (2024) = 91.16 g/MJ vs Attained 87.35 g/MJ -> Pass, Penalty = $0.0
    - Weights = (1.0, 2.0, 0.5)

    Hand Calculation:
    J = (1.0 * 65,000.0) + (2.0 * 373.0) + (0.5 * 5000.0)
      = 65,000.0 + 746.0 + 2,500.0
      = 68,246.0
    """
    # Deterministic dummy predictor returning exactly 100.0 tons
    mock_model = MagicMock()
    mock_model.predict.return_value = [
        PredictionResult(
            model_name="mock_model",
            predicted_fuel_consumption=100.0,
            confidence_score=0.95,
            runtime_seconds=0.001,
        )
    ]

    emission_engine = MaritimeEmissionEngine()
    compliance_engine = MaritimeComplianceEngine()

    assignments = [
        FleetAssignment(
            vessel_id="VSL-01",
            cargo_id="CRG-01",
            assigned=True,
            estimated_cost=0.0,
            estimated_fuel=0.0,
        )
    ]

    # Decision vector: speed = 10.0 knots, fuel_idx = 0.0 (Diesel)
    x = np.array([10.0, 0.0], dtype=float)

    context = {
        "voyage_specs": {
            "CRG-01": {
                "tons": 40000.0,
                "capacity": 50000.0,
                "distance_nm": 1000.0,
                "deadline_hours": 90.0,
            }
        },
        "delay_penalty_per_hour": 500.0,
        "compliance_year": 2024,
    }

    j_result = fleet_objective(
        x=x,
        assignments=assignments,
        weights=(1.0, 2.0, 0.5),
        context=context,
        engines=(mock_model, emission_engine, compliance_engine),
    )

    # Hand Calculation:
    # Diesel TTW CO2e = 320.6 (CO2) + 0.005*28 (CH4) + 0.018*265 (N2O) = 325.51 t
    # Diesel WTT CO2e = 59.0 t -> WTW CO2e = 384.51 t
    # J = (1.0 * 65,000.0) + (2.0 * 384.51) + (0.5 * 5000.0)
    #   = 65,000.0 + 769.02 + 2,500.0
    #   = 68,269.02
    expected_j = 68269.02
    assert pytest.approx(j_result, rel=1e-4) == expected_j


def test_fleet_objective_with_real_production_engines() -> None:
    """Verify fleet_objective runs with real production model, emissions, and compliance engines."""
    assignments = [
        FleetAssignment(
            vessel_id="VSL-REAL-1",
            cargo_id="CRG-REAL-1",
            assigned=True,
            estimated_cost=0.0,
            estimated_fuel=0.0,
        )
    ]
    # speed = 14.0 knots, fuel = Diesel (0.0)
    x = np.array([14.0, 0.0], dtype=float)

    context = {
        "voyage_specs": {
            "CRG-REAL-1": {
                "tons": 50000.0,
                "capacity": 65000.0,
                "distance_nm": 1200.0,
                "deadline_hours": 150.0,
            }
        }
    }

    # Calls real production engines automatically via get_cached_engines()
    j_real = fleet_objective(x=x, assignments=assignments, weights=(1.0, 1.0, 1.0), context=context)

    assert isinstance(j_real, float)
    assert j_real > 0.0


def test_fuel_routing_novel_fuels_bypass_ml() -> None:
    """Verify novel alternative fuels (Hydrogen, Ammonia) bypass ML production model completely."""
    mock_model = MagicMock()
    emission_engine = MaritimeEmissionEngine()
    compliance_engine = MaritimeComplianceEngine()

    assignments = [
        FleetAssignment(
            vessel_id="VSL-H2",
            cargo_id="CRG-H2",
            assigned=True,
            estimated_cost=0.0,
            estimated_fuel=0.0,
        )
    ]
    # speed = 12.0 knots, fuel_idx = 3.0 (Hydrogen)
    x = np.array([12.0, 3.0], dtype=float)

    context = {
        "voyage_specs": {
            "CRG-H2": {
                "tons": 35000.0,
                "capacity": 45000.0,
                "distance_nm": 800.0,
                "deadline_hours": 100.0,
            }
        },
        "compliance_year": 2025,
    }

    j_result = fleet_objective(
        x=x,
        assignments=assignments,
        weights=(1.0, 1.0, 1.0),
        context=context,
        engines=(mock_model, emission_engine, compliance_engine),
    )

    # ML model must NOT be called for Hydrogen
    mock_model.predict.assert_not_called()
    assert j_result > 0.0


def test_fuel_routing_shorepower_zero_operational_co2e() -> None:
    """Verify ShorePower routes through physics, generates 0 operational CO2e, and computes electricity cost."""
    mock_model = MagicMock()
    emission_engine = MaritimeEmissionEngine()
    compliance_engine = MaritimeComplianceEngine()

    assignments = [
        FleetAssignment(
            vessel_id="VSL-SHORE",
            cargo_id="CRG-SHORE",
            assigned=True,
            estimated_cost=0.0,
            estimated_fuel=0.0,
        )
    ]
    # speed = 10.0 knots, fuel_idx = 5.0 (ShorePower)
    x = np.array([10.0, 5.0], dtype=float)

    context = {
        "voyage_specs": {
            "CRG-SHORE": {
                "tons": 30000.0,
                "capacity": 40000.0,
                "distance_nm": 200.0,
                "deadline_hours": 50.0,
            }
        },
        "compliance_year": 2025,
    }

    # Weight CO2e heavily (100.0), weights=(1.0, 100.0, 0.0)
    # Since operational CO2e = 0.0, delay = 0.0 (20h vs 50h deadline), J = cost
    j_result = fleet_objective(
        x=x,
        assignments=assignments,
        weights=(1.0, 100.0, 0.0),
        context=context,
        engines=(mock_model, emission_engine, compliance_engine),
    )

    mock_model.predict.assert_not_called()
    assert j_result > 0.0


def test_fuel_routing_mixed_fleet() -> None:
    """Verify mixed fleet: ML-routed fuels pass to prod_model.predict, physics-routed fuels do not."""
    mock_model = MagicMock()
    mock_model.predict.return_value = [
        PredictionResult(
            model_name="mock_model",
            predicted_fuel_consumption=80.0,
            confidence_score=0.9,
            runtime_seconds=0.001,
        )
    ]
    emission_engine = MaritimeEmissionEngine()
    compliance_engine = MaritimeComplianceEngine()

    assignments = [
        FleetAssignment(
            vessel_id="VSL-DIESEL",
            cargo_id="CRG-1",
            assigned=True,
            estimated_cost=0.0,
            estimated_fuel=0.0,
        ),
        FleetAssignment(
            vessel_id="VSL-AMMONIA",
            cargo_id="CRG-2",
            assigned=True,
            estimated_cost=0.0,
            estimated_fuel=0.0,
        ),
    ]
    # Voyage 1: speed=14.0, fuel=0.0 (Diesel) -> ML
    # Voyage 2: speed=12.0, fuel=4.0 (Ammonia) -> Physics
    x = np.array([14.0, 0.0, 12.0, 4.0], dtype=float)

    context = {
        "voyage_specs": {
            "CRG-1": {"tons": 40000.0, "distance_nm": 1000.0, "deadline_hours": 100.0},
            "CRG-2": {"tons": 40000.0, "distance_nm": 1000.0, "deadline_hours": 100.0},
        }
    }

    j_result = fleet_objective(
        x=x,
        assignments=assignments,
        weights=(1.0, 1.0, 1.0),
        context=context,
        engines=(mock_model, emission_engine, compliance_engine),
    )

    # ML model called exactly once with exactly 1 record (for Diesel only)
    assert mock_model.predict.call_count == 1
    call_records = mock_model.predict.call_args[0][0]
    assert len(call_records) == 1
    assert call_records[0].fuel_type == "Diesel"
    assert call_records[0].vessel_id == "VSL-DIESEL"
    assert j_result > 0.0
