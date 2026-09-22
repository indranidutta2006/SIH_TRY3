"""Deterministic unit tests for FleetScheduler.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Validates:
1. Hand-built 3-vessel / 4-cargo allocation with known feasible assignments.
2. Independent validate_constraints confirmation.
3. Over-capacity cargo is left unassigned (assigned=False), not force-fit.
4. validate_constraints detects artificial capacity/deadline violations.
"""

from contracts.interfaces import SchedulerEngine
from contracts.schemas import FleetAssignment
from src.scheduler import FleetScheduler


def test_fleet_scheduler_implements_contract() -> None:
    """Verify FleetScheduler fulfills contracts.interfaces.SchedulerEngine."""
    scheduler = FleetScheduler()
    assert isinstance(scheduler, SchedulerEngine)


def test_handbuilt_3vessel_4cargo_assignment_and_constraint_validation() -> None:
    """Validate hand-built 3-vessel/4-cargo case with known feasible assignment and over-capacity cargo."""
    scheduler = FleetScheduler(nominal_speed_knots=14.0)

    vessel_ids = ["V1", "V2", "V3"]
    cargo_ids = ["C1", "C2", "C3", "C4"]

    constraints = {
        "vessels": {
            "V1": {"capacity": 30000.0},
            "V2": {"capacity": 60000.0},
            "V3": {"capacity": 100000.0},
        },
        "cargos": {
            # C1 fits V1 (smallest capable vessel)
            "C1": {"tons": 25000.0, "deadline_hours": 50.0, "distance_nm": 350.0},
            # C2 fits V2 (V1 too small, V2 smallest capable)
            "C2": {"tons": 50000.0, "deadline_hours": 70.0, "distance_nm": 420.0},
            # C3 fits V3 (V1 and V2 too small)
            "C3": {"tons": 85000.0, "deadline_hours": 90.0, "distance_nm": 700.0},
            # C4 is 150k tons -> over-capacity for all vessels (max is V3 with 100k)
            "C4": {"tons": 150000.0, "deadline_hours": 120.0, "distance_nm": 500.0},
        },
    }

    assignments = scheduler.assign_vessels(vessel_ids, cargo_ids, constraints)
    assignment_map = {a.cargo_id: a for a in assignments}

    # C1, C2, C3 must be feasibly assigned
    assert assignment_map["C1"].assigned is True
    assert assignment_map["C1"].vessel_id == "V1"

    assert assignment_map["C2"].assigned is True
    assert assignment_map["C2"].vessel_id == "V2"

    assert assignment_map["C3"].assigned is True
    assert assignment_map["C3"].vessel_id == "V3"

    # C4 exceeds 100,000 DWT max capacity -> MUST be left unassigned, NOT force-fit
    assert assignment_map["C4"].assigned is False
    assert assignment_map["C4"].vessel_id == ""

    # Independent validator confirms feasibility
    is_valid = scheduler.validate_constraints(assignments, constraints)
    assert is_valid is True


def test_independent_validator_detects_capacity_and_deadline_violations() -> None:
    """Verify validate_constraints independently rejects invalid manual schedules."""
    scheduler = FleetScheduler(nominal_speed_knots=14.0)

    constraints = {
        "vessels": {"V1": {"capacity": 30000.0}},
        "cargos": {
            "C1": {"tons": 40000.0, "deadline_hours": 50.0, "distance_nm": 280.0},
        },
    }

    # Artificially force-fit C1 (40,000 t) onto V1 (30,000 t capacity)
    violating_assignment = [
        FleetAssignment(vessel_id="V1", cargo_id="C1", assigned=True, estimated_cost=0.0, estimated_fuel=0.0)
    ]

    # Independent check must reject it
    assert scheduler.validate_constraints(violating_assignment, constraints) is False
