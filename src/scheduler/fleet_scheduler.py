"""Fleet scheduling engine implementing greedy feasible-first allocation.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Fulfills contracts.interfaces.SchedulerEngine.
Allocates vessels to pending cargo orders and provides an independent constraint validation engine.
"""

from collections.abc import Sequence
import logging
from typing import Any, Final

from contracts.exceptions import SchedulerError
from contracts.interfaces import SchedulerEngine
from contracts.schemas import FleetAssignment

logger = logging.getLogger("maritime_system")

DEFAULT_NOMINAL_SPEED_KNOTS: Final[float] = 14.0
DEFAULT_VOYAGE_DISTANCE_NM: Final[float] = 1000.0


class FleetScheduler(SchedulerEngine):
    """Greedy feasible-first commercial vessel allocator and constraint auditor."""

    def __init__(self, nominal_speed_knots: float = DEFAULT_NOMINAL_SPEED_KNOTS) -> None:
        """Initialize scheduler with design cruising speed for timetable projection."""
        self.nominal_speed = nominal_speed_knots
        self.logger = logger

    def _extract_vessel_capacities(
        self, vessel_ids: Sequence[str], constraints: dict[str, Any]
    ) -> dict[str, float]:
        """Extract capacity (DWT) for each vessel from heterogeneous constraint payloads."""
        caps: dict[str, float] = {}
        for vid in vessel_ids:
            cap = 0.0
            if "vessels" in constraints and vid in constraints["vessels"]:
                v_entry = constraints["vessels"][vid]
                cap = float(v_entry.get("capacity", v_entry.get("dwt", 0.0)))
            elif "vessel_capacities" in constraints and vid in constraints["vessel_capacities"]:
                cap = float(constraints["vessel_capacities"][vid])
            elif vid in constraints and isinstance(constraints[vid], dict):
                cap = float(constraints[vid].get("capacity", constraints[vid].get("dwt", 0.0)))
            caps[vid] = cap
        return caps

    def _extract_cargo_requirements(
        self, cargo_ids: Sequence[str], constraints: dict[str, Any]
    ) -> dict[str, dict[str, float]]:
        """Extract tons, deadline_hours, and distance_nm per cargo shipment."""
        reqs: dict[str, dict[str, float]] = {}
        for cid in cargo_ids:
            tons = 0.0
            deadline = float("inf")
            distance = DEFAULT_VOYAGE_DISTANCE_NM

            if "cargos" in constraints and cid in constraints["cargos"]:
                c_entry = constraints["cargos"][cid]
                tons = float(c_entry.get("tons", c_entry.get("weight", 0.0)))
                deadline = float(c_entry.get("deadline_hours", c_entry.get("deadline", float("inf"))))
                distance = float(c_entry.get("distance_nm", DEFAULT_VOYAGE_DISTANCE_NM))
            elif "cargo_tons" in constraints and cid in constraints["cargo_tons"]:
                tons = float(constraints["cargo_tons"][cid])
                if "cargo_deadlines" in constraints and cid in constraints["cargo_deadlines"]:
                    deadline = float(constraints["cargo_deadlines"][cid])
                if "cargo_distances" in constraints and cid in constraints["cargo_distances"]:
                    distance = float(constraints["cargo_distances"][cid])
            elif cid in constraints and isinstance(constraints[cid], dict):
                c_entry = constraints[cid]
                tons = float(c_entry.get("tons", 0.0))
                deadline = float(c_entry.get("deadline_hours", float("inf")))
                distance = float(c_entry.get("distance_nm", DEFAULT_VOYAGE_DISTANCE_NM))

            reqs[cid] = {"tons": tons, "deadline_hours": deadline, "distance_nm": distance}
        return reqs

    def assign_vessels(
        self,
        vessel_ids: Sequence[str],
        cargo_ids: Sequence[str],
        constraints: dict[str, Any],
    ) -> Sequence[FleetAssignment]:
        """Assign available vessels to pending cargo shipments using greedy feasible-first logic.

        Steps:
        1. Sort cargos by deadline ascending.
        2. Assign each cargo to the smallest-capacity capable vessel that does not exceed its schedule.
        3. Leave cargo unassigned (assigned=False) rather than violating capacity or deadline.
        """
        if not vessel_ids:
            return [
                FleetAssignment(vessel_id="", cargo_id=cid, assigned=False, estimated_cost=0.0, estimated_fuel=0.0)
                for cid in cargo_ids
            ]

        vessel_caps = self._extract_vessel_capacities(vessel_ids, constraints)
        cargo_reqs = self._extract_cargo_requirements(cargo_ids, constraints)

        # 1. Sort cargos by deadline ascending
        sorted_cargos = sorted(
            cargo_ids,
            key=lambda cid: (cargo_reqs[cid]["deadline_hours"], cargo_reqs[cid]["tons"]),
        )

        vessel_schedules: dict[str, float] = {vid: 0.0 for vid in vessel_ids}
        assignments: list[FleetAssignment] = []

        # 2. Feasible-first greedy assignment
        for cid in sorted_cargos:
            req = cargo_reqs[cid]
            c_tons = req["tons"]
            c_deadline = req["deadline_hours"]
            c_dist = req["distance_nm"]
            voyage_hours = c_dist / self.nominal_speed

            # Filter capable vessels respecting capacity and schedule
            eligible_vessels = [
                vid
                for vid in vessel_ids
                if vessel_caps[vid] >= c_tons
                and (vessel_schedules[vid] + voyage_hours) <= c_deadline
            ]

            if eligible_vessels:
                # Smallest-capacity vessel heuristic, tie-break by earliest available schedule
                chosen_vessel = min(
                    eligible_vessels,
                    key=lambda v: (vessel_caps[v], vessel_schedules[v]),
                )
                vessel_schedules[chosen_vessel] += voyage_hours
                assignments.append(
                    FleetAssignment(
                        vessel_id=chosen_vessel,
                        cargo_id=cid,
                        assigned=True,
                        estimated_cost=0.0,
                        estimated_fuel=0.0,
                    )
                )
                self.logger.debug(
                    "Assigned cargo %s (%.1f t) to vessel %s (cap: %.1f t, finish: %.1f h)",
                    cid,
                    c_tons,
                    chosen_vessel,
                    vessel_caps[chosen_vessel],
                    vessel_schedules[chosen_vessel],
                )
            else:
                # Leave unassigned rather than violating capacity or deadline
                assignments.append(
                    FleetAssignment(
                        vessel_id="",
                        cargo_id=cid,
                        assigned=False,
                        estimated_cost=0.0,
                        estimated_fuel=0.0,
                    )
                )
                self.logger.debug("Cargo %s could not be feasibly assigned within capacity/deadline.", cid)

        return assignments

    def validate_constraints(
        self,
        assignments: Sequence[FleetAssignment],
        constraints: dict[str, Any],
    ) -> bool:
        """Recompute capacity and deadline feasibility independently from assign_vessels logic."""
        all_vessels = list(
            constraints.get("vessels", {}).keys()
            or constraints.get("vessel_capacities", {}).keys()
            or [a.vessel_id for a in assignments if a.vessel_id]
        )
        all_cargos = list(
            constraints.get("cargos", {}).keys()
            or constraints.get("cargo_tons", {}).keys()
            or [a.cargo_id for a in assignments]
        )

        vessel_caps = self._extract_vessel_capacities(all_vessels, constraints)
        cargo_reqs = self._extract_cargo_requirements(all_cargos, constraints)

        vessel_timelines: dict[str, float] = {vid: 0.0 for vid in all_vessels}

        for assignment in assignments:
            if not assignment.assigned:
                continue

            vid = assignment.vessel_id
            cid = assignment.cargo_id

            if vid not in vessel_caps or cid not in cargo_reqs:
                self.logger.warning("Assignment contains unknown vessel (%s) or cargo (%s)", vid, cid)
                return False

            req = cargo_reqs[cid]
            # 1. Independent capacity check
            if req["tons"] > vessel_caps[vid]:
                self.logger.warning(
                    "Capacity violation: Cargo %s (%.1f t) exceeds vessel %s capacity (%.1f t)",
                    cid,
                    req["tons"],
                    vid,
                    vessel_caps[vid],
                )
                return False

            # 2. Independent deadline check
            voyage_hours = req["distance_nm"] / self.nominal_speed
            arrival_time = vessel_timelines[vid] + voyage_hours
            if arrival_time > req["deadline_hours"]:
                self.logger.warning(
                    "Deadline violation: Vessel %s arrives at %.1f h for cargo %s with deadline %.1f h",
                    vid,
                    arrival_time,
                    cid,
                    req["deadline_hours"],
                )
                return False

            vessel_timelines[vid] = arrival_time

        return True
