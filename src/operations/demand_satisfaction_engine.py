"""Cargo Demand Satisfaction Engine for maritime fleet operations.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 2 Deliverable: Explicitly verifies cargo demand fulfillment, computes service-level gaps,
identifies unserved cargo volume, and enforces operational satisfaction constraints.
"""

from dataclasses import asdict
import logging
from typing import Any

from contracts.schemas import (
    DemandSatisfactionMetrics,
    FleetCompositionResult,
    OptimizationScenario,
)

logger = logging.getLogger("maritime_system")


class CargoDemandSatisfactionEngine:
    """Evaluates and enforces cargo demand satisfaction under operational scenarios."""

    def __init__(self) -> None:
        """Initialize demand satisfaction engine."""
        self.logger = logger

    def evaluate_demand_satisfaction(
        self,
        scenario: OptimizationScenario | None = None,
        cargo_demand: float | None = None,
        delivered_cargo: float | None = None,
        fleet_composition: FleetCompositionResult | None = None,
        fleet_mix: FleetCompositionResult | dict[str, int] | None = None,
        capacity_plan: Any | None = None,
        deployment_plan: dict[str, list[dict[str, Any]]] | None = None,
        service_level: float | None = None,
    ) -> DemandSatisfactionMetrics:
        """Evaluate cargo delivery fulfillment against scenario target and service level.

        Workflow:
            1. Resolve required demand (prioritizing explicit cargo_demand, then forecasted_demand, then scenario cargo_demand).
            2. Compute delivered cargo volume from explicit parameter, fleet composition, deployment plan, or capacity plan.
            3. Calculate satisfaction rate, capped strictly at 1.0 (100%).
            4. Compute unserved cargo and distance to target service level.

        Args:
            scenario: Operational scenario context containing demand and service level targets.
            cargo_demand: Explicit cargo demand volume in metric tons.
            delivered_cargo: Explicitly provided delivered tonnage, if precomputed.
            fleet_composition: Fleet composition result providing total annual fleet capacity.
            fleet_mix: Fleet composition result or size/fuel dictionary.
            capacity_plan: Capacity optimization result providing vessel-level sizing.
            deployment_plan: Route-to-vessel deployment allocations.
            service_level: Optional explicit target service level (default from scenario or 0.95).

        Returns:
            DemandSatisfactionMetrics with satisfaction rate, unserved volume, gap, and status.
        """
        # 1. Resolve required demand
        if cargo_demand is not None:
            required_demand = float(cargo_demand)
        elif scenario is not None:
            required_demand = (
                scenario.forecasted_demand
                if scenario.forecasted_demand is not None
                else scenario.cargo_demand
            )
        else:
            required_demand = 0.0

        # 2. Resolve target service level
        target_sl = (
            service_level
            if service_level is not None
            else (scenario.service_level if scenario is not None else 0.95)
        )

        # 3. Resolve delivered cargo
        if delivered_cargo is not None:
            actual_delivered = float(delivered_cargo)
        elif fleet_composition is not None:
            actual_delivered = float(fleet_composition.total_capacity)
        elif fleet_mix is not None:
            if isinstance(fleet_mix, FleetCompositionResult):
                actual_delivered = float(fleet_mix.total_capacity)
            elif isinstance(fleet_mix, dict):
                actual_delivered = self._sum_fleet_mix_capacity(fleet_mix, capacity_plan)
            else:
                actual_delivered = 0.0
        elif deployment_plan is not None:
            actual_delivered = self._sum_deployment_capacity(deployment_plan)
        elif capacity_plan is not None:
            trips = getattr(capacity_plan, "optimal_trips", 1)
            cap = getattr(capacity_plan, "recommended_capacity", 0.0)
            actual_delivered = float(cap * trips)
        else:
            actual_delivered = 0.0

        # 4. Handle zero or negative demand edge case
        if required_demand <= 0.0:
            self.logger.info("Scenario demand is zero or negative (%.1f tons). Marked fully satisfied.", required_demand)
            return DemandSatisfactionMetrics(
                required_demand=max(0.0, required_demand),
                delivered_cargo=max(0.0, actual_delivered),
                demand_satisfaction_rate=1.0,
                unserved_cargo=0.0,
                service_level_gap=0.0,
                is_satisfied=True,
                status="SATISFIED",
            )

        # 5. Standard demand satisfaction metrics
        delivered_effective = max(0.0, actual_delivered)
        raw_ratio = delivered_effective / required_demand
        satisfaction_rate = min(1.0, max(0.0, raw_ratio))
        unserved = max(0.0, required_demand - delivered_effective)
        service_gap = max(0.0, target_sl - satisfaction_rate)
        is_satisfied = satisfaction_rate >= target_sl

        status = "SATISFIED" if is_satisfied else "UNSATISFIED"

        self.logger.info(
            "Demand Evaluation: Required=%.1f, Delivered=%.1f (%.1f%%), Target=%.1f%%, Status=%s",
            required_demand,
            delivered_effective,
            satisfaction_rate * 100.0,
            target_sl * 100.0,
            status,
        )

        return DemandSatisfactionMetrics(
            required_demand=round(required_demand, 2),
            delivered_cargo=round(delivered_effective, 2),
            demand_satisfaction_rate=round(satisfaction_rate, 4),
            unserved_cargo=round(unserved, 2),
            service_level_gap=round(service_gap, 4),
            is_satisfied=is_satisfied,
            status=status,
        )

    def calculate_minimum_fleet_capacity(self, scenario: OptimizationScenario) -> float:
        """Calculate minimum aggregate fleet capacity required to satisfy service level."""
        req = scenario.forecasted_demand if scenario.forecasted_demand is not None else scenario.cargo_demand
        return max(0.0, req * scenario.service_level)

    def _sum_deployment_capacity(self, deployment_plan: dict[str, list[dict[str, Any]]]) -> float:
        """Sum allocated capacity from an operational deployment plan."""
        total = 0.0
        for vessels in deployment_plan.values():
            for v in vessels:
                annual_v = float(v.get("voyages_per_year", v.get("annual_voyages", 1)))
                dwt = float(v.get("allocated_capacity_dwt", 0.0))
                total += dwt * 0.85 * annual_v
        return total

    def _sum_fleet_mix_capacity(
        self,
        fleet_mix: dict[str, int],
        capacity_plan: Any | None = None,
    ) -> float:
        """Estimate annual cargo throughput for a fleet mix dictionary."""
        # Standard annual capacities per vessel class (DWT * 0.85 * annual voyages)
        annual_class_throughput = {
            "feeder": 12000.0 * 0.85 * 35,
            "medium": 45000.0 * 0.85 * 20,
            "large": 120000.0 * 0.85 * 10,
        }
        total = 0.0
        for k, count in fleet_mix.items():
            k_clean = k.lower().split("_")[0]
            if k_clean in annual_class_throughput and count > 0:
                total += annual_class_throughput[k_clean] * count

        if total == 0.0 and capacity_plan is not None:
            total_ships = sum(v for k, v in fleet_mix.items() if k in {"feeder", "medium", "large"})
            cap = getattr(capacity_plan, "recommended_capacity", 45000.0)
            trips = getattr(capacity_plan, "optimal_trips", 20)
            total = cap * 0.85 * trips * max(1, total_ships)

        return total


# Canonical alias for Phase 2 operational consistency
DemandSatisfactionEngine = CargoDemandSatisfactionEngine
