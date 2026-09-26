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
        scenario: OptimizationScenario,
        delivered_cargo: float | None = None,
        fleet_composition: FleetCompositionResult | None = None,
        deployment_plan: dict[str, list[dict[str, Any]]] | None = None,
    ) -> DemandSatisfactionMetrics:
        """Evaluate cargo delivery fulfillment against scenario target and service level.

        Workflow:
            1. Resolve required demand (prioritizing forecasted_demand if provided, else cargo_demand).
            2. Compute delivered cargo volume from explicit parameter, fleet composition, or deployment plan.
            3. Calculate satisfaction rate, capped strictly at 1.0 (100%).
            4. Compute unserved cargo and distance to target service level.

        Args:
            scenario: Operational scenario context containing demand and service level targets.
            delivered_cargo: Explicitly provided delivered tonnage, if precomputed.
            fleet_composition: Fleet composition result providing total annual fleet capacity.
            deployment_plan: Route-to-vessel deployment allocations.

        Returns:
            DemandSatisfactionMetrics with satisfaction rate, unserved volume, gap, and status.
        """
        # 1. Resolve required demand
        required_demand = (
            scenario.forecasted_demand
            if scenario.forecasted_demand is not None
            else scenario.cargo_demand
        )

        # 2. Resolve delivered cargo
        if delivered_cargo is not None:
            actual_delivered = float(delivered_cargo)
        elif fleet_composition is not None:
            actual_delivered = float(fleet_composition.total_capacity)
        elif deployment_plan is not None:
            actual_delivered = self._sum_deployment_capacity(deployment_plan)
        else:
            actual_delivered = 0.0

        # 3. Handle zero or negative demand edge case
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

        # 4. Standard demand satisfaction metrics
        delivered_effective = max(0.0, actual_delivered)
        raw_ratio = delivered_effective / required_demand
        satisfaction_rate = min(1.0, max(0.0, raw_ratio))
        unserved = max(0.0, required_demand - delivered_effective)
        service_gap = max(0.0, scenario.service_level - satisfaction_rate)
        is_satisfied = satisfaction_rate >= scenario.service_level

        status = "SATISFIED" if is_satisfied else "UNSATISFIED"

        self.logger.info(
            "Demand Evaluation: Required=%.1f, Delivered=%.1f (%.1f%%), Target=%.1f%%, Status=%s",
            required_demand,
            delivered_effective,
            satisfaction_rate * 100.0,
            scenario.service_level * 100.0,
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
                total += float(v.get("allocated_capacity_dwt", 0.0))
        return total
