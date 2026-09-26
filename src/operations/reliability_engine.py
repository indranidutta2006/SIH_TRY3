"""Schedule Reliability Engine for maritime green fleet operations.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 2 Deliverable: Computes normalized schedule reliability, decomposed score components,
route-level reliability metrics, and port congestion / weather disruption impacts.
"""

from collections.abc import Sequence
import logging
from typing import Any, Final

from contracts.schemas import (
    OptimizationScenario,
    ReliabilityMetrics,
    SpeedOptimizationResult,
)

logger = logging.getLogger("maritime_system")


class ScheduleReliabilityEngine:
    """Evaluates schedule reliability, delay demurrage, and on-time service levels."""

    def __init__(
        self,
        weight_on_time: float = 1.0,
        weight_delay: float = 0.5,
        weight_missed: float = 0.5,
    ) -> None:
        """Initialize reliability engine with standardized weighting constants.

        Args:
            weight_on_time: Multiplier for on-time arrival rate (default 1.0).
            weight_delay: Penalty multiplier for transit delay ratio (default 0.5).
            weight_missed: Penalty multiplier for missed voyage rate (default 0.5).
        """
        self.w1: Final[float] = weight_on_time
        self.w2: Final[float] = weight_delay
        self.w3: Final[float] = weight_missed
        self.logger = logger

    def evaluate_schedule_reliability(
        self,
        scenario: OptimizationScenario,
        speed_result: SpeedOptimizationResult | None = None,
        deployment_plan: dict[str, list[dict[str, Any]]] | None = None,
        simulated_delays: Sequence[float] | None = None,
        missed_voyages: int = 0,
        total_voyages: int | None = None,
    ) -> ReliabilityMetrics:
        """Evaluate overall schedule reliability across fleet operations and corridors.

        Mathematical Formulation:
            missed_rate = missed_voyages / total_voyages
            delay_ratio = min(1.0, average_delay_hours / deadline_hours)
            Reliability = max(0.0, min(100.0, 100 * (w1 * on_time_rate - w2 * delay_ratio - w3 * missed_rate)))

        Args:
            scenario: Operational scenario context.
            speed_result: Cruising speed recommendation with estimated ETA and delays.
            deployment_plan: Optional route-to-vessel deployment allocations.
            simulated_delays: Optional list of observed voyage delays for Monte Carlo or historical simulation.
            missed_voyages: Number of aborted or unperformed voyages.
            total_voyages: Total scheduled voyages across the evaluation window.

        Returns:
            ReliabilityMetrics with normalized score, score breakdown, and route reliability.
        """
        deadline = max(scenario.deadline_hours, 1.0)

        # 1. Resolve voyage delay distribution and enforce strict observation cardinality
        if simulated_delays is not None:
            delays = list(simulated_delays)
            if total_voyages is not None and len(delays) != total_voyages:
                raise ValueError(
                    f"Observation cardinality mismatch: received {len(delays)} delay observations "
                    f"for total_voyages={total_voyages}. Number of observations must match number of voyages evaluated."
                )
            n_voyages = total_voyages if total_voyages is not None else (len(delays) + missed_voyages)
            avg_delay = sum(delays) / max(len(delays), 1) if delays else 0.0
            max_delay = max(delays) if delays else 0.0
            on_time_count = sum(1 for d in delays if d <= 0.0)
            on_time_rate = on_time_count / max(n_voyages, 1)
        else:
            # Derive from physical transit dynamics, weather resistance, and port bottlenecks
            cruising_speed = speed_result.optimal_speed if speed_result is not None else 14.0
            transit_hours = (scenario.route_distance / max(cruising_speed, 1.0)) * scenario.weather_factor
            port_hours = max(0.0, (scenario.port_delay_factor - 1.0) * 12.0)
            base_delay = speed_result.delay_hours if speed_result is not None else 0.0
            effective_delay = max(base_delay, max(0.0, transit_hours + port_hours - deadline))

            delays = [effective_delay]
            n_voyages = total_voyages or 10
            avg_delay = effective_delay
            max_delay = effective_delay
            on_time_rate = 1.0 if effective_delay <= 0.0 else max(0.0, 1.0 - (effective_delay / deadline))

        # 2. Normalize rates
        missed_count = max(0, missed_voyages)
        missed_rate = missed_count / max(n_voyages, 1)
        delay_ratio = min(1.0, max(0.0, avg_delay / deadline))

        # 3. Decomposed score calculation
        on_time_comp = 100.0 * self.w1 * on_time_rate
        delay_pen = 100.0 * self.w2 * delay_ratio
        missed_pen = 100.0 * self.w3 * missed_rate

        raw_score = on_time_comp - delay_pen - missed_pen
        reliability_score = max(0.0, min(100.0, raw_score))

        score_breakdown = {
            "on_time_component": round(on_time_comp, 2),
            "delay_penalty": round(delay_pen, 2),
            "missed_voyage_penalty": round(missed_pen, 2),
        }

        # 4. Evaluate route-level reliability and corridor metrics
        route_metrics = self.evaluate_route_metrics(
            scenario=scenario,
            speed_result=speed_result,
            deployment_plan=deployment_plan,
        )
        route_rel = route_metrics["route_reliability"]
        route_delays = route_metrics["route_delay"]
        route_demands = route_metrics["route_demand"]
        route_utilizations = route_metrics["route_utilization"]

        # 5. Build reliability trace for dashboard explainability
        reliability_trace: list[dict[str, Any]] = []
        for i, (r_id, r_score) in enumerate(route_rel.items(), start=1):
            reliability_trace.append({
                "iteration": i,
                "route_id": r_id,
                "reliability_score": r_score,
                "average_delay_hours": round(route_delays.get(r_id, avg_delay), 1),
                "weather_severity": scenario.weather_factor,
                "port_congestion": scenario.port_delay_factor,
            })

        self.logger.info(
            "Schedule Reliability: Score=%.1f/100, On-Time=%.1f%%, Delay=%.1fh, Missed=%d/%d",
            reliability_score,
            on_time_rate * 100.0,
            avg_delay,
            missed_count,
            n_voyages,
        )

        return ReliabilityMetrics(
            reliability_score=round(reliability_score, 2),
            on_time_arrival_rate=round(on_time_rate, 4),
            average_delay_hours=round(avg_delay, 2),
            max_delay_hours=round(max_delay, 2),
            missed_voyages=missed_count,
            total_voyages=n_voyages,
            score_breakdown=score_breakdown,
            route_reliability=route_rel,
            reliability_trace=tuple(reliability_trace),
            route_delays=route_delays,
            route_demands=route_demands,
            route_utilizations=route_utilizations,
        )

    def evaluate_route_reliability(
        self,
        scenario: OptimizationScenario,
        speed_result: SpeedOptimizationResult | None = None,
        deployment_plan: dict[str, list[dict[str, Any]]] | None = None,
    ) -> dict[str, float]:
        """Compute individual schedule reliability scores across distinct maritime corridors."""
        return self.evaluate_route_metrics(scenario, speed_result, deployment_plan)["route_reliability"]

    def evaluate_route_metrics(
        self,
        scenario: OptimizationScenario,
        speed_result: SpeedOptimizationResult | None = None,
        deployment_plan: dict[str, list[dict[str, Any]]] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Compute corridor-level reliability, delays, demand allocations, and utilizations."""
        routes = list(scenario.routes) if scenario.routes else [
            {"route_id": "ROUTE-ASIA-EUR-01", "origin": "Shanghai", "destination": "Rotterdam", "distance_nm": scenario.route_distance},
            {"route_id": "ROUTE-IND-ME-02", "origin": "JNPT Mumbai", "destination": "Jebel Ali", "distance_nm": scenario.route_distance * 0.45},
            {"route_id": "ROUTE-IND-SGP-03", "origin": "Chennai", "destination": "Singapore", "distance_nm": scenario.route_distance * 0.55},
        ]

        cruising_speed = speed_result.optimal_speed if speed_result is not None else 14.0
        base_deadline = max(scenario.deadline_hours, 1.0)
        route_demand_per_corridor = scenario.cargo_demand / max(1, len(routes))

        rel_dict: dict[str, float] = {}
        delay_dict: dict[str, float] = {}
        demand_dict: dict[str, float] = {}
        util_dict: dict[str, float] = {}

        for i, route in enumerate(routes):
            r_id = route.get("route_id", f"ROUTE-{i+1}")
            dist = float(route.get("distance_nm", scenario.route_distance))
            r_deadline = max(24.0, (dist / max(scenario.route_distance, 1.0)) * base_deadline)

            transit_hours = (dist / max(cruising_speed, 1.0)) * scenario.weather_factor
            port_hours = max(0.0, (scenario.port_delay_factor - 1.0) * 10.0)
            total_time = transit_hours + port_hours

            delay = max(0.0, total_time - r_deadline)
            delay_ratio = min(1.0, delay / r_deadline)
            on_time = 1.0 if delay <= 0.0 else max(0.0, 1.0 - delay_ratio)

            score = max(0.0, min(100.0, 100.0 * (self.w1 * on_time - self.w2 * delay_ratio)))
            rel_dict[r_id] = round(score, 2)
            delay_dict[r_id] = round(delay, 2)
            demand_dict[r_id] = round(route_demand_per_corridor, 2)

            if deployment_plan and r_id in deployment_plan and deployment_plan[r_id]:
                v_list = deployment_plan[r_id]
                allocated_dwt = sum(float(v.get("allocated_capacity_dwt", 0.0)) for v in v_list)
                if allocated_dwt > 0:
                    util = min(1.0, max(0.0, route_demand_per_corridor / allocated_dwt))
                else:
                    util = 0.85
            else:
                util = 0.85
            util_dict[r_id] = round(util, 4)

        return {
            "route_reliability": rel_dict,
            "route_delay": delay_dict,
            "route_demand": demand_dict,
            "route_utilization": util_dict,
        }

    def evaluate_from_observations(
        self,
        total_voyages: int,
        on_time_voyages: int,
        missed_voyages: int,
        average_delay_hours: float,
        deadline_hours: float,
        route_reliability: dict[str, float] | None = None,
        observed_delays: Sequence[float] | None = None,
    ) -> ReliabilityMetrics:
        """Construct ReliabilityMetrics from empirical or historical telemetry records."""
        if total_voyages <= 0:
            raise ValueError(f"total_voyages must be strictly positive, got {total_voyages}")
        if observed_delays is not None and len(observed_delays) != total_voyages:
            raise ValueError(
                f"Observation cardinality mismatch: received {len(observed_delays)} delay observations "
                f"for total_voyages={total_voyages}. Number of observations must match number of voyages evaluated."
            )
        if on_time_voyages < 0 or missed_voyages < 0:
            raise ValueError("Voyage counts must be non-negative")
        if on_time_voyages + missed_voyages > total_voyages:
            raise ValueError(
                f"Inconsistent observation counts: on_time_voyages ({on_time_voyages}) + "
                f"missed_voyages ({missed_voyages}) exceeds total_voyages ({total_voyages})."
            )

        n_voyages = total_voyages
        deadline = max(1.0, deadline_hours)

        on_time_rate = max(0.0, min(1.0, on_time_voyages / n_voyages))
        missed_rate = max(0.0, min(1.0, missed_voyages / n_voyages))
        delay_ratio = min(1.0, max(0.0, average_delay_hours / deadline))

        on_time_comp = 100.0 * self.w1 * on_time_rate
        delay_pen = 100.0 * self.w2 * delay_ratio
        missed_pen = 100.0 * self.w3 * missed_rate

        raw_score = on_time_comp - delay_pen - missed_pen
        score = max(0.0, min(100.0, raw_score))

        breakdown = {
            "on_time_component": round(on_time_comp, 2),
            "delay_penalty": round(delay_pen, 2),
            "missed_voyage_penalty": round(missed_pen, 2),
        }

        routes = route_reliability or {"SYSTEM-OVERALL": round(score, 2)}
        route_delays = {r: round(average_delay_hours, 2) for r in routes}
        route_demands = {r: 0.0 for r in routes}
        route_utilizations = {r: 0.85 for r in routes}

        return ReliabilityMetrics(
            reliability_score=round(score, 2),
            on_time_arrival_rate=round(on_time_rate, 4),
            average_delay_hours=round(average_delay_hours, 2),
            max_delay_hours=round(average_delay_hours * 1.5, 2),
            missed_voyages=missed_voyages,
            total_voyages=total_voyages,
            score_breakdown=breakdown,
            route_reliability=routes,
            reliability_trace=(),
            route_delays=route_delays,
            route_demands=route_demands,
            route_utilizations=route_utilizations,
        )
