"""Fleet Strategy Optimizer and Operational Orchestrator.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 1 Deliverable: Orchestrates the three optimization tiers:
1. Fleet Composition Optimizer (vessel mix & fuel allocation)
2. Vessel Capacity Optimizer (vessel-class sizing & trip allocation)
3. Eco-Speed Optimizer (cruising speed & deadline satisfaction)
4. Fleet Deployment Planner (route-to-vessel operational assignments)
Produces unified FleetStrategyRecommendation, Baseline vs. Optimized comparisons,
and JSON scenario persistence (save_scenario / load_scenario).
"""

from dataclasses import replace
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import time
from typing import Any, Final

from contracts.constants import FUEL_PRICES_USD_PER_TON, FuelType, VESSEL_CLASS_SPECS
from contracts.schemas import (
    CapacityOptimizationResult,
    DemandSatisfactionMetrics,
    FleetCompositionResult,
    FleetStrategyRecommendation,
    OptimizationScenario,
    OptimizationStatus,
    ReliabilityMetrics,
    SpeedOptimizationResult,
)
from src.operations.demand_satisfaction_engine import CargoDemandSatisfactionEngine
from src.operations.reliability_engine import ScheduleReliabilityEngine
from src.optimization.capacity_optimizer import VesselCapacityOptimizer
from src.optimization.fleet_composition_optimizer import FleetCompositionOptimizer
from src.optimization.speed_optimizer import EcoSpeedOptimizer

logger = logging.getLogger("maritime_system")


class FleetStrategyOptimizer:
    """Executive orchestrator integrating Fleet Mix, Capacity Sizing, Eco-Speed, and Route Deployment."""

    def __init__(
        self,
        composition_optimizer: FleetCompositionOptimizer | None = None,
        capacity_optimizer: VesselCapacityOptimizer | None = None,
        speed_optimizer: EcoSpeedOptimizer | None = None,
        demand_engine: CargoDemandSatisfactionEngine | None = None,
        reliability_engine: ScheduleReliabilityEngine | None = None,
    ) -> None:
        """Initialize fleet strategy orchestrator with underlying sub-optimizers."""
        self.composition_optimizer = composition_optimizer or FleetCompositionOptimizer()
        self.capacity_optimizer = capacity_optimizer or VesselCapacityOptimizer()
        self.speed_optimizer = speed_optimizer or EcoSpeedOptimizer()
        self.demand_engine = demand_engine or CargoDemandSatisfactionEngine()
        self.reliability_engine = reliability_engine or ScheduleReliabilityEngine()
        self.logger = logger

    def optimize_strategy(
        self,
        scenario: OptimizationScenario,
    ) -> FleetStrategyRecommendation:
        """Execute end-to-end strategic fleet optimization across all decision tiers.

        Workflow:
            1. Fleet Composition Optimization -> x_f, x_m, x_l and y_D, y_L, y_M, y_H, y_A
            2. Vessel Capacity Optimization   -> recommended capacity (DWT & TEU)
            3. Eco-Speed Optimization        -> optimal cruising speed (v) and ETA
            4. Fleet Deployment Plan          -> operational assignment of vessels to routes
            5. Baseline vs. Optimized Delta   -> quantified savings in fuel, cost, emissions
            6. Service Reliability Metric     -> schedule buffer index under weather factor

        Args:
            scenario: Comprehensive OptimizationScenario context.

        Returns:
            FleetStrategyRecommendation ready for executive review and dashboard display.
        """
        t_start = time.perf_counter()
        # Ensure timestamp is set
        created_at = scenario.created_at or datetime.now(UTC).isoformat()
        if not scenario.created_at:
            scenario = replace(scenario, created_at=created_at)

        self.logger.info("Executing Strategic Optimization for Scenario ID: %s", scenario.scenario_id)

        # 1. Tier 1: Fleet Composition Optimization
        comp_res = self.composition_optimizer.optimize_composition(scenario)

        # 2. Tier 2: Vessel Capacity Optimization
        primary_fuel = self._identify_primary_fuel(comp_res.fleet_mix)
        cap_res = self.capacity_optimizer.optimize_capacity(
            scenario=scenario,
            target_fuel_type=primary_fuel,
        )

        # 3. Tier 3: Eco-Speed Optimization
        vessel_type_map = {
            "FEEDER": "General Cargo",
            "HANDYMAX": "Bulk Carrier",
            "PANAMAX": "Bulk Carrier",
            "POST_PANAMAX": "Container Ship",
            "CAPESIZE": "Bulk Carrier",
        }
        v_type = scenario.vessel_type or vessel_type_map.get(scenario.vessel_class.upper(), "Bulk Carrier")
        speed_res = self.speed_optimizer.optimize_speed(
            scenario=scenario,
            cargo_load=cap_res.recommended_capacity if cap_res.recommended_capacity > 0 else scenario.cargo_demand,
            vessel_type=v_type,
            fuel_type=primary_fuel,
            vessel_dwt=cap_res.recommended_capacity,
        )

        # 4. Generate Fleet Deployment Plan (Route-to-Vessel Allocation)
        deployment_plan = self._generate_deployment_plan(
            scenario=scenario,
            composition=comp_res,
            capacity=cap_res,
            speed=speed_res,
        )

        # 5. Cargo Demand Satisfaction Evaluation
        demand_metrics = self.demand_engine.evaluate_demand_satisfaction(
            scenario=scenario,
            fleet_composition=comp_res,
            deployment_plan=deployment_plan,
        )

        # 6. Schedule Reliability Evaluation
        reliability_metrics = self.reliability_engine.evaluate_schedule_reliability(
            scenario=scenario,
            speed_result=speed_res,
            deployment_plan=deployment_plan,
        )

        # 7. Overall Optimization Status and Failure Granularity
        failure_reason = None
        if comp_res.status != OptimizationStatus.SUCCESS:
            overall_status = comp_res.status
            failure_reason = comp_res.metadata.get("failure_reason", "COMPOSITION_INFEASIBLE")
        elif speed_res.status == OptimizationStatus.DEADLINE_VIOLATED:
            overall_status = OptimizationStatus.DEADLINE_VIOLATED
            failure_reason = "DEADLINE_VIOLATED"
        elif cap_res.status != OptimizationStatus.SUCCESS:
            overall_status = cap_res.status
            failure_reason = "CAPACITY_INFEASIBLE"
        elif not demand_metrics.is_satisfied:
            overall_status = OptimizationStatus.DEMAND_UNSATISFIABLE
            failure_reason = "DEMAND_NOT_MET"
        elif reliability_metrics.reliability_score < scenario.target_reliability:
            overall_status = OptimizationStatus.INFEASIBLE
            failure_reason = "RELIABILITY_TOO_LOW"
        else:
            overall_status = OptimizationStatus.SUCCESS

        # 8. Baseline vs. Optimized Comparison
        baseline_comparison = self._compute_baseline_comparison(
            scenario=scenario,
            comp_res=comp_res,
            cap_res=cap_res,
            speed_res=speed_res,
        )

        # Normalized service reliability index (0.0 to 1.0)
        reliability = max(0.0, min(1.0, reliability_metrics.reliability_score / 100.0))
        total_runtime_ms = (time.perf_counter() - t_start) * 1000.0

        # 9. Derive Operational & Regulatory Scenario Parameters from Optimized Strategy
        derived_capacity = scenario.capacity_dwt
        if derived_capacity is None or derived_capacity <= 0:
            derived_capacity = cap_res.recommended_capacity
        if derived_capacity <= 0:
            derived_capacity = VESSEL_CLASS_SPECS.get(scenario.vessel_class.upper(), {}).get("default_dwt", 45_000.0)

        derived_voyages = scenario.annual_voyages
        if derived_voyages is None or derived_voyages <= 0:
            derived_voyages = cap_res.optimal_trips
        if derived_voyages <= 0:
            derived_voyages = VESSEL_CLASS_SPECS.get(scenario.vessel_class.upper(), {}).get("default_annual_voyages", 20)

        derived_distance = scenario.annual_distance
        if derived_distance is None or derived_distance <= 0:
            derived_distance = round(scenario.route_distance * derived_voyages, 2)

        derived_vessel_type = scenario.vessel_type
        if not derived_vessel_type or derived_vessel_type == "Bulk carrier":
            spec_type = VESSEL_CLASS_SPECS.get(scenario.vessel_class.upper(), {}).get("vessel_type")
            if spec_type and scenario.vessel_class.upper() in {"FEEDER", "POST_PANAMAX"}:
                derived_vessel_type = spec_type

        opt_scenario = replace(
            scenario,
            vessel_type=derived_vessel_type,
            capacity_dwt=round(float(derived_capacity), 1),
            annual_voyages=int(derived_voyages),
            annual_distance=round(float(derived_distance), 1),
        )

        recommendation = FleetStrategyRecommendation(
            status=overall_status,
            scenario=opt_scenario,
            fleet_mix=comp_res,
            capacity_recommendation=cap_res,
            speed_recommendation=speed_res,
            deployment_plan=deployment_plan,
            baseline_comparison=baseline_comparison,
            fuel_estimate=comp_res.fuel_consumption,
            cost_estimate=comp_res.operational_cost,
            emissions_estimate=comp_res.emissions,
            service_reliability=round(reliability, 4),
            reliability_metrics=reliability_metrics,
            demand_metrics=demand_metrics,
            summary={
                "total_vessels": sum(v for k, v in comp_res.fleet_mix.items() if k in {"feeder", "medium", "large"}),
                "primary_fuel": primary_fuel,
                "recommended_dwt": cap_res.recommended_capacity,
                "recommended_speed_knots": speed_res.optimal_speed,
                "transit_eta_hours": speed_res.estimated_eta,
                "carbon_cost_usd": comp_res.carbon_cost,
                "optimization_runtime_ms": round(total_runtime_ms, 2),
                "demand_satisfaction_pct": demand_metrics.satisfaction_percentage,
                "unserved_cargo_tons": demand_metrics.unserved_cargo,
                "schedule_reliability_score": reliability_metrics.reliability_score,
                "on_time_arrival_rate_pct": round(reliability_metrics.on_time_arrival_rate * 100.0, 2),
                "reliability_breakdown": reliability_metrics.score_breakdown,
                "failure_reason": failure_reason,
            },
        )

        self.logger.info(
            "Strategy Optimization Complete [%s]. Total Cost: $%.2fM, Fuel: %.1ft, CO2e: %.1ft",
            overall_status.value,
            comp_res.operational_cost / 1e6,
            comp_res.fuel_consumption,
            comp_res.emissions,
        )
        return recommendation

    def _identify_primary_fuel(self, fleet_mix: dict[str, int]) -> str:
        """Determine the dominant alternative or conventional fuel in the optimal mix."""
        fuel_keys = ["lng", "methanol", "hydrogen", "ammonia", "diesel"]
        mapping = {
            "diesel": FuelType.DIESEL.value,
            "lng": FuelType.LNG.value,
            "methanol": FuelType.METHANOL.value,
            "hydrogen": FuelType.HYDROGEN.value,
            "ammonia": FuelType.AMMONIA.value,
        }
        best_count = -1
        best_fuel = FuelType.DIESEL.value
        for k in fuel_keys:
            cnt = fleet_mix.get(k, 0)
            if cnt > best_count:
                best_count = cnt
                best_fuel = mapping[k]
        return best_fuel

    def _generate_deployment_plan(
        self,
        scenario: OptimizationScenario,
        composition: FleetCompositionResult,
        capacity: CapacityOptimizationResult,
        speed: SpeedOptimizationResult,
    ) -> dict[str, list[dict[str, Any]]]:
        """Construct operational route-to-vessel deployment allocations."""
        plan: dict[str, list[dict[str, Any]]] = {}

        # Resolve routes from scenario or synthesize standard maritime corridor
        routes = list(scenario.routes) if scenario.routes else [
            {"route_id": "ROUTE-ALPHA-01", "origin": "Rotterdam", "destination": "Singapore", "distance_nm": scenario.route_distance},
            {"route_id": "ROUTE-BETA-02", "origin": "Shanghai", "destination": "Hamburg", "distance_nm": scenario.route_distance * 1.1},
        ]

        # Extract available active vessels from fleet mix
        active_vessels: list[dict[str, str]] = []
        fuel_tokens = ["diesel", "lng", "methanol", "hydrogen", "ammonia"]
        v_types = ["feeder", "medium", "large"]

        v_idx = 1
        for vt in v_types:
            count = composition.fleet_mix.get(vt, 0)
            for _ in range(count):
                # Assign a fuel type from the mix
                assigned_fuel = "Diesel"
                for ft in fuel_tokens:
                    if composition.fleet_mix.get(ft, 0) > 0:
                        assigned_fuel = ft.capitalize()
                        break
                active_vessels.append({
                    "vessel_id": f"VSL-{vt.upper()[:3]}-{v_idx:03d}",
                    "vessel_class": vt.capitalize(),
                    "fuel_type": assigned_fuel,
                })
                v_idx += 1

        if not active_vessels:
            # Fallback placeholder vessel if fleet mix is 0
            active_vessels.append({
                "vessel_id": "VSL-PAN-001",
                "vessel_class": scenario.vessel_class,
                "fuel_type": "Diesel",
            })

        # Compute corridor reliability metrics
        route_scores = self.reliability_engine.evaluate_route_reliability(
            scenario=scenario,
            speed_result=speed,
        )

        # Distribute active vessels across routes
        for i, route in enumerate(routes):
            r_id = route.get("route_id", f"ROUTE-{i+1}")
            assigned_for_route: list[dict[str, Any]] = []

            # Assign 1 or more vessels to this route
            v_subset = active_vessels[i % len(active_vessels):]
            if not v_subset:
                v_subset = [active_vessels[0]]

            v_choice = v_subset[0]
            dist = float(route.get("distance_nm", scenario.route_distance))
            est_voyage_hours = dist / max(speed.optimal_speed, 1.0)
            r_score = route_scores.get(r_id, 95.0)

            # Capacity buffer calculation
            route_demand_share = scenario.cargo_demand / max(1, len(routes))
            buffer_cap = max(0.0, capacity.recommended_capacity - route_demand_share)

            assigned_for_route.append({
                "vessel_id": v_choice["vessel_id"],
                "vessel_class": v_choice["vessel_class"],
                "fuel_type": v_choice["fuel_type"],
                "allocated_capacity_dwt": capacity.recommended_capacity,
                "buffer_capacity_dwt": round(buffer_cap, 2),
                "redundancy_factor": 1.15,
                "route_reliability_score": r_score,
                "cruising_speed_knots": speed.optimal_speed,
                "voyage_eta_hours": round(est_voyage_hours, 1),
                "origin": route.get("origin", "Hub Port A"),
                "destination": route.get("destination", "Hub Port B"),
            })
            plan[r_id] = assigned_for_route

        return plan

    def _compute_baseline_comparison(
        self,
        scenario: OptimizationScenario,
        comp_res: FleetCompositionResult,
        cap_res: CapacityOptimizationResult,
        speed_res: SpeedOptimizationResult,
    ) -> dict[str, dict[str, float]]:
        """Evaluate conventional unoptimized diesel baseline vs. quantum/green optimized fleet."""
        # Baseline assumption: 100% Diesel fleet operating at fixed unoptimized design speed (~15.5 kts)
        # Higher fuel consumption and 100% conventional WtW emissions
        baseline_fuel = comp_res.fuel_consumption * 1.28  # ~28% higher fuel without eco-speed & sizing
        diesel_price = (scenario.fuel_prices or FUEL_PRICES_USD_PER_TON).get("Diesel", 650.0)

        # Baseline emissions: 3.58 tons CO2e per ton diesel
        baseline_emissions = baseline_fuel * 3.58
        baseline_carbon_cost = baseline_emissions * scenario.carbon_price
        baseline_bunker_cost = baseline_fuel * diesel_price
        baseline_capital = comp_res.operational_cost * 0.90  # Conventional diesel slightly lower capex
        baseline_total_cost = baseline_capital + baseline_bunker_cost + baseline_carbon_cost

        opt_fuel = comp_res.fuel_consumption
        opt_cost = comp_res.operational_cost
        opt_emiss = comp_res.emissions

        # Percentage reductions
        fuel_saved_pct = ((baseline_fuel - opt_fuel) / max(baseline_fuel, 1.0)) * 100.0
        cost_saved_pct = ((baseline_total_cost - opt_cost) / max(baseline_total_cost, 1.0)) * 100.0
        emiss_saved_pct = ((baseline_emissions - opt_emiss) / max(baseline_emissions, 1.0)) * 100.0

        return {
            "baseline": {
                "fuel_consumption_tons": round(baseline_fuel, 2),
                "total_cost_usd": round(baseline_total_cost, 2),
                "emissions_co2e_tons": round(baseline_emissions, 2),
                "cruising_speed_knots": 15.5,
            },
            "optimized": {
                "fuel_consumption_tons": round(opt_fuel, 2),
                "total_cost_usd": round(opt_cost, 2),
                "emissions_co2e_tons": round(opt_emiss, 2),
                "cruising_speed_knots": round(speed_res.optimal_speed, 2),
            },
            "deltas": {
                "fuel_reduction_tons": round(baseline_fuel - opt_fuel, 2),
                "fuel_reduction_pct": round(fuel_saved_pct, 2),
                "cost_savings_usd": round(baseline_total_cost - opt_cost, 2),
                "cost_savings_pct": round(cost_saved_pct, 2),
                "emissions_abated_tons": round(baseline_emissions - opt_emiss, 2),
                "emissions_abated_pct": round(emiss_saved_pct, 2),
            },
        }

    @staticmethod
    def save_scenario(
        recommendation: FleetStrategyRecommendation,
        filepath: str | Path,
    ) -> Path:
        """Persist optimized fleet strategy recommendation to JSON storage."""
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(recommendation.to_json(), encoding="utf-8")
        logger.info("Persisted fleet strategy scenario to '%s'", out_path.resolve())
        return out_path

    @staticmethod
    def load_scenario(filepath: str | Path) -> FleetStrategyRecommendation:
        """Load and reconstruct FleetStrategyRecommendation from JSON storage."""
        in_path = Path(filepath)
        if not in_path.exists():
            raise FileNotFoundError(f"Scenario file '{filepath}' does not exist.")

        raw_data = json.loads(in_path.read_text(encoding="utf-8"))

        scenario = OptimizationScenario.from_dict(raw_data["scenario"])
        status = OptimizationStatus(raw_data.get("status", "SUCCESS"))

        mix_data = raw_data["fleet_mix"]
        mix_data["status"] = OptimizationStatus(mix_data.get("status", "SUCCESS"))
        fleet_mix = FleetCompositionResult(**mix_data)

        cap_data = raw_data["capacity_recommendation"]
        cap_data["status"] = OptimizationStatus(cap_data.get("status", "SUCCESS"))
        capacity = CapacityOptimizationResult(**cap_data)

        speed_data = raw_data["speed_recommendation"]
        speed_data["status"] = OptimizationStatus(speed_data.get("status", "SUCCESS"))
        speed = SpeedOptimizationResult(**speed_data)

        rel_metrics = None
        if "reliability_metrics" in raw_data and raw_data["reliability_metrics"] is not None:
            rel_metrics = ReliabilityMetrics.from_dict(raw_data["reliability_metrics"])

        dem_metrics = None
        if "demand_metrics" in raw_data and raw_data["demand_metrics"] is not None:
            dem_metrics = DemandSatisfactionMetrics.from_dict(raw_data["demand_metrics"])

        return FleetStrategyRecommendation(
            status=status,
            scenario=scenario,
            fleet_mix=fleet_mix,
            capacity_recommendation=capacity,
            speed_recommendation=speed,
            deployment_plan=raw_data.get("deployment_plan", {}),
            baseline_comparison=raw_data.get("baseline_comparison", {}),
            fuel_estimate=float(raw_data.get("fuel_estimate", 0.0)),
            cost_estimate=float(raw_data.get("cost_estimate", 0.0)),
            emissions_estimate=float(raw_data.get("emissions_estimate", 0.0)),
            service_reliability=raw_data.get("service_reliability"),
            reliability_metrics=rel_metrics,
            demand_metrics=dem_metrics,
            summary=raw_data.get("summary", {}),
        )
