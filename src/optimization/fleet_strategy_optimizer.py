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

from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import time
from typing import Any, Final

from contracts.constants import FUEL_PRICES_USD_PER_TON, FuelType
from contracts.schemas import (
    CapacityOptimizationResult,
    FleetCompositionResult,
    FleetStrategyRecommendation,
    OptimizationScenario,
    OptimizationStatus,
    SpeedOptimizationResult,
)
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
    ) -> None:
        """Initialize fleet strategy orchestrator with underlying sub-optimizers."""
        self.composition_optimizer = composition_optimizer or FleetCompositionOptimizer()
        self.capacity_optimizer = capacity_optimizer or VesselCapacityOptimizer()
        self.speed_optimizer = speed_optimizer or EcoSpeedOptimizer()
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
            scenario = OptimizationScenario(
                cargo_demand=scenario.cargo_demand,
                route_distance=scenario.route_distance,
                deadline_hours=scenario.deadline_hours,
                scenario_id=scenario.scenario_id,
                created_at=created_at,
                carbon_price=scenario.carbon_price,
                budget=scenario.budget,
                weather_factor=scenario.weather_factor,
                vessel_class=scenario.vessel_class,
                service_level=scenario.service_level,
                max_transition_rate=scenario.max_transition_rate,
                fuel_prices=scenario.fuel_prices,
                routes=scenario.routes,
                weights=scenario.weights,
            )

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
            "PANAMAX": "Bulk Carrier",
            "POST_PANAMAX": "Container Ship",
            "CAPESIZE": "Bulk Carrier",
        }
        v_type = vessel_type_map.get(scenario.vessel_class.upper(), "Bulk Carrier")
        speed_res = self.speed_optimizer.optimize_speed(
            scenario=scenario,
            cargo_load=cap_res.recommended_capacity if cap_res.recommended_capacity > 0 else scenario.cargo_demand,
            vessel_type=v_type,
            fuel_type=primary_fuel,
            vessel_dwt=cap_res.recommended_capacity,
        )

        # 4. Overall Optimization Status
        # If any sub-optimizer failed, bubble up the critical status
        if comp_res.status != OptimizationStatus.SUCCESS:
            overall_status = comp_res.status
        elif speed_res.status == OptimizationStatus.DEADLINE_VIOLATED:
            overall_status = OptimizationStatus.DEADLINE_VIOLATED
        elif cap_res.status != OptimizationStatus.SUCCESS:
            overall_status = cap_res.status
        else:
            overall_status = OptimizationStatus.SUCCESS

        # 5. Generate Fleet Deployment Plan (Route-to-Vessel Allocation)
        deployment_plan = self._generate_deployment_plan(
            scenario=scenario,
            composition=comp_res,
            capacity=cap_res,
            speed=speed_res,
        )

        # 6. Baseline vs. Optimized Comparison
        baseline_comparison = self._compute_baseline_comparison(
            scenario=scenario,
            comp_res=comp_res,
            cap_res=cap_res,
            speed_res=speed_res,
        )

        # 7. Service Reliability Estimation
        # Buffer between transit duration and deadline scaled by weather
        delay_ratio = speed_res.delay_hours / max(scenario.deadline_hours, 1.0)
        reliability = max(0.0, min(1.0, 1.0 - (delay_ratio * scenario.weather_factor)))

        total_runtime_ms = (time.perf_counter() - t_start) * 1000.0

        recommendation = FleetStrategyRecommendation(
            status=overall_status,
            scenario=scenario,
            fleet_mix=comp_res,
            capacity_recommendation=cap_res,
            speed_recommendation=speed_res,
            deployment_plan=deployment_plan,
            baseline_comparison=baseline_comparison,
            fuel_estimate=comp_res.fuel_consumption,
            cost_estimate=comp_res.operational_cost,
            emissions_estimate=comp_res.emissions,
            service_reliability=round(reliability, 4),
            summary={
                "total_vessels": sum(v for k, v in comp_res.fleet_mix.items() if k in {"feeder", "medium", "large"}),
                "primary_fuel": primary_fuel,
                "recommended_dwt": cap_res.recommended_capacity,
                "recommended_speed_knots": speed_res.optimal_speed,
                "transit_eta_hours": speed_res.estimated_eta,
                "carbon_cost_usd": comp_res.carbon_cost,
                "optimization_runtime_ms": round(total_runtime_ms, 2),
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

            assigned_for_route.append({
                "vessel_id": v_choice["vessel_id"],
                "vessel_class": v_choice["vessel_class"],
                "fuel_type": v_choice["fuel_type"],
                "allocated_capacity_dwt": capacity.recommended_capacity,
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
            summary=raw_data.get("summary", {}),
        )
