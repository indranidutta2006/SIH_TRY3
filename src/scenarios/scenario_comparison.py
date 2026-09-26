"""Multi-Scenario Comparative Sensitivity Analysis Engine.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 4 Deliverable: Evaluates fleet operational resilience and economic robustness across
parameterized macro-economic stress scenarios:
1. Baseline Scenario ($80/t carbon, nominal prices)
2. Low Carbon Price Scenario ($30/t carbon)
3. High Carbon Price Scenario ($150/t carbon)
4. Fuel Price Shock Scenario (+50% fossil bunker price)
5. High Demand Growth Scenario (+30% cargo demand)
6. Strict Regulation Scenario (Accelerated regulatory stringency)
"""

from collections.abc import Sequence
import logging
from typing import Any, Final

from contracts.constants import FUEL_PRICES_USD_PER_TON
from contracts.schemas import (
    OptimizationScenario,
    ScenarioComparisonResult,
    ScenarioDefinition,
)
from src.optimization.fleet_strategy_optimizer import FleetStrategyOptimizer

logger = logging.getLogger("maritime_system")

# Standardized canonical macro-economic scenario definitions
DEFAULT_SCENARIOS: Final[tuple[ScenarioDefinition, ...]] = (
    ScenarioDefinition(
        name="Baseline Scenario",
        description="Standard market conditions with $80/t carbon price, nominal bunker fuel spreads, and baseline cargo demand.",
        carbon_price=80.0,
        demand_multiplier=1.0,
        fossil_fuel_multiplier=1.0,
        alt_fuel_multiplier=1.0,
        regulation_factor=1.0,
        weather_severity_multiplier=1.0,
    ),
    ScenarioDefinition(
        name="Low Carbon Price",
        description="Delayed maritime ETS and IMO market-based measures with depressed $30/t carbon price.",
        carbon_price=30.0,
        demand_multiplier=1.0,
        fossil_fuel_multiplier=1.0,
        alt_fuel_multiplier=1.0,
        regulation_factor=0.85,
        weather_severity_multiplier=1.0,
    ),
    ScenarioDefinition(
        name="High Carbon Price",
        description="Aggressive global carbon taxation reaching $150/t, penalizing fossil fuel combustion.",
        carbon_price=150.0,
        demand_multiplier=1.0,
        fossil_fuel_multiplier=1.0,
        alt_fuel_multiplier=0.95,
        regulation_factor=1.15,
        weather_severity_multiplier=1.0,
    ),
    ScenarioDefinition(
        name="Fuel Price Shock",
        description="Geopolitical crude supply disruption causing a 50% spike in fossil bunker fuel prices.",
        carbon_price=80.0,
        demand_multiplier=1.0,
        fossil_fuel_multiplier=1.50,
        alt_fuel_multiplier=1.10,
        regulation_factor=1.0,
        weather_severity_multiplier=1.0,
    ),
    ScenarioDefinition(
        name="High Demand Growth",
        description="Global trade expansion driving a 30% increase in corridor cargo throughput requirements.",
        carbon_price=80.0,
        demand_multiplier=1.30,
        fossil_fuel_multiplier=1.0,
        alt_fuel_multiplier=1.0,
        regulation_factor=1.0,
        weather_severity_multiplier=1.0,
    ),
    ScenarioDefinition(
        name="Strict Regulation",
        description="Accelerated IMO CII reduction slope and statutory FuelEU compliance stringency increase.",
        carbon_price=110.0,
        demand_multiplier=1.0,
        fossil_fuel_multiplier=1.05,
        alt_fuel_multiplier=0.90,
        regulation_factor=1.25,
        weather_severity_multiplier=1.08,
    ),
)


class MultiScenarioAnalyzer:
    """Evaluates fleet operational and economic performance across multiple macro scenarios."""

    def __init__(self, optimizer: FleetStrategyOptimizer | None = None) -> None:
        """Initialize scenario comparison engine."""
        self.optimizer = optimizer or FleetStrategyOptimizer()
        self.logger = logger

    def compare_scenarios(
        self,
        base_scenario: OptimizationScenario,
        scenarios: Sequence[ScenarioDefinition] | None = None,
    ) -> ScenarioComparisonResult:
        """Execute comparative optimization across all parameterized scenarios.

        Args:
            base_scenario: Canonical baseline scenario context.
            scenarios: Collection of ScenarioDefinition instances (defaults to canonical 6).

        Returns:
            ScenarioComparisonResult capturing comparative metrics, rankings, and sensitivity.
        """
        eval_scenarios = tuple(scenarios) if scenarios is not None else DEFAULT_SCENARIOS
        self.logger.info("Executing Multi-Scenario Comparison across %d scenarios", len(eval_scenarios))

        metrics_comparison: dict[str, dict[str, float]] = {}
        scenario_names: list[str] = []

        for scen_def in eval_scenarios:
            self.logger.info("Evaluating scenario: %s", scen_def.name)
            scenario_names.append(scen_def.name)

            # 1. Transform fuel prices based on fossil and alternative fuel multipliers
            base_prices = dict(base_scenario.fuel_prices or FUEL_PRICES_USD_PER_TON)
            adjusted_fuel_prices: dict[str, float] = {}
            for k, price in base_prices.items():
                k_lower = k.lower()
                if k_lower in {"diesel", "mgo", "vlsfo", "lng", "hfo"}:
                    mult = scen_def.fossil_fuel_multiplier
                else:
                    mult = scen_def.alt_fuel_multiplier
                adj = round(price * mult, 2)
                adjusted_fuel_prices[k] = adj
                adjusted_fuel_prices[k.capitalize()] = adj
                adjusted_fuel_prices[k_lower] = adj

            # 2. Construct parameterized scenario with all scenario shocks
            adjusted_scenario = OptimizationScenario(
                cargo_demand=base_scenario.cargo_demand * scen_def.demand_multiplier,
                route_distance=base_scenario.route_distance,
                deadline_hours=base_scenario.deadline_hours,
                budget=base_scenario.budget * max(scen_def.demand_multiplier, 1.0),
                carbon_price=scen_def.carbon_price,
                weather_factor=base_scenario.weather_factor * scen_def.weather_severity_multiplier,
                port_delay_factor=base_scenario.port_delay_factor,
                target_reliability=base_scenario.target_reliability,
                service_level=base_scenario.service_level,
                vessel_class=base_scenario.vessel_class,
                scenario_id=f"SCEN-{scen_def.name.upper().replace(' ', '-')}",
                max_transition_rate=base_scenario.max_transition_rate,
                fuel_prices=adjusted_fuel_prices,
                regulation_factor=scen_def.regulation_factor,
            )

            # Optimize strategy under adjusted conditions
            rec = self.optimizer.optimize_strategy(adjusted_scenario)
            comp = rec.fleet_mix
            rel = rec.reliability_metrics
            dem = rec.demand_metrics
            v_cnt = float(rec.summary.get("total_vessels", sum(v for k, v in comp.fleet_mix.items() if k in {"feeder", "medium", "large"})))
            if v_cnt <= 0:
                v_cnt = float(max(1, sum(comp.fleet_mix.values())))
            rel_score = rel.reliability_score if rel else base_scenario.target_reliability
            dem_pct = dem.satisfaction_percentage if dem else 100.0
            opt_spd = rec.speed_recommendation.optimal_speed if rec.speed_recommendation else 14.0

            metrics_comparison[scen_def.name] = {
                "fuel_consumption_tons": round(comp.fuel_consumption, 1),
                "operational_cost_usd": round(comp.operational_cost, 2),
                "emissions_tons": round(comp.emissions, 1),
                "reliability_score": round(rel_score, 1),
                "demand_satisfaction_pct": round(dem_pct, 1),
                "vessel_count": v_cnt,
                "speed_knots": round(opt_spd, 2),
                "carbon_price": scen_def.carbon_price,
                "fossil_fuel_multiplier": scen_def.fossil_fuel_multiplier,
                "alt_fuel_multiplier": scen_def.alt_fuel_multiplier,
                "regulation_factor": scen_def.regulation_factor,
                "diesel_price_usd": adjusted_fuel_prices.get("Diesel", 650.0),
                "lng_price_usd": adjusted_fuel_prices.get("LNG", 800.0),
                "methanol_price_usd": adjusted_fuel_prices.get("Methanol", 950.0),
            }

        # Ranking based on operational cost and emissions
        # Lower cost + lower emissions = higher rank
        def rank_key(name: str) -> float:
            m = metrics_comparison[name]
            return m["operational_cost_usd"] + (m["emissions_tons"] * 100.0)

        ranking = tuple(sorted(scenario_names, key=rank_key))
        best_scenario = ranking[0]
        worst_scenario = ranking[-1]

        # Compute sensitivity spreads relative to baseline
        base_metrics = metrics_comparison.get("Baseline Scenario", metrics_comparison[scenario_names[0]])
        sensitivity = {}
        for name, m in metrics_comparison.items():
            if name == "Baseline Scenario":
                continue
            cost_delta_pct = ((m["operational_cost_usd"] - base_metrics["operational_cost_usd"]) / max(base_metrics["operational_cost_usd"], 1e-4)) * 100.0
            emiss_delta_pct = ((m["emissions_tons"] - base_metrics["emissions_tons"]) / max(base_metrics["emissions_tons"], 1e-4)) * 100.0
            sensitivity[name] = {
                "cost_impact_pct": round(cost_delta_pct, 2),
                "emissions_impact_pct": round(emiss_delta_pct, 2),
            }

        return ScenarioComparisonResult(
            scenario_names=tuple(scenario_names),
            metrics_comparison=metrics_comparison,
            ranking=ranking,
            best_scenario=best_scenario,
            worst_scenario=worst_scenario,
            sensitivity_analysis=sensitivity,
            metadata={
                "base_scenario_id": base_scenario.scenario_id,
                "total_scenarios_evaluated": len(scenario_names),
            },
        )
