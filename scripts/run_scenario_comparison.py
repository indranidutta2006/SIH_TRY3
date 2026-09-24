"""Execute green fleet scenario comparison across all six marine fuel types.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Simulates a representative small fleet operating under:
1. Diesel (Conventional baseline)
2. LNG (Fossil transition fuel)
3. Methanol (Low-carbon alternative)
4. Hydrogen (Zero-emission green carrier)
5. Ammonia (Zero-carbon clean fuel)
6. ShorePower (Electrified zero-combustion port/coastal connection)

Routes Diesel/LNG/Methanol through the trained production ML model and
Hydrogen/Ammonia/ShorePower through the first-principles FuelPhysicsEngine.
Outputs results to outputs/reports/scenario_comparison.json.
"""

import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from logging_config import configure_logging
from contracts.constants import FuelType
from contracts.schemas import ScenarioResult
from src.optimization.scenario_analysis import ScenarioAnalysisEngine

logger = logging.getLogger("maritime_system")

REPRESENTATIVE_FLEET = [
    "VSL-BC-001",  # Bulk Carrier
    "VSL-CS-002",  # Container Ship
    "VSL-OT-003",  # Oil Tanker
    "VSL-BC-004",  # Bulk Carrier
    "VSL-GC-005",  # General Cargo
]

OPERATIONAL_SPECS = {
    "distance_nm": 1200.0,
    "speed_knots": 14.0,
    "cargo_tons": 35000.0,
    "vessel_dwt": 45000.0,
    "weather_factor": 1.05,
    "sea_state": 3,
    "compliance_year": 2025,
}

ALL_SCENARIO_FUELS = [
    ("Scenario_Diesel_Baseline", FuelType.DIESEL.value),
    ("Scenario_LNG_Transition", FuelType.LNG.value),
    ("Scenario_Methanol_LowCarbon", FuelType.METHANOL.value),
    ("Scenario_Hydrogen_Green", FuelType.HYDROGEN.value),
    ("Scenario_Ammonia_ZeroCarbon", FuelType.AMMONIA.value),
    ("Scenario_ShorePower_Electric", FuelType.SHORE_POWER.value),
]


def run_scenario_comparison(
    output_path: str = "outputs/reports/scenario_comparison.json",
) -> dict[str, Any]:
    """Execute complete 6-fuel scenario simulation and export comparative JSON report."""
    configure_logging(level="INFO")
    logger.info("Initializing Green Fleet Scenario Comparison across 6 fuel pathways...")

    engine = ScenarioAnalysisEngine()
    results: list[ScenarioResult] = []

    for name, fuel in ALL_SCENARIO_FUELS:
        params = dict(OPERATIONAL_SPECS)
        params["fuel_type"] = fuel
        res = engine.run_scenario(
            scenario_name=name,
            vessel_fleet=REPRESENTATIVE_FLEET,
            operational_parameters=params,
        )
        results.append(res)
        logger.info(
            "Completed %s: Fuel=%.2f t, WTW CO2e=%.2f t, Total Cost=$%.2f USD",
            res.scenario_name,
            res.fuel_consumption,
            res.total_emissions,
            res.total_cost,
        )

    # Multi-criteria tradeoff rankings
    balanced_rank = engine.compare_scenarios(results)
    cost_rank = engine.rank_by_cost(results)
    emissions_rank = engine.rank_by_emissions(results)

    report_payload: dict[str, Any] = {
        "metadata": {
            "title": "Green Fleet Alternative Fuel Transition Scenario Benchmark",
            "problem_id": "SIH26138",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "fleet_size": len(REPRESENTATIVE_FLEET),
            "vessels": list(REPRESENTATIVE_FLEET),
            "operational_parameters": OPERATIONAL_SPECS,
            "routing_architecture": {
                "ML_routed_fuels": ["Diesel", "LNG", "Methanol"],
                "Physics_routed_fuels": ["Hydrogen", "Ammonia", "ShorePower"],
            },
        },
        "scenarios": [r.to_dict() for r in results],
        "rankings": {
            "balanced_multicriteria_tradeoff": [r.scenario_name for r in balanced_rank],
            "lowest_cost_first": [r.scenario_name for r in cost_rank],
            "lowest_emissions_first": [r.scenario_name for r in emissions_rank],
        },
        "summary_table": [
            {
                "scenario": r.scenario_name,
                "fuel_type": r.fuel_type,
                "fuel_consumption_tons": r.fuel_consumption,
                "total_emissions_co2e_tons": r.total_emissions,
                "fuel_cost_usd": r.fuel_cost_usd,
                "fueleu_penalty_eur": r.fueleu_penalty_eur,
                "fueleu_penalty_usd": r.fueleu_penalty_usd,
                "exchange_rate_eur_to_usd": r.exchange_rate_eur_to_usd,
                "total_cost_usd": r.total_cost,
            }
            for r in results
        ],
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    logger.info("Saved scenario comparison report to %s", out_file.resolve())

    return report_payload


if __name__ == "__main__":
    report = run_scenario_comparison()
    print("\n=== GREEN FLEET SCENARIO BENCHMARK SUMMARY ===")
    for item in report["summary_table"]:
        print(
            f"{item['scenario']:<30} | Fuel: {item['fuel_type']:<11} | "
            f"Cons: {item['fuel_consumption_tons']:>8.2f} t | "
            f"CO2e: {item['total_emissions_co2e_tons']:>8.2f} t | "
            f"Bunker: ${item.get('fuel_cost_usd', 0.0):>10.2f} | "
            f"Penalty: €{item.get('fueleu_penalty_eur', 0.0):>9.2f} (${item.get('fueleu_penalty_usd', 0.0):>9.2f}) | "
            f"Total Cost: ${item['total_cost_usd']:>11.2f}"
        )
    print("\nBalanced Multicriteria Ranking:", report["rankings"]["balanced_multicriteria_tradeoff"])
