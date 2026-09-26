"""Industrial Case Studies Suite for Maritime Green Fleet Optimization.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 4 Deliverable: Evaluates 4 canonical industrial maritime case studies:
- Case Study A: Conventional Diesel Fleet (High fossil exposure & carbon tax liability)
- Case Study B: LNG Transition Fleet (Dual-fuel bridging technology)
- Case Study C: Methanol Transition Fleet (Low retrofit capex green corridor)
- Case Study D: Hydrogen Future Fleet (Deep decarbonization with fuel cell / combustion)

For each case study, generates:
1. Fleet Mix & Capacity Allocation
2. Fuel Strategy & Speeds
3. Cost & Financial Economics
4. Lifecycle Assessment (LCA: WTW, WTT, TTW)
5. Forward Regulatory Compliance (CII & FuelEU)
6. Executive Management Recommendation
"""

import json
import logging
from pathlib import Path
import sys
from typing import Any

import pandas as pd

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from contracts.schemas import OptimizationScenario
from src.decision_support.executive_recommendation_engine import ExecutiveRecommendationEngine
from src.lifecycle.lifecycle_assessment_engine import MaritimeLifecycleAssessmentEngine
from src.optimization.fleet_strategy_optimizer import FleetStrategyOptimizer
from src.strategy.fuel_transition_planner import FuelTransitionPlanner
from src.strategy.regulatory_forecast import RegulatoryForecastEngine

logger = logging.getLogger("maritime_system")


class IndustrialCaseStudySuite:
    """Orchestrates comprehensive industrial maritime case studies."""

    def __init__(self) -> None:
        """Initialize case study suite with platform engines."""
        self.optimizer = FleetStrategyOptimizer()
        self.lca_engine = MaritimeLifecycleAssessmentEngine()
        self.reg_engine = RegulatoryForecastEngine()
        self.transition_planner = FuelTransitionPlanner(
            lca_engine=self.lca_engine,
            regulatory_engine=self.reg_engine,
        )
        self.exec_engine = ExecutiveRecommendationEngine()
        self.logger = logger

    def run_all_case_studies(
        self,
        output_dir: str | Path = "case_studies/reports",
    ) -> dict[str, dict[str, Any]]:
        """Execute all 4 industrial case studies and export summary reports."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        case_definitions = {
            "Case A: Conventional Diesel Fleet": {
                "scenario": OptimizationScenario(
                    cargo_demand=220_000.0,
                    route_distance=3200.0,
                    deadline_hours=240.0,
                    budget=100_000_000.0,
                    carbon_price=80.0,
                    max_transition_rate=0.0,  # 100% conventional diesel
                    scenario_id="CASE-A-DIESEL",
                ),
                "primary_fuel": "Diesel",
                "secondary_fuel": "Diesel",
                "vessel_class": "PANAMAX",
            },
            "Case B: LNG Transition Fleet": {
                "scenario": OptimizationScenario(
                    cargo_demand=250_000.0,
                    route_distance=3500.0,
                    deadline_hours=260.0,
                    budget=120_000_000.0,
                    carbon_price=80.0,
                    max_transition_rate=0.50,  # 50% dual-fuel LNG
                    scenario_id="CASE-B-LNG",
                ),
                "primary_fuel": "LNG",
                "secondary_fuel": "Diesel",
                "vessel_class": "PANAMAX",
            },
            "Case C: Methanol Transition Fleet": {
                "scenario": OptimizationScenario(
                    cargo_demand=280_000.0,
                    route_distance=4000.0,
                    deadline_hours=300.0,
                    budget=140_000_000.0,
                    carbon_price=90.0,
                    max_transition_rate=0.40,  # 40% green methanol
                    scenario_id="CASE-C-METHANOL",
                ),
                "primary_fuel": "Methanol",
                "secondary_fuel": "Hydrogen",
                "vessel_class": "PANAMAX",
            },
            "Case D: Hydrogen Future Fleet": {
                "scenario": OptimizationScenario(
                    cargo_demand=300_000.0,
                    route_distance=4500.0,
                    deadline_hours=320.0,
                    budget=180_000_000.0,
                    carbon_price=100.0,
                    max_transition_rate=0.75,  # 75% hydrogen / fuel-cell
                    scenario_id="CASE-D-HYDROGEN",
                ),
                "primary_fuel": "Hydrogen",
                "secondary_fuel": "Ammonia",
                "vessel_class": "CAPESIZE",
            },
        }

        case_results: dict[str, dict[str, Any]] = {}
        summary_rows = []

        for case_name, cfg in case_definitions.items():
            self.logger.info("Executing %s...", case_name)
            scen: OptimizationScenario = cfg["scenario"]
            v_class = cfg["vessel_class"]

            # 1. Strategy Optimization
            strat_rec = self.optimizer.optimize_strategy(scen)
            comp = strat_rec.fleet_mix
            rel = strat_rec.reliability_metrics
            v_cnt = strat_rec.summary.get("total_vessels", sum(v for k, v in comp.fleet_mix.items() if k in {"feeder", "medium", "large"}))
            if v_cnt <= 0:
                v_cnt = max(1, sum(comp.fleet_mix.values()))

            fuel_mix_raw = {
                k: comp.fleet_mix.get(k, 0)
                for k in ["diesel", "lng", "methanol", "hydrogen", "ammonia"]
                if comp.fleet_mix.get(k, 0) > 0
            }
            if not fuel_mix_raw:
                fuel_mix_raw = {"diesel": v_cnt}
            total_fuel_units = sum(fuel_mix_raw.values())
            fuel_shares = {k: v / max(total_fuel_units, 1) for k, v in fuel_mix_raw.items()}

            # 2. Multi-Year Transition Roadmap
            roadmap = self.transition_planner.plan_transition(
                scenario=scen,
                vessel_class=v_class,
                primary_green_fuel=cfg["primary_fuel"],
                secondary_green_fuel=cfg["secondary_fuel"],
            )

            # 3. Lifecycle Assessment (LCA)
            fuel_split = {
                f: comp.fuel_consumption * sh
                for f, sh in fuel_shares.items()
                if sh > 0.0
            }
            lca_results = self.lca_engine.assess_fleet_lifecycle(
                fuel_consumption=fuel_split,
                carbon_price_usd=scen.carbon_price,
            )
            total_wtw = sum(r.well_to_wake_emissions for r in lca_results.values())
            total_wtt = sum(r.well_to_tank_emissions for r in lca_results.values())
            total_ttw = sum(r.tank_to_wake_emissions for r in lca_results.values())

            # 4. Forward Regulatory Forecast
            forecast = self.reg_engine.forecast_compliance_trajectory(
                vessel_type="Bulk carrier",
                capacity_dwt=45000.0 if v_class == "PANAMAX" else 95000.0,
                annual_fuel_consumption_tons=comp.fuel_consumption / max(v_cnt, 1),
                annual_distance_nm=scen.route_distance * 20,
                fuel_shares=fuel_shares,
                start_year=2026,
                end_year=2035,
            )

            # 5. Executive Recommendation
            exec_rec = self.exec_engine.generate_recommendation(
                strategy_recommendation=strat_rec,
                scenario=scen,
                roadmap=roadmap,
                forecast=forecast,
            )

            rel_score = rel.reliability_score if rel else scen.target_reliability
            dem_pct = strat_rec.demand_metrics.satisfaction_percentage if strat_rec.demand_metrics else 100.0
            opt_spd = strat_rec.speed_recommendation.optimal_speed if strat_rec.speed_recommendation else 14.0

            record = {
                "case_name": case_name,
                "scenario_id": scen.scenario_id,
                "vessel_class": v_class,
                "vessel_count": v_cnt,
                "optimal_speed_knots": round(opt_spd, 2),
                "fuel_strategy": fuel_shares,
                "annual_fuel_consumption_tons": round(comp.fuel_consumption, 1),
                "annual_cost_usd": round(comp.operational_cost, 2),
                "lifecycle_emissions": {
                    "tank_to_wake_tons": round(total_ttw, 1),
                    "well_to_tank_tons": round(total_wtt, 1),
                    "well_to_wake_tons": round(total_wtw, 1),
                },
                "reliability_score": round(rel_score, 1),
                "demand_satisfaction_pct": round(dem_pct, 1),
                "roi_percentage": exec_rec.roi_percentage,
                "payback_years": exec_rec.payback_years,
                "payback_status": exec_rec.payback_status,
                "priority_actions": list(exec_rec.priority_actions),
                "roadmap_summary": {
                    "total_capex_usd": roadmap.total_transition_capex,
                    "cumulative_emissions_tons": roadmap.cumulative_emissions_tons,
                    "emissions_reduction_pct": roadmap.emissions_reduction_pct,
                },
            }

            case_results[case_name] = record

            summary_rows.append({
                "Case Study": case_name,
                "Class": v_class,
                "Vessels": v_cnt,
                "Speed (kn)": round(opt_spd, 1),
                "Annual Cost ($M)": round(comp.operational_cost / 1e6, 2),
                "WTW CO2e (t)": round(total_wtw, 0),
                "Reliability": f"{rel_score:.1f}%",
                "ROI (%)": f"{exec_rec.roi_percentage}%" if exec_rec.roi_percentage is not None else "N/A",
                "Payback": f"{exec_rec.payback_years} yrs" if exec_rec.payback_years is not None else exec_rec.payback_status,
            })

        # Export JSON report
        json_file = out_path / "industrial_case_studies.json"
        json_file.write_text(json.dumps(case_results, indent=2), encoding="utf-8")

        # Export Summary CSV
        df = pd.DataFrame(summary_rows)
        csv_file = out_path / "case_studies_summary.csv"
        df.to_csv(csv_file, index=False)

        self.logger.info("Exported industrial case studies to %s", out_path.resolve())
        return case_results


if __name__ == "__main__":
    suite = IndustrialCaseStudySuite()
    suite.run_all_case_studies()
