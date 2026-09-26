"""Multi-Year Maritime Fuel Transition & Fleet Modernization Planner.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 4 Deliverable: Solves multi-period fleet decarbonization planning over 2026–2040 horizons,
co-optimizing retrofits, newbuild capital expenditure, operational bunker costs, and regulatory penalties.
Outputs actionable year-by-year TransitionRoadmap with TransitionMilestone checkpoints.
"""

from collections.abc import Sequence
import logging
from typing import Any, Final

import numpy as np

from contracts.schemas import (
    EvidenceCategory,
    OptimizationScenario,
    TransitionMilestone,
    TransitionRoadmap,
)
from src.lifecycle.lifecycle_assessment_engine import MaritimeLifecycleAssessmentEngine
from src.strategy.regulatory_forecast import RegulatoryForecastEngine

logger = logging.getLogger("maritime_system")

# Retrofit capital expenditure estimates per vessel class (USD)
RETROFIT_CAPEX_USD: Final[dict[str, dict[str, float]]] = {
    "HANDYMAX": {
        "LNG": 6_500_000.0,
        "Methanol": 4_200_000.0,
        "Ammonia": 9_000_000.0,
        "Hydrogen": 11_500_000.0,
    },
    "PANAMAX": {
        "LNG": 9_800_000.0,
        "Methanol": 6_500_000.0,
        "Ammonia": 13_500_000.0,
        "Hydrogen": 17_000_000.0,
    },
    "CAPESIZE": {
        "LNG": 15_000_000.0,
        "Methanol": 10_500_000.0,
        "Ammonia": 21_000_000.0,
        "Hydrogen": 26_000_000.0,
    },
}

# Technology Commercial Readiness Level (earliest viable operational deployment year)
TECH_READINESS_YEAR: Final[dict[str, int]] = {
    "Diesel": 2020,
    "LNG": 2022,
    "Methanol": 2025,
    "Ammonia": 2029,
    "Hydrogen": 2032,
}


class FuelTransitionPlanner:
    """Plans multi-period fleet decarbonization trajectories and dual-fuel retrofit roadmaps."""

    def __init__(
        self,
        lca_engine: MaritimeLifecycleAssessmentEngine | None = None,
        regulatory_engine: RegulatoryForecastEngine | None = None,
    ) -> None:
        """Initialize transition planner with LCA and regulatory forecasting engines."""
        self.lca_engine = lca_engine or MaritimeLifecycleAssessmentEngine()
        self.regulatory_engine = regulatory_engine or RegulatoryForecastEngine()
        self.logger = logger

    def plan_transition(
        self,
        scenario: OptimizationScenario,
        target_years: tuple[int, ...] = (2026, 2028, 2030, 2035, 2040),
        vessel_class: str = "PANAMAX",
        primary_green_fuel: str = "Methanol",
        secondary_green_fuel: str = "Hydrogen",
        max_retrofit_rate_per_period: float = 0.25,
        carbon_price_escalation_pct: float = 0.05,
        fuel_pathways: dict[str, str] | None = None,
    ) -> TransitionRoadmap:
        """Generate multi-year fleet transition roadmap minimizing total cost and emissions.

        Args:
            scenario: Base OptimizationScenario context.
            target_years: Milestone years to plan (default: 2026, 2028, 2030, 2035, 2040).
            vessel_class: Vessel classification (HANDYMAX, PANAMAX, CAPESIZE).
            primary_green_fuel: Near/medium-term transitional fuel.
            secondary_green_fuel: Long-term deep decarbonization fuel.
            max_retrofit_rate_per_period: Maximum fraction of fleet convertible per planning interval.
            carbon_price_escalation_pct: Annual carbon price escalation rate.
            fuel_pathways: Optional custom feedstock pathways mapping (e.g. {'Hydrogen': 'green'}).

        Returns:
            TransitionRoadmap capturing milestone decisions, costs, and emissions trajectory.
        """
        self.logger.info("Generating Fuel Transition Roadmap for vessel class: %s", vessel_class)

        # Baseline fleet sizing to satisfy scenario cargo demand
        nominal_capacity = 45000.0 if vessel_class == "PANAMAX" else (25000.0 if vessel_class == "HANDYMAX" else 95000.0)
        annual_voyages = 20
        required_vessels = max(4, int(np.ceil(scenario.cargo_demand / (nominal_capacity * annual_voyages))))

        milestones: list[TransitionMilestone] = []
        total_capex = 0.0
        total_opex = 0.0
        cumulative_emissions = 0.0

        current_alt_share = 0.0
        current_secondary_share = 0.0
        carbon_price = scenario.carbon_price

        # Track baseline emissions for reduction calculation
        base_annual_fuel_per_vessel = 3200.0
        base_emissions_per_year = required_vessels * base_annual_fuel_per_vessel * 3.206

        v_class_clean = vessel_class.upper() if vessel_class.upper() in RETROFIT_CAPEX_USD else "PANAMAX"
        prev_yr = target_years[0]

        for idx, yr in enumerate(target_years):
            # 1. Update carbon price with annual escalation
            elapsed_years = yr - prev_yr
            carbon_price *= ((1.0 + carbon_price_escalation_pct) ** elapsed_years)
            prev_yr = yr

            # 2. Compute fuel share targets based on tech availability & maximum transition rate
            if yr >= TECH_READINESS_YEAR.get(primary_green_fuel, 2026):
                # Transition into primary green fuel
                current_alt_share = min(0.85, current_alt_share + max_retrofit_rate_per_period)

            if yr >= TECH_READINESS_YEAR.get(secondary_green_fuel, 2032):
                # Transition into secondary advanced fuel
                current_secondary_share = min(0.50, current_secondary_share + (max_retrofit_rate_per_period * 0.75))

            f_secondary = round(current_secondary_share, 3)
            f_primary = round(max(0.0, current_alt_share - f_secondary), 3)
            f_diesel = round(max(0.0, 1.0 - (f_primary + f_secondary)), 3)

            fuel_shares = {
                "Diesel": f_diesel,
                primary_green_fuel: f_primary,
                secondary_green_fuel: f_secondary,
            }

            # 3. Calculate Retrofit Capex for newly converted vessels
            num_alt_vessels = int(round(required_vessels * (f_primary + f_secondary)))
            prev_alt_vessels = int(round(required_vessels * (milestones[-1].fuel_shares.get(primary_green_fuel, 0.0) + milestones[-1].fuel_shares.get(secondary_green_fuel, 0.0)))) if milestones else 0
            new_conversions = max(0, num_alt_vessels - prev_alt_vessels)

            unit_capex = RETROFIT_CAPEX_USD[v_class_clean].get(primary_green_fuel, 6_000_000.0)
            period_capex = float(new_conversions) * unit_capex
            total_capex += period_capex

            # 4. Calculate Operational Fuel, LCA, and Compliance Costs
            fleet_fuel_tons = required_vessels * base_annual_fuel_per_vessel
            fuel_split = {f: fleet_fuel_tons * sh for f, sh in fuel_shares.items() if sh > 0.0}

            pathways_map = {
                "Diesel": "fossil",
                primary_green_fuel: "e_methanol" if primary_green_fuel == "Methanol" else ("bio_lng" if primary_green_fuel == "LNG" else "green"),
                secondary_green_fuel: "green",
            }
            if fuel_pathways:
                pathways_map.update(fuel_pathways)

            lca_results = self.lca_engine.assess_fleet_lifecycle(
                fuel_consumption=fuel_split,
                pathways=pathways_map,
                carbon_price_usd=carbon_price,
            )

            annual_emissions = sum(res.well_to_wake_emissions for res in lca_results.values())
            annual_fuel_cost = sum(res.lifecycle_cost for res in lca_results.values())
            period_opex = annual_fuel_cost + (required_vessels * 2_500_000.0)  # Maintenance & crew
            total_opex += period_opex
            cumulative_emissions += annual_emissions

            # 5. Regulatory Checkpoint
            reg_res = self.regulatory_engine.forecast_compliance_trajectory(
                vessel_type="Bulk carrier",
                capacity_dwt=nominal_capacity,
                annual_fuel_consumption_tons=base_annual_fuel_per_vessel,
                annual_distance_nm=scenario.route_distance * annual_voyages,
                fuel_shares=fuel_shares,
                start_year=yr,
                end_year=yr,
                fuel_pathways=pathways_map,
            )

            cii_rate = reg_res.future_cii_ratings.get(yr, "C")
            pen_usd = reg_res.projected_penalties_usd.get(yr, 0.0) * required_vessels

            milestone = TransitionMilestone(
                year=yr,
                fleet_mix={v_class_clean: required_vessels},
                fuel_shares=fuel_shares,
                avg_speed_knots=13.5 if f_diesel > 0.5 else 14.2,
                capex_usd=round(period_capex, 2),
                opex_usd=round(period_opex, 2),
                annual_emissions_tons=round(annual_emissions, 2),
                cii_rating=cii_rate,
                fueleu_penalty_usd=round(pen_usd, 2),
                metadata={
                    "retrofitted_vessels_this_period": new_conversions,
                    "cumulative_retrofitted_vessels": num_alt_vessels,
                    "carbon_price_usd": round(carbon_price, 2),
                    "evidence": EvidenceCategory.MODELLED.value,
                },
            )
            milestones.append(milestone)

        final_emissions = milestones[-1].annual_emissions_tons
        emiss_reduction_pct = ((base_emissions_per_year - final_emissions) / max(base_emissions_per_year, 1e-4)) * 100.0
        total_cost = total_capex + total_opex

        # Regulatory risk score (0 to 100, where 0 is zero risk)
        penalties_total = sum(m.fueleu_penalty_usd for m in milestones)
        risk_score = min(100.0, (penalties_total / max(total_cost * 0.1, 1e-4)) * 100.0)

        return TransitionRoadmap(
            roadmap_id=f"ROADMAP-{v_class_clean}-{target_years[0]}-{target_years[-1]}",
            start_year=target_years[0],
            end_year=target_years[-1],
            milestones=tuple(milestones),
            total_transition_capex=round(total_capex, 2),
            total_transition_opex=round(total_opex, 2),
            total_transition_cost=round(total_cost, 2),
            cumulative_emissions_tons=round(cumulative_emissions, 2),
            emissions_reduction_pct=round(emiss_reduction_pct, 2),
            regulatory_risk_score=round(risk_score, 2),
            metadata={
                "vessel_class": v_class_clean,
                "fleet_size": required_vessels,
                "primary_fuel": primary_green_fuel,
                "secondary_fuel": secondary_green_fuel,
            },
        )
