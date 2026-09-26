"""Comprehensive unit and integration tests for Phase 4 Decision Intelligence & Strategic Decarbonization.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 4 Tests:
1. MaritimeLifecycleAssessmentEngine: FuelLifecycleProfile, Grey/Blue/Green/Bio/E-fuels, WTT, TTW, WTW, GHG intensity.
2. RegulatoryForecastEngine: 2026-2030 Statutory CII (MEPC.400(83)), 2031-2040 Scenario CII, FuelEU milestones,
   Article 23 penalty formula with consecutive multiplier, estimated_compliance_probability with Monte Carlo metadata.
3. FuelTransitionPlanner: Multi-year roadmaps (2026-2040), retrofit capex, tech readiness, cumulative emissions.
4. ExecutiveRecommendationEngine: Real investment ROI, payback logic (payback_years=None when benefit<=0),
   priority actions, risk matrix, evidence categorization layer.
5. MultiScenarioAnalyzer: Parameterized ScenarioDefinition objects across 6 canonical macro scenarios.
6. IndustrialCaseStudySuite: End-to-end execution of Cases A, B, C, D and report exports.
7. ExecutiveReportGenerator: Multi-format generation (PDF, Markdown, JSON).
"""

from pathlib import Path
import pytest
import numpy as np

from contracts.schemas import (
    EvidenceCategory,
    ExecutiveRecommendation,
    FuelLifecycleProfile,
    LifecycleAssessmentResult,
    OptimizationScenario,
    RegulatoryForecastResult,
    ScenarioComparisonResult,
    ScenarioDefinition,
    TransitionRoadmap,
)
from case_studies.run_case_studies import IndustrialCaseStudySuite
from src.decision_support.executive_recommendation_engine import ExecutiveRecommendationEngine
from src.lifecycle.lifecycle_assessment_engine import (
    DEFAULT_LIFECYCLE_PROFILES,
    MaritimeLifecycleAssessmentEngine,
)
from src.optimization.fleet_strategy_optimizer import FleetStrategyOptimizer
from src.reporting.executive_report_generator import ExecutiveReportGenerator
from src.scenarios.scenario_comparison import (
    DEFAULT_SCENARIOS,
    MultiScenarioAnalyzer,
)
from src.strategy.fuel_transition_planner import FuelTransitionPlanner
from src.strategy.regulatory_forecast import RegulatoryForecastEngine


@pytest.fixture
def base_scenario() -> OptimizationScenario:
    """Canonical test scenario for decision intelligence testing."""
    return OptimizationScenario(
        cargo_demand=150_000.0,
        route_distance=3500.0,
        deadline_hours=260.0,
        budget=80_000_000.0,
        carbon_price=80.0,
        scenario_id="SCEN-TEST-DECISION",
        max_transition_rate=0.40,
        target_reliability=90.0,
        vessel_class="PANAMAX",
    )


# =============================================================================
# 1. LIFECYCLE ASSESSMENT (LCA) TESTS
# =============================================================================

def test_lca_engine_default_profiles() -> None:
    """Verify default profiles cover all 5 fuels and distinct feedstock pathways."""
    engine = MaritimeLifecycleAssessmentEngine()
    
    # Check all fuels exist
    for fuel in ("Diesel", "LNG", "Methanol", "Hydrogen", "Ammonia"):
        profile = engine.get_profile(fuel)
        assert isinstance(profile, FuelLifecycleProfile)
        assert profile.energy_density_mj_per_ton > 0.0
        assert profile.cost_per_ton_usd > 0.0

    # Check pathway distinctions for Hydrogen
    grey_h2 = engine.get_profile("Hydrogen", "grey")
    blue_h2 = engine.get_profile("Hydrogen", "blue")
    green_h2 = engine.get_profile("Hydrogen", "green")
    
    assert grey_h2.production_emission_factor > blue_h2.production_emission_factor
    assert blue_h2.production_emission_factor > green_h2.production_emission_factor
    assert green_h2.renewable_fraction == 1.0


def test_lca_engine_calculations() -> None:
    """Verify Well-to-Wake (WTW) equals Well-to-Tank (WTT) plus Tank-to-Wake (TTW)."""
    engine = MaritimeLifecycleAssessmentEngine()
    
    # 1000 tons of Fossil LNG
    res = engine.assess_fuel_lifecycle(
        fuel_name="LNG",
        consumption_tons=1000.0,
        pathway="fossil",
        carbon_price_usd_per_ton=80.0,
    )
    
    assert isinstance(res, LifecycleAssessmentResult)
    assert res.fuel_consumption_tons == 1000.0
    assert res.well_to_tank_emissions == pytest.approx(
        res.fuel_production_emissions + res.fuel_transport_emissions + res.fuel_storage_emissions, abs=1e-2
    )
    assert res.well_to_wake_emissions == pytest.approx(
        res.well_to_tank_emissions + res.tank_to_wake_emissions, abs=1e-2
    )
    assert res.energy_content_mj > 0.0
    assert res.emission_intensity_g_per_mj > 0.0
    assert res.lifecycle_cost > 0.0


def test_lca_fleet_assessment() -> None:
    """Verify fleet-wide multi-fuel assessment."""
    engine = MaritimeLifecycleAssessmentEngine()
    fuel_split = {"Diesel": 800.0, "Methanol": 400.0}
    pathways = {"Diesel": "fossil", "Methanol": "e_methanol"}
    
    fleet_res = engine.assess_fleet_lifecycle(fuel_consumption=fuel_split, pathways=pathways)
    assert len(fleet_res) == 2
    assert "Diesel" in fleet_res
    assert "Methanol" in fleet_res
    assert fleet_res["Methanol"].pathway == "e_methanol"


# =============================================================================
# 2. REGULATORY FORECAST ENGINE TESTS
# =============================================================================

def test_regulatory_forecast_statutory_vs_scenario_cii() -> None:
    """Verify strict separation of 2026-2030 statutory CII from 2031-2040 scenario extrapolation."""
    engine = RegulatoryForecastEngine()
    
    # 2026-2030: Must be STATUTORY under MEPC.400(83)
    z_2026, ev_2026 = engine.resolve_cii_reduction_factor(2026)
    assert z_2026 == 0.11000
    assert ev_2026 == EvidenceCategory.STATUTORY
    
    z_2030, ev_2030 = engine.resolve_cii_reduction_factor(2030)
    assert z_2030 == 0.21500
    assert ev_2030 == EvidenceCategory.STATUTORY
    
    # 2035: Must be SCENARIO projection
    z_2035, ev_2035 = engine.resolve_cii_reduction_factor(2035)
    assert z_2035 > z_2030
    assert ev_2035 == EvidenceCategory.SCENARIO


def test_regulatory_forecast_fueleu_milestones() -> None:
    """Verify statutory FuelEU Maritime milestones and modelled interpolations."""
    engine = RegulatoryForecastEngine()
    
    # Exact statutory milestone years
    r_2025, ev_2025 = engine.resolve_fueleu_target_reduction(2025)
    assert r_2025 == 0.020  # -2%
    assert ev_2025 == EvidenceCategory.STATUTORY
    
    r_2030, ev_2030 = engine.resolve_fueleu_target_reduction(2030)
    assert r_2030 == 0.060  # -6%
    assert ev_2030 == EvidenceCategory.STATUTORY
    
    r_2040, ev_2040 = engine.resolve_fueleu_target_reduction(2040)
    assert r_2040 == 0.310  # -31%
    assert ev_2040 == EvidenceCategory.STATUTORY
    
    # Intermediate year: MODELLED interpolation
    r_2028, ev_2028 = engine.resolve_fueleu_target_reduction(2028)
    assert 0.020 < r_2028 < 0.060
    assert ev_2028 == EvidenceCategory.MODELLED


def test_regulatory_forecast_estimated_compliance_probability() -> None:
    """Verify estimated compliance probability under operational uncertainty with metadata."""
    engine = RegulatoryForecastEngine()
    
    res = engine.forecast_compliance_trajectory(
        vessel_type="Bulk carrier",
        capacity_dwt=45000.0,
        annual_fuel_consumption_tons=3000.0,
        annual_distance_nm=70000.0,
        fuel_shares={"Diesel": 0.8, "LNG": 0.2},
        start_year=2026,
        end_year=2030,
        monte_carlo_trials=100,
        random_seed=42,
    )
    
    assert isinstance(res, RegulatoryForecastResult)
    assert len(res.planning_horizon) == 5
    for yr in res.planning_horizon:
        prob = res.estimated_compliance_probability[yr]
        assert 0.0 <= prob <= 1.0
        assert yr in res.future_cii_ratings
        assert yr in res.future_fueleu_status
        assert yr in res.projected_penalties_usd
        
    # Check uncertainty metadata
    meta = res.metadata
    assert meta["monte_carlo_trials"] == 100
    assert meta["random_seed"] == 42
    assert "uncertainty_sources" in meta
    assert len(meta["uncertainty_sources"]) >= 3


# =============================================================================
# 3. FUEL TRANSITION PLANNER TESTS
# =============================================================================

def test_fuel_transition_planner(base_scenario: OptimizationScenario) -> None:
    """Verify multi-period transition planning and milestone accumulation."""
    planner = FuelTransitionPlanner()
    roadmap = planner.plan_transition(
        scenario=base_scenario,
        target_years=(2026, 2028, 2030, 2035, 2040),
        vessel_class="PANAMAX",
        primary_green_fuel="Methanol",
        secondary_green_fuel="Hydrogen",
    )
    
    assert isinstance(roadmap, TransitionRoadmap)
    assert len(roadmap.milestones) == 5
    assert roadmap.total_transition_capex >= 0.0
    assert roadmap.total_transition_opex > 0.0
    assert roadmap.total_transition_cost > 0.0
    assert roadmap.cumulative_emissions_tons > 0.0
    assert 0.0 <= roadmap.regulatory_risk_score <= 100.0
    
    # Final milestone should have reduced emissions relative to first milestone
    first_emiss = roadmap.milestones[0].annual_emissions_tons
    last_emiss = roadmap.milestones[-1].annual_emissions_tons
    assert last_emiss <= first_emiss


# =============================================================================
# 4. EXECUTIVE RECOMMENDATION ENGINE TESTS
# =============================================================================

def test_executive_recommendation_engine(base_scenario: OptimizationScenario) -> None:
    """Verify ROI, payback period logic, priority actions, and evidence items."""
    opt = FleetStrategyOptimizer()
    strat_rec = opt.optimize_strategy(base_scenario)
    
    exec_engine = ExecutiveRecommendationEngine()
    rec = exec_engine.generate_recommendation(
        strategy_recommendation=strat_rec,
        scenario=base_scenario,
        investment_horizon_years=10,
    )
    
    assert isinstance(rec, ExecutiveRecommendation)
    assert rec.total_investment_capex >= 0.0
    assert isinstance(rec.expected_cost_savings_usd, float)
    assert isinstance(rec.expected_emissions_reduction_tons, float)
    
    # Payback logic test: if benefit <= 0, payback_years must be None and status NO_PAYBACK
    if rec.annual_net_benefit_usd <= 0.0:
        assert rec.payback_years is None
        assert rec.payback_status == "NO_PAYBACK"
    else:
        if rec.total_investment_capex > 0.0:
            assert rec.payback_years > 0.0
            assert rec.payback_status in ("OPTIMAL", "ACCEPTABLE", "EXTENDED")
            
    assert len(rec.priority_actions) >= 3
    assert len(rec.risk_and_mitigations) >= 3
    assert len(rec.evidence_items) >= 4
    assert len(rec.executive_summary_text) > 50


# =============================================================================
# 5. MULTI-SCENARIO SENSITIVITY ANALYZER TESTS
# =============================================================================

def test_multi_scenario_analyzer(base_scenario: OptimizationScenario) -> None:
    """Verify evaluation across all 6 parameterized ScenarioDefinition objects."""
    analyzer = MultiScenarioAnalyzer()
    res = analyzer.compare_scenarios(base_scenario=base_scenario)
    
    assert isinstance(res, ScenarioComparisonResult)
    assert len(res.scenario_names) == 6
    assert len(res.metrics_comparison) == 6
    assert len(res.ranking) == 6
    assert res.best_scenario in res.scenario_names
    assert res.worst_scenario in res.scenario_names
    assert len(res.sensitivity_analysis) == 5  # 5 non-baseline scenarios

    # Verify scenario shocks are propagated
    fps_metrics = res.metrics_comparison["Fuel Price Shock"]
    assert fps_metrics["fossil_fuel_multiplier"] == 1.50
    assert fps_metrics["alt_fuel_multiplier"] == 1.10
    assert fps_metrics["diesel_price_usd"] == 975.0  # 650.0 * 1.50

    strict_metrics = res.metrics_comparison["Strict Regulation"]
    assert strict_metrics["regulation_factor"] == 1.25

    # Fuel Price Shock must result in higher operational cost than baseline
    base_cost = res.metrics_comparison["Baseline Scenario"]["operational_cost_usd"]
    fps_cost = fps_metrics["operational_cost_usd"]
    assert fps_cost > base_cost


# =============================================================================
# 6. INDUSTRIAL CASE STUDIES SUITE TESTS
# =============================================================================

def test_industrial_case_studies(tmp_path: Path) -> None:
    """Verify end-to-end execution of Cases A, B, C, D and report exports."""
    suite = IndustrialCaseStudySuite()
    results = suite.run_all_case_studies(output_dir=tmp_path)
    
    assert len(results) == 4
    for case_key in (
        "Case A: Conventional Diesel Fleet",
        "Case B: LNG Transition Fleet",
        "Case C: Methanol Transition Fleet",
        "Case D: Hydrogen Future Fleet",
    ):
        assert case_key in results
        c_res = results[case_key]
        assert "annual_cost_usd" in c_res
        assert "lifecycle_emissions" in c_res
        assert "reliability_score" in c_res
        assert "priority_actions" in c_res
        
    # Check exported files
    assert (tmp_path / "industrial_case_studies.json").exists()
    assert (tmp_path / "case_studies_summary.csv").exists()


# =============================================================================
# 7. EXECUTIVE REPORT GENERATOR TESTS
# =============================================================================

def test_executive_report_generator(base_scenario: OptimizationScenario, tmp_path: Path) -> None:
    """Verify generation of PDF, Markdown, and JSON reports."""
    opt = FleetStrategyOptimizer()
    strat_rec = opt.optimize_strategy(base_scenario)
    exec_engine = ExecutiveRecommendationEngine()
    rec = exec_engine.generate_recommendation(strategy_recommendation=strat_rec, scenario=base_scenario)
    
    gen = ExecutiveReportGenerator()
    reports = gen.generate_all_reports(
        recommendation=rec,
        scenario=base_scenario,
        strategy_rec=strat_rec,
        output_dir=tmp_path,
    )
    
    assert reports["pdf"].exists()
    assert reports["markdown"].exists()
    assert reports["json"].exists()
    assert reports["pdf"].stat().st_size > 1000
    assert reports["markdown"].stat().st_size > 500
    assert reports["json"].stat().st_size > 500


# =============================================================================
# 8. STREAMLIT PAGE 9 DASHBOARD INTEGRATION TEST
# =============================================================================

def test_page_decision_intelligence_render(tmp_path: Path) -> None:
    """Verify that Page 9 renders and handles executive report generation without API mismatches."""
    from unittest.mock import MagicMock, patch
    from app.dashboard.page_decision_intelligence import render_decision_intelligence_page

    with patch("streamlit.title"), \
         patch("streamlit.markdown"), \
         patch("streamlit.expander") as mock_exp, \
         patch("streamlit.columns", side_effect=lambda n: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))]), \
         patch("streamlit.number_input", side_effect=[250000.0, 3500.0, 260.0]), \
         patch("streamlit.slider", side_effect=[80.0, 10]), \
         patch("streamlit.selectbox", side_effect=["PANAMAX", "Methanol", "e_methanol", "Hydrogen"]), \
         patch("streamlit.subheader"), \
         patch("streamlit.metric"), \
         patch("streamlit.info"), \
         patch("streamlit.plotly_chart"), \
         patch("streamlit.dataframe"), \
         patch("streamlit.caption"), \
         patch("streamlit.success"), \
         patch("streamlit.button", return_value=True):
        mock_exp.return_value.__enter__.return_value = MagicMock()
        render_decision_intelligence_page()


# =============================================================================
# 9. GRANULAR LCA PATHWAYS & REGULATORY DIVERGENCE TESTS (ISSUE 3)
# =============================================================================

def test_regulatory_forecast_granular_lca_pathway_divergence() -> None:
    """Verify that Grey vs Green Hydrogen and Fossil vs E-Methanol diverge in FuelEU compliance."""
    engine = RegulatoryForecastEngine()

    # 1. Hydrogen: Grey (SMR) vs Green (Electrolysis) in 2030 (target = 85.69 gCO2eq/MJ)
    forecast_grey_h2 = engine.forecast_compliance_trajectory(
        vessel_type="Bulk carrier",
        capacity_dwt=45000.0,
        annual_fuel_consumption_tons=2000.0,
        annual_distance_nm=50000.0,
        fuel_shares={"Hydrogen": 1.0},
        start_year=2030,
        end_year=2030,
        fuel_pathways={"Hydrogen": "grey"},
    )
    forecast_green_h2 = engine.forecast_compliance_trajectory(
        vessel_type="Bulk carrier",
        capacity_dwt=45000.0,
        annual_fuel_consumption_tons=2000.0,
        annual_distance_nm=50000.0,
        fuel_shares={"Hydrogen": 1.0},
        start_year=2030,
        end_year=2030,
        fuel_pathways={"Hydrogen": "green"},
    )

    # Grey Hydrogen has ~87.5 g/MJ > 85.69 g/MJ -> DEFICIT with financial penalty
    assert forecast_grey_h2.metadata["fuel_pathways"]["Hydrogen"] == "grey"
    assert forecast_grey_h2.metadata["fuel_ghg_intensities_g_per_mj"]["Hydrogen"] == pytest.approx(87.5, abs=0.5)
    assert forecast_grey_h2.future_fueleu_status[2030] == "DEFICIT"
    assert forecast_grey_h2.projected_penalties_eur[2030] > 100_000.0

    # Green Hydrogen has ~2.92 g/MJ << 85.69 g/MJ -> COMPLIANT with zero penalty
    assert forecast_green_h2.metadata["fuel_pathways"]["Hydrogen"] == "green"
    assert forecast_green_h2.metadata["fuel_ghg_intensities_g_per_mj"]["Hydrogen"] == pytest.approx(2.92, abs=0.5)
    assert forecast_green_h2.future_fueleu_status[2030] == "COMPLIANT"
    assert forecast_green_h2.projected_penalties_eur[2030] == 0.0

    # 2. Methanol: Fossil (Natural Gas) vs E-Methanol (DAC+H2) in 2030
    forecast_fossil_meth = engine.forecast_compliance_trajectory(
        vessel_type="Bulk carrier",
        capacity_dwt=45000.0,
        annual_fuel_consumption_tons=3000.0,
        annual_distance_nm=50000.0,
        fuel_shares={"Methanol": 1.0},
        start_year=2030,
        end_year=2030,
        fuel_pathways={"Methanol": "fossil"},
    )
    forecast_e_meth = engine.forecast_compliance_trajectory(
        vessel_type="Bulk carrier",
        capacity_dwt=45000.0,
        annual_fuel_consumption_tons=3000.0,
        annual_distance_nm=50000.0,
        fuel_shares={"Methanol": 1.0},
        start_year=2030,
        end_year=2030,
        fuel_pathways={"Methanol": "e_methanol"},
    )

    # Fossil Methanol (89.2 g/MJ) is in DEFICIT in 2030
    assert forecast_fossil_meth.future_fueleu_status[2030] == "DEFICIT"
    assert forecast_fossil_meth.projected_penalties_eur[2030] > 0.0

    # E-Methanol (~5.03 g/MJ) is COMPLIANT in 2030
    assert forecast_e_meth.future_fueleu_status[2030] == "COMPLIANT"
    assert forecast_e_meth.projected_penalties_eur[2030] == 0.0


def test_regulatory_forecast_profile_and_intensity_overrides() -> None:
    """Verify direct injection of custom FuelLifecycleProfile, custom intensities, and direct overrides."""
    engine = RegulatoryForecastEngine()

    # 1. Custom GHG intensity map override
    res_custom = engine.forecast_compliance_trajectory(
        vessel_type="Bulk carrier",
        capacity_dwt=45000.0,
        annual_fuel_consumption_tons=2500.0,
        annual_distance_nm=60000.0,
        fuel_shares={"Methanol": 0.5, "Diesel": 0.5},
        start_year=2026,
        end_year=2028,
        custom_ghg_intensities={"Methanol": 15.0, "Diesel": 89.0},
    )
    assert res_custom.metadata["fuel_ghg_intensities_g_per_mj"]["Methanol"] == 15.0
    assert res_custom.metadata["fuel_ghg_intensities_g_per_mj"]["Diesel"] == 89.0

    # 2. Custom FuelLifecycleProfile injection
    custom_profile = FuelLifecycleProfile(
        fuel_name="Ammonia",
        production_pathway="certified_solar",
        production_emission_factor=0.04,
        transport_emission_factor=0.06,
        storage_emission_factor=0.0,
        tank_to_wake_factor=0.0,
        energy_density_mj_per_ton=18600.0,
        renewable_fraction=1.0,
        cost_per_ton_usd=1200.0,
    )
    res_prof = engine.forecast_compliance_trajectory(
        vessel_type="Bulk carrier",
        capacity_dwt=45000.0,
        annual_fuel_consumption_tons=2000.0,
        annual_distance_nm=50000.0,
        fuel_shares={"Ammonia": 1.0},
        start_year=2030,
        end_year=2030,
        lifecycle_profiles=[custom_profile],
    )
    assert res_prof.metadata["fuel_pathways"]["Ammonia"] == "certified_solar"
    assert res_prof.future_fueleu_status[2030] == "COMPLIANT"

    # 3. Direct fleet-wide override
    res_direct = engine.forecast_compliance_trajectory(
        vessel_type="Bulk carrier",
        capacity_dwt=45000.0,
        annual_fuel_consumption_tons=2000.0,
        annual_distance_nm=50000.0,
        fuel_shares={"Diesel": 1.0},
        start_year=2030,
        end_year=2030,
        wtw_ghg_intensity=20.0,
        annual_energy_mj=80_000_000.0,
    )
    assert res_direct.metadata["effective_ghg_intensity_g_per_mj"] == 20.0
    assert res_direct.future_fueleu_status[2030] == "COMPLIANT"


# =============================================================================
# 10. OPERATIONAL ENERGY BALANCE & LHV RECONCILIATION TESTS (ISSUE 4)
# =============================================================================

def test_regulatory_forecast_operational_energy_balance() -> None:
    """Verify operational energy balance E = sum(m_i * LHV_i) using fuel-specific LHV.

    Proves that 41,000 MJ/t is not used as the operational fuel energy input, but is strictly
    reserved for the statutory Article 23 / Annex IV penalty conversion formula.
    """
    engine = RegulatoryForecastEngine()

    # 1. Verification of standalone calculate_operational_energy_balance for individual fuels
    fuels_and_lhv = {
        "Hydrogen": 120000.0,
        "Methanol": 19900.0,
        "LNG": 49100.0,
        "Ammonia": 18600.0,
        "Diesel": 42700.0,
    }
    for fuel, expected_lhv in fuels_and_lhv.items():
        tot_e, weighted_lhv, e_map, m_map, profiles = engine.calculate_operational_energy_balance(
            fuel_consumption={fuel: 1000.0}
        )
        assert m_map[fuel] == 1000.0
        assert weighted_lhv == pytest.approx(expected_lhv, abs=1e-2)
        assert tot_e == pytest.approx(1000.0 * expected_lhv, abs=1e-2)
        assert e_map[fuel] == pytest.approx(1000.0 * expected_lhv, abs=1e-2)
        # Ensure it is NOT the default 41,000 MJ/t (unless fuel happened to be exactly 41,000)
        assert tot_e != 41_000_000.0

    # 2. Heterogeneous multi-fuel mix: 600t Diesel + 400t LNG
    # E = 600 * 42,700 + 400 * 49,100 = 25,620,000 + 19,640,000 = 45,260,000 MJ
    tot_e, weighted_lhv, e_map, m_map, _ = engine.calculate_operational_energy_balance(
        fuel_consumption={"Diesel": 600.0, "LNG": 400.0}
    )
    expected_energy = (600.0 * 42700.0) + (400.0 * 49100.0)
    assert tot_e == pytest.approx(expected_energy, abs=1e-2)
    assert weighted_lhv == pytest.approx(expected_energy / 1000.0, abs=1e-2)
    assert e_map["Diesel"] == pytest.approx(600.0 * 42700.0, abs=1e-2)
    assert e_map["LNG"] == pytest.approx(400.0 * 49100.0, abs=1e-2)

    # 3. Forecast compliance trajectory using direct fuel_consumption mapping
    forecast = engine.forecast_compliance_trajectory(
        vessel_type="Bulk carrier",
        capacity_dwt=45000.0,
        annual_distance_nm=50000.0,
        fuel_consumption={"Diesel": 600.0, "LNG": 400.0},
        start_year=2030,
        end_year=2030,
    )
    meta = forecast.metadata
    assert meta["operational_energy_mj"] == pytest.approx(expected_energy, abs=1.0)
    assert meta["energy_by_fuel_mj"]["Diesel"] == pytest.approx(600.0 * 42700.0, abs=1.0)
    assert meta["energy_by_fuel_mj"]["LNG"] == pytest.approx(400.0 * 49100.0, abs=1.0)
    assert meta["fuel_mass_tons"]["Diesel"] == 600.0
    assert meta["fuel_mass_tons"]["LNG"] == 400.0
    assert meta["fuel_lhv_mj_per_ton"]["Diesel"] == 42700.0
    assert meta["fuel_lhv_mj_per_ton"]["LNG"] == 49100.0
    assert meta["statutory_penalty_vlsfo_equivalent_factor_mj_per_ton"] == 41000.0

    # 4. Strict verification of decoupling: Operational energy vs Statutory penalty conversion
    # For 1,000 tons of Fossil Methanol in 2040:
    # Target intensity (2040: -31%) = 91.16 * 0.69 = 62.9004 g/MJ
    # Actual intensity (fossil methanol) = 89.196 g/MJ
    # Deficit intensity = 89.196 - 62.9004 = 26.2956 g/MJ
    # Operational energy supplied: E = 1,000 * 19,900 = 19,900,000 MJ (NOT 41,000,000 MJ!)
    # Compliance balance = -26.2956 * 19,900,000 = -523,282,440 g
    # Statutory penalty (Article 23): (|balance| / (actual * 41,000)) * 2,400
    # = (523,282,440 / (89.196 * 41,000)) * 2,400
    # = (523,282,440 / 3,657,036) * 2,400 = 143.089 * 2,400 = 343,414 EUR
    forecast_meth = engine.forecast_compliance_trajectory(
        vessel_type="Bulk carrier",
        capacity_dwt=45000.0,
        annual_distance_nm=50000.0,
        fuel_consumption={"Methanol": 1000.0},
        fuel_pathways={"Methanol": "fossil"},
        start_year=2040,
        end_year=2040,
    )
    # Check operational energy is exactly 1,000 t * 19,900 MJ/t = 19.9M MJ
    assert forecast_meth.metadata["operational_energy_mj"] == 19_900_000.0
    # Check penalty calculation reflects true operational energy combined with 41,000 penalty denominator
    expected_pen_eur = (523_282_440 / (89.196 * 41000.0)) * 2400.0
    assert forecast_meth.projected_penalties_eur[2040] == pytest.approx(expected_pen_eur, rel=1e-2)



