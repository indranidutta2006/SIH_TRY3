"""Page 9: Maritime Decision Intelligence & Strategic Decarbonization.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 4 Deliverable: Transforms optimization outputs into an executive decision-support system:
1. Executive Recommendation & Modernization ROI/Payback Card.
2. Multi-Year Fuel Transition Roadmap (2026–2040).
3. Well-to-Wake Lifecycle Assessment (LCA) Breakdown (WTT, TTW, Feedstocks).
4. Statutory & Scenario-Projected Regulatory Risk (IMO CII & EU FuelEU).
5. Multi-Scenario Stress Test Sensitivity Matrix.
6. Evidence vs. Assumption Classification Layer.
7. Executive PDF / Markdown / JSON Report Downloads.
"""

from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from contracts.schemas import EvidenceCategory, OptimizationScenario
from src.decision_support.executive_recommendation_engine import ExecutiveRecommendationEngine
from src.lifecycle.lifecycle_assessment_engine import MaritimeLifecycleAssessmentEngine
from src.optimization.fleet_strategy_optimizer import FleetStrategyOptimizer
from src.reporting.executive_report_generator import ExecutiveReportGenerator
from src.scenarios.scenario_comparison import MultiScenarioAnalyzer
from src.strategy.fuel_transition_planner import FuelTransitionPlanner
from src.strategy.regulatory_forecast import RegulatoryForecastEngine


def render_decision_intelligence_page() -> None:
    """Render the executive decision intelligence and strategic decarbonization interface."""
    st.title("🧭 Maritime Decision Intelligence & Strategy")
    st.markdown(
        """
        **Executive Decision-Support Platform:** Synthesizes hydrodynamic fuel predictions, 
        quantum-inspired optimization, and lifecycle emissions into actionable fleet modernization roadmaps, 
        investment ROI economics, and regulatory risk mitigation plans.
        """
    )

    optimizer = FleetStrategyOptimizer()
    lca_engine = MaritimeLifecycleAssessmentEngine()
    reg_engine = RegulatoryForecastEngine()
    transition_planner = FuelTransitionPlanner(lca_engine=lca_engine, regulatory_engine=reg_engine)
    exec_engine = ExecutiveRecommendationEngine()
    scenario_analyzer = MultiScenarioAnalyzer(optimizer=optimizer)
    report_gen = ExecutiveReportGenerator()

    # 1. Operational Parameters Configuration
    with st.expander("⚙️ Strategic Planning & Corridor Context", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            cargo_demand = st.number_input(
                "Annual Demand (tons)",
                min_value=50_000.0,
                max_value=1_000_000.0,
                value=250_000.0,
                step=25_000.0,
            )
            route_distance = st.number_input(
                "Corridor Distance (nm)",
                min_value=500.0,
                max_value=12_000.0,
                value=3500.0,
                step=250.0,
            )
        with c2:
            deadline_hours = st.number_input(
                "Delivery Window (hours)",
                min_value=100.0,
                max_value=600.0,
                value=260.0,
                step=10.0,
            )
            carbon_price = st.slider(
                "ETS Carbon Price ($/ton)",
                min_value=20.0,
                max_value=200.0,
                value=80.0,
                step=5.0,
            )
        with c3:
            vessel_class = st.selectbox("Vessel Class", ["PANAMAX", "HANDYMAX", "CAPESIZE"], index=0)
            vessel_type = st.selectbox(
                "IMO Vessel Type",
                ["Bulk carrier", "Containership", "Tanker", "General cargo ship", "Gas carrier", "LNG carrier"],
                index=0,
                help="IMO Resolution MEPC.353(78) vessel category determining statutory CII reference lines and rating boundaries."
            )
            primary_green_fuel = st.selectbox("Primary Transition Fuel", ["Methanol", "LNG", "Hydrogen", "Ammonia"], index=0)
            pathway_options = {
                "Methanol": ["e_methanol", "bio_methanol", "fossil"],
                "LNG": ["bio_lng", "fossil"],
                "Hydrogen": ["green", "blue", "grey"],
                "Ammonia": ["green", "blue", "grey"],
            }
            primary_pathway = st.selectbox(
                "Primary Feedstock Pathway",
                pathway_options.get(primary_green_fuel, ["green", "fossil"]),
                index=0,
                help="Granular LCA production pathway affecting FuelEU GHG intensity and compliance penalties."
            )
        with c4:
            secondary_green_fuel = st.selectbox("Secondary Fuel (Post-2032)", ["Hydrogen", "Ammonia", "Methanol"], index=0)
            secondary_pathway = st.selectbox(
                "Secondary Feedstock Pathway",
                pathway_options.get(secondary_green_fuel, ["green", "fossil"]),
                index=0,
                help="Granular LCA production pathway for secondary fuel post-2032."
            )
            horizon_years = st.slider("Investment Horizon (years)", 5, 20, 10)

    scenario = OptimizationScenario(
        cargo_demand=cargo_demand,
        route_distance=route_distance,
        deadline_hours=deadline_hours,
        budget=120_000_000.0,
        carbon_price=carbon_price,
        vessel_class=vessel_class,
        vessel_type=vessel_type,
        scenario_id="SCEN-DECISION-INTEL",
        max_transition_rate=0.50,
        target_reliability=90.0,
    )

    # 2. Run Engines
    strat_rec = optimizer.optimize_strategy(scenario)
    opt_scenario = strat_rec.scenario  # Parameters derived from naval-architecture capacity optimizer
    fleet_mix_res = strat_rec.fleet_mix
    total_vessels = strat_rec.summary.get("total_vessels", sum(v for k, v in fleet_mix_res.fleet_mix.items() if k in {"feeder", "medium", "large"}))
    if total_vessels <= 0:
        total_vessels = max(1, sum(fleet_mix_res.fleet_mix.values()))

    st.caption(
        f"🧭 **Strategy Derived Parameters:** Class: `{opt_scenario.vessel_class}` | "
        f"IMO Category: `{opt_scenario.vessel_type}` | "
        f"Capacity: `{opt_scenario.capacity_dwt:,.0f} DWT` | "
        f"Operational Frequency: `{opt_scenario.annual_voyages} voyages/year` | "
        f"Annual Operating Distance: `{opt_scenario.annual_distance:,.0f} nm`"
    )

    # Extract normalized fuel shares from fleet_mix
    fuel_tokens = ["diesel", "lng", "methanol", "hydrogen", "ammonia"]
    fuel_counts = {k: fleet_mix_res.fleet_mix.get(k, 0) for k in fuel_tokens if fleet_mix_res.fleet_mix.get(k, 0) > 0}
    if not fuel_counts:
        fuel_counts = {"diesel": total_vessels}
    total_fuel_units = max(1, sum(fuel_counts.values()))
    fuel_shares = {k: v / total_fuel_units for k, v in fuel_counts.items()}

    fuel_pathways = {
        "Diesel": "fossil",
        "diesel": "fossil",
        primary_green_fuel: primary_pathway,
        primary_green_fuel.lower(): primary_pathway,
        primary_green_fuel.capitalize(): primary_pathway,
        secondary_green_fuel: secondary_pathway,
        secondary_green_fuel.lower(): secondary_pathway,
        secondary_green_fuel.capitalize(): secondary_pathway,
    }

    roadmap = transition_planner.plan_transition(
        scenario=opt_scenario,
        vessel_class=vessel_class,
        primary_green_fuel=primary_green_fuel,
        secondary_green_fuel=secondary_green_fuel,
        fuel_pathways=fuel_pathways,
    )
    forecast = reg_engine.forecast_compliance_trajectory(
        vessel_type=opt_scenario.vessel_type,
        capacity_dwt=opt_scenario.capacity_dwt,
        annual_fuel_consumption_tons=fleet_mix_res.fuel_consumption / max(total_vessels, 1),
        annual_distance_nm=opt_scenario.annual_distance,
        fuel_shares=fuel_shares,
        start_year=2026,
        end_year=2040,
        fuel_pathways=fuel_pathways,
    )
    rec = exec_engine.generate_recommendation(
        strategy_recommendation=strat_rec,
        scenario=opt_scenario,
        roadmap=roadmap,
        forecast=forecast,
        investment_horizon_years=horizon_years,
    )

    # 3. Executive Decision & Financial ROI KPIs
    st.subheader("💡 Executive Recommendation & Capital ROI Summary")

    # Evidence badge helpers
    def _ev_badge(evidence: str) -> str:
        return "✅ MODELLED" if evidence == "MODELLED" else ("📜 STATUTORY" if evidence == "STATUTORY" else "⚠️ ASSUMED")

    def _fmt_millions(val: float) -> str:
        if val < 0:
            return f"-${abs(val) / 1e6:.2f}M"
        return f"+${val / 1e6:.2f}M"

    def _fmt_net(val: float) -> str:
        if val < 0:
            return f"-${abs(val) / 1e6:.2f}M"
        return f"${val / 1e6:.2f}M"

    bd = rec.economics_breakdown  # evidence-tagged breakdown dict

    k1, k2, k3, k4 = st.columns(4)
    capex_ev = bd.get("total_retrofit_capex_evidence", "ASSUMED")
    k1.metric(
        f"Modernization Capex [{_ev_badge(capex_ev)}]",
        f"${rec.total_investment_capex / 1e6:.1f}M",
        f"{total_vessels} Vessels",
        help=bd.get("total_retrofit_capex_source", ""),
    )
    k2.metric(
        f"{horizon_years}-Year ROI",
        f"{rec.roi_percentage}%" if rec.roi_percentage is not None else "N/A",
        f"Net Annual: {_fmt_net(rec.annual_net_benefit_usd)}",
    )
    k3.metric(
        "Simple Payback",
        f"{rec.payback_years} yrs" if rec.payback_years is not None else rec.payback_status,
        f"Status: {rec.payback_status}",
    )
    savings_ev = "MODELLED"  # fuel + carbon are always modelled; penalty may be assumed
    k4.metric(
        f"Annual Net Savings [✅ MODELLED]",
        _fmt_net(rec.expected_cost_savings_usd),
        f"{rec.expected_cost_savings_pct:+.1f}% vs Diesel Baseline",
    )

    st.info(rec.executive_summary_text)

    # Economics waterfall chart — decomposed ROI components with evidence colouring
    st.markdown("#### 📊 Annual Benefit Decomposition (Evidence-Classified Waterfall)")
    fuel_sav = bd.get("annual_fuel_savings_usd", 0.0)
    carbon_sav = bd.get("annual_carbon_savings_usd", 0.0)
    penalty_av = bd.get("annual_penalty_avoidance_usd", 0.0)
    opex_delta = bd.get("annual_opex_delta_usd", 0.0)
    net_benefit = bd.get("annual_net_benefit_usd", rec.annual_net_benefit_usd)
    penalty_ev = bd.get("annual_penalty_avoidance_evidence", "ASSUMED")

    COLOUR_MODELLED = "#059669"   # green
    COLOUR_ASSUMED  = "#D97706"   # amber
    COLOUR_NET      = "#1D4ED8"   # blue
    COLOUR_NEGATIVE = "#DC2626"   # red

    fig_wf = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative", "relative", "relative", "relative", "total"],
        x=[
            f"⛽ Fuel Savings\n[✅ MODELLED]",
            f"🌿 Carbon Savings\n[✅ MODELLED]",
            f"📋 FuelEU Penalty\nAvoided [{_ev_badge(penalty_ev)}]",
            f"⚙️ Additional OPEX\n[⚠️ ASSUMED]",
            f"💰 Net Annual\nBenefit",
        ],
        y=[fuel_sav, carbon_sav, penalty_av, -opex_delta, 0],
        connector={"line": {"color": "rgb(100,100,100)", "width": 1}},
        increasing={"marker": {"color": COLOUR_MODELLED}},
        decreasing={"marker": {"color": COLOUR_NEGATIVE}},
        totals={"marker": {"color": COLOUR_NET}},
        text=[
            _fmt_millions(fuel_sav),
            _fmt_millions(carbon_sav),
            _fmt_millions(penalty_av),
            f"-${opex_delta / 1e6:.2f}M",
            _fmt_net(net_benefit),
        ],
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>Amount: $%{y:,.0f}<extra></extra>"
        ),
    ))
    fig_wf.update_layout(
        title=f"Annual Economic Benefit Stack — Investment Horizon {horizon_years} Years",
        yaxis_title="USD (Annual)",
        yaxis_tickformat="$,.0f",
        showlegend=False,
        height=420,
        margin=dict(t=60, b=20),
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    # Itemised breakdown expander
    with st.expander("🔍 Full Itemised Economics Breakdown (Evidence Tags)", expanded=False):
        breakdown_rows = [
            {
                "Component": "Annual Fuel Savings / (Impact)",
                "Amount (USD/yr)": _fmt_millions(bd.get("annual_fuel_savings_usd", 0.0)),
                "Evidence": _ev_badge(bd.get("annual_fuel_savings_evidence", "MODELLED")),
                "Source Note": bd.get("annual_fuel_savings_note", "—"),
            },
            {
                "Component": "Annual Carbon Savings / (Impact)",
                "Amount (USD/yr)": _fmt_millions(bd.get("annual_carbon_savings_usd", 0.0)),
                "Evidence": _ev_badge(bd.get("annual_carbon_savings_evidence", "MODELLED")),
                "Source Note": bd.get("annual_carbon_savings_note", "—"),
            },
            {
                "Component": "FuelEU Penalty Avoided",
                "Amount (USD/yr)": _fmt_millions(bd.get("annual_penalty_avoidance_usd", 0.0)),
                "Evidence": _ev_badge(bd.get("annual_penalty_avoidance_evidence", "ASSUMED")),
                "Source Note": bd.get("annual_penalty_avoidance_note", "—"),
            },
            {
                "Component": "Additional OPEX Delta",
                "Amount (USD/yr)": f"-${bd.get('annual_opex_delta_usd', 0) / 1e6:.3f}M",
                "Evidence": _ev_badge(bd.get("annual_opex_delta_evidence", "ASSUMED")),
                "Source Note": bd.get("annual_opex_delta_note", "—"),
            },
            {
                "Component": "= Net Annual Benefit",
                "Amount (USD/yr)": _fmt_net(bd.get("annual_net_benefit_usd", rec.annual_net_benefit_usd)),
                "Evidence": "—",
                "Source Note": "Fuel + Carbon + Penalty − OPEX Delta",
            },
            {
                "Component": "Total Retrofit CAPEX (once)",
                "Amount (USD/yr)": f"${bd.get('total_retrofit_capex_usd', rec.total_investment_capex) / 1e6:.1f}M",
                "Evidence": _ev_badge(bd.get("total_retrofit_capex_evidence", "ASSUMED")),
                "Source Note": bd.get("total_retrofit_capex_source", "—"),
            },
        ]
        st.dataframe(pd.DataFrame(breakdown_rows), use_container_width=True, hide_index=True)

        # FuelEU penalty avoidance year-by-year trajectory (if MODELLED)
        yearly_pav = bd.get("yearly_penalty_avoidance_usd", {})
        if yearly_pav:
            st.markdown("**FuelEU Penalty Avoidance — Annual Trajectory**")
            yearly_df = pd.DataFrame([
                {"Year": yr, "Penalty Avoidance (USD)": v}
                for yr, v in sorted(yearly_pav.items())
            ])
            fig_pen = px.bar(
                yearly_df, x="Year", y="Penalty Avoidance (USD)",
                title="Annual FuelEU Penalty Avoidance: Baseline Diesel vs. Optimised Fleet",
                color_discrete_sequence=[COLOUR_MODELLED],
            )
            fig_pen.update_layout(yaxis_tickformat="$,.0f", height=300)
            st.plotly_chart(fig_pen, use_container_width=True)


    # 4. Multi-Year Transition Roadmap Chart
    st.subheader("📈 Multi-Year Fleet Transition Roadmap (2026–2040)")
    st.caption("Constraint-based transition heuristic bounded by Technology Readiness Levels (TRL) and drydock conversion limits.")
    milestone_rows = []
    for m in roadmap.milestones:
        for f, sh in m.fuel_shares.items():
            if sh > 0:
                milestone_rows.append({
                    "Year": str(m.year),
                    "Fuel": f,
                    "Share (%)": sh * 100.0,
                    "Emissions (t)": m.annual_emissions_tons,
                    "Capex ($M)": m.capex_usd / 1e6,
                    "CII Rating": m.cii_rating,
                })
    df_m = pd.DataFrame(milestone_rows)
    fig_roadmap = px.bar(
        df_m,
        x="Year",
        y="Share (%)",
        color="Fuel",
        title="Fleet Fuel Share Trajectory across Planning Milestones",
        barmode="stack",
        color_discrete_map={
            "Diesel": "#64748B",
            "LNG": "#0284C7",
            "Methanol": "#10B981",
            "Hydrogen": "#8B5CF6",
            "Ammonia": "#F59E0B",
        },
    )
    st.plotly_chart(fig_roadmap, use_container_width=True)

    # 5. Lifecycle Assessment (LCA) Breakdown
    st.subheader("🌿 Well-to-Wake (WTW) Lifecycle Emissions Decomposition")

    # Milestone year options from transition roadmap
    milestone_years = [m.year for m in roadmap.milestones]
    default_lca_idx = milestone_years.index(2030) if 2030 in milestone_years else (len(milestone_years) - 1)

    lca_milestone_year = st.selectbox(
        "Evaluation Milestone Year for LCA Decomposition",
        milestone_years,
        index=default_lca_idx,
        help="Select which roadmap milestone year to evaluate under full Well-to-Wake LCA with chosen feedstock pathways."
    )

    target_milestone = next((m for m in roadmap.milestones if m.year == lca_milestone_year), roadmap.milestones[0])
    eval_fuel_shares = {f: sh for f, sh in target_milestone.fuel_shares.items() if sh > 0.0}

    c_lca1, c_lca2 = st.columns(2)
    with c_lca1:
        # Assess fuel mix under LCA engine with user-selected pathways
        fuel_split = {
            f: fleet_mix_res.fuel_consumption * sh
            for f, sh in eval_fuel_shares.items()
            if sh > 0.0
        }
        lca_res = lca_engine.assess_fleet_lifecycle(
            fuel_consumption=fuel_split,
            pathways=fuel_pathways,
            carbon_price_usd=carbon_price,
        )
        lca_rows = []
        for f, res in lca_res.items():
            fuel_label = f"{f} ({res.pathway})"
            lca_rows.append({"Fuel": fuel_label, "Stage": "Tank-to-Wake (Direct Combustion)", "CO2e (tons)": res.tank_to_wake_emissions})
            lca_rows.append({"Fuel": fuel_label, "Stage": "Well-to-Tank (Upstream Feedstock & Prod)", "CO2e (tons)": res.fuel_production_emissions})
            lca_rows.append({"Fuel": fuel_label, "Stage": "Well-to-Tank (Transport & Logistics)", "CO2e (tons)": res.fuel_transport_emissions})
            if res.fuel_storage_emissions > 0:
                lca_rows.append({"Fuel": fuel_label, "Stage": "Well-to-Tank (Methane Slip / Fugitive)", "CO2e (tons)": res.fuel_storage_emissions})

        df_lca = pd.DataFrame(lca_rows)
        fig_lca = px.bar(
            df_lca,
            x="Fuel",
            y="CO2e (tons)",
            color="Stage",
            title=f"Well-to-Wake Lifecycle Footprint Breakdown ({lca_milestone_year} Milestone)",
            barmode="stack",
            color_discrete_sequence=["#EF4444", "#3B82F6", "#10B981", "#F59E0B"],
        )
        st.plotly_chart(fig_lca, use_container_width=True)

    with c_lca2:
        # Carbon cost & FuelEU penalty intensity
        intensities = [
            {
                "Fuel": f"{f} ({res.pathway})",
                "WTW Intensity (gCO2e/MJ)": res.emission_intensity_g_per_mj,
                "Cost ($/t)": res.lifecycle_cost / max(res.fuel_consumption_tons, 1e-4),
            }
            for f, res in lca_res.items()
        ]
        df_int = pd.DataFrame(intensities)
        fig_int = px.scatter(
            df_int,
            x="WTW Intensity (gCO2e/MJ)",
            y="Cost ($/t)",
            text="Fuel",
            size=[30] * len(df_int),
            title="Fuel Lifecycle Abatement Cost vs GHG Intensity",
            color="Fuel",
        )
        fig_int.update_traces(textposition="top center")
        st.plotly_chart(fig_int, use_container_width=True)

    # 6. Forward Regulatory Risk & Compliance Forecast (2026–2040)
    st.subheader("⚖️ Regulatory Forecast: IMO CII & EU FuelEU Maritime Exposure")
    c_reg1, c_reg2 = st.columns(2)
    with c_reg1:
        # Penalty trajectory
        years_list = list(forecast.planning_horizon)
        penalties_list = [forecast.projected_penalties_usd[y] for y in years_list]
        fig_pen = go.Figure()
        fig_pen.add_trace(go.Bar(
            x=[str(y) for y in years_list],
            y=penalties_list,
            marker_color="#DC2626",
            name="Projected FuelEU Deficit Penalty ($)",
        ))
        fig_pen.update_layout(title="Annual Projected FuelEU Maritime Penalties ($USD)")
        st.plotly_chart(fig_pen, use_container_width=True)

    with c_reg2:
        # Compliance probability under operational uncertainty
        probs_list = [forecast.estimated_compliance_probability[y] * 100.0 for y in years_list]
        fig_prob = go.Figure()
        fig_prob.add_trace(go.Scatter(
            x=[str(y) for y in years_list],
            y=probs_list,
            mode="lines+markers",
            line=dict(color="#10B981", width=3),
            name="Estimated Compliance Probability",
        ))
        fig_prob.update_layout(
            title="Estimated Compliance Probability Under Operational Uncertainty (%)",
            yaxis=dict(range=[0, 105]),
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    # 7. Multi-Scenario Stress Test Sensitivity Matrix
    st.subheader("🌪️ Multi-Scenario Stress Test Sensitivity Analysis")
    scen_res = scenario_analyzer.compare_scenarios(base_scenario=scenario)
    scen_table_data = []
    for s_name, data in scen_res.metrics_comparison.items():
        scen_table_data.append({
            "Scenario Name": s_name,
            "Carbon ($/t)": data["carbon_price"],
            "Fossil Mult": f"{data.get('fossil_fuel_multiplier', 1.0):.2f}x",
            "Alt Mult": f"{data.get('alt_fuel_multiplier', 1.0):.2f}x",
            "Reg Stringency": f"{data.get('regulation_factor', 1.0):.2f}x",
            "Cost ($M)": round(data["operational_cost_usd"] / 1e6, 2),
            "WTW CO2e (t)": round(data["emissions_tons"], 0),
            "Reliability": f"{data['reliability_score']:.1f}%",
            "Speed (kn)": data["speed_knots"],
            "Vessels": int(data["vessel_count"]),
        })
    df_scen = pd.DataFrame(scen_table_data)
    st.dataframe(df_scen, use_container_width=True)

    # 8. Evidence vs. Assumption Classification Layer
    st.subheader("🏷️ Evidence vs. Assumption Classification Layer")
    st.caption("Distinguishing official statutory facts from engineering models and scenario projections.")
    ev_df = pd.DataFrame(list(rec.evidence_items))
    st.dataframe(ev_df, use_container_width=True)

    # 9. Priority Actions & Strategic Risk Mitigations
    st.subheader("📋 Executive Action Plan & Operational Risk Matrix")
    c_act, c_risk = st.columns(2)
    with c_act:
        st.markdown("#### Priority 1–3 Year Actions")
        for a in rec.priority_actions:
            st.markdown(f"- {a}")
    with c_risk:
        st.markdown("#### Operational Risk Mitigation Matrix")
        df_risk = pd.DataFrame(list(rec.risk_and_mitigations))
        st.dataframe(df_risk, use_container_width=True)

    # 10. Automated Multi-Format Report Downloads
    st.subheader("📥 Export Executive Decision Package")
    b1, b2, b3 = st.columns(3)
    if b1.button("Generate Executive Decision Reports", type="primary"):
        reports = report_gen.generate_all_reports(
            recommendation=rec,
            scenario=scenario,
            strategy_rec=strat_rec,
            roadmap=roadmap,
            forecast=forecast,
            scenario_res=scen_res,
        )
        st.success(f"Reports successfully generated in `outputs/reports/`!")
        st.markdown(f"- **PDF:** `{reports['pdf']}`")
        st.markdown(f"- **Markdown:** `{reports['markdown']}`")
        st.markdown(f"- **JSON:** `{reports['json']}`")
