"""Page 6: Strategic Fleet Optimization (Fleet Mix, Capacity, Speed & Deployment).

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Serves interactive decision-support for:
1. Recommended Fleet Mix (Vessel classes & Alternative Fuels)
2. Recommended Vessel Capacity & Cargo Utilization
3. Recommended Eco-Speed & Arrival Timetable
4. Route-to-Vessel Deployment Plan
5. Baseline vs. Optimized Tradeoff Delta Analysis
6. Scenario Persistence (JSON save / load)
"""

from datetime import UTC, datetime
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from contracts.schemas import OptimizationScenario, OptimizationStatus
from src.optimization.fleet_strategy_optimizer import FleetStrategyOptimizer


def render_strategy_page() -> None:
    """Render the Fleet Strategy Optimization executive dashboard page."""
    st.title("🚢 Fleet Strategy Optimization: Mix, Capacity & Eco-Speed")
    st.markdown(
        """
        **SIH26138 Core Requirement:** Integrated decision platform determining the **optimal mix of vessel types**, 
        **capacities**, and **cruising speeds** while minimizing fuel consumption, operational cost, and 
        lifecycle greenhouse gas emissions under monetized carbon pricing and statutory compliance.
        """
    )

    optimizer = FleetStrategyOptimizer()

    # 1. Interactive Scenario Parameter Configuration
    with st.expander("🛠️ Operational Scenario Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            cargo_demand = st.number_input(
                "Annual Cargo Demand (metric tons)",
                min_value=5000.0,
                max_value=2_000_000.0,
                value=250_000.0,
                step=25000.0,
                help="Total transport volume requirement to deliver over the planning period.",
            )
            route_distance = st.number_input(
                "Representative Voyage Distance (nm)",
                min_value=200.0,
                max_value=15000.0,
                value=3500.0,
                step=250.0,
                help="Average corridor voyage distance (e.g. Rotterdam to Singapore or Transpacific).",
            )
            deadline_hours = st.number_input(
                "Transit Arrival Deadline (hours)",
                min_value=24.0,
                max_value=1000.0,
                value=260.0,
                step=10.0,
                help="Maximum allowable transit time before commercial delay demurrage triggers.",
            )

        with col2:
            carbon_price = st.slider(
                "Carbon Price ($/ton CO₂e)",
                min_value=0.0,
                max_value=250.0,
                value=80.0,
                step=10.0,
                help="Monetized shadow carbon price (e.g. EU ETS / IMO Net-Zero Framework carbon levy).",
            )
            vessel_class = st.selectbox(
                "Target Vessel Class",
                ["PANAMAX", "FEEDER", "POST_PANAMAX", "CAPESIZE"],
                index=0,
                help="Naval architectural design class governing draft limits and capacity scaling.",
            )
            weather_factor = st.slider(
                "Adverse Weather Severity Factor",
                min_value=1.0,
                max_value=1.50,
                value=1.05,
                step=0.05,
                help="Hydrodynamic resistance multiplier due to rough seas, head winds, and fouling.",
            )

        with col3:
            budget_millions = st.slider(
                "Capital & Charter Budget ($ Millions)",
                min_value=10.0,
                max_value=300.0,
                value=120.0,
                step=10.0,
                help="Maximum capital expenditure / long-term charter financing ceiling.",
            )
            max_transition_rate = st.slider(
                "Max Green Fuel Transition Rate",
                min_value=0.0,
                max_value=1.0,
                value=0.40,
                step=0.05,
                help="Maximum fraction of fleet allowed to transition to alternative green fuels per cycle.",
            )
            service_level = st.slider(
                "Target Minimum Service Level",
                min_value=0.80,
                max_value=1.00,
                value=0.95,
                step=0.01,
                help="Contractual cargo fulfillment threshold.",
            )
            solver_mode = st.selectbox(
                "Fleet Composition Solver",
                ["Deterministic MIP (Fast Enumeration)", "Quantum-Inspired QPSO (Metaheuristic)"],
                index=0,
                help="Optimization algorithm for the discrete fleet composition layer.",
            )
            solver_key = "qpso" if "QPSO" in solver_mode else "deterministic"

    scenario = OptimizationScenario(
        cargo_demand=float(cargo_demand),
        route_distance=float(route_distance),
        deadline_hours=float(deadline_hours),
        scenario_id="SCENARIO-EXEC-01",
        created_at=datetime.now(UTC).isoformat(),
        carbon_price=float(carbon_price),
        budget=float(budget_millions * 1e6),
        weather_factor=float(weather_factor),
        vessel_class=str(vessel_class),
        service_level=float(service_level),
        max_transition_rate=float(max_transition_rate),
    )

    run_col, save_col, load_col = st.columns([2, 1, 1])
    with run_col:
        run_clicked = st.button("⚡ Execute Strategic Fleet Optimization", type="primary", use_container_width=True)

    # State management for results
    if "current_recommendation" not in st.session_state or run_clicked:
        with st.spinner(f"Optimizing Fleet Composition ({solver_mode}), Capacity Sizing, Eco-Speed, and Deployment Schedule..."):
            rec = optimizer.optimize_strategy(scenario, solver=solver_key)
            st.session_state["current_recommendation"] = rec

    rec = st.session_state["current_recommendation"]

    # 2. Status Banner
    solver_badge = rec.summary.get("composition_solver", rec.fleet_mix.metadata.get("solver_name", "deterministic"))
    if rec.status == OptimizationStatus.SUCCESS:
        st.success(f"✅ Strategy Optimization Converged Successfully [Status: {rec.status.value} | Solver: {solver_badge}]")
    elif rec.status == OptimizationStatus.DEADLINE_VIOLATED:
        st.warning(f"⚠️ Transit Deadline Infeasible under safe maximum speed [Status: {rec.status.value}]")
    elif rec.status == OptimizationStatus.BUDGET_EXCEEDED:
        st.error(f"❌ Fleet Capital Budget Exceeded [Status: {rec.status.value}]")
    elif rec.status == OptimizationStatus.DEMAND_UNSATISFIABLE:
        st.error(f"❌ Fleet Size Limit cannot satisfy target cargo demand [Status: {rec.status.value}]")
    else:
        st.error(f"❌ Infeasible constraints detected [Status: {rec.status.value}]")

    st.markdown("---")

    # 3. High-Level Executive Summary Cards
    st.subheader("🎯 Executive KPI Overview")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    total_ships = sum(v for k, v in rec.fleet_mix.fleet_mix.items() if k in {"feeder", "medium", "large"})
    kpi1.metric("Active Fleet Size", f"{total_ships} vessels", f"{rec.fleet_mix.fleet_mix.get('medium', 0)} Panamax")
    kpi2.metric("Optimal Capacity", f"{rec.capacity_recommendation.recommended_capacity:,.0f} DWT", f"{rec.capacity_recommendation.capacity_teu:,.0f} TEU")
    kpi3.metric("Cruising Speed", f"{rec.speed_recommendation.optimal_speed:.1f} kts", f"ETA: {rec.speed_recommendation.estimated_eta:.1f} h")
    kpi4.metric("Annual Fuel Mass", f"{rec.fuel_estimate:,.1f} t", f"${rec.cost_estimate/1e6:.2f}M Total Cost")
    kpi5.metric("Lifecycle CO₂e", f"{rec.emissions_estimate:,.1f} t", f"Reliability: {rec.service_reliability*100:.1f}%" if rec.service_reliability else "N/A")

    st.markdown("---")

    # 4. Baseline vs. Optimized Comparison Table & Deltas
    st.subheader("📊 Baseline vs. Optimized Strategy Impact")
    b_data = rec.baseline_comparison
    if b_data and "baseline" in b_data and "optimized" in b_data and "deltas" in b_data:
        comp_df = pd.DataFrame([
            {
                "Key Performance Metric": "Annual Fuel Consumption (tons)",
                "Conventional Baseline": f"{b_data['baseline']['fuel_consumption_tons']:,.1f}",
                "Green Optimized Strategy": f"{b_data['optimized']['fuel_consumption_tons']:,.1f}",
                "Absolute Reduction": f"-{b_data['deltas']['fuel_reduction_tons']:,.1f} t",
                "Relative Savings": f"-{b_data['deltas']['fuel_reduction_pct']:.2f}%",
            },
            {
                "Key Performance Metric": "Total Monetized Cost (USD)",
                "Conventional Baseline": f"${b_data['baseline']['total_cost_usd']:,.2f}",
                "Green Optimized Strategy": f"${b_data['optimized']['total_cost_usd']:,.2f}",
                "Absolute Reduction": f"-${b_data['deltas']['cost_savings_usd']:,.2f}",
                "Relative Savings": f"-{b_data['deltas']['cost_savings_pct']:.2f}%",
            },
            {
                "Key Performance Metric": "Well-to-Wake Lifecycle CO₂e (tons)",
                "Conventional Baseline": f"{b_data['baseline']['emissions_co2e_tons']:,.1f}",
                "Green Optimized Strategy": f"{b_data['optimized']['emissions_co2e_tons']:,.1f}",
                "Absolute Reduction": f"-{b_data['deltas']['emissions_abated_tons']:,.1f} t",
                "Relative Savings": f"-{b_data['deltas']['emissions_abated_pct']:.2f}%",
            },
            {
                "Key Performance Metric": "Operational Cruising Speed (knots)",
                "Conventional Baseline": f"{b_data['baseline']['cruising_speed_knots']:.1f} kts",
                "Green Optimized Strategy": f"{b_data['optimized']['cruising_speed_knots']:.1f} kts",
                "Absolute Reduction": f"{b_data['optimized']['cruising_speed_knots'] - b_data['baseline']['cruising_speed_knots']:+.1f} kts",
                "Relative Savings": "Eco-Steaming",
            },
        ])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # 5. Visual Analytics: Fleet Mix & Eco-Speed Curve
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("🚢 Recommended Fleet Composition")
        mix = rec.fleet_mix.fleet_mix

        size_df = pd.DataFrame([
            {"Category": "Feeder (5k-15k DWT)", "Count": mix.get("feeder", 0)},
            {"Category": "Medium / Panamax (25k-55k DWT)", "Count": mix.get("medium", 0)},
            {"Category": "Large / Capesize (80k-180k DWT)", "Count": mix.get("large", 0)},
        ])
        fuel_df = pd.DataFrame([
            {"Fuel": "Diesel (VLSFO)", "Count": mix.get("diesel", 0)},
            {"Fuel": "LNG (Methane)", "Count": mix.get("lng", 0)},
            {"Fuel": "Methanol (Green)", "Count": mix.get("methanol", 0)},
            {"Fuel": "Hydrogen (Fuel Cell)", "Count": mix.get("hydrogen", 0)},
            {"Fuel": "Ammonia (Zero Carbon)", "Count": mix.get("ammonia", 0)},
        ])

        fig_mix = px.bar(
            fuel_df,
            x="Fuel",
            y="Count",
            color="Fuel",
            title=f"Fuel Technology Mix (Total: {total_ships} vessels)",
            text="Count",
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        st.plotly_chart(fig_mix, use_container_width=True)

    with chart_col2:
        st.subheader("⚡ Speed vs. Fuel / Cost Tradeoff Frontier")
        # Synthesize speed curve for visual Pareto demonstration
        speeds = [9.0 + i * 0.5 for i in range(25)]
        curve_data = []
        for s in speeds:
            transit = scenario.route_distance / s
            delay = max(0.0, transit - scenario.deadline_hours)
            # Power law ~ s^3
            fuel_approx = 0.0018 * (rec.capacity_recommendation.recommended_capacity ** (2/3)) * (s ** 3) * (transit / 24.0) * 0.04
            cost_approx = (fuel_approx * 650.0) + (fuel_approx * 3.15 * scenario.carbon_price) + (delay * 600.0)
            curve_data.append({"Speed (knots)": s, "Estimated Fuel (tons)": fuel_approx, "Total Trip Cost ($)": cost_approx, "Delay (hours)": delay})

        df_curve = pd.DataFrame(curve_data)
        fig_speed = px.line(
            df_curve,
            x="Speed (knots)",
            y="Total Trip Cost ($)",
            title="Operational Speed Optimization Curve (Demurrage vs Fuel Drag)",
            markers=True,
        )
        # Highlight optimal point
        fig_speed.add_trace(go.Scatter(
            x=[rec.speed_recommendation.optimal_speed],
            y=[df_curve.loc[(df_curve["Speed (knots)"] - rec.speed_recommendation.optimal_speed).abs().idxmin(), "Total Trip Cost ($)"]],
            mode="markers+text",
            name="Optimal Eco-Speed",
            marker=dict(color="#00CC96", size=16, symbol="star"),
            text=[f"Optimal: {rec.speed_recommendation.optimal_speed:.1f} kts"],
            textposition="top center",
        ))
        st.plotly_chart(fig_speed, use_container_width=True)

    st.markdown("---")

    # 6. Fleet Deployment Schedule Table
    st.subheader("🗺️ Operational Fleet Deployment Plan")
    st.markdown("Automated assignment of active fleet assets to maritime trade corridors:")

    deploy_rows = []
    for r_id, vessels in rec.deployment_plan.items():
        for v in vessels:
            deploy_rows.append({
                "Route ID": r_id,
                "Origin": v.get("origin", "Hub Port A"),
                "Destination": v.get("destination", "Hub Port B"),
                "Assigned Vessel": v.get("vessel_id", ""),
                "Vessel Class": v.get("vessel_class", ""),
                "Fuel Powertrain": v.get("fuel_type", ""),
                "Allocated Capacity (DWT)": f"{v.get('allocated_capacity_dwt', 0):,.0f}",
                "Cruising Speed (kts)": f"{v.get('cruising_speed_knots', 0):.1f}",
                "Estimated ETA (h)": f"{v.get('voyage_eta_hours', 0):.1f}",
            })

    if deploy_rows:
        st.dataframe(pd.DataFrame(deploy_rows), use_container_width=True, hide_index=True)

    # 6.1 QPSO Telemetry Expander (if QPSO solver selected)
    if "qpso" in str(rec.fleet_mix.metadata.get("solver_name", "")).lower():
        with st.expander("🔬 Quantum-Inspired Particle Swarm Convergence Telemetry", expanded=False):
            trace_df = pd.DataFrame(rec.fleet_mix.metadata.get("optimization_trace", []))
            if not trace_df.empty and "score" in trace_df.columns:
                fig_trace = go.Figure()
                fig_trace.add_trace(go.Scatter(
                    x=trace_df["iteration"],
                    y=trace_df["score"],
                    mode="lines+markers",
                    name="Global Best Score (gbest)",
                    line=dict(color="#00CC96", width=2),
                ))
                fig_trace.update_layout(
                    title="QPSO Delta-Potential Contraction-Expansion Convergence",
                    xaxis_title="Swarm Iteration",
                    yaxis_title="Scalarized Objective Score",
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_trace, use_container_width=True)
            st.caption(
                f"QPSO Evaluations: {rec.fleet_mix.metadata.get('n_evaluations', 0)} | "
                f"Convergence Score: {rec.fleet_mix.metadata.get('convergence_score', 1.0)} | "
                f"Solver Runtime: {rec.fleet_mix.metadata.get('runtime_ms', 0)}ms"
            )

    # 7. Scenario JSON Persistence
    st.markdown("---")
    st.subheader("💾 Scenario Persistence & Audit Export")
    col_dl, col_path = st.columns([1, 2])
    with col_dl:
        scenario_json = rec.to_json()
        st.download_button(
            label="📥 Download Strategy Recommendation (JSON)",
            data=scenario_json,
            file_name=f"fleet_strategy_{rec.scenario.scenario_id}.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_path:
        st.caption(f"Persisted Scenario ID: `{rec.scenario.scenario_id}` | Solver: `{rec.fleet_mix.metadata.get('solver_name', 'deterministic_combinatorial_mip')}` | Iterations: `{rec.fleet_mix.metadata.get('iterations', 0)}`")
