"""Page 7: Operational Reliability & Cargo Demand Satisfaction.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 2 Deliverable: Interactive executive dashboard for:
1. Schedule Reliability Monitoring (Score, On-Time Arrival Rate, Average Delays)
2. Explainable Reliability Score Decomposition (On-Time Reward, Delay Penalty, Missed Voyage Penalty)
3. Cargo Demand Satisfaction & Service-Level Gap Analysis
4. Route-Level Reliability & Corridor Stress Testing
5. Port Congestion & Adverse Weather Disruption Sensitivity
6. Operational Deployment Plan with Buffer Capacity & Redundancy
"""

from datetime import UTC, datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from contracts.schemas import OptimizationScenario, OptimizationStatus
from src.operations.demand_satisfaction_engine import CargoDemandSatisfactionEngine
from src.operations.reliability_engine import ScheduleReliabilityEngine
from src.optimization.fleet_strategy_optimizer import FleetStrategyOptimizer


def render_reliability_page() -> None:
    """Render the Operational Reliability and Cargo Demand Satisfaction page."""
    st.title("🛡️ Operational Reliability & Cargo Demand Satisfaction")
    st.markdown(
        """
        **SIH26138 Core Requirement:** Ensure **operational reliability**, **cargo demand satisfaction**, 
        and statutory compliance under stochastic weather disruptions, port turnaround bottlenecks, 
        and dynamic supply chain demand fluctuations.
        """
    )

    strategy_optimizer = FleetStrategyOptimizer()
    reliability_engine = ScheduleReliabilityEngine()
    demand_engine = CargoDemandSatisfactionEngine()

    # 1. Operational & Environmental Controls
    with st.expander("⚙️ Operational Reliability & Stress Testing Controls", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            cargo_demand = st.number_input(
                "Required Cargo Demand (metric tons)",
                min_value=10_000.0,
                max_value=2_000_000.0,
                value=250_000.0,
                step=25_000.0,
                help="Total contracted cargo volume required across the planning period.",
            )
            has_forecast = st.checkbox("Apply Stochastic Forecast Demand", value=False)
            forecasted_demand = None
            if has_forecast:
                forecasted_demand = st.number_input(
                    "Forecasted Demand (tons)",
                    min_value=10_000.0,
                    max_value=2_500_000.0,
                    value=280_000.0,
                    step=10_000.0,
                    help="Upper-bound demand forecast accounting for market surge.",
                )

            service_level_target = st.slider(
                "Target Service Level (%)",
                min_value=80.0,
                max_value=100.0,
                value=95.0,
                step=1.0,
                help="Minimum acceptable proportion of demand delivered without penalty.",
            ) / 100.0

        with col2:
            target_reliability = st.slider(
                "Target Reliability Threshold (/100)",
                min_value=60.0,
                max_value=99.0,
                value=90.0,
                step=1.0,
                help="Statutory minimum schedule reliability score required by charterers.",
            )
            port_delay_factor = st.slider(
                "Port Congestion Delay Multiplier",
                min_value=1.0,
                max_value=2.0,
                value=1.0,
                step=0.05,
                help="Turnaround slowdown factor at key hub ports (1.0 = normal, 1.5 = severe congestion).",
            )
            weather_factor = st.slider(
                "Adverse Sea State / Weather Multiplier",
                min_value=1.0,
                max_value=1.40,
                value=1.05,
                step=0.05,
                help="Admiralty hydrodynamic resistance penalty from rough sea states and winds.",
            )

        with col3:
            vessel_class = st.selectbox(
                "Primary Vessel Class",
                ["PANAMAX", "FEEDER", "POST_PANAMAX", "CAPESIZE"],
                index=0,
            )
            route_distance = st.number_input(
                "Key Corridor Distance (nm)",
                min_value=500.0,
                max_value=15000.0,
                value=3500.0,
                step=250.0,
            )
            deadline_hours = st.number_input(
                "Transit Deadline (hours)",
                min_value=48.0,
                max_value=800.0,
                value=260.0,
                step=10.0,
            )

    # 2. Build Operational Scenario and Run Optimization
    scenario = OptimizationScenario(
        cargo_demand=cargo_demand,
        route_distance=route_distance,
        deadline_hours=deadline_hours,
        scenario_id="SCEN-RELIABILITY-LIVE",
        created_at=datetime.now(UTC).isoformat(),
        carbon_price=80.0,
        budget=120_000_000.0,
        weather_factor=weather_factor,
        vessel_class=vessel_class,
        service_level=service_level_target,
        max_transition_rate=0.40,
        target_reliability=target_reliability,
        port_delay_factor=port_delay_factor,
        forecasted_demand=forecasted_demand,
    )

    with st.spinner("Running Integrated Reliability & Demand Satisfaction Optimization..."):
        rec = strategy_optimizer.optimize_strategy(scenario)

    rel = rec.reliability_metrics or reliability_engine.evaluate_schedule_reliability(scenario, rec.speed_recommendation)
    dem = rec.demand_metrics or demand_engine.evaluate_demand_satisfaction(scenario, fleet_composition=rec.fleet_mix)

    # 3. Status Alert Banner
    if rec.status == OptimizationStatus.SUCCESS:
        st.success(f"✅ **Fleet Strategy Feasible & Reliable**: Status `{rec.status.value}` — Demand fully satisfied and reliability above target threshold.")
    elif rec.status == OptimizationStatus.DEMAND_UNSATISFIABLE:
        st.error(f"❌ **Cargo Demand Unsatisfiable**: Status `{rec.status.value}` — Unserved volume: {dem.unserved_cargo:,.0f} tons. Failure Reason: `{rec.summary.get('failure_reason')}`.")
    elif rec.status == OptimizationStatus.DEADLINE_VIOLATED:
        st.error(f"⚠️ **Schedule Deadline Violated**: Status `{rec.status.value}` — Fastest possible transit plus port congestion exceeds {deadline_hours:.1f}h limit.")
    else:
        st.warning(f"⚠️ **Optimization Feasibility Alert**: Status `{rec.status.value}` — Failure Reason: `{rec.summary.get('failure_reason')}` (Target Reliability: {target_reliability:.1f} vs Achieved: {rel.reliability_score:.1f}).")

    st.markdown("---")

    # 4. Top-Level Executive KPI Scorecard
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric(
            label="Schedule Reliability Score",
            value=f"{rel.reliability_score:.1f} / 100",
            delta=f"{rel.reliability_score - target_reliability:+.1f} vs Target",
            delta_color="normal" if rel.reliability_score >= target_reliability else "inverse",
        )
    with kpi2:
        st.metric(
            label="On-Time Arrival Rate",
            value=f"{rel.on_time_arrival_rate * 100.0:.1f}%",
            delta=f"Total: {rel.total_voyages} voyages",
        )
    with kpi3:
        st.metric(
            label="Demand Satisfaction",
            value=f"{dem.satisfaction_percentage:.1f}%",
            delta=f"Service Level: {service_level_target * 100.0:.0f}%",
            delta_color="normal" if dem.is_satisfied else "inverse",
        )
    with kpi4:
        st.metric(
            label="Unserved Cargo Volume",
            value=f"{dem.unserved_cargo:,.0f} tons",
            delta="Zero Shortfall" if dem.unserved_cargo == 0 else "Cargo Deficit",
            delta_color="normal" if dem.unserved_cargo == 0 else "inverse",
        )
    with kpi5:
        st.metric(
            label="Average Delay",
            value=f"{rel.average_delay_hours:.1f} hrs",
            delta=f"Max: {rel.max_delay_hours:.1f} hrs",
            delta_color="inverse" if rel.average_delay_hours > 0 else "normal",
        )

    st.markdown("---")

    # 5. Visual Analytical Tiers
    tab1, tab2, tab3 = st.tabs([
        "📊 Reliability Explainability & Decomposition",
        "🗺️ Route Corridors & Contingency Deployment",
        "📈 Stress Testing & Sensitivity Curves",
    ])

    with tab1:
        st.subheader("Explainable Reliability Score Decomposition")
        st.markdown(
            """
            The normalized reliability score is rigorously decomposed into rewarded on-time adherence 
            and penalized delay / cancellation components:
            $$\\text{Reliability} = \\max\\left(0, \\min\\left(100, 100 \\times (w_1 \\cdot \\text{on\\_time} - w_2 \\cdot \\text{delay\\_ratio} - w_3 \\cdot \\text{missed\\_rate})\\right)\\right)$$
            """
        )

        col_dec1, col_dec2 = st.columns([1, 1])

        with col_dec1:
            breakdown = rel.score_breakdown
            fig_decomp = go.Figure(
                go.Waterfall(
                    name="Score Breakdown",
                    orientation="v",
                    measure=["absolute", "relative", "relative", "total"],
                    x=["On-Time Component", "Transit Delay Penalty", "Missed Voyage Penalty", "Final Reliability Score"],
                    y=[
                        breakdown.get("on_time_component", 100.0),
                        -breakdown.get("delay_penalty", 0.0),
                        -breakdown.get("missed_voyage_penalty", 0.0),
                        rel.reliability_score,
                    ],
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                    decreasing={"marker": {"color": "#EF553B"}},
                    increasing={"marker": {"color": "#00CC96"}},
                    totals={"marker": {"color": "#2CA02C"}},
                )
            )
            fig_decomp.update_layout(
                title="Waterfall Decomposition of Schedule Reliability Score",
                yaxis_title="Points",
                height=380,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_decomp, use_container_width=True)

        with col_dec2:
            # Demand satisfaction donut chart
            deliv = dem.delivered_cargo
            unserv = dem.unserved_cargo
            fig_dem = go.Figure(
                data=[
                    go.Pie(
                        labels=["Delivered Cargo", "Unserved Demand"],
                        values=[deliv, unserv],
                        hole=0.55,
                        marker_colors=["#10B981", "#EF4444" if unserv > 0 else "#D1D5DB"],
                    )
                ]
            )
            fig_dem.update_layout(
                title=f"Cargo Demand Fulfillment (Target: {service_level_target * 100.0:.0f}%)",
                height=380,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_dem, use_container_width=True)

    with tab2:
        st.subheader("Route Corridor Reliability & Operational Deployment")
        st.markdown(
            "Individual corridor reliability scores under prevailing sea conditions and port congestion factors:"
        )

        col_r1, col_r2 = st.columns([1, 1])

        with col_r1:
            route_df = pd.DataFrame(
                [{"Route Corridor": r_id, "Reliability Score": score} for r_id, score in rel.route_reliability.items()]
            )
            fig_route = px.bar(
                route_df,
                x="Route Corridor",
                y="Reliability Score",
                color="Reliability Score",
                color_continuous_scale="RdYlGn",
                range_y=[0, 100],
                title="Corridor-Level Reliability Comparison",
            )
            fig_route.add_hline(y=target_reliability, line_dash="dash", line_color="red", annotation_text="Target")
            fig_route.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_route, use_container_width=True)

        with col_r2:
            st.markdown("**Corridor Operational Statistics**")
            st.dataframe(route_df, use_container_width=True, hide_index=True)
            st.info(
                f"💡 **Buffer Capacity**: Fleet vessels maintain a minimum 15% DWT capacity redundancy "
                f"to absorb demand surges without violating departure windows."
            )

        st.markdown("### Operational Fleet Deployment Plan with Buffer Redundancy")
        deploy_rows = []
        for r_id, vessels in rec.deployment_plan.items():
            for v in vessels:
                deploy_rows.append({
                    "Route ID": r_id,
                    "Vessel ID": v.get("vessel_id"),
                    "Class": v.get("vessel_class"),
                    "Fuel Powertrain": v.get("fuel_type"),
                    "Allocated DWT": f"{v.get('allocated_capacity_dwt', 0.0):,.0f}",
                    "Buffer Capacity (DWT)": f"{v.get('buffer_capacity_dwt', 0.0):,.0f}",
                    "Redundancy Factor": f"{v.get('redundancy_factor', 1.15):.2f}x",
                    "Corridor Reliability": f"{v.get('route_reliability_score', 90.0):.1f}%",
                    "Cruising Speed": f"{v.get('cruising_speed_knots', 14.0):.1f} kts",
                    "Transit ETA": f"{v.get('voyage_eta_hours', 0.0):.1f}h",
                })
        st.dataframe(pd.DataFrame(deploy_rows), use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Disruption Stress Testing & Sensitivity Curves")
        st.markdown(
            "Simulating the resilience of fleet schedule reliability under severe port congestion and adverse sea states:"
        )

        congestion_factors = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]
        sim_results = []

        for cf in congestion_factors:
            test_scenario = OptimizationScenario(
                cargo_demand=cargo_demand,
                route_distance=route_distance,
                deadline_hours=deadline_hours,
                weather_factor=weather_factor,
                port_delay_factor=cf,
            )
            sim_rel = reliability_engine.evaluate_schedule_reliability(
                scenario=test_scenario,
                speed_result=rec.speed_recommendation,
            )
            sim_results.append({
                "Port Congestion Factor": f"{cf:.1f}x",
                "Reliability Score": sim_rel.reliability_score,
                "Average Delay (hrs)": sim_rel.average_delay_hours,
                "On-Time Rate (%)": sim_rel.on_time_arrival_rate * 100.0,
            })

        sim_df = pd.DataFrame(sim_results)

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            fig_sens = px.line(
                sim_df,
                x="Port Congestion Factor",
                y="Reliability Score",
                markers=True,
                title="Reliability Degradation vs. Port Congestion",
            )
            fig_sens.add_hline(y=target_reliability, line_dash="dash", line_color="red", annotation_text="Target Limit")
            fig_sens.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_sens, use_container_width=True)

        with col_s2:
            fig_delay = px.bar(
                sim_df,
                x="Port Congestion Factor",
                y="Average Delay (hrs)",
                title="Average Voyage Delays under Port Bottlenecks",
                color="Average Delay (hrs)",
                color_continuous_scale="Reds",
            )
            fig_delay.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_delay, use_container_width=True)
