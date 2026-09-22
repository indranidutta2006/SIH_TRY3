"""Page 3: Green Fleet Optimization & Swarm Scalability (QPSO vs PSO & Pareto)."""

import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_optimization_page() -> None:
    """Render Green Fleet Swarm Optimization & Scalability page."""
    st.title("⚡ Green Fleet Swarm Optimization & Scalability")
    st.markdown(
        """
        **Objective 2:** Multivariable fleet optimization minimizing fuel cost, Well-to-Wake CO₂e emissions,
        and timetable delay penalties using Quantum-Behaved Particle Swarm Optimization (**QPSO**) vs classical PSO.
        """
    )

    report_path = Path("outputs/reports/optimization_benchmark.json")
    if not report_path.exists():
        st.warning(f"Optimization report `{report_path}` not found. Please run the build script.")
        return

    data = json.loads(report_path.read_text(encoding="utf-8"))
    summary = data.get("summary", [])
    detailed = data.get("detailed_runs", {})
    pareto_data = data.get("pareto_analysis_medium", {})

    # 1. Scalability Summary Table
    st.subheader("Swarm Scalability Benchmark (Equal-Budget Protocol)")
    sum_rows = []
    for item in summary:
        sum_rows.append(
            {
                "Tier": item.get("tier", "").capitalize(),
                "Assigned Voyages (D=2N)": item.get("assigned_voyages", 0),
                "QPSO Best Score": item.get("qpso_best_score", 0.0),
                "PSO Best Score": item.get("pso_best_score", 0.0),
                "QPSO Improvement (%)": f"{item.get('qpso_improvement_pct', 0.0):+.2f}%",
                "Evaluation Budget": item.get("evaluations_per_algo", 0),
                "QPSO Wall Clock (s)": item.get("qpso_runtime_s", 0.0),
                "PSO Wall Clock (s)": item.get("pso_runtime_s", 0.0),
            }
        )

    st.dataframe(pd.DataFrame(sum_rows), use_container_width=True)

    # 2. Convergence Trajectories Plot
    st.subheader("Convergence Trajectories (QPSO Quantum Tunneling vs Classical PSO)")
    selected_tier = st.selectbox("Select Problem Tier to View Convergence", ["small", "medium", "large"], index=0)

    if selected_tier in detailed:
        qpso_hist = detailed[selected_tier].get("QPSO", {}).get("history", [])
        pso_hist = detailed[selected_tier].get("PSO", {}).get("history", [])

        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(y=qpso_hist, mode="lines+markers", name="QPSO (Quantum Swarm)", line=dict(color="#00CC96", width=3)))
        fig_conv.add_trace(go.Scatter(y=pso_hist, mode="lines+markers", name="PSO (Classical Baseline)", line=dict(color="#EF553B", width=2, dash="dash")))

        fig_conv.update_layout(
            title=f"{selected_tier.capitalize()} Tier Iteration History",
            xaxis_title="Iteration",
            yaxis_title="Objective Function Score J",
            hovermode="x unified",
        )
        st.plotly_chart(fig_conv, use_container_width=True)

    st.markdown("---")

    # 3. Bi-Objective Pareto Tradeoff Frontier
    st.subheader("🎯 Bi-Objective Pareto Tradeoff Frontier (NSGA-II)")
    st.markdown("Tradeoff frontier balancing operational fuel bunkering cost against Well-to-Wake greenhouse gas emissions.")

    front = pareto_data.get("pareto_front", [])
    if front:
        p_df = pd.DataFrame(front)
        fig_pareto = px.scatter(
            p_df,
            x="fuel_cost_usd",
            y="co2e_tons",
            text=[f"Point {i+1}" for i in range(len(p_df))],
            title="Medium Fleet Pareto Frontier: Fuel Cost vs Lifecycle CO₂e",
            labels={"fuel_cost_usd": "Fleet Fuel Cost (USD)", "co2e_tons": "Well-to-Wake CO₂e (metric tons)"},
        )
        fig_pareto.update_traces(marker=dict(size=14, color="#636EFA", symbol="diamond"), textposition="top right")
        st.plotly_chart(fig_pareto, use_container_width=True)

        st.dataframe(p_df, use_container_width=True)
