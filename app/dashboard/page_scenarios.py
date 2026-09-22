"""Page 5: Macro Scenario Analysis & Alternative Fuel Transition."""

import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st


def render_scenarios_page() -> None:
    """Render Macro Scenario Analysis & Alternative Fuels page."""
    st.title("🌱 Macro Scenario Analysis & Alternative Fuel Pathways")
    st.markdown(
        """
        Simulate fleetwide transition pathways across **six maritime fuel technologies**:
        Conventional Diesel, LNG, Methanol, Green Hydrogen, Zero-Carbon Ammonia, and Electrified ShorePower.
        """
    )

    report_path = Path("outputs/reports/scenario_comparison.json")
    if not report_path.exists():
        st.warning(f"Scenario report `{report_path}` not found. Please run the build script.")
        return

    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = report.get("summary_table", [])
    rankings = report.get("rankings", {})

    # 1. Routing Architecture Guardrail Badge
    st.info(
        "🛡️ **Strict Architectural Routing Guardrail:**\n"
        "- **Historical ML Regressor:** Diesel, LNG, Methanol (`ProductionModelManager`)\n"
        "- **First-Principles Naval Hydrodynamics:** Hydrogen, Ammonia, ShorePower (`MaritimeFuelPhysicsEngine`)"
    )

    # 2. 6-Fuel Comparison Table
    st.subheader("Fleetwide Scenario Comparison Table (5 Vessels, 1,200 nm Voyage)")
    df_scen = pd.DataFrame(summary)
    st.dataframe(df_scen, use_container_width=True)

    # 3. Comparative Charts
    col1, col2 = st.columns(2)
    with col1:
        fig_cost = px.bar(
            df_scen,
            x="fuel_type",
            y="total_cost_usd",
            color="fuel_type",
            title="Total Voyage Operating Cost ($ USD)",
            labels={"total_cost_usd": "Operating Cost ($ USD)", "fuel_type": "Fuel Option"},
        )
        st.plotly_chart(fig_cost, use_container_width=True)

    with col2:
        fig_emiss = px.bar(
            df_scen,
            x="fuel_type",
            y="total_emissions_co2e_tons",
            color="fuel_type",
            title="Well-to-Wake Lifecycle CO₂e Emissions (metric tons)",
            labels={"total_emissions_co2e_tons": "WTW CO₂e (tons)", "fuel_type": "Fuel Option"},
        )
        st.plotly_chart(fig_emiss, use_container_width=True)

    st.markdown("---")

    # 4. Multi-Criteria Tradeoff Ranking
    st.subheader("⚖️ Multi-Criteria Balanced Tradeoff Ranking")
    st.caption("Normalized dimensionless scoring: 0.5 × (Cost / Max_Cost) + 0.5 × (Emissions / Max_Emissions)")

    balanced_list = rankings.get("balanced_multicriteria_tradeoff", [])
    for idx, sc_name in enumerate(balanced_list, start=1):
        st.markdown(f"**{idx}. {sc_name}**")
