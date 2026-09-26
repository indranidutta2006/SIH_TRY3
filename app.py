"""Main Streamlit Executive Dashboard entrypoint for SIH26138.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Serves 5 functional analytical views:
1. Fleet Telemetry & Ingestion Data Explorer
2. Fuel Consumption Prediction & Quantum Modeling (QIFCP)
3. Green Fleet Optimization & Swarm Scalability (QPSO vs PSO & Pareto)
4. Statutory Decarbonization & Compliance (IMO CII & FuelEU Maritime)
5. Macro Scenario Analysis & Alternative Fuel Pathways
"""

import sys
from pathlib import Path
import streamlit as st

# Ensure project root is on sys.path for direct headless execution
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.dashboard import (
    render_benchmarking_page,
    render_compliance_page,
    render_optimization_page,
    render_overview_page,
    render_prediction_page,
    render_reliability_page,
    render_scenarios_page,
    render_strategy_page,
)

# 1. Streamlit Global Page Configuration
st.set_page_config(
    page_title="Quantum Green Fleet Navigator | SIH26138",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/cargo-ship.png", width=72)
st.sidebar.title("Fleet Navigator")
st.sidebar.caption("SIH26138: Quantum & Green Optimization")

page_selection = st.sidebar.radio(
    "Navigation Menu",
    [
        "1. Fleet Telemetry & Explorer",
        "2. Fuel Prediction & QIFCP",
        "3. Swarm Optimization (QPSO)",
        "4. Statutory Compliance (IMO/EU)",
        "5. Scenario Analysis & Alternative Fuels",
        "6. Fleet Strategy Optimization",
        "7. Operational Reliability",
        "8. Benchmarking & Validation",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Architecture Status:**
    - 🟢 Production Model: `HistGBDT`
    - ⚛️ Quantum Model: `QIFCP`
    - ⚡ Swarm Engine: `QPSO (Swarm)`
    - ⚖️ Decarbonization: `IMO CII / FuelEU`
    - 🛡️ Physics Engine: `Admiralty + LHV`
    - 🎯 Strategy Engine: `Mix + Capacity + Speed`
    - ⚓ Reliability Engine: `Demand + Schedule Buffer`
    - 📊 Benchmarking: `Quantum Advantage + Baselines`
    """
)

st.sidebar.markdown("---")
st.sidebar.caption("© 2026 Smart India Hackathon | Problem SIH26138")

# 3. Page Routing
if page_selection == "1. Fleet Telemetry & Explorer":
    render_overview_page()
elif page_selection == "2. Fuel Prediction & QIFCP":
    render_prediction_page()
elif page_selection == "3. Swarm Optimization (QPSO)":
    render_optimization_page()
elif page_selection == "4. Statutory Compliance (IMO/EU)":
    render_compliance_page()
elif page_selection == "5. Scenario Analysis & Alternative Fuels":
    render_scenarios_page()
elif page_selection == "6. Fleet Strategy Optimization":
    render_strategy_page()
elif page_selection == "7. Operational Reliability":
    render_reliability_page()
elif page_selection == "8. Benchmarking & Validation":
    render_benchmarking_page()

