"""Page 1: Fleet Telemetry & Ingestion Data Explorer."""

from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st


def render_overview_page() -> None:
    """Render Executive Overview & Fleet Telemetry explorer."""
    st.title("🚢 Fleet Telemetry & Data Explorer")
    st.markdown(
        """
        **SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization**  
        Ingested immutable voyage telemetry representing commercial vessel operations, hydrodynamic resistance,
        environmental sea states, and multi-fuel bunkering regimes.
        """
    )

    data_file = Path("data/raw/voyages_sample.csv")
    if not data_file.exists():
        st.warning(f"Telemetry file `{data_file}` not found. Please run the build script.")
        return

    df = pd.read_csv(data_file)

    # 1. High-level KPI summary cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Voyages", f"{len(df):,}")
    with col2:
        avg_dwt = df["vessel_dwt"].mean() if "vessel_dwt" in df.columns else 0.0
        st.metric("Avg Vessel DWT", f"{avg_dwt:,.0f} t")
    with col3:
        avg_speed = df["speed_knots"].mean() if "speed_knots" in df.columns else 0.0
        st.metric("Avg Speed", f"{avg_speed:.1f} kts")
    with col4:
        avg_fuel = df["fuel_consumption"].mean() if "fuel_consumption" in df.columns else 0.0
        st.metric("Avg Fuel Cons.", f"{avg_fuel:.1f} t")
    with col5:
        total_co2 = df["co2_emissions"].sum() if "co2_emissions" in df.columns else 0.0
        st.metric("Total CO₂ Tracked", f"{total_co2:,.0f} t")

    st.markdown("---")

    # 2. Interactive filtering
    st.subheader("Filter & Explore Voyages")
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        selected_types = st.multiselect(
            "Vessel Types",
            options=sorted(df["vessel_type"].dropna().unique()),
            default=sorted(df["vessel_type"].dropna().unique()),
        )
    with filter_col2:
        selected_fuels = st.multiselect(
            "Fuel Types",
            options=sorted(df["fuel_type"].dropna().unique()),
            default=sorted(df["fuel_type"].dropna().unique()),
        )

    filtered_df = df[
        (df["vessel_type"].isin(selected_types)) & (df["fuel_type"].isin(selected_fuels))
    ]

    # 3. Visualizations
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig_type = px.histogram(
            filtered_df,
            x="vessel_type",
            color="fuel_type",
            barmode="group",
            title="Voyage Distribution by Vessel and Fuel Type",
        )
        st.plotly_chart(fig_type, use_container_width=True)

    with chart_col2:
        fig_scatter = px.scatter(
            filtered_df.sample(min(1000, len(filtered_df)), random_state=42),
            x="speed_knots",
            y="fuel_consumption",
            color="vessel_type",
            size="cargo_tons",
            hover_data=["voyage_id", "vessel_dwt", "weather_factor"],
            title="Speed vs Fuel Consumption (Cubic Drag Law)",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # 4. Data table preview
    st.subheader("Raw Telemetry Sample")
    st.dataframe(filtered_df.head(100), use_container_width=True)
