"""Page 2: Fuel Consumption Prediction & Quantum-Inspired Modeling (QIFCP).

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Presents the canonical 5-seed vessel-disjoint prediction benchmark comparing the frozen
production predictor (Physics + Residual QIFCP) against naval physics and classical tree references.
"""

from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

from contracts.schemas import VoyageRecord
from src.prediction.benchmark_loader import (
    DEFAULT_CANONICAL_BENCHMARK_PATH,
    get_canonical_benchmark_metadata,
    get_canonical_prediction_dataframe,
    load_canonical_prediction_benchmark,
    resolve_canonical_benchmark_path,
)
from src.prediction.fuel_prediction_service import get_fuel_prediction_service


def render_prediction_page() -> None:
    """Render Prediction & QIFCP comparative analysis page."""
    st.title("🧠 Fuel Consumption Prediction & Quantum Modeling")
    st.markdown(
        """
        **Objective 1:** High-precision fuel consumption estimation across nonlinear hydrodynamic drag,
        weather resistance, and payload states using the frozen **Physics-Informed Residual QIFCP** architecture.
        """
    )

    report_path = resolve_canonical_benchmark_path()
    if not report_path.exists():
        st.warning(f"Canonical prediction benchmark report `{report_path}` not found. Please ensure reports are generated.")
        return

    try:
        data = load_canonical_prediction_benchmark(report_path)
        meta = get_canonical_benchmark_metadata(data)
        leaderboard_df = get_canonical_prediction_dataframe(data)
    except Exception as exc:
        st.error(f"Error loading canonical prediction benchmark: {exc}")
        return

    # Production Model Status Banner
    st.success(
        f"**Model Status:** `PRODUCTION / FROZEN`  \n"
        f"**Canonical Architecture:** `PhysicsInformedQIFCPRegressor` (K=3, M=15, adaptive entanglement, grouped γ, λ=1.0)  \n"
        f"**Accuracy Claim:** {meta.accuracy_claim}  \n"
        f"**Evaluation Context:** {meta.subtitle}  \n"
        f"**Dataset:** `{meta.dataset_path}`"
    )

    # 1. Canonical 5-Seed Leaderboard Table
    st.subheader("Model Benchmark Leaderboard")
    st.caption(f"{meta.subtitle} • Dataset: `{meta.dataset_path}`")

    display_cols = [
        "Model Architecture",
        "Status",
        "R² Score",
        "RMSE (tons)",
        "MAE (tons)",
        "sMAPE (%)",
        "Inference (ms / 100)",
        "Architecture Class",
    ]
    st.dataframe(
        leaderboard_df[display_cols],
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Note: Results represent unpooled sample means across 5 independent vessel-disjoint splits (GroupShuffleSplit on vessel_id). "
        "Synthetic benchmark evaluation on disjoint vessel groups; real-world commercial vessel deployment requires dynamic recalibration "
        "using operational telemetry (noon reports, torque meters, AIS)."
    )

    # 2. Performance Comparison Charts (Chart A & Chart B)
    col1, col2 = st.columns(2)

    status_color_map = {
        "PRODUCTION / FROZEN": "#1565C0",  # Dark Blue
        "REFERENCE": "#757575",            # Neutral Gray
        "ABLATION": "#FB8C00",             # Amber / Orange
    }

    with col1:
        fig_r2 = px.bar(
            leaderboard_df,
            x="Model Architecture",
            y="R² Score",
            color="Status",
            color_discrete_map=status_color_map,
            text="R² Score",
            title="Model Accuracy (R² Score)",
        )
        fig_r2.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_r2.update_yaxes(range=[0.90, 1.00])
        fig_r2.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_r2, use_container_width=True)

    with col2:
        fig_rmse = px.bar(
            leaderboard_df,
            x="Model Architecture",
            y="RMSE (tons)",
            color="Status",
            color_discrete_map=status_color_map,
            text="RMSE (tons)",
            title="Prediction Error (RMSE)",
        )
        fig_rmse.update_traces(texttemplate="%{text:.1f} t", textposition="outside")
        fig_rmse.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_rmse, use_container_width=True)

    st.markdown("---")

    # 3. Interactive Prediction Sandbox
    st.subheader("🔮 Live Voyage Fuel Prediction Sandbox")
    st.markdown("Run real-time inference using the frozen production model (**Physics + Residual QIFCP**).")

    with st.form("prediction_form"):
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            v_type = st.selectbox("Vessel Type", ["Bulk Carrier", "Container", "Tanker", "General Cargo"])
            fuel_type = st.selectbox("Fuel Type", ["Diesel", "LNG", "Methanol", "Hydrogen", "Ammonia", "ShorePower"])
            distance_nm = st.number_input("Voyage Distance (nm)", min_value=100.0, max_value=20000.0, value=1200.0, step=50.0)
        with pcol2:
            speed_knots = st.number_input("Speed (knots)", min_value=8.0, max_value=25.0, value=14.0, step=0.5)
            cargo_tons = st.number_input("Cargo Payload (metric tons)", min_value=1000.0, max_value=300000.0, value=35000.0, step=1000.0)
            vessel_dwt = st.number_input("Vessel DWT", min_value=2000.0, max_value=350000.0, value=45000.0, step=1000.0)
        with pcol3:
            weather_factor = st.slider("Weather Severity Factor", min_value=1.0, max_value=1.5, value=1.05, step=0.01)
            sea_state = st.slider("Sea State (Beaufort)", min_value=0, max_value=7, value=3, step=1)
            submitted = st.form_submit_button("⚡ Predict Fuel Consumption", use_container_width=True)

    if submitted:
        try:
            fuel_service = get_fuel_prediction_service()
            hours_at_sea = distance_nm / max(speed_knots, 1.0)
            record = VoyageRecord(
                voyage_id="SANDBOX-001",
                vessel_id="VSL-SANDBOX",
                vessel_type=v_type,
                vessel_dwt=vessel_dwt,
                cargo_tons=cargo_tons,
                distance_nm=distance_nm,
                speed_knots=speed_knots,
                hours_at_sea=hours_at_sea,
                fuel_type=fuel_type,
                weather_factor=weather_factor,
                sea_state=sea_state,
                data_source="sandbox",
                is_synthetic=True,
                fuel_consumption=None,
                co2_emissions=None,
            )

            pred = fuel_service.predict(record)
            pred_fuel = pred.predicted_fuel_consumption

            res1, res2, res3 = st.columns(3)
            with res1:
                st.metric("Predicted Fuel", f"{pred_fuel:.2f} metric tons")
            with res2:
                tons_per_day = pred_fuel / (hours_at_sea / 24.0) if hours_at_sea > 0 else 0.0
                st.metric("Consumption Rate", f"{tons_per_day:.2f} t/day")
            with res3:
                model_name = pred.model_name
                st.metric("Model Deployed", model_name)

        except Exception as exc:
            st.error(f"Inference error: {exc}")
