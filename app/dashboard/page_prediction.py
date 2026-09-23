"""Page 2: Fuel Consumption Prediction & Quantum-Inspired Modeling (QIFCP)."""

import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

from contracts.schemas import VoyageRecord
from src.prediction.model_manager import ProductionModelManager


def render_prediction_page() -> None:
    """Render Prediction & QIFCP comparative analysis page."""
    st.title("🧠 Fuel Consumption Prediction & Quantum Modeling")
    st.markdown(
        """
        **Objective 1:** High-precision fuel consumption estimation across nonlinear hydrodynamic drag,
        weather resistance, and payload states using classical ensembles and Quantum-Inspired Modeling (**QIFCP**).
        """
    )

    report_path = Path("outputs/reports/prediction_benchmark.json")
    if not report_path.exists():
        st.warning(f"Prediction report `{report_path}` not found. Please run the build script.")
        return

    data = json.loads(report_path.read_text(encoding="utf-8"))
    models = data.get("models", {})

    # 1. 4-Model Comparison Table
    st.subheader("Model Benchmark Leaderboard (5,000 Verified Test Voyages)")
    rows = []
    for name, m in models.items():
        rows.append(
            {
                "Model Architecture": m.get("model_name", name),
                "R² Score": m.get("r2", 0.0),
                "RMSE (tons)": m.get("rmse", 0.0),
                "MAE (tons)": m.get("mae", 0.0),
                "MAPE (%)": m.get("mape", 0.0),
                "Inference (ms / 100)": m.get("infer_ms_per_100_samples", 0.0),
                "Fit Time (s)": m.get("fit_time_seconds", 0.0),
            }
        )

    leaderboard_df = pd.DataFrame(rows).sort_values(by="R² Score", ascending=False)
    st.dataframe(leaderboard_df, use_container_width=True)

    # ── Per-Source Normalized Error Subsection ──────────────────────────────
    st.subheader("Per-Source Normalized Error")

    norm_rows = []
    for name, m in models.items():
        by_source = m.get("by_source", {})
        for src, sm in by_source.items():
            norm_rows.append(
                {
                    "Model": m.get("model_name", name),
                    "Source": src,
                    "NRMSE_mean": sm.get("nrmse_mean", None),
                    "MAPE_pct": sm.get("mape_pct", None),
                    "R²": sm.get("r2", None),
                }
            )

    if norm_rows:
        norm_df = pd.DataFrame(norm_rows)

        def _color_nrmse(val):
            """Green if NRMSE < 1.0 (error < target mean), red if > 1.0."""
            if val is None:
                return ""
            if val < 1.0:
                return "color: #2e7d32"  # green
            return "color: #c62828"  # red

        styled = (
            norm_df.style
            .format({"NRMSE_mean": "{:.4f}", "MAPE_pct": "{:.2f}%", "R²": "{:.4f}"}, na_rep="—")
            .map(_color_nrmse, subset=["NRMSE_mean"])
        )
        st.dataframe(styled, use_container_width=True)

        st.caption(
            "FuelCast's low raw RMSE (56.50 t) partly reflects its smaller target scale "
            "(mean ≈ 57 t vs. ≈ 871 t for Mock); normalized metrics (NRMSE, MAPE) should be "
            "used for fair cross-source comparison."
        )

        st.info(
            "**Cross-source trade-off:** Tree-based models (RF, HistGBDT) achieve excellent accuracy on Mock "
            "(NRMSE 0.11) but suffer severe leaf shrinkage on FuelCast's low-rate regime (NRMSE 2.66–2.88). "
            "QIFCP's continuous quantum projection is more stable across sources (FuelCast NRMSE 0.99) but "
            "less accurate than trees on Mock (NRMSE 0.34 vs. 0.11). Neither model class is categorically "
            "superior — the choice depends on whether source-balanced fairness or peak single-source accuracy "
            "is prioritised."
        )
    else:
        st.info("Per-source normalized metrics not available. Re-run the benchmark with `--target-mode rate`.")

    # 2. Performance Comparison Charts
    col1, col2 = st.columns(2)
    with col1:
        fig_r2 = px.bar(
            leaderboard_df,
            x="Model Architecture",
            y="R² Score",
            color="R² Score",
            color_continuous_scale="Blues",
            title="Model Accuracy (R² Score)",
        )
        st.plotly_chart(fig_r2, use_container_width=True)

    with col2:
        fig_rmse = px.bar(
            leaderboard_df,
            x="Model Architecture",
            y="RMSE (tons)",
            color="RMSE (tons)",
            color_continuous_scale="Reds_r",
            title="Root Mean Squared Error (Lower is Better)",
        )
        st.plotly_chart(fig_rmse, use_container_width=True)

    st.markdown("---")

    # 3. Interactive Prediction Sandbox
    st.subheader("🔮 Live Voyage Fuel Prediction Sandbox")
    st.markdown("Run real-time inference using the verified production model (`HistGradientBoosting`).")

    with st.form("prediction_form"):
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            v_type = st.selectbox("Vessel Type", ["Bulk Carrier", "Container", "Tanker", "RoRo", "LNG Carrier"])
            fuel_type = st.selectbox("Fuel Type", ["Diesel", "LNG", "Methanol"])
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
            manager = ProductionModelManager()
            prod_model = manager.get_best_model()

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

            preds = prod_model.predict([record])
            pred_fuel = preds[0].predicted_fuel_consumption

            res1, res2, res3 = st.columns(3)
            with res1:
                st.metric("Predicted Fuel", f"{pred_fuel:.2f} metric tons")
            with res2:
                tons_per_day = pred_fuel / (hours_at_sea / 24.0)
                st.metric("Consumption Rate", f"{tons_per_day:.2f} t/day")
            with res3:
                model_name = preds[0].model_name
                st.metric("Model Deployed", model_name)

        except Exception as exc:
            st.error(f"Inference error: {exc}")
