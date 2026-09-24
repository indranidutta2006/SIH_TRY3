"""Page 4: Statutory Decarbonization & Compliance (IMO CII & FuelEU Maritime)."""

import streamlit as st

from contracts.exceptions import ComplianceError, DataValidationError
from src.compliance.compliance_engine import MaritimeComplianceEngine


def render_compliance_page() -> None:
    """Render Statutory Compliance & Environmental Decarbonization page."""
    st.title("⚖️ Statutory Decarbonization & Regulatory Compliance")
    st.markdown(
        """
        Automated compliance auditing enforcing **IMO Carbon Intensity Indicator (CII)** ratings
        and **European Union FuelEU Maritime** statutory financial penalties.
        """
    )

    compliance_engine = MaritimeComplianceEngine()

    tab_cii, tab_fueleu = st.tabs(["🏛️ IMO Carbon Intensity Indicator (CII)", "🇪🇺 EU FuelEU Maritime Statutory Penalties"])

    with tab_cii:
        st.subheader("IMO CII Letter Rating Calculator (MEPC.353(78) G2 & MEPC.400(83))")
        col1, col2 = st.columns(2)
        with col1:
            v_type = st.selectbox(
                "Vessel Category",
                [
                    "Bulk Carrier",
                    "Container",
                    "Tanker",
                    "General Cargo",
                    "LNG Carrier",
                    "Ro-Ro Cargo Ship",
                    "Ro-Ro Vehicle Carrier",
                    "Ro-Ro Passenger Ship",
                    "Gas Carrier",
                ],
                key="cii_type",
            )
            is_gt_vessel = any(k in v_type.lower() for k in ["ro-ro", "roro", "passenger", "vehicle"])
            cap_type_choice = st.radio(
                "Statutory Capacity Metric (MEPC.353(78) G2)",
                ["DWT (Deadweight Tonnage)", "GT (Gross Tonnage)"],
                index=1 if is_gt_vessel else 0,
                horizontal=True,
                key=f"cap_type_{v_type}",
                help="Ro-Ro and Passenger vessels use Gross Tonnage (GT). Cargo, bulk, tankers, and LNG carriers use DWT.",
            )
            cap_code = "GT" if "GT" in cap_type_choice else "DWT"
            statutory_required_metric = "GT" if is_gt_vessel else "DWT"
            if cap_code != statutory_required_metric:
                st.warning(
                    f"⚠️ **Incompatible Capacity Unit:** IMO Resolution MEPC.353(78) mandates **{statutory_required_metric}** "
                    f"for '{v_type}'. Submitting with {cap_code} will be rejected as a statutory mismatch."
                )

            cap_val = st.number_input(
                f"Vessel Capacity ({cap_code})",
                min_value=1000.0,
                max_value=400000.0,
                value=65000.0,
                step=1000.0,
                key="cii_capacity",
                help=f"Operational capacity in {cap_code} as required under IMO Resolution MEPC.353(78).",
            )
            dist = st.number_input("Annual Distance (nm)", min_value=500.0, max_value=150000.0, value=45000.0, step=500.0, key="cii_dist")
        with col2:
            co2 = st.number_input("Annual CO₂ Emissions (metric tons)", min_value=10.0, max_value=500000.0, value=12500.0, step=50.0, key="cii_co2")
            year = st.selectbox("Compliance Assessment Year", [2023, 2024, 2025, 2026, 2027, 2030], index=2, key="cii_year")

        if st.button("Evaluate IMO CII Rating", use_container_width=True):
            try:
                res = compliance_engine.evaluate_cii(
                    vessel_type=v_type,
                    capacity=cap_val,
                    capacity_type=cap_code,
                    vessel_dwt=cap_val if cap_code == "DWT" else None,
                    vessel_gt=cap_val if cap_code == "GT" else None,
                    annual_distance_nm=dist,
                    annual_co2_tons=co2,
                    year=year,
                )
            except (DataValidationError, ComplianceError) as e:
                st.error(f"❌ Statutory Error: {e.message}")
                return

            grade_color = {
                "A": "green",
                "B": "blue",
                "C": "orange",
                "D": "purple",
                "E": "red",
            }.get(res.cii_rating, "gray")

            rcol1, rcol2, rcol3 = st.columns(3)
            with rcol1:
                st.markdown(f"### Rating: :{grade_color}[Grade {res.cii_rating}]")
            with rcol2:
                metric_unit = res.capacity_metric.lower()
                st.metric("Attained CII", f"{res.attained_cii:.2f} gCO₂/{metric_unit}·nm")
            with rcol3:
                st.metric("Required Target CII", f"{res.required_cii:.2f} gCO₂/{metric_unit}·nm")

            d1, d2, d3, d4 = res.rating_boundaries
            st.caption(
                f"IMO MEPC.353(78) G2 Reference Branch: a={res.reference_line_a}, c={res.reference_line_c} | "
                f"Statutory Metric: {res.capacity_metric} | CII Ratio: {res.cii_ratio:.3f} | Compliance Status: {res.compliance_status}"
            )
            st.caption(
                f"IMO MEPC.354(78) G4 Rating Boundaries: "
                f"Grade A ≤ {d1:.2f} | Grade B ≤ {d2:.2f} | Grade C ≤ {d3:.2f} | Grade D ≤ {d4:.2f} | Grade E > {d4:.2f}"
            )

    with tab_fueleu:
        st.subheader("EU FuelEU Maritime GHG-Intensity Compliance & Penalty Estimator (Regulation (EU) 2023/1805)")
        st.caption(
            "ℹ️ **Estimator Scope:** Evaluates Article 4 Well-to-Wake GHG intensity targets and Annex IV / Article 23 "
            "statutory penalties, including the Article 23(2) consecutive-deficit multiplier $[1 + (n - 1) / 10]$. "
            "Does not simulate Article 5 RFNBO quotas, Article 6 Onshore Power Supply (OPS) port mandates, or pooling/banking flexibilities."
        )
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            fe_year = st.selectbox("Assessment Year", [2025, 2030, 2035, 2040, 2045, 2050], index=0, key="fe_year")
            ghg_intensity = st.number_input(
                "Operational GHG Intensity (gCO₂eq / MJ)",
                min_value=0.0,
                max_value=120.0,
                value=91.0,
                step=0.5,
                key="fe_ghg",
                help="Reference baseline for marine diesel is 91.16 gCO2eq/MJ.",
            )
        with fcol2:
            energy_mj = st.number_input(
                "Total On-board Energy Consumed (MJ)",
                min_value=1000.0,
                max_value=1e10,
                value=50000000.0,
                step=1000000.0,
                key="fe_energy",
            )
        with fcol3:
            fe_consecutive = st.number_input(
                "Consecutive Deficit Periods (n)",
                min_value=1,
                max_value=10,
                value=1,
                step=1,
                key="fe_consecutive",
                help="Article 23(2) statutory multiplier: 1 + (n - 1) / 10 applied when deficit persists across consecutive years.",
            )

        if st.button("Calculate FuelEU Compliance & Penalties", use_container_width=True):
            fe_res = compliance_engine.evaluate_fueleu(
                ghg_intensity=ghg_intensity,
                energy_used_mj=energy_mj,
                year=fe_year,
                consecutive_deficit_periods=fe_consecutive,
            )

            pcol1, pcol2, pcol3 = st.columns(3)
            with pcol1:
                status_text = "PASSED" if fe_res.fueleu_pass else "VIOLATION (PENALTY ACCRUED)"
                color = "green" if fe_res.fueleu_pass else "red"
                st.markdown(f"### Status: :{color}[{status_text}]")
            with pcol2:
                penalty_eur = fe_res.penalty_eur
                st.metric("Estimated Penalty (€)", f"€{penalty_eur:,.2f}")
            with pcol3:
                st.metric("Compliance Status", fe_res.compliance_status)

            if fe_res.penalty_multiplier > 1.0:
                st.caption(
                    f"⚠️ Article 23(2) Consecutive Deficit Multiplier Active: "
                    f"**{fe_res.penalty_multiplier:.1f}×** applied for {fe_res.consecutive_deficit_periods} consecutive reporting periods."
                )
