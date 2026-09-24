"""Page 4: Statutory Decarbonization & Compliance (IMO CII & FuelEU Maritime)."""

import streamlit as st

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
        st.subheader("IMO CII Letter Rating Calculator (MEPC.337(76))")
        col1, col2 = st.columns(2)
        with col1:
            v_type = st.selectbox("Vessel Category", ["Bulk Carrier", "Container", "Tanker", "RoRo", "LNG Carrier"], key="cii_type")
            dwt = st.number_input("Vessel DWT", min_value=1000.0, max_value=400000.0, value=65000.0, step=1000.0, key="cii_dwt")
            dist = st.number_input("Annual Distance (nm)", min_value=500.0, max_value=150000.0, value=45000.0, step=500.0, key="cii_dist")
        with col2:
            co2 = st.number_input("Annual CO₂ Emissions (metric tons)", min_value=10.0, max_value=500000.0, value=12500.0, step=50.0, key="cii_co2")
            year = st.selectbox("Compliance Assessment Year", [2023, 2024, 2025, 2026, 2027, 2030], index=2, key="cii_year")

        if st.button("Evaluate IMO CII Rating", use_container_width=True):
            res = compliance_engine.evaluate_cii(
                vessel_type=v_type,
                vessel_dwt=dwt,
                annual_distance_nm=dist,
                annual_co2_tons=co2,
                year=year,
            )

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
                st.metric("Attained CII", f"{res.attained_cii:.2f} gCO₂/dwt·nm")
            with rcol3:
                st.metric("Required Target CII", f"{res.required_cii:.2f} gCO₂/dwt·nm")

            st.caption(f"CII Ratio (Attained / Required): {res.cii_ratio:.3f} | Compliance Status: {res.compliance_status}")

    with tab_fueleu:
        st.subheader("EU FuelEU Maritime GHG Intensity & Penalty Accounting (Regulation (EU) 2023/1805)")
        fcol1, fcol2 = st.columns(2)
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

        if st.button("Calculate FuelEU Compliance & Penalties", use_container_width=True):
            fe_res = compliance_engine.evaluate_fueleu(
                ghg_intensity=ghg_intensity,
                energy_used_mj=energy_mj,
                year=fe_year,
            )

            pcol1, pcol2, pcol3 = st.columns(3)
            with pcol1:
                status_text = "PASSED" if fe_res.fueleu_pass else "VIOLATION (PENALTY ACCRUED)"
                color = "green" if fe_res.fueleu_pass else "red"
                st.markdown(f"### Status: :{color}[{status_text}]")
            with pcol2:
                penalty_eur = fe_res.penalty_eur
                st.metric("FuelEU Penalty (€)", f"€{penalty_eur:,.2f}")
            with pcol3:
                st.metric("Compliance Status", fe_res.compliance_status)
