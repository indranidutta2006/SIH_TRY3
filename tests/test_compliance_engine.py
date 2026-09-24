"""Deterministic unit tests for MaritimeComplianceEngine.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Validates IMO CII rating calculations and EU FuelEU Maritime compliance penalties.
"""

import pytest

from contracts.exceptions import ComplianceError, DataValidationError
from contracts.interfaces import ComplianceEngine
from contracts.schemas import (
    CIIResult,
    ComplianceAssessment,
    ComplianceResult,
    FuelEUResult,
)
from src.compliance import CII_Z_FACTORS, MaritimeComplianceEngine


def test_compliance_engine_implements_contract() -> None:
    """Verify MaritimeComplianceEngine fulfills contracts.interfaces.ComplianceEngine."""
    engine = MaritimeComplianceEngine()
    assert isinstance(engine, ComplianceEngine)


def test_cii_rating_bands() -> None:
    """Verify IMO CII operational letter ratings from A through E."""
    engine = MaritimeComplianceEngine()
    dwt = 80000.0
    distance = 5000.0
    year = 2024

    # Superior performance -> Grade A (attained ratio ~0.51 <= 0.83)
    res_a = engine.evaluate_cii(co2_emissions=800.0, cargo_tons=dwt, distance_nm=distance, year=year)
    assert res_a.cii_rating == "A"
    assert res_a.compliance_status == "COMPLIANT"
    assert res_a.fueleu_pass is None  # FuelEU is not evaluated during CII assessment

    # Moderate baseline performance -> Grade C (attained ratio ~0.98 in [0.94, 1.06])
    res_c = engine.evaluate_cii(co2_emissions=1550.0, cargo_tons=dwt, distance_nm=distance, year=year)
    assert res_c.cii_rating == "C"
    assert res_c.compliance_status == "COMPLIANT"
    assert res_c.fueleu_pass is None

    # Highly polluting -> Grade E (attained ratio ~1.9 > 1.19)
    res_e = engine.evaluate_cii(co2_emissions=3000.0, cargo_tons=dwt, distance_nm=distance, year=year)
    assert res_e.cii_rating == "E"
    assert res_e.compliance_status == "NON_COMPLIANT"
    assert res_e.fueleu_pass is None  # CII failure does NOT falsely imply FuelEU failure


def test_fueleu_pass_on_compliant_ghg_intensity() -> None:
    """Verify zero penalty and pass status when GHG intensity is below FuelEU threshold."""
    engine = MaritimeComplianceEngine()
    # 2025 target is 89.3368 gCO2eq/MJ
    # Green fuel with GHG intensity 60.0 gCO2eq/MJ
    res = engine.evaluate_fueleu(ghg_intensity=60.0, energy_used_mj=1000000.0, year=2025)

    assert isinstance(res, ComplianceResult)
    assert res.fueleu_pass is True
    assert res.penalty_eur == 0.0
    assert res.compliance_score == 0.0


def test_fueleu_penalty_on_excess_ghg_intensity() -> None:
    """Verify positive financial penalty when GHG intensity breaches FuelEU target."""
    engine = MaritimeComplianceEngine()
    # 2025 target is 89.3368 gCO2eq/MJ
    # Heavy fuel with GHG intensity 95.0 gCO2eq/MJ
    energy_mj = 50000000.0  # 50,000 GJ
    res = engine.evaluate_fueleu(ghg_intensity=95.0, energy_used_mj=energy_mj, year=2025)

    assert isinstance(res, ComplianceResult)
    assert res.fueleu_pass is False
    assert res.penalty_eur > 0.0
    assert res.compliance_score > 0.0


def test_compliance_validation_errors() -> None:
    """Verify invalid inputs raise DataValidationError or ComplianceError."""
    engine = MaritimeComplianceEngine()

    with pytest.raises(DataValidationError):
        engine.evaluate_cii(co2_emissions=-10.0, cargo_tons=50000.0, distance_nm=1000.0, year=2024)

    with pytest.raises(DataValidationError):
        engine.evaluate_cii(co2_emissions=10.0, cargo_tons=-50000.0, distance_nm=1000.0, year=2024)

    with pytest.raises(ComplianceError):
        engine.evaluate_cii(co2_emissions=10.0, cargo_tons=50000.0, distance_nm=1000.0, year=2015)

    with pytest.raises(DataValidationError):
        engine.evaluate_fueleu(ghg_intensity=-5.0, energy_used_mj=1000.0, year=2025)


def test_cii_dashboard_call_pattern() -> None:
    """Verify evaluate_cii works with dashboard keyword arguments."""
    engine = MaritimeComplianceEngine()
    res = engine.evaluate_cii(
        vessel_type="Bulk Carrier",
        vessel_dwt=65000.0,
        annual_distance_nm=45000.0,
        annual_co2_tons=12500.0,
        year=2025,
    )
    assert isinstance(res, ComplianceResult)
    assert res.cii_rating in ("A", "B", "C", "D", "E")
    assert res.attained_cii > 0.0
    assert res.required_cii > 0.0
    assert res.cii_ratio > 0.0
    assert res.compliance_status in ("COMPLIANT", "NON_COMPLIANT")
    assert res.penalty_eur == 0.0


def test_compliance_result_canonical_schema_fields() -> None:
    """Verify ComplianceResult contains all canonical attributes for both CII and FuelEU."""
    engine = MaritimeComplianceEngine()
    res_cii = engine.evaluate_cii(
        co2_emissions=1000.0,
        cargo_tons=50000.0,
        distance_nm=3000.0,
        year=2024,
    )
    assert hasattr(res_cii, "cii_rating")
    assert hasattr(res_cii, "attained_cii")
    assert hasattr(res_cii, "required_cii")
    assert hasattr(res_cii, "cii_ratio")
    assert hasattr(res_cii, "fueleu_pass")
    assert hasattr(res_cii, "fueleu_target")
    assert hasattr(res_cii, "ghg_intensity")
    assert hasattr(res_cii, "penalty_eur")
    assert hasattr(res_cii, "compliance_status")
    assert hasattr(res_cii, "compliance_score")

    fe_res = engine.evaluate_fueleu(
        ghg_intensity=95.0,
        energy_used_mj=50000000.0,
        year=2025,
    )
    assert fe_res.compliance_status == "NON_COMPLIANT"
    assert fe_res.penalty_eur > 0.0
    assert fe_res.fueleu_target > 0.0
    assert fe_res.ghg_intensity == 95.0


@pytest.mark.parametrize(
    "year, expected_z",
    [
        (2023, 0.050),
        (2024, 0.070),
        (2025, 0.090),
        (2026, 0.110),
        (2027, 0.13625),
        (2028, 0.16250),
        (2029, 0.18875),
        (2030, 0.21500),
    ],
)
def test_cii_z_factors_mepc_400_83(year: int, expected_z: float) -> None:
    """Verify statutory annual reduction factor Z conforming to IMO Resolution MEPC.400(83)."""
    engine = MaritimeComplianceEngine()
    assert engine.get_cii_z_factor(year) == pytest.approx(expected_z, abs=1e-5)

    # Verify required CII calculation respects the statutory Z-factor
    dwt = 70000.0
    baseline_cii = 4745.0 * (dwt ** -0.622)
    expected_required = baseline_cii * (1.0 - expected_z)

    res = engine.evaluate_cii(
        co2_emissions=1200.0,
        cargo_tons=dwt,
        distance_nm=4000.0,
        year=year,
    )
    assert res.required_cii == pytest.approx(expected_required, rel=1e-4)


def test_cii_future_years_2027_through_2030() -> None:
    """Verify explicit compliance evaluations for assessment years 2027, 2028, 2029, and 2030."""
    engine = MaritimeComplianceEngine()
    dwt = 65000.0
    dist = 45000.0
    co2 = 12500.0
    baseline_cii = 4745.0 * (dwt ** -0.622)

    z_targets = {
        2027: 0.13625,
        2028: 0.16250,
        2029: 0.18875,
        2030: 0.21500,
    }

    for y, z in z_targets.items():
        res = engine.evaluate_cii(
            vessel_type="Bulk Carrier",
            vessel_dwt=dwt,
            annual_distance_nm=dist,
            annual_co2_tons=co2,
            year=y,
        )
        expected_req = round(baseline_cii * (1.0 - z), 4)
        assert res.required_cii == pytest.approx(expected_req, abs=1e-3)
        expected_ratio = round(res.attained_cii / res.required_cii, 4)
        assert res.cii_ratio == pytest.approx(expected_ratio, abs=1e-3)


@pytest.mark.parametrize("invalid_year", [2015, 2020, 2022, 2031, 2035])
def test_cii_unsupported_years_raise_compliance_error(invalid_year: int) -> None:
    """Verify that reporting years outside the statutory MEPC.400(83) range (2023–2030) are rejected."""
    engine = MaritimeComplianceEngine()
    with pytest.raises(ComplianceError):
        engine.get_cii_z_factor(invalid_year)

    with pytest.raises(ComplianceError):
        engine.evaluate_cii(
            co2_emissions=1000.0,
            cargo_tons=50000.0,
            distance_nm=3000.0,
            year=invalid_year,
        )


def test_cii_general_cargo_mepc353_78_split() -> None:
    """Verify General Cargo split curve at 20,000 DWT under Resolution MEPC.353(78)."""
    engine = MaritimeComplianceEngine()

    # Large general cargo (>= 20,000 DWT): a=31948, c=0.792
    a_lg, c_lg, eff_lg, metric_lg = engine.resolve_cii_reference_line("General Cargo", 25000.0)
    assert a_lg == 31948.0
    assert c_lg == 0.792
    assert metric_lg == "DWT"

    # Small general cargo (< 20,000 DWT): a=588, c=0.3885
    a_sm, c_sm, eff_sm, metric_sm = engine.resolve_cii_reference_line("General Cargo", 15000.0)
    assert a_sm == 588.0
    assert c_sm == 0.3885
    assert metric_sm == "DWT"


def test_cii_lng_carrier_mepc353_78_split() -> None:
    """Verify LNG Carrier tiered curves under Resolution MEPC.353(78)."""
    engine = MaritimeComplianceEngine()

    # Large LNG carrier (>= 100,000 DWT): a=9.827, c=0.0
    a_lg, c_lg, _, metric_lg = engine.resolve_cii_reference_line("LNG Carrier", 120000.0)
    assert a_lg == 9.827
    assert c_lg == 0.0
    assert metric_lg == "DWT"

    # Medium LNG carrier (65,000 - 100,000 DWT): a=1.4479e14, c=2.673
    a_md, c_md, _, metric_md = engine.resolve_cii_reference_line("LNG Carrier", 80000.0)
    assert a_md == pytest.approx(1.4479e14, rel=1e-5)
    assert c_md == 2.673
    assert metric_md == "DWT"

    # Small LNG carrier (< 65,000 DWT): a=1.4779e14, c=2.673
    a_sm, c_sm, _, metric_sm = engine.resolve_cii_reference_line("LNG Carrier", 50000.0)
    assert a_sm == pytest.approx(1.4779e14, rel=1e-5)
    assert c_sm == 2.673
    assert metric_sm == "DWT"


def test_cii_roro_mepc353_78_gt_metric() -> None:
    """Verify Ro-Ro categories use Gross Tonnage (GT) and correct statutory curves."""
    engine = MaritimeComplianceEngine()

    # Ro-Ro Vehicle Carrier >= 30,000 GT: a=3627, c=0.590, metric=GT
    a_vc_lg, c_vc_lg, _, m_vc_lg = engine.resolve_cii_reference_line("Ro-Ro Vehicle Carrier", 45000.0)
    assert a_vc_lg == 3627.0
    assert c_vc_lg == 0.590
    assert m_vc_lg == "GT"

    # Ro-Ro Vehicle Carrier < 30,000 GT: a=330, c=0.329, metric=GT
    a_vc_sm, c_vc_sm, _, m_vc_sm = engine.resolve_cii_reference_line("Ro-Ro Vehicle Carrier", 20000.0)
    assert a_vc_sm == 330.0
    assert c_vc_sm == 0.329
    assert m_vc_sm == "GT"

    # Ro-Ro Cargo Ship: a=1967, c=0.485, metric=GT
    a_ro, c_ro, _, m_ro = engine.resolve_cii_reference_line("Ro-Ro Cargo Ship", 25000.0)
    assert a_ro == 1967.0
    assert c_ro == 0.485
    assert m_ro == "GT"

    # Ro-Ro Passenger Ship: a=2023, c=0.460, metric=GT
    a_pax, c_pax, _, m_pax = engine.resolve_cii_reference_line("Ro-Ro Passenger Ship", 35000.0)
    assert a_pax == 2023.0
    assert c_pax == 0.460
    assert m_pax == "GT"


def test_cii_bulk_carrier_capacity_cap_at_279k() -> None:
    """Verify Bulk Carrier calculation capacity is capped at 279,000 DWT per MEPC.353(78)."""
    engine = MaritimeComplianceEngine()
    a, c, eff_cap, metric = engine.resolve_cii_reference_line("Bulk Carrier", 350000.0)
    assert eff_cap == 279000.0
    assert a == 4745.0
    assert c == 0.622
    assert metric == "DWT"


def test_cii_evaluate_with_gt_capacity() -> None:
    """Verify evaluate_cii correctly handles GT capacity basis for Ro-Ro ships."""
    engine = MaritimeComplianceEngine()
    res = engine.evaluate_cii(
        vessel_type="Ro-Ro Cargo Ship",
        capacity=30000.0,
        capacity_type="GT",
        annual_distance_nm=40000.0,
        annual_co2_tons=8000.0,
        year=2025,
    )
    assert res.capacity_metric == "GT"
    assert res.reference_line_a == 1967.0
    assert res.reference_line_c == 0.485

    # Attained CII = (8000 * 1e6) / (30000 * 40000) = 6.6667 gCO2 / GT*nm
    expected_attained = (8000.0 * 1e6) / (30000.0 * 40000.0)
    assert res.attained_cii == pytest.approx(expected_attained, abs=1e-3)


def test_cii_rating_boundaries_mepc354_78() -> None:
    """Verify Resolution MEPC.354(78) G4 ship-type specific rating boundary vectors."""
    engine = MaritimeComplianceEngine()

    # Bulk Carrier: 0.86, 0.94, 1.06, 1.18
    assert engine.resolve_cii_rating_boundaries("Bulk Carrier", 60000.0) == (0.86, 0.94, 1.06, 1.18)

    # Tanker: 0.82, 0.93, 1.08, 1.28
    assert engine.resolve_cii_rating_boundaries("Tanker", 50000.0) == (0.82, 0.93, 1.08, 1.28)

    # Containership: 0.83, 0.94, 1.07, 1.19
    assert engine.resolve_cii_rating_boundaries("Container", 40000.0) == (0.83, 0.94, 1.07, 1.19)

    # LNG Carrier (>= 100k DWT): 0.89, 0.98, 1.06, 1.13
    assert engine.resolve_cii_rating_boundaries("LNG Carrier", 120000.0) == (0.89, 0.98, 1.06, 1.13)

    # LNG Carrier (< 100k DWT): 0.78, 0.92, 1.10, 1.37
    assert engine.resolve_cii_rating_boundaries("LNG Carrier", 80000.0) == (0.78, 0.92, 1.10, 1.37)

    # Ro-Ro Vehicle Carrier: 0.86, 0.94, 1.06, 1.16
    assert engine.resolve_cii_rating_boundaries("Ro-Ro Vehicle Carrier", 45000.0) == (0.86, 0.94, 1.06, 1.16)

    # Ro-Ro Cargo Ship: 0.76, 0.89, 1.08, 1.27
    assert engine.resolve_cii_rating_boundaries("Ro-Ro Cargo Ship", 25000.0) == (0.76, 0.89, 1.08, 1.27)

    # Ro-Ro Passenger Ship: 0.76, 0.92, 1.14, 1.30
    assert engine.resolve_cii_rating_boundaries("Ro-Ro Passenger Ship", 35000.0) == (0.76, 0.92, 1.14, 1.30)


def test_cii_rating_classification_ship_type_differences() -> None:
    """Verify that ship-type-specific boundaries correctly classify edge cases."""
    engine = MaritimeComplianceEngine()

    # 1. Tanker with ratio 1.25:
    # Under universal threshold (1.19), this would be E. Under Tanker G4 vector (d4=1.28), this is Grade D.
    # Compute emissions to get ratio ~1.25
    dwt = 50000.0
    dist = 40000.0
    baseline = 5247.0 * (dwt ** -0.610)
    z = engine.get_cii_z_factor(2025)
    req = baseline * (1.0 - z)
    target_attained = req * 1.25
    co2 = (target_attained * dwt * dist) / 1e6

    res_tanker = engine.evaluate_cii(
        vessel_type="Tanker",
        capacity=dwt,
        annual_distance_nm=dist,
        annual_co2_tons=co2,
        year=2025,
    )
    assert res_tanker.cii_ratio == pytest.approx(1.25, abs=0.01)
    assert res_tanker.cii_rating == "D"  # Would have been E under old universal 1.19!

    # 2. LNG Carrier (120,000 DWT) with ratio 0.88:
    # Under universal threshold (0.83), this would be B. Under Large LNG G4 vector (d1=0.89), this is Grade A!
    baseline_lng = 9.827
    req_lng = baseline_lng * (1.0 - z)
    target_attained_lng = req_lng * 0.88
    co2_lng = (target_attained_lng * 120000.0 * dist) / 1e6

    res_lng = engine.evaluate_cii(
        vessel_type="LNG Carrier",
        capacity=120000.0,
        annual_distance_nm=dist,
        annual_co2_tons=co2_lng,
        year=2025,
    )
    assert res_lng.cii_ratio == pytest.approx(0.88, abs=0.01)
    assert res_lng.cii_rating == "A"  # Would have been B under old universal 0.83!


def test_cii_does_not_overload_fueleu_pass() -> None:
    """Verify evaluating CII leaves fueleu_pass as None and does not conflate regulations."""
    engine = MaritimeComplianceEngine()
    # Grade E non-compliant vessel under CII
    res_e = engine.evaluate_cii(
        co2_emissions=3000.0,
        cargo_tons=80000.0,
        distance_nm=5000.0,
        year=2025,
    )
    assert res_e.cii_rating == "E"
    assert res_e.compliance_status == "NON_COMPLIANT"
    # Essential semantic guarantee: CII failure must NOT mark FuelEU as failed!
    assert res_e.fueleu_pass is None
    # Concrete canonical metric is cii_ratio
    assert res_e.cii_ratio > 1.0


def test_decoupled_cii_and_fueleu_result_objects() -> None:
    """Verify assess_cii and assess_fueleu return dedicated, decoupled result objects."""
    engine = MaritimeComplianceEngine()

    # Dedicated CII assessment
    cii_res = engine.assess_cii(
        vessel_type="Bulk Carrier",
        vessel_dwt=75000.0,
        annual_distance_nm=40000.0,
        annual_co2_tons=10000.0,
        year=2025,
    )
    assert isinstance(cii_res, CIIResult)
    assert cii_res.cii_rating in ("A", "B", "C", "D", "E")
    assert cii_res.attained_cii > 0.0
    assert cii_res.required_cii > 0.0
    assert cii_res.cii_ratio > 0.0
    assert isinstance(cii_res.is_compliant, bool)
    assert not hasattr(cii_res, "penalty_eur")
    assert not hasattr(cii_res, "fueleu_pass")

    # Dedicated FuelEU assessment
    fe_res = engine.assess_fueleu(
        ghg_intensity=95.0,
        energy_used_mj=50_000_000.0,
        year=2025,
    )
    assert isinstance(fe_res, FuelEUResult)
    assert fe_res.fueleu_pass is False
    assert fe_res.penalty_eur > 0.0
    assert fe_res.is_compliant is False
    assert not hasattr(fe_res, "cii_rating")
    assert not hasattr(fe_res, "attained_cii")


def test_compliance_assessment_container() -> None:
    """Verify ComplianceAssessment aggregates decoupled CII and FuelEU results cleanly."""
    engine = MaritimeComplianceEngine()

    assessment = engine.assess_compliance(
        cii_params={
            "vessel_type": "Containership",
            "vessel_dwt": 50000.0,
            "annual_distance_nm": 60000.0,
            "annual_co2_tons": 18000.0,
            "year": 2025,
        },
        fueleu_params={
            "ghg_intensity": 70.0,
            "energy_used_mj": 80_000_000.0,
            "year": 2025,
        },
    )
    assert isinstance(assessment, ComplianceAssessment)
    assert isinstance(assessment.cii, CIIResult)
    assert isinstance(assessment.fueleu, FuelEUResult)
    assert assessment.fueleu.fueleu_pass is True
    assert assessment.fueleu.penalty_eur == 0.0
    d = assessment.to_dict()
    assert "cii" in d and "fueleu" in d




