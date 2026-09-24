"""Deterministic unit tests for MaritimeComplianceEngine.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Validates IMO CII rating calculations and EU FuelEU Maritime compliance penalties.
"""

import pytest

from contracts.exceptions import ComplianceError, DataValidationError
from contracts.interfaces import ComplianceEngine
from contracts.schemas import ComplianceResult
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
    assert res_a.fueleu_pass is True

    # Moderate baseline performance -> Grade C (attained ratio ~0.98 in [0.94, 1.06])
    res_c = engine.evaluate_cii(co2_emissions=1550.0, cargo_tons=dwt, distance_nm=distance, year=year)
    assert res_c.cii_rating == "C"
    assert res_c.fueleu_pass is True

    # Highly polluting -> Grade E (attained ratio ~1.9 > 1.19)
    res_e = engine.evaluate_cii(co2_emissions=3000.0, cargo_tons=dwt, distance_nm=distance, year=year)
    assert res_e.cii_rating == "E"
    assert res_e.fueleu_pass is False


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


