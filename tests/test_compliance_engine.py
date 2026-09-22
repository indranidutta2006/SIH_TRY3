"""Deterministic unit tests for MaritimeComplianceEngine.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Validates IMO CII rating calculations and EU FuelEU Maritime compliance penalties.
"""

import pytest

from contracts.exceptions import ComplianceError, DataValidationError
from contracts.interfaces import ComplianceEngine
from contracts.schemas import ComplianceResult
from src.compliance import MaritimeComplianceEngine


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
