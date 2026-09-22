"""Deterministic unit tests validating contract compliance of MaritimeEmissionEngine.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Verifies full adherence to contracts.interfaces.EmissionEngine and contracts.schemas.EmissionResult,
testing calculate_ttw(), calculate_wtw(), input boundary validation, and lifecycle logical consistency.
"""

import pytest

from contracts.exceptions import DataValidationError
from contracts.interfaces import EmissionEngine
from contracts.schemas import EmissionResult
from src.prediction.emission_engine import (
    CH4_FACTORS,
    CO2_FACTORS,
    N2O_FACTORS,
    WTT_CO2E_FACTORS,
    MaritimeEmissionEngine,
)


def test_maritime_emission_engine_implements_interface() -> None:
    """Ensure MaritimeEmissionEngine strictly implements the abstract contracts.interfaces.EmissionEngine."""
    engine = MaritimeEmissionEngine()
    assert isinstance(engine, EmissionEngine)
    assert hasattr(engine, "calculate_ttw")
    assert hasattr(engine, "calculate_wtw")


def test_calculate_ttw_returns_contract_schema_instance() -> None:
    """Ensure calculate_ttw() returns a direct contracts.schemas.EmissionResult instance."""
    engine = MaritimeEmissionEngine()
    ttw = engine.calculate_ttw(fuel_consumption=100.0, fuel_type="Diesel")

    assert isinstance(ttw, EmissionResult)
    assert type(ttw) is EmissionResult
    assert ttw.fuel_type == "Diesel"
    assert ttw.co2 > 0.0
    assert ttw.ch4 >= 0.0
    assert ttw.n2o >= 0.0
    assert ttw.co2e >= ttw.co2


def test_calculate_wtw_returns_contract_schema_instance() -> None:
    """Ensure calculate_wtw() returns a direct contracts.schemas.EmissionResult instance."""
    engine = MaritimeEmissionEngine()
    wtw = engine.calculate_wtw(fuel_consumption=100.0, fuel_type="Diesel")

    assert isinstance(wtw, EmissionResult)
    assert type(wtw) is EmissionResult
    assert wtw.fuel_type == "Diesel"
    assert wtw.co2 > 0.0
    assert wtw.co2e >= wtw.co2


def test_all_supported_fuels_ttw_and_wtw() -> None:
    """Verify TTW and WTW calculations across all supported marine fuel types."""
    engine = MaritimeEmissionEngine()
    supported_fuels = ["Diesel", "LNG", "Methanol", "Ammonia", "Hydrogen", "ShorePower"]

    for fuel in supported_fuels:
        ttw = engine.calculate_ttw(50.0, fuel)
        wtw = engine.calculate_wtw(50.0, fuel)

        assert ttw.fuel_type == fuel
        assert wtw.fuel_type == fuel
        assert isinstance(ttw, EmissionResult)
        assert isinstance(wtw, EmissionResult)


def test_logical_consistency_wtw_greater_than_or_equal_to_ttw() -> None:
    """Ensure WTW CO2e is strictly >= TTW CO2e for all fuels."""
    engine = MaritimeEmissionEngine()
    non_zero_fuels = ["Diesel", "LNG", "Methanol"]

    for fuel in non_zero_fuels:
        ttw = engine.calculate_ttw(100.0, fuel)
        wtw = engine.calculate_wtw(100.0, fuel)

        # For fossil / hydrocarbon fuels, upstream WTT emissions must increase total lifecycle footprint
        assert wtw.co2e > ttw.co2e, f"WTW CO2e ({wtw.co2e}) must exceed TTW CO2e ({ttw.co2e}) for {fuel}"
        assert wtw.co2 == ttw.co2, f"Direct combustion CO2 must match between WTW ({wtw.co2}) and TTW ({ttw.co2}) for {fuel}"


def test_zero_carbon_fuels_yield_zero_emissions() -> None:
    """Verify zero-carbon power sources (Ammonia, Hydrogen, ShorePower) evaluate to 0.0."""
    engine = MaritimeEmissionEngine()
    zero_carbon_fuels = ["Ammonia", "Hydrogen", "ShorePower"]

    for fuel in zero_carbon_fuels:
        ttw = engine.calculate_ttw(100.0, fuel)
        wtw = engine.calculate_wtw(100.0, fuel)

        assert ttw.co2 == 0.0
        assert ttw.co2e == 0.0
        assert wtw.co2 == 0.0
        assert wtw.co2e == 0.0


def test_negative_fuel_consumption_raises_validation_error() -> None:
    """Ensure negative fuel consumption raises DataValidationError."""
    engine = MaritimeEmissionEngine()

    with pytest.raises(DataValidationError) as exc_ttw:
        engine.calculate_ttw(fuel_consumption=-10.0, fuel_type="Diesel")
    assert "cannot be negative" in str(exc_ttw.value)

    with pytest.raises(DataValidationError) as exc_wtw:
        engine.calculate_wtw(fuel_consumption=-10.0, fuel_type="Diesel")
    assert "cannot be negative" in str(exc_wtw.value)


def test_unsupported_fuel_raises_validation_error() -> None:
    """Ensure unapproved fuel strings raise DataValidationError."""
    engine = MaritimeEmissionEngine()

    with pytest.raises(DataValidationError) as exc_ttw:
        engine.calculate_ttw(fuel_consumption=50.0, fuel_type="CoalTar")
    assert "Unsupported fuel type" in str(exc_ttw.value)

    with pytest.raises(DataValidationError) as exc_wtw:
        engine.calculate_wtw(fuel_consumption=50.0, fuel_type="NuclearSteam")
    assert "Unsupported fuel type" in str(exc_wtw.value)


def test_deterministic_output() -> None:
    """Verify repeated calculations for identical inputs yield identical outputs."""
    engine = MaritimeEmissionEngine()
    ttw1 = engine.calculate_ttw(75.5, "LNG")
    ttw2 = engine.calculate_ttw(75.5, "LNG")
    wtw1 = engine.calculate_wtw(75.5, "LNG")
    wtw2 = engine.calculate_wtw(75.5, "LNG")

    assert ttw1 == ttw2
    assert wtw1 == wtw2


def test_diesel_specific_regulatory_values() -> None:
    """Verify exact numerical statutory values for Diesel combustion."""
    engine = MaritimeEmissionEngine()
    ttw = engine.calculate_ttw(100.0, "Diesel")

    # 100 * 3.206 = 320.6
    assert pytest.approx(ttw.co2, rel=1e-4) == 320.6
    # 100 * 0.00005 = 0.005
    assert pytest.approx(ttw.ch4, rel=1e-4) == 0.005
    # 100 * 0.00018 = 0.018
    assert pytest.approx(ttw.n2o, rel=1e-4) == 0.018
    # co2e = 320.6 + (28 * 0.005) + (265 * 0.018) = 320.6 + 0.14 + 4.77 = 325.51
    expected_co2e = 320.6 + (28.0 * 0.005) + (265.0 * 0.018)
    assert pytest.approx(ttw.co2e, rel=1e-3) == expected_co2e
