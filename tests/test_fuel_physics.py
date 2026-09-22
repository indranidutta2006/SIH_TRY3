"""Unit tests for the first-principles maritime fuel physics engine.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Validates Admiralty hydrodynamic resistance, alternative fuel heating values,
ShorePower electrical routing, and mathematical consistency.
"""

import pytest

from contracts.constants import FuelType
from contracts.interfaces import FuelPhysicsEngine
from src.physics.fuel_physics_engine import (
    LHV_MJ_PER_KG,
    MaritimeFuelPhysicsEngine,
    PhysicsCalculationError,
)


def test_implements_fuel_physics_interface() -> None:
    """Verify MaritimeFuelPhysicsEngine implements contracts.interfaces.FuelPhysicsEngine."""
    engine = MaritimeFuelPhysicsEngine()
    assert isinstance(engine, FuelPhysicsEngine)


def test_hydrogen_mass_smaller_than_diesel_for_fixed_energy() -> None:
    """Assert Hydrogen fuel mass is significantly smaller than Diesel for identical energy requirements.

    Hydrogen has a much higher Lower Heating Value (~120.0 MJ/kg) than Diesel (~42.7 MJ/kg),
    so less physical fuel mass is required to deliver the same propulsion energy.
    """
    engine = MaritimeFuelPhysicsEngine()
    dist = 1500.0
    spd = 14.0
    cargo = 45000.0
    weather = 1.1

    diesel_mass = engine.calculate_fuel_use(
        distance_nm=dist,
        speed_knots=spd,
        cargo_tons=cargo,
        weather_factor=weather,
        fuel_type="Diesel",
    )

    hydrogen_mass = engine.calculate_fuel_use(
        distance_nm=dist,
        speed_knots=spd,
        cargo_tons=cargo,
        weather_factor=weather,
        fuel_type="Hydrogen",
    )

    assert diesel_mass > 0.0
    assert hydrogen_mass > 0.0
    # Mass ratio must closely track inverse LHV ratio: 42.7 / 120.0 ~ 0.3558
    assert hydrogen_mass < diesel_mass
    ratio = hydrogen_mass / diesel_mass
    expected_ratio = LHV_MJ_PER_KG["Diesel"] / LHV_MJ_PER_KG["Hydrogen"]
    assert pytest.approx(ratio, rel=0.01) == expected_ratio


def test_ammonia_mass_greater_than_diesel_for_fixed_energy() -> None:
    """Assert Ammonia requires more fuel mass than Diesel due to lower energy density (~18.6 vs ~42.7 MJ/kg)."""
    engine = MaritimeFuelPhysicsEngine()
    dist = 1000.0
    spd = 12.0
    cargo = 30000.0
    weather = 1.0

    diesel_mass = engine.calculate_fuel_use(
        distance_nm=dist,
        speed_knots=spd,
        cargo_tons=cargo,
        weather_factor=weather,
        fuel_type="Diesel",
    )
    ammonia_mass = engine.calculate_fuel_use(
        distance_nm=dist,
        speed_knots=spd,
        cargo_tons=cargo,
        weather_factor=weather,
        fuel_type="Ammonia",
    )

    assert ammonia_mass > diesel_mass
    ratio = ammonia_mass / diesel_mass
    expected_ratio = LHV_MJ_PER_KG["Diesel"] / LHV_MJ_PER_KG["Ammonia"]
    assert pytest.approx(ratio, rel=0.01) == expected_ratio


def test_shore_power_fuel_consumption_zero_always() -> None:
    """Assert ShorePower fuel_consumption is identically 0.0 while electrical energy is tracked."""
    engine = MaritimeFuelPhysicsEngine()

    fuel_use = engine.calculate_fuel_use(
        distance_nm=800.0,
        speed_knots=15.0,
        cargo_tons=50000.0,
        weather_factor=1.05,
        fuel_type="ShorePower",
    )

    assert fuel_use == 0.0
    # Electrical energy must be positive and tracked
    assert engine.last_energy_mwh > 0.0


def test_calculate_energy_consistency_round_trip() -> None:
    """Assert calculate_energy is mathematically consistent with calculate_fuel_use for round-trip."""
    engine = MaritimeFuelPhysicsEngine()

    for fuel in ["Diesel", "LNG", "Methanol", "Hydrogen", "Ammonia"]:
        fuel_mass = engine.calculate_fuel_use(
            distance_nm=1200.0,
            speed_knots=14.0,
            cargo_tons=40000.0,
            weather_factor=1.0,
            fuel_type=fuel,
        )
        # Expected thermal fuel energy tracked by engine
        tracked_mwh = engine.last_energy_mwh
        # Derived energy from mass
        derived_mwh = engine.calculate_energy(fuel_mass, fuel)

        # Must match to within rounding precision
        assert pytest.approx(derived_mwh, rel=0.005) == tracked_mwh


def test_calculate_energy_shore_power_raises_error() -> None:
    """Assert calculate_energy for ShorePower raises PhysicsCalculationError with clear explanation."""
    engine = MaritimeFuelPhysicsEngine()

    with pytest.raises(PhysicsCalculationError, match="ShorePower"):
        engine.calculate_energy(0.0, "ShorePower")


def test_invalid_parameters_raise_physics_error() -> None:
    """Assert physically impossible distances, velocities, or cargo tonnage raise PhysicsCalculationError."""
    engine = MaritimeFuelPhysicsEngine()

    with pytest.raises(PhysicsCalculationError):
        engine.calculate_fuel_use(distance_nm=-100.0, speed_knots=12.0, cargo_tons=1000.0, weather_factor=1.0, fuel_type="Diesel")

    with pytest.raises(PhysicsCalculationError):
        engine.calculate_fuel_use(distance_nm=100.0, speed_knots=-10.0, cargo_tons=1000.0, weather_factor=1.0, fuel_type="Diesel")

    with pytest.raises(PhysicsCalculationError):
        engine.calculate_fuel_use(distance_nm=100.0, speed_knots=12.0, cargo_tons=-50.0, weather_factor=1.0, fuel_type="Diesel")

    with pytest.raises(PhysicsCalculationError):
        engine.calculate_fuel_use(distance_nm=100.0, speed_knots=12.0, cargo_tons=1000.0, weather_factor=0.0, fuel_type="Diesel")

    with pytest.raises(PhysicsCalculationError):
        engine.calculate_fuel_use(distance_nm=100.0, speed_knots=12.0, cargo_tons=1000.0, weather_factor=1.0, fuel_type="Kerosene")
