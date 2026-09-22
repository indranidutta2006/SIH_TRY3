"""Naval architecture hydrodynamic and alternative fuel physics calculation engine.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Implements contracts.interfaces.FuelPhysicsEngine for first-principles energy,
propulsion resistance (Admiralty formula), and alternative fuel mass conversions
(Hydrogen, Ammonia, ShorePower) unrepresented in classical ML training sets.
"""

import logging
from typing import Final

from contracts.constants import FuelType
from contracts.exceptions import MaritimeSystemError
from contracts.interfaces import FuelPhysicsEngine

logger = logging.getLogger("maritime_system")


class PhysicsCalculationError(MaritimeSystemError):
    """Raised when hydrodynamic or thermodynamic physical calculations encounter invalid inputs."""


# Lower Heating Values (LHV) in MJ/kg
# Comments indicate verification status against IMO / ICCT statutory guidelines
LHV_MJ_PER_KG: Final[dict[str, float]] = {
    "Diesel": 42.7,     # approximate — verify against IMO/ICCT before final report
    "LNG": 50.0,        # approximate — verify against IMO/ICCT before final report
    "Methanol": 19.9,   # approximate — verify against IMO/ICCT before final report
    "Hydrogen": 120.0,  # approximate — verify against IMO/ICCT before final report
    "Ammonia": 18.6,    # approximate — verify against IMO/ICCT before final report
}

# Standard brake thermal efficiency for marine engines / fuel cells
DEFAULT_THERMAL_EFFICIENCY: Final[float] = 0.45  # ~45% brake thermal efficiency
ELECTRICAL_EFFICIENCY: Final[float] = 0.95        # ~95% electric motor / shore drive efficiency


def normalize_fuel_name(fuel_type: str) -> str:
    """Canonicalize fuel type string matching system FuelType tokens."""
    clean = fuel_type.strip().lower()
    mapping = {
        "diesel": FuelType.DIESEL.value,
        "mgo": FuelType.DIESEL.value,
        "mdo": FuelType.DIESEL.value,
        "vlsfo": FuelType.DIESEL.value,
        "lng": FuelType.LNG.value,
        "methanol": FuelType.METHANOL.value,
        "hydrogen": FuelType.HYDROGEN.value,
        "ammonia": FuelType.AMMONIA.value,
        "shorepower": FuelType.SHORE_POWER.value,
        "shore_power": FuelType.SHORE_POWER.value,
        "shore power": FuelType.SHORE_POWER.value,
    }
    return mapping.get(clean, fuel_type.strip())


class MaritimeFuelPhysicsEngine(FuelPhysicsEngine):
    """Naval architectural physics engine implementing contracts.interfaces.FuelPhysicsEngine."""

    def __init__(
        self,
        thermal_efficiency: float = DEFAULT_THERMAL_EFFICIENCY,
        electrical_efficiency: float = ELECTRICAL_EFFICIENCY,
        default_admiralty_coeff: float = 500.0,
    ) -> None:
        """Initialize physics engine with propulsion and energy efficiency constants."""
        self.thermal_efficiency = thermal_efficiency
        self.electrical_efficiency = electrical_efficiency
        self.default_admiralty_coeff = default_admiralty_coeff
        self._last_energy_mwh: float = 0.0

    @property
    def last_energy_mwh(self) -> float:
        """Energy in MWh computed in the most recent calculate_fuel_use invocation."""
        return self._last_energy_mwh

    def calculate_fuel_use(
        self,
        distance_nm: float,
        speed_knots: float,
        cargo_tons: float,
        weather_factor: float,
        fuel_type: str,
        vessel_dwt: float | None = None,
        admiralty_coeff: float | None = None,
    ) -> float:
        """Calculate total theoretical fuel consumption using hydrodynamic resistance principles.

        Reuses the classical Admiralty propulsion formula:
            P_propulsion = (Delta^(2/3) * V^3) / C_adm
        and converts total mechanical propulsion energy into fuel mass via fuel Lower Heating Value (LHV).
        For ShorePower, returns 0.0 metric tons combusted fuel mass while tracking electrical MWh.

        Args:
            distance_nm: Voyage distance in nautical miles.
            speed_knots: Vessel operational velocity in knots.
            cargo_tons: Vessel displacement / deadweight payload in metric tons.
            weather_factor: Weather severity multiplier (>= 1.0).
            fuel_type: Fuel classification token (Diesel, LNG, Methanol, Hydrogen, Ammonia, ShorePower).
            vessel_dwt: Optional vessel deadweight tonnage (defaults to estimated 1.25 * cargo_tons).
            admiralty_coeff: Optional custom admiralty coefficient (defaults to engine standard).

        Returns:
            Computed fuel mass consumption in metric tons (0.0 for ShorePower).

        Raises:
            PhysicsCalculationError: If input parameters violate physical constraints.
        """
        if distance_nm <= 0.0:
            raise PhysicsCalculationError(
                f"Voyage distance must be positive, got {distance_nm} nm",
                details={"distance_nm": distance_nm},
            )
        if speed_knots <= 0.0:
            raise PhysicsCalculationError(
                f"Vessel speed must be positive, got {speed_knots} knots",
                details={"speed_knots": speed_knots},
            )
        if cargo_tons < 0.0:
            raise PhysicsCalculationError(
                f"Cargo tons cannot be negative, got {cargo_tons}",
                details={"cargo_tons": cargo_tons},
            )
        if weather_factor <= 0.0:
            raise PhysicsCalculationError(
                f"Weather factor must be positive, got {weather_factor}",
                details={"weather_factor": weather_factor},
            )

        canonical_fuel = normalize_fuel_name(fuel_type)

        # 1. Hull displacement calculation (lightweight + payload)
        dwt = vessel_dwt if vessel_dwt is not None and vessel_dwt > 0.0 else max(cargo_tons * 1.25, 10000.0)
        displacement_tons = (0.20 * dwt) + cargo_tons

        # 2. Main engine propulsion power (kW) via Admiralty formula
        c_adm = admiralty_coeff if admiralty_coeff is not None and admiralty_coeff > 0.0 else self.default_admiralty_coeff
        propulsion_power_kw = (displacement_tons ** (2.0 / 3.0) * (speed_knots ** 3.0)) / c_adm

        # 3. Auxiliary and hotel load (kW)
        auxiliary_power_kw = 0.05 * (dwt ** 0.6) * 100.0
        total_power_kw = propulsion_power_kw + auxiliary_power_kw

        # 4. Total mechanical energy requirement (MWh & MJ)
        hours_at_sea = distance_nm / max(speed_knots, 1.0)
        mech_energy_kwh = total_power_kw * hours_at_sea * max(weather_factor, 1.0)
        mech_energy_mj = mech_energy_kwh * 3.6

        # 5. Route ShorePower vs Combusted Chemical Fuels
        if canonical_fuel == FuelType.SHORE_POWER.value:
            # ShorePower uses direct electrical propulsion / grid power
            electrical_energy_mwh = (mech_energy_kwh / 1000.0) / self.electrical_efficiency
            self._last_energy_mwh = float(round(electrical_energy_mwh, 4))
            return 0.0

        if canonical_fuel not in LHV_MJ_PER_KG:
            raise PhysicsCalculationError(
                f"Unsupported fuel type '{fuel_type}'. Supported fuels: {list(LHV_MJ_PER_KG.keys()) + ['ShorePower']}",
                details={"fuel_type": fuel_type},
            )

        # Thermal energy input needed from fuel combustion
        fuel_energy_mj = mech_energy_mj / self.thermal_efficiency
        fuel_energy_mwh = fuel_energy_mj / 3600.0
        self._last_energy_mwh = float(round(fuel_energy_mwh, 4))

        # 6. Mass of fuel consumed (tons) using lower heating value (LHV)
        lhv_mj_per_kg = LHV_MJ_PER_KG[canonical_fuel]
        fuel_mass_kg = fuel_energy_mj / lhv_mj_per_kg
        fuel_mass_tons = fuel_mass_kg / 1000.0

        return float(round(fuel_mass_tons, 4))

    def calculate_energy(
        self,
        fuel_consumption: float,
        fuel_type: str,
    ) -> float:
        """Derive mechanical and thermal propulsion energy in megawatt-hours (MWh).

        Converts fuel mass into energy via:
            energy_mwh = fuel_consumption_tons * 1000 * LHV_MJ_PER_KG[fuel_type] / 3600

        Args:
            fuel_consumption: Total fuel consumed in metric tons.
            fuel_type: Fuel classification token denoting Lower Heating Value (LHV).

        Returns:
            Calculated total thermal fuel energy in megawatt-hours (MWh).

        Raises:
            PhysicsCalculationError: If called for ShorePower or invalid fuel types.
        """
        canonical_fuel = normalize_fuel_name(fuel_type)

        if canonical_fuel == FuelType.SHORE_POWER.value:
            raise PhysicsCalculationError(
                "ShorePower consumes zero combusted fuel mass. Energy must be obtained "
                "from electrical consumption tracking (last_energy_mwh), not calculate_energy via fuel mass.",
                details={"fuel_type": fuel_type, "fuel_consumption": fuel_consumption},
            )

        if canonical_fuel not in LHV_MJ_PER_KG:
            raise PhysicsCalculationError(
                f"Unsupported fuel type '{fuel_type}'. Supported: {list(LHV_MJ_PER_KG.keys())}",
                details={"fuel_type": fuel_type},
            )

        if fuel_consumption < 0.0:
            raise PhysicsCalculationError(
                f"Fuel consumption cannot be negative, got {fuel_consumption} tons",
                details={"fuel_consumption": fuel_consumption},
            )

        lhv = LHV_MJ_PER_KG[canonical_fuel]
        energy_mwh = (fuel_consumption * 1000.0 * lhv) / 3600.0
        return float(round(energy_mwh, 4))
