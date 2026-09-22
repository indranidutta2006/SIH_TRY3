"""Maritime emissions calculation engine conforming to contracts.interfaces.EmissionEngine.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Calculates Tank-to-Wake (TTW) combustion emissions and Well-to-Wake (WTW) lifecycle emissions
across multiple greenhouse gas species (CO2, CH4, N2O, and CO2e) adhering to IMO MEPC
and FuelEU Maritime regulatory standards.
"""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import json
import logging
from typing import Any, Final

from contracts.exceptions import DataValidationError, PredictionError
from contracts.interfaces import EmissionEngine as BaseEmissionEngine
from contracts.schemas import EmissionResult, VoyageRecord

logger = logging.getLogger("maritime_system")

# Tank-to-Wake (TTW) direct operational CO2 emission factors (metric tons CO2 per metric ton fuel)
# Resolution MEPC.245(66) / MEPC.281(70) / FuelEU Maritime Regulation (EU) 2023/1805
CO2_FACTORS: Final[dict[str, float]] = {
    "Diesel": 3.206,      # IMO standard MDO/MGO (3.206 t CO2 / t fuel)
    "LNG": 2.750,         # Liquefied Natural Gas (2.750 t CO2 / t fuel)
    "Methanol": 1.375,    # Fossil/bio-methanol combustion factor (1.375 t CO2 / t fuel)
    "Ammonia": 0.0,       # Non-carbon alternative fuel
    "Hydrogen": 0.0,      # Non-carbon alternative fuel
    "ShorePower": 0.0,    # Zero direct operational combustion emissions
}

# Alias for backward compatibility with earlier factor imports
FUEL_EMISSION_FACTORS: Final[dict[str, float]] = CO2_FACTORS

# Tank-to-Wake (TTW) Methane (CH4) emission factors (metric tons CH4 per metric ton fuel)
# Derived from IMO 4th GHG Study & FuelEU Maritime default emission values
CH4_FACTORS: Final[dict[str, float]] = {
    "Diesel": 0.00005,    # 0.05 kg CH4 / t fuel
    "LNG": 0.00200,       # 2.0 kg CH4 / t fuel (dual-fuel methane slip allowance)
    "Methanol": 0.0,      # Negligible operational methane slip
    "Ammonia": 0.0,       # Zero methane
    "Hydrogen": 0.0,      # Zero methane
    "ShorePower": 0.0,    # Zero operational methane
}

# Tank-to-Wake (TTW) Nitrous Oxide (N2O) emission factors (metric tons N2O per metric ton fuel)
# Derived from IMO 4th GHG Study & IPCC 2006 guidelines for marine diesel engines
N2O_FACTORS: Final[dict[str, float]] = {
    "Diesel": 0.00018,    # 0.18 kg N2O / t fuel
    "LNG": 0.00011,       # 0.11 kg N2O / t fuel
    "Methanol": 0.00011,  # 0.11 kg N2O / t fuel
    "Ammonia": 0.0,       # Standard clean combustion baseline
    "Hydrogen": 0.0,      # Zero nitrous oxide
    "ShorePower": 0.0,    # Zero operational nitrous oxide
}

# Well-to-Tank (WTT) upstream greenhouse gas emissions (metric tons CO2e per metric ton fuel)
# Covers extraction, refining, liquefaction, bunkering, and transport (FuelEU Maritime / JEC WTW v5)
WTT_CO2E_FACTORS: Final[dict[str, float]] = {
    "Diesel": 0.590,      # Upstream refining and supply chain (0.590 t CO2e / t fuel)
    "LNG": 0.700,         # Upstream extraction, liquefaction, transport (0.700 t CO2e / t fuel)
    "Methanol": 0.400,    # Natural gas reforming supply chain (0.400 t CO2e / t fuel)
    "Ammonia": 0.0,       # Baseline assumption: green renewable supply chain
    "Hydrogen": 0.0,      # Baseline assumption: green electrolysis supply chain
    "ShorePower": 0.0,    # Baseline assumption: zero-emission shore connection
}

# 100-year Global Warming Potential (GWP) metrics (IPCC Fifth Assessment Report - AR5 / FuelEU Maritime)
GWP_CH4: Final[float] = 28.0
GWP_N2O: Final[float] = 265.0


def normalize_fuel_type(fuel_type: str) -> str:
    """Normalize input fuel string to standard canonical case.

    Args:
        fuel_type: Raw fuel string (e.g. 'diesel', 'DIESEL', 'LNG').

    Returns:
        Canonical title-cased or uppercase fuel name.
    """
    clean = fuel_type.strip()
    lookup = {
        "diesel": "Diesel",
        "mgo": "Diesel",
        "mdo": "Diesel",
        "vlsfo": "Diesel",
        "lng": "LNG",
        "methanol": "Methanol",
        "ammonia": "Ammonia",
        "hydrogen": "Hydrogen",
        "shorepower": "ShorePower",
        "shore_power": "ShorePower",
    }
    return lookup.get(clean.lower(), clean)


@dataclass(slots=True, frozen=True)
class CarbonEmissionResult(EmissionResult):
    """Backward-compatible emission result providing extended fuel provenance attributes."""

    fuel_consumption: float = 0.0
    emission_factor: float = 0.0
    calculation_method: str = "IMO_MEPC_Tank_to_Wake"

    @property
    def co2_emissions(self) -> float:
        """Alias returning CO2 emissions for backward compatibility."""
        return self.co2

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass including backward-compatible aliases."""
        data = asdict(self)
        data["co2_emissions"] = self.co2
        return data

    def to_json(self) -> str:
        """Serialize dataclass to JSON formatted string."""
        return json.dumps(self.to_dict(), indent=2)


class MaritimeEmissionEngine(BaseEmissionEngine):
    """Lifecycle maritime GHG emission accounting engine implementing contracts.interfaces.EmissionEngine."""

    def __init__(
        self,
        co2_factors: dict[str, float] | None = None,
        ch4_factors: dict[str, float] | None = None,
        n2o_factors: dict[str, float] | None = None,
        wtt_factors: dict[str, float] | None = None,
    ) -> None:
        """Initialize the maritime emission engine with statutory and lifecycle factors.

        Args:
            co2_factors: Mapping of fuel type to TTW CO2 emission factor.
            ch4_factors: Mapping of fuel type to TTW CH4 emission factor.
            n2o_factors: Mapping of fuel type to TTW N2O emission factor.
            wtt_factors: Mapping of fuel type to WTT CO2e factor.
        """
        self.co2_factors = co2_factors if co2_factors is not None else dict(CO2_FACTORS)
        self.ch4_factors = ch4_factors if ch4_factors is not None else dict(CH4_FACTORS)
        self.n2o_factors = n2o_factors if n2o_factors is not None else dict(N2O_FACTORS)
        self.wtt_factors = wtt_factors if wtt_factors is not None else dict(WTT_CO2E_FACTORS)
        self.logger = logger

    def _validate_inputs(self, fuel_consumption: float, fuel_type: str) -> str:
        """Validate fuel consumption magnitude and verify supported fuel classification.

        Args:
            fuel_consumption: Consumed fuel mass in metric tons.
            fuel_type: Fuel classification token.

        Returns:
            Normalized canonical fuel classification string.

        Raises:
            DataValidationError: If consumption is negative or fuel type is unsupported.
        """
        if fuel_consumption < 0.0:
            raise DataValidationError(
                f"Fuel consumption cannot be negative: {fuel_consumption} tons.",
                details={"fuel_consumption": fuel_consumption},
            )

        norm_fuel = normalize_fuel_type(fuel_type)
        if norm_fuel not in self.co2_factors:
            supported = sorted(self.co2_factors.keys())
            raise DataValidationError(
                f"Unsupported fuel type '{fuel_type}' (normalized: '{norm_fuel}'). "
                f"Supported fuels: {supported}",
                details={"fuel_type": fuel_type, "supported_fuels": supported},
            )
        return norm_fuel

    def calculate_ttw(
        self,
        fuel_consumption: float,
        fuel_type: str,
    ) -> EmissionResult:
        """Compute Tank-to-Wake (direct operational combustion) atmospheric emissions.

        Fulfills contracts.interfaces.EmissionEngine.calculate_ttw.

        Args:
            fuel_consumption: Mass of fuel burned in metric tons.
            fuel_type: Fuel classification token.

        Returns:
            EmissionResult containing CO2, CH4, N2O, and CO2e totals in metric tons.

        Raises:
            DataValidationError: When input bounds or fuel types fail validation.
        """
        norm_fuel = self._validate_inputs(fuel_consumption, fuel_type)

        co2_factor = self.co2_factors[norm_fuel]
        ch4_factor = self.ch4_factors.get(norm_fuel, 0.0)
        n2o_factor = self.n2o_factors.get(norm_fuel, 0.0)

        co2 = fuel_consumption * co2_factor
        ch4 = fuel_consumption * ch4_factor
        n2o = fuel_consumption * n2o_factor
        co2e = co2 + (GWP_CH4 * ch4) + (GWP_N2O * n2o)

        return EmissionResult(
            fuel_type=norm_fuel,
            co2=round(float(co2), 4),
            ch4=round(float(ch4), 6),
            n2o=round(float(n2o), 6),
            co2e=round(float(co2e), 4),
        )

    def calculate_wtw(
        self,
        fuel_consumption: float,
        fuel_type: str,
    ) -> EmissionResult:
        """Compute Well-to-Wake (full lifecycle supply chain plus operational) emissions.

        Fulfills contracts.interfaces.EmissionEngine.calculate_wtw.
        WTW is derived as Tank-to-Wake (TTW) combustion emissions plus Well-to-Tank (WTT)
        upstream production, processing, transport, and bunkering emissions.

        Args:
            fuel_consumption: Mass of fuel consumed in metric tons.
            fuel_type: Fuel classification token.

        Returns:
            EmissionResult quantifying lifecycle emissions across all pollutants.

        Raises:
            DataValidationError: When input bounds or fuel types fail validation.
        """
        ttw = self.calculate_ttw(fuel_consumption, fuel_type)
        wtt_factor = self.wtt_factors.get(ttw.fuel_type, 0.0)
        wtt_co2e = fuel_consumption * wtt_factor

        wtw_co2e = ttw.co2e + wtt_co2e

        return EmissionResult(
            fuel_type=ttw.fuel_type,
            co2=ttw.co2,
            ch4=round(float(ttw.ch4), 6),
            n2o=round(float(ttw.n2o), 6),
            co2e=round(float(wtw_co2e), 4),
        )

    def get_emission_factor(self, fuel_type: str) -> float:
        """Retrieve statutory TTW carbon factor for a specific fuel type.

        Args:
            fuel_type: Fuel classification token.

        Returns:
            Emission factor in tons CO2 per ton fuel.

        Raises:
            DataValidationError: If fuel type is unknown or unsupported.
        """
        norm_fuel = normalize_fuel_type(fuel_type)
        if norm_fuel not in self.co2_factors:
            supported = sorted(self.co2_factors.keys())
            raise DataValidationError(
                f"Unsupported fuel type '{fuel_type}' (normalized: '{norm_fuel}'). "
                f"Supported fuels: {supported}",
                details={"fuel_type": fuel_type, "supported_fuels": supported},
            )
        return self.co2_factors[norm_fuel]


    def calculate_emissions(
        self,
        fuel_consumption: float,
        fuel_type: str,
        calculation_method: str = "IMO_MEPC_Tank_to_Wake",
    ) -> CarbonEmissionResult:
        """Calculate direct operational CO2 emissions for fuel burned.

        Backward-compatible delegator calling calculate_ttw and returning CarbonEmissionResult.

        Args:
            fuel_consumption: Mass of fuel burned in metric tons.
            fuel_type: Type of marine fuel consumed.
            calculation_method: Provenance tag describing the calculation standard.

        Returns:
            Structured CarbonEmissionResult instance (subclass of contracts.schemas.EmissionResult).
        """
        ttw = self.calculate_ttw(fuel_consumption, fuel_type)
        factor = self.get_emission_factor(fuel_type)

        return CarbonEmissionResult(
            fuel_type=ttw.fuel_type,
            co2=ttw.co2,
            ch4=ttw.ch4,
            n2o=ttw.n2o,
            co2e=ttw.co2e,
            fuel_consumption=round(float(fuel_consumption), 4),
            emission_factor=factor,
            calculation_method=calculation_method,
        )

    def calculate_voyage_emissions(
        self,
        voyage: VoyageRecord,
        predicted_fuel: float | None = None,
    ) -> CarbonEmissionResult:
        """Calculate emissions for a specific VoyageRecord using predicted or recorded fuel.

        Args:
            voyage: VoyageRecord instance.
            predicted_fuel: Optional fuel prediction override (if voyage.fuel_consumption is None).

        Returns:
            CarbonEmissionResult instance.

        Raises:
            PredictionError: If no fuel consumption value is available.
        """
        fuel = (
            predicted_fuel
            if predicted_fuel is not None
            else voyage.fuel_consumption
        )
        if fuel is None:
            raise PredictionError(
                f"Cannot compute emissions for voyage {voyage.voyage_id}: "
                "no fuel consumption provided and record target is null.",
                details={"voyage_id": voyage.voyage_id},
            )

        return self.calculate_emissions(
            fuel_consumption=fuel,
            fuel_type=voyage.fuel_type,
            calculation_method="IMO_MEPC_Predicted_Voyage",
        )

    def batch_calculate_emissions(
        self,
        fuels: Sequence[float],
        fuel_types: Sequence[str],
    ) -> list[CarbonEmissionResult]:
        """Compute emissions across a batch of voyage predictions.

        Args:
            fuels: Sequence of fuel consumption values in metric tons.
            fuel_types: Sequence of corresponding fuel types.

        Returns:
            List of CarbonEmissionResult instances.

        Raises:
            DataValidationError: If array lengths mismatch.
        """
        if len(fuels) != len(fuel_types):
            raise DataValidationError(
                f"Mismatched batch inputs: {len(fuels)} fuels vs {len(fuel_types)} fuel types.",
                details={"fuels_len": len(fuels), "fuel_types_len": len(fuel_types)},
            )

        return [
            self.calculate_emissions(fuel, f_type)
            for fuel, f_type in zip(fuels, fuel_types, strict=False)
        ]


# Concrete implementations and backward-compatibility aliases
CarbonEmissionEngine = MaritimeEmissionEngine

