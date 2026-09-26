"""Maritime Lifecycle Assessment (LCA) Engine.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 4 Deliverable: Evaluates Well-to-Wake (WTW) lifecycle emissions, upstream Well-to-Tank (WTT)
production and transport emissions, direct Tank-to-Wake (TTW) combustion, and total lifecycle cost.
Supports configurable fuel pathways:
- Hydrogen: Grey (SMR), Blue (SMR + CCS), Green (Electrolysis)
- Methanol: Fossil, Bio, E-Methanol
- LNG: Fossil, Bio-LNG
- Ammonia: Grey, Blue, Green
- Diesel: Standard MGO/HFO, Biodiesel
"""

from collections.abc import Mapping
import logging
from typing import Any, Final

from contracts.schemas import (
    EvidenceCategory,
    FuelLifecycleProfile,
    LifecycleAssessmentResult,
)

logger = logging.getLogger("maritime_system")

# Standardized default lifecycle profiles grounded in IMO 4th GHG Study & FuelEU Maritime Annex I/II
DEFAULT_LIFECYCLE_PROFILES: Final[dict[str, dict[str, FuelLifecycleProfile]]] = {
    "Diesel": {
        "fossil": FuelLifecycleProfile(
            fuel_name="Diesel",
            production_pathway="fossil",
            production_emission_factor=0.450,  # Upstream crude extraction & refining
            transport_emission_factor=0.140,   # Marine bunkering & distribution
            storage_emission_factor=0.0,
            tank_to_wake_factor=3.206,         # IMO MEPC standard MDO direct combustion
            energy_density_mj_per_ton=42700.0, # Lower Heating Value
            renewable_fraction=0.0,
            cost_per_ton_usd=650.0,
            metadata={"evidence": EvidenceCategory.MODELLED.value},
        ),
        "biodiesel": FuelLifecycleProfile(
            fuel_name="Diesel",
            production_pathway="biodiesel",
            production_emission_factor=0.250,  # Hydrotreated vegetable oil (HVO)
            transport_emission_factor=0.100,
            storage_emission_factor=0.0,
            tank_to_wake_factor=3.206,         # Combustion (offset by biogenic feedstock under FuelEU)
            energy_density_mj_per_ton=37200.0,
            renewable_fraction=1.0,
            cost_per_ton_usd=1100.0,
            metadata={"evidence": EvidenceCategory.MODELLED.value},
        ),
    },
    "LNG": {
        "fossil": FuelLifecycleProfile(
            fuel_name="LNG",
            production_pathway="fossil",
            production_emission_factor=0.500,  # Extraction & liquefaction
            transport_emission_factor=0.150,   # Cryogenic LNG transport
            storage_emission_factor=0.056,     # Methane slip allowance (2 kg/t CH4 * 28 GWP)
            tank_to_wake_factor=2.750,         # Direct combustion CO2
            energy_density_mj_per_ton=49100.0,
            renewable_fraction=0.0,
            cost_per_ton_usd=800.0,
            metadata={"evidence": EvidenceCategory.MODELLED.value},
        ),
        "bio_lng": FuelLifecycleProfile(
            fuel_name="LNG",
            production_pathway="bio_lng",
            production_emission_factor=0.180,  # Anaerobic digestion biomethane
            transport_emission_factor=0.070,
            storage_emission_factor=0.056,
            tank_to_wake_factor=2.750,
            energy_density_mj_per_ton=49100.0,
            renewable_fraction=1.0,
            cost_per_ton_usd=1400.0,
            metadata={"evidence": EvidenceCategory.MODELLED.value},
        ),
    },
    "Methanol": {
        "fossil": FuelLifecycleProfile(
            fuel_name="Methanol",
            production_pathway="fossil",
            production_emission_factor=0.300,  # Natural gas steam reforming
            transport_emission_factor=0.100,
            storage_emission_factor=0.0,
            tank_to_wake_factor=1.375,
            energy_density_mj_per_ton=19900.0,
            renewable_fraction=0.0,
            cost_per_ton_usd=550.0,
            metadata={"evidence": EvidenceCategory.MODELLED.value},
        ),
        "bio_methanol": FuelLifecycleProfile(
            fuel_name="Methanol",
            production_pathway="bio_methanol",
            production_emission_factor=0.120,  # Biomass gasification
            transport_emission_factor=0.080,
            storage_emission_factor=0.0,
            tank_to_wake_factor=1.375,
            energy_density_mj_per_ton=19900.0,
            renewable_fraction=1.0,
            cost_per_ton_usd=950.0,
            metadata={"evidence": EvidenceCategory.MODELLED.value},
        ),
        "e_methanol": FuelLifecycleProfile(
            fuel_name="Methanol",
            production_pathway="e_methanol",
            production_emission_factor=0.040,  # DAC CO2 + Green H2 synthesis
            transport_emission_factor=0.060,
            storage_emission_factor=0.0,
            tank_to_wake_factor=1.375,
            energy_density_mj_per_ton=19900.0,
            renewable_fraction=1.0,
            cost_per_ton_usd=1350.0,
            metadata={"evidence": EvidenceCategory.MODELLED.value},
        ),
    },
    "Hydrogen": {
        "grey": FuelLifecycleProfile(
            fuel_name="Hydrogen",
            production_pathway="grey",
            production_emission_factor=9.000,  # Unabated Steam Methane Reforming (SMR)
            transport_emission_factor=1.500,   # High-pressure 700 bar compression / liquid trucking
            storage_emission_factor=0.0,
            tank_to_wake_factor=0.0,           # Zero direct combustion/fuel cell emission
            energy_density_mj_per_ton=120000.0,
            renewable_fraction=0.0,
            cost_per_ton_usd=2200.0,
            metadata={"evidence": EvidenceCategory.MODELLED.value},
        ),
        "blue": FuelLifecycleProfile(
            fuel_name="Hydrogen",
            production_pathway="blue",
            production_emission_factor=1.500,  # SMR with 90% Carbon Capture & Storage (CCS)
            transport_emission_factor=1.000,
            storage_emission_factor=0.0,
            tank_to_wake_factor=0.0,
            energy_density_mj_per_ton=120000.0,
            renewable_fraction=0.0,
            cost_per_ton_usd=3200.0,
            metadata={"evidence": EvidenceCategory.MODELLED.value},
        ),
        "green": FuelLifecycleProfile(
            fuel_name="Hydrogen",
            production_pathway="green",
            production_emission_factor=0.250,  # Renewable PEM/Alkaline electrolysis
            transport_emission_factor=0.100,   # Localized pipeline/port terminal
            storage_emission_factor=0.0,
            tank_to_wake_factor=0.0,
            energy_density_mj_per_ton=120000.0,
            renewable_fraction=1.0,
            cost_per_ton_usd=4500.0,
            metadata={"evidence": EvidenceCategory.MODELLED.value},
        ),
    },
    "Ammonia": {
        "grey": FuelLifecycleProfile(
            fuel_name="Ammonia",
            production_pathway="grey",
            production_emission_factor=1.900,  # Fossil Haber-Bosch synthesis
            transport_emission_factor=0.300,
            storage_emission_factor=0.0,
            tank_to_wake_factor=0.0,
            energy_density_mj_per_ton=18600.0,
            renewable_fraction=0.0,
            cost_per_ton_usd=700.0,
            metadata={"evidence": EvidenceCategory.MODELLED.value},
        ),
        "blue": FuelLifecycleProfile(
            fuel_name="Ammonia",
            production_pathway="blue",
            production_emission_factor=0.450,  # Haber-Bosch with CCS
            transport_emission_factor=0.200,
            storage_emission_factor=0.0,
            tank_to_wake_factor=0.0,
            energy_density_mj_per_ton=18600.0,
            renewable_fraction=0.0,
            cost_per_ton_usd=1050.0,
            metadata={"evidence": EvidenceCategory.MODELLED.value},
        ),
        "green": FuelLifecycleProfile(
            fuel_name="Ammonia",
            production_pathway="green",
            production_emission_factor=0.080,  # Green Hydrogen Haber-Bosch
            transport_emission_factor=0.120,
            storage_emission_factor=0.0,
            tank_to_wake_factor=0.0,
            energy_density_mj_per_ton=18600.0,
            renewable_fraction=1.0,
            cost_per_ton_usd=1500.0,
            metadata={"evidence": EvidenceCategory.MODELLED.value},
        ),
    },
}


def normalize_fuel_name(fuel_name: str) -> str:
    """Normalize fuel name string to canonical casing."""
    f = str(fuel_name).strip()
    if f.upper() == "LNG":
        return "LNG"
    return f.capitalize()


class MaritimeLifecycleAssessmentEngine:
    """Computes comprehensive Well-to-Wake lifecycle footprints across maritime fuel pathways."""

    def __init__(
        self,
        custom_profiles: Mapping[str, Mapping[str, FuelLifecycleProfile]] | None = None,
    ) -> None:
        """Initialize LCA engine with default or custom lifecycle profiles."""
        self.profiles: dict[str, dict[str, FuelLifecycleProfile]] = {}
        for fuel, pathways in DEFAULT_LIFECYCLE_PROFILES.items():
            self.profiles[fuel] = dict(pathways)
        if custom_profiles:
            for fuel, pathways in custom_profiles.items():
                if fuel not in self.profiles:
                    self.profiles[fuel] = {}
                self.profiles[fuel].update(dict(pathways))
        self.logger = logger

    def get_profile(self, fuel_name: str, pathway: str | None = None) -> FuelLifecycleProfile:
        """Retrieve the FuelLifecycleProfile for given fuel and pathway with alias and casing resilience."""
        norm_fuel = normalize_fuel_name(fuel_name)

        if norm_fuel not in self.profiles:
            supported = sorted(self.profiles.keys())
            raise ValueError(
                f"Unknown or unsupported marine fuel '{fuel_name}' (normalized: '{norm_fuel}'). "
                f"Available supported fuels are: {supported}."
            )

        pathways = self.profiles[norm_fuel]
        if pathway:
            pw_key = str(pathway).strip().lower().replace("-", "_").replace(" ", "_")
            if pw_key in pathways:
                return pathways[pw_key]

            # Alias checking for alternative fuel terminology
            aliases: dict[str, str] = {
                "e_fuel": "e_methanol",
                "efuel": "e_methanol",
                "emethanol": "e_methanol",
                "biomethanol": "bio_methanol",
                "biolng": "bio_lng",
                "bio_fuel": "bio_methanol" if norm_fuel == "Methanol" else "bio_lng",
                "fossil_lng": "fossil",
                "fossil_diesel": "fossil",
                "fossil_methanol": "fossil",
                "smr": "grey",
                "electrolysis": "green",
                "renewable": "green",
            }
            mapped_pw = aliases.get(pw_key)
            if mapped_pw and mapped_pw in pathways:
                return pathways[mapped_pw]

            supported_pws = sorted(pathways.keys())
            raise ValueError(
                f"Unknown feedstock pathway '{pathway}' for fuel '{norm_fuel}'. "
                f"Supported pathways for {norm_fuel} are: {supported_pws}."
            )

        # Return default pathway for fuel
        default_keys = {"Diesel": "fossil", "LNG": "fossil", "Methanol": "fossil", "Hydrogen": "green", "Ammonia": "green"}
        target_key = default_keys.get(norm_fuel, list(pathways.keys())[0])
        return pathways.get(target_key, list(pathways.values())[0])

    def assess_fuel_lifecycle(
        self,
        fuel_name: str,
        consumption_tons: float,
        pathway: str | None = None,
        carbon_price_usd_per_ton: float = 80.0,
        fueleu_penalty_usd: float = 0.0,
    ) -> LifecycleAssessmentResult:
        """Calculate detailed Well-to-Wake (WTW) lifecycle footprint for given fuel consumption."""
        profile = self.get_profile(fuel_name, pathway)

        c = max(0.0, float(consumption_tons))
        prod_emiss = c * profile.production_emission_factor
        trans_emiss = c * profile.transport_emission_factor
        stor_emiss = c * profile.storage_emission_factor
        wtt_emiss = prod_emiss + trans_emiss + stor_emiss
        ttw_emiss = c * profile.tank_to_wake_factor
        wtw_emiss = wtt_emiss + ttw_emiss

        energy_mj = c * profile.energy_density_mj_per_ton
        ghg_intensity = (wtw_emiss * 1e6) / max(energy_mj, 1e-4) if energy_mj > 0 else 0.0

        bunker_cost = c * profile.cost_per_ton_usd
        carbon_tax = wtw_emiss * carbon_price_usd_per_ton
        total_lifecycle_cost = bunker_cost + carbon_tax + fueleu_penalty_usd

        return LifecycleAssessmentResult(
            fuel_type=profile.fuel_name,
            pathway=profile.production_pathway,
            fuel_consumption_tons=round(c, 2),
            tank_to_wake_emissions=round(ttw_emiss, 2),
            well_to_tank_emissions=round(wtt_emiss, 2),
            well_to_wake_emissions=round(wtw_emiss, 2),
            fuel_production_emissions=round(prod_emiss, 2),
            fuel_transport_emissions=round(trans_emiss, 2),
            fuel_storage_emissions=round(stor_emiss, 2),
            lifecycle_cost=round(total_lifecycle_cost, 2),
            energy_content_mj=round(energy_mj, 1),
            emission_intensity_g_per_mj=round(ghg_intensity, 2),
            metadata={
                "bunker_cost_usd": round(bunker_cost, 2),
                "carbon_tax_usd": round(carbon_tax, 2),
                "fueleu_penalty_usd": round(fueleu_penalty_usd, 2),
                "renewable_fraction": profile.renewable_fraction,
                "evidence_type": profile.metadata.get("evidence", EvidenceCategory.MODELLED.value),
            },
        )

    def assess_fleet_lifecycle(
        self,
        fuel_consumption: Mapping[str, float],
        pathways: Mapping[str, str] | None = None,
        carbon_price_usd: float = 80.0,
        fueleu_penalties: Mapping[str, float] | None = None,
    ) -> dict[str, LifecycleAssessmentResult]:
        """Compute multi-fuel lifecycle assessments for an entire fleet or voyage profile with casing-resilient pathway mapping."""
        results: dict[str, LifecycleAssessmentResult] = {}
        path_map = pathways or {}
        pen_map = fueleu_penalties or {}

        # Pre-normalize pathways map for flexible lookup
        norm_path_map: dict[str, str] = {}
        for k, v in path_map.items():
            norm_path_map[k] = v
            norm_path_map[k.lower()] = v
            norm_path_map[normalize_fuel_name(k)] = v

        norm_pen_map: dict[str, float] = {}
        for k, v in pen_map.items():
            val = float(v)
            norm_pen_map[k] = val
            norm_pen_map[k.lower()] = val
            norm_pen_map[normalize_fuel_name(k)] = val

        for fuel, tons in fuel_consumption.items():
            norm_f = normalize_fuel_name(fuel)
            pw = norm_path_map.get(fuel) or norm_path_map.get(norm_f) or norm_path_map.get(fuel.lower())
            pen = norm_pen_map.get(fuel) or norm_pen_map.get(norm_f) or norm_pen_map.get(fuel.lower(), 0.0)
            results[fuel] = self.assess_fuel_lifecycle(
                fuel_name=fuel,
                consumption_tons=tons,
                pathway=pw,
                carbon_price_usd_per_ton=carbon_price_usd,
                fueleu_penalty_usd=pen,
            )

        return results
