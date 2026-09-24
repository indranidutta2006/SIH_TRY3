"""Macro-level green fleet scenario simulation and tradeoff analysis engine.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Implements contracts.interfaces.ScenarioEngine to model multi-fuel fleet transition pathways,
enforcing strict routing:
- Diesel, LNG, Methanol -> ProductionModelManager (Learned Machine Learning Regressor)
- Hydrogen, Ammonia, ShorePower -> MaritimeFuelPhysicsEngine (First-Principles Hydrodynamic Engine)
Computes Well-to-Wake (WTW) lifecycle emissions, FuelEU statutory compliance penalties,
and balanced multi-criteria tradeoff rankings without dimensional magnitude distortion.
"""

import logging
from typing import Any, Final, Sequence

from contracts.constants import FuelType
from contracts.exceptions import MaritimeSystemError
from contracts.interfaces import ScenarioEngine
from contracts.schemas import ScenarioResult, VoyageRecord
from src.compliance.compliance_engine import MaritimeComplianceEngine
from src.physics.fuel_physics_engine import MaritimeFuelPhysicsEngine, normalize_fuel_name
from src.prediction.emission_engine import MaritimeEmissionEngine
from src.prediction.model_manager import ProductionModelManager

logger = logging.getLogger("maritime_system")

# Fuels supported by historical ML training distribution
ML_ROUTED_FUELS: Final[set[str]] = {
    FuelType.DIESEL.value,
    FuelType.LNG.value,
    FuelType.METHANOL.value,
}

# Alternative fuels strictly routed through first-principles physics engine
PHYSICS_ROUTED_FUELS: Final[set[str]] = {
    FuelType.HYDROGEN.value,
    FuelType.AMMONIA.value,
    FuelType.SHORE_POWER.value,
}

# Standard economic baseline fuel prices (USD per metric ton)
STANDARD_FUEL_PRICES_USD: Final[dict[str, float]] = {
    "Diesel": 650.0,
    "LNG": 800.0,
    "Methanol": 950.0,
    "Hydrogen": 2500.0,
    "Ammonia": 1200.0,
    "ShorePower": 300.0,
}

# Fuel Lower Calorific Values (MJ per metric ton) for FuelEU intensity evaluation
FUEL_LCV_MJ_PER_TON: Final[dict[str, float]] = {
    "Diesel": 42700.0,
    "LNG": 49100.0,
    "Methanol": 19900.0,
    "Hydrogen": 120000.0,
    "Ammonia": 18600.0,
    "ShorePower": 3600.0,
}


class ScenarioAnalysisEngine(ScenarioEngine):
    """Scenario simulation and green fleet comparison engine implementing ScenarioEngine."""

    def __init__(
        self,
        model_manager: ProductionModelManager | None = None,
        physics_engine: MaritimeFuelPhysicsEngine | None = None,
        emission_engine: MaritimeEmissionEngine | None = None,
        compliance_engine: MaritimeComplianceEngine | None = None,
    ) -> None:
        """Initialize scenario engine with shared or injected subsystem engines."""
        self._model_manager = model_manager
        self._physics_engine = physics_engine or MaritimeFuelPhysicsEngine()
        self._emission_engine = emission_engine or MaritimeEmissionEngine()
        self._compliance_engine = compliance_engine or MaritimeComplianceEngine()

    @property
    def model_manager(self) -> ProductionModelManager:
        """Lazily initialize ProductionModelManager only when an ML-routed scenario executes."""
        if self._model_manager is None:
            self._model_manager = ProductionModelManager()
        return self._model_manager

    @property
    def physics_engine(self) -> MaritimeFuelPhysicsEngine:
        """Retrieve the physics calculation engine."""
        return self._physics_engine

    @property
    def emission_engine(self) -> MaritimeEmissionEngine:
        """Retrieve the greenhouse gas emission engine."""
        return self._emission_engine

    @property
    def compliance_engine(self) -> MaritimeComplianceEngine:
        """Retrieve the statutory compliance engine."""
        return self._compliance_engine

    def run_scenario(
        self,
        scenario_name: str,
        vessel_fleet: Sequence[str],
        operational_parameters: dict[str, Any],
    ) -> ScenarioResult:
        """Simulate fleet operation under an environmental, speed, or alternative fuel regime.

        Strict Routing Guardrail:
        - Diesel, LNG, Methanol are routed through ProductionModelManager().get_best_model()
        - Hydrogen, Ammonia, ShorePower are routed through MaritimeFuelPhysicsEngine()

        Args:
            scenario_name: Identifier for the scenario run.
            vessel_fleet: List of vessel identifiers evaluated in the run.
            operational_parameters: Dict containing fuel_type, distance_nm, speed_knots,
                cargo_tons, weather_factor, compliance_year, and optional overrides.

        Returns:
            ScenarioResult summarizing cumulative costs, emissions, and fuel consumption.
        """
        raw_fuel = operational_parameters.get("fuel_type", FuelType.DIESEL.value)
        fuel_type = normalize_fuel_name(raw_fuel)
        compliance_year = int(operational_parameters.get("compliance_year", 2025))
        fuel_prices = operational_parameters.get("fuel_prices", STANDARD_FUEL_PRICES_USD)
        price_per_ton = float(fuel_prices.get(fuel_type, STANDARD_FUEL_PRICES_USD.get(fuel_type, 650.0)))
        vessel_specs = operational_parameters.get("vessel_specs", {})

        total_fuel_tons = 0.0
        total_co2e_tons = 0.0
        total_cost_usd = 0.0

        is_ml_route = fuel_type in ML_ROUTED_FUELS
        logger.info(
            "Executing scenario '%s' (fuel=%s, fleet_size=%d, routing=%s)",
            scenario_name,
            fuel_type,
            len(vessel_fleet),
            "ML" if is_ml_route else "Physics",
        )

        # 1. Compute fuel consumption per vessel respecting the strict routing boundary
        if is_ml_route:
            # Route ML-trained fuels (Diesel, LNG, Methanol) through ProductionModelManager
            records: list[VoyageRecord] = []
            for v_id in vessel_fleet:
                spec = vessel_specs.get(v_id, {})
                dist = float(spec.get("distance_nm", operational_parameters.get("distance_nm", 1000.0)))
                spd = float(spec.get("speed_knots", operational_parameters.get("speed_knots", 14.0)))
                cargo = float(spec.get("cargo_tons", operational_parameters.get("cargo_tons", 40000.0)))
                dwt = float(spec.get("vessel_dwt", spec.get("capacity", operational_parameters.get("vessel_dwt", max(cargo * 1.25, 50000.0)))))
                v_type = str(spec.get("vessel_type", operational_parameters.get("vessel_type", "Bulk Carrier")))
                weather = float(spec.get("weather_factor", operational_parameters.get("weather_factor", 1.0)))
                sea = int(spec.get("sea_state", operational_parameters.get("sea_state", 3)))
                hours = dist / max(spd, 1.0)

                record = VoyageRecord(
                    voyage_id=f"SCEN-{scenario_name}-{v_id}",
                    vessel_id=v_id,
                    vessel_type=v_type,
                    vessel_dwt=dwt,
                    cargo_tons=cargo,
                    distance_nm=dist,
                    speed_knots=spd,
                    hours_at_sea=hours,
                    fuel_type=fuel_type,
                    weather_factor=weather,
                    sea_state=sea,
                    data_source="scenario_engine",
                    is_synthetic=True,
                    fuel_consumption=None,
                    co2_emissions=None,
                )
                records.append(record)

            best_model = self.model_manager.get_best_model()
            predictions = best_model.predict(records)
            fuel_per_vessel = [float(p.predicted_fuel_consumption) for p in predictions]
        else:
            # Route non-ML alternative fuels (Hydrogen, Ammonia, ShorePower) through FuelPhysicsEngine
            # ProductionModelManager is NEVER called or touched here
            fuel_per_vessel = []
            for v_id in vessel_fleet:
                spec = vessel_specs.get(v_id, {})
                dist = float(spec.get("distance_nm", operational_parameters.get("distance_nm", 1000.0)))
                spd = float(spec.get("speed_knots", operational_parameters.get("speed_knots", 14.0)))
                cargo = float(spec.get("cargo_tons", operational_parameters.get("cargo_tons", 40000.0)))
                dwt = float(spec.get("vessel_dwt", spec.get("capacity", operational_parameters.get("vessel_dwt", max(cargo * 1.25, 50000.0)))))
                weather = float(spec.get("weather_factor", operational_parameters.get("weather_factor", 1.0)))

                fuel_val = self.physics_engine.calculate_fuel_use(
                    distance_nm=dist,
                    speed_knots=spd,
                    cargo_tons=cargo,
                    weather_factor=weather,
                    fuel_type=fuel_type,
                    vessel_dwt=dwt,
                )
                fuel_per_vessel.append(fuel_val)

        # 2. Lifecycle Emissions, Energy, and FuelEU Compliance Penalties
        lcv_mj_per_ton = FUEL_LCV_MJ_PER_TON.get(fuel_type, 42700.0)

        for fuel_tons in fuel_per_vessel:
            total_fuel_tons += fuel_tons

            # Calculate Well-to-Wake CO2e emissions
            emiss_result = self.emission_engine.calculate_wtw(fuel_tons, fuel_type)
            co2e = float(emiss_result.co2e)
            total_co2e_tons += co2e

            # Energy and statutory compliance calculation
            if fuel_type == FuelType.SHORE_POWER.value:
                # Shore power electrical MWh directly translated into energy and cost
                energy_mwh = self.physics_engine.last_energy_mwh
                voyage_energy_mj = energy_mwh * 3600.0
                fueleu_penalty = 0.0  # Zero operational GHG emissions
                voyage_cost = (energy_mwh * price_per_ton)  # price_per_ton acts as USD/MWh for shore power
            else:
                voyage_energy_mj = fuel_tons * lcv_mj_per_ton
                ghg_intensity = (co2e * 1e6) / voyage_energy_mj if voyage_energy_mj > 0.0 else 0.0
                comp_result = self.compliance_engine.evaluate_fueleu(
                    ghg_intensity=ghg_intensity,
                    energy_used_mj=voyage_energy_mj,
                    year=compliance_year,
                )
                fueleu_penalty = float(comp_result.penalty_eur)
                voyage_cost = (fuel_tons * price_per_ton) + fueleu_penalty

            total_cost_usd += voyage_cost

        result = ScenarioResult(
            scenario_name=scenario_name,
            fuel_type=fuel_type,
            total_cost=float(round(total_cost_usd, 2)),
            total_emissions=float(round(total_co2e_tons, 2)),
            fuel_consumption=float(round(total_fuel_tons, 2)),
        )
        return result

    def compare_scenarios(
        self,
        scenario_results: Sequence[ScenarioResult],
    ) -> list[ScenarioResult]:
        """Rank and analyze tradeoff frontiers across multiple executed scenario runs.

        To avoid unweighted scalar magnitude distortion (where ~$10^6 fuel cost swamps ~$10^3 tons CO2e),
        scenarios are evaluated using a balanced, dimensionless multi-criteria score:
            Score(s) = 0.5 * (Cost(s) / Max_Cost) + 0.5 * (Emissions(s) / Max_Emissions)

        Args:
            scenario_results: Sequence of executed scenario outcomes.

        Returns:
            Ranked list of ScenarioResult instances sorted ascending by normalized tradeoff score
            (lowest combined cost-and-emissions score ranked first).
        """
        if not scenario_results:
            return []

        if len(scenario_results) == 1:
            return list(scenario_results)

        max_cost = max(r.total_cost for r in scenario_results)
        max_emiss = max(r.total_emissions for r in scenario_results)

        denom_cost = max_cost if max_cost > 0.0 else 1.0
        denom_emiss = max_emiss if max_emiss > 0.0 else 1.0

        def score_scenario(r: ScenarioResult) -> float:
            norm_cost = r.total_cost / denom_cost
            norm_emiss = r.total_emissions / denom_emiss
            return 0.5 * norm_cost + 0.5 * norm_emiss

        ranked = sorted(scenario_results, key=score_scenario)
        return ranked

    @staticmethod
    def rank_by_cost(scenario_results: Sequence[ScenarioResult]) -> list[ScenarioResult]:
        """Rank scenarios strictly by total financial cost in ascending order."""
        return sorted(scenario_results, key=lambda r: r.total_cost)

    @staticmethod
    def rank_by_emissions(scenario_results: Sequence[ScenarioResult]) -> list[ScenarioResult]:
        """Rank scenarios strictly by total Well-to-Wake CO2e emissions in ascending order."""
        return sorted(scenario_results, key=lambda r: r.total_emissions)
