"""Fleet-level multi-engine scalar objective function.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Decodes decision vectors into per-voyage (speed_knots, fuel_type_index) pairs, evaluating:
1. Fuel consumption prediction via ProductionModelManager.
2. Lifecycle Well-to-Wake CO2e emissions via MaritimeEmissionEngine.
3. FuelEU Maritime deficit penalty via MaritimeComplianceEngine.
Computes scalar J = w1 * total_fuel_cost + w2 * total_co2e + w3 * schedule_delay_penalty.
"""

from collections.abc import Sequence
import logging
from typing import Any, Final

import numpy as np

from contracts.constants import DEFAULT_EUR_TO_USD_FX_RATE
from contracts.schemas import FleetAssignment, VoyageRecord
from src.compliance.compliance_engine import MaritimeComplianceEngine
from src.physics.fuel_physics_engine import MaritimeFuelPhysicsEngine
from src.prediction.emission_engine import MaritimeEmissionEngine
from src.prediction.model_manager import ProductionModelManager

logger = logging.getLogger("maritime_system")

SUPPORTED_FUEL_CHOICES: Final[tuple[str, ...]] = (
    "Diesel",
    "LNG",
    "Methanol",
    "Hydrogen",
    "Ammonia",
    "ShorePower",
)

ML_ROUTED_FUELS: Final[set[str]] = {
    "Diesel",
    "LNG",
    "Methanol",
}

PHYSICS_ROUTED_FUELS: Final[set[str]] = {
    "Hydrogen",
    "Ammonia",
    "ShorePower",
}

# Standard statutory and market baseline prices (USD per metric ton)
STANDARD_FUEL_PRICES_USD: Final[dict[str, float]] = {
    "Diesel": 650.0,
    "LNG": 800.0,
    "Methanol": 950.0,
    "Hydrogen": 2500.0,
    "Ammonia": 1200.0,
    "ShorePower": 300.0,
}

# Lower Calorific Values (LCV in MJ per metric ton)
FUEL_LCV_MJ_PER_TON: Final[dict[str, float]] = {
    "Diesel": 42700.0,
    "LNG": 49100.0,
    "Methanol": 19900.0,
    "Hydrogen": 120000.0,
    "Ammonia": 18600.0,
    "ShorePower": 3600.0,
}

# Module-level engine caches for zero-overhead swarm evaluation
_CACHED_PROD_MODEL = None
_CACHED_EMISSION_ENGINE = None
_CACHED_COMPLIANCE_ENGINE = None
_CACHED_PHYSICS_ENGINE = None


def get_cached_engines() -> tuple[Any, MaritimeEmissionEngine, MaritimeComplianceEngine, MaritimeFuelPhysicsEngine]:
    """Retrieve or lazily initialize shared production model, emission, compliance, and physics engines."""
    global _CACHED_PROD_MODEL, _CACHED_EMISSION_ENGINE, _CACHED_COMPLIANCE_ENGINE, _CACHED_PHYSICS_ENGINE
    if _CACHED_PROD_MODEL is None:
        _CACHED_PROD_MODEL = ProductionModelManager().get_best_model()
    if _CACHED_EMISSION_ENGINE is None:
        _CACHED_EMISSION_ENGINE = MaritimeEmissionEngine()
    if _CACHED_COMPLIANCE_ENGINE is None:
        _CACHED_COMPLIANCE_ENGINE = MaritimeComplianceEngine()
    if _CACHED_PHYSICS_ENGINE is None:
        _CACHED_PHYSICS_ENGINE = MaritimeFuelPhysicsEngine()
    return _CACHED_PROD_MODEL, _CACHED_EMISSION_ENGINE, _CACHED_COMPLIANCE_ENGINE, _CACHED_PHYSICS_ENGINE


def fleet_objective(
    x: np.ndarray,
    assignments: Sequence[FleetAssignment],
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    context: dict[str, Any] | None = None,
    engines: tuple[Any, ...] | None = None,
) -> float:
    """Evaluate fleet-level objective function unifying all three engines.

    Args:
        x: Flattened float decision vector: 2 values per assigned voyage (speed, fuel_idx).
        assignments: Sequence of FleetAssignment instances from the scheduler.
        weights: Tuple of (w1_cost, w2_emissions, w3_delay) weight multipliers.
        context: Optional dictionary containing voyage_specs and penalty rates.
        engines: Optional (model_engine, emission_engine, compliance_engine[, physics_engine]) tuple.

    Returns:
        Scalar cost J to minimize.
    """
    assigned_voyages = [a for a in assignments if a.assigned]
    n_voyages = len(assigned_voyages)
    if n_voyages == 0:
        return 0.0

    if len(x) < 2 * n_voyages:
        raise ValueError(
            f"Decision vector length {len(x)} is insufficient for {n_voyages} assigned voyages (requires {2 * n_voyages})."
        )

    w1, w2, w3 = weights
    ctx = context or {}
    voyage_specs = ctx.get("voyage_specs", {})
    vessel_specs = ctx.get("vessel_specs", ctx.get("vessels", {}))
    hourly_delay_rate = float(ctx.get("delay_penalty_per_hour", 500.0))
    compliance_year = int(ctx.get("compliance_year", 2025))
    eur_to_usd_rate = float(ctx.get("eur_to_usd_rate", ctx.get("fx_rate", DEFAULT_EUR_TO_USD_FX_RATE)))

    # Resolve engines (production model, emission engine, compliance engine, physics engine)
    if engines is not None:
        if len(engines) == 4:
            prod_model, emission_engine, compliance_engine, physics_engine = engines
        elif len(engines) == 3:
            prod_model, emission_engine, compliance_engine = engines
            physics_engine = MaritimeFuelPhysicsEngine()
        else:
            raise ValueError(f"engines tuple must have 3 or 4 elements, got {len(engines)}")
    else:
        prod_model, emission_engine, compliance_engine, physics_engine = get_cached_engines()

    total_fuel_cost = 0.0
    total_co2e = 0.0
    schedule_delay_penalty = 0.0

    # 1. Decode decision vector into per-voyage telemetry and route ML vs Physics
    ml_records: list[VoyageRecord] = []
    ml_indices: list[int] = []
    voyage_data: list[dict[str, Any]] = []

    for i, assignment in enumerate(assigned_voyages):
        speed_knots = float(x[2 * i])
        fuel_idx = int(np.clip(round(float(x[2 * i + 1])), 0, len(SUPPORTED_FUEL_CHOICES) - 1))
        fuel_type = SUPPORTED_FUEL_CHOICES[fuel_idx]

        v_id = assignment.vessel_id
        c_id = assignment.cargo_id

        # Resolve hierarchical voyage and vessel specifications from context
        v_entry = vessel_specs.get(v_id, {}) if isinstance(vessel_specs, dict) else {}
        if not v_entry and isinstance(voyage_specs, dict):
            v_entry = voyage_specs.get(v_id, {})
        c_entry = voyage_specs.get(c_id, {}) if isinstance(voyage_specs, dict) else {}
        pair_entry = (
            voyage_specs.get(f"{v_id}_{c_id}", voyage_specs.get(f"{v_id}-{c_id}", {}))
            if isinstance(voyage_specs, dict)
            else {}
        )
        spec = {**v_entry, **c_entry, **pair_entry}

        cargo_tons = float(spec.get("cargo_tons", spec.get("tons", spec.get("weight", 40000.0))))
        vessel_dwt = float(spec.get("vessel_dwt", spec.get("capacity", spec.get("dwt", max(cargo_tons * 1.2, 50000.0)))))
        distance_nm = float(spec.get("distance_nm", 1000.0))
        deadline_hours = float(spec.get("deadline_hours", 120.0))
        weather_factor = float(spec.get("weather_factor", 1.0))
        sea_state = int(spec.get("sea_state", 3))
        vessel_type = str(spec.get("vessel_type", "Bulk Carrier"))

        hours_at_sea = distance_nm / max(speed_knots, 1.0)

        v_info: dict[str, Any] = {
            "fuel_type": fuel_type,
            "hours_at_sea": hours_at_sea,
            "deadline_hours": deadline_hours,
            "fuel_tons": 0.0,
            "energy_mwh": 0.0,
        }

        if fuel_type in ML_ROUTED_FUELS:
            # Route ML-trained fuels (Diesel, LNG, Methanol) through ProductionModelManager
            record = VoyageRecord(
                voyage_id=f"VY-{v_id}-{c_id}",
                vessel_id=v_id,
                vessel_type=vessel_type,
                vessel_dwt=vessel_dwt,
                cargo_tons=cargo_tons,
                distance_nm=distance_nm,
                speed_knots=speed_knots,
                hours_at_sea=hours_at_sea,
                fuel_type=fuel_type,
                weather_factor=weather_factor,
                sea_state=sea_state,
                data_source="fleet_optimizer",
                is_synthetic=True,
                fuel_consumption=None,
                co2_emissions=None,
            )
            ml_records.append(record)
            ml_indices.append(i)
        else:
            # Route alternative novel fuels (Hydrogen, Ammonia, ShorePower) through FuelPhysicsEngine
            # ProductionModelManager is NEVER invoked for these fuels
            fuel_val = physics_engine.calculate_fuel_use(
                distance_nm=distance_nm,
                speed_knots=speed_knots,
                cargo_tons=cargo_tons,
                weather_factor=weather_factor,
                fuel_type=fuel_type,
                vessel_dwt=vessel_dwt,
            )
            v_info["fuel_tons"] = float(fuel_val)
            if fuel_type == "ShorePower":
                v_info["energy_mwh"] = float(physics_engine.last_energy_mwh)

        voyage_data.append(v_info)

    # 2. Vectorized ML prediction for ML-routed voyages
    if ml_records:
        pred_results = prod_model.predict(ml_records)
        for k, pred_res in enumerate(pred_results):
            orig_idx = ml_indices[k]
            voyage_data[orig_idx]["fuel_tons"] = float(pred_res.predicted_fuel_consumption)

    # 3. Calculate Well-to-Wake CO2e emissions, FuelEU penalty, and costs
    for v_info in voyage_data:
        fuel_type = v_info["fuel_type"]
        hours_at_sea = v_info["hours_at_sea"]
        deadline_hours = v_info["deadline_hours"]
        fuel_consumed = v_info["fuel_tons"]

        price_per_ton = STANDARD_FUEL_PRICES_USD.get(fuel_type, 650.0)

        if fuel_type == "ShorePower":
            energy_mwh = v_info["energy_mwh"]
            voyage_fuel_cost = energy_mwh * price_per_ton
            co2e = 0.0  # Zero operational GHG emissions
        else:
            emiss_res = emission_engine.calculate_wtw(fuel_consumed, fuel_type)
            co2e = float(emiss_res.co2e)

            lcv = FUEL_LCV_MJ_PER_TON.get(fuel_type, 42700.0)
            energy_used_mj = fuel_consumed * lcv
            ghg_intensity = (co2e * 1e6) / energy_used_mj if energy_used_mj > 0.0 else 0.0

            comp_res = compliance_engine.evaluate_fueleu(
                ghg_intensity=ghg_intensity,
                energy_used_mj=energy_used_mj,
                year=compliance_year,
            )
            fueleu_penalty_eur = float(comp_res.penalty_eur)
            fueleu_penalty_usd = fueleu_penalty_eur * eur_to_usd_rate
            voyage_fuel_cost = (fuel_consumed * price_per_ton) + fueleu_penalty_usd

        total_fuel_cost += voyage_fuel_cost
        total_co2e += co2e

        delay_hours = max(0.0, hours_at_sea - deadline_hours)
        schedule_delay_penalty += delay_hours * hourly_delay_rate

    # Scalar combination J = w1 * cost + w2 * co2e + w3 * delay
    # Supports optional reference normalization: J = w1*(Cost/Ref_Cost) + w2*(CO2e/Ref_CO2e) + w3*(Delay/Ref_Delay)
    if bool(ctx.get("normalize", False)):
        ref_cost = float(ctx.get("ref_cost", 100000.0 * n_voyages))
        ref_co2e = float(ctx.get("ref_co2e", 500.0 * n_voyages))
        ref_delay = float(ctx.get("ref_delay", 5000.0 * n_voyages))

        norm_cost = total_fuel_cost / max(ref_cost, 1.0)
        norm_co2e = total_co2e / max(ref_co2e, 1.0)
        norm_delay = schedule_delay_penalty / max(ref_delay, 1.0)
        j_scalar = (w1 * norm_cost) + (w2 * norm_co2e) + (w3 * norm_delay)
    else:
        j_scalar = (w1 * total_fuel_cost) + (w2 * total_co2e) + (w3 * schedule_delay_penalty)

    return float(j_scalar)
