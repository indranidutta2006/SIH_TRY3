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

from contracts.schemas import FleetAssignment, VoyageRecord
from src.compliance.compliance_engine import MaritimeComplianceEngine
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


def get_cached_engines() -> tuple[Any, MaritimeEmissionEngine, MaritimeComplianceEngine]:
    """Retrieve or lazily initialize shared production model, emission, and compliance engines."""
    global _CACHED_PROD_MODEL, _CACHED_EMISSION_ENGINE, _CACHED_COMPLIANCE_ENGINE
    if _CACHED_PROD_MODEL is None:
        _CACHED_PROD_MODEL = ProductionModelManager().get_best_model()
    if _CACHED_EMISSION_ENGINE is None:
        _CACHED_EMISSION_ENGINE = MaritimeEmissionEngine()
    if _CACHED_COMPLIANCE_ENGINE is None:
        _CACHED_COMPLIANCE_ENGINE = MaritimeComplianceEngine()
    return _CACHED_PROD_MODEL, _CACHED_EMISSION_ENGINE, _CACHED_COMPLIANCE_ENGINE


def fleet_objective(
    x: np.ndarray,
    assignments: Sequence[FleetAssignment],
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    context: dict[str, Any] | None = None,
    engines: tuple[Any, Any, Any] | None = None,
) -> float:
    """Evaluate fleet-level objective function unifying all three engines.

    Args:
        x: Flattened float decision vector: 2 values per assigned voyage (speed, fuel_idx).
        assignments: Sequence of FleetAssignment instances from the scheduler.
        weights: Tuple of (w1_cost, w2_emissions, w3_delay) weight multipliers.
        context: Optional dictionary containing voyage_specs and penalty rates.
        engines: Optional (model_engine, emission_engine, compliance_engine) tuple.

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
    hourly_delay_rate = float(ctx.get("delay_penalty_per_hour", 500.0))
    compliance_year = int(ctx.get("compliance_year", 2025))

    # Resolve engines (production model, emission engine, compliance engine)
    if engines is not None:
        prod_model, emission_engine, compliance_engine = engines
    else:
        prod_model, emission_engine, compliance_engine = get_cached_engines()

    total_fuel_cost = 0.0
    total_co2e = 0.0
    schedule_delay_penalty = 0.0

    # 1. Decode decision vector into per-voyage telemetry and collect batch records
    records: list[VoyageRecord] = []
    voyage_meta: list[tuple[str, float, float]] = []

    for i, assignment in enumerate(assigned_voyages):
        speed_knots = float(x[2 * i])
        fuel_idx = int(np.clip(round(float(x[2 * i + 1])), 0, len(SUPPORTED_FUEL_CHOICES) - 1))
        fuel_type = SUPPORTED_FUEL_CHOICES[fuel_idx]

        v_id = assignment.vessel_id
        c_id = assignment.cargo_id

        spec = voyage_specs.get(c_id, voyage_specs.get(v_id, {}))
        cargo_tons = float(spec.get("tons", 40000.0))
        vessel_dwt = float(spec.get("capacity", max(cargo_tons * 1.2, 50000.0)))
        distance_nm = float(spec.get("distance_nm", 1000.0))
        deadline_hours = float(spec.get("deadline_hours", 120.0))

        hours_at_sea = distance_nm / max(speed_knots, 1.0)

        # Construct VoyageRecord for model inference adhering to frozen schema
        record = VoyageRecord(
            voyage_id=f"VY-{v_id}-{c_id}",
            vessel_id=v_id,
            vessel_type="Bulk Carrier",
            vessel_dwt=vessel_dwt,
            cargo_tons=cargo_tons,
            distance_nm=distance_nm,
            speed_knots=speed_knots,
            hours_at_sea=hours_at_sea,
            fuel_type=fuel_type,
            weather_factor=1.0,
            sea_state=3,
            data_source="fleet_optimizer",
            is_synthetic=True,
            fuel_consumption=None,
            co2_emissions=None,
        )
        records.append(record)
        voyage_meta.append((fuel_type, hours_at_sea, deadline_hours))

    # 2. Engine 1: Predict fuel consumption for all voyages in a single vectorized batch
    pred_results = prod_model.predict(records)

    # 3. Engine 2 & 3: Calculate Well-to-Wake CO2e emissions and FuelEU penalty
    for i, pred_res in enumerate(pred_results):
        fuel_type, hours_at_sea, deadline_hours = voyage_meta[i]
        predicted_fuel = float(pred_res.predicted_fuel_consumption)

        emiss_res = emission_engine.calculate_wtw(predicted_fuel, fuel_type)
        co2e = float(emiss_res.co2e)

        lcv = FUEL_LCV_MJ_PER_TON.get(fuel_type, 42700.0)
        energy_used_mj = predicted_fuel * lcv
        ghg_intensity = (co2e * 1e6) / energy_used_mj if energy_used_mj > 0.0 else 0.0

        comp_res = compliance_engine.evaluate_fueleu(
            ghg_intensity=ghg_intensity,
            energy_used_mj=energy_used_mj,
            year=compliance_year,
        )
        fueleu_penalty = float(comp_res.compliance_score) if not comp_res.fueleu_pass else 0.0

        # Accumulate costs, emissions, and schedule delays
        price_per_ton = STANDARD_FUEL_PRICES_USD.get(fuel_type, 650.0)
        voyage_fuel_cost = (predicted_fuel * price_per_ton) + fueleu_penalty
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
