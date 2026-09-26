"""Base interface and unified candidate evaluation for benchmark optimization solvers.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 3 Deliverable: Ensures fair, apples-to-apples comparison across all optimization solvers
by sharing the exact same objective formulation, physical equations, operational constraints,
and regulatory compliance models.
"""

from abc import ABC, abstractmethod
import hashlib
import json
import logging
import time
from collections.abc import Sequence
from typing import Any, Final

import numpy as np

from contracts.constants import (
    DEFAULT_EUR_TO_USD_FX_RATE,
    FUEL_PRICES_USD_PER_TON,
    FuelType,
)
from contracts.schemas import (
    BenchmarkResult,
    OptimizationScenario,
)
from src.compliance.compliance_engine import MaritimeComplianceEngine
from src.operations.demand_satisfaction_engine import CargoDemandSatisfactionEngine
from src.operations.reliability_engine import ScheduleReliabilityEngine
from src.physics.fuel_physics_engine import MaritimeFuelPhysicsEngine
from src.prediction.emission_engine import MaritimeEmissionEngine

logger = logging.getLogger("maritime_system")


def compute_scenario_fingerprint(scenario: OptimizationScenario) -> str:
    """Compute an immutable deterministic SHA-256 fingerprint for scenario physics and economics."""
    payload = {
        "demand": float(scenario.forecasted_demand if scenario.forecasted_demand is not None else scenario.cargo_demand),
        "distance": float(scenario.route_distance),
        "deadline": float(scenario.deadline_hours),
        "weather": float(scenario.weather_factor),
        "carbon": float(scenario.carbon_price),
        "budget": float(scenario.budget),
        "weights": tuple(float(w) for w in scenario.weights),
        "service_level": float(scenario.service_level),
        "max_transition": float(scenario.max_transition_rate),
        "reliability": float(scenario.target_reliability),
        "port_delay": float(scenario.port_delay_factor),
        "scenario_id": str(scenario.scenario_id or ""),
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class BenchmarkEvaluationCache:
    """Isolated evaluation cache for an individual solver execution.

    Cross-solver cache sharing is prohibited to preserve fair runtime and search comparison.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.hits: int = 0
        self.misses: int = 0

    def get(self, key: tuple[Any, ...]) -> dict[str, Any] | None:
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, key: tuple[Any, ...], value: dict[str, Any]) -> None:
        self._cache[key] = value

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round((self.hits / total) * 100.0, 2) if total > 0 else 0.0

    def clear(self) -> None:
        self._cache.clear()
        self.hits = 0
        self.misses = 0

VESSEL_SPECS: Final[dict[str, dict[str, Any]]] = {
    "feeder": {
        "dwt": 12000.0,
        "design_speed_knots": 13.0,
        "admiralty_coeff": 420.0,
        "annual_voyages": 35,
        "daily_charter_usd": 12000.0,
        "capex_usd": 25_000_000.0,
        "max_available": 20,
    },
    "medium": {
        "dwt": 45000.0,
        "design_speed_knots": 14.5,
        "admiralty_coeff": 520.0,
        "annual_voyages": 20,
        "daily_charter_usd": 22000.0,
        "capex_usd": 50_000_000.0,
        "max_available": 15,
    },
    "large": {
        "dwt": 120000.0,
        "design_speed_knots": 15.5,
        "admiralty_coeff": 650.0,
        "annual_voyages": 10,
        "daily_charter_usd": 38000.0,
        "capex_usd": 95_000_000.0,
        "max_available": 10,
    },
}

FUEL_CAPEX_MULTIPLIER: Final[dict[str, float]] = {
    "diesel": 1.00,
    "lng": 1.15,
    "methanol": 1.20,
    "hydrogen": 1.40,
    "ammonia": 1.35,
}


class BaseBenchmarkSolver(ABC):
    """Abstract base class for all benchmark optimization solvers."""

    def __init__(self, solver_name: str) -> None:
        """Initialize benchmark solver with identification name and engines."""
        self.solver_name = solver_name
        self.physics_engine = MaritimeFuelPhysicsEngine()
        self.emission_engine = MaritimeEmissionEngine()
        self.compliance_engine = MaritimeComplianceEngine()
        self.demand_engine = CargoDemandSatisfactionEngine()
        self.reliability_engine = ScheduleReliabilityEngine()
        self.logger = logger

    @abstractmethod
    def solve(
        self,
        scenario: OptimizationScenario,
        max_iterations: int = 50,
        seed: int = 42,
    ) -> BenchmarkResult:
        """Execute benchmark solver against the unified operational scenario.

        Args:
            scenario: Comprehensive OptimizationScenario context.
            max_iterations: Maximum search iterations or generation count.
            seed: Reproducibility random seed.

        Returns:
            BenchmarkResult capturing standardized performance metrics.
        """
        pass

    @staticmethod
    def allocate_fuel_mix(
        total_vessels: int,
        alt_count: int,
        scenario: OptimizationScenario,
        shares: Sequence[float] | None = None,
    ) -> dict[str, int]:
        """Canonical 5-fuel allocation preserving exact vessel counts and scenario intent.

        Distributes vessels across (diesel, lng, methanol, hydrogen, ammonia) such that:
            sum(fuel_mix.values()) == total_vessels
            sum(alt_fuels) == alt_count
        """
        total = max(1, int(total_vessels))
        alts = min(max(0, int(alt_count)), total)
        diesel = total - alts

        alt_fuels = ["lng", "methanol", "hydrogen", "ammonia"]

        if alts == 0:
            return {
                "diesel": total,
                "lng": 0,
                "methanol": 0,
                "hydrogen": 0,
                "ammonia": 0,
            }

        scen_id = (scenario.scenario_id or "").upper()

        if shares is not None:
            if len(shares) == 5:
                # 5-fuel allocation directly across (diesel, lng, methanol, hydrogen, ammonia)
                raw_w = np.array(shares, dtype=float)
                if np.sum(raw_w) <= 0 or not np.all(raw_w >= 0):
                    raw_w = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
                norm_w5 = raw_w / np.sum(raw_w)
                # Enforce max transition rate constraint on alternative fuel proportion
                alt_prop = float(np.sum(norm_w5[1:]))
                if alt_prop > scenario.max_transition_rate and scenario.max_transition_rate < 1.0:
                    scale = scenario.max_transition_rate / max(alt_prop, 1e-6)
                    norm_w5[1:] *= scale
                    norm_w5[0] = 1.0 - float(np.sum(norm_w5[1:]))

                exact_counts5 = norm_w5 * total
                base_counts5 = np.floor(exact_counts5).astype(int)
                rem5 = total - int(np.sum(base_counts5))
                if rem5 > 0:
                    remainders5 = exact_counts5 - base_counts5
                    sorted_indices5 = sorted(range(5), key=lambda i: (-remainders5[i], i))
                    for idx in sorted_indices5[:rem5]:
                        base_counts5[idx] += 1

                return {
                    "diesel": int(base_counts5[0]),
                    "lng": int(base_counts5[1]),
                    "methanol": int(base_counts5[2]),
                    "hydrogen": int(base_counts5[3]),
                    "ammonia": int(base_counts5[4]),
                }
            elif len(shares) == len(alt_fuels):
                raw_w = np.array(shares, dtype=float)
                if np.sum(raw_w) > 0 and np.all(raw_w >= 0):
                    norm_w = raw_w / np.sum(raw_w)
                else:
                    norm_w = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
            else:
                norm_w = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
        elif "HYDROGEN" in scen_id:
            norm_w = np.array([0.0, 0.0, 1.0, 0.0], dtype=float)
        elif "AMMONIA" in scen_id:
            norm_w = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        elif "METHANOL" in scen_id:
            norm_w = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)
        elif "LNG" in scen_id:
            norm_w = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        else:
            # Default general benchmark: balanced representation across all 4 alternative fuels
            norm_w = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)

        # Largest remainder method (Hare-Niemeyer) for exact integer representation
        exact_counts = norm_w * alts
        base_counts = np.floor(exact_counts).astype(int)
        remainder = alts - int(np.sum(base_counts))

        if remainder > 0:
            remainders = exact_counts - base_counts
            sorted_indices = sorted(range(len(alt_fuels)), key=lambda i: (-remainders[i], i))
            for idx in sorted_indices[:remainder]:
                base_counts[idx] += 1

        allocated: dict[str, int] = {
            "diesel": int(diesel),
            "lng": int(base_counts[0]),
            "methanol": int(base_counts[1]),
            "hydrogen": int(base_counts[2]),
            "ammonia": int(base_counts[3]),
        }

        # Sanity validation
        if sum(allocated.values()) != total:
            raise ValueError(
                f"Fuel mix sum {sum(allocated.values())} does not match total vessels {total}"
            )
        if sum(allocated[k] for k in alt_fuels) != alts:
            raise ValueError(
                f"Alternative fuel sum {sum(allocated[k] for k in alt_fuels)} does not match alt_count {alts}"
            )

        return allocated

    @classmethod
    def decode_and_repair(
        cls,
        vec: np.ndarray,
        scenario: OptimizationScenario,
    ) -> tuple[int, int, int, dict[str, int], float, float]:
        """Canonical 9D continuous-to-discrete decoding and deterministic repair operator.

        Returns:
            (x_feeder, x_medium, x_large, fuel_mix, speed_knots, repair_distance)
        """
        # 1. Bounds clipping & integer rounding
        raw_f = float(vec[0])
        raw_m = float(vec[1])
        raw_l = float(vec[2])
        xf = int(np.clip(round(raw_f), 0, VESSEL_SPECS["feeder"]["max_available"]))
        xm = int(np.clip(round(raw_m), 0, VESSEL_SPECS["medium"]["max_available"]))
        xl = int(np.clip(round(raw_l), 0, VESSEL_SPECS["large"]["max_available"]))

        # 2. Total fleet repair (at least 1 vessel if all rounded to zero)
        tot = xf + xm + xl
        if tot == 0:
            xm = 1
            tot = 1

        # 3. Alternative fuel count & 5-fuel allocation
        alt_ratio = float(np.clip(vec[3], 0.0, scenario.max_transition_rate))
        alt_count = int(round(tot * alt_ratio))

        if len(vec) >= 9:
            raw_shares = [float(vec[4]), float(vec[5]), float(vec[6]), float(vec[7])]
            speed = float(np.clip(vec[8], 9.5, 17.5))
        else:
            raw_shares = None
            speed = float(np.clip(vec[4], 9.5, 17.5))

        fuel_mix = cls.allocate_fuel_mix(tot, alt_count, scenario, shares=raw_shares)

        # 4. Compute repair distance: L2 distance between raw continuous coordinates and decoded discrete coordinates
        repaired_coords = [
            float(xf),
            float(xm),
            float(xl),
            float(sum(fuel_mix[k] for k in ("lng", "methanol", "hydrogen", "ammonia")) / tot),
            float(fuel_mix["lng"] / max(1, alt_count)),
            float(fuel_mix["methanol"] / max(1, alt_count)),
            float(fuel_mix["hydrogen"] / max(1, alt_count)),
            float(fuel_mix["ammonia"] / max(1, alt_count)),
            speed,
        ]
        raw_coords = [float(vec[i]) if i < len(vec) else 0.0 for i in range(9)]
        repair_distance = float(np.linalg.norm(np.array(repaired_coords) - np.array(raw_coords)))

        return xf, xm, xl, fuel_mix, speed, round(repair_distance, 4)

    def evaluate_candidate(
        self,
        x_f: int,
        x_m: int,
        x_l: int,
        fuel_mix: dict[str, int],
        speed_knots: float,
        scenario: OptimizationScenario,
        cache: BenchmarkEvaluationCache | None = None,
    ) -> dict[str, Any]:
        """Unified candidate evaluation under exact maritime physical and operational constraints.

        Returns a dictionary of raw metrics:
            objective_score, fuel_tons, cost_usd, emissions_tons,
            reliability_score, demand_satisfaction_rate, is_feasible, penalties.
        """
        fp = compute_scenario_fingerprint(scenario)
        speed = float(np.clip(speed_knots, 9.0, 18.0))
        speed_rounded = round(speed, 2)

        cache_key = (
            int(x_f),
            int(x_m),
            int(x_l),
            int(fuel_mix.get("diesel", 0)),
            int(fuel_mix.get("lng", 0)),
            int(fuel_mix.get("methanol", 0)),
            int(fuel_mix.get("hydrogen", 0)),
            int(fuel_mix.get("ammonia", 0)),
            speed_rounded,
            fp,
        )

        if cache is not None:
            cached_val = cache.get(cache_key)
            if cached_val is not None:
                return cached_val
        total_vessels = x_f + x_m + x_l
        fuel_prices = scenario.fuel_prices or FUEL_PRICES_USD_PER_TON
        w_cost, w_fuel, w_emiss = scenario.weights

        # 1. Zero vessels penalty
        if total_vessels == 0:
            return {
                "objective_score": 1e9,
                "fuel_consumption": 0.0,
                "operational_cost": 0.0,
                "emissions": 0.0,
                "reliability_score": 0.0,
                "demand_satisfaction_rate": 0.0,
                "feasible_solution": False,
                "failure_reason": "ZERO_VESSELS",
            }

        # 2. Demand Satisfaction & Fleet Capacity
        ann_capacity = (
            x_f * VESSEL_SPECS["feeder"]["dwt"] * 0.85 * VESSEL_SPECS["feeder"]["annual_voyages"]
            + x_m * VESSEL_SPECS["medium"]["dwt"] * 0.85 * VESSEL_SPECS["medium"]["annual_voyages"]
            + x_l * VESSEL_SPECS["large"]["dwt"] * 0.85 * VESSEL_SPECS["large"]["annual_voyages"]
        )

        effective_demand = scenario.forecasted_demand if scenario.forecasted_demand is not None else scenario.cargo_demand
        dem_metrics = self.demand_engine.evaluate_demand_satisfaction(
            scenario=scenario,
            delivered_cargo=ann_capacity,
        )

        demand_deficit = max(0.0, (effective_demand * scenario.service_level) - ann_capacity)
        demand_penalty = (demand_deficit / max(effective_demand, 1.0)) * 500.0

        # 3. Capital & Charter Costs
        base_capital = (
            x_f * VESSEL_SPECS["feeder"]["capex_usd"] * 0.10
            + x_m * VESSEL_SPECS["medium"]["capex_usd"] * 0.10
            + x_l * VESSEL_SPECS["large"]["capex_usd"] * 0.10
        )

        weighted_capex_mult = sum(
            fuel_mix.get(k, 0) * FUEL_CAPEX_MULTIPLIER.get(k, 1.0)
            for k in ("diesel", "lng", "methanol", "hydrogen", "ammonia")
        ) / total_vessels

        total_capital = base_capital * weighted_capex_mult
        budget_excess = max(0.0, total_capital - scenario.budget)
        budget_penalty = (budget_excess / max(scenario.budget, 1.0)) * 1000.0

        # 4. Green Transition Rate Constraint
        alt_count = sum(fuel_mix.get(k, 0) for k in ("lng", "methanol", "hydrogen", "ammonia"))
        alt_fraction = alt_count / total_vessels
        transition_excess = max(0.0, alt_fraction - scenario.max_transition_rate)
        transition_penalty = transition_excess * 1000.0

        # 5. Speed, Transit Time & Delays
        speed = float(np.clip(speed_knots, 9.0, 18.0))
        transit_hours = (scenario.route_distance / speed) * scenario.weather_factor
        port_delay = max(0.0, (scenario.port_delay_factor - 1.0) * 12.0)
        total_duration = transit_hours + port_delay
        delay_hours = max(0.0, total_duration - scenario.deadline_hours)

        # 6. Fuel & Emissions
        weighted_dwt = (
            x_f * VESSEL_SPECS["feeder"]["dwt"]
            + x_m * VESSEL_SPECS["medium"]["dwt"]
            + x_l * VESSEL_SPECS["large"]["dwt"]
        ) / total_vessels

        weighted_adm = (
            x_f * VESSEL_SPECS["feeder"]["admiralty_coeff"]
            + x_m * VESSEL_SPECS["medium"]["admiralty_coeff"]
            + x_l * VESSEL_SPECS["large"]["admiralty_coeff"]
        ) / total_vessels

        total_fuel = 0.0
        total_emissions = 0.0
        total_bunker_cost = 0.0

        fuel_type_map = {
            "diesel": FuelType.DIESEL.value,
            "lng": FuelType.LNG.value,
            "methanol": FuelType.METHANOL.value,
            "hydrogen": FuelType.HYDROGEN.value,
            "ammonia": FuelType.AMMONIA.value,
        }

        # Representative single-voyage fuel and emissions scaled across active vessels
        for f_key, count in fuel_mix.items():
            if count <= 0:
                continue
            f_type = fuel_type_map.get(f_key, FuelType.DIESEL.value)
            f_price = fuel_prices.get(f_type, 650.0)

            fuel_vsl = self.physics_engine.calculate_fuel_use(
                distance_nm=scenario.route_distance,
                speed_knots=speed,
                cargo_tons=weighted_dwt * 0.85,
                weather_factor=scenario.weather_factor,
                fuel_type=f_type,
                vessel_dwt=weighted_dwt,
                admiralty_coeff=weighted_adm,
            )
            emiss_vsl = float(self.emission_engine.calculate_wtw(fuel_vsl, f_type).co2e)

            total_fuel += fuel_vsl * count
            total_emissions += emiss_vsl * count
            total_bunker_cost += fuel_vsl * count * f_price

        carbon_cost = total_emissions * scenario.carbon_price
        delay_cost = delay_hours * total_vessels * 500.0  # $500/hr delay demurrage
        total_operational_cost = total_capital + total_bunker_cost + carbon_cost + delay_cost

        # 7. Schedule Reliability
        total_voyages = max(1, total_vessels)
        simulated_delays = [delay_hours] * total_voyages

        assert len(simulated_delays) == total_voyages, (
            f"Observation cardinality violation: len(simulated_delays)={len(simulated_delays)} "
            f"must equal total_voyages={total_voyages}"
        )
        if len(simulated_delays) != total_voyages:
            raise ValueError(
                f"Benchmark evaluation cardinality mismatch: len(simulated_delays)={len(simulated_delays)} != total_voyages={total_voyages}"
            )
        if total_voyages < 1:
            raise ValueError(f"Benchmark evaluation invalid voyages: total_voyages={total_voyages} must be >= 1")
        if not all(np.isfinite(d) for d in simulated_delays):
            raise ValueError("Benchmark evaluation simulated delays contain non-finite values")

        rel_metrics = self.reliability_engine.evaluate_schedule_reliability(
            scenario=scenario,
            simulated_delays=simulated_delays,
            missed_voyages=0,
            total_voyages=total_voyages,
        )

        reliability_deficit = max(0.0, scenario.target_reliability - rel_metrics.reliability_score)
        reliability_penalty = reliability_deficit * 20.0

        # 8. Composite Objective Function
        normalized_cost = total_operational_cost / 1_000_000.0
        normalized_fuel = total_fuel / 100.0
        normalized_emiss = total_emissions / 300.0

        composite_score = (
            w_cost * normalized_cost
            + w_fuel * normalized_fuel
            + w_emiss * normalized_emiss
            + demand_penalty
            + budget_penalty
            + transition_penalty
            + reliability_penalty
        )

        is_feasible = (
            demand_deficit <= 1e-4
            and budget_excess <= 1e-4
            and transition_excess <= 1e-4
            and reliability_deficit <= 1e-4
        )

        res_dict = {
            "objective_score": round(composite_score, 4),
            "fuel_consumption": round(total_fuel, 2),
            "operational_cost": round(total_operational_cost, 2),
            "emissions": round(total_emissions, 2),
            "reliability_score": round(rel_metrics.reliability_score, 2),
            "demand_satisfaction_rate": round(dem_metrics.demand_satisfaction_rate, 4),
            "feasible_solution": is_feasible,
            "fleet_mix": {**fuel_mix, "feeder": x_f, "medium": x_m, "large": x_l},
            "speed_knots": speed,
            "ann_capacity": round(ann_capacity, 2),
        }

        if cache is not None:
            cache.put(cache_key, res_dict)

        return res_dict
