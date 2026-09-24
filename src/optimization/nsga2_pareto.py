"""Multi-Objective Pareto solver with NSGA-II environmental selection and differential-evolution variation.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Solves the bi-objective optimization problem:
  f1(x) = Total Fuel Cost (USD)
  f2(x) = Total Lifecycle CO2e Emissions (metric tons)
over the scheduled fleet decision vector on the medium problem size tier.
Combines differential-evolution variation (DE/rand/1/bin continuous search) with
canonical Deb et al. (2002) fast non-dominated sorting and crowding-distance
environmental selection (hybrid DE-NSGA-II / DEMO architecture).
"""

from collections.abc import Sequence
import logging
import time
from typing import Any

import numpy as np

from contracts.schemas import FleetAssignment, VoyageRecord
from src.optimization.fleet_objective import (
    FUEL_LCV_MJ_PER_TON,
    ML_ROUTED_FUELS,
    PHYSICS_ROUTED_FUELS,
    STANDARD_FUEL_PRICES_USD,
    SUPPORTED_FUEL_CHOICES,
    get_cached_engines,
)
from src.optimization.fleet_optimizer import generate_fleet_problem
from src.physics.fuel_physics_engine import MaritimeFuelPhysicsEngine

logger = logging.getLogger("maritime_system")


def evaluate_bi_objective(
    x: np.ndarray,
    assignments: Sequence[FleetAssignment],
    context: dict[str, Any],
    engines: tuple[Any, ...],
) -> tuple[float, float]:
    """Evaluate dual objectives: (total_fuel_cost, total_co2e)."""
    assigned_voyages = [a for a in assignments if a.assigned]
    if len(engines) == 4:
        prod_model, emission_engine, compliance_engine, physics_engine = engines
    elif len(engines) == 3:
        prod_model, emission_engine, compliance_engine = engines
        physics_engine = MaritimeFuelPhysicsEngine()
    else:
        raise ValueError(f"Expected 3 or 4 engines, got {len(engines)}")

    voyage_specs = context.get("voyage_specs", {})
    vessel_specs = context.get("vessel_specs", context.get("vessels", {}))
    compliance_year = int(context.get("compliance_year", 2025))

    total_cost = 0.0
    total_co2e = 0.0

    # 1. Decode decision vector into per-voyage telemetry and route ML vs Physics
    ml_records: list[VoyageRecord] = []
    ml_indices: list[int] = []
    voyage_data: list[dict[str, Any]] = []

    for i, a in enumerate(assigned_voyages):
        speed = float(x[2 * i])
        fuel_idx = int(np.clip(round(float(x[2 * i + 1])), 0, len(SUPPORTED_FUEL_CHOICES) - 1))
        fuel_type = SUPPORTED_FUEL_CHOICES[fuel_idx]

        # Resolve hierarchical voyage and vessel specifications from context
        v_entry = vessel_specs.get(a.vessel_id, {}) if isinstance(vessel_specs, dict) else {}
        if not v_entry and isinstance(voyage_specs, dict):
            v_entry = voyage_specs.get(a.vessel_id, {})
        c_entry = voyage_specs.get(a.cargo_id, {}) if isinstance(voyage_specs, dict) else {}
        pair_entry = (
            voyage_specs.get(f"{a.vessel_id}_{a.cargo_id}", voyage_specs.get(f"{a.vessel_id}-{a.cargo_id}", {}))
            if isinstance(voyage_specs, dict)
            else {}
        )
        spec = {**v_entry, **c_entry, **pair_entry}

        cargo_tons = float(spec.get("cargo_tons", spec.get("tons", spec.get("weight", 40000.0))))
        vessel_dwt = float(spec.get("vessel_dwt", spec.get("capacity", spec.get("dwt", max(cargo_tons * 1.2, 50000.0)))))
        distance_nm = float(spec.get("distance_nm", 1000.0))
        weather_factor = float(spec.get("weather_factor", 1.0))
        sea_state = int(spec.get("sea_state", 3))
        vessel_type = str(spec.get("vessel_type", "Bulk Carrier"))
        hours_at_sea = distance_nm / max(speed, 1.0)

        v_info: dict[str, Any] = {
            "fuel_type": fuel_type,
            "hours_at_sea": hours_at_sea,
            "fuel_tons": 0.0,
            "energy_mwh": 0.0,
        }

        if fuel_type in ML_ROUTED_FUELS:
            # Route ML-trained fuels (Diesel, LNG, Methanol) through ProductionModelManager
            record = VoyageRecord(
                voyage_id=f"VY-{a.vessel_id}-{a.cargo_id}",
                vessel_id=a.vessel_id,
                vessel_type=vessel_type,
                vessel_dwt=vessel_dwt,
                cargo_tons=cargo_tons,
                distance_nm=distance_nm,
                speed_knots=speed,
                hours_at_sea=hours_at_sea,
                fuel_type=fuel_type,
                weather_factor=weather_factor,
                sea_state=sea_state,
                data_source="nsga2_solver",
                is_synthetic=True,
                fuel_consumption=None,
                co2_emissions=None,
            )
            ml_records.append(record)
            ml_indices.append(i)
        else:
            # Route novel alternative fuels (Hydrogen, Ammonia, ShorePower) through FuelPhysicsEngine
            # ProductionModelManager is NEVER invoked for these fuels
            fuel_val = physics_engine.calculate_fuel_use(
                distance_nm=distance_nm,
                speed_knots=speed,
                cargo_tons=cargo_tons,
                weather_factor=weather_factor,
                fuel_type=fuel_type,
                vessel_dwt=vessel_dwt,
            )
            v_info["fuel_tons"] = float(fuel_val)
            if fuel_type == "ShorePower":
                v_info["energy_mwh"] = float(physics_engine.last_energy_mwh)

        voyage_data.append(v_info)

    # 2. Batch predict ML-routed voyages
    if ml_records:
        preds = prod_model.predict(ml_records)
        for k, pred in enumerate(preds):
            orig_idx = ml_indices[k]
            voyage_data[orig_idx]["fuel_tons"] = float(pred.predicted_fuel_consumption)

    # 3. Compute cost and emissions
    for v_info in voyage_data:
        fuel_type = v_info["fuel_type"]
        fuel_t = v_info["fuel_tons"]
        price = STANDARD_FUEL_PRICES_USD.get(fuel_type, 650.0)

        if fuel_type == "ShorePower":
            energy_mwh = v_info["energy_mwh"]
            total_cost += energy_mwh * price
            # Zero operational CO2e for shore power
        else:
            emiss = emission_engine.calculate_wtw(fuel_t, fuel_type)
            co2e_t = float(emiss.co2e)

            lcv = FUEL_LCV_MJ_PER_TON.get(fuel_type, 42700.0)
            energy_mj = fuel_t * lcv
            intensity = (co2e_t * 1e6) / energy_mj if energy_mj > 0 else 0.0

            comp = compliance_engine.evaluate_fueleu(intensity, energy_mj, year=compliance_year)
            fueleu_penalty = float(comp.penalty_eur)

            total_cost += (fuel_t * price) + fueleu_penalty
            total_co2e += co2e_t

    return float(total_cost), float(total_co2e)


def non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    """Fast non-dominated sorting algorithm for Pareto front identification."""
    n = len(objectives)
    domination_counts = np.zeros(n, dtype=int)
    dominated_solutions: list[list[int]] = [[] for _ in range(n)]
    fronts: list[list[int]] = [[]]

    for p in range(n):
        for q in range(p + 1, n):
            # True if p dominates q
            p_dom_q = np.all(objectives[p] <= objectives[q]) and np.any(objectives[p] < objectives[q])
            # True if q dominates p
            q_dom_p = np.all(objectives[q] <= objectives[p]) and np.any(objectives[q] < objectives[p])

            if p_dom_q:
                dominated_solutions[p].append(q)
                domination_counts[q] += 1
            elif q_dom_p:
                dominated_solutions[q].append(p)
                domination_counts[p] += 1

        if domination_counts[p] == 0:
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front: list[int] = []
        for p in fronts[i]:
            for q in dominated_solutions[p]:
                domination_counts[q] -= 1
                if domination_counts[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return [f for f in fronts if len(f) > 0]


def calculate_crowding_distance(objectives: np.ndarray) -> np.ndarray:
    """Calculate NSGA-II crowding distance for a front in objective space.

    Conforms strictly to Deb et al. (2002), Section III-B (Crowding-Distance Computation).
    For each objective:
      1. Boundary solutions (minimal and maximal objective values) are assigned infinity (inf).
      2. Intermediate solutions are assigned the normalized cuboid distance between their neighbors:
         d_i = sum_m (f_m(i+1) - f_m(i-1)) / (f_m^max - f_m^min)

    Args:
        objectives: 2D array of shape (L, M) containing objective values for L individuals.

    Returns:
        1D array of shape (L,) containing crowding distances. Boundary individuals have distance = inf.
    """
    objectives = np.asarray(objectives, dtype=float)
    if objectives.ndim != 2:
        raise ValueError(f"objectives must be a 2D array, got shape {objectives.shape}")

    n_individuals, n_objectives = objectives.shape
    if n_individuals == 0:
        return np.array([], dtype=float)
    if n_individuals <= 2:
        return np.full(n_individuals, np.inf, dtype=float)

    distances = np.zeros(n_individuals, dtype=float)

    for m in range(n_objectives):
        sorted_indices = np.argsort(objectives[:, m])
        obj_min = float(objectives[sorted_indices[0], m])
        obj_max = float(objectives[sorted_indices[-1], m])
        obj_range = obj_max - obj_min

        # Boundary solutions receive infinite distance
        distances[objectives[:, m] == obj_min] = np.inf
        distances[objectives[:, m] == obj_max] = np.inf

        if obj_range > 0:
            for k in range(1, n_individuals - 1):
                idx = sorted_indices[k]
                if not np.isinf(distances[idx]):
                    diff = float(objectives[sorted_indices[k + 1], m] - objectives[sorted_indices[k - 1], m])
                    distances[idx] += diff / obj_range

    return distances


class ParetoFleetOptimizer:
    """Multi-objective fleet optimization solver using NSGA-II environmental selection with DE variation."""

    def __init__(self, random_state: int = 42) -> None:
        """Initialize Pareto optimizer."""
        self.random_state = random_state
        self.logger = logger
        self.engines = get_cached_engines()

    def optimize(
        self,
        assignments: list[Any],
        context: dict[str, Any],
        population_size: int = 30,
        max_generations: int = 50,
    ) -> dict[str, Any]:
        """Execute bi-objective optimization using NSGA-II environmental selection with DE variation."""
        assigned_voyages = [a for a in assignments if getattr(a, "assigned", True)]
        n_voyages = len(assigned_voyages)
        if n_voyages == 0:
            return {"pareto_front": [], "runtime_seconds": 0.0}

        dim = 2 * n_voyages
        lb = np.array([10.0, 0.0] * n_voyages, dtype=float)
        ub = np.array([20.0, 5.0] * n_voyages, dtype=float)

        rng = np.random.default_rng(self.random_state)
        start_time = time.perf_counter()

        # 1. Initialize random population
        pop = rng.uniform(lb, ub, size=(population_size, dim))
        objs = np.zeros((population_size, 2), dtype=float)

        for i in range(population_size):
            objs[i] = evaluate_bi_objective(pop[i], assignments, context, self.engines)

        # 2. Evolutionary generations
        for gen in range(max_generations):
            # Create offspring via differential-evolution variation (DE/rand/1/bin mutation & binomial crossover)
            offspring = np.zeros_like(pop)
            for i in range(population_size):
                idxs = rng.choice(population_size, size=3, replace=False)
                mutant = pop[idxs[0]] + 0.8 * (pop[idxs[1]] - pop[idxs[2]])
                cross_mask = rng.uniform(0.0, 1.0, size=dim) < 0.7
                offspring[i] = np.where(cross_mask, mutant, pop[i])
                offspring[i] = np.clip(offspring[i], lb, ub)

            offspring_objs = np.zeros((population_size, 2), dtype=float)
            for i in range(population_size):
                offspring_objs[i] = evaluate_bi_objective(offspring[i], assignments, context, self.engines)

            # Combine parent and offspring (2N population)
            combined_pop = np.vstack([pop, offspring])
            combined_objs = np.vstack([objs, offspring_objs])

            # Non-dominated sort
            fronts = non_dominated_sort(combined_objs)

            # Select best N individuals via non-dominated fronts and canonical crowding distance
            new_pop: list[np.ndarray] = []
            new_objs: list[np.ndarray] = []
            for front in fronts:
                if len(new_pop) + len(front) <= population_size:
                    for idx in front:
                        new_pop.append(combined_pop[idx])
                        new_objs.append(combined_objs[idx])
                else:
                    needed = population_size - len(new_pop)
                    front_objs = combined_objs[front]
                    crowding_dist = calculate_crowding_distance(front_objs)
                    # Descending sort by crowding distance to prioritize boundary and isolated solutions
                    sorted_front_order = np.argsort(-crowding_dist)
                    for i in sorted_front_order[:needed]:
                        idx = front[i]
                        new_pop.append(combined_pop[idx])
                        new_objs.append(combined_objs[idx])
                    break

            pop = np.array(new_pop)
            objs = np.array(new_objs)

        runtime = time.perf_counter() - start_time
        final_fronts = non_dominated_sort(objs)
        pareto_indices = final_fronts[0] if final_fronts else []

        pareto_points = objs[pareto_indices]
        # Sort by cost ascending
        sort_order = np.argsort(pareto_points[:, 0])
        sorted_points = pareto_points[sort_order]

        return {
            "assigned_voyages": n_voyages,
            "runtime_seconds": round(runtime, 4),
            "generations": max_generations,
            "population_size": population_size,
            "pareto_front_size": len(pareto_indices),
            "pareto_front": [
                {"fuel_cost_usd": round(float(pt[0]), 2), "co2e_tons": round(float(pt[1]), 2)}
                for pt in sorted_points
            ],
        }

    def solve_medium_pareto(
        self,
        population_size: int = 20,
        generations: int = 15,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Solve bi-objective trade-off for medium fleet tier (20 vessels / 50 cargos)."""
        _, _, constraints, assignments = generate_fleet_problem(
            n_vessels=20, n_cargos=50, seed=seed
        )
        context = {
            "voyage_specs": constraints["cargos"],
            "vessel_specs": constraints.get("vessels", {}),
            "compliance_year": 2025,
        }
        res = self.optimize(
            assignments=assignments,
            context=context,
            population_size=population_size,
            max_generations=generations,
        )
        res["tier"] = "medium"
        res["vessels"] = 20
        res["cargos"] = 50
        return res


def run_nsga2_pareto(
    assignments: list[Any],
    context: dict[str, Any],
    population_size: int = 30,
    max_generations: int = 50,
    random_state: int = 42,
) -> dict[str, Any]:
    """Convenience wrapper around ParetoFleetOptimizer for scripted use."""
    optimizer = ParetoFleetOptimizer(random_state=random_state)
    return optimizer.optimize(
        assignments=assignments,
        context=context,
        population_size=population_size,
        max_generations=max_generations,
    )
