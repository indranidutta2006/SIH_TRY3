"""Fleet-level optimization runner and scalability benchmarking engine.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Coordinates fleet scheduling, problem generation, parameter bounding, and equal-budget
comparisons between QPSO and PSO across small, medium, and large fleet operational tiers.
"""

from collections.abc import Sequence
import logging
import time
from typing import Any, Final

import numpy as np

from contracts.constants import OptimizerType
from contracts.exceptions import OptimizationError
from contracts.schemas import FleetAssignment, OptimizationResult
from src.optimization.fleet_objective import fleet_objective, get_cached_engines
from src.optimization.pso_baseline import PSOOptimizer
from src.optimization.qpso import QPSOOptimizer
from src.scheduler.fleet_scheduler import FleetScheduler

logger = logging.getLogger("maritime_system")

FLEET_TIER_CONFIGS: Final[dict[str, dict[str, Any]]] = {
    "small": {
        "n_vessels": 5,
        "n_cargos": 10,
        "population_size": 15,
        "max_iterations": 20,  # 300 evaluations
    },
    "medium": {
        "n_vessels": 20,
        "n_cargos": 50,
        "population_size": 20,
        "max_iterations": 25,  # 500 evaluations
    },
    "large": {
        "n_vessels": 50,
        "n_cargos": 200,
        "population_size": 25,
        "max_iterations": 30,  # 750 evaluations
    },
}


def generate_fleet_problem(
    n_vessels: int,
    n_cargos: int,
    seed: int = 42,
) -> tuple[list[str], list[str], dict[str, Any], Sequence[FleetAssignment]]:
    """Deterministically generate fleet constraints and initial feasible assignment schedule.

    Args:
        n_vessels: Number of commercial cargo vessels.
        n_cargos: Number of pending cargo shipments.
        seed: Random seed for deterministic generation.

    Returns:
        Tuple of (vessel_ids, cargo_ids, constraints_dict, assignments).
    """
    rng = np.random.default_rng(seed)

    vessel_ids = [f"VSL-{i+1:03d}" for i in range(n_vessels)]
    cargo_ids = [f"CRG-{j+1:04d}" for j in range(n_cargos)]

    vessel_types = ("Bulk Carrier", "Container Ship", "Tanker", "General Cargo")
    vessels_spec: dict[str, dict[str, Any]] = {}
    for i, vid in enumerate(vessel_ids):
        capacity = float(rng.choice([35000.0, 55000.0, 75000.0, 110000.0]))
        vtype = vessel_types[i % len(vessel_types)]
        vessels_spec[vid] = {
            "capacity": capacity,
            "vessel_dwt": capacity,
            "vessel_type": vtype,
        }

    cargos_spec: dict[str, dict[str, Any]] = {}
    for j, cid in enumerate(cargo_ids):
        tons = float(rng.uniform(15000.0, 65000.0))
        distance_nm = float(rng.uniform(800.0, 2500.0))
        transit_h = distance_nm / 14.0
        deadline_hours = float(transit_h + rng.uniform(36.0, 96.0))
        weather_factor = float(round(1.0 + (0.05 * (j % 5)), 2))
        sea_state = int(2 + (j % 4))
        cargos_spec[cid] = {
            "tons": tons,
            "cargo_tons": tons,
            "distance_nm": distance_nm,
            "deadline_hours": deadline_hours,
            "weather_factor": weather_factor,
            "sea_state": sea_state,
        }

    constraints: dict[str, Any] = {
        "vessels": vessels_spec,
        "cargos": cargos_spec,
    }

    scheduler = FleetScheduler(nominal_speed_knots=14.0)
    assignments = scheduler.assign_vessels(vessel_ids, cargo_ids, constraints)

    return vessel_ids, cargo_ids, constraints, assignments


class FleetOptimizationRunner:
    """Orchestrates equal-budget QPSO and PSO optimization runs over scheduled fleets."""

    def __init__(self, random_state: int = 42) -> None:
        """Initialize runner with default random state."""
        self.random_state = random_state
        self.logger = logger
        # Preload engines
        self.engines = get_cached_engines()

    def optimize_tier(
        self,
        tier: str,
        optimizer_name: str = "QPSO",
        weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
        seed: int = 42,
        normalize: bool = False,
    ) -> tuple[OptimizationResult, Sequence[FleetAssignment], dict[str, Any]]:
        """Run single optimization algorithm on a specified problem tier.

        Args:
            tier: Problem tier ('small', 'medium', or 'large').
            optimizer_name: 'QPSO' or 'PSO'.
            weights: Objective weight tuple (w_cost, w_co2e, w_delay).
            seed: Deterministic seed.
            normalize: Whether to use reference baseline normalization in objective.

        Returns:
            Tuple of (OptimizationResult, assignments, context).
        """
        if tier not in FLEET_TIER_CONFIGS:
            raise OptimizationError(
                f"Unknown fleet tier '{tier}'. Supported: {list(FLEET_TIER_CONFIGS.keys())}",
                details={"tier": tier},
            )

        cfg = FLEET_TIER_CONFIGS[tier]
        _, _, constraints, assignments = generate_fleet_problem(
            n_vessels=cfg["n_vessels"],
            n_cargos=cfg["n_cargos"],
            seed=seed,
        )

        assigned_voyages = [a for a in assignments if a.assigned]
        if not assigned_voyages:
            raise OptimizationError("No voyages could be assigned feasibly by scheduler.")

        context = {
            "voyage_specs": constraints["cargos"],
            "vessel_specs": constraints.get("vessels", {}),
            "delay_penalty_per_hour": 500.0,
            "compliance_year": 2025,
            "normalize": normalize,
        }

        # Decision vector bounds: [speed_knots in (10, 20), fuel_type in (0, 5)] per voyage
        bounds: list[tuple[float, float]] = []
        for _ in assigned_voyages:
            bounds.append((10.0, 20.0))
            bounds.append((0.0, 5.0))

        # Build closure over cached engines for high-throughput evaluation
        def target_objective(x: np.ndarray) -> float:
            return fleet_objective(
                x=x,
                assignments=assignments,
                weights=weights,
                context=context,
                engines=self.engines,
            )

        hypers = {
            "population_size": cfg["population_size"],
            "max_iterations": cfg["max_iterations"],
            "seed": seed,
        }

        opt_key = optimizer_name.strip().upper()
        if opt_key in ("QPSO", "QUANTUM_PSO"):
            optimizer = QPSOOptimizer(random_state=seed)
        elif opt_key in ("PSO", "CLASSICAL_PSO"):
            optimizer = PSOOptimizer(random_state=seed)
        else:
            raise OptimizationError(
                f"Unsupported optimizer '{optimizer_name}'. Supported: QPSO, PSO"
            )

        self.logger.info(
            "Running %s on '%s' tier (%d assigned voyages, budget: %d evaluations, normalize=%s)",
            opt_key,
            tier,
            len(assigned_voyages),
            cfg["population_size"] * cfg["max_iterations"],
            normalize,
        )

        result = optimizer.optimize(
            objective_function=target_objective,
            parameter_bounds=bounds,
            hyperparameters=hypers,
        )

        return result, assignments, context

    def run_scalability_sweep(
        self,
        tiers: Sequence[str] = ("small", "medium", "large"),
        algorithms: Sequence[str] = ("QPSO", "PSO"),
        seed: int = 42,
        normalize: bool = False,
    ) -> dict[str, Any]:
        """Execute full scalability sweep across tiers and algorithms under identical budgets.

        Returns:
            Structured benchmark dictionary ready for report export and convergence plotting.
        """
        sweep_results: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seed": seed,
            "normalize": normalize,
            "tiers": {},
        }

        for tier in tiers:
            sweep_results["tiers"][tier] = {}
            for algo in algorithms:
                res, assignments, _ = self.optimize_tier(
                    tier=tier,
                    optimizer_name=algo,
                    seed=seed,
                    normalize=normalize,
                )
                assigned_count = sum(1 for a in assignments if a.assigned)
                sweep_results["tiers"][tier][algo] = {
                    "optimizer_name": res.optimizer_name,
                    "problem_size": tier,
                    "assigned_voyages": assigned_count,
                    "best_score": res.best_score,
                    "wall_clock_seconds": res.runtime_seconds,
                    "n_evaluations": res.n_evaluations,
                    "iterations": res.iterations,
                    "converged": res.converged,
                    "history": list(res.history),
                }

        return sweep_results
