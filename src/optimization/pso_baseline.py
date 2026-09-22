"""Classical Particle Swarm Optimization (PSO) baseline solver.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Implements canonical velocity-clamped inertia particle swarm optimization baseline.
Conforms strictly to contracts.interfaces.OptimizationEngine.
"""

from collections.abc import Callable
import logging
from typing import Any

import numpy as np

from contracts.constants import OptimizerType
from src.optimization.base import SwarmOptimizationEngine

logger = logging.getLogger("maritime_system")


class PSOOptimizer(SwarmOptimizationEngine):
    """Canonical Particle Swarm Optimization (PSO) baseline solver."""

    def __init__(self, random_state: int = 42) -> None:
        """Initialize PSO baseline with canonical identifier and seed."""
        super().__init__(optimizer_name=OptimizerType.PSO.value, random_state=random_state)

    def _run_optimization(
        self,
        objective_function: Callable[[np.ndarray], float],
        lb: np.ndarray,
        ub: np.ndarray,
        population_size: int,
        max_iterations: int,
        seed: int,
        hyperparameters: dict[str, Any],
    ) -> tuple[np.ndarray, float, int, tuple[float, ...], bool]:
        """Execute classical PSO search under strict budget discipline."""
        rng = np.random.default_rng(seed)
        dim = len(lb)

        w_start = float(hyperparameters.get("w_start", 0.9))
        w_end = float(hyperparameters.get("w_end", 0.4))
        c1 = float(hyperparameters.get("c1", 1.5))
        c2 = float(hyperparameters.get("c2", 1.5))

        # Velocity bounds: maximum 20% of the dynamic range per dimension
        v_max = 0.2 * (ub - lb)
        v_min = -v_max

        # 1. Initialize swarm positions and velocities
        X = rng.uniform(lb, ub, size=(population_size, dim))
        V = rng.uniform(v_min, v_max, size=(population_size, dim))

        # 2. Initial evaluation of population (population_size evaluations)
        pbest = X.copy()
        pbest_scores = np.empty(population_size, dtype=float)
        for i in range(population_size):
            pbest_scores[i] = float(objective_function(X[i]))

        n_evaluations = population_size
        best_idx = int(np.argmin(pbest_scores))
        gbest = pbest[best_idx].copy()
        gbest_score = float(pbest_scores[best_idx])

        history: list[float] = [gbest_score]

        # 3. Velocity-clamped iterations (max_iterations - 1 iterations)
        # Total evaluations = population_size + (max_iterations - 1) * population_size
        #                   = population_size * max_iterations exactly.
        for t in range(1, max_iterations):
            w = w_start - (t / max_iterations) * (w_start - w_end)

            for i in range(population_size):
                r1 = rng.uniform(0.0, 1.0, size=dim)
                r2 = rng.uniform(0.0, 1.0, size=dim)

                V[i] = (
                    w * V[i]
                    + c1 * r1 * (pbest[i] - X[i])
                    + c2 * r2 * (gbest - X[i])
                )
                V[i] = np.clip(V[i], v_min, v_max)
                X[i] = np.clip(X[i] + V[i], lb, ub)

                score = float(objective_function(X[i]))
                n_evaluations += 1

                if score < pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest[i] = X[i].copy()
                    if score < gbest_score:
                        gbest_score = score
                        gbest = X[i].copy()

            history.append(gbest_score)

        converged = bool(len(history) >= 5 and abs(history[-1] - history[-5]) < 1e-4)
        return gbest, gbest_score, n_evaluations, tuple(history), converged
