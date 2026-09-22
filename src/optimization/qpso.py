"""Quantum-Behaved Particle Swarm Optimization (QPSO) engine.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Implements quantum delta-potential-well search with mean best position (mbest)
and contraction-expansion alpha scheduling. Conforms strictly to contracts.interfaces.OptimizationEngine.
"""

from collections.abc import Callable
import logging
from typing import Any

import numpy as np

from contracts.constants import OptimizerType
from src.optimization.base import SwarmOptimizationEngine

logger = logging.getLogger("maritime_system")


class QPSOOptimizer(SwarmOptimizationEngine):
    """Quantum-Behaved Particle Swarm Optimization (QPSO) solver."""

    def __init__(self, random_state: int = 42) -> None:
        """Initialize QPSO optimizer with canonical identifier and seed."""
        super().__init__(optimizer_name=OptimizerType.QPSO.value, random_state=random_state)

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
        """Execute QPSO search under strict budget discipline."""
        rng = np.random.default_rng(seed)
        dim = len(lb)
        alpha_start = float(hyperparameters.get("alpha_start", 1.0))
        alpha_end = float(hyperparameters.get("alpha_end", 0.5))

        # 1. Initialize swarm positions uniformly within parameter bounds
        X = rng.uniform(lb, ub, size=(population_size, dim))

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

        # 3. Iterative quantum delta-potential updates (max_iterations - 1 iterations)
        # Total evaluations = population_size + (max_iterations - 1) * population_size
        #                   = population_size * max_iterations exactly.
        for t in range(1, max_iterations):
            alpha = alpha_start - (t / max_iterations) * (alpha_start - alpha_end)
            mbest = np.mean(pbest, axis=0)

            for i in range(population_size):
                phi = rng.uniform(0.0, 1.0, size=dim)
                p = phi * pbest[i] + (1.0 - phi) * gbest

                u = rng.uniform(0.0, 1.0, size=dim)
                u = np.clip(u, 1e-12, 1.0)
                signs = rng.choice([-1.0, 1.0], size=dim)

                step = signs * alpha * np.abs(mbest - X[i]) * np.log(1.0 / u)
                X[i] = np.clip(p + step, lb, ub)

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
