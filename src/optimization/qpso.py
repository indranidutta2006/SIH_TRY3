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


def calculate_swarm_diversity(X: np.ndarray, mbest: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> float:
    """Calculate normalized swarm spatial diversity D_t in [0, 1]."""
    span = float(np.linalg.norm(ub - lb))
    if span <= 1e-12:
        return 0.0
    mean_dist = float(np.mean(np.linalg.norm(X - mbest, axis=1)))
    return float(np.clip(mean_dist / span, 0.0, 1.0))


def compute_adaptive_alpha(
    t: int,
    max_iterations: int,
    diversity: float,
    stagnation_count: int,
    alpha_start: float = 1.0,
    alpha_end: float = 0.5,
    alpha_min: float = 0.35,
    alpha_max: float = 1.25,
    d_target: float = 0.15,
    s_patience: int = 5,
    c_d: float = 0.25,
    c_s: float = 0.20,
) -> float:
    """Compute diversity and stagnation responsive bounded contraction-expansion coefficient.

    Strictly satisfies alpha_min <= alpha_t <= alpha_max for all iterations.
    """
    prog = min(max(float(t) / max(1.0, float(max_iterations)), 0.0), 1.0)
    base_alpha = alpha_start - prog * (alpha_start - alpha_end)

    # Diversity adjustment: increases step size when swarm collapses prematurely
    delta_div = c_d * max(0.0, 1.0 - (diversity / max(d_target, 1e-6)))

    # Stagnation adjustment: increases exploration when gbest fails to improve
    delta_stag = c_s * min(1.0, max(0.0, float(stagnation_count) / max(s_patience, 1)))

    raw_alpha = base_alpha * (1.0 + delta_div + delta_stag)
    return float(np.clip(raw_alpha, alpha_min, alpha_max))


class QPSOOptimizer(SwarmOptimizationEngine):
    """Quantum-Behaved Particle Swarm Optimization (QPSO) solver with adaptive alpha."""

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
        # Initialise RNG (kept for possible future use) and delegate to reusable kernel.
        from src.optimization.qpso_kernel import run_qpso
        # Assemble hyperparameters matching original expectations.
        hyperparams = {
            "alpha_start": float(hyperparameters.get("alpha_start", 1.0)),
            "alpha_end": float(hyperparameters.get("alpha_end", 0.5)),
            "alpha_min": float(hyperparameters.get("alpha_min", 0.35)),
            "alpha_max": float(hyperparameters.get("alpha_max", 1.25)),
            "adaptive_alpha": bool(hyperparameters.get("adaptive_alpha", True)),
            "stagnation_patience": int(hyperparameters.get("stagnation_patience", 5)),
            "reinit_fraction": float(hyperparameters.get("reinit_fraction", 0.25)),
            "early_stop_patience": int(hyperparameters.get("early_stop_patience", 15)),
            "improvement_tolerance": float(hyperparameters.get("improvement_tolerance", 1e-4)),
        }
        # Run the shared QPSO kernel.
        gbest, gbest_score, n_evaluations, history, meta = run_qpso(
            objective_function=objective_function,
            lb=lb,
            ub=ub,
            population_size=population_size,
            max_iterations=max_iterations,
            seed=seed,
            hyperparameters=hyperparams,
            early_stopping=early_stopping,
            elite_capacity=5,
        )
        converged = meta.get("converged", False)
        return gbest, gbest_score, n_evaluations, tuple(history), converged
# Legacy implementation removed after delegating to qpso_kernel
# Legacy implementation removed after delegating to qpso_kernel
        alpha_end = float(hyperparameters.get("alpha_end", 0.5))
        alpha_min = float(hyperparameters.get("alpha_min", 0.35))
        alpha_max = float(hyperparameters.get("alpha_max", 1.25))
        use_adaptive_alpha = bool(hyperparameters.get("adaptive_alpha", True))
        stagnation_patience = int(hyperparameters.get("stagnation_patience", 5))
        reinit_fraction = float(hyperparameters.get("reinit_fraction", 0.25))
        early_stopping = bool(hyperparameters.get("early_stopping", False))
        early_stop_patience = int(hyperparameters.get("early_stop_patience", 15))
        tolerance = float(hyperparameters.get("improvement_tolerance", 1e-4))

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
        alpha_history: list[float] = [alpha_start]
        diversity_history: list[float] = []
        stagnation_counter = 0

        # 3. Iterative quantum delta-potential updates (max_iterations - 1 iterations)
        for t in range(1, max_iterations):
            mbest = np.mean(pbest, axis=0)
            div = calculate_swarm_diversity(X, mbest, lb, ub)
            diversity_history.append(div)

            if use_adaptive_alpha:
                alpha = compute_adaptive_alpha(
                    t=t,
                    max_iterations=max_iterations,
                    diversity=div,
                    stagnation_count=stagnation_counter,
                    alpha_start=alpha_start,
                    alpha_end=alpha_end,
                    alpha_min=alpha_min,
                    alpha_max=alpha_max,
                )
            else:
                alpha = float(np.clip(
                    alpha_start - (t / max_iterations) * (alpha_start - alpha_end),
                    alpha_min,
                    alpha_max,
                ))

            alpha_history.append(alpha)

            # Vectorized quantum update for all particles
            phi = rng.uniform(0.0, 1.0, size=(population_size, dim))
            p = phi * pbest + (1.0 - phi) * gbest
            u = rng.uniform(1e-12, 1.0, size=(population_size, dim))
            signs = rng.choice(np.array([-1.0, 1.0]), size=(population_size, dim))
            step = signs * alpha * np.abs(mbest - X) * np.log(1.0 / u)
            X_new = np.clip(p + step, lb, ub)

            # Stagnation re-exploration: replace worst particles' update with quantum exploration around gbest
            if stagnation_counter >= stagnation_patience and reinit_fraction > 0:
                n_reinit = max(1, int(round(reinit_fraction * population_size)))
                worst_indices = np.argsort(pbest_scores)[-n_reinit:]
                for idx in worst_indices:
                    s_u = rng.uniform(1e-12, 1.0, size=dim)
                    s_signs = rng.choice(np.array([-1.0, 1.0]), size=dim)
                    s_step = s_signs * alpha * np.abs(mbest - X[idx]) * np.log(1.0 / s_u)
                    X_new[idx] = np.clip(gbest + s_step, lb, ub)

            X = X_new
            improved = False

            for i in range(population_size):
                score = float(objective_function(X[i]))
                n_evaluations += 1

                if score < pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest[i] = X[i].copy()
                    if score < gbest_score - tolerance:
                        gbest_score = score
                        gbest = X[i].copy()
                        improved = True

            if improved:
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            history.append(gbest_score)

            if early_stopping and stagnation_counter >= early_stop_patience:
                break

        converged = bool(len(history) >= 5 and abs(history[-1] - history[-5]) < 1e-4)
        return gbest, gbest_score, n_evaluations, tuple(history), converged
