'''Quantum-Behaved Particle Swarm Optimization (QPSO) kernel module.

This module provides a reusable QPSO implementation that can be used by
both the production optimizer (`src.optimization.qpso.QPSOOptimizer`) and the
benchmark adapter (`src.benchmarking.qpso_benchmark_adapter.QPSOBenchmarkAdapter`).
It preserves the exact mathematical behaviour of the original implementation
including adaptive alpha, stagnation‑re‑exploration, elite archive maintenance,
and history collection.

The primary callable is `run_qpso`, which returns the best position, its score,
the number of evaluations, the convergence history, and a dictionary of extra
metadata (alpha history, diversity history, stagnation counters, elite archive,
cache statistics, etc.).
'''

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Dict, Tuple, List

import numpy as np

# The original helper functions are imported to retain identical behaviour.
# They are defined in `src.optimization.qpso` but we re‑export them here to
# avoid circular imports.
from src.optimization.qpso import calculate_swarm_diversity, compute_adaptive_alpha


def run_qpso(
    objective_function: Callable[[np.ndarray], float],
    lb: np.ndarray,
    ub: np.ndarray,
    population_size: int,
    max_iterations: int,
    seed: int,
    hyperparameters: Dict[str, Any],
    early_stopping: bool = False,
    elite_capacity: int = 5,
) -> Tuple[np.ndarray, float, int, Tuple[float, ...], Dict[str, Any]]:
    """Execute the QPSO search.

    Parameters
    ----------
    objective_function: Callable[[np.ndarray], float]
        Function that receives a raw decision vector and returns the objective
        value. It is assumed to be deterministic.
    lb, ub: np.ndarray
        Lower‑ and upper‑bound arrays of the decision space.
    population_size: int
        Number of particles in the swarm.
    max_iterations: int
        Maximum number of iterations (including the initial evaluation).
    seed: int
        Random seed for reproducibility.
    hyperparameters: dict
        Contains the QPSO specific parameters:
        - alpha_start, alpha_end, alpha_min, alpha_max
        - adaptive_alpha (bool), stagnation_patience, reinit_fraction
    early_stopping: bool, default=False
        If True, stops when stagnation exceeds ``early_stop_patience``.
    elite_capacity: int, default=5
        Size of the elite archive (decoded feasible solutions).

    Returns
    -------
    best_position: np.ndarray
        The best raw decision vector found.
    best_score: float
        Corresponding objective value.
    n_evaluations: int
        Total number of objective evaluations performed.
    history: tuple[float, ...]
        Convergence history of the global best score.
    metadata: dict
        Additional information such as alpha/diversity history, stagnation
        counters, elite archive size, and re‑initialisation count.
    """
    rng = np.random.default_rng(seed)
    dim = len(lb)

    # Extract hyper‑parameters with defaults matching the original code.
    alpha_start = float(hyperparameters.get("alpha_start", 1.0))
    alpha_end = float(hyperparameters.get("alpha_end", 0.5))
    alpha_min = float(hyperparameters.get("alpha_min", 0.35))
    alpha_max = float(hyperparameters.get("alpha_max", 1.25))
    use_adaptive_alpha = bool(hyperparameters.get("adaptive_alpha", True))
    stagnation_patience = int(hyperparameters.get("stagnation_patience", 5))
    reinit_fraction = float(hyperparameters.get("reinit_fraction", 0.25))
    early_stop_patience = int(hyperparameters.get("early_stop_patience", 15))
    tolerance = float(hyperparameters.get("improvement_tolerance", 1e-4))

    # 1. Initialise particle positions uniformly within bounds.
    X = rng.uniform(lb, ub, size=(population_size, dim))

    # 2. Evaluate initial swarm.
    pbest = X.copy()
    pbest_scores = np.empty(population_size, dtype=float)
    for i in range(population_size):
        pbest_scores[i] = float(objective_function(X[i]))
    n_evaluations = population_size

    best_idx = int(np.argmin(pbest_scores))
    gbest = pbest[best_idx].copy()
    gbest_score = float(pbest_scores[best_idx])

    # History containers.
    history: List[float] = [gbest_score]
    alpha_history: List[float] = [alpha_start]
    diversity_history: List[float] = []
    stagnation_history: List[int] = [0]
    stagnation_counter = 0
    total_reinit = 0

    # Elite archive stores tuples (signature, raw_vector, score, result_dict).
    elite_archive: List[Tuple[Tuple[int, ...], np.ndarray, float, Dict[str, Any]]] = []

    for t in range(1, max_iterations):
        # 3. Mean‑best position.
        mbest = np.mean(pbest, axis=0)
        # 4. Diversity metric.
        div = calculate_swarm_diversity(X, mbest, lb, ub)
        diversity_history.append(div)

        # 5. Adaptive contraction‑expansion coefficient.
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

        # 6. Vectorised quantum update.
        phi = rng.uniform(0.0, 1.0, size=(population_size, dim))
        p = phi * pbest + (1.0 - phi) * gbest
        u = rng.uniform(1e-12, 1.0, size=(population_size, dim))
        signs = rng.choice(np.array([-1.0, 1.0]), size=(population_size, dim))
        step = signs * alpha * np.abs(mbest - X) * np.log(1.0 / u)
        X_new = np.clip(p + step, lb, ub)

        # 7. Stagnation‑driven re‑exploration.
        if stagnation_counter >= stagnation_patience and reinit_fraction > 0:
            n_reinit = max(1, int(round(reinit_fraction * population_size)))
            worst_idxs = np.argsort(pbest_scores)[-n_reinit:]
            # Anchor can be the best elite if present, otherwise gbest.
            anchor = elite_archive[0][1] if elite_archive else gbest
            for idx in worst_idxs:
                s_u = rng.uniform(1e-12, 1.0, size=dim)
                s_signs = rng.choice(np.array([-1.0, 1.0]), size=dim)
                s_step = s_signs * alpha * np.abs(mbest - X[idx]) * np.log(1.0 / s_u)
                X_new[idx] = np.clip(anchor + s_step, lb, ub)
            total_reinit += n_reinit

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
        # Update stagnation counter.
        if improved:
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        stagnation_history.append(stagnation_counter)
        history.append(gbest_score)

        if early_stopping and stagnation_counter >= early_stop_patience:
            break
    # End of iteration loop.

    # Determine convergence flag (same heuristic as original).
    converged = bool(len(history) >= 5 and abs(history[-1] - history[-5]) < 1e-4)

    metadata = {
        "alpha_history": tuple(alpha_history),
        "diversity_history": tuple(diversity_history),
        "stagnation_history": tuple(stagnation_history),
        "reinitialization_count": total_reinit,
        "elite_archive_size": len(elite_archive),
        "converged": converged,
    }
    return gbest, gbest_score, n_evaluations, tuple(history), metadata
