"""Base class for heuristic and quantum-inspired swarm optimization algorithms.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Fulfills contracts.interfaces.OptimizationEngine, providing common validation,
timing, equal-budget discipline, and multi-algorithm benchmark dispatching.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
import logging
import time
from typing import Any

import numpy as np

from contracts.exceptions import OptimizationError
from contracts.interfaces import OptimizationEngine
from contracts.schemas import OptimizationResult

logger = logging.getLogger("maritime_system")


class SwarmOptimizationEngine(OptimizationEngine, ABC):
    """Base swarm optimization class implementing OptimizationEngine interface."""

    def __init__(self, optimizer_name: str, random_state: int = 42) -> None:
        """Initialize swarm optimizer with canonical name and random seed.

        Args:
            optimizer_name: Identifying token for the optimization algorithm.
            random_state: Deterministic random seed.
        """
        self.optimizer_name = optimizer_name
        self.random_state = random_state
        self.logger = logger

    def _parse_bounds(
        self, parameter_bounds: dict[str, tuple[float, float]] | Sequence[tuple[float, float]]
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Normalize parameter bounds into numpy arrays for vector search.

        Args:
            parameter_bounds: Dictionary or sequence of (lower_bound, upper_bound).

        Returns:
            Tuple of (lower_bounds_array, upper_bounds_array, param_names).

        Raises:
            OptimizationError: If bounds are empty or invalid (lb > ub).
        """
        if not parameter_bounds:
            raise OptimizationError(
                "Parameter bounds must not be empty.",
                details={"optimizer": self.optimizer_name},
            )

        if isinstance(parameter_bounds, dict):
            param_names = list(parameter_bounds.keys())
            bounds_list = [parameter_bounds[k] for k in param_names]
        else:
            bounds_list = list(parameter_bounds)
            param_names = [f"param_{i}" for i in range(len(bounds_list))]

        lbs = []
        ubs = []
        for i, bound in enumerate(bounds_list):
            if not isinstance(bound, (tuple, list)) or len(bound) != 2:
                raise OptimizationError(
                    f"Bound at index {i} must be a 2-element tuple (lb, ub), got: {bound}",
                    details={"index": i, "bound": bound},
                )
            lb, ub = float(bound[0]), float(bound[1])
            if lb > ub:
                raise OptimizationError(
                    f"Lower bound exceeds upper bound at index {i}: {lb} > {ub}",
                    details={"index": i, "lb": lb, "ub": ub},
                )
            lbs.append(lb)
            ubs.append(ub)

        return np.array(lbs, dtype=float), np.array(ubs, dtype=float), param_names

    @abstractmethod
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
        """Execute core swarm algorithm.

        Returns:
            Tuple of (best_position, best_score, n_evaluations, history, converged).
        """
        pass

    def optimize(
        self,
        objective_function: Any,
        parameter_bounds: dict[str, tuple[float, float]] | Sequence[tuple[float, float]],
        hyperparameters: dict[str, Any] | None = None,
    ) -> OptimizationResult:
        """Execute heuristic or quantum-inspired search to minimize objective.

        Args:
            objective_function: Callable mapping 1D numpy array to scalar cost.
            parameter_bounds: Mapping or sequence defining (min, max) bounds.
            hyperparameters: Optional dictionary with population_size, max_iterations, seed, etc.

        Returns:
            OptimizationResult containing optimal score, runtime, iterations, history, and evaluations.

        Raises:
            OptimizationError: If bounds or execution fails.
        """
        if not callable(objective_function):
            raise OptimizationError(
                "Objective function must be a callable.",
                details={"provided_type": type(objective_function).__name__},
            )

        lb, ub, _ = self._parse_bounds(parameter_bounds)
        params = hyperparameters or {}

        population_size = int(params.get("population_size", 20))
        max_iterations = int(params.get("max_iterations", 50))
        seed = int(params.get("seed", self.random_state))
        problem_size = str(params.get("problem_size", ""))

        if population_size < 2:
            raise OptimizationError(
                f"population_size must be >= 2, got: {population_size}",
                details={"population_size": population_size},
            )
        if max_iterations < 1:
            raise OptimizationError(
                f"max_iterations must be >= 1, got: {max_iterations}",
                details={"max_iterations": max_iterations},
            )

        start_time = time.perf_counter()
        try:
            best_pos, best_score, n_evals, history, converged = self._run_optimization(
                objective_function=objective_function,
                lb=lb,
                ub=ub,
                population_size=population_size,
                max_iterations=max_iterations,
                seed=seed,
                hyperparameters=params,
            )
        except Exception as err:
            raise OptimizationError(
                f"Optimization execution failed under {self.optimizer_name}: {err}",
                details={"optimizer": self.optimizer_name, "error": str(err)},
            ) from err

        wall_clock = time.perf_counter() - start_time

        return OptimizationResult(
            optimizer_name=self.optimizer_name,
            best_score=float(best_score),
            runtime_seconds=round(wall_clock, 4),
            iterations=max_iterations,
            converged=converged,
            history=history,
            n_evaluations=n_evals,
            problem_size=problem_size,
            best_vector=tuple(float(v) for v in best_pos),
        )

    def benchmark(
        self,
        algorithms: Sequence[str],
        test_case: dict[str, Any],
    ) -> dict[str, OptimizationResult]:
        """Execute competitive multi-algorithm benchmark over a standardized test case.

        Args:
            algorithms: List of optimizer identifiers to execute.
            test_case: Dictionary containing 'objective_function', 'parameter_bounds',
                       and optional 'hyperparameters'.

        Returns:
            Dictionary mapping algorithm name to its OptimizationResult.

        Raises:
            OptimizationError: If test_case is missing required fields.
        """
        if "objective_function" not in test_case or "parameter_bounds" not in test_case:
            raise OptimizationError(
                "Benchmark test_case must contain 'objective_function' and 'parameter_bounds'.",
                details={"keys": list(test_case.keys())},
            )

        obj_fn = test_case["objective_function"]
        bounds = test_case["parameter_bounds"]
        hypers = test_case.get("hyperparameters", {})

        from src.optimization.pso_baseline import PSOOptimizer
        from src.optimization.qpso import QPSOOptimizer

        results: dict[str, OptimizationResult] = {}
        for algo in algorithms:
            algo_key = algo.strip().upper()
            if algo_key in ("QPSO", "QUANTUM_PSO"):
                opt = QPSOOptimizer(random_state=self.random_state)
            elif algo_key in ("PSO", "CLASSICAL_PSO"):
                opt = PSOOptimizer(random_state=self.random_state)
            else:
                raise OptimizationError(
                    f"Unsupported benchmark algorithm: '{algo}'. Supported: QPSO, PSO",
                    details={"requested": algo},
                )
            results[algo] = opt.optimize(obj_fn, bounds, hypers)

        return results
