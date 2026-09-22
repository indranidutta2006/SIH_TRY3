"""Deterministic unit tests for QPSO and PSO swarm optimization engines.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Validates exact evaluation budget accounting, bit-identical seed reproducibility,
contract compliance with OptimizationEngine, and parameter boundary enforcement.
"""

import numpy as np
import pytest

from contracts.interfaces import OptimizationEngine
from contracts.schemas import OptimizationResult
from src.optimization import PSOOptimizer, QPSOOptimizer, SwarmOptimizationEngine


def sphere_objective(x: np.ndarray) -> float:
    """Standard continuous convex benchmark function with minimum at x=0."""
    return float(np.sum(x**2))


def test_swarm_optimizers_conform_to_contract() -> None:
    """Verify both QPSO and PSO strictly fulfill OptimizationEngine interface."""
    qpso = QPSOOptimizer(random_state=42)
    pso = PSOOptimizer(random_state=42)

    assert isinstance(qpso, OptimizationEngine)
    assert isinstance(qpso, SwarmOptimizationEngine)
    assert isinstance(pso, OptimizationEngine)
    assert isinstance(pso, SwarmOptimizationEngine)


@pytest.mark.parametrize("optimizer_cls", [QPSOOptimizer, PSOOptimizer])
def test_evaluation_budget_exact_equality(optimizer_cls: type[SwarmOptimizationEngine]) -> None:
    """Assert n_evaluations == population_size * max_iterations exactly."""
    population_size = 15
    max_iterations = 25
    expected_evaluations = population_size * max_iterations

    bounds = {"x0": (-5.0, 5.0), "x1": (-5.0, 5.0), "x2": (-5.0, 5.0)}
    hyperparameters = {
        "population_size": population_size,
        "max_iterations": max_iterations,
        "seed": 123,
    }

    optimizer = optimizer_cls(random_state=123)
    result = optimizer.optimize(
        objective_function=sphere_objective,
        parameter_bounds=bounds,
        hyperparameters=hyperparameters,
    )

    assert isinstance(result, OptimizationResult)
    assert result.n_evaluations == expected_evaluations
    assert result.iterations == max_iterations
    assert len(result.history) == max_iterations


@pytest.mark.parametrize("optimizer_cls", [QPSOOptimizer, PSOOptimizer])
def test_bit_identical_history_reproducibility(optimizer_cls: type[SwarmOptimizationEngine]) -> None:
    """Assert bit-identical convergence history across two separate runs with the same seed."""
    bounds = {"x0": (-10.0, 10.0), "x1": (-10.0, 10.0)}
    hyperparameters = {
        "population_size": 20,
        "max_iterations": 30,
        "seed": 999,
    }

    opt1 = optimizer_cls(random_state=999)
    res1 = opt1.optimize(sphere_objective, bounds, hyperparameters)

    opt2 = optimizer_cls(random_state=999)
    res2 = opt2.optimize(sphere_objective, bounds, hyperparameters)

    assert res1.best_score == res2.best_score
    assert res1.history == res2.history
    assert res1.n_evaluations == res2.n_evaluations


@pytest.mark.parametrize("optimizer_cls", [QPSOOptimizer, PSOOptimizer])
def test_convergence_towards_optimum(optimizer_cls: type[SwarmOptimizationEngine]) -> None:
    """Assert both optimizers decrease the sphere objective value significantly from start."""
    bounds = {"x0": (-5.0, 5.0), "x1": (-5.0, 5.0)}
    hyperparameters = {
        "population_size": 25,
        "max_iterations": 40,
        "seed": 42,
    }

    optimizer = optimizer_cls(random_state=42)
    res = optimizer.optimize(sphere_objective, bounds, hyperparameters)

    # Initial score should be higher than final best_score
    assert res.history[0] > res.best_score
    assert res.best_score < 1.0  # Near the true optimum 0.0


def test_benchmark_method_executes_without_type_error() -> None:
    """Assert benchmark() executes multi-algorithm run without TypeError or failure."""
    qpso = QPSOOptimizer(random_state=42)
    test_case = {
        "objective_function": sphere_objective,
        "parameter_bounds": [(-5.0, 5.0), (-5.0, 5.0)],
        "hyperparameters": {"population_size": 10, "max_iterations": 15, "seed": 42},
    }

    bench_results = qpso.benchmark(algorithms=["QPSO", "PSO"], test_case=test_case)

    assert "QPSO" in bench_results
    assert "PSO" in bench_results
    assert bench_results["QPSO"].n_evaluations == 150
    assert bench_results["PSO"].n_evaluations == 150
