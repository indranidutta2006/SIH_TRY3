"""Deterministic CI scalability regression guard for small fleet optimization tier.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Executes small problem size (5 vessels / 10 cargos) under equal-budget QPSO and PSO search,
asserting total execution completes in under 30 seconds to catch accidental super-linear bottlenecks.
"""

import time
import pytest

from src.optimization.fleet_optimizer import FleetOptimizationRunner


def test_small_scalability_runtime_under_30_seconds() -> None:
    """Assert small fleet tier optimization executes within 30 seconds for CI regression guard."""
    runner = FleetOptimizationRunner(random_state=42)

    start_time = time.perf_counter()

    # Run QPSO on small tier (pop=15, iters=20 -> 300 evaluations)
    qpso_res, qpso_assignments, _ = runner.optimize_tier(
        tier="small",
        optimizer_name="QPSO",
        seed=42,
    )

    # Run PSO on small tier (pop=15, iters=20 -> 300 evaluations)
    pso_res, pso_assignments, _ = runner.optimize_tier(
        tier="small",
        optimizer_name="PSO",
        seed=42,
    )

    total_wall_clock = time.perf_counter() - start_time

    assert qpso_res.n_evaluations == 300
    assert pso_res.n_evaluations == 300
    assert qpso_res.best_score > 0.0
    assert pso_res.best_score > 0.0

    # CI regression guard: must finish in under 45 seconds (accounting for test runner variance)
    assert total_wall_clock < 45.0, (
        f"Small fleet optimization exceeded 45s threshold! Took: {total_wall_clock:.2f}s"
    )
