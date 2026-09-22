"""Benchmark fleet optimization algorithms across scalability tiers.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Executes equal-budget QPSO vs PSO comparison across:
- Small (5 vessels / 10 cargos)
- Medium (20 vessels / 50 cargos)
- Large (50 vessels / 200 cargos)
across multiple random seeds to ensure statistical significance,
and optionally computes the bi-objective Pareto frontier on the medium tier.
Outputs results to outputs/reports/optimization_benchmark.json.
"""

from collections.abc import Sequence
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

# Ensure project root is in sys.path when invoked directly from scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from logging_config import configure_logging
from src.optimization.fleet_optimizer import FleetOptimizationRunner
from src.optimization.nsga2_pareto import ParetoFleetOptimizer

logger = logging.getLogger("maritime_system")


def run_optimization_benchmark(
    output_report: str = "outputs/reports/optimization_benchmark.json",
    seed: int | None = None,
    seeds: Sequence[int] | None = None,
    include_pareto: bool = True,
    normalize: bool = True,
) -> dict[str, Any]:
    """Execute complete multi-seed scalability sweep and export benchmark report."""
    configure_logging(level="INFO")

    if seeds is not None:
        eval_seeds = tuple(seeds)
    elif seed is not None:
        eval_seeds = (seed,)
    else:
        eval_seeds = (42, 101, 2024)

    logger.info(
        "Starting Fleet Optimization Scalability Benchmark (seeds=%s, normalize=%s)",
        eval_seeds,
        normalize,
    )

    all_seed_sweeps: dict[int, dict[str, Any]] = {}
    tiers = ("small", "medium", "large")
    algorithms = ("QPSO", "PSO")

    for s in eval_seeds:
        logger.info("--> Executing scalability sweep for seed=%d...", s)
        runner = FleetOptimizationRunner(random_state=s)
        sweep = runner.run_scalability_sweep(
            tiers=tiers,
            algorithms=algorithms,
            seed=s,
            normalize=normalize,
        )
        all_seed_sweeps[s] = sweep

    # Aggregate across seeds
    summary_table: list[dict[str, Any]] = []
    multi_seed_summary: dict[str, Any] = {}

    for tier in tiers:
        qpso_scores = [all_seed_sweeps[s]["tiers"][tier]["QPSO"]["best_score"] for s in eval_seeds]
        pso_scores = [all_seed_sweeps[s]["tiers"][tier]["PSO"]["best_score"] for s in eval_seeds]
        qpso_times = [all_seed_sweeps[s]["tiers"][tier]["QPSO"]["wall_clock_seconds"] for s in eval_seeds]
        pso_times = [all_seed_sweeps[s]["tiers"][tier]["PSO"]["wall_clock_seconds"] for s in eval_seeds]

        qpso_mean = float(np.mean(qpso_scores))
        qpso_std = float(np.std(qpso_scores))
        pso_mean = float(np.mean(pso_scores))
        pso_std = float(np.std(pso_scores))

        improvement_pct = 0.0
        if pso_mean > 0:
            improvement_pct = round(((pso_mean - qpso_mean) / pso_mean) * 100.0, 2)

        assigned_count = all_seed_sweeps[eval_seeds[0]]["tiers"][tier]["QPSO"]["assigned_voyages"]
        evals = all_seed_sweeps[eval_seeds[0]]["tiers"][tier]["QPSO"]["n_evaluations"]

        summary_table.append(
            {
                "tier": tier,
                "assigned_voyages": assigned_count,
                "qpso_best_score": round(qpso_mean, 4),
                "qpso_score_std": round(qpso_std, 4),
                "pso_best_score": round(pso_mean, 4),
                "pso_score_std": round(pso_std, 4),
                "qpso_improvement_pct": improvement_pct,
                "qpso_runtime_s": round(float(np.mean(qpso_times)), 2),
                "pso_runtime_s": round(float(np.mean(pso_times)), 2),
                "evaluations_per_algo": evals,
                "n_seeds": len(eval_seeds),
            }
        )

        multi_seed_summary[tier] = {
            "seeds": list(eval_seeds),
            "qpso_scores": [round(x, 4) for x in qpso_scores],
            "pso_scores": [round(x, 4) for x in pso_scores],
            "qpso_mean": round(qpso_mean, 4),
            "qpso_std": round(qpso_std, 4),
            "pso_mean": round(pso_mean, 4),
            "pso_std": round(pso_std, 4),
            "mean_improvement_pct": improvement_pct,
        }

    benchmark_data: dict[str, Any] = {
        "benchmark_metadata": {
            "title": "Quantum-Inspired Fleet Optimization Scalability Benchmark",
            "problem_id": "SIH26138",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seeds": list(eval_seeds),
            "normalize": normalize,
            "algorithms": list(algorithms),
            "tiers": list(tiers),
        },
        "summary": summary_table,
        "detailed_runs": all_seed_sweeps[eval_seeds[0]]["tiers"],
        "multi_seed_runs": {str(s): all_seed_sweeps[s]["tiers"] for s in eval_seeds},
        "multi_seed_summary": multi_seed_summary,
    }

    # Optional bi-objective Pareto analysis on medium tier
    if include_pareto:
        logger.info("Computing bi-objective Pareto front on medium tier (seed=%d)...", eval_seeds[0])
        pareto_solver = ParetoFleetOptimizer(random_state=eval_seeds[0])
        pareto_res = pareto_solver.solve_medium_pareto(population_size=15, generations=10, seed=eval_seeds[0])
        benchmark_data["pareto_analysis_medium"] = pareto_res

    out_path = Path(output_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(benchmark_data, indent=2), encoding="utf-8")
    logger.info("Saved optimization benchmark report to %s (%d bytes)", out_path.resolve(), out_path.stat().st_size)

    return benchmark_data


if __name__ == "__main__":
    run_optimization_benchmark()
