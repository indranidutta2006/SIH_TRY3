"""Scalability sweep and peak memory profiling suite.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 3 Deliverable: Benchmarks solver performance across fleet scales [10, 50, 100, 250, 500, 1000]
measuring runtime complexity, peak memory consumption (tracemalloc), and solution quality.
"""

import time
import tracemalloc
from typing import Any

from contracts.schemas import OptimizationScenario
from src.benchmarking.qpso_benchmark_adapter import QPSOBenchmarkAdapter
from src.benchmarking.solvers.classical_pso_solver import ClassicalPSOSolver
from src.benchmarking.solvers.greedy_solver import GreedyFleetSolver


class ScalabilitySuite:
    """Evaluates algorithmic scaling across problem dimensions and fleet capacities."""

    def __init__(self) -> None:
        """Initialize scalability suite with representative comparison solvers."""
        self.solvers = {
            "QPSO": QPSOBenchmarkAdapter(),
            "Classical PSO": ClassicalPSOSolver(),
            "Greedy": GreedyFleetSolver(),
        }

    def run_scalability_sweep(
        self,
        vessel_counts: tuple[int, ...] = (10, 50, 100, 250, 500, 1000),
        base_demand_per_vessel: float = 20_000.0,
        max_iterations: int = 20,
    ) -> list[dict[str, Any]]:
        """Execute scalability sweep over fine-grained fleet sizes measuring time, memory, and quality.

        Args:
            vessel_counts: Fleet sizes to benchmark (default: 10, 50, 100, 250, 500, 1000).
            base_demand_per_vessel: Proportional demand scaling per vessel.
            max_iterations: Iteration budget per solver run.

        Returns:
            List of dictionaries capturing scale records.
        """
        records: list[dict[str, Any]] = []

        for count in vessel_counts:
            # Scale scenario proportionately with fleet size
            scenario = OptimizationScenario(
                cargo_demand=count * base_demand_per_vessel,
                route_distance=3500.0,
                deadline_hours=280.0,
                budget=float(count) * 12_000_000.0,
                scenario_id=f"SCEN-SCALE-{count:04d}",
            )

            for name, solver in self.solvers.items():
                tracemalloc.start()
                t0 = time.perf_counter()

                res = solver.solve(scenario, max_iterations=max_iterations, seed=42)

                runtime = time.perf_counter() - t0
                current_mem, peak_mem = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                peak_mb = peak_mem / (1024.0 * 1024.0)

                records.append({
                    "fleet_size": count,
                    "solver_name": name,
                    "runtime_seconds": round(runtime, 4),
                    "peak_memory_mb": round(peak_mb, 3),
                    "objective_score": res.objective_score,
                    "objective": res.objective_score,
                    "fuel_consumption": res.fuel_consumption,
                    "operational_cost": res.operational_cost,
                    "feasible": res.feasible_solution,
                    "feasibility": res.feasible_solution,
                    "n_evaluations": res.n_evaluations,
                })

        return records

    def analyze_complexity_order(self, records: list[dict[str, Any]]) -> dict[str, str]:
        """Classify empirical runtime complexity for each solver."""
        summary: dict[str, str] = {}
        for name in self.solvers:
            s_recs = [r for r in records if r["solver_name"] == name]
            if len(s_recs) < 2:
                continue
            ratio = s_recs[-1]["runtime_seconds"] / max(s_recs[0]["runtime_seconds"], 1e-4)
            size_ratio = s_recs[-1]["fleet_size"] / max(s_recs[0]["fleet_size"], 1)

            if ratio <= size_ratio * 1.5:
                summary[name] = "O(N) Near-Linear"
            elif ratio <= (size_ratio ** 2) * 1.5:
                summary[name] = "O(N^2) Polynomial"
            else:
                summary[name] = "O(e^N) Exponential"

        return summary
