import time
from typing import Any
import numpy as np

from contracts.schemas import BenchmarkResult, OptimizationScenario
from src.benchmarking.solvers.base_solver import (
    BaseBenchmarkSolver,
    BenchmarkEvaluationCache,
    VESSEL_SPECS,
)
from src.optimization.qpso_kernel import run_qpso

class QPSOBenchmarkAdapter(BaseBenchmarkSolver):
    """QPSO solver adapter conforming strictly to the benchmark evaluation interface.

    The heavy‑lifting QPSO dynamics are delegated to the shared kernel
    :func:`src.optimization.qpso_kernel.run_qpso`.  This adapter retains its
    domain‑specific decoding, repair, and evaluation logic, as well as elite‑
    archive handling and cache statistics collection.
    """

    def __init__(
        self,
        population_size: int = 20,
        alpha_start: float = 1.0,
        alpha_end: float = 0.5,
        alpha_min: float = 0.35,
        alpha_max: float = 1.25,
        use_adaptive_alpha: bool = True,
        stagnation_patience: int = 5,
        reinit_fraction: float = 0.25,
        elite_capacity: int = 5,
    ) -> None:
        super().__init__(solver_name="Quantum-Inspired PSO (QPSO)")
        self.population_size = population_size
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.use_adaptive_alpha = use_adaptive_alpha
        self.stagnation_patience = stagnation_patience
        self.reinit_fraction = reinit_fraction
        self.elite_capacity = elite_capacity

    def solve(
        self,
        scenario: OptimizationScenario,
        max_iterations: int = 50,
        seed: int = 42,
        early_stopping: bool = False,
    ) -> BenchmarkResult:
        """Execute QPSO search via the shared kernel.

        The kernel works on *raw* decision vectors.  This wrapper converts the
        vector to the decoded problem representation, evaluates it (with caching)
        and maintains an elite‑archive of feasible decoded solutions.
        """
        start_time = time.perf_counter()
        rng = np.random.default_rng(seed)

        # Isolated cache for this solver run
        cache = BenchmarkEvaluationCache()

        # Decision‑vector bounds (9 dimensions)
        lb = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.5], dtype=float)
        ub = np.array([
            float(VESSEL_SPECS["feeder"]["max_available"]),
            float(VESSEL_SPECS["medium"]["max_available"]),
            float(VESSEL_SPECS["large"]["max_available"]),
            scenario.max_transition_rate,
            1.0,
            1.0,
            1.0,
            1.0,
            17.5,
        ], dtype=float)

        repair_distances: list[float] = []
        elite_archive: list[tuple[tuple[int, ...], np.ndarray, float, dict[str, Any]]] = []
        best_score = float("inf")
        best_eval_dict: dict[str, Any] | None = None

        # Objective wrapper for the kernel
        def objective(vec: np.ndarray) -> float:
            nonlocal best_score, best_eval_dict
            xf, xm, xl, fuel_mix, speed, rep_dist = self.decode_and_repair(vec, scenario)
            repair_distances.append(rep_dist)
            res = self.evaluate_candidate(xf, xm, xl, fuel_mix, speed, scenario, cache=cache)
            score = float(res["objective_score"])
            if score < best_score:
                best_score = score
                best_eval_dict = res
            # Elite‑archive maintenance (decoded feasible solutions only)
            if res.get("feasible_solution", False):
                sig = (
                    xf,
                    xm,
                    xl,
                    fuel_mix.get("diesel", 0),
                    fuel_mix.get("lng", 0),
                    fuel_mix.get("methanol", 0),
                    fuel_mix.get("hydrogen", 0),
                    fuel_mix.get("ammonia", 0),
                )
                existing_idx = next((i for i, entry in enumerate(elite_archive) if entry[0] == sig), None)
                if existing_idx is not None:
                    if score < elite_archive[existing_idx][2]:
                        elite_archive[existing_idx] = (sig, vec.copy(), score, res)
                else:
                    if len(elite_archive) < self.elite_capacity:
                        elite_archive.append((sig, vec.copy(), score, res))
                    else:
                        worst_idx = max(range(len(elite_archive)), key=lambda i: elite_archive[i][2])
                        if score < elite_archive[worst_idx][2]:
                            elite_archive[worst_idx] = (sig, vec.copy(), score, res)
            return score

        # Hyper‑parameters for the kernel
        hyperparams = {
            "alpha_start": self.alpha_start,
            "alpha_end": self.alpha_end,
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
            "adaptive_alpha": self.use_adaptive_alpha,
            "stagnation_patience": self.stagnation_patience,
            "reinit_fraction": self.reinit_fraction,
            "early_stop_patience": 15,
            "improvement_tolerance": 1e-4,
        }

        # Run the shared QPSO kernel
        best_vec, _, eval_count, history, meta = run_qpso(
            objective_function=objective,
            lb=lb,
            ub=ub,
            population_size=self.population_size,
            max_iterations=max_iterations,
            seed=seed,
            hyperparameters=hyperparams,
            early_stopping=early_stopping,
            elite_capacity=self.elite_capacity,
        )

        elapsed = time.perf_counter() - start_time
        assert best_eval_dict is not None

        # Compare with best elite (if any) and prefer the better decoded solution
        if elite_archive:
            best_elite = min(elite_archive, key=lambda e: e[2])
            if best_elite[2] < best_eval_dict["objective_score"]:
                best_eval_dict = best_elite[3]

        conv_info = {
            "initial_objective": round(float(history[0]), 4),
            "final_objective": round(float(history[-1]), 4),
            "improvement_pct": round(
                float(history[0] - history[-1]) / max(abs(history[0]), 1e-4) * 100.0,
                2,
            ),
            "history": list(history),
            "n_evaluations": eval_count,
        }

        return BenchmarkResult(
            solver_name=self.solver_name,
            runtime_seconds=round(elapsed, 4),
            iterations=max_iterations,
            objective_score=best_eval_dict["objective_score"],
            fuel_consumption=best_eval_dict["fuel_consumption"],
            operational_cost=best_eval_dict["operational_cost"],
            emissions=best_eval_dict["emissions"],
            reliability_score=best_eval_dict["reliability_score"],
            demand_satisfaction_rate=best_eval_dict["demand_satisfaction_rate"],
            convergence_score=round(
                float(history[0] - history[-1]) / max(abs(history[0]), 1e-4),
                4,
            ),
            feasible_solution=best_eval_dict["feasible_solution"],
            n_evaluations=eval_count,
            convergence_information=conv_info,
            metadata={
                "fleet_mix": best_eval_dict["fleet_mix"],
                "speed_knots": best_eval_dict["speed_knots"],
                "history": list(history),
                "alpha_history": list(meta.get("alpha_history", [])),
                "diversity_history": list(meta.get("diversity_history", [])),
                "stagnation_history": list(meta.get("stagnation_history", [])),
                "reinitialization_count": meta.get("reinitialization_count", 0),
                "elite_archive_size": len(elite_archive),
                "cache_hits": cache.hits,
                "cache_misses": cache.misses,
                "cache_hit_rate": cache.hit_rate,
                "unique_evaluations": cache.misses,
                "mean_repair_distance": round(
                    float(np.mean(repair_distances)), 4
                )
                if repair_distances
                else 0.0,
                "is_quantum": True,
            },
        )
