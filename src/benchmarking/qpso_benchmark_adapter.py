import time
from typing import Any

import numpy as np

from contracts.schemas import BenchmarkResult, OptimizationScenario
from src.benchmarking.solvers.base_solver import (
    BaseBenchmarkSolver,
    BenchmarkEvaluationCache,
    VESSEL_SPECS,
)
from src.optimization.qpso import calculate_swarm_diversity, compute_adaptive_alpha


class QPSOBenchmarkAdapter(BaseBenchmarkSolver):
    """QPSO solver adapter conforming strictly to the benchmark evaluation interface."""

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
    ) -> BenchmarkResult:
        """Execute QPSO search using quantum delta-potential-well dynamics."""
        start_time = time.perf_counter()
        rng = np.random.default_rng(seed)

        # Isolated evaluation cache for this single solver run (no cross-solver sharing)
        cache = BenchmarkEvaluationCache()

        # Dimension 9: [x_f, x_m, x_l, alt_ratio, s_lng, s_methanol, s_hydrogen, s_ammonia, speed]
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
        dim = len(lb)

        eval_count = 0
        repair_distances: list[float] = []

        # Decoded-state elite archive: list of (signature, vec, score, res_dict)
        elite_archive: list[tuple[tuple[int, ...], np.ndarray, float, dict[str, Any]]] = []

        def decode_and_eval(vec: np.ndarray) -> tuple[float, dict[str, Any]]:
            nonlocal eval_count
            eval_count += 1

            xf, xm, xl, fuel_mix, speed, rep_dist = self.decode_and_repair(vec, scenario)
            repair_distances.append(rep_dist)

            res = self.evaluate_candidate(xf, xm, xl, fuel_mix, speed, scenario, cache=cache)
            score = res["objective_score"]

            # Update elite archive if solution is feasible
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

            return score, res

        # 1. Initialize swarm positions uniformly in parameter bounds
        X = rng.uniform(lb, ub, size=(self.population_size, dim))
        pbest = X.copy()
        pbest_scores = np.full(self.population_size, float("inf"))
        eval_dicts: list[dict[str, Any]] = [{}] * self.population_size

        for i in range(self.population_size):
            pbest_scores[i], eval_dicts[i] = decode_and_eval(X[i])

        best_idx = int(np.argmin(pbest_scores))
        gbest = pbest[best_idx].copy()
        gbest_score = float(pbest_scores[best_idx])
        best_eval_dict = eval_dicts[best_idx]

        history: list[float] = [gbest_score]
        alpha_history: list[float] = [self.alpha_start]
        diversity_history: list[float] = []
        stagnation_history: list[int] = [0]
        stagnation_counter = 0
        total_reinit_count = 0

        # 2. Quantum delta-potential updates
        for t in range(1, max_iterations):
            mbest = np.mean(pbest, axis=0)
            div = calculate_swarm_diversity(X, mbest, lb, ub)
            diversity_history.append(div)

            if self.use_adaptive_alpha:
                alpha = compute_adaptive_alpha(
                    t=t,
                    max_iterations=max_iterations,
                    diversity=div,
                    stagnation_count=stagnation_counter,
                    alpha_start=self.alpha_start,
                    alpha_end=self.alpha_end,
                    alpha_min=self.alpha_min,
                    alpha_max=self.alpha_max,
                )
            else:
                alpha = float(np.clip(
                    self.alpha_start - (t / max_iterations) * (self.alpha_start - self.alpha_end),
                    self.alpha_min,
                    self.alpha_max,
                ))

            alpha_history.append(alpha)

            # Vectorized quantum delta-potential update
            phi = rng.uniform(0.0, 1.0, size=(self.population_size, dim))
            p = phi * pbest + (1.0 - phi) * gbest
            u = rng.uniform(1e-12, 1.0, size=(self.population_size, dim))
            signs = rng.choice(np.array([-1.0, 1.0]), size=(self.population_size, dim))
            step = signs * alpha * np.abs(mbest - X) * np.log(1.0 / u)
            X_new = np.clip(p + step, lb, ub)

            # Stagnation-triggered quantum re-exploration of worst particles
            if stagnation_counter >= self.stagnation_patience and self.reinit_fraction > 0:
                n_reinit = max(1, int(round(self.reinit_fraction * self.population_size)))
                worst_indices = np.argsort(pbest_scores)[-n_reinit:]
                anchor = elite_archive[0][1] if elite_archive else gbest
                for idx in worst_indices:
                    s_u = rng.uniform(1e-12, 1.0, size=dim)
                    s_signs = rng.choice(np.array([-1.0, 1.0]), size=dim)
                    s_step = s_signs * alpha * np.abs(mbest - X[idx]) * np.log(1.0 / s_u)
                    X_new[idx] = np.clip(anchor + s_step, lb, ub)
                total_reinit_count += n_reinit

            X = X_new
            improved = False

            for i in range(self.population_size):
                score, res = decode_and_eval(X[i])
                if score < pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest[i] = X[i].copy()
                    if score < gbest_score - 1e-4:
                        gbest_score = score
                        gbest = X[i].copy()
                        best_eval_dict = res
                        improved = True

            if improved:
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            stagnation_history.append(stagnation_counter)
            history.append(gbest_score)

        elapsed = time.perf_counter() - start_time
        assert best_eval_dict is not None

        # Verify best elite solution if available
        if elite_archive:
            best_elite = min(elite_archive, key=lambda e: e[2])
            if best_elite[2] < best_eval_dict["objective_score"]:
                best_eval_dict = best_elite[3]

        conv_info = {
            "initial_objective": round(float(history[0]), 4),
            "final_objective": round(float(history[-1]), 4),
            "improvement_pct": round(float(history[0] - history[-1]) / max(abs(history[0]), 1e-4) * 100.0, 2),
            "history": history,
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
            convergence_score=round(float(history[0] - history[-1]) / max(abs(history[0]), 1e-4), 4),
            feasible_solution=best_eval_dict["feasible_solution"],
            n_evaluations=eval_count,
            convergence_information=conv_info,
            metadata={
                "fleet_mix": best_eval_dict["fleet_mix"],
                "speed_knots": best_eval_dict["speed_knots"],
                "history": history,
                "alpha_history": alpha_history,
                "diversity_history": diversity_history,
                "stagnation_history": stagnation_history,
                "reinitialization_count": total_reinit_count,
                "elite_archive_size": len(elite_archive),
                "cache_hits": cache.hits,
                "cache_misses": cache.misses,
                "cache_hit_rate": cache.hit_rate,
                "unique_evaluations": cache.misses,
                "mean_repair_distance": round(float(np.mean(repair_distances)), 4) if repair_distances else 0.0,
                "is_quantum": True,
            },
        )
