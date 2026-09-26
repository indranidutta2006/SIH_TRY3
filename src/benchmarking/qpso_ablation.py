"""QPSO Ablation Study Module.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 3 Deliverable: Systematically isolate each enhancement across 7 variants (A through G).

Variants:
- Variant A: Baseline QPSO (linear alpha, no re-exploration, no elite archive, naive round)
- Variant B: Adaptive Alpha (diversity & stagnation modulated alpha)
- Variant C: Adaptive Alpha + Stagnation Re-exploration (worst-particle update)
- Variant D: Adaptive Alpha + Re-exploration + Elite Archive (K=5)
- Variant E: Adaptive Alpha + Re-exploration + Elite Archive + Discrete Repair
- Variant F: Full Enhanced QPSO (All mechanisms active + memoized evaluation cache)
- Variant G: Full Enhanced QPSO + Deterministic Local Refinement (Secondary ablation)
"""

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from contracts.schemas import BenchmarkResult, OptimizationScenario
from src.benchmarking.solvers.base_solver import (
    BaseBenchmarkSolver,
    BenchmarkEvaluationCache,
    VESSEL_SPECS,
)
from src.optimization.qpso import calculate_swarm_diversity, compute_adaptive_alpha


@dataclass(frozen=True)
class AblationConfig:
    variant_id: str
    name: str
    use_adaptive_alpha: bool
    use_reexploration: bool
    use_elite_archive: bool
    use_discrete_repair: bool
    use_local_refinement: bool
    description: str


ABLATION_VARIANTS: dict[str, AblationConfig] = {
    "A": AblationConfig(
        variant_id="A",
        name="Baseline QPSO",
        use_adaptive_alpha=False,
        use_reexploration=False,
        use_elite_archive=False,
        use_discrete_repair=False,
        use_local_refinement=False,
        description="Linear alpha decay, standard delta-potential, naive rounding",
    ),
    "B": AblationConfig(
        variant_id="B",
        name="Adaptive Alpha",
        use_adaptive_alpha=True,
        use_reexploration=False,
        use_elite_archive=False,
        use_discrete_repair=False,
        use_local_refinement=False,
        description="Diversity- & stagnation-modulated contraction-expansion alpha",
    ),
    "C": AblationConfig(
        variant_id="C",
        name="Adaptive Alpha + Re-exploration",
        use_adaptive_alpha=True,
        use_reexploration=True,
        use_elite_archive=False,
        use_discrete_repair=False,
        use_local_refinement=False,
        description="Stagnation-triggered quantum re-exploration of worst particles",
    ),
    "D": AblationConfig(
        variant_id="D",
        name="Adaptive Alpha + Re-expl + Elite Archive",
        use_adaptive_alpha=True,
        use_reexploration=True,
        use_elite_archive=True,
        use_discrete_repair=False,
        use_local_refinement=False,
        description="Elite archive tracking diverse feasible discrete states",
    ),
    "E": AblationConfig(
        variant_id="E",
        name="Adaptive Alpha + Re-expl + Archive + Repair",
        use_adaptive_alpha=True,
        use_reexploration=True,
        use_elite_archive=True,
        use_discrete_repair=True,
        use_local_refinement=False,
        description="Canonical discrete repair operator with minimum fleet guarantee",
    ),
    "F": AblationConfig(
        variant_id="F",
        name="Full Enhanced QPSO",
        use_adaptive_alpha=True,
        use_reexploration=True,
        use_elite_archive=True,
        use_discrete_repair=True,
        use_local_refinement=False,
        description="Full production QPSO with isolated candidate evaluation cache",
    ),
    "G": AblationConfig(
        variant_id="G",
        name="Full Enhanced QPSO + Local Refinement",
        use_adaptive_alpha=True,
        use_reexploration=True,
        use_elite_archive=True,
        use_discrete_repair=True,
        use_local_refinement=True,
        description="Full Enhanced QPSO with post-hoc deterministic 1-opt local search",
    ),
}


class QPSOAblationSolver(BaseBenchmarkSolver):
    """Configurable QPSO solver specifically for ablation experiments."""

    def __init__(self, config: AblationConfig, population_size: int = 20) -> None:
        super().__init__(solver_name=f"QPSO-Variant-{config.variant_id} ({config.name})")
        self.config = config
        self.population_size = population_size
        self.alpha_start = 1.0
        self.alpha_end = 0.5
        self.alpha_min = 0.35
        self.alpha_max = 1.25
        self.stagnation_patience = 5
        self.reinit_fraction = 0.25
        self.elite_capacity = 5

    def solve(
        self,
        scenario: OptimizationScenario,
        max_iterations: int = 50,
        seed: int = 42,
    ) -> BenchmarkResult:
        start_time = time.perf_counter()
        rng = np.random.default_rng(seed)

        # Isolated evaluation cache for this run
        cache = BenchmarkEvaluationCache()

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
        elite_archive: list[tuple[tuple[int, ...], np.ndarray, float, dict[str, Any]]] = []

        def decode_and_eval(vec: np.ndarray) -> tuple[float, dict[str, Any]]:
            nonlocal eval_count
            eval_count += 1

            if self.config.use_discrete_repair:
                xf, xm, xl, fuel_mix, speed, rep_dist = self.decode_and_repair(vec, scenario)
                repair_distances.append(rep_dist)
            else:
                xf = int(round(vec[0]))
                xm = int(round(vec[1]))
                xl = int(round(vec[2]))
                tot = xf + xm + xl
                if tot == 0:
                    tot = 1
                    xm = 1
                alt_ratio = float(np.clip(vec[3], 0.0, scenario.max_transition_rate))
                alt_count = int(round(tot * alt_ratio))
                shares = [float(vec[4]), float(vec[5]), float(vec[6]), float(vec[7])]
                speed = float(np.clip(vec[8], 9.5, 17.5))
                fuel_mix = self.allocate_fuel_mix(tot, alt_count, scenario, shares=shares)
                repair_distances.append(0.0)

            res = self.evaluate_candidate(xf, xm, xl, fuel_mix, speed, scenario, cache=cache)
            score = res["objective_score"]

            if self.config.use_elite_archive and res.get("feasible_solution", False):
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

        # Swarm initialization
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
        stagnation_counter = 0

        for t in range(1, max_iterations):
            mbest = np.mean(pbest, axis=0)

            if self.config.use_adaptive_alpha:
                div = calculate_swarm_diversity(X, mbest, lb, ub)
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

            # Standard quantum update
            phi = rng.uniform(0.0, 1.0, size=(self.population_size, dim))
            p = phi * pbest + (1.0 - phi) * gbest
            u = rng.uniform(1e-12, 1.0, size=(self.population_size, dim))
            signs = rng.choice(np.array([-1.0, 1.0]), size=(self.population_size, dim))
            step = signs * alpha * np.abs(mbest - X) * np.log(1.0 / u)
            X_new = np.clip(p + step, lb, ub)

            # Stagnation re-exploration
            if (
                self.config.use_reexploration
                and stagnation_counter >= self.stagnation_patience
                and self.reinit_fraction > 0
            ):
                n_reinit = max(1, int(round(self.reinit_fraction * self.population_size)))
                worst_indices = np.argsort(pbest_scores)[-n_reinit:]
                anchor = elite_archive[0][1] if elite_archive else gbest
                for idx in worst_indices:
                    s_u = rng.uniform(1e-12, 1.0, size=dim)
                    s_signs = rng.choice(np.array([-1.0, 1.0]), size=dim)
                    s_step = s_signs * alpha * np.abs(mbest - X[idx]) * np.log(1.0 / s_u)
                    X_new[idx] = np.clip(anchor + s_step, lb, ub)

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

            history.append(gbest_score)

        # Check elite archive
        if self.config.use_elite_archive and elite_archive:
            best_elite = min(elite_archive, key=lambda e: e[2])
            if best_elite[2] < best_eval_dict["objective_score"]:
                best_eval_dict = best_elite[3]

        # Variant G: Optional deterministic 1-opt local search refinement
        if self.config.use_local_refinement and best_eval_dict is not None:
            current_best_score = best_eval_dict["objective_score"]
            cur_vec = gbest.copy()
            # Try small perturbations in vessel counts and speed
            for dim_idx, delta in [(0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0), (2, 1.0), (2, -1.0), (8, 0.5), (8, -0.5)]:
                cand_vec = cur_vec.copy()
                cand_vec[dim_idx] = np.clip(cand_vec[dim_idx] + delta, lb[dim_idx], ub[dim_idx])
                c_score, c_res = decode_and_eval(cand_vec)
                if c_score < current_best_score and c_res.get("feasible_solution", False):
                    current_best_score = c_score
                    best_eval_dict = c_res
                    cur_vec = cand_vec

        elapsed = time.perf_counter() - start_time
        assert best_eval_dict is not None

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
                "variant_id": self.config.variant_id,
                "variant_name": self.config.name,
                "fleet_mix": best_eval_dict["fleet_mix"],
                "speed_knots": best_eval_dict["speed_knots"],
                "history": history,
                "cache_hits": cache.hits,
                "cache_misses": cache.misses,
                "cache_hit_rate": cache.hit_rate,
                "unique_evaluations": cache.misses,
                "mean_repair_distance": round(float(np.mean(repair_distances)), 4) if repair_distances else 0.0,
                "is_quantum": True,
            },
        )


def run_ablation_study(
    scenario: OptimizationScenario,
    seeds: list[int] | range = range(100, 130),
    max_iterations: int = 50,
    population_size: int = 20,
) -> dict[str, Any]:
    """Execute all 7 ablation variants across multiple seeds and compute summary metrics."""
    seed_list = list(seeds)
    results: dict[str, list[BenchmarkResult]] = {vid: [] for vid in ABLATION_VARIANTS}

    for vid, config in ABLATION_VARIANTS.items():
        solver = QPSOAblationSolver(config=config, population_size=population_size)
        for seed in seed_list:
            res = solver.solve(scenario, max_iterations=max_iterations, seed=seed)
            results[vid].append(res)

    summary: dict[str, dict[str, Any]] = {}
    for vid, config in ABLATION_VARIANTS.items():
        res_list = results[vid]
        objs = [r.objective_score for r in res_list]
        runtimes = [r.runtime_seconds for r in res_list]
        evals = [r.n_evaluations for r in res_list]
        unique_evals = [r.metadata.get("unique_evaluations", r.n_evaluations) for r in res_list]
        feasibles = [r.feasible_solution for r in res_list]
        fuels = [r.fuel_consumption for r in res_list]
        costs = [r.operational_cost for r in res_list]
        emissions = [r.emissions for r in res_list]

        summary[vid] = {
            "variant_id": vid,
            "name": config.name,
            "description": config.description,
            "runs": len(res_list),
            "mean_objective": round(float(np.mean(objs)), 4),
            "std_objective": round(float(np.std(objs)), 4),
            "best_objective": round(float(np.min(objs)), 4),
            "worst_objective": round(float(np.max(objs)), 4),
            "feasible_rate": round(float(sum(feasibles)) / len(feasibles) * 100.0, 2),
            "mean_fuel_t": round(float(np.mean(fuels)), 2),
            "mean_cost_usd": round(float(np.mean(costs)), 2),
            "mean_emissions_tco2e": round(float(np.mean(emissions)), 2),
            "mean_runtime_s": round(float(np.mean(runtimes)), 4),
            "mean_total_evals": round(float(np.mean(evals)), 1),
            "mean_unique_evals": round(float(np.mean(unique_evals)), 1),
        }

    return {
        "n_seeds": len(seed_list),
        "seeds": seed_list,
        "max_iterations": max_iterations,
        "population_size": population_size,
        "variants": summary,
    }
