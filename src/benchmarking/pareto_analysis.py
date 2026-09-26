"""Multi-objective Pareto frontier extraction and non-dominated sorting.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 3 Deliverable: Extracts non-dominated Pareto frontiers across Fuel, Cost, Emissions,
and Reliability objectives, supporting multi-criteria trade-off decision intelligence.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from contracts.schemas import OptimizationScenario, ParetoResult


class ParetoAnalyzer:
    """Extracts non-dominated Pareto frontiers across multi-dimensional maritime criteria."""

    def extract_pareto_front(
        self,
        candidate_evaluations: Sequence[dict[str, Any]],
        scenario_id: str = "SCENARIO-001",
        solver_name: str = "Multi-Objective Benchmark",
    ) -> ParetoResult:
        """Perform fast non-dominated sorting across Fuel, Cost, Emissions, and Reliability.

        Dominance Criterion:
            Point A dominates B (A ≻ B) if:
                Cost(A) <= Cost(B)
                Fuel(A) <= Fuel(B)
                Emissions(A) <= Emissions(B)
                Reliability(A) >= Reliability(B)
            with strict inequality in at least one objective.

        Args:
            candidate_evaluations: List of candidate metrics dicts containing fuel, cost, emissions, reliability.
            scenario_id: Scenario identifier.
            solver_name: Solver or suite name.

        Returns:
            ParetoResult with pareto_points, dominated_points, and hypervolume indicator.
        """
        if not candidate_evaluations:
            return ParetoResult(
                scenario_id=scenario_id,
                solver_name=solver_name,
                pareto_points=(),
                dominated_points=(),
                hypervolume=0.0,
            )

        n = len(candidate_evaluations)
        costs = np.array([float(c.get("operational_cost", 1e9)) for c in candidate_evaluations])
        fuels = np.array([float(c.get("fuel_consumption", 1e9)) for c in candidate_evaluations])
        emiss = np.array([float(c.get("emissions", 1e9)) for c in candidate_evaluations])
        relis = np.array([float(c.get("reliability_score", 0.0)) for c in candidate_evaluations])

        # Vectorized or pair-wise non-dominated check
        is_dominated = np.zeros(n, dtype=bool)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Does j dominate i?
                # j <= i for min objectives, j >= i for max objectives
                j_no_worse = (
                    costs[j] <= costs[i]
                    and fuels[j] <= fuels[i]
                    and emiss[j] <= emiss[i]
                    and relis[j] >= relis[i]
                )
                j_strictly_better = (
                    costs[j] < costs[i]
                    or fuels[j] < fuels[i]
                    or emiss[j] < emiss[i]
                    or relis[j] > relis[i]
                )
                if j_no_worse and j_strictly_better:
                    is_dominated[i] = True
                    break

        pareto_indices = np.where(~is_dominated)[0]
        dominated_indices = np.where(is_dominated)[0]

        pareto_points: list[dict[str, Any]] = []
        for idx in pareto_indices:
            p_dict = dict(candidate_evaluations[idx])
            p_dict["operational_cost"] = round(float(costs[idx]), 2)
            p_dict["fuel_consumption"] = round(float(fuels[idx]), 2)
            p_dict["emissions"] = round(float(emiss[idx]), 2)
            p_dict["reliability_score"] = round(float(relis[idx]), 2)
            p_dict["objective_score"] = round(float(candidate_evaluations[idx].get("objective_score", 0.0)), 4)
            pareto_points.append(p_dict)

        dominated_points: list[dict[str, Any]] = []
        for idx in dominated_indices:
            d_dict = dict(candidate_evaluations[idx])
            d_dict["operational_cost"] = round(float(costs[idx]), 2)
            d_dict["fuel_consumption"] = round(float(fuels[idx]), 2)
            d_dict["emissions"] = round(float(emiss[idx]), 2)
            d_dict["reliability_score"] = round(float(relis[idx]), 2)
            d_dict["objective_score"] = round(float(candidate_evaluations[idx].get("objective_score", 0.0)), 4)
            dominated_points.append(d_dict)

        # Calculate normalized 2D Hypervolume indicator between Cost and Fuel relative to nadir point
        c_min, c_max = float(np.min(costs)), float(np.max(costs) + 1e-4)
        f_min, f_max = float(np.min(fuels)), float(np.max(fuels) + 1e-4)

        norm_p = []
        for p in pareto_points:
            nc = (p["operational_cost"] - c_min) / (c_max - c_min)
            nf = (p["fuel_consumption"] - f_min) / (f_max - f_min)
            norm_p.append((nc, nf))

        # Sort by first objective
        norm_p.sort(key=lambda pt: pt[0])
        hv = 0.0
        cur_f = 1.0
        for nc, nf in norm_p:
            if nf < cur_f:
                hv += (1.0 - nc) * (cur_f - nf)
                cur_f = nf

        return ParetoResult(
            scenario_id=scenario_id,
            solver_name=solver_name,
            pareto_points=tuple(pareto_points),
            dominated_points=tuple(dominated_points),
            hypervolume=round(float(np.clip(hv, 0.0, 1.0)), 4),
            metadata={
                "total_candidates": n,
                "pareto_front_size": len(pareto_points),
                "dominance_ratio": round(len(pareto_points) / max(n, 1), 4),
            },
        )
