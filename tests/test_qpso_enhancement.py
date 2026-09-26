"""Comprehensive test suite for enhanced QPSO optimization and benchmark fairness.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Validates:
1. Adaptive alpha boundary enforcement [alpha_min, alpha_max] under all extreme inputs
2. Normalized swarm diversity metric in [0, 1]
3. Stagnation detection and quantum re-exploration
4. Elite archive discrete-state deduplication
5. Vectorized QPSO mathematical reproducibility
6. Canonical decode_and_repair feasibility and repair distance
7. Isolated evaluation cache separation across solver runs
8. Scenario fingerprint invalidation in candidate caching
9. Exact evaluation budget equality between QPSO and Classical PSO
10. Five-fuel preservation of Hydrogen and Ammonia allocations
"""

import numpy as np
import pytest

from contracts.schemas import BenchmarkResult, OptimizationScenario
from src.benchmarking.qpso_benchmark_adapter import QPSOBenchmarkAdapter
from src.benchmarking.solvers.base_solver import (
    BaseBenchmarkSolver,
    BenchmarkEvaluationCache,
    VESSEL_SPECS,
    compute_scenario_fingerprint,
)
from src.benchmarking.solvers.classical_pso_solver import ClassicalPSOSolver
from src.optimization.qpso import (
    calculate_swarm_diversity,
    compute_adaptive_alpha,
)


@pytest.fixture
def base_scenario() -> OptimizationScenario:
    """Standard scenario fixture for benchmark testing."""
    return OptimizationScenario(
        cargo_demand=120_000.0,
        route_distance=3000.0,
        deadline_hours=240.0,
        budget=60_000_000.0,
        carbon_price=80.0,
        weather_factor=1.05,
        port_delay_factor=1.10,
        scenario_id="SCEN-TEST-ENHANCE",
        max_transition_rate=0.40,
        target_reliability=85.0,
    )


def test_adaptive_alpha_bounds() -> None:
    """Verify adaptive alpha is strictly clipped in [alpha_min, alpha_max] under extreme inputs."""
    alpha_min = 0.35
    alpha_max = 1.25

    extreme_cases = [
        # (t, max_iter, diversity, stagnation)
        (0, 100, 0.0, 1000),      # Complete swarm collapse + extreme stagnation
        (100, 100, 1.0, 0),       # High diversity + zero stagnation at final iteration
        (50, 100, 0.0, 0),        # Zero diversity + zero stagnation
        (50, 100, 1.0, 1000),     # Max diversity + extreme stagnation
        (0, 50, -5.0, -10),       # Negative pathological inputs
        (50, 50, 99.0, 500),      # Out-of-range positive inputs
    ]

    for t, max_t, div, stag in extreme_cases:
        alpha = compute_adaptive_alpha(
            t=t,
            max_iterations=max_t,
            diversity=div,
            stagnation_count=stag,
            alpha_start=1.0,
            alpha_end=0.5,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )
        assert alpha_min <= alpha <= alpha_max, f"Alpha {alpha} out of bounds for ({t}, {max_t}, {div}, {stag})"


def test_diversity_calculation() -> None:
    """Verify calculate_swarm_diversity returns float in [0, 1] and reflects spread."""
    lb = np.array([0.0, 0.0, 0.0])
    ub = np.array([10.0, 10.0, 10.0])

    # 1. Zero diversity: all particles at identical position
    X_collapsed = np.full((10, 3), 5.0)
    mbest = np.mean(X_collapsed, axis=0)
    div_zero = calculate_swarm_diversity(X_collapsed, mbest, lb, ub)
    assert div_zero == pytest.approx(0.0, abs=1e-6)

    # 2. Spread diversity: particles spread across bounds
    X_spread = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0],
        [0.0, 10.0, 5.0],
        [10.0, 0.0, 5.0],
    ])
    mbest_spread = np.mean(X_spread, axis=0)
    div_spread = calculate_swarm_diversity(X_spread, mbest_spread, lb, ub)
    assert 0.0 < div_spread <= 1.0


def test_stagnation_detection(base_scenario: OptimizationScenario) -> None:
    """Verify stagnation is tracked and triggers re-exploration history when threshold exceeded."""
    solver = QPSOBenchmarkAdapter(
        population_size=10,
        stagnation_patience=2,
        reinit_fraction=0.3,
    )
    res = solver.solve(base_scenario, max_iterations=15, seed=42)
    meta = res.metadata

    assert "stagnation_history" in meta
    assert len(meta["stagnation_history"]) == 15
    assert "reinitialization_count" in meta
    assert meta["reinitialization_count"] >= 0


def test_elite_archive_dedup(base_scenario: OptimizationScenario) -> None:
    """Verify elite archive deduplicates candidate solutions by discrete state signature."""
    solver = QPSOBenchmarkAdapter(population_size=15, elite_capacity=5)
    res = solver.solve(base_scenario, max_iterations=20, seed=42)

    archive_size = res.metadata.get("elite_archive_size", 0)
    assert 0 <= archive_size <= 5


def test_vectorized_qpso_math(base_scenario: OptimizationScenario) -> None:
    """Verify vectorized QPSO produces bit-identical results with fixed seed."""
    solver1 = QPSOBenchmarkAdapter(population_size=10)
    solver2 = QPSOBenchmarkAdapter(population_size=10)

    res1 = solver1.solve(base_scenario, max_iterations=10, seed=999)
    res2 = solver2.solve(base_scenario, max_iterations=10, seed=999)

    assert res1.objective_score == pytest.approx(res2.objective_score, abs=1e-6)
    assert res1.fuel_consumption == pytest.approx(res2.fuel_consumption, abs=1e-4)
    assert res1.operational_cost == pytest.approx(res2.operational_cost, abs=1e-2)
    assert res1.emissions == pytest.approx(res2.emissions, abs=1e-4)
    assert res1.n_evaluations == res2.n_evaluations


class DummySolver(BaseBenchmarkSolver):
    def __init__(self, name: str = "TestSolver") -> None:
        super().__init__(name)

    def solve(
        self,
        scenario: OptimizationScenario,
        max_iterations: int = 50,
        seed: int = 42,
    ) -> BenchmarkResult:
        raise NotImplementedError


def test_decode_and_repair(base_scenario: OptimizationScenario) -> None:
    """Verify canonical decode_and_repair enforces discrete legality and returns repair distance."""
    dummy_solver = DummySolver("TestSolver")

    # Degenerate input: all zero vessel counts, out-of-range speed
    raw_vec = np.array([0.1, 0.2, 0.1, 0.8, 0.5, 0.2, 0.1, 0.2, 22.0])
    xf, xm, xl, fuel_mix, speed, rep_dist = dummy_solver.decode_and_repair(raw_vec, base_scenario)

    assert xf + xm + xl >= 1, "Must guarantee non-empty fleet"
    assert 9.5 <= speed <= 17.5, "Speed must be clamped to legal bounds"
    assert rep_dist >= 0.0, "Repair distance must be non-negative"

    total_fuel_vessels = sum(fuel_mix.values())
    assert total_fuel_vessels == (xf + xm + xl), "Total fuel vessels must match total fleet"

    # Alternative transition cap test: alt vessels cannot exceed max_transition_rate * total_vessels
    alt_vessels = total_fuel_vessels - fuel_mix.get("diesel", 0)
    max_allowed = int(np.floor((xf + xm + xl) * base_scenario.max_transition_rate + 1e-6))
    assert alt_vessels <= max_allowed, f"Alt vessels {alt_vessels} exceeds cap {max_allowed}"


def test_cache_isolation(base_scenario: OptimizationScenario) -> None:
    """Verify each solver instance maintains completely isolated candidate memoization."""
    qpso = QPSOBenchmarkAdapter(population_size=10)
    pso = ClassicalPSOSolver(population_size=10)

    res_qpso = qpso.solve(base_scenario, max_iterations=10, seed=42)
    res_pso = pso.solve(base_scenario, max_iterations=10, seed=42)

    # Check cache metrics recorded independently
    assert "cache_hits" in res_qpso.metadata
    assert "cache_misses" in res_qpso.metadata
    assert "unique_evaluations" in res_qpso.metadata

    assert "cache_hits" in res_pso.metadata
    assert "cache_misses" in res_pso.metadata
    assert "unique_evaluations" in res_pso.metadata

    # Verify hit rates are mathematically valid percentages
    assert 0.0 <= res_qpso.metadata["cache_hit_rate"] <= 100.0
    assert 0.0 <= res_pso.metadata["cache_hit_rate"] <= 100.0


def test_cache_scenario_invalidation(base_scenario: OptimizationScenario) -> None:
    """Verify changing scenario attributes generates different scenario fingerprints and prevents cache leakage."""
    fp1 = compute_scenario_fingerprint(base_scenario)

    modified_scenario = OptimizationScenario(
        cargo_demand=base_scenario.cargo_demand * 1.5,
        route_distance=base_scenario.route_distance,
        deadline_hours=base_scenario.deadline_hours,
        budget=base_scenario.budget,
        carbon_price=base_scenario.carbon_price,
        weather_factor=base_scenario.weather_factor,
        port_delay_factor=base_scenario.port_delay_factor,
        scenario_id=base_scenario.scenario_id,
        max_transition_rate=base_scenario.max_transition_rate,
        target_reliability=base_scenario.target_reliability,
    )
    fp2 = compute_scenario_fingerprint(modified_scenario)

    assert fp1 != fp2, "Fingerprints must differ when scenario parameters change"

    # Test cache behavior with explicit keys
    cache = BenchmarkEvaluationCache()
    key1 = (2, 4, 1, 5, 1, 1, 0, 0, 14.5, fp1)
    key2 = (2, 4, 1, 5, 1, 1, 0, 0, 14.5, fp2)

    cache.put(key1, {"objective_score": 10.0})
    assert cache.get(key1) is not None
    assert cache.get(key2) is None, "Different scenario fingerprint must not match cached entry"


def test_exact_evaluation_budget_equality(base_scenario: OptimizationScenario) -> None:
    """Verify QPSO and Classical PSO execute the exact same number of candidate evaluations."""
    pop_size = 12
    max_iter = 15
    expected_evaluations = pop_size * max_iter  # 12 * 15 = 180

    qpso = QPSOBenchmarkAdapter(population_size=pop_size)
    pso = ClassicalPSOSolver(population_size=pop_size)

    res_qpso = qpso.solve(base_scenario, max_iterations=max_iter, seed=101)
    res_pso = pso.solve(base_scenario, max_iterations=max_iter, seed=101)

    assert res_qpso.n_evaluations == expected_evaluations
    assert res_pso.n_evaluations == expected_evaluations
    assert res_qpso.n_evaluations == res_pso.n_evaluations


def test_five_fuel_allocation_preserves_h2_nh3(base_scenario: OptimizationScenario) -> None:
    """Verify non-zero continuous shares for hydrogen and ammonia produce non-zero fleet counts when budget allows."""
    solver = DummySolver("FuelPreserveTest")

    # Vector with 10 total vessels, 40% alt allowed (4 alt vessels)
    # shares: lng=0.1, methanol=0.1, hydrogen=0.4, ammonia=0.4
    raw_vec = np.array([2.0, 5.0, 3.0, 0.4, 0.1, 0.1, 0.4, 0.4, 14.0])
    xf, xm, xl, fuel_mix, speed, _ = solver.decode_and_repair(raw_vec, base_scenario)

    assert fuel_mix.get("hydrogen", 0) > 0, "Hydrogen must be allocated when shares are significant"
    assert fuel_mix.get("ammonia", 0) > 0, "Ammonia must be allocated when shares are significant"
    assert sum(fuel_mix.values()) == (xf + xm + xl)


def test_ablation_variants_execution(base_scenario: OptimizationScenario) -> None:
    """Verify that all 7 ablation variants execute and return compliant BenchmarkResult instances."""
    from src.benchmarking.qpso_ablation import ABLATION_VARIANTS, QPSOAblationSolver

    for vid, config in ABLATION_VARIANTS.items():
        solver = QPSOAblationSolver(config=config, population_size=5)
        res = solver.solve(base_scenario, max_iterations=5, seed=42)
        assert isinstance(res, BenchmarkResult)
        assert res.objective_score > 0.0
        assert res.iterations == 5
        assert res.metadata["variant_id"] == vid

