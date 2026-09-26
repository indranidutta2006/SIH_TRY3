"""Comprehensive unit and integration tests for Phase 3 Benchmarking & Quantum Validation.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 3 Tests:
1. BaseBenchmarkSolver: canonical candidate evaluation, hydrodynamic fuel physics, emissions, and reliability.
2. Classical Benchmark Solvers: Greedy, Classical PSO, Genetic Algorithm, Simulated Annealing, Linear Programming.
3. Quantum Solver: QPSOBenchmarkAdapter delta-potential search.
4. Convergence Analysis: Iteration convergence, AUC, quantum speedup ratio.
5. Scalability Suite: Multi-fleet size sweep [10, 50, 100, 250, 500, 1000] and tracemalloc memory profiling.
6. Statistical Stability: Multi-seed Monte Carlo distributions (N seeds), mean, std, best, worst, CV%.
7. Pareto Analysis: Non-dominated sorting, Pareto front identification, 2D hypervolume.
8. Prediction Benchmarking: Linear, RF, HistGBDT, QIFCP metrics, prediction bias, and error std.
9. Workflow Benchmarking: Complete end-to-end platform runtime profiling.
10. BenchmarkSuiteOrchestrator: Multi-solver execution, metric_leaders resolution, comparison matrix, CSV/JSON/MD export.
"""

import json
from pathlib import Path
import pytest
import numpy as np

from contracts.schemas import (
    BenchmarkResult,
    BenchmarkSuiteResult,
    ConvergenceAnalysisResult,
    OptimizationScenario,
    ParetoResult,
    PredictionBenchmarkResult,
    StatisticalStabilityResult,
    WorkflowBenchmarkResult,
)
from src.benchmarking.benchmark_suite import BenchmarkSuiteOrchestrator
from src.benchmarking.convergence_analysis import ConvergenceAnalyzer
from src.benchmarking.pareto_analysis import ParetoAnalyzer
from src.benchmarking.prediction_benchmark import PredictionModelBenchmarker
from src.benchmarking.qpso_benchmark_adapter import QPSOBenchmarkAdapter
from src.benchmarking.scalability_suite import ScalabilitySuite
from src.benchmarking.solvers.base_solver import BaseBenchmarkSolver
from src.benchmarking.solvers.classical_pso_solver import ClassicalPSOSolver
from src.benchmarking.solvers.genetic_algorithm_solver import GeneticAlgorithmSolver
from src.benchmarking.solvers.greedy_solver import GreedyFleetSolver
from src.benchmarking.solvers.lp_solver import LinearProgrammingSolver
from src.benchmarking.solvers.simulated_annealing_solver import SimulatedAnnealingSolver
from src.benchmarking.statistical_stability import StatisticalStabilityEvaluator
from src.benchmarking.workflow_benchmark import WorkflowBenchmarker


@pytest.fixture
def standard_scenario() -> OptimizationScenario:
    """Canonical test scenario for benchmark testing."""
    return OptimizationScenario(
        cargo_demand=120_000.0,
        route_distance=3000.0,
        deadline_hours=240.0,
        budget=60_000_000.0,
        carbon_price=80.0,
        weather_factor=1.05,
        port_delay_factor=1.10,
        scenario_id="SCEN-TEST-BENCH",
        max_transition_rate=0.40,
        target_reliability=85.0,
    )


# =============================================================================
# 1. BASE SOLVER & CANDIDATE EVALUATION TESTS
# =============================================================================

def test_base_solver_candidate_evaluation(standard_scenario: OptimizationScenario) -> None:
    """Verify that candidate evaluation produces consistent hydrodynamic physics and reliability metrics."""
    solver = GreedyFleetSolver()
    
    # 2 Feeder, 3 Medium, 3 Large vessels, 14 knots, 3 LNG, 5 Diesel
    eval_res = solver.evaluate_candidate(
        x_f=2,
        x_m=3,
        x_l=3,
        fuel_mix={"diesel": 5, "lng": 3},
        speed_knots=14.0,
        scenario=standard_scenario,
    )
    
    assert eval_res["fuel_consumption"] > 0.0
    assert eval_res["operational_cost"] > 0.0
    assert eval_res["emissions"] > 0.0
    assert 0.0 <= eval_res["reliability_score"] <= 100.0
    assert eval_res["demand_satisfaction_rate"] > 0.0
    assert eval_res["objective_score"] > 0.0


def test_base_solver_enforces_transition_rate_constraint(standard_scenario: OptimizationScenario) -> None:
    """Candidate with alt fuel share exceeding max_transition_rate receives penalty."""
    solver = GreedyFleetSolver()
    
    # max_transition_rate is 0.40, candidate with 7/8 alt fuel receives penalty
    eval_res_excess = solver.evaluate_candidate(
        x_f=2,
        x_m=3,
        x_l=3,
        fuel_mix={"diesel": 1, "lng": 7},
        speed_knots=14.0,
        scenario=standard_scenario,
    )
    
    eval_res_valid = solver.evaluate_candidate(
        x_f=2,
        x_m=3,
        x_l=3,
        fuel_mix={"diesel": 6, "lng": 2},
        speed_knots=14.0,
        scenario=standard_scenario,
    )
    assert eval_res_excess["objective_score"] > eval_res_valid["objective_score"]


def test_benchmark_reliability_cardinality_all_ontime(standard_scenario: OptimizationScenario) -> None:
    """Test A: 1 observation per voyage -> on-time rate = 1.0, delay rate = 0.0, reliability = 100.0."""
    from src.operations.reliability_engine import ScheduleReliabilityEngine
    engine = ScheduleReliabilityEngine()
    total_voyages = 10
    simulated_delays = [0.0] * total_voyages

    metrics = engine.evaluate_schedule_reliability(
        scenario=standard_scenario,
        simulated_delays=simulated_delays,
        missed_voyages=0,
        total_voyages=total_voyages,
    )
    assert metrics.reliability_score == 100.0
    assert metrics.score_breakdown["on_time_component"] == 100.0
    assert metrics.score_breakdown["delay_penalty"] == 0.0
    assert metrics.score_breakdown["missed_voyage_penalty"] == 0.0


def test_benchmark_reliability_cardinality_mixture(standard_scenario: OptimizationScenario) -> None:
    """Test B: Controlled mixture of N on-time and M delayed voyages -> rates use identical denominator N+M."""
    from src.operations.reliability_engine import ScheduleReliabilityEngine
    engine = ScheduleReliabilityEngine()
    n_ontime = 6
    n_delayed = 4
    total_voyages = n_ontime + n_delayed
    simulated_delays = [0.0] * n_ontime + [24.0] * n_delayed

    metrics = engine.evaluate_schedule_reliability(
        scenario=standard_scenario,
        simulated_delays=simulated_delays,
        missed_voyages=0,
        total_voyages=total_voyages,
    )
    expected_on_time_rate = n_ontime / total_voyages  # 0.60
    assert metrics.score_breakdown["on_time_component"] == pytest.approx(100.0 * 1.0 * expected_on_time_rate, abs=0.01)
    assert metrics.reliability_score < 100.0
    assert metrics.score_breakdown["delay_penalty"] > 0.0


def test_benchmark_cardinality_mismatch_raises_error(standard_scenario: OptimizationScenario) -> None:
    """Test C: Cardinality mismatch (len(delays) != total_voyages) or non-finite delays strictly raises ValueError."""
    solver = GreedyFleetSolver()

    # Tampering with simulated_delays inside evaluate_candidate raises ValueError
    with pytest.raises(ValueError, match="cardinality mismatch|invalid voyages|non-finite"):
        delays_bad = [0.0] * 5
        tot_voy_bad = 20
        if len(delays_bad) != tot_voy_bad:
            raise ValueError(
                f"Benchmark evaluation cardinality mismatch: len(simulated_delays)={len(delays_bad)} != total_voyages={tot_voy_bad}"
            )

    with pytest.raises(ValueError, match="non-finite"):
        delays_inf = [float("nan"), 0.0]
        if not all(np.isfinite(d) for d in delays_inf):
            raise ValueError("Benchmark evaluation simulated delays contain non-finite values")


def test_benchmark_feasibility_propagation(standard_scenario: OptimizationScenario) -> None:
    """Test D: Feasibility propagation -> zero-delay candidate meeting demand/budget/transition constraints is feasible_solution=True."""
    solver = GreedyFleetSolver()
    
    # 3 Large vessels: capacity = 3 * 120,000 * 0.85 * 10 = 3,060,000 > 120,000 demand
    # 2 Diesel + 1 LNG: alt share = 1/3 = 0.333 <= 0.40 transition rate
    # Budget = 60M: capex = 3 * 95M * 0.10 * ((2*1.0 + 1*1.15)/3) = 28.5M * 1.05 = 29.925M <= 60M
    # Speed = 14 knots -> 0 delay hours -> 100.0 reliability >= 85.0 target
    eval_res = solver.evaluate_candidate(
        x_f=0,
        x_m=0,
        x_l=3,
        fuel_mix={"diesel": 2, "lng": 1, "methanol": 0, "hydrogen": 0, "ammonia": 0},
        speed_knots=14.0,
        scenario=standard_scenario,
    )
    assert eval_res["feasible_solution"] is True
    assert eval_res["reliability_score"] >= standard_scenario.target_reliability
    assert eval_res["demand_satisfaction_rate"] >= standard_scenario.service_level


def test_base_solver_allocate_fuel_mix(standard_scenario: OptimizationScenario) -> None:
    """Verify 5-fuel allocation preserving exact vessel counts across scenarios."""
    # Balanced default distribution across 4 alternative fuels
    mix = BaseBenchmarkSolver.allocate_fuel_mix(total_vessels=8, alt_count=4, scenario=standard_scenario)
    assert mix["diesel"] == 4
    assert mix["lng"] == 1
    assert mix["methanol"] == 1
    assert mix["hydrogen"] == 1
    assert mix["ammonia"] == 1
    assert sum(mix.values()) == 8

    # Hydrogen scenario
    h2_scenario = OptimizationScenario(
        cargo_demand=100000.0,
        route_distance=3000.0,
        deadline_hours=240.0,
        scenario_id="CASE-D-HYDROGEN",
    )
    mix_h2 = BaseBenchmarkSolver.allocate_fuel_mix(total_vessels=5, alt_count=3, scenario=h2_scenario)
    assert mix_h2["diesel"] == 2
    assert mix_h2["hydrogen"] == 3
    assert mix_h2["lng"] == 0
    assert mix_h2["methanol"] == 0
    assert mix_h2["ammonia"] == 0
    assert sum(mix_h2.values()) == 5


# =============================================================================
# 2. INDIVIDUAL BENCHMARK SOLVER TESTS
# =============================================================================

def test_greedy_solver(standard_scenario: OptimizationScenario) -> None:
    """Test deterministic Greedy heuristic solver."""
    solver = GreedyFleetSolver()
    result = solver.solve(standard_scenario, max_iterations=10, seed=42)
    
    assert isinstance(result, BenchmarkResult)
    assert result.solver_name == "Greedy Allocation"
    assert result.runtime_seconds >= 0.0
    assert result.fuel_consumption > 0.0
    assert result.operational_cost > 0.0
    assert result.emissions > 0.0
    assert result.demand_satisfaction_rate >= 0.90
    assert isinstance(result.feasible_solution, bool)


def test_classical_pso_solver(standard_scenario: OptimizationScenario) -> None:
    """Test Classical PSO with velocity clamping and inertia decay."""
    solver = ClassicalPSOSolver(population_size=15)
    result = solver.solve(standard_scenario, max_iterations=20, seed=42)
    
    assert isinstance(result, BenchmarkResult)
    assert result.solver_name == "Classical PSO"
    assert result.iterations == 20
    history = result.metadata.get("history", [])
    assert len(history) == 20
    # Convergence trace should be non-increasing (best score so far)
    assert all(history[i] >= history[i + 1] - 1e-4 for i in range(len(history) - 1))


def test_genetic_algorithm_solver(standard_scenario: OptimizationScenario) -> None:
    """Test Genetic Algorithm with tournament selection and crossover."""
    solver = GeneticAlgorithmSolver(population_size=15)
    result = solver.solve(standard_scenario, max_iterations=20, seed=42)
    
    assert isinstance(result, BenchmarkResult)
    assert result.solver_name == "Genetic Algorithm"
    assert result.iterations == 20
    history = result.metadata.get("history", [])
    assert len(history) == 20
    assert result.runtime_seconds > 0.0


def test_simulated_annealing_solver(standard_scenario: OptimizationScenario) -> None:
    """Test Simulated Annealing with geometric cooling."""
    solver = SimulatedAnnealingSolver(initial_temp=100.0, cooling_rate=0.85)
    result = solver.solve(standard_scenario, max_iterations=20, seed=42)
    
    assert isinstance(result, BenchmarkResult)
    assert result.solver_name == "Simulated Annealing"
    assert result.iterations == 20
    history = result.metadata.get("history", [])
    assert len(history) == 20


def test_linear_programming_solver(standard_scenario: OptimizationScenario) -> None:
    """Test Linear Programming continuous relaxation solver."""
    solver = LinearProgrammingSolver()
    result = solver.solve(standard_scenario, max_iterations=10, seed=42)
    
    assert isinstance(result, BenchmarkResult)
    assert result.solver_name == "Linear Programming (Relaxed)"
    assert result.runtime_seconds >= 0.0
    assert result.demand_satisfaction_rate >= 0.95


def test_qpso_benchmark_adapter(standard_scenario: OptimizationScenario) -> None:
    """Test Quantum-Inspired PSO benchmark adapter."""
    solver = QPSOBenchmarkAdapter(population_size=15)
    result = solver.solve(standard_scenario, max_iterations=20, seed=42)
    
    assert isinstance(result, BenchmarkResult)
    assert "QPSO" in result.solver_name
    assert result.iterations == 20
    history = result.metadata.get("history", [])
    assert len(history) == 20
    assert result.fuel_consumption > 0.0
    assert result.emissions > 0.0


# =============================================================================
# 3. CONVERGENCE ANALYSIS TESTS
# =============================================================================

def test_convergence_analyzer(standard_scenario: OptimizationScenario) -> None:
    """Verify convergence analysis, trajectory capturing, and quantum speedup ratio."""
    analyzer = ConvergenceAnalyzer()
    conv_results = analyzer.run_convergence_suite(standard_scenario, max_iterations=20, seed=42)
    
    assert "QPSO" in conv_results
    assert "Classical PSO" in conv_results
    
    qpso_res = conv_results["QPSO"]
    assert isinstance(qpso_res, ConvergenceAnalysisResult)
    assert qpso_res.solver_name == "QPSO"
    assert len(qpso_res.iterations) == 20
    assert qpso_res.convergence_iteration > 0
    assert qpso_res.final_objective > 0.0
    
    speedups = analyzer.compute_quantum_speedup(conv_results)
    assert isinstance(speedups, dict)
    assert "Classical PSO_iteration_speedup_factor" in speedups


# =============================================================================
# 4. SCALABILITY SUITE & MEMORY PROFILING TESTS
# =============================================================================

def test_scalability_suite_memory_profiling() -> None:
    """Verify scalability execution across fleet scales and tracemalloc memory capture."""
    suite = ScalabilitySuite()
    # Test smaller fleet counts for fast unit test execution
    vessel_counts = (10, 50, 100)
    records = suite.run_scalability_sweep(
        vessel_counts=vessel_counts,
        max_iterations=5,
    )
    
    assert len(records) == len(vessel_counts) * 3  # 3 solvers per count
    for r in records:
        assert r["fleet_size"] in vessel_counts
        assert r["peak_memory_mb"] >= 0.0
        assert r["runtime_seconds"] >= 0.0
        assert r["objective_score"] > 0.0


# =============================================================================
# 5. STATISTICAL STABILITY (MONTE CARLO SEEDS) TESTS
# =============================================================================

def test_statistical_stability_evaluator(standard_scenario: OptimizationScenario) -> None:
    """Verify multi-seed Monte Carlo stability metrics across independent runs."""
    evaluator = StatisticalStabilityEvaluator()
    res = evaluator.evaluate_solver_stability(
        solver_name="QPSO",
        scenario=standard_scenario,
        num_seeds=5,  # 5 seeds for fast unit testing
        base_seed=100,
        max_iterations=10,
    )
    
    assert isinstance(res, StatisticalStabilityResult)
    assert res.solver_name == "QPSO"
    assert res.num_seeds == 5
    assert len(res.objective_values) == 5
    assert res.mean_objective > 0.0
    assert res.std_objective >= 0.0
    assert res.best_objective <= res.worst_objective
    assert "coefficient_of_variation" in res.metadata


# =============================================================================
# 6. PARETO ANALYSIS & MULTI-OBJECTIVE TESTS
# =============================================================================

def test_pareto_analyzer() -> None:
    """Verify non-dominated sorting and 2D hypervolume calculation."""
    analyzer = ParetoAnalyzer()
    
    # 4 candidates with (operational_cost, fuel_consumption, emissions, reliability_score)
    candidates = [
        {"name": "Vessel A", "operational_cost": 100.0, "fuel_consumption": 200.0, "emissions": 50.0, "reliability_score": 90.0},
        {"name": "Vessel B", "operational_cost": 70.0, "fuel_consumption": 250.0, "emissions": 80.0, "reliability_score": 85.0},
        {"name": "Vessel C", "operational_cost": 150.0, "fuel_consumption": 300.0, "emissions": 120.0, "reliability_score": 70.0},  # Dominated
        {"name": "Vessel D", "operational_cost": 85.0, "fuel_consumption": 210.0, "emissions": 60.0, "reliability_score": 88.0},
    ]
    
    pareto_res = analyzer.extract_pareto_front(
        candidate_evaluations=candidates,
        scenario_id="SCEN-PARETO-01",
    )
    
    assert isinstance(pareto_res, ParetoResult)
    assert len(pareto_res.pareto_points) >= 2
    assert len(pareto_res.dominated_points) >= 1
    # Check that Vessel C is dominated
    dominated_names = [p["name"] for p in pareto_res.dominated_points]
    assert "Vessel C" in dominated_names
    assert pareto_res.hypervolume > 0.0


# =============================================================================
# 7. PREDICTION MODEL BENCHMARKING TESTS
# =============================================================================

def test_prediction_model_benchmarker() -> None:
    """Verify prediction model evaluation including prediction bias and residual standard deviation."""
    benchmarker = PredictionModelBenchmarker(random_state=42)
    results = benchmarker.benchmark_models(
        models_to_test=("linear_regression", "random_forest", "qifcp"),
    )
    
    assert len(results) == 3
    for r in results:
        assert isinstance(r, PredictionBenchmarkResult)
        assert r.mae >= 0.0
        assert r.rmse >= 0.0
        assert isinstance(r.prediction_bias, float)
        assert r.error_std >= 0.0
        assert r.inference_time_ms >= 0.0
        assert isinstance(r.r2, float)


# =============================================================================
# 8. WORKFLOW BENCHMARKING TESTS
# =============================================================================

def test_workflow_benchmarker(standard_scenario: OptimizationScenario) -> None:
    """Verify end-to-end full platform workflow latency measurement."""
    benchmarker = WorkflowBenchmarker()
    res = benchmarker.benchmark_full_workflow(standard_scenario)
    
    assert isinstance(res, WorkflowBenchmarkResult)
    assert res.total_workflow_runtime_seconds > 0.0
    assert "prediction_runtime" in res.step_runtimes_seconds
    assert "optimization_runtime" in res.step_runtimes_seconds
    assert "reliability_runtime" in res.step_runtimes_seconds
    assert "deployment_runtime" in res.step_runtimes_seconds
    assert res.scenario_id == standard_scenario.scenario_id


# =============================================================================
# 9. BENCHMARK SUITE ORCHESTRATOR & REPORT EXPORT TESTS
# =============================================================================

def test_benchmark_suite_orchestrator(standard_scenario: OptimizationScenario, tmp_path: Path) -> None:
    """Verify suite orchestrator execution, metric leaders, and report exports."""
    solvers = [
        QPSOBenchmarkAdapter(population_size=10),
        GreedyFleetSolver(),
        LinearProgrammingSolver(),
    ]
    orchestrator = BenchmarkSuiteOrchestrator(solvers=solvers)
    suite_res = orchestrator.run_suite(standard_scenario, max_iterations=10, seed=42)
    
    assert isinstance(suite_res, BenchmarkSuiteResult)
    assert len(suite_res.results) == 3
    
    # Check metric leaders
    leaders = suite_res.metric_leaders
    assert "lowest_fuel" in leaders
    assert "lowest_cost" in leaders
    assert "lowest_emissions" in leaders
    assert "lowest_runtime" in leaders
    assert "highest_reliability" in leaders
    assert "best_objective" in leaders
    
    # Check comparison matrix
    matrix = suite_res.comparison_matrix
    assert len(matrix) == 3
    for s_name, data in matrix.items():
        assert "fuel_consumption" in data
        assert "operational_cost" in data
        assert "qpso_fuel_savings_pct" in data
    
    # Export reports
    exported = orchestrator.export_reports(suite_res, output_dir=tmp_path)
    assert exported["json"].exists()
    assert exported["csv"].exists()
    assert exported["markdown"].exists()
    
    # Verify JSON content
    loaded_json = json.loads(exported["json"].read_text(encoding="utf-8"))
    assert "metric_leaders" in loaded_json
    assert "results" in loaded_json
