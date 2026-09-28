"""Automated regression tests for SIH26138 benchmark dashboard data wiring and isolation.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Verifies that:
1. Full Solver Matrix loads canonical integrated solver benchmark data.
2. Convergence dynamics loads iteration trajectory data across seeds.
3. Scalability loads strictly scalability benchmark records and NEVER raw voyage rows.
4. Prediction benchmark uses Physics + Residual QIFCP with frozen metrics.
5. Statistical stability loads multi-seed Monte Carlo distributions.
6. QPSO vs Classical PSO and ablation load canonical head-to-head and A-G variants.
7. Forbidden voyage columns (voyage_id, vessel_id, cargo_tons, etc.) are strictly rejected.
8. Missing artifacts trigger clear error states rather than silent substitution.
"""

from pathlib import Path
import json
import pytest
import pandas as pd

from src.benchmarking.benchmark_data_loader import (
    BENCHMARK_DATA_SOURCES,
    FORBIDDEN_VOYAGE_COLUMNS,
    BenchmarkArtifactError,
    load_convergence_data,
    load_prediction_accuracy_data,
    load_qpso_ablation_data,
    load_qpso_head_to_head_data,
    load_scalability_data,
    load_solver_matrix_data,
    load_statistical_stability_data,
    resolve_benchmark_artifact_path,
)


def test_data_source_registry_paths_exist() -> None:
    """Verify all canonical benchmark data sources are registered and resolve to valid files."""
    for key, path in BENCHMARK_DATA_SOURCES.items():
        resolved = resolve_benchmark_artifact_path(key)
        assert resolved.exists(), f"Benchmark data source '{key}' resolved to non-existent path: {resolved}"
        assert resolved.is_file(), f"Benchmark data source '{key}' is not a regular file: {resolved}"


def test_full_solver_matrix_loads_canonical_data() -> None:
    """Verify Full Solver Matrix loads all 6 algorithmic classes with canonical metrics."""
    solver_df, provenance = load_solver_matrix_data()

    assert isinstance(solver_df, pd.DataFrame)
    assert len(solver_df) >= 6

    solvers = list(solver_df["Solver"])
    expected_solvers = [
        "QPSO",
        "Classical PSO",
        "Genetic Algorithm",
        "Simulated Annealing",
        "Linear Programming (Relaxed)",
        "Greedy Allocation",
    ]
    for exp in expected_solvers:
        assert exp in solvers, f"Expected solver '{exp}' missing from solver matrix"

    # Verify required metric columns
    required_cols = [
        "Solver",
        "Category",
        "Mean Objective",
        "Feasible Rate (%)",
        "Mean Fuel (t)",
        "Mean Cost ($)",
        "Mean CO2e (t)",
        "Mean Runtime (ms)",
        "Counted Evaluations",
    ]
    for col in required_cols:
        assert col in solver_df.columns, f"Required column '{col}' missing from solver matrix"

    # Verify QPSO canonical values
    qpso_row = solver_df[solver_df["Solver"] == "QPSO"].iloc[0]
    assert qpso_row["Mean Objective"] == pytest.approx(8.8309, abs=0.001)
    assert qpso_row["Feasible Rate (%)"] == 100.0
    assert qpso_row["Counted Evaluations"] == 1000

    # Verify Classical PSO canonical values
    pso_row = solver_df[solver_df["Solver"] == "Classical PSO"].iloc[0]
    assert pso_row["Mean Objective"] == pytest.approx(10.6612, abs=0.001)
    assert pso_row["Feasible Rate (%)"] == 100.0
    assert pso_row["Counted Evaluations"] == 1000


def test_convergence_dynamics_loads_canonical_trajectories() -> None:
    """Verify Convergence Dynamics loads 50-iteration descent trajectories across 30 seeds."""
    conv_df, summary = load_convergence_data()

    assert isinstance(conv_df, pd.DataFrame)
    assert len(conv_df) == 50, "Convergence trajectory must contain exactly 50 iterations"
    assert "Iteration" in conv_df.columns
    assert "QPSO (Mean across 30 seeds)" in conv_df.columns
    assert "Classical PSO (Mean across 30 seeds)" in conv_df.columns

    # Verify monotonic or non-increasing overall progression
    assert conv_df["QPSO (Mean across 30 seeds)"].iloc[-1] <= conv_df["QPSO (Mean across 30 seeds)"].iloc[0]
    assert conv_df["Classical PSO (Mean across 30 seeds)"].iloc[-1] <= conv_df["Classical PSO (Mean across 30 seeds)"].iloc[0]

    # Verify canonical final convergence score
    assert conv_df["QPSO (Mean across 30 seeds)"].iloc[-1] == pytest.approx(8.8309, abs=0.001)

    # Verify summary fields
    assert summary["best_known_objective"] == pytest.approx(8.8309, abs=0.001)
    assert summary["qpso_iterations_to_2pct"] == pytest.approx(13.5, abs=0.1)
    assert summary["pso_iterations_to_2pct"] == pytest.approx(19.2, abs=0.1)
    assert summary["n_seeds"] == 30


def test_scalability_curves_loads_strictly_scalability_rows() -> None:
    """CRITICAL ACCEPTANCE TEST: Verify Scalability tab loads strictly scalability records.

    Asserts that the dataset contains fleet_size, solver_name, runtime_seconds, objective,
    and explicitly asserts that NO raw voyage telemetry columns exist.
    """
    scale_df, meta = load_scalability_data()

    assert isinstance(scale_df, pd.DataFrame)
    assert not scale_df.empty

    # 1. Assert required scalability dimensions
    assert "fleet_size" in scale_df.columns
    assert "solver_name" in scale_df.columns
    assert "runtime_seconds" in scale_df.columns
    assert "objective" in scale_df.columns
    assert "peak_memory_mb" in scale_df.columns
    assert "n_evaluations" in scale_df.columns

    # 2. Assert fleet size dimension values
    fleet_sizes = sorted(scale_df["fleet_size"].unique().tolist())
    for exp_size in [10, 50, 100, 250, 500]:
        assert exp_size in fleet_sizes, f"Fleet size {exp_size} missing from scalability dataset"

    # 3. Assert solver dimension values
    solvers = set(scale_df["solver_name"].unique())
    assert "QPSO" in solvers
    assert "Classical PSO" in solvers
    assert "Greedy" in solvers

    # 4. STRICT INVARIANT: Must NOT contain raw voyage telemetry columns
    for forbidden in FORBIDDEN_VOYAGE_COLUMNS:
        assert forbidden not in scale_df.columns, (
            f"CORRUPTION: Raw voyage telemetry column '{forbidden}' found in scalability dataframe!"
        )


def test_prediction_accuracy_loads_canonical_qifcp() -> None:
    """Verify Prediction Accuracy uses Physics + Residual QIFCP as frozen production model."""
    pred_df, meta = load_prediction_accuracy_data()

    assert isinstance(pred_df, pd.DataFrame)
    assert len(pred_df) >= 5

    # Check production model designation
    prod_rows = pred_df[pred_df["Status"] == "PRODUCTION / FROZEN"]
    assert len(prod_rows) == 1
    prod_row = prod_rows.iloc[0]
    assert prod_row["Model Architecture"] == "Physics + Residual QIFCP"
    assert prod_row["R² Score"] == pytest.approx(0.9933, abs=0.001)
    assert prod_row["RMSE (tons)"] == pytest.approx(61.55, abs=0.1)
    assert prod_row["MAE (tons)"] == pytest.approx(40.07, abs=0.1)

    # Check reference models are present
    model_names = list(pred_df["Model Architecture"])
    for ref in ["Naval Physics Baseline", "Random Forest", "HistGradientBoosting", "Grouped QIFCP Direct"]:
        assert ref in model_names


def test_statistical_stability_loads_30_seed_distributions() -> None:
    """Verify Statistical Stability loads Monte Carlo statistics and 30-seed distributions."""
    stab_df, dists, meta = load_statistical_stability_data()

    assert isinstance(stab_df, pd.DataFrame)
    assert len(stab_df) >= 4

    algorithms = list(stab_df["Algorithm"])
    for algo in ["QPSO", "Classical PSO", "Genetic Algorithm", "Simulated Annealing"]:
        assert algo in algorithms
        assert algo in dists
        assert len(dists[algo]) == 30, f"Algorithm '{algo}' must have exactly 30 seed-level samples"

    # Verify QPSO stability characteristics (near-zero CV)
    qpso_row = stab_df[stab_df["Algorithm"] == "QPSO"].iloc[0]
    assert qpso_row["CV (%)"] < 1.0, "QPSO coefficient of variation must be near-zero"


def test_qpso_head_to_head_and_ablation_loaders() -> None:
    """Verify Head-to-Head summary and A-G ablation study loaders."""
    h2h_sum, h2h_seeds, h2h_meta = load_qpso_head_to_head_data()

    # Head-to-Head 30 seeds
    assert h2h_meta["n_seeds"] == 30
    assert h2h_meta["qpso_wins"] == 6
    assert h2h_meta["pso_wins"] == 0
    assert h2h_meta["ties"] == 24
    assert len(h2h_seeds) == 30

    # Ablation study Variants A through G
    abl_df, abl_meta = load_qpso_ablation_data()
    assert isinstance(abl_df, pd.DataFrame)
    assert len(abl_df) == 7

    variants = list(abl_df["Variant"])
    for expected_var in ["A", "B", "C", "D", "E", "F", "G"]:
        assert expected_var in variants


def test_missing_artifact_raises_clear_error() -> None:
    """Verify that attempting to load a non-existent artifact raises FileNotFoundError with diagnostic message."""
    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_benchmark_artifact_path("non_existent_fake_benchmark_artifact.json")
    assert "Canonical benchmark artifact not found" in str(exc_info.value)
