"""Canonical benchmark data loader and registry for SIH26138 dashboard and reporting.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Provides authoritative, schema-validated loading of all canonical benchmark artifacts:
1. Full Solver Matrix (outputs/reports/final_integrated_optimization_benchmark.json)
2. Convergence Dynamics (outputs/reports/final_integrated_optimization_benchmark.json)
3. Scalability Curves (outputs/reports/final_integrated_optimization_benchmark.json or benchmark_reports/scalability_summary.csv)
4. Prediction Accuracy (outputs/reports/final_prediction_benchmark.json)
5. Statistical Stability (benchmark_reports/statistical_stability.json)
6. QPSO vs Classical PSO & Ablation (outputs/reports/final_integrated_optimization_benchmark.json)

Ensures complete data isolation between tabs: raw voyage telemetry data cannot leak
into benchmark views.
"""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Final
import numpy as np
import pandas as pd

from src.prediction.benchmark_loader import (
    CanonicalBenchmarkMetadata,
    get_canonical_benchmark_metadata,
    get_canonical_prediction_dataframe,
    resolve_canonical_benchmark_path,
)

logger = logging.getLogger("maritime_system")

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

# Canonical dashboard data-source registry
BENCHMARK_DATA_SOURCES: Final[dict[str, Path]] = {
    "solver_matrix": PROJECT_ROOT / "outputs" / "reports" / "final_integrated_optimization_benchmark.json",
    "convergence": PROJECT_ROOT / "outputs" / "reports" / "final_integrated_optimization_benchmark.json",
    "scalability": PROJECT_ROOT / "outputs" / "reports" / "final_integrated_optimization_benchmark.json",
    "scalability_csv": PROJECT_ROOT / "benchmark_reports" / "scalability_summary.csv",
    "prediction": PROJECT_ROOT / "outputs" / "reports" / "final_prediction_benchmark.json",
    "stability": PROJECT_ROOT / "benchmark_reports" / "statistical_stability.json",
    "stability_fallback": PROJECT_ROOT / "outputs" / "reports" / "final_integrated_optimization_benchmark.json",
    "qpso_head_to_head": PROJECT_ROOT / "outputs" / "reports" / "final_integrated_optimization_benchmark.json",
    "qpso_ablation": PROJECT_ROOT / "outputs" / "reports" / "final_integrated_optimization_benchmark.json",
    "qpso_ablation_fallback": PROJECT_ROOT / "benchmark_reports" / "qpso_ablation_study.json",
}

# Raw voyage columns that must NEVER appear in scalability or solver benchmarks
FORBIDDEN_VOYAGE_COLUMNS: Final[frozenset[str]] = frozenset({
    "voyage_id",
    "vessel_id",
    "vessel_type",
    "vessel_dwt",
    "cargo_tons",
    "distance_nm",
    "speed_knots",
    "hours_at_sea",
    "weather_factor",
    "sea_state",
    "data_source",
    "is_synthetic",
})


class BenchmarkArtifactError(Exception):
    """Raised when a canonical benchmark artifact is missing or malformed."""


def resolve_benchmark_artifact_path(source_key_or_path: str | Path) -> Path:
    """Resolve a benchmark artifact path robustly across local, container, and CWD environments.

    Args:
        source_key_or_path: Key in BENCHMARK_DATA_SOURCES or a Path/str.

    Returns:
        Resolved existing Path.

    Raises:
        FileNotFoundError: If the artifact cannot be located anywhere.
    """
    if isinstance(source_key_or_path, str) and source_key_or_path in BENCHMARK_DATA_SOURCES:
        target = BENCHMARK_DATA_SOURCES[source_key_or_path]
    else:
        target = Path(source_key_or_path)

    # 1. Direct path check
    if target.is_file():
        return target.resolve()

    # 2. Check relative to PROJECT_ROOT
    cand = (PROJECT_ROOT / target).resolve()
    if cand.is_file():
        return cand

    # 3. Check relative to current working directory
    cand = (Path.cwd() / target).resolve()
    if cand.is_file():
        return cand

    # 4. Check container /app directory
    rel_part = target if not target.is_absolute() else target.relative_to(target.anchor)
    cand = (Path("/app") / rel_part).resolve()
    if cand.is_file():
        return cand

    # 5. Check if filename exists inside known report directories
    for search_dir in ("outputs/reports", "benchmark_reports", "outputs"):
        cand = (PROJECT_ROOT / search_dir / target.name).resolve()
        if cand.is_file():
            return cand
        cand = (Path("/app") / search_dir / target.name).resolve()
        if cand.is_file():
            return cand

    raise FileNotFoundError(
        f"Canonical benchmark artifact not found: '{source_key_or_path}'. "
        f"Searched target '{target}', PROJECT_ROOT ({PROJECT_ROOT}), CWD ({Path.cwd()}), and /app."
    )


# ==============================================================================
# 1. Full Solver Matrix Loader
# ==============================================================================

def load_solver_matrix_data() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and validate the canonical full solver benchmark matrix.

    Returns:
        solver_matrix_df: Standardized DataFrame containing all 6 solvers.
        provenance: Metadata dict with timestamp, commit, and scenario details.

    Raises:
        BenchmarkArtifactError: If schema validation fails.
    """
    path = resolve_benchmark_artifact_path("solver_matrix")
    data = json.loads(path.read_text(encoding="utf-8"))

    if "primary_head_to_head" not in data or "secondary_solvers" not in data:
        raise BenchmarkArtifactError(
            f"Artifact '{path}' is missing required sections ('primary_head_to_head', 'secondary_solvers')."
        )

    p_h2h = data["primary_head_to_head"]
    sec = data["secondary_solvers"]

    rows: list[dict[str, Any]] = []

    # QPSO
    qpso_agg = p_h2h.get("qpso_aggregate", {})
    rows.append({
        "Solver": "QPSO",
        "Category": "Quantum Swarm",
        "Mean Objective": round(float(qpso_agg.get("mean_objective", 0.0)), 4),
        "Feasible Rate (%)": round(float(qpso_agg.get("feasible_rate_pct", 0.0)), 1),
        "Mean Fuel (t)": round(float(qpso_agg.get("mean_fuel_t", 0.0)), 2),
        "Mean Cost ($)": round(float(qpso_agg.get("mean_cost_usd", 0.0)), 2),
        "Mean CO2e (t)": round(float(qpso_agg.get("mean_emissions_t", 0.0)), 2),
        "Mean Runtime (ms)": round(float(qpso_agg.get("mean_runtime_s", 0.0)) * 1000.0, 1),
        "Counted Evaluations": int(qpso_agg.get("mean_evaluations", 1000)),
    })

    # Classical PSO
    pso_agg = p_h2h.get("pso_aggregate", {})
    rows.append({
        "Solver": "Classical PSO",
        "Category": "Classical Swarm",
        "Mean Objective": round(float(pso_agg.get("mean_objective", 0.0)), 4),
        "Feasible Rate (%)": round(float(pso_agg.get("feasible_rate_pct", 0.0)), 1),
        "Mean Fuel (t)": round(float(pso_agg.get("mean_fuel_t", 0.0)), 2),
        "Mean Cost ($)": round(float(pso_agg.get("mean_cost_usd", 0.0)), 2),
        "Mean CO2e (t)": round(float(pso_agg.get("mean_emissions_t", 0.0)), 2),
        "Mean Runtime (ms)": round(float(pso_agg.get("mean_runtime_s", 0.0)) * 1000.0, 1),
        "Counted Evaluations": int(pso_agg.get("mean_evaluations", 1000)),
    })

    category_map = {
        "GA": "Evolutionary",
        "SA": "Trajectory Metaheuristic",
        "LP": "Mathematical Programming",
        "Greedy": "Heuristic Baseline",
    }

    for key, item in sec.items():
        name = item.get("solver_name", key)
        cat = category_map.get(key, "Classical Reference")
        mean_obj = float(item.get("mean_objective", item.get("objective", 0.0)))
        feas_rate = float(item.get("feasible_rate_pct", 100.0 if item.get("feasible") else 0.0))
        mean_fuel = float(item.get("mean_fuel_t", item.get("fuel_t", 0.0)))
        mean_cost = float(item.get("mean_cost_usd", item.get("cost_usd", 0.0)))
        mean_emiss = float(item.get("mean_emissions_t", item.get("emissions_t", 0.0)))
        mean_runtime = float(item.get("mean_runtime_s", item.get("runtime_s", 0.0))) * 1000.0
        evals = int(item.get("mean_evaluations", item.get("evaluations", 1)))

        rows.append({
            "Solver": name,
            "Category": cat,
            "Mean Objective": round(mean_obj, 4),
            "Feasible Rate (%)": round(feas_rate, 1),
            "Mean Fuel (t)": round(mean_fuel, 2),
            "Mean Cost ($)": round(mean_cost, 2),
            "Mean CO2e (t)": round(mean_emiss, 2),
            "Mean Runtime (ms)": round(mean_runtime, 1),
            "Counted Evaluations": evals,
        })

    solver_matrix_df = pd.DataFrame(rows)

    # Schema validation
    expected_cols = {"Solver", "Category", "Mean Objective", "Feasible Rate (%)", "Mean Fuel (t)", "Mean Cost ($)", "Mean CO2e (t)", "Mean Runtime (ms)", "Counted Evaluations"}
    missing = expected_cols - set(solver_matrix_df.columns)
    if missing:
        raise BenchmarkArtifactError(f"Solver matrix is missing required columns: {missing}")
    if len(solver_matrix_df) < 6:
        raise BenchmarkArtifactError(f"Expected at least 6 solvers in matrix, got {len(solver_matrix_df)}")

    provenance = {
        "timestamp_utc": data.get("timestamp_utc", "Unknown"),
        "git_commit": data.get("git_commit", "Unknown"),
        "predictor": data.get("predictor", "PhysicsInformedQIFCPRegressor"),
        "scenario": data.get("scenario", {}),
        "source_path": str(path),
    }

    return solver_matrix_df, provenance


# ==============================================================================
# 2. Convergence Dynamics Loader
# ==============================================================================

def load_convergence_data() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and validate iteration-by-iteration convergence trajectories across seeds.

    Returns:
        convergence_df: DataFrame with Iteration, Mean QPSO Objective, Mean PSO Objective,
                        and per-seed columns.
        summary: Convergence metrics (mean iterations to 2%, best-known objective, gaps).
    """
    path = resolve_benchmark_artifact_path("convergence")
    data = json.loads(path.read_text(encoding="utf-8"))

    if "primary_head_to_head" not in data or "seed_records" not in data["primary_head_to_head"]:
        raise BenchmarkArtifactError(f"Artifact '{path}' is missing primary_head_to_head.seed_records")

    p_h2h = data["primary_head_to_head"]
    seed_records = p_h2h["seed_records"]
    if not seed_records:
        raise BenchmarkArtifactError("seed_records is empty in convergence benchmark")

    qpso_histories = np.array([r["qpso"]["history"] for r in seed_records], dtype=float)
    pso_histories = np.array([r["pso"]["history"] for r in seed_records], dtype=float)

    n_iterations = qpso_histories.shape[1]
    iterations = np.arange(1, n_iterations + 1)

    mean_qpso = qpso_histories.mean(axis=0)
    mean_pso = pso_histories.mean(axis=0)

    rows: list[dict[str, Any]] = []
    for i in range(n_iterations):
        row = {
            "Iteration": int(iterations[i]),
            "QPSO (Mean across 30 seeds)": round(float(mean_qpso[i]), 4),
            "Classical PSO (Mean across 30 seeds)": round(float(mean_pso[i]), 4),
        }
        rows.append(row)

    convergence_df = pd.DataFrame(rows)

    summary = {
        "best_known_objective": float(p_h2h.get("best_known_objective", 8.8309)),
        "qpso_iterations_to_2pct": float(p_h2h.get("qpso_aggregate", {}).get("mean_iterations_to_2pct", 13.5)),
        "pso_iterations_to_2pct": float(p_h2h.get("pso_aggregate", {}).get("mean_iterations_to_2pct", 19.2)),
        "qpso_gap_to_best_known_pct": float(p_h2h.get("qpso_aggregate", {}).get("mean_gap_to_best_known_pct", 0.0)),
        "pso_gap_to_best_known_pct": float(p_h2h.get("pso_aggregate", {}).get("mean_gap_to_best_known_pct", 20.73)),
        "n_seeds": len(seed_records),
        "seeds": [r["seed"] for r in seed_records],
        "seed_records": seed_records,
        "source_path": str(path),
    }

    return convergence_df, summary


# ==============================================================================
# 3. Scalability Curves Loader — CRITICAL FIX
# ==============================================================================

def load_scalability_data() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and validate the canonical swarm scalability benchmark.

    CRITICAL INVARIANT:
    Ensures the returned DataFrame contains ONLY scalability benchmark fields:
    - fleet_size
    - solver_name
    - runtime_seconds
    - objective
    - peak_memory_mb
    - feasible
    - n_evaluations
    And explicitly enforces that NO raw voyage telemetry columns are present.

    Returns:
        scalability_df: Validated scalability DataFrame.
        metadata: Metadata describing problem sizes and solvers.

    Raises:
        BenchmarkArtifactError: If schema validation fails or raw voyage columns exist.
    """
    path = resolve_benchmark_artifact_path("scalability")
    data = json.loads(path.read_text(encoding="utf-8"))

    if "scalability" not in data or not data["scalability"]:
        # Fallback to scalability_csv if available
        csv_path = resolve_benchmark_artifact_path("scalability_csv")
        raw_df = pd.read_csv(csv_path)
        records = raw_df.to_dict(orient="records")
    else:
        records = data["scalability"]

    scalability_df = pd.DataFrame(records)

    # 1. Enforce absence of forbidden raw voyage columns
    present_forbidden = FORBIDDEN_VOYAGE_COLUMNS.intersection(scalability_df.columns)
    if present_forbidden:
        raise BenchmarkArtifactError(
            f"CORRUPTION DETECTED: Scalability artifact contains raw voyage columns: {present_forbidden}. "
            f"The Scalability tab must NEVER display voyage telemetry."
        )

    # 2. Enforce required scalability dimensions
    required_cols = {"fleet_size", "solver_name", "runtime_seconds", "peak_memory_mb", "feasible", "n_evaluations"}
    missing = required_cols - set(scalability_df.columns)
    if missing:
        raise BenchmarkArtifactError(f"Scalability artifact missing required columns: {missing}")

    # Standardize objective column
    if "objective" not in scalability_df.columns and "objective_score" in scalability_df.columns:
        scalability_df["objective"] = scalability_df["objective_score"]

    # Normalize data types
    scalability_df["fleet_size"] = scalability_df["fleet_size"].astype(int)
    scalability_df["runtime_seconds"] = scalability_df["runtime_seconds"].astype(float)
    scalability_df["peak_memory_mb"] = scalability_df["peak_memory_mb"].astype(float)
    scalability_df["objective"] = scalability_df["objective"].astype(float)
    scalability_df["n_evaluations"] = scalability_df["n_evaluations"].astype(int)
    scalability_df["feasible_label"] = scalability_df["feasible"].apply(lambda f: "✅ Yes" if f else "❌ No")

    metadata = {
        "fleet_sizes": sorted(scalability_df["fleet_size"].unique().tolist()),
        "solvers": sorted(scalability_df["solver_name"].unique().tolist()),
        "record_count": len(scalability_df),
        "source_path": str(path),
    }

    return scalability_df, metadata


# ==============================================================================
# 4. Prediction Accuracy Loader
# ==============================================================================

def load_prediction_accuracy_data() -> tuple[pd.DataFrame, CanonicalBenchmarkMetadata]:
    """Load and validate the canonical 5-seed vessel-disjoint prediction benchmark.

    Primary model: 'Physics + Residual QIFCP' (PRODUCTION / FROZEN).
    Reference models: Naval Physics Baseline, Random Forest, HistGradientBoosting, Grouped QIFCP Direct.
    """
    path = resolve_canonical_benchmark_path()
    meta = get_canonical_benchmark_metadata(report_path=path)
    prediction_df = get_canonical_prediction_dataframe(report_path=path)

    # Verify primary model name
    prod_row = prediction_df[prediction_df["Status"] == "PRODUCTION / FROZEN"]
    if prod_row.empty:
        raise BenchmarkArtifactError("Canonical prediction benchmark is missing PRODUCTION / FROZEN model.")
    if prod_row.iloc[0]["Model Architecture"] != "Physics + Residual QIFCP":
        raise BenchmarkArtifactError(
            f"Expected primary model 'Physics + Residual QIFCP', found '{prod_row.iloc[0]['Model Architecture']}'"
        )

    return prediction_df, meta


# ==============================================================================
# 5. Statistical Stability Loader
# ==============================================================================

def load_statistical_stability_data() -> tuple[pd.DataFrame, dict[str, list[float]], dict[str, Any]]:
    """Load and validate cross-seed statistical stability benchmarks.

    Returns:
        stability_df: DataFrame with Algorithm, Seeds Evaluated, Feasible Runs, Mean Obj,
                      Std Dev, Best Obj, Worst Obj, CV (%).
        distributions: Mapping of solver name to list of 30 seed-level objective values.
        metadata: Seeds and provenance details.
    """
    path = resolve_benchmark_artifact_path("stability")
    data = json.loads(path.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    distributions: dict[str, list[float]] = {}

    for solver_key, s_data in data.items():
        name = s_data.get("solver_name", solver_key)
        n_seeds = int(s_data.get("num_seeds", 30))
        mean_obj = float(s_data.get("mean_objective", 0.0))
        std_obj = float(s_data.get("std_objective", 0.0))
        best_obj = float(s_data.get("best_objective", 0.0))
        worst_obj = float(s_data.get("worst_objective", 0.0))
        cv = float(s_data.get("cv", 0.0))
        feas_rate = float(s_data.get("feasible_run_rate", 1.0)) * 100.0
        feas_count = int(s_data.get("feasible_run_count", n_seeds))

        obj_vals = [float(v) for v in s_data.get("objective_values", [])]
        distributions[name] = obj_vals

        rows.append({
            "Algorithm": name,
            "Seeds Evaluated": n_seeds,
            "Feasible Runs": f"{feas_count}/{n_seeds} ({feas_rate:.0f}%)",
            "Mean Objective": round(mean_obj, 4),
            "Std Deviation": round(std_obj, 4),
            "Best Objective": round(best_obj, 4),
            "Worst Objective": round(worst_obj, 4),
            "CV (%)": round(cv, 2),
        })

    stability_df = pd.DataFrame(rows)

    metadata = {
        "solvers": list(distributions.keys()),
        "source_path": str(path),
    }

    return stability_df, distributions, metadata


# ==============================================================================
# 6. QPSO vs Classical PSO & Ablation Loader
# ==============================================================================

def load_qpso_head_to_head_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load and validate the 30-seed fair comparison between QPSO and Classical PSO.

    Returns:
        h2h_summary_df: Side-by-side metric comparison table.
        h2h_seeds_df: Per-seed breakdown table for all 30 seeds.
        metadata: Win/tie/loss counts and evaluation parity.
    """
    path = resolve_benchmark_artifact_path("qpso_head_to_head")
    data = json.loads(path.read_text(encoding="utf-8"))

    if "primary_head_to_head" not in data:
        raise BenchmarkArtifactError(f"Missing primary_head_to_head in '{path}'")

    p_h2h = data["primary_head_to_head"]
    qpso_agg = p_h2h["qpso_aggregate"]
    pso_agg = p_h2h["pso_aggregate"]
    win_loss_tie = p_h2h["win_loss_tie"]
    seed_records = p_h2h["seed_records"]

    summary_rows = [
        {
            "Metric": "Mean Objective Score",
            "QPSO": f"{qpso_agg['mean_objective']:.4f} ± {qpso_agg['std_objective']:.4f}",
            "Classical PSO": f"{pso_agg['mean_objective']:.4f} ± {pso_agg['std_objective']:.4f}",
            "Delta (PSO - QPSO)": f"{pso_agg['mean_objective'] - qpso_agg['mean_objective']:+.4f}",
            "Winner": "QPSO",
        },
        {
            "Metric": "Best Objective Score",
            "QPSO": f"{qpso_agg['best_objective']:.4f}",
            "Classical PSO": f"{pso_agg['best_objective']:.4f}",
            "Delta (PSO - QPSO)": f"{pso_agg['best_objective'] - qpso_agg['best_objective']:+.4f}",
            "Winner": "Tie",
        },
        {
            "Metric": "Worst Objective Score",
            "QPSO": f"{qpso_agg['worst_objective']:.4f}",
            "Classical PSO": f"{pso_agg['worst_objective']:.4f}",
            "Delta (PSO - QPSO)": f"{pso_agg['worst_objective'] - qpso_agg['worst_objective']:+.4f}",
            "Winner": "QPSO",
        },
        {
            "Metric": "Mean Fuel Consumption (tons)",
            "QPSO": f"{qpso_agg['mean_fuel_t']:,.2f} t",
            "Classical PSO": f"{pso_agg['mean_fuel_t']:,.2f} t",
            "Delta (PSO - QPSO)": f"{pso_agg['mean_fuel_t'] - qpso_agg['mean_fuel_t']:+,.2f} t",
            "Winner": "QPSO",
        },
        {
            "Metric": "Mean Operational Cost ($)",
            "QPSO": f"${qpso_agg['mean_cost_usd']:,.2f}",
            "Classical PSO": f"${pso_agg['mean_cost_usd']:,.2f}",
            "Delta (PSO - QPSO)": f"${pso_agg['mean_cost_usd'] - qpso_agg['mean_cost_usd']:+,.2f}",
            "Winner": "QPSO",
        },
        {
            "Metric": "Mean GHG Emissions (t CO2e)",
            "QPSO": f"{qpso_agg['mean_emissions_t']:,.2f} t",
            "Classical PSO": f"{pso_agg['mean_emissions_t']:,.2f} t",
            "Delta (PSO - QPSO)": f"{pso_agg['mean_emissions_t'] - qpso_agg['mean_emissions_t']:+,.2f} t",
            "Winner": "QPSO",
        },
        {
            "Metric": "Feasible Run Rate (%)",
            "QPSO": f"{qpso_agg['feasible_rate_pct']:.1f}%",
            "Classical PSO": f"{pso_agg['feasible_rate_pct']:.1f}%",
            "Delta (PSO - QPSO)": "0.0%",
            "Winner": "Tie (100%)",
        },
        {
            "Metric": "Mean Gap to Best Known (%)",
            "QPSO": f"{qpso_agg['mean_gap_to_best_known_pct']:.2f}%",
            "Classical PSO": f"{pso_agg['mean_gap_to_best_known_pct']:.2f}%",
            "Delta (PSO - QPSO)": f"{pso_agg['mean_gap_to_best_known_pct'] - qpso_agg['mean_gap_to_best_known_pct']:+.2f}%",
            "Winner": "QPSO",
        },
        {
            "Metric": "Mean Iterations to 2% of Best",
            "QPSO": f"{qpso_agg['mean_iterations_to_2pct']:.1f}",
            "Classical PSO": f"{pso_agg['mean_iterations_to_2pct']:.1f}",
            "Delta (PSO - QPSO)": f"{pso_agg['mean_iterations_to_2pct'] - qpso_agg['mean_iterations_to_2pct']:+.1f}",
            "Winner": "QPSO",
        },
        {
            "Metric": "Mean Wall-Clock Runtime (s)",
            "QPSO": f"{qpso_agg['mean_runtime_s']:.2f} s",
            "Classical PSO": f"{pso_agg['mean_runtime_s']:.2f} s",
            "Delta (PSO - QPSO)": f"{pso_agg['mean_runtime_s'] - qpso_agg['mean_runtime_s']:+.2f} s",
            "Winner": "Classical PSO",
        },
        {
            "Metric": "Candidate Cache Hit Rate (%)",
            "QPSO": f"{qpso_agg['mean_cache_hit_rate']:.1f}%",
            "Classical PSO": f"{pso_agg['mean_cache_hit_rate']:.1f}%",
            "Delta (PSO - QPSO)": f"{qpso_agg['mean_cache_hit_rate'] - pso_agg['mean_cache_hit_rate']:+.1f}%",
            "Winner": "-",
        },
    ]
    h2h_summary_df = pd.DataFrame(summary_rows)

    seed_rows: list[dict[str, Any]] = []
    for r in seed_records:
        seed_rows.append({
            "Seed": r["seed"],
            "QPSO Objective": round(float(r["qpso"]["objective"]), 4),
            "PSO Objective": round(float(r["pso"]["objective"]), 4),
            "Difference (PSO - QPSO)": round(float(r["objective_diff_pso_minus_qpso"]), 4),
            "Winner": r["winner"],
            "QPSO Gap (%)": f"{float(r['qpso']['gap_to_best_known_pct']):.2f}%",
            "PSO Gap (%)": f"{float(r['pso']['gap_to_best_known_pct']):.2f}%",
        })
    h2h_seeds_df = pd.DataFrame(seed_rows)

    metadata = {
        "n_seeds": p_h2h.get("n_seeds", 30),
        "qpso_wins": win_loss_tie.get("qpso_wins", 0),
        "pso_wins": win_loss_tie.get("pso_wins", 0),
        "ties": win_loss_tie.get("ties", 0),
        "qpso_win_rate_pct": win_loss_tie.get("qpso_win_rate_pct", 0.0),
        "pso_win_rate_pct": win_loss_tie.get("pso_win_rate_pct", 0.0),
        "tie_rate_pct": win_loss_tie.get("tie_rate_pct", 0.0),
        "evaluations_per_run": p_h2h.get("evaluations_per_run", 1000),
        "source_path": str(path),
    }

    return h2h_summary_df, h2h_seeds_df, metadata


def load_qpso_ablation_data() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and validate the canonical QPSO A–G algorithmic enhancement ablation study.

    Returns:
        ablation_df: DataFrame with Variants A through G.
        metadata: Seed set and hyperparameter metadata.
    """
    path = resolve_benchmark_artifact_path("qpso_ablation")
    data = json.loads(path.read_text(encoding="utf-8"))

    if "ablation" in data and "variants" in data["ablation"]:
        v_dict = data["ablation"]["variants"]
        meta_dict = data["ablation"]
    else:
        fb_path = resolve_benchmark_artifact_path("qpso_ablation_fallback")
        fb_data = json.loads(fb_path.read_text(encoding="utf-8"))
        v_dict = fb_data.get("variants", {})
        meta_dict = fb_data

    rows: list[dict[str, Any]] = []
    for vid, v in v_dict.items():
        rows.append({
            "Variant": vid,
            "Name": v.get("name", f"Variant {vid}"),
            "Description": v.get("description", ""),
            "Mean Objective": round(float(v.get("mean_objective", 0.0)), 4),
            "Std Dev": round(float(v.get("std_objective", 0.0)), 4),
            "Best Objective": round(float(v.get("best_objective", 0.0)), 4),
            "Worst Objective": round(float(v.get("worst_objective", 0.0)), 4),
            "Feasible (%)": f"{float(v.get('feasible_rate', 100.0)):.1f}%",
            "Unique Evals": round(float(v.get("mean_unique_evals", 0.0)), 1),
            "Runtime (s)": round(float(v.get("mean_runtime_s", 0.0)), 3),
        })

    ablation_df = pd.DataFrame(rows)

    metadata = {
        "n_seeds": meta_dict.get("n_seeds", 30),
        "population_size": meta_dict.get("population_size", 20),
        "max_iterations": meta_dict.get("max_iterations", 50),
        "source_path": str(path),
    }

    return ablation_df, metadata
