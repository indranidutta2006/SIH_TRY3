"""Canonical prediction benchmark loader for dashboard and reporting.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Authoritatively loads and transforms canonical 5-seed vessel-disjoint benchmark metrics
from outputs/reports/final_prediction_benchmark.json without hard-coded numbers.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import pandas as pd


PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CANONICAL_BENCHMARK_PATH: Path = (
    PROJECT_ROOT / "outputs" / "reports" / "final_prediction_benchmark.json"
)

# Authoritative model mapping and display order for the final benchmark
CANONICAL_BENCHMARK_MODELS = (
    ("physics_residual_qifcp", "Physics + Residual QIFCP", "PRODUCTION / FROZEN"),
    ("physics_baseline", "Naval Physics Baseline", "REFERENCE"),
    ("random_forest", "Random Forest", "REFERENCE"),
    ("hist_gradient_boosting", "HistGradientBoosting", "REFERENCE"),
    ("adaptive_grouped_gamma", "Grouped QIFCP Direct", "ABLATION"),
)


@dataclass(frozen=True)
class CanonicalBenchmarkMetadata:
    benchmark_name: str
    benchmark_version: str
    dataset_path: str
    seeds: list[int]
    split_protocol: str
    target_mode: str
    aggregation_method: str
    subtitle: str
    accuracy_claim: str


def resolve_canonical_benchmark_path(
    report_path: Path | str | None = None,
) -> Path:
    """Resolve the canonical prediction benchmark path robustly across environments.

    Searches in:
    1. Direct file path if given and existing
    2. Given path relative to PROJECT_ROOT
    3. Given path relative to current working directory
    4. Canonical DEFAULT_CANONICAL_BENCHMARK_PATH (PROJECT_ROOT / outputs/reports/...)
    5. Standard container path (/app/outputs/reports/final_prediction_benchmark.json)
    6. CWD-relative outputs/reports/final_prediction_benchmark.json

    Returns:
        Resolved existing Path if found, otherwise canonical target Path.
    """
    if report_path is not None:
        p = Path(report_path)
        if p.is_file():
            return p.resolve()
        candidate = (PROJECT_ROOT / p).resolve()
        if candidate.is_file():
            return candidate
        candidate = (Path.cwd() / p).resolve()
        if candidate.is_file():
            return candidate

    # 4. Check canonical PROJECT_ROOT path
    if DEFAULT_CANONICAL_BENCHMARK_PATH.is_file():
        return DEFAULT_CANONICAL_BENCHMARK_PATH.resolve()

    # 5. Check container /app directory
    container_cand = Path("/app/outputs/reports/final_prediction_benchmark.json")
    if container_cand.is_file():
        return container_cand.resolve()

    # 6. Check CWD relative path
    cwd_cand = (Path.cwd() / "outputs" / "reports" / "final_prediction_benchmark.json").resolve()
    if cwd_cand.is_file():
        return cwd_cand

    # Fallback to absolute canonical path
    return DEFAULT_CANONICAL_BENCHMARK_PATH.resolve()


def load_canonical_prediction_benchmark(
    report_path: Path | str = DEFAULT_CANONICAL_BENCHMARK_PATH,
) -> dict[str, Any]:
    """Load the raw canonical prediction benchmark JSON artifact.

    Args:
        report_path: Path to outputs/reports/final_prediction_benchmark.json.

    Returns:
        Parsed JSON dictionary.

    Raises:
        FileNotFoundError: If the report file does not exist.
    """
    resolved = resolve_canonical_benchmark_path(report_path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Canonical prediction benchmark report missing at: {resolved}. "
            f"Searched relative to PROJECT_ROOT ({PROJECT_ROOT}) and CWD ({Path.cwd()})."
        )
    return json.loads(resolved.read_text(encoding="utf-8"))


def get_canonical_benchmark_metadata(
    data: dict[str, Any] | None = None,
    report_path: Path | str = DEFAULT_CANONICAL_BENCHMARK_PATH,
) -> CanonicalBenchmarkMetadata:
    """Extract metadata and context labels from canonical benchmark data."""
    if data is None:
        data = load_canonical_prediction_benchmark(report_path)

    raw_meta = data.get("metadata", {})
    canon_ds = raw_meta.get("canonical_dataset", {})
    ds_path = canon_ds.get("path", "data/raw/voyages_sample.csv").replace("\\", "/")

    return CanonicalBenchmarkMetadata(
        benchmark_name=raw_meta.get("benchmark_name", "SIH26138 Final Canonical Prediction Benchmark"),
        benchmark_version=raw_meta.get("benchmark_version", "3.0.0"),
        dataset_path=ds_path,
        seeds=list(raw_meta.get("seeds", [42, 43, 44, 45, 46])),
        split_protocol=raw_meta.get(
            "split_protocol",
            "GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed) on vessel_id",
        ),
        target_mode=raw_meta.get("target_mode", "absolute"),
        aggregation_method=raw_meta.get(
            "aggregation_method",
            "Unpooled per-seed sample mean and sample standard deviation (ddof=1)",
        ),
        subtitle="5-seed vessel-disjoint evaluation • GroupShuffleSplit by vessel_id • Mean across seeds",
        accuracy_claim="Physics + Residual QIFCP — canonical 5-seed result",
    )


def get_canonical_prediction_dataframe(
    data: dict[str, Any] | None = None,
    report_path: Path | str = DEFAULT_CANONICAL_BENCHMARK_PATH,
) -> pd.DataFrame:
    """Build standardized leaderboard dataframe from canonical benchmark data.

    Returns DataFrame containing strictly:
    - Model Architecture
    - Status (PRODUCTION / FROZEN, REFERENCE, ABLATION)
    - R² Score
    - RMSE (tons)
    - MAE (tons)
    - sMAPE (%)
    - Inference (ms / 100)
    - Architecture Class
    """
    if data is None:
        data = load_canonical_prediction_benchmark(report_path)

    aggregate = data.get("aggregate_results", {})
    model_specs = data.get("model_specifications", {})

    rows = []
    for model_id, display_name, status in CANONICAL_BENCHMARK_MODELS:
        if model_id not in aggregate:
            continue
        agg_m = aggregate[model_id]
        spec = model_specs.get(model_id, {})

        r2_val = float(agg_m.get("r2", {}).get("mean", 0.0))
        rmse_val = float(agg_m.get("rmse", {}).get("mean", 0.0))
        mae_val = float(agg_m.get("mae", {}).get("mean", 0.0))
        smape_val = float(agg_m.get("smape", {}).get("mean", 0.0))
        infer_val = float(agg_m.get("infer_ms_per_100_samples", {}).get("mean", 0.0))
        arch_class = spec.get("class", spec.get("class_name", ""))

        rows.append(
            {
                "Model Architecture": display_name,
                "Status": status,
                "R² Score": round(r2_val, 4),
                "RMSE (tons)": round(rmse_val, 2),
                "MAE (tons)": round(mae_val, 2),
                "sMAPE (%)": round(smape_val, 2),
                "Inference (ms / 100)": round(infer_val, 3),
                "Architecture Class": arch_class,
                "Model Key": model_id,
            }
        )

    # Optional secondary: check if linear regression exists in artifact
    if "linear_regression" in aggregate:
        agg_lin = aggregate["linear_regression"]
        spec_lin = model_specs.get("linear_regression", {})
        rows.append(
            {
                "Model Architecture": "Linear Regression",
                "Status": "REFERENCE",
                "R² Score": round(float(agg_lin.get("r2", {}).get("mean", 0.0)), 4),
                "RMSE (tons)": round(float(agg_lin.get("rmse", {}).get("mean", 0.0)), 2),
                "MAE (tons)": round(float(agg_lin.get("mae", {}).get("mean", 0.0)), 2),
                "sMAPE (%)": round(float(agg_lin.get("smape", {}).get("mean", 0.0)), 2),
                "Inference (ms / 100)": round(
                    float(agg_lin.get("infer_ms_per_100_samples", {}).get("mean", 0.0)), 3
                ),
                "Architecture Class": spec_lin.get("class", "LinearRegression"),
                "Model Key": "linear_regression",
            }
        )

    return pd.DataFrame(rows)
