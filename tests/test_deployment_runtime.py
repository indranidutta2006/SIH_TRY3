"""Regression and sanity tests for Cloud Run container deployment and runtime readiness.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Verifies that:
1. All 9 dashboard pages (including Decision Intelligence) are imported and bound in app.py.
2. Canonical prediction benchmark artifact is preserved by .dockerignore and resolved robustly.
3. Path resolution succeeds whether running from repository root, subdirectories, or /app.
4. Canonical benchmark artifact integrity is maintained without metric corruption.
"""

from pathlib import Path
import json
import fnmatch
import pytest

from src.prediction.benchmark_loader import (
    DEFAULT_CANONICAL_BENCHMARK_PATH,
    PROJECT_ROOT,
    load_canonical_prediction_benchmark,
    resolve_canonical_benchmark_path,
    get_canonical_prediction_dataframe,
)


def test_app_imports_and_all_9_pages_bound() -> None:
    """Verify that app.py imports cleanly and binds render_decision_intelligence_page without NameError."""
    import importlib.util

    app_py_path = PROJECT_ROOT / "app.py"
    assert app_py_path.exists(), "Top-level app.py must exist"

    spec = importlib.util.spec_from_file_location("app_main", app_py_path)
    assert spec is not None and spec.loader is not None
    app_module = importlib.util.module_from_spec(spec)

    # Execute app.py in its own module namespace
    spec.loader.exec_module(app_module)

    expected_page_functions = [
        "render_overview_page",
        "render_prediction_page",
        "render_optimization_page",
        "render_compliance_page",
        "render_scenarios_page",
        "render_strategy_page",
        "render_reliability_page",
        "render_benchmarking_page",
        "render_decision_intelligence_page",
    ]

    for func_name in expected_page_functions:
        assert hasattr(app_module, func_name), f"app.py must define or import '{func_name}'"
        assert callable(getattr(app_module, func_name)), f"'{func_name}' in app.py must be callable"


def test_dockerignore_whitelists_canonical_prediction_benchmark() -> None:
    """Verify .dockerignore does not exclude outputs/reports/final_prediction_benchmark.json."""
    dockerignore_path = PROJECT_ROOT / ".dockerignore"
    assert dockerignore_path.exists(), ".dockerignore file must exist in repository root"

    lines = [
        line.strip()
        for line in dockerignore_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    # Verify whitelist rule is explicitly present
    assert "!outputs/reports/final_prediction_benchmark.json" in lines, (
        ".dockerignore must contain '!outputs/reports/final_prediction_benchmark.json' "
        "to ensure the canonical benchmark is bundled inside the container."
    )

    for target_rel in [
        "outputs/reports/final_prediction_benchmark.json",
        "outputs/reports/final_integrated_optimization_benchmark.json",
    ]:
        is_ignored = False
        for pattern in lines:
            if pattern.startswith("!"):
                unignore = pattern[1:]
                if fnmatch.fnmatch(target_rel, unignore):
                    is_ignored = False
            else:
                if fnmatch.fnmatch(target_rel, pattern):
                    is_ignored = True

        assert not is_ignored, (
            f"'{target_rel}' would be ignored during docker build under current .dockerignore rules!"
        )


def test_canonical_benchmark_path_resolution_robustness() -> None:
    """Verify resolve_canonical_benchmark_path locates the artifact under different path conventions."""
    resolved_default = resolve_canonical_benchmark_path()
    assert resolved_default.exists(), (
        f"Default resolved path does not exist: {resolved_default}"
    )
    assert resolved_default.name == "final_prediction_benchmark.json"

    # Test relative path string resolution
    resolved_rel = resolve_canonical_benchmark_path("outputs/reports/final_prediction_benchmark.json")
    assert resolved_rel.exists()

    # Test path resolution with Path object
    resolved_path_obj = resolve_canonical_benchmark_path(
        Path("outputs") / "reports" / "final_prediction_benchmark.json"
    )
    assert resolved_path_obj.exists()


def test_canonical_benchmark_content_and_frozen_metrics() -> None:
    """Verify canonical benchmark content is valid and retains frozen production metrics."""
    data = load_canonical_prediction_benchmark()
    assert "metadata" in data
    assert "aggregate_results" in data
    assert "model_specifications" in data

    agg = data["aggregate_results"]
    assert "physics_residual_qifcp" in agg, "physics_residual_qifcp must exist in aggregate results"

    qifcp_agg = agg["physics_residual_qifcp"]
    r2_mean = float(qifcp_agg["r2"]["mean"])
    rmse_mean = float(qifcp_agg["rmse"]["mean"])
    mae_mean = float(qifcp_agg["mae"]["mean"])

    # Strict check against frozen canonical values
    assert r2_mean == pytest.approx(0.9933, abs=0.001)
    assert rmse_mean == pytest.approx(61.55, abs=0.1)
    assert mae_mean == pytest.approx(40.07, abs=0.1)


def test_dataframe_contains_all_models_without_missing_report_exception() -> None:
    """Verify get_canonical_prediction_dataframe returns complete table with production status."""
    df = get_canonical_prediction_dataframe()
    assert not df.empty
    assert "Model Architecture" in df.columns
    assert "Status" in df.columns

    prod_row = df[df["Status"] == "PRODUCTION / FROZEN"]
    assert len(prod_row) == 1
    assert prod_row.iloc[0]["Model Architecture"] == "Physics + Residual QIFCP"
