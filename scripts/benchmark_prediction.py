"""Benchmark prediction models including baselines and QPSO-tuned QIFCP.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Evaluates LinearRegression, RandomForest, HistGradientBoosting, and QIFCP (Objective 1)
on the validated voyage dataset and generates outputs/reports/prediction_benchmark.json.
"""

import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

# Ensure project root is in sys.path when invoked directly from scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from contracts.schemas import VoyageRecord
from logging_config import configure_logging
from src.ingestion.dataset_loader import CSVDatasetLoader
from src.ingestion.feature_pipeline import FeatureEngineeringPipeline
from src.prediction.metrics import evaluate_predictions
from src.prediction.model_registry import ModelRegistry
from src.prediction.qifcp import QIFCPRegressor

logger = logging.getLogger("maritime_system")


def run_prediction_benchmark(
    dataset_path: str = "data/raw/voyages_sample.csv",
    output_report: str = "outputs/reports/prediction_benchmark.json",
    sample_size: int = 5000,
    test_split: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Execute end-to-end benchmark across all baseline and quantum-inspired regressors."""
    configure_logging(level="INFO")
    logger.info("Starting Objective 1 Prediction Benchmark on %s", dataset_path)

    loader = CSVDatasetLoader()
    records = loader.load_data(dataset_path)

    if sample_size and len(records) > sample_size:
        records = records[:sample_size]

    pipeline = FeatureEngineeringPipeline()
    X_df, y_series = pipeline.get_training_features_and_target(records, encode_categoricals=True)
    X = X_df.to_numpy(dtype=float)
    y = y_series.to_numpy(dtype=float)

    n_samples = len(X)
    split_idx = int(n_samples * (1.0 - test_split))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    registry = ModelRegistry(random_state=random_state)
    models_to_test = ["linear_regression", "random_forest", "hist_gradient_boosting", "qifcp"]

    benchmark_summary: dict[str, Any] = {
        "dataset_samples": n_samples,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features_count": X.shape[1],
        "models": {},
    }

    for model_name in models_to_test:
        logger.info("Benchmarking model: %s", model_name)
        model = registry.create_model(model_name)

        fit_start = time.perf_counter()
        if model_name == "qifcp" and isinstance(model, QIFCPRegressor):
            # Tune QIFCP hyperparameters using QPSO under equal-budget discipline
            model.tune_with_qpso(X_train, y_train, population_size=8, max_iterations=10)
        else:
            model.fit(X_train, y_train)
        fit_time = time.perf_counter() - fit_start

        infer_start = time.perf_counter()
        preds = model.predict(X_test)
        infer_time = time.perf_counter() - infer_start

        metrics = evaluate_predictions(y_test, preds, model_name=model_name)
        model_results = {
            "model_name": model_name,
            "rmse": metrics.rmse,
            "mae": metrics.mae,
            "mape": metrics.mape,
            "r2": metrics.r2,
            "fit_time_seconds": round(fit_time, 4),
            "infer_time_seconds": round(infer_time, 4),
            "infer_ms_per_100_samples": round((infer_time / len(X_test)) * 100000.0, 2),
        }
        if model_name == "qifcp" and isinstance(model, QIFCPRegressor):
            model_results["tuned_gamma"] = round(model.gamma, 4)
            model_results["tuned_alpha_reg"] = round(model.alpha_reg, 4)
            model_results["tuning_history"] = list(model.tuning_history_)

        benchmark_summary["models"][model_name] = model_results
        logger.info(
            "Model %s - RMSE: %.2f, MAPE: %.2f%%, R2: %.4f, Fit: %.2fs",
            model_name,
            metrics.rmse,
            metrics.mape * 100.0,
            metrics.r2,
            fit_time,
        )

    out_path = Path(output_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(benchmark_summary, indent=2), encoding="utf-8")
    logger.info("Saved prediction benchmark report to %s", out_path.resolve())

    return benchmark_summary


if __name__ == "__main__":
    run_prediction_benchmark()
