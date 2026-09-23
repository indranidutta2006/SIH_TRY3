"""Benchmark prediction models including baselines and QPSO-tuned QIFCP.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Evaluates LinearRegression, RandomForest, HistGradientBoosting, and QIFCP (Objective 1)
on the validated voyage dataset and generates outputs/reports/prediction_benchmark.json.
Supports GroupShuffleSplit by vessel_id, per-data-source metrics breakdown, absolute vs. rate
target modes, and segmented per-data-source routing architectures.
"""

import argparse
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

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

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
    target_mode: str = "absolute",
) -> dict[str, Any]:
    """Execute end-to-end benchmark across all baseline and quantum-inspired regressors.

    Args:
        dataset_path: Path to CSV dataset file.
        output_report: Destination path for JSON report.
        sample_size: Maximum records to use (0 or None for full dataset).
        test_split: Test partition ratio for GroupShuffleSplit.
        random_state: Deterministic random seed.
        target_mode: 'absolute' (predicts fuel_consumption directly) or
                     'rate' (predicts fuel_consumption / hours_at_sea, then
                     reconstructs absolute consumption as rate * hours_at_sea).

    Returns:
        Dictionary of benchmark metrics and metadata.
    """
    configure_logging(level="INFO")
    logger.info(
        "Starting Objective 1 Prediction Benchmark on %s [Target Mode: %s]",
        dataset_path,
        target_mode,
    )

    loader = CSVDatasetLoader()
    records = loader.load_data(dataset_path)

    if sample_size and len(records) > sample_size:
        records = records[:sample_size]

    valid_records = [rec for rec in records if rec.fuel_consumption is not None]
    if len(valid_records) != len(records):
        logger.warning(
            "Filtered %d records with missing fuel_consumption target.",
            len(records) - len(valid_records),
        )
        records = valid_records

    pipeline = FeatureEngineeringPipeline()
    X_df, _ = pipeline.get_training_features_and_target(records, encode_categoricals=True)
    X = X_df.to_numpy(dtype=float)
    n_samples = len(X)

    # Extract parallel metadata for group splitting and rate transformation
    vessel_groups = np.array([str(rec.vessel_id) for rec in records])
    data_sources = np.array([str(rec.data_source) for rec in records])
    hours_at_sea = np.array([max(float(rec.hours_at_sea), 1e-4) for rec in records])
    y_abs = np.array([float(rec.fuel_consumption) for rec in records])
    y_rate = y_abs / hours_at_sea

    # Choose training target based on target_mode
    if target_mode == "rate":
        y = y_rate
    elif target_mode == "absolute":
        y = y_abs
    else:
        raise ValueError(f"Unknown target_mode '{target_mode}'. Choose 'absolute' or 'rate'.")

    # GroupShuffleSplit ensures zero vessel leakage between train and test sets
    gss = GroupShuffleSplit(n_splits=1, test_size=test_split, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=vessel_groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train = y[train_idx]
    y_test_abs = y_abs[test_idx]
    test_hours = hours_at_sea[test_idx]
    test_sources = data_sources[test_idx]

    train_vessels = sorted(set(vessel_groups[train_idx]))
    test_vessels = sorted(set(vessel_groups[test_idx]))
    unique_sources = sorted({str(src) for src in test_sources})

    logger.info(
        "Group-based split (by vessel_id): Train=%d samples (%d vessels), Test=%d samples (%d vessels), Overlap=%s",
        len(X_train),
        len(train_vessels),
        len(X_test),
        len(test_vessels),
        not set(train_vessels).isdisjoint(set(test_vessels)),
    )
    test_source_counts = {str(src): int(np.sum(test_sources == src)) for src in unique_sources}
    logger.info("Test set data_source distribution: %s", test_source_counts)

    registry = ModelRegistry(random_state=random_state)
    models_to_test = ["linear_regression", "random_forest", "hist_gradient_boosting", "qifcp"]

    benchmark_summary: dict[str, Any] = {
        "dataset_samples": n_samples,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_vessels_count": len(train_vessels),
        "test_vessels_count": len(test_vessels),
        "test_source_distribution": test_source_counts,
        "target_mode": target_mode,
        "approach": "global",
        "split_method": "GroupShuffleSplit(groups=vessel_id, zero_vessel_leakage)",
        "features_count": X.shape[1],
        "models": {},
    }

    for model_name in models_to_test:
        logger.info("Benchmarking model: %s (target_mode=%s)", model_name, target_mode)
        model = registry.create_model(model_name)

        fit_start = time.perf_counter()
        if model_name == "qifcp" and isinstance(model, QIFCPRegressor):
            # Tune QIFCP hyperparameters using QPSO under equal-budget discipline
            model.tune_with_qpso(X_train, y_train, population_size=8, max_iterations=10)
        else:
            model.fit(X_train, y_train)
        fit_time = time.perf_counter() - fit_start

        infer_start = time.perf_counter()
        preds_raw = model.predict(X_test)
        infer_time = time.perf_counter() - infer_start

        # Reconstruct absolute consumption if trained in rate mode
        if target_mode == "rate":
            preds_abs = preds_raw * test_hours
        else:
            preds_abs = preds_raw

        # (1) Compute Global Metrics on reconstructed absolute consumption
        metrics = evaluate_predictions(y_test_abs, preds_abs, model_name=model_name)

        # (2) Compute metrics separately per data_source group
        by_source: dict[str, dict[str, Any]] = {}
        for src in unique_sources:
            src_str = str(src)
            src_mask = test_sources == src
            if np.sum(src_mask) > 0:
                y_src = y_test_abs[src_mask]
                p_src = preds_abs[src_mask]
                src_metrics = evaluate_predictions(y_src, p_src, model_name=f"{model_name}_{src_str}")

                # Normalized error metrics for cross-source comparability
                mean_target = float(np.mean(y_src))
                nrmse_mean = float(src_metrics.rmse / mean_target) if mean_target > 0 else float("inf")
                nmae_mean = float(src_metrics.mae / mean_target) if mean_target > 0 else float("inf")
                mape_pct = float(
                    np.mean(np.abs((y_src - p_src) / np.clip(np.abs(y_src), 1e-8, None))) * 100
                )

                by_source[src_str] = {
                    "count": int(np.sum(src_mask)),
                    "rmse": src_metrics.rmse,
                    "mae": src_metrics.mae,
                    "mape": src_metrics.mape,
                    "r2": src_metrics.r2,
                    "mean_target": round(mean_target, 2),
                    "nrmse_mean": round(nrmse_mean, 4),
                    "nmae_mean": round(nmae_mean, 4),
                    "mape_pct": round(mape_pct, 2),
                }

        model_results: dict[str, Any] = {
            "model_name": model_name,
            "target_mode": target_mode,
            "approach": "global",
            "rmse": metrics.rmse,
            "mae": metrics.mae,
            "mape": metrics.mape,
            "r2": metrics.r2,
            "by_source": by_source,
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
            "Model %-22s [GLOBAL] - RMSE: %8.2f | MAE: %8.2f | MAPE: %6.2f%% | R2: %7.4f | Fit: %.2fs",
            model_name,
            metrics.rmse,
            metrics.mae,
            metrics.mape,
            metrics.r2,
            fit_time,
        )
        for src, sm in by_source.items():
            logger.info(
                "  -> %-10s (N=%4d): RMSE: %8.2f | MAE: %8.2f | MAPE: %6.2f%% | R2: %7.4f",
                src,
                sm["count"],
                sm["rmse"],
                sm["mae"],
                sm["mape"],
                sm["r2"],
            )

    out_path = Path(output_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(benchmark_summary, indent=2), encoding="utf-8")
    logger.info("Saved prediction benchmark report to %s", out_path.resolve())

    return benchmark_summary


def run_segmented_benchmark(
    dataset_path: str = "data/processed/voyages_blended.csv",
    output_report: str = "outputs/reports/prediction_benchmark_segmented.json",
    sample_size: int = 5000,
    test_split: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Execute segmented benchmarking where a specialized model is trained per data_source.

    Uses the exact same GroupShuffleSplit by vessel_id so that comparisons against
    the global models are perfectly controlled and fair.

    Args:
        dataset_path: Path to CSV dataset file.
        output_report: Destination path for JSON report.
        sample_size: Maximum records to use.
        test_split: Test partition ratio for GroupShuffleSplit.
        random_state: Deterministic random seed.

    Returns:
        Dictionary of segmented benchmark metrics and metadata.
    """
    configure_logging(level="INFO")
    logger.info(
        "Starting Segmented (Per-Data-Source) Prediction Benchmark on %s",
        dataset_path,
    )

    loader = CSVDatasetLoader()
    records = loader.load_data(dataset_path)

    if sample_size and len(records) > sample_size:
        records = records[:sample_size]

    valid_records = [rec for rec in records if rec.fuel_consumption is not None]
    if len(valid_records) != len(records):
        records = valid_records

    pipeline = FeatureEngineeringPipeline()
    X_df, _ = pipeline.get_training_features_and_target(records, encode_categoricals=True)
    X = X_df.to_numpy(dtype=float)
    n_samples = len(X)

    vessel_groups = np.array([str(rec.vessel_id) for rec in records])
    data_sources = np.array([str(rec.data_source) for rec in records])
    y_abs = np.array([float(rec.fuel_consumption) for rec in records])

    # Exact same GroupShuffleSplit as global models
    gss = GroupShuffleSplit(n_splits=1, test_size=test_split, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y_abs, groups=vessel_groups))

    test_sources = data_sources[test_idx]
    train_sources = data_sources[train_idx]
    train_vessels = sorted(set(vessel_groups[train_idx]))
    test_vessels = sorted(set(vessel_groups[test_idx]))
    unique_sources = sorted({str(src) for src in test_sources})

    logger.info(
        "Segmented Group Split (vessel_id): Train=%d samples (%d vessels), Test=%d samples (%d vessels)",
        len(train_idx),
        len(train_vessels),
        len(test_idx),
        len(test_vessels),
    )
    test_source_counts = {str(src): int(np.sum(test_sources == src)) for src in unique_sources}

    registry = ModelRegistry(random_state=random_state)
    models_to_test = ["linear_regression", "random_forest", "hist_gradient_boosting", "qifcp"]

    benchmark_summary: dict[str, Any] = {
        "dataset_samples": n_samples,
        "train_samples": len(train_idx),
        "test_samples": len(test_idx),
        "train_vessels_count": len(train_vessels),
        "test_vessels_count": len(test_vessels),
        "test_source_distribution": test_source_counts,
        "target_mode": "absolute",
        "approach": "segmented",
        "split_method": "GroupShuffleSplit(groups=vessel_id, zero_vessel_leakage)",
        "features_count": X.shape[1],
        "models": {},
    }

    for model_name in models_to_test:
        logger.info("Benchmarking segmented models for: %s", model_name)
        all_segmented_preds = np.zeros(len(test_idx), dtype=float)
        by_source: dict[str, dict[str, Any]] = {}
        total_fit_time = 0.0
        total_infer_time = 0.0

        for src in unique_sources:
            src_str = str(src)
            tr_mask = train_sources == src
            te_mask = test_sources == src

            X_tr_src = X[train_idx][tr_mask]
            y_tr_src = y_abs[train_idx][tr_mask]
            X_te_src = X[test_idx][te_mask]
            y_te_src = y_abs[test_idx][te_mask]

            model_src = registry.create_model(model_name)

            t0 = time.perf_counter()
            if model_name == "qifcp" and isinstance(model_src, QIFCPRegressor):
                model_src.tune_with_qpso(X_tr_src, y_tr_src, population_size=8, max_iterations=10)
            else:
                model_src.fit(X_tr_src, y_tr_src)
            fit_t = time.perf_counter() - t0
            total_fit_time += fit_t

            t1 = time.perf_counter()
            preds_src = model_src.predict(X_te_src)
            infer_t = time.perf_counter() - t1
            total_infer_time += infer_t

            all_segmented_preds[te_mask] = preds_src

            src_metrics = evaluate_predictions(y_te_src, preds_src, model_name=f"{model_name}_{src_str}_seg")
            by_source[src_str] = {
                "count": int(np.sum(te_mask)),
                "rmse": src_metrics.rmse,
                "mae": src_metrics.mae,
                "mape": src_metrics.mape,
                "r2": src_metrics.r2,
                "fit_time_seconds": round(fit_t, 4),
            }

        # Global evaluation across all concatenated segmented predictions
        global_metrics = evaluate_predictions(y_abs[test_idx], all_segmented_preds, model_name=f"{model_name}_segmented")

        model_results: dict[str, Any] = {
            "model_name": model_name,
            "target_mode": "absolute",
            "approach": "segmented",
            "rmse": global_metrics.rmse,
            "mae": global_metrics.mae,
            "mape": global_metrics.mape,
            "r2": global_metrics.r2,
            "by_source": by_source,
            "fit_time_seconds": round(total_fit_time, 4),
            "infer_time_seconds": round(total_infer_time, 4),
            "infer_ms_per_100_samples": round((total_infer_time / len(test_idx)) * 100000.0, 2),
        }

        benchmark_summary["models"][model_name] = model_results
        logger.info(
            "Segmented %-22s [COMBINED] - RMSE: %8.2f | MAE: %8.2f | MAPE: %6.2f%% | R2: %7.4f | Total Fit: %.2fs",
            model_name,
            global_metrics.rmse,
            global_metrics.mae,
            global_metrics.mape,
            global_metrics.r2,
            total_fit_time,
        )
        for src, sm in by_source.items():
            logger.info(
                "  -> %-10s (N=%4d): RMSE: %8.2f | MAE: %8.2f | MAPE: %6.2f%% | R2: %7.4f",
                src,
                sm["count"],
                sm["rmse"],
                sm["mae"],
                sm["mape"],
                sm["r2"],
            )

    out_path = Path(output_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(benchmark_summary, indent=2), encoding="utf-8")
    logger.info("Saved segmented benchmark report to %s", out_path.resolve())

    return benchmark_summary


def compare_all_paradigms(
    dataset_path: str = "data/processed/voyages_blended.csv",
    sample_size: int = 5000,
    random_state: int = 42,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run Global-Absolute, Global-Rate, and Segmented paradigms and display a unified table."""
    print("=" * 115)
    print("PARADIGM 1: GLOBAL ABSOLUTE TARGET (Universal model on fuel_consumption)")
    print("=" * 115)
    res_abs = run_prediction_benchmark(
        dataset_path=dataset_path,
        output_report="outputs/reports/prediction_benchmark_absolute.json",
        sample_size=sample_size,
        random_state=random_state,
        target_mode="absolute",
    )

    print("\n" + "=" * 115)
    print("PARADIGM 2: GLOBAL RATE TARGET (Universal model on fuel_consumption / hours_at_sea)")
    print("=" * 115)
    res_rate = run_prediction_benchmark(
        dataset_path=dataset_path,
        output_report="outputs/reports/prediction_benchmark_rate.json",
        sample_size=sample_size,
        random_state=random_state,
        target_mode="rate",
    )

    print("\n" + "=" * 115)
    print("PARADIGM 3: SEGMENTED ARCHITECTURE (Specialized model trained per data_source)")
    print("=" * 115)
    res_seg = run_segmented_benchmark(
        dataset_path=dataset_path,
        output_report="outputs/reports/prediction_benchmark_segmented.json",
        sample_size=sample_size,
        random_state=random_state,
    )

    # Set default report to segmented or rate as chosen production reference
    default_out = Path("outputs/reports/prediction_benchmark.json")
    default_out.write_text(json.dumps(res_seg, indent=2), encoding="utf-8")

    models = list(res_abs["models"].keys())
    sources = list(res_abs["test_source_distribution"].keys())

    print("\n" + "=" * 120)
    print("UNIFIED MULTI-PARADIGM BENCHMARK COMPARISON TABLE")
    print("=" * 120)

    for m_name in models:
        m_abs = res_abs["models"][m_name]
        m_rate = res_rate["models"][m_name]
        m_seg = res_seg["models"][m_name]

        print(f"\nModel Architecture: {m_name.upper()}")
        print("-" * 120)
        print(f"{'Data Scope':<18} | {'Formulation / Approach':<22} | {'RMSE (tons)':>12} | {'MAE (tons)':>12} | {'MAPE (%)':>10} | {'R2 Score':>10}")
        print("-" * 120)

        # Global rows
        print(f"{'GLOBAL':<18} | {'Global-Absolute':<22} | {m_abs['rmse']:>12.2f} | {m_abs['mae']:>12.2f} | {m_abs['mape']:>9.2f}% | {m_abs['r2']:>10.4f}")
        print(f"{'GLOBAL':<18} | {'Global-Rate':<22} | {m_rate['rmse']:>12.2f} | {m_rate['mae']:>12.2f} | {m_rate['mape']:>9.2f}% | {m_rate['r2']:>10.4f}")
        print(f"{'GLOBAL':<18} | {'Segmented (Routed)':<22} | {m_seg['rmse']:>12.2f} | {m_seg['mae']:>12.2f} | {m_seg['mape']:>9.2f}% | {m_seg['r2']:>10.4f}")
        print("." * 120)

        # Per source rows
        for src in sources:
            s_abs = m_abs["by_source"].get(src, {})
            s_rate = m_rate["by_source"].get(src, {})
            s_seg = m_seg["by_source"].get(src, {})
            s_count = s_abs.get("count", 0)
            src_label = f"{src} (N={s_count})"

            print(f"{src_label:<18} | {'Global-Absolute':<22} | {s_abs.get('rmse', 0.0):>12.2f} | {s_abs.get('mae', 0.0):>12.2f} | {s_abs.get('mape', 0.0):>9.2f}% | {s_abs.get('r2', 0.0):>10.4f}")
            print(f"{src_label:<18} | {'Global-Rate':<22} | {s_rate.get('rmse', 0.0):>12.2f} | {s_rate.get('mae', 0.0):>12.2f} | {s_rate.get('mape', 0.0):>9.2f}% | {s_rate.get('r2', 0.0):>10.4f}")
            print(f"{src_label:<18} | {'Segmented (Routed)':<22} | {s_seg.get('rmse', 0.0):>12.2f} | {s_seg.get('mae', 0.0):>12.2f} | {s_seg.get('mape', 0.0):>9.2f}% | {s_seg.get('r2', 0.0):>10.4f}")
            print("." * 120)

    return res_abs, res_rate, res_seg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prediction benchmark.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/raw/voyages_sample.csv",
        help="Dataset path (default: data/raw/voyages_sample.csv)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Sample size (default: 5000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/reports/prediction_benchmark.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--target-mode",
        type=str,
        choices=["absolute", "rate"],
        default="absolute",
        help="Target formulation: 'absolute' (fuel_consumption) or 'rate' (fuel_consumption / hours_at_sea)",
    )
    parser.add_argument(
        "--segmented",
        action="store_true",
        help="Train separate specialized models per data_source group",
    )
    parser.add_argument(
        "--compare-modes",
        action="store_true",
        help="Run global absolute, global rate, and segmented modes side-by-side in a combined comparison table",
    )

    args = parser.parse_args()

    if args.compare_modes:
        compare_all_paradigms(
            dataset_path=args.data,
            sample_size=args.sample_size,
        )
    elif args.segmented:
        run_segmented_benchmark(
            dataset_path=args.data,
            sample_size=args.sample_size,
            output_report=args.output,
        )
    else:
        run_prediction_benchmark(
            dataset_path=args.data,
            sample_size=args.sample_size,
            output_report=args.output,
            target_mode=args.target_mode,
        )
