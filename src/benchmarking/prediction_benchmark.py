"""Predictive model benchmarking and error distribution profiling.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 3 Deliverable: Compares classical regressors against the Quantum-Inspired Predictor (QIFCP),
measuring MAE, RMSE, R², latency, prediction bias, and residual standard deviation.
"""

import logging
import time
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from contracts.schemas import PredictionBenchmarkResult
from src.ingestion.feature_pipeline import FeatureEngineeringPipeline
from src.prediction.model_registry import ModelRegistry

logger = logging.getLogger("maritime_system")


class PredictionModelBenchmarker:
    """Evaluates and compares classical vs quantum predictive model performance."""

    def __init__(self, random_state: int = 42) -> None:
        """Initialize benchmarker with model registry and feature pipeline."""
        self.random_state = random_state
        self.registry = ModelRegistry(random_state=random_state)
        self.pipeline = FeatureEngineeringPipeline()
        self.logger = logger

    def benchmark_models(
        self,
        X_train: np.ndarray | None = None,
        y_train: np.ndarray | None = None,
        X_test: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
        models_to_test: tuple[str, ...] = (
            "linear_regression",
            "random_forest",
            "hist_gradient_boosting",
            "qifcp",
        ),
    ) -> list[PredictionBenchmarkResult]:
        """Execute cross-model benchmarking over standardized evaluation split.

        Args:
            X_train: Training feature matrix. If None, synthesizes canonical verification data.
            y_train: Training targets.
            X_test: Test feature matrix.
            y_test: Test targets.
            models_to_test: Identifiers of models to evaluate.

        Returns:
            List of PredictionBenchmarkResult instances.
        """
        # If no data passed, synthesize representative hydrodynamic evaluation data
        if X_train is None or y_train is None or X_test is None or y_test is None:
            X_train, y_train, X_test, y_test = self._generate_synthetic_eval_data()

        results: list[PredictionBenchmarkResult] = []

        for model_id in models_to_test:
            self.logger.info("Benchmarking model: %s", model_id)
            model = self.registry.create_model(model_id)

            # 1. Fit model
            t_fit_start = time.perf_counter()
            model.fit(X_train, y_train)
            fit_time = time.perf_counter() - t_fit_start

            # 2. Inference latency measurement (per 1,000 samples)
            t_inf_start = time.perf_counter()
            y_pred = model.predict(X_test)
            inf_time = time.perf_counter() - t_inf_start
            n_samples = len(X_test)
            latency_per_1k = (inf_time / max(n_samples, 1)) * 1000.0 * 1000.0  # ms per 1,000 samples

            # 3. Standard metrics
            mae = float(mean_absolute_error(y_test, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2 = float(r2_score(y_test, y_pred))

            # 4. Error distribution metrics (bias and standard deviation)
            residuals = y_pred - y_test
            bias = float(np.mean(residuals))
            error_std = float(np.std(residuals))

            is_quantum = "qifcp" in model_id.lower()

            results.append(
                PredictionBenchmarkResult(
                    model_name=self._format_model_name(model_id),
                    mae=round(mae, 4),
                    rmse=round(rmse, 4),
                    r2=round(r2, 4),
                    inference_time_ms=round(latency_per_1k, 3),
                    prediction_bias=round(bias, 4),
                    error_std=round(error_std, 4),
                    is_quantum=is_quantum,
                    metadata={
                        "fit_time_seconds": round(fit_time, 4),
                        "test_samples": n_samples,
                    },
                )
            )

        return results

    def _format_model_name(self, model_id: str) -> str:
        """Translate model identifier into human-readable label."""
        mapping = {
            "linear_regression": "Linear Regression (Baseline)",
            "random_forest": "Random Forest",
            "hist_gradient_boosting": "HistGBDT (Production Tree)",
            "qifcp": "Quantum-Inspired Predictor (QIFCP)",
        }
        return mapping.get(model_id, model_id.replace("_", " ").title())

    def _generate_synthetic_eval_data(
        self,
        n_samples: int = 2000,
        n_features: int = 12,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate physically consistent synthetic evaluation dataset for deterministic verification."""
        rng = np.random.default_rng(self.random_state)

        # Features: [speed, dwt, cargo, distance, weather, draft, lcv, ...]
        X = rng.normal(0.0, 1.0, size=(n_samples, n_features))
        # Admiralty-inspired non-linear target: quadratic + cubic interaction + noise
        speed_col = X[:, 0]
        dwt_col = np.abs(X[:, 1]) + 0.5
        y = 2.5 * (speed_col ** 3) * (dwt_col ** 0.67) + 10.0 + rng.normal(0.0, 0.5, size=n_samples)
        y = np.clip(y, 1.0, 150.0)

        split = int(n_samples * 0.8)
        return X[:split], y[:split], X[split:], y[split:]
