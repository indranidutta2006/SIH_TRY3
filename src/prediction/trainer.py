"""Training engine and pipeline orchestration for baseline fuel consumption models.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 2A training subsystem training, evaluating, comparing, and persisting baseline regressors
(LinearRegression, RandomForest, HistGradientBoosting) with strict leakage validation.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.model_selection import train_test_split

from contracts.exceptions import DataValidationError, PredictionError
from contracts.schemas import VoyageRecord
from src.ingestion.dataset_loader import CSVDatasetLoader
from src.ingestion.feature_pipeline import FeatureEngineeringPipeline
from src.ingestion.validators import ValidationEngine
from src.prediction.metrics import (
    ModelComparisonResult,
    ModelMetrics,
    evaluate_predictions,
    select_best_model,
)
from src.prediction.model_registry import (
    DEFAULT_RANDOM_STATE,
    ModelRegistry,
    normalize_model_name,
)

logger = logging.getLogger("maritime_system")


class PredictionTrainer:
    """Trains, benchmarks, and persists maritime fuel consumption prediction models."""

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        test_size: float = 0.2,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> None:
        """Initialize trainer with model registry and split configurations.

        Args:
            registry: ModelRegistry instance or None for default.
            test_size: Proportion of dataset allocated to out-of-sample evaluation.
            random_state: Seed ensuring deterministic splits and initializations.
        """
        self.registry = registry or ModelRegistry(random_state=random_state)
        self.test_size = test_size
        self.random_state = random_state
        self.logger = logger

    def train_and_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_names: Sequence[str] | None = None,
    ) -> tuple[dict[str, RegressorMixin], ModelComparisonResult]:
        """Train candidate models, evaluate on held-out test split, and rank results.

        Args:
            X: Model feature matrix (without target leakage).
            y: Target label series representing fuel_consumption.
            model_names: Optional subset of model names to train. Defaults to all registered.

        Returns:
            Tuple of (dict of trained models, ModelComparisonResult).

        Raises:
            PredictionError: If training fails or data is insufficient.
        """
        if X.empty or y.empty:
            raise PredictionError("Cannot train on empty feature matrix or target vector.")
        if len(X) != len(y):
            raise PredictionError(
                f"Feature matrix rows ({len(X)}) does not match target labels ({len(y)})."
            )

        target_models = (
            [normalize_model_name(m) for m in model_names]
            if model_names is not None
            else self.registry.get_supported_models()
        )

        self.logger.info(
            "Splitting dataset: N=%d with test_size=%.2f (seed=%d)",
            len(X),
            self.test_size,
            self.random_state,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        trained_models: dict[str, RegressorMixin] = {}
        metrics_by_model: dict[str, ModelMetrics] = {}

        for model_name in target_models:
            self.logger.info("Training baseline model '%s'...", model_name)
            model = self.registry.create_model(model_name)
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            except Exception as err:
                raise PredictionError(
                    f"Failed to fit/predict model '{model_name}': {err}",
                    details={"model_name": model_name, "error": str(err)},
                ) from err

            metrics = evaluate_predictions(y_test, y_pred, model_name=model_name)
            trained_models[model_name] = model
            metrics_by_model[model_name] = metrics

        best_model_name = select_best_model(metrics_by_model, primary_metric="rmse")
        timestamp_str = datetime.now(UTC).isoformat()

        comparison = ModelComparisonResult(
            best_model_name=best_model_name,
            metrics_by_model=metrics_by_model,
            training_timestamp=timestamp_str,
        )

        self.logger.info(
            "Benchmark completed. Best performing model: '%s' (RMSE=%.4f)",
            best_model_name,
            metrics_by_model[best_model_name].rmse,
        )
        return trained_models, comparison

    def save_artifacts(
        self,
        trained_models: dict[str, RegressorMixin],
        comparison_result: ModelComparisonResult,
        artifacts_dir: Path | str = "artifacts",
        feature_columns: list[str] | None = None,
        target_mode: str = "absolute",
    ) -> dict[str, Path]:
        """Persist trained model binaries and evaluation metrics to storage.

        Args:
            trained_models: Dictionary mapping canonical model names to fitted regressors.
            comparison_result: ModelComparisonResult benchmark summary.
            artifacts_dir: Root directory for model and metrics storage.
            feature_columns: Optional list of feature column names used in training.
            target_mode: 'absolute' or 'rate' — stamped into every artifact payload so
                         PredictionInferenceEngine applies the correct inverse transform
                         (rate x implied_hours) at inference time.

        Returns:
            Dictionary mapping artifact names to their persisted Path locations.
        """
        root_path = Path(artifacts_dir)
        models_dir = root_path / "models"
        metrics_dir = root_path / "metrics"

        models_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: dict[str, Path] = {}

        for model_name, model in trained_models.items():
            filename = self.registry.get_artifact_filename(model_name)
            model_path = models_dir / filename

            # Package model artifact with training metadata, feature schema, and target mode.
            # target_mode is critical: it tells the inference engine whether to multiply
            # raw predictions by implied_hours to reconstruct absolute fuel consumption.
            artifact_payload = {
                "model": model,
                "model_name": model_name,
                "feature_columns": feature_columns or [],
                "timestamp": comparison_result.training_timestamp,
                "target_mode": target_mode,  # "absolute" or "rate"
            }
            joblib.dump(artifact_payload, model_path)
            saved_paths[f"model_{model_name}"] = model_path
            self.logger.info(
                "Persisted model '%s' [target_mode=%s] to %s",
                model_name,
                target_mode,
                model_path,
            )

        # Save metrics comparison JSON — include training_config block for transparency
        import json as _json
        raw_metrics = _json.loads(comparison_result.to_json())
        raw_metrics["training_config"] = {
            "target_mode": target_mode,
            "target_description": (
                "fuel_consumption (metric tons)"
                if target_mode == "absolute"
                else "fuel_consumption / hours_at_sea (tons per hour) — "
                     "inference multiplies by implied_hours to return metric tons"
            ),
        }
        metrics_path = metrics_dir / "baseline_metrics.json"
        metrics_path.write_text(_json.dumps(raw_metrics, indent=2), encoding="utf-8")
        saved_paths["baseline_metrics"] = metrics_path
        self.logger.info("Persisted benchmark metrics to %s", metrics_path)

        return saved_paths



def train_all_models(
    dataset_path: Path | str = "data/raw/voyages_sample.csv",
    artifacts_dir: Path | str = "artifacts",
    test_size: float = 0.2,
    random_state: int = DEFAULT_RANDOM_STATE,
    target_mode: str = "absolute",
) -> ModelComparisonResult:
    """Execute end-to-end baseline model training, evaluation, and persistence pipeline.

    Workflow:
    1. Load raw CSV voyage records via CSVDatasetLoader.
    2. Validate dataset boundaries and uniqueness via ValidationEngine.
    3. Generate maritime domain features via FeatureEngineeringPipeline.
    4. Extract feature matrix X and target y with strict leakage guards.
    5. Optionally transform y to fuel-rate form (fuel_consumption / hours_at_sea).
    6. Train all registered baseline models (LinearRegression, RF, HistGBDT).
    7. Evaluate out-of-sample performance metrics on held-out test split.
    8. Persist trained model artifacts (with target_mode stamped) and baseline_metrics.json.
    9. Return structured ModelComparisonResult.

    Args:
        dataset_path: Path to input CSV dataset.
        artifacts_dir: Destination folder for model and metric artifacts.
        test_size: Out-of-sample validation split ratio.
        random_state: Random seed for deterministic reproducibility.
        target_mode: 'absolute' trains y = fuel_consumption (metric tons).
                     'rate' trains y = fuel_consumption / hours_at_sea (tons/h).
                     The mode is stored in every artifact so the inference engine
                     applies the matching inverse transform (rate x implied_hours).

    Returns:
        ModelComparisonResult summarising comparative metrics and best model selection.

    Raises:
        ValueError: If target_mode is not 'absolute' or 'rate'.
    """
    if target_mode not in ("absolute", "rate"):
        raise ValueError(f"target_mode must be 'absolute' or 'rate', got '{target_mode}'.")

    logger.info(
        "Initiating baseline prediction training pipeline from %s [target_mode=%s]",
        dataset_path,
        target_mode,
    )

    # 1. Ingestion
    loader = CSVDatasetLoader()
    records: list[VoyageRecord] = loader.load_data(dataset_path)

    # 2. Validation
    validator = ValidationEngine()
    validator.validate_dataset(records, strict=True)

    # 3 & 4. Feature Extraction & Leakage Guardrails
    pipeline = FeatureEngineeringPipeline()
    X, y_abs = pipeline.get_training_features_and_target(
        records, encode_categoricals=True
    )

    # Explicitly enforce target leakage prohibition
    pipeline.check_for_target_leakage(X)

    # 5. Target transformation
    if target_mode == "rate":
        # Compute hours_at_sea from the raw records — NOT from the feature matrix
        # (hours_at_sea was intentionally dropped from X to remove proxy leakage).
        # The 1e-4 floor matches the clamp used in benchmark_prediction.py.
        hours = pd.Series(
            [max(float(r.hours_at_sea), 1e-4) for r in records],
            index=y_abs.index,
            name="hours_at_sea",
        )
        y = (y_abs / hours).rename("fuel_rate_tons_per_hour")
        logger.info(
            "Rate target mode: y = fuel_consumption / hours_at_sea  "
            "(mean rate = %.4f t/h, stdev = %.4f t/h)",
            float(y.mean()),
            float(y.std()),
        )
    else:
        y = y_abs

    # 6 & 7. Training & Evaluation
    trainer = PredictionTrainer(test_size=test_size, random_state=random_state)
    trained_models, comparison_result = trainer.train_and_evaluate(X, y)

    # 8. Persistence — target_mode is forwarded so artifacts carry it
    trainer.save_artifacts(
        trained_models=trained_models,
        comparison_result=comparison_result,
        artifacts_dir=artifacts_dir,
        feature_columns=list(X.columns),
        target_mode=target_mode,
    )

    logger.info(
        "Baseline training pipeline completed successfully. Best model: %s [target_mode=%s]",
        comparison_result.best_model_name,
        target_mode,
    )
    return comparison_result

