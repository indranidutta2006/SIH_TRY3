"""Prediction inference engine for deployed maritime fuel consumption models.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 2A inference module generating predictions from persisted baseline models,
validating feature schema integrity, and enforcing runtime target leakage guardrails.
"""

from collections.abc import Sequence
import logging
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin

from contracts.exceptions import DataValidationError, PredictionError
from contracts.interfaces import PredictionEngine
from contracts.schemas import PredictionResult, VoyageRecord
from src.ingestion.feature_pipeline import (
    FORBIDDEN_LEAKAGE_COLUMNS,
    FeatureEngineeringPipeline,
)
from src.prediction.metrics import evaluate_predictions
from src.prediction.model_registry import normalize_model_name

logger = logging.getLogger("maritime_system")


class PredictionInferenceEngine(PredictionEngine):
    """Inference engine managing model loading, feature schema validation, and prediction generation."""

    def __init__(
        self,
        model: RegressorMixin | None = None,
        model_name: str = "custom_model",
        feature_columns: list[str] | None = None,
        feature_pipeline: FeatureEngineeringPipeline | None = None,
    ) -> None:
        """Initialize inference engine with model and optional feature expectations.

        Args:
            model: Fitted scikit-learn regressor. If None, must be loaded via load_model.
            model_name: Identifier for the underlying model.
            feature_columns: Expected feature names and ordering required by model.
            feature_pipeline: Pipeline used to generate features from VoyageRecord sequences.
        """
        self.model = model
        self.model_name = normalize_model_name(model_name)
        self.feature_columns = feature_columns
        self.pipeline = feature_pipeline or FeatureEngineeringPipeline()
        self.logger = logger

    @classmethod
    def load_from_artifact(cls, artifact_path: Path | str) -> "PredictionInferenceEngine":
        """Factory constructor instantiating engine from a serialized model artifact.

        Args:
            artifact_path: File path to persisted .pkl / .joblib model bundle.

        Returns:
            Configured PredictionInferenceEngine instance.

        Raises:
            PredictionError: If artifact file does not exist or payload is invalid.
        """
        path = Path(artifact_path)
        if not path.exists():
            raise PredictionError(
                f"Model artifact file does not exist: {path.resolve()}",
                details={"artifact_path": str(path)},
            )

        try:
            payload = joblib.load(path)
        except Exception as err:
            raise PredictionError(
                f"Failed to load model artifact from {path.resolve()}: {err}",
                details={"artifact_path": str(path), "error": str(err)},
            ) from err

        if isinstance(payload, dict) and "model" in payload:
            model = payload["model"]
            model_name = payload.get("model_name", path.stem)
            feature_columns = payload.get("feature_columns")
        elif hasattr(payload, "predict"):
            model = payload
            model_name = path.stem
            feature_columns = None
        else:
            raise PredictionError(
                f"Loaded artifact from {path.resolve()} is not a recognized model or bundle.",
                details={"artifact_path": str(path)},
            )

        logger.info("Loaded model '%s' from %s", model_name, path.name)
        return cls(
            model=model,
            model_name=model_name,
            feature_columns=feature_columns,
        )

    def validate_feature_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate feature matrix against leakage, null values, and column schema.

        Args:
            X: Input feature DataFrame.

        Returns:
            Validated and column-aligned DataFrame ready for model predict call.

        Raises:
            DataValidationError: If target leakage is present.
            PredictionError: If required features are missing or invalid.
        """
        if X.empty:
            raise PredictionError("Cannot run inference on empty feature matrix.")

        # 1. Target Leakage Guardrails
        self.pipeline.check_for_target_leakage(X)

        # 2. Schema / Feature Column Compatibility Check
        if self.feature_columns:
            missing_cols = set(self.feature_columns) - set(X.columns)
            if missing_cols:
                raise PredictionError(
                    f"Feature compatibility error: missing required feature columns: {sorted(missing_cols)}",
                    details={"missing_columns": sorted(missing_cols)},
                )
            # Reorder columns to match training distribution exactly
            aligned_X = X[self.feature_columns].copy()
        else:
            aligned_X = X.copy()

        # 3. Non-nullability check for numeric predictors
        if aligned_X.isna().any().any():
            null_cols = aligned_X.columns[aligned_X.isna().any()].tolist()
            raise PredictionError(
                f"Feature matrix contains null values in columns: {null_cols}",
                details={"null_columns": null_cols},
            )

        return aligned_X

    def predict_df(self, X: pd.DataFrame) -> np.ndarray:
        """Generate raw numeric predictions for a prepared feature DataFrame.

        Args:
            X: Preprocessed feature DataFrame.

        Returns:
            1-dimensional numpy array of predicted fuel consumption (metric tons).

        Raises:
            PredictionError: If model is not loaded or inference fails.
        """
        if self.model is None:
            raise PredictionError(
                "Prediction model is uninitialized. Call load_from_artifact() or train() first."
            )

        valid_X = self.validate_feature_matrix(X)
        try:
            raw_preds = self.model.predict(valid_X)
            preds = np.asarray(raw_preds, dtype=float)
            if preds.ndim > 1:
                preds = preds.ravel()
            return preds
        except Exception as err:
            raise PredictionError(
                f"Inference execution failed on model '{self.model_name}': {err}",
                details={"model_name": self.model_name, "error": str(err)},
            ) from err

    def predict(
        self, feature_records: Sequence[VoyageRecord]
    ) -> list[PredictionResult]:
        """Generate structured PredictionResult instances for input voyage telemetry.

        Fulfills the abstract contract in contracts.interfaces.PredictionEngine.

        Args:
            feature_records: Sequence of VoyageRecord instances for inference.

        Returns:
            List of structured PredictionResult instances.

        Raises:
            PredictionError: If inference fails or records are empty.
        """
        if not feature_records:
            raise PredictionError("Cannot generate predictions on empty voyage sequence.")

        start_time = time.perf_counter()

        # Extract features using inference pipeline (handles fuel_consumption=None)
        X = self.pipeline.get_inference_features(
            feature_records, encode_categoricals=True
        )

        preds = self.predict_df(X)
        elapsed_total = time.perf_counter() - start_time
        per_record_runtime = elapsed_total / max(len(feature_records), 1)

        results: list[PredictionResult] = []
        for pred in preds:
            # Physical boundary: fuel consumption is strictly positive
            bounded_pred = float(max(pred, 0.0))
            results.append(
                PredictionResult(
                    model_name=self.model_name,
                    predicted_fuel_consumption=round(bounded_pred, 4),
                    # Nominal schema placeholder preserved for PredictionResult contract compatibility.
                    # Statistical confidence is evaluated via empirical R2, RMSE, MAE, MAPE, and NRMSE metrics.
                    confidence_score=0.95,
                    runtime_seconds=round(per_record_runtime, 6),
                )
            )

        return results

    def train(
        self,
        training_records: Sequence[VoyageRecord],
        target_field: str = "fuel_consumption",
    ) -> None:
        """Fit model using voyage records conforming to contracts.interfaces.PredictionEngine.

        Args:
            training_records: Sequence of labeled VoyageRecord instances.
            target_field: Target field name (must be 'fuel_consumption').

        Raises:
            PredictionError: If training fails or target_field is invalid.
        """
        if target_field != "fuel_consumption":
            raise PredictionError(
                f"Unsupported target_field '{target_field}'. Only 'fuel_consumption' is supported."
            )

        X, y = self.pipeline.get_training_features_and_target(
            training_records, encode_categoricals=True
        )
        self.feature_columns = list(X.columns)

        if self.model is None:
            # Default to HistGradientBoostingRegressor if model unassigned
            from src.prediction.model_registry import ModelRegistry

            self.model = ModelRegistry().create_model("hist_gradient_boosting")
            self.model_name = "hist_gradient_boosting"

        self.model.fit(X, y)
        self.logger.info("Fitted model '%s' on %d records.", self.model_name, len(training_records))

    def evaluate(self, evaluation_records: Sequence[VoyageRecord]) -> dict[str, float]:
        """Calculate regression metrics conforming to contracts.interfaces.PredictionEngine.

        Args:
            evaluation_records: Sequence of VoyageRecord instances with ground truth fuel_consumption.

        Returns:
            Dictionary containing 'rmse', 'mae', 'mape', and 'r2'.

        Raises:
            PredictionError: If ground truth is missing or evaluation fails.
        """
        if not evaluation_records:
            raise PredictionError("Cannot evaluate on empty evaluation records.")

        X, y_true = self.pipeline.get_training_features_and_target(
            evaluation_records, encode_categoricals=True
        )
        preds = self.predict_df(X)
        metrics = evaluate_predictions(y_true, preds, model_name=self.model_name)
        return {
            "rmse": metrics.rmse,
            "mae": metrics.mae,
            "mape": metrics.mape,
            "r2": metrics.r2,
        }
