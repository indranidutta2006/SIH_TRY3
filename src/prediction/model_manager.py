"""Production model manager for lifecycle tracking, selection, and inference deployment.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 2B production model manager discovering model artifacts, reading empirical metrics,
and deploying the best performing regressor dynamically based on baseline_metrics.json.
"""

import json
import logging
from pathlib import Path
from typing import Any

from contracts.exceptions import PredictionError
from src.prediction.metrics import ModelMetrics
from src.prediction.model_registry import (
    MODEL_FILENAME_MAP,
    ModelRegistry,
    normalize_model_name,
)
from src.prediction.predictor import PredictionInferenceEngine

logger = logging.getLogger("maritime_system")


class ProductionModelManager:
    """Discovers, evaluates, loads, and manages production-ready predictive regressors."""

    def __init__(
        self,
        artifacts_dir: Path | str = "artifacts",
    ) -> None:
        """Initialize the model manager with root artifacts location.

        Args:
            artifacts_dir: Root directory path containing models/ and metrics/ subdirectories.
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.models_dir = self.artifacts_dir / "models"
        self.metrics_dir = self.artifacts_dir / "metrics"
        self.metrics_file = self.metrics_dir / "baseline_metrics.json"
        self.logger = logger
        self._loaded_models: dict[str, PredictionInferenceEngine] = {}

    def _read_metrics_manifest(self) -> dict[str, Any]:
        """Read and parse the baseline metrics JSON manifest.

        Returns:
            Dictionary representation of metrics manifest.

        Raises:
            PredictionError: If metrics file does not exist or is malformed.
        """
        if not self.metrics_file.exists():
            raise PredictionError(
                f"Production metrics file does not exist: {self.metrics_file.resolve()}. "
                "Train baseline models first via train_all_models().",
                details={"metrics_file": str(self.metrics_file)},
            )
        try:
            return json.loads(self.metrics_file.read_text(encoding="utf-8"))
        except Exception as err:
            raise PredictionError(
                f"Failed to parse metrics manifest from {self.metrics_file.resolve()}: {err}",
                details={"metrics_file": str(self.metrics_file), "error": str(err)},
            ) from err

    def list_available_models(self) -> list[str]:
        """Discover and list all available model names with persisted binaries.

        Returns:
            List of canonical model names that have persisted .pkl artifacts.
        """
        if not self.models_dir.exists():
            return []

        available: list[str] = []
        for model_name, filename in MODEL_FILENAME_MAP.items():
            if (self.models_dir / filename).exists():
                available.append(model_name)
        return sorted(available)

    def get_model_metrics(
        self, model_name: str | None = None
    ) -> dict[str, ModelMetrics] | ModelMetrics:
        """Retrieve evaluation metrics for a specific model or all models.

        Args:
            model_name: Optional canonical or raw model identifier. If None, returns all.

        Returns:
            ModelMetrics instance or dictionary of ModelMetrics by model name.

        Raises:
            PredictionError: If requested model is not found in metrics.
        """
        manifest = self._read_metrics_manifest()
        metrics_raw = manifest.get("metrics_by_model", {})

        metrics_parsed = {
            name: ModelMetrics(
                model_name=m["model_name"],
                mae=float(m["mae"]),
                rmse=float(m["rmse"]),
                mape=float(m["mape"]),
                r2=float(m["r2"]),
            )
            for name, m in metrics_raw.items()
        }

        if model_name is None:
            return metrics_parsed

        norm_name = normalize_model_name(model_name)
        if norm_name not in metrics_parsed:
            raise PredictionError(
                f"Metrics for model '{model_name}' (normalized: '{norm_name}') not found. "
                f"Available in manifest: {list(metrics_parsed.keys())}",
                details={"model_name": model_name, "available": list(metrics_parsed.keys())},
            )
        return metrics_parsed[norm_name]

    def get_best_model_name(self) -> str:
        """Retrieve the canonical identifier of the best model recorded in metrics.

        Returns:
            Best model name string as audited in baseline_metrics.json.
        """
        manifest = self._read_metrics_manifest()
        best_name = manifest.get("best_model_name")
        if not best_name:
            raise PredictionError(
                "Metrics manifest does not specify 'best_model_name'.",
                details={"manifest": manifest},
            )
        return normalize_model_name(best_name)

    def load_model(self, model_name: str) -> PredictionInferenceEngine:
        """Load a specific model artifact by name and return its inference engine.

        Caches loaded engines in memory for fast repeated inference.

        Args:
            model_name: Identifier of model to load (e.g. 'hist_gradient_boosting').

        Returns:
            Configured PredictionInferenceEngine instance.

        Raises:
            PredictionError: If model file does not exist.
        """
        norm_name = normalize_model_name(model_name)
        if norm_name in self._loaded_models:
            return self._loaded_models[norm_name]

        filename = ModelRegistry.get_artifact_filename(norm_name)
        model_path = self.models_dir / filename

        if not model_path.exists():
            raise PredictionError(
                f"Model binary '{filename}' does not exist in {self.models_dir.resolve()}.",
                details={"model_name": norm_name, "model_path": str(model_path)},
            )

        engine = PredictionInferenceEngine.load_from_artifact(model_path)
        self._loaded_models[norm_name] = engine
        return engine

    def get_best_model(self) -> PredictionInferenceEngine:
        """Automatically load and return the best performing production model.

        Uses baseline_metrics.json as source of truth.

        Returns:
            PredictionInferenceEngine configured with the best benchmarked model.
        """
        best_name = self.get_best_model_name()
        self.logger.info("Auto-deploying best production model: '%s'", best_name)
        return self.load_model(best_name)
