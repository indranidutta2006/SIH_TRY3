"""Statistical and regression evaluation metrics for fuel prediction models.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 2A evaluation module computing MAE, RMSE, MAPE, and R2 with zero-division protection
and structured model comparison reporting.
"""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from contracts.exceptions import PredictionError

logger = logging.getLogger("maritime_system")


@dataclass(frozen=True, slots=True)
class ModelMetrics:
    """Regression performance evaluation metrics for a single predictive model."""

    model_name: str
    mae: float
    rmse: float
    mape: float
    r2: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize dataclass to JSON formatted string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass(frozen=True, slots=True)
class ModelComparisonResult:
    """Comparative benchmarking results across multiple trained models."""

    best_model_name: str
    metrics_by_model: dict[str, ModelMetrics]
    training_timestamp: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize comparative summary to dictionary."""
        return {
            "best_model_name": self.best_model_name,
            "metrics_by_model": {
                name: m.to_dict() for name, m in self.metrics_by_model.items()
            },
            "training_timestamp": self.training_timestamp,
        }

    def to_json(self) -> str:
        """Serialize comparative summary to JSON formatted string."""
        return json.dumps(self.to_dict(), indent=2)


def _to_numpy_array(data: Sequence[float] | np.ndarray | pd.Series) -> np.ndarray:
    """Convert input sequence or pandas Series into a clean 1D float numpy array.

    Args:
        data: Ground truth or predicted values.

    Returns:
        1-dimensional float64 numpy array.

    Raises:
        PredictionError: If input is empty or contains non-numeric values.
    """
    if isinstance(data, (pd.Series, pd.DataFrame)):
        arr = data.to_numpy(dtype=float)
    else:
        arr = np.asarray(data, dtype=float)

    if arr.ndim > 1:
        arr = arr.ravel()

    if arr.size == 0:
        raise PredictionError("Cannot calculate metrics on an empty dataset.")

    return arr


def compute_mae(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    y_pred: Sequence[float] | np.ndarray | pd.Series,
) -> float:
    """Calculate Mean Absolute Error (MAE) between true and predicted fuel consumption.

    Args:
        y_true: Actual ground truth observations.
        y_pred: Predicted values from regression model.

    Returns:
        Computed MAE as float.
    """
    y_t = _to_numpy_array(y_true)
    y_p = _to_numpy_array(y_pred)
    if y_t.shape != y_p.shape:
        raise PredictionError(
            f"Shape mismatch: y_true shape {y_t.shape} != y_pred shape {y_p.shape}."
        )
    return float(np.mean(np.abs(y_t - y_p)))


def compute_rmse(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    y_pred: Sequence[float] | np.ndarray | pd.Series,
) -> float:
    """Calculate Root Mean Squared Error (RMSE).

    Args:
        y_true: Actual ground truth observations.
        y_pred: Predicted values from regression model.

    Returns:
        Computed RMSE as float.
    """
    y_t = _to_numpy_array(y_true)
    y_p = _to_numpy_array(y_pred)
    if y_t.shape != y_p.shape:
        raise PredictionError(
            f"Shape mismatch: y_true shape {y_t.shape} != y_pred shape {y_p.shape}."
        )
    mse = np.mean((y_t - y_p) ** 2)
    return float(np.sqrt(mse))


def compute_mape(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    y_pred: Sequence[float] | np.ndarray | pd.Series,
    epsilon: float = 1e-6,
) -> float:
    """Calculate Mean Absolute Percentage Error (MAPE) expressed as a percentage.

    Safely protects against division by zero using a floor threshold epsilon.

    Args:
        y_true: Actual ground truth observations.
        y_pred: Predicted values from regression model.
        epsilon: Floor threshold to prevent division by zero.

    Returns:
        Computed MAPE percentage (e.g. 4.5 for 4.5%).
    """
    y_t = _to_numpy_array(y_true)
    y_p = _to_numpy_array(y_pred)
    if y_t.shape != y_p.shape:
        raise PredictionError(
            f"Shape mismatch: y_true shape {y_t.shape} != y_pred shape {y_p.shape}."
        )
    denominator = np.where(np.abs(y_t) < epsilon, epsilon, np.abs(y_t))
    percentage_errors = np.abs((y_t - y_p) / denominator) * 100.0
    return float(np.mean(percentage_errors))


def compute_r2(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    y_pred: Sequence[float] | np.ndarray | pd.Series,
) -> float:
    """Calculate Coefficient of Determination (R^2).

    Args:
        y_true: Actual ground truth observations.
        y_pred: Predicted values from regression model.

    Returns:
        Computed R2 score as float (upper bounded at 1.0).
    """
    y_t = _to_numpy_array(y_true)
    y_p = _to_numpy_array(y_pred)
    if y_t.shape != y_p.shape:
        raise PredictionError(
            f"Shape mismatch: y_true shape {y_t.shape} != y_pred shape {y_p.shape}."
        )
    ss_res = np.sum((y_t - y_p) ** 2)
    mean_y = np.mean(y_t)
    ss_tot = np.sum((y_t - mean_y) ** 2)
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - (ss_res / ss_tot))


def evaluate_predictions(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    y_pred: Sequence[float] | np.ndarray | pd.Series,
    model_name: str,
) -> ModelMetrics:
    """Calculate comprehensive evaluation metrics for a single model's predictions.

    Args:
        y_true: Ground truth actuals.
        y_pred: Model predictions.
        model_name: Identifier for the evaluated model.

    Returns:
        ModelMetrics dataclass populated with MAE, RMSE, MAPE, and R2.
    """
    mae = compute_mae(y_true, y_pred)
    rmse = compute_rmse(y_true, y_pred)
    mape = compute_mape(y_true, y_pred)
    r2 = compute_r2(y_true, y_pred)

    metrics = ModelMetrics(
        model_name=model_name,
        mae=round(mae, 4),
        rmse=round(rmse, 4),
        mape=round(mape, 4),
        r2=round(r2, 4),
    )

    logger.info(
        "Model [%s] Evaluation: RMSE=%.4f | MAE=%.4f | MAPE=%.2f%% | R2=%.4f",
        model_name,
        rmse,
        mae,
        mape,
        r2,
    )
    return metrics


def select_best_model(
    metrics_by_model: dict[str, ModelMetrics],
    primary_metric: str = "rmse",
) -> str:
    """Select the best performing model based on actual empirical metrics.

    Args:
        metrics_by_model: Dictionary mapping model names to ModelMetrics.
        primary_metric: Metric to optimize ('rmse', 'mae', 'mape', or 'r2').

    Returns:
        Name of the best-performing model.

    Raises:
        PredictionError: If metrics dictionary is empty or metric is unsupported.
    """
    if not metrics_by_model:
        raise PredictionError("Cannot select best model from empty metrics dictionary.")

    metric_lower = primary_metric.strip().lower()
    if metric_lower in {"rmse", "mae", "mape"}:
        # Lower is better
        return min(
            metrics_by_model.keys(),
            key=lambda name: getattr(metrics_by_model[name], metric_lower),
        )
    elif metric_lower == "r2":
        # Higher is better
        return max(
            metrics_by_model.keys(),
            key=lambda name: getattr(metrics_by_model[name], "r2"),
        )
    else:
        raise PredictionError(
            f"Unsupported primary metric '{primary_metric}'. Must be 'rmse', 'mae', 'mape', or 'r2'."
        )
