"""Model registry for baseline maritime fuel consumption regressors.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 2A baseline model factory providing deterministic instantiation and hyperparameter
management for LinearRegression, RandomForest, and HistGradientBoosting regressors.
"""

from collections.abc import Callable
import logging
from typing import Any, Final

from sklearn.base import RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from contracts.exceptions import PredictionError

logger = logging.getLogger("maritime_system")

DEFAULT_RANDOM_STATE: Final[int] = 42

# Canonical model identifiers mapped to human-readable filenames
MODEL_FILENAME_MAP: Final[dict[str, str]] = {
    "linear_regression": "linear_regression.pkl",
    "random_forest": "random_forest.pkl",
    "hist_gradient_boosting": "hist_gradient_boosting.pkl",
    "qifcp": "qifcp.pkl",
}


def normalize_model_name(name: str) -> str:
    """Normalize model identifier strings into canonical lowercase snake_case format.

    Args:
        name: Input model identifier (e.g. 'LinearRegression', 'random_forest').

    Returns:
        Canonical snake_case model identifier.
    """
    cleaned = name.strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "linearregression": "linear_regression",
        "linear_regression": "linear_regression",
        "linear": "linear_regression",
        "randomforestregressor": "random_forest",
        "random_forest": "random_forest",
        "randomforest": "random_forest",
        "rf": "random_forest",
        "histgradientboostingregressor": "hist_gradient_boosting",
        "hist_gradient_boosting": "hist_gradient_boosting",
        "histgradientboosting": "hist_gradient_boosting",
        "hgbt": "hist_gradient_boosting",
        "hist_gbdt": "hist_gradient_boosting",
        "qifcp": "qifcp",
        "quantum_inspired": "qifcp",
        "quantum": "qifcp",
        "qnn": "qifcp",
    }
    return mapping.get(cleaned, cleaned)


class ModelRegistry:
    """Factory and catalog for deterministic baseline prediction regressors."""

    def __init__(self, random_state: int = DEFAULT_RANDOM_STATE) -> None:
        """Initialize the model registry with a default random seed.

        Args:
            random_state: Seed for deterministic model initialization.
        """
        self.random_state = random_state
        self.logger = logger
        self._builders: dict[str, Callable[..., RegressorMixin]] = {
            "linear_regression": self._build_linear_regression,
            "random_forest": self._build_random_forest,
            "hist_gradient_boosting": self._build_hist_gradient_boosting,
            "qifcp": self._build_qifcp,
        }

    @staticmethod
    def get_supported_models(include_all: bool = False) -> list[str]:
        """Return list of canonical identifiers for registered models.

        Args:
            include_all: If True, returns all models including quantum estimators.
                         If False (default), returns the standard baseline models.

        Returns:
            List of supported model key strings.
        """
        if include_all:
            return list(MODEL_FILENAME_MAP.keys())
        return ["linear_regression", "random_forest", "hist_gradient_boosting"]

    @staticmethod
    def get_artifact_filename(model_name: str) -> str:
        """Retrieve standard persistence filename for a given model.

        Args:
            model_name: Canonical or raw model identifier.

        Returns:
            Standard pickle filename (e.g. 'linear_regression.pkl').

        Raises:
            PredictionError: If model_name is not registered.
        """
        norm_name = normalize_model_name(model_name)
        if norm_name not in MODEL_FILENAME_MAP:
            raise PredictionError(
                f"Unknown model name '{model_name}'. Supported models: {list(MODEL_FILENAME_MAP.keys())}",
                details={"model_name": model_name},
            )
        return MODEL_FILENAME_MAP[norm_name]

    def _build_linear_regression(self, **kwargs: Any) -> LinearRegression:
        """Instantiate an Ordinary Least Squares LinearRegression model."""
        params = {"fit_intercept": True, "copy_X": True}
        params.update(kwargs)
        return LinearRegression(**params)

    def _build_random_forest(self, **kwargs: Any) -> RandomForestRegressor:
        """Instantiate a deterministic RandomForestRegressor model."""
        params = {
            "n_estimators": 100,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": self.random_state,
            "n_jobs": -1,
        }
        params.update(kwargs)
        return RandomForestRegressor(**params)

    def _build_hist_gradient_boosting(
        self, **kwargs: Any
    ) -> HistGradientBoostingRegressor:
        """Instantiate a deterministic HistGradientBoostingRegressor model."""
        params = {
            "max_iter": 100,
            "max_depth": 10,
            "min_samples_leaf": 20,
            "learning_rate": 0.1,
            "random_state": self.random_state,
        }
        params.update(kwargs)
        return HistGradientBoostingRegressor(**params)

    def _build_qifcp(self, **kwargs: Any) -> RegressorMixin:
        """Instantiate a Quantum-Inspired Fuel Consumption Predictor."""
        from src.prediction.qifcp import QIFCPRegressor

        params = {
            "gamma": 0.5,
            "alpha_reg": 1.0,
            "n_entanglement_pairs": 15,
            "random_state": self.random_state,
        }
        params.update(kwargs)
        return QIFCPRegressor(**params)

    def create_model(
        self,
        model_name: str,
        **kwargs: Any,
    ) -> RegressorMixin:
        """Instantiate a configured regressor instance by model identifier.

        Args:
            model_name: Identifier of the regressor to instantiate.
            **kwargs: Hyperparameter overrides.

        Returns:
            Configured scikit-learn regressor instance.

        Raises:
            PredictionError: If model_name is not supported.
        """
        norm_name = normalize_model_name(model_name)
        builder = self._builders.get(norm_name)
        if builder is None:
            raise PredictionError(
                f"Model '{model_name}' (normalized: '{norm_name}') is not registered. "
                f"Supported: {self.get_supported_models()}",
                details={"model_name": model_name, "supported": self.get_supported_models()},
            )
        model = builder(**kwargs)
        self.logger.debug("Created model instance for '%s'", norm_name)
        return model

    def create_all_models(
        self,
        **kwargs: Any,
    ) -> dict[str, RegressorMixin]:
        """Instantiate instances of all registered baseline models.

        Args:
            **kwargs: Hyperparameter overrides applied where applicable.

        Returns:
            Dictionary mapping canonical model names to regressor instances.
        """
        return {
            name: self.create_model(name, **kwargs)
            for name in self.get_supported_models()
        }
