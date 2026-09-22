"""Deterministic unit tests for ModelRegistry.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 2A tests verifying model instantiation, deterministic random states,
model normalization, and rejection of unknown model types.
"""

import pytest
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from contracts.exceptions import PredictionError
from src.prediction.model_registry import (
    DEFAULT_RANDOM_STATE,
    ModelRegistry,
    normalize_model_name,
)


def test_supported_models_list() -> None:
    """Ensure registry defines the expected set of baseline regression models."""
    supported = ModelRegistry.get_supported_models()
    expected = ["linear_regression", "random_forest", "hist_gradient_boosting"]
    assert sorted(supported) == sorted(expected)


def test_artifact_filenames() -> None:
    """Ensure standard artifact filenames map correctly for all models."""
    assert ModelRegistry.get_artifact_filename("linear_regression") == "linear_regression.pkl"
    assert ModelRegistry.get_artifact_filename("random_forest") == "random_forest.pkl"
    assert ModelRegistry.get_artifact_filename("hist_gradient_boosting") == "hist_gradient_boosting.pkl"


def test_artifact_filename_invalid_model_raises_error() -> None:
    """Ensure querying artifact filename for unknown model raises PredictionError."""
    with pytest.raises(PredictionError) as exc_info:
        ModelRegistry.get_artifact_filename("quantum_flux_regressor")
    assert "Unknown model name" in str(exc_info.value)


def test_model_name_normalization() -> None:
    """Ensure alias strings normalize to canonical snake_case identifiers."""
    assert normalize_model_name("LinearRegression") == "linear_regression"
    assert normalize_model_name("RandomForestRegressor") == "random_forest"
    assert normalize_model_name("RF") == "random_forest"
    assert normalize_model_name("HistGradientBoostingRegressor") == "hist_gradient_boosting"
    assert normalize_model_name("HGBT") == "hist_gradient_boosting"


def test_model_instantiation() -> None:
    """Verify correct scikit-learn regressor types are constructed."""
    registry = ModelRegistry(random_state=42)

    lr = registry.create_model("linear_regression")
    assert isinstance(lr, LinearRegression)

    rf = registry.create_model("random_forest")
    assert isinstance(rf, RandomForestRegressor)
    assert rf.random_state == 42

    hgbt = registry.create_model("hist_gradient_boosting")
    assert isinstance(hgbt, HistGradientBoostingRegressor)
    assert hgbt.random_state == 42


def test_create_all_models() -> None:
    """Ensure create_all_models returns all three supported regressor instances."""
    registry = ModelRegistry()
    models = registry.create_all_models()
    assert len(models) == 3
    assert "linear_regression" in models
    assert "random_forest" in models
    assert "hist_gradient_boosting" in models


def test_unknown_model_raises_prediction_error() -> None:
    """Ensure requesting an unsupported model raises PredictionError."""
    registry = ModelRegistry()
    with pytest.raises(PredictionError) as exc_info:
        registry.create_model("unsupported_neural_net")
    assert "is not registered" in str(exc_info.value)
