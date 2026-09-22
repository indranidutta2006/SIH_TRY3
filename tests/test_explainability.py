"""Deterministic unit tests for PredictionExplainer.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 2B tests verifying feature importance extraction for RandomForest and HistGradientBoosting,
global importance rankings, local voyage driver attribution, and mathematical normalization.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from contracts.schemas import VoyageRecord
from src.prediction.explainability import PredictionExplainer


@pytest.fixture
def synthetic_training_data() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Fixture providing synthetic features and target where feature_0 dominates."""
    np.random.seed(42)
    n_samples = 100
    feature_names = ["speed_knots", "distance_nm", "cargo_tons", "weather_factor"]

    X_data = {
        "speed_knots": np.random.uniform(10.0, 20.0, n_samples),
        "distance_nm": np.random.uniform(500.0, 5000.0, n_samples),
        "cargo_tons": np.random.uniform(10000.0, 100000.0, n_samples),
        "weather_factor": np.random.uniform(1.0, 1.4, n_samples),
    }
    X = pd.DataFrame(X_data)
    # Fuel has strong cubic dependency on speed and linear on distance
    y = pd.Series(
        (X["speed_knots"] ** 3.0) * 0.1 + (X["distance_nm"] * 0.05),
        name="fuel_consumption",
    )
    return X, y, feature_names


def test_random_forest_feature_importances(
    synthetic_training_data: tuple[pd.DataFrame, pd.Series, list[str]],
) -> None:
    """Verify RandomForestRegressor feature importances are extracted and sum to 1.0."""
    X, y, feature_names = synthetic_training_data
    rf = RandomForestRegressor(n_estimators=30, random_state=42)
    rf.fit(X, y)

    explainer = PredictionExplainer(rf, feature_names=feature_names)
    importances = explainer.get_feature_importances()

    assert len(importances) == len(feature_names)
    assert pytest.approx(sum(importances.values()), abs=1e-5) == 1.0
    assert all(val >= 0.0 for val in importances.values())


def test_hist_gradient_boosting_importances_tree_nodes(
    synthetic_training_data: tuple[pd.DataFrame, pd.Series, list[str]],
) -> None:
    """Verify HistGradientBoostingRegressor feature importances via internal tree nodes."""
    X, y, feature_names = synthetic_training_data
    hgbt = HistGradientBoostingRegressor(max_iter=30, random_state=42)
    hgbt.fit(X, y)

    explainer = PredictionExplainer(hgbt, feature_names=feature_names)
    importances = explainer.get_feature_importances()

    assert len(importances) == len(feature_names)
    assert pytest.approx(sum(importances.values()), abs=1e-5) == 1.0
    assert all(val >= 0.0 for val in importances.values())


def test_hist_gradient_boosting_permutation_importance(
    synthetic_training_data: tuple[pd.DataFrame, pd.Series, list[str]],
) -> None:
    """Verify HistGradientBoostingRegressor feature importances using permutation importance."""
    X, y, feature_names = synthetic_training_data
    hgbt = HistGradientBoostingRegressor(max_iter=30, random_state=42)
    hgbt.fit(X, y)

    explainer = PredictionExplainer(
        hgbt, feature_names=feature_names, background_X=X, background_y=y
    )
    importances = explainer.get_feature_importances(X=X, y=y)

    assert len(importances) == len(feature_names)
    assert pytest.approx(sum(importances.values()), abs=1e-5) == 1.0
    # speed_knots should be dominant due to cubic relationship
    assert importances["speed_knots"] > importances["weather_factor"]


def test_global_explanation_ranking(
    synthetic_training_data: tuple[pd.DataFrame, pd.Series, list[str]],
) -> None:
    """Verify global explanation ranks features descending by importance."""
    X, y, feature_names = synthetic_training_data
    rf = RandomForestRegressor(n_estimators=30, random_state=42)
    rf.fit(X, y)

    explainer = PredictionExplainer(rf, feature_names=feature_names)
    report = explainer.get_global_explanation(top_k=3)

    assert len(report) == 3
    assert report[0]["rank"] == 1
    assert report[1]["rank"] == 2
    assert report[2]["rank"] == 3
    assert report[0]["importance"] >= report[1]["importance"]
    assert report[1]["importance"] >= report[2]["importance"]


def test_local_explanation_structure(
    synthetic_training_data: tuple[pd.DataFrame, pd.Series, list[str]],
) -> None:
    """Verify structure and content of local explanation dictionary."""
    X, y, feature_names = synthetic_training_data
    rf = RandomForestRegressor(n_estimators=30, random_state=42)
    rf.fit(X, y)

    explainer = PredictionExplainer(rf, feature_names=feature_names, background_X=X)
    voyage_features = pd.DataFrame(
        [
            {
                "speed_knots": 19.5,  # High speed -> positive driver
                "distance_nm": 600.0,  # Low distance -> negative driver
                "cargo_tons": 50000.0,
                "weather_factor": 1.05,
            }
        ]
    )

    explanation = explainer.explain_voyage(voyage_features)

    assert "top_positive_drivers" in explanation
    assert "top_negative_drivers" in explanation
    assert "dominant_factor" in explanation
    assert isinstance(explanation["top_positive_drivers"], list)
    assert isinstance(explanation["top_negative_drivers"], list)
    assert explanation["dominant_factor"] in feature_names


def test_local_explanation_with_voyage_record() -> None:
    """Verify local explanation directly on a VoyageRecord instance."""
    from src.prediction.model_registry import ModelRegistry

    # Train a fast model on minimal dummy dataset with pipeline features
    from src.ingestion.feature_pipeline import FeatureEngineeringPipeline

    pipeline = FeatureEngineeringPipeline()
    sample_voyage = VoyageRecord(
        voyage_id="VY-EXP-001",
        vessel_id="VSL-001",
        vessel_type="Bulk Carrier",
        vessel_dwt=100000.0,
        cargo_tons=80000.0,
        distance_nm=2500.0,
        speed_knots=16.0,
        hours_at_sea=156.25,
        fuel_type="Diesel",
        weather_factor=1.2,
        sea_state=4,
        data_source="mock",
        is_synthetic=True,
        fuel_consumption=600.0,
        co2_emissions=1923.6,
    )
    records = [sample_voyage, sample_voyage]
    X, y = pipeline.get_training_features_and_target(records)

    rf = RandomForestRegressor(n_estimators=10, random_state=42).fit(X, y)
    explainer = PredictionExplainer(
        rf, feature_names=list(X.columns), background_X=X, feature_pipeline=pipeline
    )

    explanation = explainer.explain_voyage(sample_voyage)
    assert "dominant_factor" in explanation
    assert explanation["dominant_factor"] in X.columns


def test_deterministic_output(
    synthetic_training_data: tuple[pd.DataFrame, pd.Series, list[str]],
) -> None:
    """Ensure explainer yields deterministic outputs across multiple invocations."""
    X, y, feature_names = synthetic_training_data
    rf = RandomForestRegressor(n_estimators=20, random_state=42)
    rf.fit(X, y)

    explainer1 = PredictionExplainer(rf, feature_names=feature_names)
    explainer2 = PredictionExplainer(rf, feature_names=feature_names)

    assert explainer1.get_feature_importances() == explainer2.get_feature_importances()


def test_uniform_fallback_on_zero_importances() -> None:
    """Verify fallback normalization when all raw feature importances are zero."""

    class DummyModel:
        feature_importances_ = [0.0, 0.0, 0.0]

    explainer = PredictionExplainer(DummyModel(), feature_names=["f1", "f2", "f3"])
    importances = explainer.get_feature_importances()

    assert pytest.approx(sum(importances.values()), abs=1e-5) == 1.0
    assert pytest.approx(importances["f1"], rel=1e-3) == 1.0 / 3.0
