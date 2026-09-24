"""Deterministic integration tests for baseline prediction pipeline, inference engine, and artifacts.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 2A tests verifying end-to-end training, model serialization to artifacts/,
inference fidelity, feature compatibility validation, and target leakage rejection.
"""

import json
from pathlib import Path
import pytest

from contracts.exceptions import DataValidationError, PredictionError
from contracts.interfaces import PredictionEngine
from contracts.schemas import PredictionResult, VoyageRecord
from src.ingestion.dataset_loader import CSVDatasetLoader
from src.ingestion.feature_pipeline import FeatureEngineeringPipeline
from src.prediction.metrics import ModelComparisonResult
from src.prediction.predictor import PredictionInferenceEngine
from src.prediction.trainer import PredictionTrainer, train_all_models


@pytest.fixture
def mini_dataset_records() -> list[VoyageRecord]:
    """Fixture providing a small set of valid VoyageRecord instances for fast integration tests."""
    records = []
    vessel_types = ["Bulk Carrier", "Container Ship", "Oil Tanker", "General Cargo"]
    fuel_types = ["Diesel", "LNG", "Methanol"]

    for i in range(40):
        v_type = vessel_types[i % len(vessel_types)]
        f_type = fuel_types[i % len(fuel_types)]
        dwt = 50000.0 + (i * 2000.0)
        cargo = dwt * 0.8
        dist = 1000.0 + (i * 100.0)
        speed = 12.0 + (i % 5)
        hours = dist / speed
        fuel = 100.0 + (cargo * 0.001 * dist * 0.002) + (speed ** 2.5 * 0.1)
        co2 = fuel * 3.114

        records.append(
            VoyageRecord(
                voyage_id=f"VY-MINI-{i:04d}",
                vessel_id=f"VSL-{i % 5:03d}",
                vessel_type=v_type,
                vessel_dwt=dwt,
                cargo_tons=cargo,
                distance_nm=dist,
                speed_knots=speed,
                hours_at_sea=hours,
                fuel_type=f_type,
                weather_factor=1.05 + ((i % 4) * 0.05),
                sea_state=i % 6,
                data_source="mock",
                is_synthetic=True,
                fuel_consumption=fuel,
                co2_emissions=co2,
            )
        )
    return records


def test_prediction_inference_engine_conforms_to_interface() -> None:
    """Verify PredictionInferenceEngine implements the PredictionEngine abstract contract."""
    engine = PredictionInferenceEngine()
    assert isinstance(engine, PredictionEngine)
    assert hasattr(engine, "train")
    assert hasattr(engine, "predict")
    assert hasattr(engine, "evaluate")


def test_end_to_end_training_and_artifact_generation(
    tmp_path: Path, mini_dataset_records: list[VoyageRecord]
) -> None:
    """Verify PredictionTrainer fits models, generates metrics, and persists artifacts."""
    pipeline = FeatureEngineeringPipeline()
    X, y = pipeline.get_training_features_and_target(
        mini_dataset_records, encode_categoricals=True
    )

    trainer = PredictionTrainer(test_size=0.25, random_state=42)
    trained_models, comparison = trainer.train_and_evaluate(X, y)

    assert len(trained_models) == 3
    assert comparison.best_model_name in trained_models
    assert len(comparison.metrics_by_model) == 3

    # Persist artifacts into tmp_path
    artifacts_map = trainer.save_artifacts(
        trained_models=trained_models,
        comparison_result=comparison,
        artifacts_dir=tmp_path,
        feature_columns=list(X.columns),
    )

    # Check that required model files exist
    assert (tmp_path / "models" / "linear_regression.pkl").exists()
    assert (tmp_path / "models" / "random_forest.pkl").exists()
    assert (tmp_path / "models" / "hist_gradient_boosting.pkl").exists()

    # Check metrics JSON
    metrics_file = tmp_path / "metrics" / "baseline_metrics.json"
    assert metrics_file.exists()
    metrics_data = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert "best_model_name" in metrics_data
    assert "metrics_by_model" in metrics_data


def test_inference_engine_loads_and_predicts(
    tmp_path: Path, mini_dataset_records: list[VoyageRecord]
) -> None:
    """Verify loading persisted model and generating structured PredictionResult instances."""
    pipeline = FeatureEngineeringPipeline()
    X, y = pipeline.get_training_features_and_target(
        mini_dataset_records, encode_categoricals=True
    )

    trainer = PredictionTrainer(random_state=42)
    trained_models, comparison = trainer.train_and_evaluate(X, y)
    trainer.save_artifacts(trained_models, comparison, artifacts_dir=tmp_path, feature_columns=list(X.columns))

    model_path = tmp_path / "models" / "linear_regression.pkl"
    engine = PredictionInferenceEngine.load_from_artifact(model_path)

    # Unlabelled inference voyage record (fuel_consumption=None)
    unlabelled = [
        VoyageRecord(
            voyage_id="VY-INF-001",
            vessel_id="VSL-INF",
            vessel_type="Bulk Carrier",
            vessel_dwt=70000.0,
            cargo_tons=55000.0,
            distance_nm=2500.0,
            speed_knots=13.5,
            hours_at_sea=185.18,
            fuel_type="Diesel",
            weather_factor=1.1,
            sea_state=3,
            data_source="mock",
            is_synthetic=True,
            fuel_consumption=None,
            co2_emissions=None,
        )
    ]

    results = engine.predict(unlabelled)
    assert len(results) == 1
    res = results[0]
    assert isinstance(res, PredictionResult)
    assert res.predicted_fuel_consumption > 0.0
    assert res.confidence_score > 0.0
    assert res.runtime_seconds >= 0.0


def test_feature_compatibility_validation() -> None:
    """Ensure inference engine rejects feature sets missing required columns."""
    engine = PredictionInferenceEngine(
        model=None,
        feature_columns=["vessel_dwt", "cargo_tons", "distance_nm"],
    )

    import pandas as pd

    incomplete_df = pd.DataFrame({"vessel_dwt": [50000.0], "cargo_tons": [40000.0]})
    with pytest.raises(PredictionError) as exc_info:
        engine.validate_feature_matrix(incomplete_df)

    assert "missing required feature columns" in str(exc_info.value)


def test_inference_rejects_target_leakage() -> None:
    """Ensure inference engine strictly rejects feature sets containing target leakage."""
    engine = PredictionInferenceEngine()

    import pandas as pd

    leaking_df = pd.DataFrame(
        {
            "vessel_dwt": [50000.0],
            "fuel_consumption": [200.0],  # Direct target leakage
        }
    )
    with pytest.raises(DataValidationError) as exc_info:
        engine.validate_feature_matrix(leaking_df)

    assert "Target leakage violation" in str(exc_info.value)


def test_train_all_models_convenience_function(
    tmp_path: Path, mini_dataset_records: list[VoyageRecord]
) -> None:
    """Verify train_all_models runs end-to-end and returns ModelComparisonResult."""
    # Write mini dataset to CSV
    csv_path = tmp_path / "mini_voyages.csv"
    import csv
    from dataclasses import asdict

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(mini_dataset_records[0]).keys()))
        writer.writeheader()
        for rec in mini_dataset_records:
            writer.writerow(asdict(rec))

    artifacts_dir = tmp_path / "artifacts"
    comparison = train_all_models(
        dataset_path=csv_path,
        artifacts_dir=artifacts_dir,
        test_size=0.2,
        random_state=42,
    )

    assert isinstance(comparison, ModelComparisonResult)
    assert comparison.best_model_name in {"linear_regression", "random_forest", "hist_gradient_boosting"}
    assert (artifacts_dir / "models" / "linear_regression.pkl").exists()
    assert (artifacts_dir / "models" / "random_forest.pkl").exists()
    assert (artifacts_dir / "models" / "hist_gradient_boosting.pkl").exists()
    assert (artifacts_dir / "metrics" / "baseline_metrics.json").exists()


def test_train_all_models_rate_mode(
    tmp_path: Path, mini_dataset_records: list[VoyageRecord]
) -> None:
    """Verify train_all_models with target_mode='rate' persists rate metadata and reconstructs absolute fuel."""
    import csv
    from dataclasses import asdict

    csv_path = tmp_path / "mini_voyages_rate.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(mini_dataset_records[0]).keys()))
        writer.writeheader()
        for rec in mini_dataset_records:
            writer.writerow(asdict(rec))

    artifacts_dir = tmp_path / "artifacts_rate"
    comparison = train_all_models(
        dataset_path=csv_path,
        artifacts_dir=artifacts_dir,
        test_size=0.2,
        random_state=42,
        target_mode="rate",
    )

    assert isinstance(comparison, ModelComparisonResult)
    metrics_file = artifacts_dir / "metrics" / "baseline_metrics.json"
    assert metrics_file.exists()
    metrics_data = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert metrics_data["training_config"]["target_mode"] == "rate"

    # Test loading and predicting via PredictionInferenceEngine
    best_model_path = artifacts_dir / "models" / f"{comparison.best_model_name}.pkl"
    engine = PredictionInferenceEngine.load_from_artifact(best_model_path)
    assert engine.target_mode == "rate"

    predictions = engine.predict(mini_dataset_records[:5])
    assert len(predictions) == 5
    for p in predictions:
        assert p.predicted_fuel_consumption > 0.0
        assert p.confidence_score == 0.95


def test_train_all_models_invalid_target_mode(tmp_path: Path) -> None:
    """Verify train_all_models rejects invalid target_mode."""
    with pytest.raises(ValueError, match="target_mode must be 'absolute' or 'rate'"):
        train_all_models(dataset_path=tmp_path / "nonexistent.csv", target_mode="unsupported")

