"""Unit and regression tests for Out-of-Generator Stress Test & Distribution Shift Audit.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Verifies dataset integrity, cryptographic invariance of canonical training data, physical
plausibility of shifted validation sets, report schema compliance, and prediction reproducibility.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from contracts.schemas import VoyageRecord
from scripts.benchmark_prediction import run_qifcp_out_of_generator_stress_test
from scripts.make_mock_dataset import validate_record
from src.ingestion.dataset_loader import CSVDatasetLoader
from src.ingestion.feature_pipeline import FeatureEngineeringPipeline
from src.prediction.qifcp import (
    NavalPhysicsFuelBaseline,
    PhysicsInformedQIFCPRegressor,
)

CANONICAL_DATASET = Path("data/raw/voyages_sample.csv")
CANONICAL_SHA256 = "069dfc97f9c3b75ad5e903a772095151692aa6501d07c2c648175cfe27ddcd2c"
CANONICAL_SHA256_LF = "d9e5ae1879e85cbbc971770811fb0ed5b83b33e6ce93f630c00d7a8c1abfe5af"

SHIFTED_DATASETS = {
    "level1": {
        "path": Path("data/validation/out_of_generator_level1.csv"),
        "sha256": "aa4dd04aafb14e0140d1a8f0f358e758ab4ccb6389236f4fc5986184354cf9bd",
        "sha256_lf": "a2ce8c3b5b79a230380c03f25e00a1fb48e2be213b5c9f14f2bf7f5a7ca185e6",
        "rows": 500,
    },
    "level2": {
        "path": Path("data/validation/out_of_generator_level2.csv"),
        "sha256": "116caa0eef427e31d8b3615c9f1481e48c5ca53f51c53c6f10952a5b78940ca0",
        "sha256_lf": "5ec41821d06cc64dcc8836c94dcbe1bb8a5b20cfe205f3194eb6dde0b9008580",
        "rows": 500,
    },
    "level3": {
        "path": Path("data/validation/out_of_generator_level3.csv"),
        "sha256": "1cbb26fc6cef040fc5aea6cfc473c4052cbc82191ac07a0e80e6e2a52a3d2685",
        "sha256_lf": "7727a728fec3ebeb16d558b88ba2cb4e7dd31fd91da42c4fd2ec99d5ca917ea0",
        "rows": 500,
    },
}

REPORT_JSON = Path("outputs/reports/qifcp_out_of_generator_stress_test.json")
REPORT_MD = Path("outputs/reports/qifcp_out_of_generator_stress_test.md")


def test_canonical_dataset_unmodified() -> None:
    """Ensure canonical training dataset is bitwise identical to frozen baseline hash."""
    if not CANONICAL_DATASET.exists():
        pytest.skip("Canonical dataset not found in this environment")
    actual_hash = hashlib.sha256(CANONICAL_DATASET.read_bytes()).hexdigest()
    valid_hashes = {CANONICAL_SHA256, CANONICAL_SHA256_LF}
    assert actual_hash in valid_hashes, (
        f"Canonical dataset hash altered! Expected one of {valid_hashes}, got {actual_hash}"
    )


def test_shifted_validation_datasets_exist_and_match_hashes() -> None:
    """Ensure all 3 shifted validation datasets exist with exact expected hashes and row counts."""
    for s_id, s_info in SHIFTED_DATASETS.items():
        p = s_info["path"]
        assert p.exists(), f"Shifted dataset {s_id} does not exist at {p}"
        actual_hash = hashlib.sha256(p.read_bytes()).hexdigest()
        valid_hashes = {s_info["sha256"], s_info["sha256_lf"]}
        assert actual_hash in valid_hashes, (
            f"Hash mismatch for {s_id}: expected one of {valid_hashes}, got {actual_hash}"
        )
        # Check line count (500 data rows + 1 header)
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == s_info["rows"] + 1, (
            f"Row count mismatch for {s_id}: expected {s_info['rows'] + 1} lines, got {len(lines)}"
        )


def test_shifted_datasets_physical_plausibility_and_feature_pipeline() -> None:
    """Ensure shifted records pass domain validation and cleanly transform through feature pipeline."""
    loader = CSVDatasetLoader()
    pipeline = FeatureEngineeringPipeline()

    for s_id, s_info in SHIFTED_DATASETS.items():
        recs = loader.load_data(str(s_info["path"]))
        assert len(recs) == 500, f"Expected 500 records for {s_id}"

        for rec in recs:
            assert validate_record(rec), f"Record {rec.voyage_id} in {s_id} failed physical validation"
            assert rec.fuel_consumption is not None and rec.fuel_consumption > 0.0

        X_df, y_ser = pipeline.get_training_features_and_target(recs, encode_categoricals=True)
        assert X_df.shape == (500, 27), f"Unexpected feature matrix shape for {s_id}: {X_df.shape}"
        assert not X_df.isna().any().any(), f"NaNs found in feature matrix for {s_id}"
        assert not y_ser.isna().any(), f"NaNs found in target series for {s_id}"


def test_stress_test_report_schema_and_contents() -> None:
    """Verify that stress test reports exist and conform to complete audit schema."""
    assert REPORT_JSON.exists(), f"Stress test JSON report missing at {REPORT_JSON}"
    assert REPORT_MD.exists(), f"Stress test MD report missing at {REPORT_MD}"

    data = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    assert "metadata" in data
    assert "shift_specifications" in data
    assert "seed42_frozen_results" in data
    assert "aggregate_results" in data
    assert "degradation_summary" in data
    assert "model_rankings" in data
    assert "component_analysis" in data
    assert "decision_summary" in data

    # Check that in-domain and all 3 stress levels are evaluated
    for cond in ["in_domain", "level1", "level2", "level3"]:
        assert cond in data["aggregate_results"]
        assert cond in data["model_rankings"]
        assert len(data["model_rankings"][cond]) == 5

    # Check decision summary verdict
    dec = data["decision_summary"]
    assert "FREEZE" in dec["verdict"]


def test_stress_test_predictions_reproducibility() -> None:
    """Ensure models evaluated on shifted data yield bitwise identical predictions across runs."""
    loader = CSVDatasetLoader()
    pipeline = FeatureEngineeringPipeline()

    canon_records = loader.load_data(str(CANONICAL_DATASET))
    X_canon_df, y_canon_ser = pipeline.get_training_features_and_target(canon_records)
    X_train = X_canon_df.to_numpy(dtype=float)[:300]
    y_train = y_canon_ser.to_numpy(dtype=float)[:300]

    shift_records = loader.load_data(str(SHIFTED_DATASETS["level1"]["path"]))
    X_shift_df, _ = pipeline.get_training_features_and_target(shift_records)
    X_shift = X_shift_df.to_numpy(dtype=float)

    # 1. NavalPhysicsFuelBaseline
    phys_model = NavalPhysicsFuelBaseline(feature_names=list(X_canon_df.columns))
    phys_model.fit(X_train, y_train)
    p1 = phys_model.predict(X_shift)
    p2 = phys_model.predict(X_shift)
    np.testing.assert_array_equal(p1, p2, err_msg="NavalPhysicsFuelBaseline predictions not reproducible")

    # 2. PhysicsInformedQIFCPRegressor
    pi_model = PhysicsInformedQIFCPRegressor(
        feature_names=list(X_canon_df.columns),
        random_state=42,
    )
    pi_model.fit(X_train, y_train)
    y_phys1, r1, y_fin1 = pi_model.predict_components(X_shift)
    y_phys2, r2, y_fin2 = pi_model.predict_components(X_shift)

    np.testing.assert_array_equal(y_phys1, y_phys2)
    np.testing.assert_array_equal(r1, r2)
    np.testing.assert_array_equal(y_fin1, y_fin2)


def test_degradation_metrics_mathematical_consistency() -> None:
    """Verify that delta metrics match the exact definition (shifted - in-domain)."""
    data = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    deg = data["degradation_summary"]
    agg = data["aggregate_results"]

    for lvl in ["level1", "level2", "level3"]:
        for m_id in ["physics_residual_qifcp", "naval_physics_baseline", "random_forest"]:
            if m_id not in deg[lvl]:
                continue
            d_item = deg[lvl][m_id]
            expected_delta_mae = agg[lvl][m_id]["mae"]["mean"] - agg["in_domain"][m_id]["mae"]["mean"]
            assert pytest.approx(d_item["delta_mae"], abs=1e-3) == expected_delta_mae

            expected_pct = (expected_delta_mae / agg["in_domain"][m_id]["mae"]["mean"]) * 100.0
            assert pytest.approx(d_item["pct_delta_mae"], abs=1e-1) == expected_pct
