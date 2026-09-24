"""Unit test verifying that hours_at_sea does not leak data_source identity.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
PART A: Neutralize the hours_at_sea proxy-leakage.
Validates that a single-feature classifier using ONLY hours_at_sea achieves <= 60%
accuracy on predicting data_source across the 3 classes ('mock', 'thetis_mrv', 'fuelcast'),
confirming that observational durations have been aligned with representative commercial
voyage leg distributions and source fingerprinting is eliminated.

CI note
-------
These leakage tests require a *blended* dataset that contains records from all three
sources ('mock', 'thetis_mrv', 'fuelcast').  In CI the proprietary THETIS-MRV and FuelCast
source files are absent so the blending function returns only synthetic data — making the
multi-class leakage test meaningless.  All tests that depend on real data skip gracefully
when the blended file cannot be produced with all three source labels.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.ingestion.real_data_adapter import (
    RealDataAdapter,
    blend_real_and_synthetic_datasets,
    compute_source_group,
)
from src.ingestion.feature_pipeline import FeatureEngineeringPipeline


# ---------------------------------------------------------------------------
# Module-level fixture: blended dataset
#
# Returns a Path to a CSV that contains all three source labels.
# Skips the entire module if real data is unavailable so downstream tests
# never receive a path to a file that doesn't exist.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def blended_dataset_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Ensure a blended dataset (all three sources) exists for leakage testing.

    Resolution order:
      1. Use data/processed/voyages_blended.csv if it already contains all three
         source labels (real run has been done locally).
      2. Otherwise generate a fresh blend from the synthetic CSV + real sources.
      3. Skip the entire module if real data is unavailable — a synthetic-only
         blend only has 'mock' labels and cannot test 3-class leakage.
    """
    existing = Path("data/processed/voyages_blended.csv")
    if existing.exists():
        df = pd.read_csv(existing)
        sources = set(df["data_source"].unique())
        if {"mock", "thetis_mrv", "fuelcast"}.issubset(sources):
            return existing
        # Falls through: file exists but is synthetic-only from a previous run

    synth_path = Path("data/raw/voyages_sample.csv")
    if not synth_path.exists():
        pytest.skip("Synthetic dataset voyages_sample.csv not found — cannot generate blend.")

    tmp_dir = tmp_path_factory.mktemp("leakage_test")
    out_csv = tmp_dir / "voyages_blended.csv"
    _, stats = blend_real_and_synthetic_datasets(
        synthetic_path=synth_path,
        output_path=out_csv,
        seed=42,
    )

    # If no real records were available the blend is synthetic-only; skip.
    if stats.get("real_rows", 0) == 0:
        pytest.skip(
            "No real observational records available (THETIS-MRV=0, FuelCast=0). "
            "The 3-class hours_at_sea leakage test requires all three source labels. "
            "Place THETIS-MRV and FuelCast source files in data/raw to enable these tests."
        )

    # Sanity-check that the file was actually written
    if not out_csv.exists():
        pytest.skip("blend_real_and_synthetic_datasets did not write output CSV — skipping.")

    return out_csv


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_hours_at_sea_does_not_leak_source(blended_dataset_path: Path) -> None:
    """Train single-feature model using ONLY hours_at_sea to predict data_source.

    Asserts that accuracy is <= 60% (threshold required by project specification).
    Prior to the fix, accuracy was 95.25% due to 1.0 h FuelCast timestamps.
    """
    df = pd.read_csv(blended_dataset_path)

    # Verify 3 distinct classes exist
    sources = set(df["data_source"].unique())
    assert sources == {"mock", "thetis_mrv", "fuelcast"}, f"Expected 3 sources, got {sources}"

    X = df["hours_at_sea"].to_numpy(dtype=float).reshape(-1, 1)
    y = df["data_source"].to_numpy(dtype=str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(random_state=42, max_iter=200)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))

    # Strict assertion against 60% threshold
    assert accuracy <= 0.60, (
        f"hours_at_sea proxy leakage detected: single-feature classifier achieved "
        f"{accuracy:.2%} accuracy (must be <= 60.0%). Prior unmitigated accuracy: 95.25%."
    )


def test_hours_at_sea_balanced_near_random(blended_dataset_path: Path) -> None:
    """Verify on class-balanced evaluation that accuracy is near the 33.3% random baseline."""
    df = pd.read_csv(blended_dataset_path)
    min_count = int(df["data_source"].value_counts().min())

    balanced_df = pd.concat(
        [
            df[df["data_source"] == s].sample(min_count, random_state=42)
            for s in sorted(df["data_source"].unique())
        ],
        ignore_index=True,
    )

    X = balanced_df["hours_at_sea"].to_numpy(dtype=float).reshape(-1, 1)
    y = balanced_df["data_source"].to_numpy(dtype=str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(random_state=42, max_iter=200)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    balanced_accuracy = float(accuracy_score(y_test, y_pred))

    # Balanced accuracy should be near random 33.3% baseline, well below 50%
    assert balanced_accuracy <= 0.50, (
        f"Balanced 3-class accuracy {balanced_accuracy:.2%} is significantly above "
        f"random baseline (33.33%)."
    )


def test_feature_pipeline_drops_raw_hours_at_sea() -> None:
    """Verify FeatureEngineeringPipeline drops raw hours_at_sea from feature matrix X.

    Confirms hours_at_sea is replaced by implied_hours and one-hot source_group
    to prevent target and proxy leakage during model training.

    Requires THETIS-MRV source files in data/raw — skips when unavailable.
    """
    adapter = RealDataAdapter()
    sample_records = adapter.load_thetis_mrv(limit=20)
    if len(sample_records) == 0:
        pytest.skip(
            "THETIS-MRV adapter returned 0 records — source files absent in data/raw. "
            "Place THETIS-MRV CSV files in data/raw to enable this test."
        )

    pipeline = FeatureEngineeringPipeline()
    X, y = pipeline.get_training_features_and_target(sample_records, encode_categoricals=True)

    # Raw hours_at_sea must not appear in X
    assert "hours_at_sea" not in X.columns, "Raw hours_at_sea must be dropped from feature matrix X!"

    # Replacement features must be present
    assert "implied_hours" in X.columns, "implied_hours must be present in X"
    assert any(col.startswith("source_group_") for col in X.columns), "source_group one-hot features missing"
