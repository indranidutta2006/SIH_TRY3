"""Unit tests for RealDataAdapter and dataset blending.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Verifies that observational datasets (THETIS-MRV and FuelCast) conform to the VoyageRecord
schema, preserve missing environmental fields as null, and achieve >= 90% validation pass rate.
"""

from pathlib import Path
import pytest

from src.ingestion.feature_pipeline import FeatureEngineeringPipeline
from src.ingestion.real_data_adapter import (
    RealDataAdapter,
    blend_real_and_synthetic_datasets,
)
from src.ingestion.validators import ValidationEngine


@pytest.fixture
def adapter() -> RealDataAdapter:
    """Instantiate RealDataAdapter pointing to repository data/raw."""
    return RealDataAdapter(raw_dir=Path("data/raw"))


@pytest.fixture
def validator() -> ValidationEngine:
    """Instantiate standard ValidationEngine."""
    return ValidationEngine()


def test_thetis_mrv_schema_and_null_weather(adapter: RealDataAdapter, validator: ValidationEngine) -> None:
    """Verify THETIS-MRV records conform to schema, leave weather null, and pass >=90% validation."""
    records = adapter.load_thetis_mrv(limit=500)
    assert len(records) > 0, "Failed to load THETIS-MRV records from data/raw."

    # Validate provenance and environmental field invariants
    for rec in records:
        assert rec.data_source == "thetis_mrv"
        assert rec.is_synthetic is False
        assert rec.weather_factor is None, "THETIS-MRV weather_factor must be null, not fabricated."
        assert rec.sea_state is None, "THETIS-MRV sea_state must be null, not fabricated."
        assert rec.fuel_consumption is not None and rec.fuel_consumption > 0.0
        assert rec.distance_nm > 0.0
        assert rec.hours_at_sea > 0.0

    # Validate domain pass rate >= 90%
    val_result = validator.validate_dataset(records)
    pass_rate = val_result.valid_records / val_result.total_records
    assert pass_rate >= 0.90, f"THETIS-MRV pass rate {pass_rate:.1%} fell below required 90% threshold."


def test_fuelcast_schema_and_environmental_fields(adapter: RealDataAdapter, validator: ValidationEngine) -> None:
    """Verify FuelCast records conform to schema, retain observed weather, and pass >=90% validation."""
    records = adapter.load_fuelcast(limit_per_vessel=200, limit=500)
    assert len(records) > 0, "Failed to load FuelCast records from data/raw."

    # Validate provenance and observed environmental fields
    for rec in records:
        assert rec.data_source == "fuelcast"
        assert rec.is_synthetic is False
        assert rec.weather_factor is not None and rec.weather_factor >= 1.0
        assert rec.sea_state is not None and 0 <= rec.sea_state <= 9
        assert rec.fuel_consumption is not None and rec.fuel_consumption > 0.0
        assert rec.speed_knots >= 1.0

    # Validate domain pass rate >= 90%
    val_result = validator.validate_dataset(records)
    pass_rate = val_result.valid_records / val_result.total_records
    assert pass_rate >= 0.90, f"FuelCast pass rate {pass_rate:.1%} fell below required 90% threshold."


def test_load_all_real_data_pass_rate(adapter: RealDataAdapter, validator: ValidationEngine) -> None:
    """Verify aggregated real dataset achieves >= 90% validation compliance."""
    records = adapter.load_all_real_data(target_count=600, seed=42)
    assert len(records) > 0

    sources = {rec.data_source for rec in records}
    assert "thetis_mrv" in sources
    assert "fuelcast" in sources

    val_result = validator.validate_dataset(records)
    pass_rate = val_result.valid_records / val_result.total_records
    assert pass_rate >= 0.90, f"Combined real dataset pass rate {pass_rate:.1%} below 90% threshold."


def test_blend_real_and_synthetic_ratio(tmp_path: Path) -> None:
    """Verify blend_real_and_synthetic_datasets targets roughly 50/50 real-to-synthetic ratio."""
    synth_path = Path("data/raw/voyages_sample.csv")
    if not synth_path.exists():
        pytest.skip("Synthetic dataset voyages_sample.csv not found.")

    out_csv = tmp_path / "blended_test.csv"
    blended_records, stats = blend_real_and_synthetic_datasets(
        synthetic_path=synth_path,
        target_ratio=0.5,
        output_path=out_csv,
        seed=42,
    )

    assert len(blended_records) > 0
    assert out_csv.exists()

    # Target roughly 50/50 ratio (tolerance 45% - 55%)
    real_ratio = stats["real_ratio"]
    assert 0.45 <= real_ratio <= 0.55, f"Real ratio {real_ratio:.2f} deviated from target ~0.50."

    # Verify FeatureEngineeringPipeline transforms the blended dataset cleanly without error
    pipeline = FeatureEngineeringPipeline()
    x_df, y_series = pipeline.get_training_features_and_target(blended_records[:200])
    assert not x_df.empty
    assert not y_series.empty
    assert not x_df.isna().any().any(), "Engineered feature matrix contains unexpected NaNs."
