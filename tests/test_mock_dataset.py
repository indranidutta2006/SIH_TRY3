"""Validation tests for the synthetic voyage dataset generator.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Verifies schema compliance, deterministic reproducibility, synthetic data safety tags,
and physical validity of generated mock maritime telemetry.
"""

import csv
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from contracts.schemas import VoyageRecord
from scripts.make_mock_dataset import CSV_HEADERS, generate_synthetic_voyages, write_csv


def test_generate_synthetic_voyages_in_memory() -> None:
    """Verify that in-memory generation produces validated VoyageRecord instances."""
    records = generate_synthetic_voyages(num_rows=100, seed=42)
    assert len(records) == 100
    for rec in records:
        assert isinstance(rec, VoyageRecord)
        assert rec.data_source == "mock"
        assert rec.is_synthetic is True
        assert rec.fuel_consumption is not None and rec.fuel_consumption > 0.0
        assert rec.co2_emissions is not None and rec.co2_emissions > 0.0
        assert rec.cargo_tons <= rec.vessel_dwt


def test_csv_generation_succeeds_and_columns_exist(tmp_path: Path) -> None:
    """Verify CSV generation produces a readable file with exact schema columns."""
    output_file = tmp_path / "test_voyages.csv"
    records = generate_synthetic_voyages(num_rows=50, seed=123)
    write_csv(records, output_file)

    assert output_file.exists()
    assert output_file.stat().st_size > 0

    with output_file.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == CSV_HEADERS


def test_row_count_matches_request(tmp_path: Path) -> None:
    """Verify output row count exactly matches requested number."""
    target_rows = 150
    output_file = tmp_path / "count_test.csv"
    records = generate_synthetic_voyages(num_rows=target_rows, seed=999)
    write_csv(records, output_file)

    df = pd.read_csv(output_file)
    assert len(df) == target_rows


def test_synthetic_safety_markers(tmp_path: Path) -> None:
    """Verify every row contains data_source='mock' and is_synthetic=True."""
    output_file = tmp_path / "safety_test.csv"
    records = generate_synthetic_voyages(num_rows=100, seed=777)
    write_csv(records, output_file)

    df = pd.read_csv(output_file)
    assert (df["data_source"] == "mock").all()
    assert (df["is_synthetic"] == True).all()


def test_no_negative_numeric_values(tmp_path: Path) -> None:
    """Verify that all numerical metrics across the dataset are strictly positive."""
    output_file = tmp_path / "positive_test.csv"
    records = generate_synthetic_voyages(num_rows=100, seed=555)
    write_csv(records, output_file)

    df = pd.read_csv(output_file)
    numeric_columns = [
        "vessel_dwt",
        "cargo_tons",
        "distance_nm",
        "speed_knots",
        "hours_at_sea",
        "weather_factor",
        "sea_state",
        "fuel_consumption",
        "co2_emissions",
    ]
    for col in numeric_columns:
        assert (df[col] > 0).all() or (col == "sea_state" and (df[col] >= 0).all())


def test_reproducibility_deterministic_generation(tmp_path: Path) -> None:
    """Verify that identical random seeds generate identical CSV files."""
    file_a = tmp_path / "voyages_a.csv"
    file_b = tmp_path / "voyages_b.csv"

    records_a = generate_synthetic_voyages(num_rows=50, seed=42)
    records_b = generate_synthetic_voyages(num_rows=50, seed=42)

    write_csv(records_a, file_a)
    write_csv(records_b, file_b)

    assert file_a.read_text(encoding="utf-8") == file_b.read_text(encoding="utf-8")


def test_deserialization_into_voyage_record(tmp_path: Path) -> None:
    """Verify that each serialized CSV row hydrates into a valid VoyageRecord."""
    output_file = tmp_path / "hydration_test.csv"
    records = generate_synthetic_voyages(num_rows=20, seed=333)
    write_csv(records, output_file)

    with output_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Cast numeric types for dict hydration
            row_dict = {
                "voyage_id": row["voyage_id"],
                "vessel_id": row["vessel_id"],
                "vessel_type": row["vessel_type"],
                "vessel_dwt": float(row["vessel_dwt"]),
                "cargo_tons": float(row["cargo_tons"]),
                "distance_nm": float(row["distance_nm"]),
                "speed_knots": float(row["speed_knots"]),
                "hours_at_sea": float(row["hours_at_sea"]),
                "fuel_type": row["fuel_type"],
                "weather_factor": float(row["weather_factor"]),
                "sea_state": int(row["sea_state"]),
                "data_source": row["data_source"],
                "is_synthetic": row["is_synthetic"].lower() == "true",
                "fuel_consumption": float(row["fuel_consumption"]),
                "co2_emissions": float(row["co2_emissions"]),
            }
            record = VoyageRecord.from_dict(row_dict)
            assert isinstance(record, VoyageRecord)


def test_cli_execution_with_custom_rows(tmp_path: Path) -> None:
    """Verify CLI interface runs via subprocess with custom --rows and --output."""
    custom_output = tmp_path / "cli_sample.csv"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/make_mock_dataset.py",
            "--rows",
            "75",
            "--output",
            str(custom_output),
            "--seed",
            "101",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert custom_output.exists()
    df = pd.read_csv(custom_output)
    assert len(df) == 75
    assert (df["data_source"] == "mock").all()
