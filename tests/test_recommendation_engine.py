"""Deterministic unit tests for VoyageOptimizationEngine and OptimizationRecommendation.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 2B tests verifying operational intervention rules: speed reduction, weather routing,
cargo consolidation, CO2 savings calculation, and report serialization.
"""

import json
import pytest

from contracts.schemas import VoyageRecord
from src.prediction.recommendation_engine import (
    OptimizationRecommendation,
    VoyageOptimizationEngine,
)


@pytest.fixture
def baseline_voyage() -> VoyageRecord:
    """Fixture providing a standard compliant baseline VoyageRecord."""
    return VoyageRecord(
        voyage_id="VY-OPT-001",
        vessel_id="VSL-001",
        vessel_type="Bulk Carrier",
        vessel_dwt=100000.0,
        cargo_tons=85000.0,  # 85% utilization (>= 70%)
        distance_nm=3000.0,
        speed_knots=13.0,  # <= 14.0 fleet avg
        hours_at_sea=230.76,
        fuel_type="Diesel",
        weather_factor=1.08,  # <= 1.15
        sea_state=2,
        data_source="mock",
        is_synthetic=True,
        fuel_consumption=500.0,
        co2_emissions=1603.0,
    )


def test_optimal_voyage_no_interventions(baseline_voyage: VoyageRecord) -> None:
    """Ensure compliant voyage operating inside green envelope requires no interventions."""
    engine = VoyageOptimizationEngine()
    rec = engine.evaluate_voyage(baseline_voyage, predicted_fuel_consumption=500.0)

    assert isinstance(rec, OptimizationRecommendation)
    assert rec.estimated_savings_fuel == 0.0
    assert rec.estimated_savings_co2 == 0.0
    assert rec.recommended_speed == 13.0
    assert len(rec.opportunities) == 0
    assert "optimal green fleet parameters" in rec.recommendation_text


def test_high_speed_triggers_speed_reduction(baseline_voyage: VoyageRecord) -> None:
    """Ensure speed exceeding fleet average triggers cubic speed reduction recommendation."""
    from dataclasses import replace

    high_speed_voyage = replace(baseline_voyage, speed_knots=16.0)
    engine = VoyageOptimizationEngine(fleet_average_speed=14.0, speed_reduction_pct=0.075)

    rec = engine.evaluate_voyage(high_speed_voyage, predicted_fuel_consumption=1000.0)

    assert "Speed Reduction" in rec.opportunities
    assert rec.recommended_speed < 16.0
    assert rec.recommended_speed == pytest.approx(16.0 * (1.0 - 0.075), abs=0.05)
    assert rec.estimated_savings_fuel > 0.0
    assert rec.estimated_savings_co2 > 0.0
    assert "Reduce operating speed" in rec.recommendation_text


def test_weather_mitigation_trigger(baseline_voyage: VoyageRecord) -> None:
    """Ensure weather factor exceeding 1.15 triggers meteorological rerouting."""
    from dataclasses import replace

    severe_weather_voyage = replace(baseline_voyage, weather_factor=1.28)
    engine = VoyageOptimizationEngine()

    rec = engine.evaluate_voyage(severe_weather_voyage, predicted_fuel_consumption=800.0)

    assert "Weather Routing" in rec.opportunities
    assert rec.estimated_savings_fuel > 0.0
    assert "Adverse weather factor" in rec.recommendation_text


def test_cargo_efficiency_trigger(baseline_voyage: VoyageRecord) -> None:
    """Ensure payload utilization under 70% triggers cargo consolidation advice."""
    from dataclasses import replace

    underloaded_voyage = replace(baseline_voyage, cargo_tons=40000.0)  # 40% utilization
    engine = VoyageOptimizationEngine()

    rec = engine.evaluate_voyage(underloaded_voyage, predicted_fuel_consumption=600.0)

    assert "Cargo Consolidation" in rec.opportunities
    assert "Low cargo utilization" in rec.recommendation_text


def test_compound_optimization_scenario(baseline_voyage: VoyageRecord) -> None:
    """Ensure multiple overlapping operational inefficiencies are all identified."""
    from dataclasses import replace

    inefficient_voyage = replace(
        baseline_voyage,
        speed_knots=17.5,
        weather_factor=1.25,
        cargo_tons=50000.0,  # 50% < 70%
    )
    engine = VoyageOptimizationEngine()
    rec = engine.evaluate_voyage(inefficient_voyage, predicted_fuel_consumption=1200.0)

    assert len(rec.opportunities) == 3
    assert "Speed Reduction" in rec.opportunities
    assert "Weather Routing" in rec.opportunities
    assert "Cargo Consolidation" in rec.opportunities
    assert rec.estimated_savings_fuel > 100.0
    assert rec.estimated_savings_co2 > 300.0


def test_zero_predicted_fuel_edge_case(baseline_voyage: VoyageRecord) -> None:
    """Ensure zero predicted fuel returns safe zero savings without errors."""
    engine = VoyageOptimizationEngine()
    rec = engine.evaluate_voyage(baseline_voyage, predicted_fuel_consumption=0.0)

    assert rec.estimated_savings_fuel == 0.0
    assert rec.estimated_savings_co2 == 0.0
    assert len(rec.opportunities) == 0


def test_carbon_savings_coherence_with_fuel_type(baseline_voyage: VoyageRecord) -> None:
    """Ensure carbon savings use statutory factors for the specific fuel type."""
    from dataclasses import replace

    # Diesel voyage
    diesel_voyage = replace(baseline_voyage, speed_knots=16.0, fuel_type="Diesel")
    # LNG voyage
    lng_voyage = replace(baseline_voyage, speed_knots=16.0, fuel_type="LNG")

    engine = VoyageOptimizationEngine()
    rec_diesel = engine.evaluate_voyage(diesel_voyage, predicted_fuel_consumption=1000.0)
    rec_lng = engine.evaluate_voyage(lng_voyage, predicted_fuel_consumption=1000.0)

    # Fuel savings are identical (same speed & base fuel)
    assert pytest.approx(rec_diesel.estimated_savings_fuel, rel=1e-3) == rec_lng.estimated_savings_fuel

    # Diesel emission factor (3.206) > LNG emission factor (2.750)
    assert rec_diesel.estimated_savings_co2 > rec_lng.estimated_savings_co2
    expected_diesel_co2 = rec_diesel.estimated_savings_fuel * 3.206
    assert pytest.approx(rec_diesel.estimated_savings_co2, rel=1e-2) == expected_diesel_co2


def test_recommendation_serialization(baseline_voyage: VoyageRecord) -> None:
    """Verify OptimizationRecommendation serializes cleanly to dict and JSON."""
    from dataclasses import replace

    voyage = replace(baseline_voyage, speed_knots=15.5)
    engine = VoyageOptimizationEngine()
    rec = engine.evaluate_voyage(voyage, predicted_fuel_consumption=700.0)

    d = rec.to_dict()
    assert "estimated_savings_fuel" in d
    assert "recommended_speed" in d

    json_str = rec.to_json()
    parsed = json.loads(json_str)
    assert parsed["recommended_speed"] == rec.recommended_speed
