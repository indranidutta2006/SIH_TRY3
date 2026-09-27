"""Standalone realistic smoke test for the integrated Prediction -> Fleet Optimization pipeline.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Executes an end-to-end realistic fleet optimization scenario using the frozen
PhysicsInformedQIFCPRegressor wrapped in FuelPredictionService.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))

from contracts.constants import FUEL_PRICES_USD_PER_TON, FuelType
from contracts.schemas import OptimizationScenario, OptimizationStatus
from src.compliance.compliance_engine import MaritimeComplianceEngine
from src.optimization.fleet_composition_optimizer import FleetCompositionOptimizer
from src.prediction.emission_engine import MaritimeEmissionEngine
from src.prediction.fuel_prediction_service import get_fuel_prediction_service


def run_realistic_smoke_test() -> None:
    print("=" * 80)
    print("SIH26138 REALISTIC INTEGRATED SMOKE TEST: PREDICTION -> OPTIMIZATION")
    print("=" * 80)

    # 1. Initialize Canonical Service & Engines
    service = get_fuel_prediction_service()
    compliance_engine = MaritimeComplianceEngine()
    emission_engine = MaritimeEmissionEngine()
    optimizer = FleetCompositionOptimizer(fuel_service=service)

    print("\n--- [1] PREDICTOR INITIALIZATION ---")
    print(f"Prediction Service:  {type(service).__name__}")
    print(f"Canonical Model:     {service.model_name}")
    print(f"Estimator Class:     {type(service.model).__name__}")
    print(f"Harmonic Order (K):  {service.model.harmonic_order}")
    print(f"Pair Budget (M):     {service.model.n_entanglement_pairs}")
    print(f"Entanglement Mode:   {service.model.entanglement_mode}")
    print(f"Gamma Mode:          {service.model.gamma_mode}")
    print(f"Residual Lambda:     {service.model.lambda_residual}")

    # 2. Scenario Setup
    scenario = OptimizationScenario(
        cargo_demand=120_000.0,
        route_distance=2800.0,
        deadline_hours=240.0,
        scenario_id="SCEN-REALISTIC-SMOKE",
        budget=140_000_000.0,
        carbon_price=75.0,
        weather_factor=1.12,
        service_level=0.95,
        max_transition_rate=0.40,
        target_reliability=0.90,
        port_delay_factor=1.08,
        weights=(0.45, 0.35, 0.20),
    )

    print("\n--- [2] SCENARIO SPECIFICATION ---")
    print(f"Scenario ID:         {scenario.scenario_id}")
    print(f"Annual Demand:       {scenario.cargo_demand:,.1f} metric tons")
    print(f"Route Distance:      {scenario.route_distance:,.1f} NM")
    print(f"Transit Deadline:    {scenario.deadline_hours:.1f} hours")
    print(f"Weather Factor:      {scenario.weather_factor:.2f}")
    print(f"Carbon Price:        ${scenario.carbon_price:.2f}/ton CO2e")
    print(f"CAPEX Budget:        ${scenario.budget:,.2f}")
    print(f"Max Transition Rate: {scenario.max_transition_rate * 100:.1f}%")
    print(f"Target Reliability:  {scenario.target_reliability * 100:.1f}%")

    # 3. Multi-Fuel Single-Voyage Predictions
    print("\n--- [3] PREDICTION TRACE (Single-Voyage Representative Baseline: Panamax 45k DWT, 14.0 kts) ---")
    test_dwt = 45000.0
    test_cargo = test_dwt * 0.85
    test_speed = 14.0

    fuel_samples = {}
    for fuel in ["Diesel", "LNG", "Methanol", "Hydrogen", "Ammonia", "ShorePower"]:
        t0 = time.perf_counter()
        f_pred = service.calculate_fuel(
            distance_nm=scenario.route_distance,
            speed_knots=test_speed,
            cargo_tons=test_cargo,
            weather_factor=scenario.weather_factor,
            fuel_type=fuel,
            vessel_dwt=test_dwt,
        )
        dt_us = (time.perf_counter() - t0) * 1e6
        fuel_samples[fuel] = f_pred
        extra = f"(last_energy: {service.last_energy_mwh:.2f} MWh)" if fuel == "ShorePower" else ""
        print(f"  - {fuel:<12}: {f_pred:>8.2f} metric tons {extra:<30} [{dt_us:>6.1f} us]")

    # 4. Optimization Execution
    print("\n--- [4] OPTIMIZATION TRACE ---")
    t_opt_start = time.perf_counter()
    result = optimizer.optimize_composition(scenario)
    opt_time_s = time.perf_counter() - t_opt_start

    print(f"Optimization Status: {result.status.value}")
    print(f"Solver Name:         {result.metadata.get('solver_name')}")
    print(f"Iterations:          {result.metadata.get('iterations')}")
    print(f"Runtime:             {result.metadata.get('runtime_ms'):.2f} ms (wall-clock: {opt_time_s:.2f}s)")
    print(f"Convergence Score:   {result.metadata.get('convergence_score'):.2f}")
    print(f"Selected Fleet Mix:  {result.fleet_mix}")
    print(f"Traceable Predictor: {result.metadata.get('predictor')}")
    print(f"Prediction Source:   {result.metadata.get('fuel_prediction_source')}")

    # 5. Economics & Energy
    print("\n--- [5] ECONOMICS & ENERGY ---")
    print(f"Total Fuel Consumed: {result.fuel_consumption:>12,.2f} metric tons")
    print(f"Predicted Fuel Meta: {result.metadata.get('predicted_fuel_consumption'):>12,.2f} metric tons")
    print(f"Bunker Fuel Cost:   ${result.metadata.get('fuel_cost'):>12,.2f}")
    print(f"Carbon Cost:        ${result.carbon_cost:>12,.2f}")
    print(f"Total Operational:  ${result.operational_cost:>12,.2f}")
    print(f"Composite Objective: {result.optimization_score:>12,.4f}")

    # 6. Environmental Impact
    print("\n--- [6] ENVIRONMENTAL IMPACT ---")
    print(f"Well-to-Wake CO2e:   {result.emissions:>12,.2f} metric tons")
    print(f"Metadata CO2e:       {result.metadata.get('co2_emissions'):>12,.2f} metric tons")
    transport_work = result.total_capacity * scenario.route_distance
    ei = (result.emissions * 1e6) / max(transport_work, 1.0)
    print(f"Emissions Intensity: {ei:>12.3f} gCO2e / (ton-NM)")

    # 7. Operational Logistics & Service
    print("\n--- [7] OPERATIONS & LOGISTICS ---")
    print(f"Delivered Cargo Cap: {result.total_capacity:>12,.2f} metric tons")
    print(f"Target Demand:       {scenario.cargo_demand:>12,.2f} metric tons")
    print(f"Service Level:       {result.service_level_achieved * 100:>11.2f}% (req: {scenario.service_level * 100:.1f}%)")

    # 8. Regulatory Compliance Verification
    print("\n--- [8] STATUTORY REGULATORY COMPLIANCE ---")
    reg = result.metadata.get("regulatory_breakdown", {})
    print(f"Regulatory Breakdown: {json.dumps(reg, indent=2)}")

    # Statutory compliance check using compliance engine
    cii_res = compliance_engine.evaluate_cii(
        co2_emissions=result.emissions,
        distance_nm=scenario.route_distance,
        capacity=result.total_capacity,
        vessel_type="Bulk Carrier",
        year=2024,
    )
    print(f"IMO CII Letter Rating: {cii_res.cii_rating} (Attained: {cii_res.attained_cii:.2f}, Required: {cii_res.required_cii:.2f}, Ratio: {cii_res.cii_ratio:.3f})")

    # 9. Verification Assertion
    print("\n--- [9] REPRODUCIBILITY & EXACT EQUIVALENCE ASSERTION ---")
    assert result.status == OptimizationStatus.SUCCESS, "Optimization failed"
    assert result.fuel_consumption > 0.0, "Zero fuel consumed"
    assert result.metadata.get("predicted_fuel_consumption") == result.fuel_consumption
    assert result.metadata.get("predictor") == "physics_residual_qifcp"
    assert result.metadata.get("fuel_prediction_source") == "canonical_qifcp"
    print("Verification Assertion: PASSED (Exact fuel and metadata traceability verified)")
    print("=" * 80)


if __name__ == "__main__":
    run_realistic_smoke_test()
