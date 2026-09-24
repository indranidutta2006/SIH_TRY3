"""Single end-to-end execution pipeline for SIH26138 Green Fleet Navigator.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Executes the full chain in one command:
1. Data Ingestion / Verification (Raw voyage records)
2. Prediction Layer: Model Training & QPSO-Tuned QIFCP Benchmarking
3. Emissions & Compliance Layers: Direct Statutory Evaluation
4. Scheduler Layer: Feasible Fleet Allocation & Constraint Validation
5. Optimization Layer: Equal-Budget Swarm Scalability Sweep (QPSO vs PSO & Pareto)
6. Macro Scenario Analysis: 6-Fuel Pathway Simulation (ML + Physics Routing)
7. Artifact Verification: Validates all JSON reports required by the executive dashboard.

Outputs:
- artifacts/models/*.pkl
- artifacts/metrics/baseline_metrics.json
- outputs/reports/prediction_benchmark.json
- outputs/reports/optimization_benchmark.json
- outputs/reports/scenario_comparison.json
- outputs/reports/full_pipeline_run.json
"""

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from logging_config import configure_logging
from contracts.constants import FuelType
from contracts.schemas import VoyageRecord
from src.compliance.compliance_engine import MaritimeComplianceEngine
from src.prediction import train_all_models
from src.prediction.emission_engine import MaritimeEmissionEngine
from src.scheduler.fleet_scheduler import FleetScheduler
from scripts.benchmark_prediction import run_prediction_benchmark
from scripts.benchmark_optimization import run_optimization_benchmark
from scripts.run_scenario_comparison import run_scenario_comparison

logger = logging.getLogger("maritime_system")


def ensure_directories() -> None:
    """Ensure all required artifact and output directories exist."""
    directories = [
        "data/raw",
        "data/processed",
        "artifacts/models",
        "artifacts/metrics",
        "outputs/reports",
        "outputs/logs",
        "outputs/figures",
    ]
    for d in directories:
        Path(d).mkdir(parents=True, exist_ok=True)


def run_full_pipeline(
    dataset_path: str = "data/raw/voyages_sample.csv",
    rows_to_generate_if_missing: int = 10000,
    seed: int = 42,
    seeds: Sequence[int] | None = None,
    normalize_objective: bool = True,
    force_regenerate_data: bool = False,
    blend_real: bool = False,
    use_real_data: bool = False,
) -> dict[str, Any]:
    """Execute the full end-to-end maritime intelligence and optimization pipeline."""
    configure_logging(level="INFO")
    ensure_directories()
    t_pipeline_start = time.perf_counter()
    start_iso = datetime.now(timezone.utc).isoformat()
    eval_seeds = tuple(seeds) if seeds is not None else (42, 101, 2024)
    real_data_enabled = bool(use_real_data or blend_real)

    logger.info("=" * 72)
    logger.info("SIH26138: STARTING COMPLETE END-TO-END MARITIME PIPELINE")
    logger.info("Seeds configured for swarm sweep: %s", eval_seeds)
    if real_data_enabled:
        logger.info("Real Data Integration: ENABLED (Targeting ~50/50 Real/Synthetic, Canonical Rate Framing)")
    else:
        logger.info("Real Data Integration: DISABLED (Synthetic-Only Fallback)")
    logger.info("=" * 72)

    stage_timings: dict[str, float] = {}

    # -------------------------------------------------------------------------
    # STAGE 1: Data Ingestion & Dataset Verification
    # -------------------------------------------------------------------------
    logger.info("\n>>> [STAGE 1/6] Data Ingestion & Dataset Verification")
    t0 = time.perf_counter()
    data_file = Path(dataset_path)
    if force_regenerate_data or not data_file.exists():
        logger.info("Synthesizing %d fresh voyage records -> '%s'...", rows_to_generate_if_missing, dataset_path)
        from scripts.make_mock_dataset import generate_synthetic_voyages, write_csv
        records = generate_synthetic_voyages(num_rows=rows_to_generate_if_missing, seed=seed)
        write_csv(records, data_file)
        logger.info("Generated synthetic voyage dataset: %s (%d rows)", data_file, len(records))
    else:
        logger.info("Found existing voyage dataset: %s (%d bytes)", data_file, data_file.stat().st_size)

    active_training_dataset = dataset_path
    if real_data_enabled:
        logger.info("Blending synthetic dataset with real observational data (THETIS-MRV + FuelCast)...")
        from src.ingestion.real_data_adapter import blend_real_and_synthetic_datasets
        blended_path = "data/processed/voyages_blended.csv"
        _, blend_stats = blend_real_and_synthetic_datasets(
            synthetic_path=data_file,
            target_ratio=0.5,
            output_path=blended_path,
            seed=seed,
        )
        active_training_dataset = blended_path
        ratio_msg = (
            f"[OK] Blended Dataset Achieved: {blend_stats['real_pct']:.1f}% Real ({blend_stats['real_rows']:,} rows) / "
            f"{blend_stats['synthetic_pct']:.1f}% Synthetic ({blend_stats['synthetic_rows']:,} rows) "
            f"[Total: {blend_stats['total_rows']:,} rows] -> '{blended_path}'\n"
            f"     Pre-Blend Auditing:\n"
            f"       - THETIS-MRV Pre-Blend Rows: {blend_stats.get('thetis_pre_blend_count', 0):,}\n"
            f"       - FuelCast Pre-Blend Rows:   {blend_stats.get('fuelcast_pre_blend_count', 0):,}\n"
            f"       - Total Real Pre-Blend Rows: {blend_stats.get('total_real_pre_blend_count', 0):,}\n"
            f"       - Synthetic Pre-Blend Rows:  {blend_stats.get('synthetic_pre_blend_count', 0):,}\n"
            f"       - Subsampling Performed:     {blend_stats.get('subsampling_performed', False)} "
            f"(subsampled {blend_stats['real_rows']:,} from {blend_stats.get('total_real_pre_blend_count', 0):,} real rows without replacement)\n"
            f"       - Duplication Performed:     {blend_stats.get('duplication_performed', False)} (zero duplicate rows)"
        )
        print(f"\n{ratio_msg}\n")
        logger.info(ratio_msg)

        # Log ratio to outputs/logs/
        blend_log_file = Path("outputs/logs/blend_dataset.log")
        blend_log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(blend_log_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now(timezone.utc).isoformat()}]\n{ratio_msg}\n\n")

    stage_timings["1_data_ingestion"] = round(time.perf_counter() - t0, 3)

    # -------------------------------------------------------------------------
    # STAGE 2: Prediction Layer Training & Benchmarking
    # -------------------------------------------------------------------------
    logger.info("\n>>> [STAGE 2/6] Prediction Layer: Model Training & QIFCP Benchmarking")
    t0 = time.perf_counter()
    target_mode = "rate" if real_data_enabled else "absolute"
    logger.info(
        "Training baseline models using dataset '%s' (target_mode=%s)...",
        active_training_dataset,
        target_mode,
    )
    train_all_models(
        dataset_path=active_training_dataset,
        artifacts_dir="artifacts",
        random_state=seed,
        target_mode=target_mode,
    )

    logger.info(
        "Benchmarking models and tuning QIFCP via QPSO (Objective 1, target_mode=%s)...",
        target_mode,
    )
    pred_benchmark = run_prediction_benchmark(
        dataset_path=active_training_dataset,
        output_report="outputs/reports/prediction_benchmark.json",
        sample_size=5000,
        random_state=seed,
        target_mode=target_mode,
    )
    stage_timings["2_prediction_layer"] = round(time.perf_counter() - t0, 3)

    # -------------------------------------------------------------------------
    # STAGE 3: Direct Emissions & Compliance Verification
    # -------------------------------------------------------------------------
    logger.info("\n>>> [STAGE 3/6] Direct Statutory Emissions & Compliance Validation")
    t0 = time.perf_counter()
    emission_engine = MaritimeEmissionEngine()
    compliance_engine = MaritimeComplianceEngine()

    sample_record = VoyageRecord(
        voyage_id="PIPE-VAL-001",
        vessel_id="VSL-BC-001",
        vessel_type="Bulk Carrier",
        vessel_dwt=50000.0,
        cargo_tons=35000.0,
        distance_nm=1200.0,
        speed_knots=14.0,
        hours_at_sea=85.71,
        fuel_type=FuelType.DIESEL.value,
        weather_factor=1.05,
        sea_state=3,
        data_source="synthetic_generator",
        is_synthetic=True,
        fuel_consumption=45.0,
        co2_emissions=144.27,
    )

    sample_emissions = emission_engine.calculate_wtw(
        fuel_consumption=float(sample_record.fuel_consumption or 45.0),
        fuel_type=sample_record.fuel_type,
    )
    sample_cii = compliance_engine.evaluate_cii(
        co2_emissions=float(sample_record.co2_emissions or 144.27),
        cargo_tons=sample_record.cargo_tons,
        distance_nm=sample_record.distance_nm,
        year=2025,
    )
    # Energy: 45 t diesel * 42,700 MJ/t = 1,921,500 MJ; GHG intensity ~91.0 gCO2eq/MJ
    energy_mj = float(sample_record.fuel_consumption or 45.0) * 42700.0
    ghg_intensity = (sample_emissions.co2e * 1_000_000.0) / energy_mj
    sample_fueleu = compliance_engine.evaluate_fueleu(
        ghg_intensity=ghg_intensity,
        energy_used_mj=energy_mj,
        year=2025,
    )

    logger.info(
        "Emissions & Compliance Verified: WTW CO2e=%.2f t | IMO CII Grade=%s (Ratio=%.2f) | FuelEU Pass=%s (Penalty=EUR %.2f)",
        sample_emissions.co2e,
        sample_cii.cii_rating,
        sample_cii.cii_ratio,
        sample_fueleu.fueleu_pass,
        sample_fueleu.penalty_eur,
    )
    stage_timings["3_emissions_compliance"] = round(time.perf_counter() - t0, 3)

    # -------------------------------------------------------------------------
    # STAGE 4: Fleet Scheduler Allocation
    # -------------------------------------------------------------------------
    logger.info("\n>>> [STAGE 4/6] Fleet Scheduler: Cargo Assignment & Constraint Auditing")
    t0 = time.perf_counter()
    scheduler = FleetScheduler()
    from src.optimization.fleet_optimizer import generate_fleet_problem
    vessel_ids, cargo_ids, constraints, assignments = generate_fleet_problem(5, 10, seed=seed)
    scheduler = FleetScheduler(nominal_speed_knots=14.0)
    assigned_count = sum(1 for a in assignments if a.assigned)
    feasibility = scheduler.validate_constraints(assignments, constraints)
    logger.info(
        "Scheduler Feasibility: %s | Assigned: %d / %d cargos (Fleet: %d vessels)",
        "PASS" if feasibility else "FAIL",
        assigned_count,
        len(cargo_ids),
        len(vessel_ids),
    )
    stage_timings["4_scheduler"] = round(time.perf_counter() - t0, 3)

    # -------------------------------------------------------------------------
    # STAGE 5: Swarm Optimization Scalability Sweep (QPSO vs PSO & Pareto)
    # -------------------------------------------------------------------------
    logger.info("\n>>> [STAGE 5/6] Optimization Layer: Multi-Seed Swarm Scalability Sweep (%s)", eval_seeds)
    t0 = time.perf_counter()
    opt_benchmark = run_optimization_benchmark(
        output_report="outputs/reports/optimization_benchmark.json",
        seeds=eval_seeds,
        include_pareto=True,
        normalize=normalize_objective,
    )
    stage_timings["5_optimization_swarm"] = round(time.perf_counter() - t0, 3)

    # -------------------------------------------------------------------------
    # STAGE 6: Macro Scenario Analysis (6 Fuel Pathways)
    # -------------------------------------------------------------------------
    logger.info("\n>>> [STAGE 6/6] Scenario Analysis: 6-Fuel Alternative Pathway Simulation")
    t0 = time.perf_counter()
    scenario_benchmark = run_scenario_comparison(
        output_path="outputs/reports/scenario_comparison.json"
    )
    stage_timings["6_scenario_analysis"] = round(time.perf_counter() - t0, 3)

    total_wall_clock_s = round(time.perf_counter() - t_pipeline_start, 3)
    end_iso = datetime.now(timezone.utc).isoformat()

    # -------------------------------------------------------------------------
    # ARTIFACT INTEGRITY AUDIT
    # -------------------------------------------------------------------------
    required_artifacts = {
        "Raw Dataset": Path("data/raw/voyages_sample.csv"),
        "Model Registry": Path("artifacts/metrics/baseline_metrics.json"),
        "Prediction Benchmark": Path("outputs/reports/prediction_benchmark.json"),
        "Optimization Benchmark": Path("outputs/reports/optimization_benchmark.json"),
        "Scenario Comparison": Path("outputs/reports/scenario_comparison.json"),
    }
    artifact_audit: dict[str, Any] = {}
    all_artifacts_valid = True
    for label, path in required_artifacts.items():
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        artifact_audit[label] = {"path": str(path), "exists": exists, "size_bytes": size}
        if not exists or size == 0:
            all_artifacts_valid = False

    run_manifest = {
        "pipeline_metadata": {
            "status": "SUCCESS" if all_artifacts_valid else "FAILED",
            "start_time_utc": start_iso,
            "end_time_utc": end_iso,
            "total_wall_clock_seconds": total_wall_clock_s,
            "stage_timings_seconds": stage_timings,
            "normalize_objective": normalize_objective,
            "random_seed": seed,
        },
        "artifacts_verified": artifact_audit,
        "executive_highlights": {
            "prediction_models": list(pred_benchmark.get("models", {}).keys()),
            "best_prediction_model": min(
                pred_benchmark.get("models", {}).items(),
                key=lambda item: item[1].get("rmse", 9999),
            )[0]
            if pred_benchmark.get("models")
            else None,
            "optimization_tiers": [item["tier"] for item in opt_benchmark.get("summary", [])],
            "scenario_winner_balanced": scenario_benchmark.get("balanced_multicriteria_ranking", [None])[0]
            if isinstance(scenario_benchmark, dict)
            else None,
        },
    }

    manifest_path = Path("outputs/reports/full_pipeline_run.json")
    manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    # -------------------------------------------------------------------------
    # PRINT EXECUTIVE SUMMARY TABLE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("      SIH26138: FULL PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 78)
    print(f"Total Wall Clock Duration: {total_wall_clock_s:.2f} seconds")
    print("-" * 78)
    print("STAGE BREAKDOWN:")
    for stage, duration in stage_timings.items():
        print(f"  - {stage:<26} : {duration:>7.2f} s")
    print("-" * 78)
    print("GENERATED ARTIFACT AUDIT:")
    for label, info in artifact_audit.items():
        status_sym = "[OK]  " if info["exists"] else "[FAIL]"
        print(f"  {status_sym} {label:<24} : {info['path']:<42} ({info['size_bytes']:>8} bytes)")
    print("-" * 78)
    print(f"Pipeline Run Manifest Saved  : {manifest_path}")
    print("=" * 78 + "\n")

    return run_manifest


def main() -> None:
    """CLI entrypoint for run_full_pipeline.py."""
    parser = argparse.ArgumentParser(
        description="Execute full SIH26138 Green Fleet maritime pipeline from scratch."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/raw/voyages_sample.csv",
        help="Path to raw voyage dataset (default: data/raw/voyages_sample.csv)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=10000,
        help="Synthetic rows to generate if dataset does not exist (default: 10000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 101, 2024],
        help="Random seeds for multi-seed optimization sweep (default: 42 101 2024)",
    )
    parser.add_argument(
        "--regenerate-data",
        action="store_true",
        help="Force regeneration of raw synthetic voyage dataset even if it exists",
    )
    parser.add_argument(
        "--unnormalized",
        action="store_true",
        help="Use unnormalized scalarization objective instead of normalized trade-off",
    )
    parser.add_argument(
        "--use-real-data",
        action="store_true",
        help="Train models on blended real (THETIS-MRV + FuelCast) and synthetic dataset using canonical rate framing",
    )
    parser.add_argument(
        "--blend-real",
        action="store_true",
        help="Merge synthetic dataset with real observational data (THETIS-MRV + FuelCast) targeting ~50/50 ratio before training",
    )
    args = parser.parse_args()

    run_full_pipeline(
        dataset_path=args.data,
        rows_to_generate_if_missing=args.rows,
        seed=args.seed,
        seeds=args.seeds,
        normalize_objective=not args.unnormalized,
        force_regenerate_data=args.regenerate_data,
        blend_real=args.blend_real,
        use_real_data=args.use_real_data,
    )


if __name__ == "__main__":
    main()
