"""Synthetic maritime voyage dataset generator for green fleet optimization.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Generates physically plausible, schema-conformant synthetic voyage telemetry records
to facilitate offline development and ML model prototyping prior to real-world data ingestion.
"""

import argparse
from collections.abc import Sequence
import csv
import logging
from pathlib import Path
import random
import sys
from typing import Final

# Ensure project root is in sys.path when invoked directly from scripts/
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

import config
from contracts.constants import FuelType
from contracts.schemas import VoyageRecord
import logging_config

# Allowed vessel classes and operational configurations
VESSEL_CONFIGS: Final[dict[str, dict[str, tuple[float, float]]]] = {
    "Bulk Carrier": {
        "dwt": (35000.0, 180000.0),
        "speed": (11.0, 14.5),
        "admiralty": (480.0, 560.0),
        "cargo_ratio": (0.65, 0.95),
    },
    "Container Ship": {
        "dwt": (20000.0, 200000.0),
        "speed": (15.0, 22.0),
        "admiralty": (550.0, 640.0),
        "cargo_ratio": (0.50, 0.85),
    },
    "Oil Tanker": {
        "dwt": (45000.0, 300000.0),
        "speed": (12.0, 15.5),
        "admiralty": (460.0, 540.0),
        "cargo_ratio": (0.70, 0.95),
    },
    "General Cargo": {
        "dwt": (5000.0, 35000.0),
        "speed": (10.0, 14.0),
        "admiralty": (420.0, 500.0),
        "cargo_ratio": (0.55, 0.90),
    },
}

# Permitted Phase 1 fuel types and respective conversion metrics
ALLOWED_FUELS: Final[tuple[str, ...]] = (
    FuelType.DIESEL.value,
    FuelType.LNG.value,
    FuelType.METHANOL.value,
)

# Baseline specific fuel consumption (g/kWh) by fuel energy density
SFOC_MAP: Final[dict[str, float]] = {
    FuelType.DIESEL.value: 175.0,
    FuelType.LNG.value: 145.0,
    FuelType.METHANOL.value: 345.0,
}

# Emission factors (metric tons CO2 per metric ton fuel)
CO2_FACTOR_MAP: Final[dict[str, float]] = {
    FuelType.DIESEL.value: 3.206,
    FuelType.LNG.value: 2.750,
    FuelType.METHANOL.value: 1.375,
}

CSV_HEADERS: Final[list[str]] = [
    "voyage_id",
    "vessel_id",
    "vessel_type",
    "vessel_dwt",
    "cargo_tons",
    "distance_nm",
    "speed_knots",
    "hours_at_sea",
    "fuel_type",
    "weather_factor",
    "sea_state",
    "data_source",
    "is_synthetic",
    "fuel_consumption",
    "co2_emissions",
]


class SyntheticFleet:
    """Pre-generates a realistic commercial vessel pool to preserve fleet consistency."""

    def __init__(self, fleet_size: int = 60, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random(42)
        self.vessels: list[dict[str, str | float]] = []
        vessel_types = list(VESSEL_CONFIGS.keys())

        # Distribute fuels realistically across fleet
        fuel_choices = [FuelType.DIESEL.value] * 7 + [FuelType.LNG.value] * 2 + [FuelType.METHANOL.value]

        prefix_map = {
            "Bulk Carrier": "BC",
            "Container Ship": "CS",
            "Oil Tanker": "OT",
            "General Cargo": "GC",
        }

        for idx in range(1, fleet_size + 1):
            v_type = self.rng.choice(vessel_types)
            dwt_min, dwt_max = VESSEL_CONFIGS[v_type]["dwt"]
            dwt = round(self.rng.uniform(dwt_min, dwt_max), -2)
            fuel = self.rng.choice(fuel_choices)
            prefix = prefix_map[v_type]
            v_id = f"VSL-{prefix}-{idx:03d}"

            self.vessels.append(
                {
                    "vessel_id": v_id,
                    "vessel_type": v_type,
                    "vessel_dwt": dwt,
                    "fuel_type": fuel,
                }
            )

    def sample_vessel(self) -> dict[str, str | float]:
        """Sample a vessel entity from the persistent fleet pool."""
        return self.rng.choice(self.vessels)


def calculate_physics_fuel(
    vessel_type: str,
    vessel_dwt: float,
    cargo_tons: float,
    speed_knots: float,
    hours_at_sea: float,
    weather_factor: float,
    sea_state: int,
    fuel_type: str,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Derive fuel consumption and CO2 emissions via hydrodynamic principles.

    Applies the classical Admiralty formula (Power proportional to Displacement^(2/3) * Speed^3),
    specific fuel oil consumption (SFOC), and fuel carbon factors.

    Returns:
        Tuple of (fuel_consumption_tons, co2_emissions_tons).
    """
    # 1. Total hydrodynamic displacement (lightweight ~ 20% DWT + payload)
    displacement_tons = (0.20 * vessel_dwt) + cargo_tons

    # 2. Admiralty coefficient for vessel hull efficiency
    adm_min, adm_max = VESSEL_CONFIGS[vessel_type]["admiralty"]
    admiralty_coeff = (adm_min + adm_max) / 2.0

    # 3. Main engine propulsion power (kW): P = (Delta^(2/3) * V^3) / C_adm
    propulsion_power_kw = (displacement_tons ** (2.0 / 3.0) * (speed_knots ** 3.0)) / admiralty_coeff

    # 4. Auxiliary and hotel service generator electrical load (kW)
    auxiliary_power_kw = 0.05 * (vessel_dwt ** 0.6) * 100.0

    total_power_kw = propulsion_power_kw + auxiliary_power_kw

    # 5. Specific Fuel Oil Consumption (SFOC)
    sfoc_g_kwh = SFOC_MAP[fuel_type]

    # 6. Environmental resistance multiplier
    # Weather severity and sea state increase required effective thrust
    environmental_multiplier = weather_factor * (1.0 + 0.025 * sea_state)

    # 7. Uncorrected baseline fuel consumption in metric tons
    nominal_fuel_tons = (
        (total_power_kw * sfoc_g_kwh * hours_at_sea) / 1_000_000.0
    ) * environmental_multiplier

    # 8. Minor stochastic variance representing hull fouling, trim, and current drift (sigma ~ 2%)
    stochastic_factor = rng.normal(loc=1.0, scale=0.02)
    fuel_consumption = max(0.1, float(nominal_fuel_tons * stochastic_factor))
    fuel_consumption = round(fuel_consumption, 2)

    # 9. CO2 emissions calculation using stoichiometric fuel factor
    co2_factor = CO2_FACTOR_MAP[fuel_type]
    co2_emissions = round(fuel_consumption * co2_factor, 2)

    return fuel_consumption, co2_emissions


def validate_record(record: VoyageRecord) -> bool:
    """Validate physical plausibility, categorical domains, and positive values."""
    if record.vessel_dwt <= 0.0 or record.cargo_tons <= 0.0:
        return False
    if record.cargo_tons > record.vessel_dwt:
        return False
    if record.distance_nm <= 0.0 or record.speed_knots <= 0.0 or record.hours_at_sea <= 0.0:
        return False
    if record.weather_factor < 1.0 or record.sea_state < 0:
        return False
    if record.fuel_type not in ALLOWED_FUELS:
        return False
    if record.vessel_type not in VESSEL_CONFIGS:
        return False
    if record.data_source != "mock" or not record.is_synthetic:
        return False
    if record.fuel_consumption is None or record.fuel_consumption <= 0.0:
        return False
    if record.co2_emissions is None or record.co2_emissions <= 0.0:
        return False
    return True


def generate_synthetic_voyages(
    num_rows: int,
    seed: int = 42,
) -> list[VoyageRecord]:
    """Generate a deterministic sequence of physically plausible VoyageRecord instances.

    Args:
        num_rows: Number of valid voyage records to produce.
        seed: Fixed random integer for reproducible pseudorandom generation.

    Returns:
        List of validated VoyageRecord instances.
    """
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    fleet = SyntheticFleet(fleet_size=60, rng=py_rng)
    records: list[VoyageRecord] = []
    attempts = 0
    max_attempts = num_rows * 5

    while len(records) < num_rows and attempts < max_attempts:
        attempts += 1
        voyage_idx = len(records) + 1
        voyage_id = f"VY-{voyage_idx:06d}"

        # Sample base vessel from fleet
        vessel = fleet.sample_vessel()
        v_id = str(vessel["vessel_id"])
        v_type = str(vessel["vessel_type"])
        v_dwt = float(vessel["vessel_dwt"])
        fuel_type = str(vessel["fuel_type"])

        # Determine cargo load based on vessel type capacity ratio
        ratio_min, ratio_max = VESSEL_CONFIGS[v_type]["cargo_ratio"]
        cargo_ratio = py_rng.uniform(ratio_min, ratio_max)
        cargo_tons = round(v_dwt * cargo_ratio, 1)

        # Operational distance in nautical miles (300 nm to 7500 nm)
        distance_nm = round(float(np_rng.uniform(300.0, 7500.0)), 1)

        # Operational speed in knots with variance around design speed
        spd_min, spd_max = VESSEL_CONFIGS[v_type]["speed"]
        base_speed = (spd_min + spd_max) / 2.0
        speed_knots = round(float(np_rng.normal(base_speed, 0.8)), 1)
        speed_knots = float(np.clip(speed_knots, spd_min - 1.5, spd_max + 1.5))

        # Voyage duration (hours) with minor port departure/arrival buffer
        hours_at_sea = round((distance_nm / speed_knots) + py_rng.uniform(0.5, 2.5), 2)

        # Sea state (Douglas scale 0 to 7) and correlated weather multiplier (1.00 to 1.35)
        sea_state = int(np_rng.choice([1, 2, 3, 4, 5, 6, 7], p=[0.10, 0.25, 0.30, 0.20, 0.10, 0.03, 0.02]))
        base_weather = 1.0 + (0.04 * sea_state)
        weather_factor = round(float(np.clip(np_rng.normal(base_weather, 0.02), 1.00, 1.38)), 2)

        # Compute physics-based fuel and emissions
        fuel_consumption, co2_emissions = calculate_physics_fuel(
            vessel_type=v_type,
            vessel_dwt=v_dwt,
            cargo_tons=cargo_tons,
            speed_knots=speed_knots,
            hours_at_sea=hours_at_sea,
            weather_factor=weather_factor,
            sea_state=sea_state,
            fuel_type=fuel_type,
            rng=np_rng,
        )

        record = VoyageRecord(
            voyage_id=voyage_id,
            vessel_id=v_id,
            vessel_type=v_type,
            vessel_dwt=v_dwt,
            cargo_tons=cargo_tons,
            distance_nm=distance_nm,
            speed_knots=speed_knots,
            hours_at_sea=hours_at_sea,
            fuel_type=fuel_type,
            weather_factor=weather_factor,
            sea_state=sea_state,
            data_source="mock",
            is_synthetic=True,
            fuel_consumption=fuel_consumption,
            co2_emissions=co2_emissions,
        )

        if validate_record(record):
            records.append(record)

    return records


def write_csv(records: Sequence[VoyageRecord], output_path: Path) -> None:
    """Serialize VoyageRecord instances to CSV matching schema column ordering.

    Args:
        records: Sequence of VoyageRecord dataclasses.
        output_path: Destination filesystem path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec.to_dict())


def main() -> int:
    """CLI entrypoint for synthetic voyage dataset generator."""
    default_cfg = config.get_default_config()

    parser = argparse.ArgumentParser(
        description="Generate synthetic voyage telemetry records for green fleet modeling."
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=2000,
        help="Number of synthetic voyage records to generate (default: 2000).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_cfg.data_paths.sample_voyages_file,
        help="Destination path for the generated CSV file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=default_cfg.random_seed,
        help="Random seed for deterministic generation (default: 42).",
    )

    args = parser.parse_args()

    # Configure centralized logger
    logger = logging_config.configure_logging(log_file_name="dataset_generator.log")
    logger.info("Initiating synthetic dataset generation")
    logger.info("Rows requested: %d | Seed: %d", args.rows, args.seed)
    logger.info("Target destination: %s", args.output)

    if args.rows <= 0:
        logger.error("Invalid rows parameter: %d. Must be greater than 0.", args.rows)
        return 1

    records = generate_synthetic_voyages(num_rows=args.rows, seed=args.seed)
    write_csv(records, args.output)

    logger.info("Successfully generated %d voyage records to %s", len(records), args.output)
    logger.info("Dataset generation completed with status: SUCCESS")

    print(f"Generated {len(records)} synthetic voyage records at: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
