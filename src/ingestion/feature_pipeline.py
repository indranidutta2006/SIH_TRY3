"""Feature engineering pipeline for maritime fuel consumption prediction.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 1B feature extraction engine transforming validated voyage telemetry into model-ready
training and inference matrices with strict target leakage guardrails.
"""

from collections.abc import Sequence
from dataclasses import asdict
import logging
from typing import Final

import numpy as np
import pandas as pd

from contracts.exceptions import DataValidationError
from contracts.schemas import VoyageRecord
from src.ingestion.validators import VALID_FUEL_TYPES, VALID_VESSEL_TYPES

logger = logging.getLogger("maritime_system")

# Target and direct emissions leakage tokens strictly prohibited from predictor matrix X
FORBIDDEN_LEAKAGE_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "fuel_consumption",
        "co2_emissions",
        "fuel_per_nm",
        "fuel_per_hour",
        "co2_per_nm",
        "co2_per_ton_nm",
        "fuel_rate",
        "co2_rate",
        "fuel_intensity",
        "co2_intensity",
    }
)

IDENTIFIER_COLUMNS: Final[frozenset[str]] = frozenset(
    {"voyage_id", "vessel_id", "data_source", "is_synthetic"}
)


def _records_to_dataframe(records_or_df: Sequence[VoyageRecord] | pd.DataFrame) -> pd.DataFrame:
    """Convert a sequence of VoyageRecord instances or DataFrame to a clean DataFrame copy.

    Args:
        records_or_df: Sequence of VoyageRecord dataclasses or existing pandas DataFrame.

    Returns:
        New pandas DataFrame instance.
    """
    if isinstance(records_or_df, pd.DataFrame):
        return records_or_df.copy()
    if not records_or_df:
        return pd.DataFrame()
    return pd.DataFrame([asdict(rec) for rec in records_or_df])


class FeatureEngineeringPipeline:
    """Constructs naval architectural, kinematic, and operational features for predictive modeling."""

    def __init__(
        self,
        allowed_vessel_types: frozenset[str] = VALID_VESSEL_TYPES,
        allowed_fuel_types: frozenset[str] = VALID_FUEL_TYPES,
    ) -> None:
        """Initialize the feature pipeline with deterministic categorical vocabularies.

        Args:
            allowed_vessel_types: Domain vocabulary for vessel categories.
            allowed_fuel_types: Domain vocabulary for marine fuel categories.
        """
        self.allowed_vessel_types = sorted(allowed_vessel_types)
        self.allowed_fuel_types = sorted(allowed_fuel_types)
        self.logger = logger

    def check_for_target_leakage(self, feature_df: pd.DataFrame) -> None:
        """Inspect a feature matrix and raise DataValidationError if target leakage is detected.

        Args:
            feature_df: DataFrame of candidate model features.

        Raises:
            DataValidationError: If any forbidden target or emissions columns are present.
        """
        cols = set(feature_df.columns)
        leakage_detected = cols.intersection(FORBIDDEN_LEAKAGE_COLUMNS)

        # Check also for heuristic prefixes/suffixes matching target leakage
        pattern_leakage = {
            col
            for col in cols
            if (
                col.startswith("fuel_")
                or col.startswith("co2_")
                or "fuel_consumption" in col
                or "co2_emissions" in col
            )
            and col not in {"fuel_type"}
            and not col.startswith("fuel_type_")
        }
        all_leakage = leakage_detected.union(pattern_leakage)

        if all_leakage:
            msg = (
                f"Target leakage violation: feature matrix X contains forbidden target/emissions "
                f"columns: {sorted(all_leakage)}. Target leakage violates predictive integrity."
            )
            self.logger.error(msg)
            raise DataValidationError(
                msg,
                details={"leakage_columns": sorted(all_leakage)},
            )

    def create_features(
        self,
        data: Sequence[VoyageRecord] | pd.DataFrame,
    ) -> pd.DataFrame:
        """Engineer naval architectural, hydrodynamic, and operational interaction features.

        Args:
            data: Sequence of VoyageRecord instances or DataFrame.

        Returns:
            DataFrame containing original columns augmented with engineered features.
        """
        df = _records_to_dataframe(data)
        if df.empty:
            return df

        # 1. Cargo Payload Utilization Ratios (with zero-division protection)
        dwt_safe = np.maximum(df["vessel_dwt"].to_numpy(dtype=float), 1e-6)
        cargo_safe = np.maximum(df["cargo_tons"].to_numpy(dtype=float), 0.0)
        cargo_ratio = np.clip(cargo_safe / dwt_safe, 0.0, 1.0)
        df["cargo_ratio"] = cargo_ratio
        df["cargo_utilization_pct"] = cargo_ratio * 100.0

        # 2. Transport Work (Cargo × Distance in ton-nautical miles)
        dist_safe = np.maximum(df["distance_nm"].to_numpy(dtype=float), 0.0)
        transport_work = cargo_safe * dist_safe
        df["transport_work"] = transport_work
        df["ton_nautical_miles"] = transport_work

        # 3. Hydrodynamic Power Proxy (Admiralty Coefficient Principle)
        # Displacement estimate: lightship estimated as ~20% of DWT plus deadweight payload
        speed_safe = np.maximum(df["speed_knots"].to_numpy(dtype=float), 0.0)
        displacement_est = 0.20 * dwt_safe + cargo_safe
        # Power is proportional to displacement^(2/3) * speed^3
        df["power_proxy"] = (displacement_est ** (2.0 / 3.0)) * (speed_safe ** 3.0)

        # 4. Kinematics & Speed Consistency Checks
        hours_safe = np.maximum(df["hours_at_sea"].to_numpy(dtype=float), 1e-4)
        implied_speed = dist_safe / hours_safe
        df["implied_speed"] = implied_speed
        df["speed_discrepancy"] = implied_speed - speed_safe

        # 5. Environmental & Severity Interactions
        weather_safe = np.maximum(df["weather_factor"].to_numpy(dtype=float), 1.0)
        sea_state_safe = np.maximum(df["sea_state"].to_numpy(dtype=float), 0.0)
        df["weather_speed_interaction"] = speed_safe * weather_safe
        df["weather_sea_interaction"] = weather_safe * (sea_state_safe + 1.0)

        return df

    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply deterministic one-hot encoding aligned with domain vocabularies.

        Args:
            df: DataFrame containing vessel_type and fuel_type columns.

        Returns:
            DataFrame with one-hot columns for vessel_type and fuel_type.
        """
        encoded_df = df.copy()

        # One-hot vessel types
        for v_type in self.allowed_vessel_types:
            col_name = f"vessel_type_{v_type.lower().replace(' ', '_')}"
            encoded_df[col_name] = (encoded_df["vessel_type"] == v_type).astype(int)

        # One-hot fuel types
        for f_type in self.allowed_fuel_types:
            col_name = f"fuel_type_{f_type.lower().replace(' ', '_')}"
            encoded_df[col_name] = (encoded_df["fuel_type"] == f_type).astype(int)

        # Drop original raw categorical strings
        return encoded_df.drop(columns=["vessel_type", "fuel_type"], errors="ignore")

    def get_training_features_and_target(
        self,
        data: Sequence[VoyageRecord] | pd.DataFrame,
        encode_categoricals: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Produce feature matrix X and regression target y with strict leakage guardrails.

        Args:
            data: Sequence of VoyageRecord instances or DataFrame.
            encode_categoricals: Whether to one-hot encode categorical features.

        Returns:
            Tuple of (feature DataFrame X, target Series y representing fuel_consumption).

        Raises:
            DataValidationError: If dataset is empty, target is missing, or leakage is detected.
        """
        raw_df = _records_to_dataframe(data)
        if raw_df.empty:
            raise DataValidationError("Cannot extract training features: dataset is empty.")

        if "fuel_consumption" not in raw_df.columns:
            raise DataValidationError(
                "Training dataset missing required target field: 'fuel_consumption'."
            )

        # Filter out rows where target is missing / null
        valid_target_mask = raw_df["fuel_consumption"].notna()
        if not valid_target_mask.any():
            raise DataValidationError(
                "Training dataset has 0 non-null values for 'fuel_consumption'."
            )

        clean_df = raw_df[valid_target_mask].copy()
        y: pd.Series = clean_df["fuel_consumption"].astype(float).copy()
        y.name = "fuel_consumption"

        # Engineer maritime physical features
        featured_df = self.create_features(clean_df)

        # Drop targets and identifiers from feature matrix X
        drop_cols = set(IDENTIFIER_COLUMNS).union(
            {"fuel_consumption", "co2_emissions"}
        )
        x_df = featured_df.drop(columns=[col for col in drop_cols if col in featured_df.columns])

        if encode_categoricals:
            x_df = self.encode_categoricals(x_df)

        # Strict target leakage check on X
        self.check_for_target_leakage(x_df)

        self.logger.info(
            "Prepared training set: X shape=%s, y shape=%s", x_df.shape, y.shape
        )
        return x_df, y

    def get_inference_features(
        self,
        data: Sequence[VoyageRecord] | pd.DataFrame,
        encode_categoricals: bool = True,
    ) -> pd.DataFrame:
        """Produce feature matrix X for unobserved or inference telemetry.

        Args:
            data: Sequence of VoyageRecord instances or DataFrame.
            encode_categoricals: Whether to one-hot encode categorical features.

        Returns:
            Feature matrix X with strict target leakage validation.

        Raises:
            DataValidationError: If dataset is empty or target leakage is detected.
        """
        raw_df = _records_to_dataframe(data)
        if raw_df.empty:
            raise DataValidationError("Cannot extract inference features: dataset is empty.")

        # Engineer maritime physical features
        featured_df = self.create_features(raw_df)

        # Drop targets, leakage columns, and identifiers if present in telemetry
        drop_cols = set(IDENTIFIER_COLUMNS).union(
            {"fuel_consumption", "co2_emissions"}
        )
        x_df = featured_df.drop(columns=[col for col in drop_cols if col in featured_df.columns])

        if encode_categoricals:
            x_df = self.encode_categoricals(x_df)

        # Strict target leakage check on X
        self.check_for_target_leakage(x_df)

        self.logger.info("Prepared inference set: X shape=%s", x_df.shape)
        return x_df
