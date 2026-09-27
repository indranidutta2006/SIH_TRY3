"""Unified production fuel prediction service wrapping the frozen PhysicsInformedQIFCPRegressor.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Provides a clean, domain-validated abstraction decoupling fleet-level optimization engines
from QIFCP predictive model internals.
"""

from collections.abc import Sequence
from dataclasses import asdict
import logging
from pathlib import Path
from typing import Any, Final

import numpy as np
from sklearn.base import RegressorMixin

from contracts.constants import SUPPORTED_FUELS, FuelType
from contracts.exceptions import DataValidationError, PredictionError
from contracts.schemas import PredictionResult, VoyageRecord
from src.ingestion.dataset_loader import CSVDatasetLoader
from src.ingestion.feature_pipeline import FeatureEngineeringPipeline
from src.ingestion.validators import VALID_FUEL_TYPES, VALID_VESSEL_TYPES
from src.physics.fuel_physics_engine import (
    ELECTRICAL_EFFICIENCY,
    MaritimeFuelPhysicsEngine,
    normalize_fuel_name,
)
from src.prediction.predictor import PredictionInferenceEngine

logger = logging.getLogger("maritime_system")

# Module-level singleton cache for zero-overhead swarm / solver evaluations
_GLOBAL_FUEL_SERVICE: "FuelPredictionService | None" = None


class FuelPredictionService:
    """Production fuel prediction service wrapping the frozen PhysicsInformedQIFCPRegressor.

    Ensures that fleet optimization algorithms depend on a unified service interface
    rather than importing or reimplementing QIFCP internals directly.
    """

    def __init__(
        self,
        predictor: PredictionInferenceEngine | RegressorMixin | None = None,
        dataset_path: Path | str = "data/raw/voyages_sample.csv",
        random_state: int = 42,
    ) -> None:
        """Initialize fuel prediction service with calibrated predictor.

        Args:
            predictor: Optional pre-calibrated PredictionInferenceEngine or regressor.
            dataset_path: Path to canonical voyage dataset for calibration if predictor is None.
            random_state: Deterministic seed for reproducible model instantiation.
        """
        self.dataset_path = Path(dataset_path)
        self.random_state = random_state
        self.logger = logger
        self.physics_engine = MaritimeFuelPhysicsEngine()
        self._last_energy_mwh: float = 0.0

        # Memory cache for identical candidate evaluations during optimization
        self._memo_cache: dict[tuple[Any, ...], float] = {}

        if predictor is not None:
            if isinstance(predictor, PredictionInferenceEngine):
                self._engine = predictor
            else:
                self._engine = PredictionInferenceEngine(
                    model=predictor,
                    model_name="physics_residual_qifcp",
                    target_mode="absolute",
                )
        else:
            self._engine = self._initialize_canonical_qifcp()

    @property
    def predictor_engine(self) -> PredictionInferenceEngine:
        """Access underlying PredictionInferenceEngine."""
        return self._engine

    @property
    def model(self) -> RegressorMixin | None:
        """Access underlying fitted scikit-learn regressor."""
        return self._engine.model

    @property
    def model_name(self) -> str:
        """Name of the active prediction model."""
        return getattr(self._engine, "model_name", "physics_residual_qifcp")

    @property
    def last_energy_mwh(self) -> float:
        """Energy in MWh from the most recent calculation (e.g. ShorePower)."""
        return self._last_energy_mwh

    def _initialize_canonical_qifcp(self) -> PredictionInferenceEngine:
        """Instantiate and calibrate the frozen PhysicsInformedQIFCPRegressor."""
        if not self.dataset_path.exists():
            raise PredictionError(
                f"Canonical dataset file does not exist: {self.dataset_path.resolve()}",
                details={"dataset_path": str(self.dataset_path)},
            )

        records = CSVDatasetLoader().load_data(self.dataset_path)
        valid_records = [r for r in records if r.fuel_consumption is not None]
        if not valid_records:
            raise PredictionError("No valid voyage records with fuel_consumption found.")

        pipeline = FeatureEngineeringPipeline()
        X_df, _ = pipeline.get_training_features_and_target(valid_records, encode_categoricals=True)
        feature_columns = list(X_df.columns)
        X = X_df.to_numpy(dtype=float)
        y = np.array([float(r.fuel_consumption) for r in valid_records], dtype=float)

        from src.prediction.qifcp import PhysicsInformedQIFCPRegressor

        # Frozen configuration: K=3, M=15, adaptive entanglement, grouped gamma, lambda_residual=1.0
        model = PhysicsInformedQIFCPRegressor(
            lambda_residual=1.0,
            n_entanglement_pairs=15,
            harmonic_order=3,
            qifcp_mode="v2",
            entanglement_mode="adaptive",
            gamma_mode="grouped",
            feature_names=feature_columns,
            random_state=self.random_state,
        )
        model.fit(X, y)
        self.logger.info(
            "Calibrated canonical PhysicsInformedQIFCPRegressor on %d records (features: %d)",
            len(valid_records),
            len(feature_columns),
        )

        return PredictionInferenceEngine(
            model=model,
            model_name="physics_residual_qifcp",
            feature_columns=feature_columns,
            target_mode="absolute",
        )

    def validate_inputs(
        self,
        distance_nm: float,
        speed_knots: float,
        cargo_tons: float,
        weather_factor: float,
        fuel_type: str,
        vessel_dwt: float | None = None,
        vessel_type: str = "Bulk Carrier",
    ) -> str:
        """Validate candidate operational parameters against physical and schema boundaries.

        Raises:
            DataValidationError: If any operational input is non-finite, negative, or invalid.
        """
        if not np.isfinite(distance_nm) or distance_nm <= 0.0:
            raise DataValidationError(
                f"Voyage distance must be positive and finite, got {distance_nm} NM."
            )
        if not np.isfinite(speed_knots) or speed_knots <= 0.0:
            raise DataValidationError(
                f"Vessel speed must be positive and finite, got {speed_knots} knots."
            )
        if not np.isfinite(cargo_tons) or cargo_tons < 0.0:
            raise DataValidationError(
                f"Cargo payload must be non-negative and finite, got {cargo_tons} tons."
            )
        if not np.isfinite(weather_factor) or weather_factor < 1.0:
            raise DataValidationError(
                f"Weather factor must be >= 1.0 and finite, got {weather_factor}."
            )

        norm_fuel = normalize_fuel_name(fuel_type)
        if norm_fuel not in VALID_FUEL_TYPES:
            raise DataValidationError(
                f"Unsupported fuel type '{fuel_type}' (normalized: '{norm_fuel}'). "
                f"Supported: {sorted(VALID_FUEL_TYPES)}"
            )

        if vessel_dwt is not None:
            if not np.isfinite(vessel_dwt) or vessel_dwt <= 0.0:
                raise DataValidationError(
                    f"Vessel DWT must be positive and finite, got {vessel_dwt} tons."
                )
            if cargo_tons > vessel_dwt * 1.05:
                raise DataValidationError(
                    f"Cargo payload ({cargo_tons} t) cannot exceed vessel capacity ({vessel_dwt} t)."
                )

        return norm_fuel

    def calculate_fuel(
        self,
        distance_nm: float,
        speed_knots: float,
        cargo_tons: float,
        weather_factor: float,
        fuel_type: str,
        vessel_dwt: float | None = None,
        vessel_type: str = "Bulk Carrier",
        sea_state: int = 3,
        admiralty_coeff: float | None = None,
        **kwargs: Any,
    ) -> float:
        """Predict fuel consumption (metric tons) for a single candidate voyage configuration.

        Args:
            distance_nm: Voyage distance in nautical miles.
            speed_knots: Operational speed in knots.
            cargo_tons: Cargo payload in metric tons.
            weather_factor: Weather severity multiplier (>= 1.0).
            fuel_type: Marine fuel token (Diesel, LNG, Methanol, Hydrogen, Ammonia, ShorePower).
            vessel_dwt: Optional vessel deadweight tonnage (defaults to 1.25 * cargo_tons).
            vessel_type: Commercial vessel classification (default Bulk Carrier).
            sea_state: Sea state code (default 3).
            admiralty_coeff: Optional hydrodynamic coefficient.

        Returns:
            Predicted fuel consumption in metric tons.
        """
        norm_fuel = self.validate_inputs(
            distance_nm=distance_nm,
            speed_knots=speed_knots,
            cargo_tons=cargo_tons,
            weather_factor=weather_factor,
            fuel_type=fuel_type,
            vessel_dwt=vessel_dwt,
            vessel_type=vessel_type,
        )

        # ShorePower check: 0.0 metric tons combusted; track electricity MWh
        if norm_fuel == FuelType.SHORE_POWER.value:
            dwt_eff = float(vessel_dwt if vessel_dwt is not None and vessel_dwt > 0 else max(cargo_tons * 1.25, 10000.0))
            hours = distance_nm / speed_knots
            p_kw = (dwt_eff ** 0.6) * 100.0 * 0.05
            energy_mwh = (p_kw * hours / 1000.0) / ELECTRICAL_EFFICIENCY
            self._last_energy_mwh = float(energy_mwh)
            return 0.0

        # Optimization memoization key
        eff_dwt = float(vessel_dwt if vessel_dwt is not None and vessel_dwt > 0 else max(cargo_tons * 1.25, 10000.0))
        memo_key = (
            round(distance_nm, 1),
            round(speed_knots, 2),
            round(cargo_tons, 1),
            round(weather_factor, 2),
            norm_fuel,
            round(eff_dwt, 1),
            vessel_type,
            sea_state,
        )
        if memo_key in self._memo_cache:
            return self._memo_cache[memo_key]

        # Construct single VoyageRecord and pass into inference engine
        hours_at_sea = distance_nm / max(speed_knots, 1e-4)
        record = VoyageRecord(
            voyage_id=f"PRED-{norm_fuel}",
            vessel_id="VSL-PREDICT",
            vessel_type=vessel_type if vessel_type in VALID_VESSEL_TYPES else "Bulk Carrier",
            vessel_dwt=eff_dwt,
            cargo_tons=cargo_tons,
            distance_nm=distance_nm,
            speed_knots=speed_knots,
            hours_at_sea=hours_at_sea,
            fuel_type=norm_fuel,
            weather_factor=weather_factor,
            sea_state=sea_state,
            data_source="fleet_optimizer",
            is_synthetic=True,
            fuel_consumption=None,
            co2_emissions=None,
        )

        pred_res = self._engine.predict([record])
        fuel_pred = float(pred_res[0].predicted_fuel_consumption)

        # Sanity validation on predictor output
        if not np.isfinite(fuel_pred) or fuel_pred < 0.0:
            raise PredictionError(
                f"Predictor returned non-finite or negative fuel consumption: {fuel_pred} metric tons.",
                details={"predicted_fuel": fuel_pred, "inputs": memo_key},
            )

        self._memo_cache[memo_key] = fuel_pred
        return fuel_pred

    def predict_voyage(self, record: VoyageRecord) -> float:
        """Predict fuel consumption (metric tons) for a single VoyageRecord.

        Args:
            record: Fully populated VoyageRecord.

        Returns:
            Predicted fuel consumption in metric tons.
        """
        results = self.predict_voyages([record])
        return results[0]

    def predict_voyages(self, records: Sequence[VoyageRecord]) -> list[float]:
        """Predict fuel consumption (metric tons) for a sequence of VoyageRecords.

        Args:
            records: Sequence of VoyageRecord instances.

        Returns:
            List of predicted fuel consumption values in metric tons.
        """
        if not records:
            return []

        for rec in records:
            self.validate_inputs(
                distance_nm=float(rec.distance_nm),
                speed_knots=float(rec.speed_knots),
                cargo_tons=float(rec.cargo_tons),
                weather_factor=float(rec.weather_factor),
                fuel_type=str(rec.fuel_type),
                vessel_dwt=float(rec.vessel_dwt) if rec.vessel_dwt is not None else None,
                vessel_type=str(rec.vessel_type),
            )

        pred_results = self._engine.predict(records)
        fuel_vals: list[float] = []
        for pr in pred_results:
            val = float(pr.predicted_fuel_consumption)
            if not np.isfinite(val) or val < 0.0:
                raise PredictionError(
                    f"Predictor returned invalid fuel consumption: {val}",
                    details={"result": asdict(pr)},
                )
            fuel_vals.append(val)

        return fuel_vals

    def predict(self, target: Any, **kwargs: Any) -> Any:
        """Polymorphic prediction interface for optimization and dashboard consumers.

        Accepts:
            - VoyageRecord -> returns PredictionResult
            - Sequence[VoyageRecord] -> returns list[PredictionResult]
            - Keyword arguments / operational parameters -> returns float fuel consumption
        """
        if isinstance(target, VoyageRecord):
            res = self._engine.predict([target])
            return res[0]
        elif isinstance(target, Sequence) and len(target) > 0 and isinstance(target[0], VoyageRecord):
            return self._engine.predict(target)
        elif isinstance(target, dict):
            params = {**target, **kwargs}
            return self.calculate_fuel(**params)
        elif hasattr(target, "route_distance"):
            # OptimizationScenario or similar scenario object
            speed = float(kwargs.get("speed_knots", 14.0))
            fuel_type = str(kwargs.get("fuel_type", "Diesel"))
            cargo_tons = float(kwargs.get("cargo_tons", getattr(target, "cargo_demand", 40000.0)))
            vessel_dwt = float(kwargs.get("vessel_dwt", max(cargo_tons * 1.25, 50000.0)))
            return self.calculate_fuel(
                distance_nm=float(target.route_distance),
                speed_knots=speed,
                cargo_tons=cargo_tons,
                weather_factor=float(target.weather_factor),
                fuel_type=fuel_type,
                vessel_dwt=vessel_dwt,
                vessel_type=getattr(target, "vessel_type", "Bulk Carrier"),
            )
        else:
            raise TypeError(f"Unsupported target type for FuelPredictionService.predict: {type(target)}")


def get_fuel_prediction_service() -> FuelPredictionService:
    """Retrieve or lazily initialize the shared global FuelPredictionService singleton."""
    global _GLOBAL_FUEL_SERVICE
    if _GLOBAL_FUEL_SERVICE is None:
        _GLOBAL_FUEL_SERVICE = FuelPredictionService()
    return _GLOBAL_FUEL_SERVICE
