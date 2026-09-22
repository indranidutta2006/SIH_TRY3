"""Model explainability module extracting global and local feature contributions.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 2B explainability engine extracting native scikit-learn feature importances for
RandomForest and HistGradientBoosting regressors without heavy external libraries like SHAP.
"""

from collections.abc import Sequence
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.inspection import permutation_importance

from contracts.exceptions import PredictionError
from contracts.schemas import VoyageRecord
from src.ingestion.feature_pipeline import FeatureEngineeringPipeline

logger = logging.getLogger("maritime_system")


class PredictionExplainer:
    """Provides global and local prediction explanations for maritime fuel regressors."""

    def __init__(
        self,
        model: RegressorMixin,
        feature_names: list[str],
        background_X: pd.DataFrame | None = None,
        background_y: pd.Series | None = None,
        feature_pipeline: FeatureEngineeringPipeline | None = None,
        random_state: int = 42,
    ) -> None:
        """Initialize explainer with trained model and feature definitions.

        Args:
            model: Trained scikit-learn regressor (RandomForest or HistGradientBoosting).
            feature_names: Ordered list of feature column names used during training.
            background_X: Optional reference feature DataFrame for permutation importance or baselines.
            background_y: Optional reference target series.
            feature_pipeline: Optional FeatureEngineeringPipeline instance for VoyageRecord parsing.
            random_state: Seed for deterministic permutation importance.
        """
        self.model = model
        self.feature_names = list(feature_names)
        self.background_X = background_X
        self.background_y = background_y
        self.pipeline = feature_pipeline or FeatureEngineeringPipeline()
        self.random_state = random_state
        self.logger = logger

        # Precompute reference statistics (mean and std) for local explanations
        if self.background_X is not None and not self.background_X.empty:
            aligned_bg = self.background_X[self.feature_names]
            self._feature_means = aligned_bg.mean().to_dict()
            self._feature_stds = aligned_bg.std().replace(0.0, 1.0).fillna(1.0).to_dict()
        else:
            self._feature_means = {feat: 0.0 for feat in self.feature_names}
            self._feature_stds = {feat: 1.0 for feat in self.feature_names}

    def get_feature_importances(
        self,
        X: pd.DataFrame | None = None,
        y: pd.Series | None = None,
    ) -> dict[str, float]:
        """Extract and normalize model feature importances summing strictly to 1.0.

        Supports:
        - RandomForestRegressor: uses native MDI feature_importances_.
        - HistGradientBoostingRegressor: uses permutation_importance if data is provided,
          or extracts split-frequency distributions from internal tree predictors.

        Args:
            X: Optional feature matrix override for permutation importance.
            y: Optional target series override for permutation importance.

        Returns:
            Dictionary mapping feature name to normalized importance (sum == 1.0).

        Raises:
            PredictionError: If importances cannot be derived.
        """
        n_feats = len(self.feature_names)
        raw_importances = np.zeros(n_feats, dtype=float)

        # 1. Native feature_importances_ attribute (e.g. RandomForestRegressor)
        if hasattr(self.model, "feature_importances_"):
            raw_importances = np.asarray(self.model.feature_importances_, dtype=float)

        # 2. HistGradientBoostingRegressor or models without feature_importances_
        else:
            eval_X = X if X is not None else self.background_X
            eval_y = y if y is not None else self.background_y

            if eval_X is not None and eval_y is not None and not eval_X.empty:
                # Use standard scikit-learn permutation importance
                aligned_X = eval_X[self.feature_names]
                perm_res = permutation_importance(
                    self.model,
                    aligned_X,
                    eval_y,
                    n_repeats=5,
                    random_state=self.random_state,
                )
                raw_importances = np.maximum(perm_res.importances_mean, 0.0)
            elif hasattr(self.model, "_predictors"):
                # Extract internal tree node split frequencies
                split_counts = np.zeros(n_feats, dtype=float)
                for predictor_row in self.model._predictors:
                    for tree in predictor_row:
                        if hasattr(tree, "nodes"):
                            for node in tree.nodes:
                                if not node["is_leaf"]:
                                    f_idx = int(node["feature_idx"])
                                    if 0 <= f_idx < n_feats:
                                        split_counts[f_idx] += 1.0
                raw_importances = split_counts
            elif hasattr(self.model, "coef_"):
                # Linear model coefficients absolute magnitude
                raw_importances = np.abs(np.asarray(self.model.coef_, dtype=float))
            else:
                raise PredictionError(
                    f"Model of type {type(self.model).__name__} does not expose feature importances.",
                    details={"model_type": str(type(self.model))},
                )

        # Normalize so sum(importances) == 1.0
        total = float(np.sum(raw_importances))
        if total > 0.0:
            normalized = raw_importances / total
        else:
            normalized = np.full(n_feats, 1.0 / max(n_feats, 1), dtype=float)

        importance_dict = {
            feat: round(float(imp), 6)
            for feat, imp in zip(self.feature_names, normalized, strict=False)
        }

        # Ensure exact sum == 1.0 by assigning residual to highest importance feature
        diff = 1.0 - sum(importance_dict.values())
        if abs(diff) > 1e-9:
            max_feat = max(importance_dict, key=importance_dict.get)
            importance_dict[max_feat] = round(importance_dict[max_feat] + diff, 6)

        return importance_dict

    def get_global_explanation(
        self,
        top_k: int = 10,
        X: pd.DataFrame | None = None,
        y: pd.Series | None = None,
    ) -> list[dict[str, Any]]:
        """Generate a global feature importance report ranking the top influential features.

        Args:
            top_k: Maximum number of top features to return (default 10).
            X: Optional feature matrix for importance computation.
            y: Optional target vector.

        Returns:
            List of dictionaries with feature name, importance, and rank.
        """
        importances = self.get_feature_importances(X=X, y=y)
        sorted_features = sorted(importances.items(), key=lambda item: item[1], reverse=True)

        report = [
            {
                "rank": rank,
                "feature": feat,
                "importance": round(imp, 4),
                "importance_percentage": round(imp * 100.0, 2),
            }
            for rank, (feat, imp) in enumerate(sorted_features[:top_k], start=1)
        ]
        return report

    def explain_voyage(
        self,
        voyage_or_features: VoyageRecord | pd.DataFrame | pd.Series,
        top_n: int = 3,
    ) -> dict[str, Any]:
        """Explain the local prediction drivers for a single voyage observation.

        Computes relative deviations against reference statistics multiplied by global
        feature importances to isolate upward (positive) and downward (negative) fuel drivers.

        Args:
            voyage_or_features: VoyageRecord dataclass or single-row feature DataFrame/Series.
            top_n: Number of top drivers to isolate in positive and negative directions.

        Returns:
            Dictionary containing:
            - top_positive_drivers: Features driving consumption higher than fleet average.
            - top_negative_drivers: Features driving consumption lower than fleet average.
            - dominant_factor: The single most impactful feature for this voyage.
        """
        # 1. Resolve feature vector
        if isinstance(voyage_or_features, VoyageRecord):
            x_df = self.pipeline.get_inference_features(
                [voyage_or_features], encode_categoricals=True
            )
            feature_row = x_df.iloc[0]
        elif isinstance(voyage_or_features, pd.DataFrame):
            feature_row = voyage_or_features.iloc[0]
        else:
            feature_row = voyage_or_features

        importances = self.get_feature_importances()

        impacts: list[dict[str, Any]] = []
        for feat in self.feature_names:
            if feat not in feature_row:
                continue

            val = float(feature_row[feat])
            mean_val = float(self._feature_means.get(feat, 0.0))
            std_val = float(self._feature_stds.get(feat, 1.0))
            imp = importances.get(feat, 0.0)

            # Standardized z-score deviation
            deviation = (val - mean_val) / (std_val if std_val > 0.0 else 1.0)
            impact_score = imp * deviation

            impacts.append(
                {
                    "feature": feat,
                    "impact_score": round(float(impact_score), 4),
                    "feature_value": round(val, 4),
                    "baseline_mean": round(mean_val, 4),
                }
            )

        # Separate positive and negative drivers
        pos_drivers = [d for d in impacts if d["impact_score"] > 0.0]
        neg_drivers = [d for d in impacts if d["impact_score"] < 0.0]

        # Sort positive drivers descending (highest upward pressure on fuel)
        pos_drivers.sort(key=lambda d: d["impact_score"], reverse=True)
        # Sort negative drivers ascending (highest downward relief on fuel)
        neg_drivers.sort(key=lambda d: d["impact_score"])

        # Determine dominant factor by absolute impact magnitude
        dominant = max(impacts, key=lambda d: abs(d["impact_score"])) if impacts else None

        return {
            "top_positive_drivers": [
                {"feature": d["feature"], "impact_score": d["impact_score"]}
                for d in pos_drivers[:top_n]
            ],
            "top_negative_drivers": [
                {"feature": d["feature"], "impact_score": d["impact_score"]}
                for d in neg_drivers[:top_n]
            ],
            "dominant_factor": dominant["feature"] if dominant else "none",
        }
