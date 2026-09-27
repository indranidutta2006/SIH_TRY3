"""Quantum-Inspired Fuel Consumption Predictor (QIFCP) regressor.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Fulfills Objective 1 by implementing a Quantum-Inspired feature-encoding and entanglement
surrogate regressor whose hyperparameters are optimized using QPSO (Quantum-Behaved PSO).
Conforms strictly to scikit-learn's BaseEstimator and RegressorMixin interfaces.
"""

from collections.abc import Sequence
import logging
import time
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

from contracts.exceptions import PredictionError
from src.optimization.qpso import QPSOOptimizer
from src.physics.fuel_physics_engine import LHV_MJ_PER_KG

logger = logging.getLogger("maritime_system")


DEFAULT_27_FEATURE_NAMES = (
    "vessel_dwt", "cargo_tons", "distance_nm", "speed_knots",
    "weather_factor", "sea_state", "cargo_ratio", "cargo_utilization_pct",
    "transport_work", "ton_nautical_miles", "power_proxy", "implied_hours",
    "weather_speed_interaction", "weather_sea_interaction",
    "vessel_type_bulk_carrier", "vessel_type_container_ship",
    "vessel_type_general_cargo", "vessel_type_oil_tanker",
    "fuel_type_ammonia", "fuel_type_diesel", "fuel_type_hydrogen",
    "fuel_type_lng", "fuel_type_methanol", "fuel_type_shorepower",
    "source_group_short", "source_group_medium", "source_group_long",
)

HYDRODYNAMIC_FEATURES = frozenset({
    "vessel_dwt",
    "distance_nm",
    "transport_work",
    "ton_nautical_miles",
    "power_proxy",
})

ENVIRONMENTAL_FEATURES = frozenset({
    "weather_factor",
    "sea_state",
    "weather_speed_interaction",
    "weather_sea_interaction",
})


class QIFCPRegressor(BaseEstimator, RegressorMixin):
    """Quantum-Inspired Fuel Consumption Predictor with QPSO hyperparameter tuning.

    Encodes standardized hydro-meteorological features into quantum phase angles
    theta_j = arctan(gamma * z_j) (or theta_j = arctan(gamma_g * z_j) for grouped scaling)
    and superposition-entanglement correlation states, with closed-form quantum state
    projection weights and QPSO-optimized hyperparameters.
    """

    def __init__(
        self,
        gamma: float = 0.5,
        alpha_reg: float = 1.0,
        n_entanglement_pairs: int = 15,
        harmonic_order: int = 3,
        qifcp_mode: str = "v1",
        entanglement_mode: str = "random",
        gamma_mode: str = "global",
        gamma_hydrodynamic: float | None = None,
        gamma_operational: float | None = None,
        gamma_environment: float | None = None,
        feature_names: Sequence[str] | None = None,
        random_state: int = 42,
    ) -> None:
        """Initialize QIFCP regressor with structural hyperparameters.

        Args:
            gamma: Quantum phase scaling factor for arctan projection.
            alpha_reg: L2 regularization strength for quantum state projection.
            n_entanglement_pairs: Number of pairwise quantum correlation terms.
            harmonic_order: Maximum harmonic multiplier K for multi-harmonic basis (v2).
            qifcp_mode: Model variant architecture ('v1' for legacy single harmonic, 'v2' for multi-harmonic).
            entanglement_mode: Pair selection policy ('random' for static random sampling, 'adaptive' for deterministic relevance ranking).
            gamma_mode: Phase scaling architecture ('global' for scalar gamma, 'grouped' for domain gamma_g).
            gamma_hydrodynamic: Phase scaling factor for hydrodynamic predictors (displacement, distance, power).
            gamma_operational: Phase scaling factor for operational predictors (cargo, speed, vessel/fuel indicators).
            gamma_environment: Phase scaling factor for environmental predictors (weather, sea state, interactions).
            feature_names: Optional sequence of input feature names for deterministic group assignment.
            random_state: Deterministic random seed for reproducibility.
        """
        self.gamma = gamma
        self.alpha_reg = alpha_reg
        self.n_entanglement_pairs = n_entanglement_pairs
        self.harmonic_order = harmonic_order
        self.qifcp_mode = qifcp_mode
        self.entanglement_mode = entanglement_mode
        self.gamma_mode = gamma_mode
        self.gamma_hydrodynamic = gamma_hydrodynamic
        self.gamma_operational = gamma_operational
        self.gamma_environment = gamma_environment
        self.feature_names = tuple(feature_names) if feature_names is not None else None
        self.random_state = random_state

        # Validate parameters
        if self.harmonic_order < 1:
            raise ValueError("harmonic_order must be >= 1")
        if self.qifcp_mode not in ("v1", "v2"):
            raise ValueError("qifcp_mode must be 'v1' or 'v2'")
        if self.entanglement_mode not in ("random", "adaptive"):
            raise ValueError("entanglement_mode must be 'random' or 'adaptive'")
        if self.gamma_mode not in ("global", "grouped"):
            raise ValueError("gamma_mode must be 'global' or 'grouped'")
        if self.n_entanglement_pairs < 0:
            raise ValueError("n_entanglement_pairs must be >= 0")

        # Validate gamma values
        if self.gamma <= 0.0 or not np.isfinite(self.gamma):
            raise ValueError("gamma must be positive and finite")
        if self.alpha_reg < 0.0 or not np.isfinite(self.alpha_reg):
            raise ValueError("alpha_reg must be non-negative and finite")

        if self.gamma_mode == "grouped":
            for g_name, g_val in [
                ("gamma_hydrodynamic", self.gamma_hydrodynamic),
                ("gamma_operational", self.gamma_operational),
                ("gamma_environment", self.gamma_environment),
            ]:
                if g_val is not None and (g_val <= 0.0 or not np.isfinite(g_val)):
                    raise ValueError(f"{g_name} must be positive and finite")

        # Fitted attributes
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None
        self.entanglement_indices_: list[tuple[int, int]] = []
        self.tuning_history_: tuple[float, ...] = ()
        self.best_inner_val_score_: float | None = None
        self.n_objective_evaluations_: int | None = None

        # Expose internal architectural hyperparameters for external inspection
        self.harmonic_order_ = self.harmonic_order
        self.qifcp_mode_ = self.qifcp_mode
        self.entanglement_mode_ = self.entanglement_mode
        self.gamma_mode_ = self.gamma_mode

        # Expose group gamma attributes
        self.gamma_hydrodynamic_ = float(
            self.gamma_hydrodynamic if self.gamma_hydrodynamic is not None else self.gamma
        )
        self.gamma_operational_ = float(
            self.gamma_operational if self.gamma_operational is not None else self.gamma
        )
        self.gamma_environment_ = float(
            self.gamma_environment if self.gamma_environment is not None else self.gamma
        )

        self.phase_groups_: dict[str, list[str]] = {
            "hydrodynamic": [],
            "operational": [],
            "environmental": [],
        }
        self.phase_group_indices_: dict[str, list[int]] = {
            "hydrodynamic": [],
            "operational": [],
            "environmental": [],
        }
        self.feature_gamma_vector_: np.ndarray | None = None

    def _select_random_pairs(self, n_features: int) -> list[tuple[int, int]]:
        """Deterministically select feature index pairs for quantum entanglement representation."""
        rng = np.random.default_rng(self.random_state)
        all_pairs = [
            (i, j)
            for i in range(n_features)
            for j in range(i + 1, n_features)
        ]
        if not all_pairs:
            return []
        k = min(self.n_entanglement_pairs, len(all_pairs))
        chosen_indices = rng.choice(len(all_pairs), size=k, replace=False)
        return [all_pairs[idx] for idx in chosen_indices]

    def _select_adaptive_pairs(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> list[tuple[int, int]]:
        """Deterministically select top M feature interaction pairs using training statistics.

        Calculates a training-only pair relevance score based on Maximum Relevance Minimum Redundancy:
            pair_score(i, j) = |corr(x_i, y)| * |corr(x_j, y)| * (1.0 - |corr(x_i, x_j)|)
        If y is None or has zero variance, falls back to pairwise correlation magnitude |corr(x_i, x_j)|.
        Selection is strictly deterministic with lexicographical tie-breaking by (i, j).
        """
        n_samples, n_features = X.shape
        all_pairs = [
            (i, j)
            for i in range(n_features)
            for j in range(i + 1, n_features)
        ]
        if not all_pairs or self.n_entanglement_pairs <= 0:
            return []

        # Standardize features using training statistics
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        Z = (X - mean) / std

        # Target-aware relevance scores
        y_valid = False
        r_y = np.zeros(n_features, dtype=float)
        if y is not None:
            y_arr = np.asarray(y, dtype=float).ravel()
            if len(y_arr) == n_samples:
                std_y = float(np.std(y_arr))
                if std_y > 1e-8:
                    y_c = y_arr - float(np.mean(y_arr))
                    for i in range(n_features):
                        r_y[i] = float(np.abs(np.mean(Z[:, i] * y_c) / std_y))
                    y_valid = True

        scores: list[tuple[tuple[int, int], float]] = []
        for i, j in all_pairs:
            cov = float(np.abs(np.mean(Z[:, i] * Z[:, j])))
            if y_valid:
                s = float(r_y[i] * r_y[j] * (1.0 - cov))
            else:
                s = float(cov)
            scores.append(((i, j), s))

        # Sort descending by score; tie-break deterministically by (i, j)
        scores.sort(key=lambda item: (-item[1], item[0]))
        k = min(self.n_entanglement_pairs, len(all_pairs))
        return [pair for pair, _ in scores[:k]]

    def _select_entanglement_pairs(
        self,
        n_features: int,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
    ) -> list[tuple[int, int]]:
        """Select entanglement pairs according to configured entanglement_mode."""
        if self.entanglement_mode == "adaptive" and X is not None:
            return self._select_adaptive_pairs(X, y)
        return self._select_random_pairs(n_features)

    def _resolve_phase_groups(
        self, n_features: int, feature_names: Sequence[str] | None = None
    ) -> None:
        """Deterministically map features into exactly one of three domain phase groups."""
        if feature_names is not None:
            names = [str(c) for c in feature_names]
        elif self.feature_names is not None:
            names = [str(c) for c in self.feature_names]
        elif n_features == len(DEFAULT_27_FEATURE_NAMES):
            names = list(DEFAULT_27_FEATURE_NAMES)
        else:
            names = [f"feature_{i}" for i in range(n_features)]

        if len(names) != n_features:
            names = [f"feature_{i}" for i in range(n_features)]

        hydro_names: list[str] = []
        oper_names: list[str] = []
        env_names: list[str] = []
        hydro_indices: list[int] = []
        oper_indices: list[int] = []
        env_indices: list[int] = []

        is_synthetic_generic = all(name == f"feature_{k}" for k, name in enumerate(names))

        for i, name in enumerate(names):
            lower = name.lower()
            if name in HYDRODYNAMIC_FEATURES:
                hydro_names.append(name)
                hydro_indices.append(i)
            elif name in ENVIRONMENTAL_FEATURES:
                env_names.append(name)
                env_indices.append(i)
            elif any(kw in lower for kw in ["weather", "sea", "wave", "wind", "current", "swell", "temp"]):
                env_names.append(name)
                env_indices.append(i)
            elif any(kw in lower for kw in ["dwt", "displacement", "power_proxy", "transport_work", "ton_nautical_miles", "admiralty", "draft", "draught"]):
                hydro_names.append(name)
                hydro_indices.append(i)
            elif is_synthetic_generic:
                # Deterministic balanced 3-way partition for unnamed/synthetic features
                mod = i % 3
                if mod == 0:
                    hydro_names.append(name)
                    hydro_indices.append(i)
                elif mod == 1:
                    oper_names.append(name)
                    oper_indices.append(i)
                else:
                    env_names.append(name)
                    env_indices.append(i)
            else:
                oper_names.append(name)
                oper_indices.append(i)

        self.phase_groups_ = {
            "hydrodynamic": hydro_names,
            "operational": oper_names,
            "environmental": env_names,
        }
        self.phase_group_indices_ = {
            "hydrodynamic": hydro_indices,
            "operational": oper_indices,
            "environmental": env_indices,
        }

        # Build feature-level gamma vector
        gamma_vec = np.empty(n_features, dtype=float)
        if self.gamma_mode == "grouped":
            g_hydro = float(self.gamma_hydrodynamic if self.gamma_hydrodynamic is not None else self.gamma)
            g_oper = float(self.gamma_operational if self.gamma_operational is not None else self.gamma)
            g_env = float(self.gamma_environment if self.gamma_environment is not None else self.gamma)
            if hydro_indices:
                gamma_vec[hydro_indices] = g_hydro
            if oper_indices:
                gamma_vec[oper_indices] = g_oper
            if env_indices:
                gamma_vec[env_indices] = g_env
            self.gamma_hydrodynamic_ = g_hydro
            self.gamma_operational_ = g_oper
            self.gamma_environment_ = g_env
        else:
            gamma_vec.fill(self.gamma)
            self.gamma_hydrodynamic_ = float(self.gamma)
            self.gamma_operational_ = float(self.gamma)
            self.gamma_environment_ = float(self.gamma)

        self.feature_gamma_vector_ = gamma_vec

    def _quantum_feature_map(self, X: np.ndarray) -> np.ndarray:
        """Map standardized numeric predictors into quantum superposition and entanglement basis.

        Steps:
        1. Standardize using learned mean and std to prevent arctan saturation.
        2. Compute phase angles theta = arctan(gamma * Z) or arctan(gamma_vec * Z).
        3. Extract single-qubit basis states [cos(theta), sin(theta)].
        4. Extract multi-qubit entanglement correlation states [cos(theta_i + theta_j), sin(theta_i + theta_j)].
        5. Append bias unit column.
        """
        if self.mean_ is None or self.std_ is None:
            raise PredictionError("QIFCPRegressor must be fitted before computing quantum feature map.")

        # 1. Zero-mean, unit-variance standardization
        Z = (X - self.mean_) / self.std_

        # 2. Non-saturating phase angle transformation
        gamma_vec = self.feature_gamma_vector_ if self.feature_gamma_vector_ is not None else self.gamma
        theta = np.arctan(gamma_vec * Z)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        # 3. Superposition states (multi-harmonic if v2)
        if self.qifcp_mode == "v2":
            # Generate cos(k*theta) and sin(k*theta) for k=1..K
            harmonic_blocks = []
            for k in range(1, self.harmonic_order + 1):
                harmonic_blocks.append(np.cos(k * theta))
                harmonic_blocks.append(np.sin(k * theta))
            # Concatenate along feature dimension
            superposition = np.hstack(harmonic_blocks)
        else:
            # Original single harmonic
            superposition = np.hstack([cos_theta, sin_theta])

        # 4. Entanglement states (unchanged)
        entangle_terms = []
        for i, j in self.entanglement_indices_:
            sum_theta = theta[:, i] + theta[:, j]
            entangle_terms.append(np.cos(sum_theta))
            entangle_terms.append(np.sin(sum_theta))

        blocks = [np.ones((X.shape[0], 1), dtype=float), superposition]
        if entangle_terms:
            blocks.append(np.column_stack(entangle_terms))

        Phi = np.hstack(blocks)
        # Store feature count for later inspection
        self.n_quantum_features_ = Phi.shape[1]
        return Phi

    def fit(self, X: Any, y: Any) -> "QIFCPRegressor":
        """Fit quantum state projection weights via regularized normal equations.

        Args:
            X: Training feature matrix (n_samples, n_features).
            y: Target fuel consumption values (n_samples,).

        Returns:
            Fitted QIFCPRegressor instance.
        """
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).ravel()

        if X_arr.shape[0] != y_arr.shape[0]:
            raise PredictionError(
                f"Sample count mismatch: {X_arr.shape[0]} inputs vs {y_arr.shape[0]} targets."
            )

        # Learn feature standardizers
        self.mean_ = np.mean(X_arr, axis=0)
        self.std_ = np.std(X_arr, axis=0)
        # Avoid division by zero on invariant features
        self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)

        # Resolve feature groups and compute gamma vector
        if hasattr(X, "columns"):
            col_names = [str(c) for c in X.columns]
        elif self.feature_names is not None:
            col_names = [str(c) for c in self.feature_names]
        else:
            col_names = None

        self._resolve_phase_groups(n_features=X_arr.shape[1], feature_names=col_names)

        # Deterministic entanglement graph
        self.entanglement_indices_ = self._select_entanglement_pairs(
            X_arr.shape[1], X=X_arr, y=y_arr
        )

        # Quantum feature mapping
        Phi = self._quantum_feature_map(X_arr)

        # Closed-form regularized quantum state projection: (Phi^T Phi + alpha * I)^(-1) Phi^T y
        p = Phi.shape[1]
        reg_matrix = self.alpha_reg * np.eye(p)
        reg_matrix[0, 0] = 0.0  # Do not regularize bias intercept

        A = Phi.T @ Phi + reg_matrix
        b = Phi.T @ y_arr

        try:
            self.weights_ = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            self.weights_ = np.linalg.lstsq(A, b, rcond=None)[0]

        return self

    def predict(self, X: Any) -> np.ndarray:
        """Generate predicted fuel consumption using fitted quantum projection.

        Args:
            X: Input feature matrix.

        Returns:
            1D numpy array of predicted fuel consumption.
        """
        if self.weights_ is None:
            raise PredictionError("QIFCPRegressor is not fitted. Call fit() before predict().")

        X_arr = np.asarray(X, dtype=float)
        Phi = self._quantum_feature_map(X_arr)
        preds = Phi @ self.weights_
        return np.maximum(0.0, preds)  # Fuel consumption is strictly non-negative

    def compute_phase_distribution(self, X: Any) -> np.ndarray:
        """Diagnostic utility returning phase angles theta = arctan(gamma * Z).

        Useful for asserting that phase angles do not saturate across features.
        """
        if self.mean_ is None or self.std_ is None:
            raise PredictionError("Regressor must be fitted to compute phase distribution.")
        X_arr = np.asarray(X, dtype=float)
        Z = (X_arr - self.mean_) / self.std_
        return np.arctan(self.gamma * Z)

    def tune_with_qpso(
        self,
        X: Any,
        y: Any,
        groups: Any = None,
        population_size: int = 10,
        max_iterations: int = 15,
        val_split: float = 0.2,
    ) -> "QIFCPRegressor":
        """Optimize gamma and alpha_reg hyperparameters using QPSO search.

        Directly establishes synergy between Objective 1 (Prediction) and
        Objective 2 (QPSO Optimization) under an equal-budget protocol.

        Supports vessel-grouped (group-disjoint) inner validation splitting to eliminate
        vessel data leakage during the QPSO hyperparameter search.

        Args:
            X: Training feature matrix.
            y: Target values.
            groups: Optional group identifiers (e.g. vessel_id array) to enforce
                vessel-disjoint inner validation splitting.
            population_size: Number of QPSO particles.
            max_iterations: Maximum QPSO optimization iterations.
            val_split: Fraction of groups (or samples) held out for validation.
        """
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).ravel()
        n = len(X_arr)

        if groups is not None:
            groups_arr = np.asarray(groups)
            if len(groups_arr) != n:
                raise PredictionError(
                    f"Length of groups ({len(groups_arr)}) does not match samples ({n})."
                )
            unique_groups = np.unique(groups_arr)
            if len(unique_groups) >= 2:
                gss = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=self.random_state)
                tr_idx, val_idx = next(gss.split(X_arr, y_arr, groups=groups_arr))
                X_train, X_val = X_arr[tr_idx], X_arr[val_idx]
                y_train, y_val = y_arr[tr_idx], y_arr[val_idx]
                self.inner_train_indices_ = tr_idx
                self.inner_val_indices_ = val_idx
                logger.info(
                    "QIFCP inner vessel-grouped split: %d train (%d vessels), %d val (%d vessels)",
                    len(X_train),
                    len(np.unique(groups_arr[tr_idx])),
                    len(X_val),
                    len(np.unique(groups_arr[val_idx])),
                )
            else:
                logger.warning(
                    "Only 1 unique group detected in groups; falling back to positional validation split."
                )
                split_idx = int(n * (1.0 - val_split))
                X_train, X_val = X_arr[:split_idx], X_arr[split_idx:]
                y_train, y_val = y_arr[:split_idx], y_arr[split_idx:]
                self.inner_train_indices_ = np.arange(split_idx)
                self.inner_val_indices_ = np.arange(split_idx, n)
        else:
            split_idx = int(n * (1.0 - val_split))
            X_train, X_val = X_arr[:split_idx], X_arr[split_idx:]
            y_train, y_val = y_arr[:split_idx], y_arr[split_idx:]
            self.inner_train_indices_ = np.arange(split_idx)
            self.inner_val_indices_ = np.arange(split_idx, n)

        # Search bounds:
        if self.gamma_mode == "grouped":
            param_bounds = [
                (0.05, 2.0),   # gamma_hydrodynamic
                (0.05, 2.0),   # gamma_operational
                (0.05, 2.0),   # gamma_environment
                (0.01, 50.0),  # alpha_reg
            ]
        else:
            param_bounds = [
                (0.05, 2.0),   # gamma
                (0.01, 50.0),  # alpha_reg
            ]

        def validation_objective(params: np.ndarray) -> float:
            if self.gamma_mode == "grouped":
                g_hydro = float(params[0])
                g_oper = float(params[1])
                g_env = float(params[2])
                alpha_cand = float(params[3])
                cand_model = QIFCPRegressor(
                    gamma=self.gamma,
                    alpha_reg=alpha_cand,
                    n_entanglement_pairs=self.n_entanglement_pairs,
                    harmonic_order=self.harmonic_order,
                    qifcp_mode=self.qifcp_mode,
                    entanglement_mode=self.entanglement_mode,
                    gamma_mode="grouped",
                    gamma_hydrodynamic=g_hydro,
                    gamma_operational=g_oper,
                    gamma_environment=g_env,
                    feature_names=self.feature_names,
                    random_state=self.random_state,
                )
            else:
                gamma_cand, alpha_cand = float(params[0]), float(params[1])
                cand_model = QIFCPRegressor(
                    gamma=gamma_cand,
                    alpha_reg=alpha_cand,
                    n_entanglement_pairs=self.n_entanglement_pairs,
                    harmonic_order=self.harmonic_order,
                    qifcp_mode=self.qifcp_mode,
                    entanglement_mode=self.entanglement_mode,
                    gamma_mode="global",
                    feature_names=self.feature_names,
                    random_state=self.random_state,
                )
            cand_model.fit(X_train, y_train)
            preds = cand_model.predict(X_val)
            return float(root_mean_squared_error(y_val, preds))

        qpso = QPSOOptimizer(random_state=self.random_state)
        opt_res = qpso.optimize(
            objective_function=validation_objective,
            parameter_bounds=param_bounds,
            hyperparameters={
                "population_size": population_size,
                "max_iterations": max_iterations,
                "seed": self.random_state,
            },
        )

        if opt_res.best_vector:
            if self.gamma_mode == "grouped" and len(opt_res.best_vector) >= 4:
                self.gamma_hydrodynamic = float(opt_res.best_vector[0])
                self.gamma_operational = float(opt_res.best_vector[1])
                self.gamma_environment = float(opt_res.best_vector[2])
                self.alpha_reg = float(opt_res.best_vector[3])
                self.gamma_hydrodynamic_ = self.gamma_hydrodynamic
                self.gamma_operational_ = self.gamma_operational
                self.gamma_environment_ = self.gamma_environment
            elif len(opt_res.best_vector) >= 2:
                self.gamma = float(opt_res.best_vector[0])
                self.alpha_reg = float(opt_res.best_vector[1])
                self.gamma_hydrodynamic_ = self.gamma
                self.gamma_operational_ = self.gamma
                self.gamma_environment_ = self.gamma

        self.tuning_history_ = opt_res.history
        self.best_inner_val_score_ = float(opt_res.best_score)
        self.n_objective_evaluations_ = int(opt_res.n_evaluations)
        if self.gamma_mode == "grouped":
            logger.info(
                "QPSO tuned Grouped QIFCP hyperparameters: g_hydro=%.4f, g_oper=%.4f, g_env=%.4f, alpha_reg=%.4f (best RMSE: %.4f, evals: %d)",
                self.gamma_hydrodynamic_,
                self.gamma_operational_,
                self.gamma_environment_,
                self.alpha_reg,
                opt_res.best_score,
                self.n_objective_evaluations_,
            )
        else:
            logger.info(
                "QPSO tuned QIFCP hyperparameters: gamma=%.4f, alpha_reg=%.4f (best RMSE: %.4f, evals: %d)",
                self.gamma,
                self.alpha_reg,
                opt_res.best_score,
                self.n_objective_evaluations_,
            )

        # Fit final model on all data using optimal hyperparameters
        return self.fit(X_arr, y_arr)


class NavalPhysicsFuelBaseline(BaseEstimator, RegressorMixin):
    """First-principles naval architecture fuel consumption baseline regressor.

    Implements a deterministic physics baseline grounded in Admiralty resistance
    principles, auxiliary hotel load, and thermodynamic fuel lower heating values (LHV).
    Calibrates dimensionless component multipliers (c_prop, c_aux) on training data only
    using ordinary least squares without an intercept, guaranteeing zero fuel at zero voyage scale.
    """

    def __init__(
        self,
        c_adm_nominal: float = 500.0,
        thermal_efficiency: float = 0.45,
        feature_names: Sequence[str] | None = None,
    ) -> None:
        self.c_adm_nominal = c_adm_nominal
        self.thermal_efficiency = thermal_efficiency
        self.feature_names = tuple(feature_names) if feature_names is not None else None

        self.c_prop_: float | None = None
        self.c_aux_: float | None = None
        self.physics_parameters_: dict[str, Any] = {}
        self.calibration_time_seconds_: float = 0.0

    def _extract_physical_vectors(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute theoretical propulsion energy and auxiliary energy vectors in fuel tons."""
        X_arr = np.asarray(X, dtype=float)
        fnames = list(self.feature_names) if self.feature_names is not None else list(DEFAULT_27_FEATURE_NAMES)
        name_to_idx = {name: i for i, name in enumerate(fnames)}

        def get_col(name: str, fallback_idx: int) -> np.ndarray:
            if name in name_to_idx and name_to_idx[name] < X_arr.shape[1]:
                return X_arr[:, name_to_idx[name]]
            elif fallback_idx < X_arr.shape[1]:
                return X_arr[:, fallback_idx]
            return np.zeros(X_arr.shape[0], dtype=float)

        dwt = np.maximum(get_col("vessel_dwt", 0), 1.0)
        power_proxy = np.maximum(get_col("power_proxy", 10), 0.0)
        implied_hours = np.maximum(get_col("implied_hours", 11), 0.0)
        weather_factor = np.maximum(get_col("weather_factor", 4), 1.0)

        # LHV vector
        lhv_diesel = LHV_MJ_PER_KG.get("Diesel", 42.7)
        lhv_lng = LHV_MJ_PER_KG.get("LNG", 50.0)
        lhv_methanol = LHV_MJ_PER_KG.get("Methanol", 19.9)
        lhv_ammonia = LHV_MJ_PER_KG.get("Ammonia", 18.6)
        lhv_hydrogen = LHV_MJ_PER_KG.get("Hydrogen", 120.0)

        f_diesel = get_col("fuel_type_diesel", 19)
        f_lng = get_col("fuel_type_lng", 21)
        f_methanol = get_col("fuel_type_methanol", 22)
        f_ammonia = get_col("fuel_type_ammonia", 18)
        f_hydrogen = get_col("fuel_type_hydrogen", 20)

        lhv = (
            f_diesel * lhv_diesel
            + f_lng * lhv_lng
            + f_methanol * lhv_methanol
            + f_ammonia * lhv_ammonia
            + f_hydrogen * lhv_hydrogen
        )
        lhv = np.where(lhv > 0.0, lhv, lhv_diesel)

        p_prop = power_proxy / max(self.c_adm_nominal, 1.0)
        p_aux = 0.05 * (dwt ** 0.6) * 100.0

        conversion = (implied_hours * weather_factor * 3.6 / max(self.thermal_efficiency, 1e-4)) / (lhv * 1000.0)
        e_prop = p_prop * conversion
        e_aux = p_aux * conversion
        return e_prop, e_aux

    def fit(self, X: Any, y: Any) -> "NavalPhysicsFuelBaseline":
        t0 = time.perf_counter()
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        e_prop, e_aux = self._extract_physical_vectors(X_arr)
        A = np.column_stack([e_prop, e_aux])

        # Ordinary least squares without intercept: y ~ c_prop * e_prop + c_aux * e_aux
        c, residuals, rank, s = np.linalg.lstsq(A, y_arr, rcond=None)
        self.c_prop_ = float(max(c[0], 0.0))
        self.c_aux_ = float(max(c[1], 0.0))
        self.calibration_time_seconds_ = time.perf_counter() - t0

        self.physics_parameters_ = {
            "c_prop": self.c_prop_,
            "c_aux": self.c_aux_,
            "c_adm_nominal": self.c_adm_nominal,
            "thermal_efficiency": self.thermal_efficiency,
            "effective_c_adm": float(self.c_adm_nominal / self.c_prop_) if self.c_prop_ > 0 else self.c_adm_nominal,
            "effective_aux_scale": self.c_aux_,
            "units": "dimensionless scale multipliers on theoretical metric tons",
            "fitting_method": "Ordinary least squares without intercept on X_train only",
            "training_samples": int(len(y_arr)),
        }
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.c_prop_ is None or self.c_aux_ is None:
            raise PredictionError("NavalPhysicsFuelBaseline has not been fitted yet.")
        X_arr = np.asarray(X, dtype=float)
        e_prop, e_aux = self._extract_physical_vectors(X_arr)
        y_pred = self.c_prop_ * e_prop + self.c_aux_ * e_aux
        return np.maximum(y_pred, 0.0)


class PhysicsInformedQIFCPRegressor(BaseEstimator, RegressorMixin):
    """Phase 2D: Physics-informed residual learning regressor combining
    NavalPhysicsFuelBaseline with Adaptive Grouped QIFCP.

    Decomposition:
        y_hat_final = y_hat_phys + lambda_residual * r_hat_qifcp
    where y_hat_phys explains large-scale propulsion hydrodynamic power trends,
    and QIFCP learns remaining nonlinear residual patterns r = y - y_hat_phys.
    """

    def __init__(
        self,
        lambda_residual: float = 1.0,
        gamma: float = 0.5,
        alpha_reg: float = 1.0,
        n_entanglement_pairs: int = 15,
        harmonic_order: int = 3,
        qifcp_mode: str = "v2",
        entanglement_mode: str = "adaptive",
        gamma_mode: str = "grouped",
        gamma_hydrodynamic: float | None = None,
        gamma_operational: float | None = None,
        gamma_environment: float | None = None,
        feature_names: Sequence[str] | None = None,
        random_state: int = 42,
    ) -> None:
        self.lambda_residual = lambda_residual
        self.gamma = gamma
        self.alpha_reg = alpha_reg
        self.n_entanglement_pairs = n_entanglement_pairs
        self.harmonic_order = harmonic_order
        self.qifcp_mode = qifcp_mode
        self.entanglement_mode = entanglement_mode
        self.gamma_mode = gamma_mode
        self.gamma_hydrodynamic = gamma_hydrodynamic
        self.gamma_operational = gamma_operational
        self.gamma_environment = gamma_environment
        self.feature_names = tuple(feature_names) if feature_names is not None else None
        self.random_state = random_state

        self.physics_model_ = NavalPhysicsFuelBaseline(feature_names=self.feature_names)
        self.qifcp_residual_ = QIFCPRegressor(
            gamma=self.gamma,
            alpha_reg=self.alpha_reg,
            n_entanglement_pairs=self.n_entanglement_pairs,
            harmonic_order=self.harmonic_order,
            qifcp_mode=self.qifcp_mode,
            entanglement_mode=self.entanglement_mode,
            gamma_mode=self.gamma_mode,
            gamma_hydrodynamic=self.gamma_hydrodynamic,
            gamma_operational=self.gamma_operational,
            gamma_environment=self.gamma_environment,
            feature_names=self.feature_names,
            random_state=self.random_state,
        )

        self.physics_parameters_: dict[str, Any] = {}
        self.best_inner_val_score_: float | None = None
        self.tuning_history_: tuple[float, ...] = ()
        self.n_objective_evaluations_: int | None = None
        self.training_time_seconds_: float = 0.0

    @property
    def gamma_hydrodynamic_(self) -> float | None:
        return getattr(self.qifcp_residual_, "gamma_hydrodynamic_", None)

    @property
    def gamma_operational_(self) -> float | None:
        return getattr(self.qifcp_residual_, "gamma_operational_", None)

    @property
    def gamma_environment_(self) -> float | None:
        return getattr(self.qifcp_residual_, "gamma_environment_", None)

    @property
    def alpha_reg_(self) -> float | None:
        return getattr(self.qifcp_residual_, "alpha_reg", None)

    @property
    def n_quantum_features_(self) -> int | None:
        return getattr(self.qifcp_residual_, "n_quantum_features_", None)

    def fit(self, X: Any, y: Any) -> "PhysicsInformedQIFCPRegressor":
        t0 = time.perf_counter()
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        self.physics_model_.fit(X_arr, y_arr)
        self.physics_parameters_ = self.physics_model_.physics_parameters_
        y_phys_train = self.physics_model_.predict(X_arr)
        r_train = y_arr - y_phys_train

        self.qifcp_residual_.fit(X_arr, r_train)
        self.training_time_seconds_ = time.perf_counter() - t0
        return self

    def tune_with_qpso(
        self,
        X: Any,
        y: Any,
        groups: np.ndarray | None = None,
        population_size: int = 8,
        max_iterations: int = 10,
    ) -> "PhysicsInformedQIFCPRegressor":
        t0 = time.perf_counter()
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        # 1. Calibrate physics baseline on training partition only
        self.physics_model_.fit(X_arr, y_arr)
        self.physics_parameters_ = self.physics_model_.physics_parameters_
        y_phys_train = self.physics_model_.predict(X_arr)
        r_train = y_arr - y_phys_train

        # 2. Tune QIFCP residual model using inner vessel-disjoint validation
        self.qifcp_residual_.tune_with_qpso(
            X_arr,
            r_train,
            groups=groups,
            population_size=population_size,
            max_iterations=max_iterations,
        )

        self.best_inner_val_score_ = self.qifcp_residual_.best_inner_val_score_
        self.tuning_history_ = self.qifcp_residual_.tuning_history_
        self.n_objective_evaluations_ = self.qifcp_residual_.n_objective_evaluations_
        self.training_time_seconds_ = time.perf_counter() - t0
        return self

    def predict_components(self, X: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute individual predictions: (physics_baseline, residual_correction, final_prediction)."""
        X_arr = np.asarray(X, dtype=float)
        y_phys = self.physics_model_.predict(X_arr)
        r_hat = self.qifcp_residual_.predict(X_arr)
        y_final = np.maximum(y_phys + self.lambda_residual * r_hat, 0.0)
        return y_phys, r_hat, y_final

    def predict(self, X: Any) -> np.ndarray:
        _, _, y_final = self.predict_components(X)
        return y_final
