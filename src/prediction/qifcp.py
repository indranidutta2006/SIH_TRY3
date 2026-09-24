"""Quantum-Inspired Fuel Consumption Predictor (QIFCP) regressor.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Fulfills Objective 1 by implementing a Quantum-Inspired feature-encoding and entanglement
surrogate regressor whose hyperparameters are optimized using QPSO (Quantum-Behaved PSO).
Conforms strictly to scikit-learn's BaseEstimator and RegressorMixin interfaces.
"""

import logging
import time
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

from contracts.exceptions import PredictionError
from src.optimization.qpso import QPSOOptimizer

logger = logging.getLogger("maritime_system")


class QIFCPRegressor(BaseEstimator, RegressorMixin):
    """Quantum-Inspired Fuel Consumption Predictor with QPSO hyperparameter tuning.

    Encodes standardized hydro-meteorological features into quantum phase angles
    theta_j = arctan(gamma * z_j) and superposition-entanglement correlation states,
    with closed-form quantum state projection weights and QPSO-optimized hyperparameters.
    """

    def __init__(
        self,
        gamma: float = 0.5,
        alpha_reg: float = 1.0,
        n_entanglement_pairs: int = 15,
        random_state: int = 42,
    ) -> None:
        """Initialize QIFCP regressor with structural hyperparameters.

        Args:
            gamma: Quantum phase scaling factor for arctan projection.
            alpha_reg: L2 regularization strength for quantum state projection.
            n_entanglement_pairs: Number of pairwise quantum correlation terms.
            random_state: Deterministic random seed for reproducibility.
        """
        self.gamma = gamma
        self.alpha_reg = alpha_reg
        self.n_entanglement_pairs = n_entanglement_pairs
        self.random_state = random_state

        # Fitted attributes
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None
        self.entanglement_indices_: list[tuple[int, int]] = []
        self.tuning_history_: tuple[float, ...] = ()

    def _select_entanglement_pairs(self, n_features: int) -> list[tuple[int, int]]:
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

    def _quantum_feature_map(self, X: np.ndarray) -> np.ndarray:
        """Map standardized numeric predictors into quantum superposition and entanglement basis.

        Steps:
        1. Standardize using learned mean and std to prevent arctan saturation.
        2. Compute phase angles theta = arctan(gamma * Z).
        3. Extract single-qubit basis states [cos(theta), sin(theta)].
        4. Extract multi-qubit entanglement correlation states [cos(theta_i + theta_j), sin(theta_i + theta_j)].
        5. Append bias unit column.
        """
        if self.mean_ is None or self.std_ is None:
            raise PredictionError("QIFCPRegressor must be fitted before computing quantum feature map.")

        # 1. Zero-mean, unit-variance standardization
        Z = (X - self.mean_) / self.std_

        # 2. Non-saturating phase angle transformation
        theta = np.arctan(self.gamma * Z)

        # 3. Superposition states
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # 4. Entanglement states
        entangle_terms = []
        for i, j in self.entanglement_indices_:
            sum_theta = theta[:, i] + theta[:, j]
            entangle_terms.append(np.cos(sum_theta))
            entangle_terms.append(np.sin(sum_theta))

        blocks = [np.ones((X.shape[0], 1), dtype=float), cos_theta, sin_theta]
        if entangle_terms:
            blocks.append(np.column_stack(entangle_terms))

        return np.hstack(blocks)

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

        # Deterministic entanglement graph
        self.entanglement_indices_ = self._select_entanglement_pairs(X_arr.shape[1])

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

        # Search bounds: gamma in [0.05, 2.0], alpha_reg in [0.01, 50.0]
        param_bounds = [
            (0.05, 2.0),   # gamma
            (0.01, 50.0),  # alpha_reg
        ]

        def validation_objective(params: np.ndarray) -> float:
            gamma_cand, alpha_cand = float(params[0]), float(params[1])
            model = QIFCPRegressor(
                gamma=gamma_cand,
                alpha_reg=alpha_cand,
                n_entanglement_pairs=self.n_entanglement_pairs,
                random_state=self.random_state,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
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

        if opt_res.best_vector and len(opt_res.best_vector) >= 2:
            self.gamma = float(opt_res.best_vector[0])
            self.alpha_reg = float(opt_res.best_vector[1])

        self.tuning_history_ = opt_res.history
        logger.info(
            "QPSO tuned QIFCP hyperparameters: gamma=%.4f, alpha_reg=%.4f (best RMSE: %.4f)",
            self.gamma,
            self.alpha_reg,
            opt_res.best_score,
        )

        # Fit final model on all data using optimal hyperparameters
        return self.fit(X_arr, y_arr)
