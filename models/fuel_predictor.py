"""Quantum-inspired and classical hydrodynamic fuel consumption prediction module.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Predicts vessel daily fuel consumption (MT/day) from hydrodynamic telemetry
using PennyLane quantum kernel Hilbert-space embeddings and classical gradient boosting baselines.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler


FEATURE_COLUMNS: List[str] = ["speed_kn", "load_factor", "wave_ht_m", "wind_bft"]
TARGET_COLUMN: str = "fuel_consumption_mt_day"
Tuple_Features = Tuple[np.ndarray, Optional[np.ndarray]]


class QuantumKernelPredictor:
    """Quantum Kernel Ridge Regression predictor for daily maritime fuel consumption.

    Quantum principle used: interference between feature-encoded states produces a similarity metric unavailable to classical RBF kernels.
    """

    def __init__(self, n_wires: int = 4, n_reps: int = 2, alpha: float = 0.1) -> None:
        """Initialize PennyLane quantum simulation device and Kernel Ridge estimator.

        Args:
            n_wires: Number of qubits (corresponding to the 4 normalized hydrodynamic features).
            n_reps: Number of ZZFeatureMap entangling circuit repetitions.
            alpha: L2 regularization strength for KernelRidge.
        """
        self.n_wires = n_wires
        self.n_reps = n_reps
        self.alpha = alpha
        self.scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        self.model = KernelRidge(alpha=self.alpha, kernel="precomputed")

        self.dev = qml.device("default.qubit", wires=self.n_wires)
        self._state_qnode = qml.QNode(self._circuit, self.dev)
        self.X_train_states_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None

    def _circuit(self, x: np.ndarray) -> Sequence[complex]:
        """Constructs an AngleEmbedding and 2-repetition ZZFeatureMap state-preparation circuit.

        Args:
            x: 1D array of 4 normalized hydrodynamic features [speed_norm, load_norm, wave_norm, wind_norm].

        Returns:
            Simulated state vector in the 2^n Hilbert space.
        """
        # Quantum principle used: interference between feature-encoded states produces a similarity metric unavailable to classical RBF kernels.
        qml.AngleEmbedding(features=x, wires=range(self.n_wires), rotation="Y")

        for _ in range(self.n_reps):
            for i in range(self.n_wires):
                qml.RZ(x[i], wires=i)
            for i in range(self.n_wires - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(x[i] * x[i + 1], wires=i + 1)
                qml.CNOT(wires=[i, i + 1])

        return qml.state()

    def _extract_features(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple_Features:
        if isinstance(X, pd.DataFrame):
            classes = X["class"].to_numpy() if "class" in X.columns else None
            features = X[FEATURE_COLUMNS].to_numpy(dtype=float)
            return features, classes
        X_arr = np.asarray(X, dtype=float)
        return X_arr[:, : len(FEATURE_COLUMNS)], None

    def _compute_states(self, X_norm: np.ndarray) -> np.ndarray:
        """Vectorized computation of 2^4-dimensional quantum state vectors."""
        return np.array([self._state_qnode(x) for x in X_norm])

    def kernel_matrix(self, X1_states: np.ndarray, X2_states: np.ndarray) -> np.ndarray:
        """Calculates transition probability inner products between quantum states.

        K(x1, x2) = |<psi(x1) | psi(x2)>|^2
        """
        inner_product = X1_states @ X2_states.conj().T
        return np.abs(inner_product) ** 2

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> "QuantumKernelPredictor":
        """Fits the quantum kernel ridge regression model."""
        X_mat, _ = self._extract_features(X)
        self.y_train_ = np.asarray(y, dtype=float).ravel()

        X_norm = self.scaler.fit_transform(X_mat)
        self.X_train_states_ = self._compute_states(X_norm)

        # Build Gram matrix
        K_train = self.kernel_matrix(self.X_train_states_, self.X_train_states_)
        self.model.fit(K_train, self.y_train_)
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Generates predictions using precomputed quantum kernel evaluation against training states."""
        if self.X_train_states_ is None:
            raise ValueError("Model is not fitted. Call fit() before predicting.")

        X_mat, _ = self._extract_features(X)
        X_norm = self.scaler.transform(X_mat)
        X_test_states = self._compute_states(X_norm)

        K_test = self.kernel_matrix(X_test_states, self.X_train_states_)
        return self.model.predict(K_test)

    def evaluate(
        self,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        vessel_classes: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Evaluates prediction metrics including MAE, RMSE, global R2, and per-class R2.

        Returns:
            Dict containing keys 'MAE', 'RMSE', 'R2', and 'per_class_R2'.
        """
        _, inferred_classes = self._extract_features(X_test)
        classes = vessel_classes if vessel_classes is not None else inferred_classes

        y_true = np.asarray(y_test, dtype=float).ravel()
        y_pred = self.predict(X_test)

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))

        metrics: Dict[str, Any] = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        }

        if classes is not None:
            classes_arr = np.asarray(classes)
            per_class_r2: Dict[str, float] = {}
            for c in np.unique(classes_arr):
                mask = classes_arr == c
                if np.sum(mask) > 1:
                    per_class_r2[str(c)] = float(r2_score(y_true[mask], y_pred[mask]))
            metrics["per_class_R2"] = per_class_r2

        return metrics


class ClassicalBaseline:
    """Classical benchmark predictor using GradientBoostingRegressor on the same 4 features."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=self.random_state,
        )

    def _extract_features(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple_Features:
        if isinstance(X, pd.DataFrame):
            classes = X["class"].to_numpy() if "class" in X.columns else None
            features = X[FEATURE_COLUMNS].to_numpy(dtype=float)
            return features, classes
        X_arr = np.asarray(X, dtype=float)
        return X_arr[:, : len(FEATURE_COLUMNS)], None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> "ClassicalBaseline":
        """Fits the Gradient Boosting Regressor."""
        X_mat, _ = self._extract_features(X)
        y_arr = np.asarray(y, dtype=float).ravel()
        self.model.fit(X_mat, y_arr)
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predicts daily fuel consumption using classical gradient boosted trees."""
        X_mat, _ = self._extract_features(X)
        return self.model.predict(X_mat)

    def evaluate(
        self,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        vessel_classes: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Evaluates prediction metrics including MAE, RMSE, global R2, and per-class R2.

        Returns:
            Dict containing keys 'MAE', 'RMSE', 'R2', and 'per_class_R2'.
        """
        _, inferred_classes = self._extract_features(X_test)
        classes = vessel_classes if vessel_classes is not None else inferred_classes

        y_true = np.asarray(y_test, dtype=float).ravel()
        y_pred = self.predict(X_test)

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))

        metrics: Dict[str, Any] = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        }

        if classes is not None:
            classes_arr = np.asarray(classes)
            per_class_r2: Dict[str, float] = {}
            for c in np.unique(classes_arr):
                mask = classes_arr == c
                if np.sum(mask) > 1:
                    per_class_r2[str(c)] = float(r2_score(y_true[mask], y_pred[mask]))
            metrics["per_class_R2"] = per_class_r2

        return metrics
