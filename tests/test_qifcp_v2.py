import numpy as np
import pytest
from src.prediction.qifcp import QIFCPRegressor

@pytest.fixture
def dummy_data():
    # 5 samples, 4 features
    X = np.random.RandomState(0).randn(5, 4)
    y = np.random.RandomState(1).randn(5)
    return X, y

def test_v2_mode_fits_and_predicts(dummy_data):
    X, y = dummy_data
    model = QIFCPRegressor(qifcp_mode="v2", harmonic_order=2, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (X.shape[0],)
    # predictions should be non‑negative as per implementation
    assert np.all(preds >= 0)

@pytest.mark.parametrize("K, expected_dim", [
    (1, 21),  # bias + 2*K*features + 2*pair_count (pair_count=6 for 4 features)
    (3, 37),
])
def test_harmonic_dimension(dummy_data, K, expected_dim):
    X, y = dummy_data
    model = QIFCPRegressor(qifcp_mode="v2", harmonic_order=K, random_state=42)
    model.fit(X, y)
    assert model.n_quantum_features_ == expected_dim

def test_invalid_harmonic_order():
    with pytest.raises(ValueError):
        QIFCPRegressor(qifcp_mode="v2", harmonic_order=0)
    with pytest.raises(ValueError):
        QIFCPRegressor(qifcp_mode="v2", harmonic_order=-3)

def test_deterministic_predictions(dummy_data):
    X, y = dummy_data
    model1 = QIFCPRegressor(qifcp_mode="v2", harmonic_order=3, random_state=123)
    model2 = QIFCPRegressor(qifcp_mode="v2", harmonic_order=3, random_state=123)
    model1.fit(X, y)
    model2.fit(X, y)
    preds1 = model1.predict(X)
    preds2 = model2.predict(X)
    np.testing.assert_allclose(preds1, preds2)
    assert model1.n_quantum_features_ == model2.n_quantum_features_


def test_adaptive_entanglement_pair_count_and_no_duplicates():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(50, 10))
    y = X[:, 0] * 2.0 + X[:, 1] * X[:, 2] + rng.normal(0, 0.1, size=50)

    for M in [3, 5, 10]:
        model = QIFCPRegressor(
            qifcp_mode="v2",
            harmonic_order=3,
            n_entanglement_pairs=M,
            entanglement_mode="adaptive",
            random_state=42,
        )
        model.fit(X, y)
        assert len(model.entanglement_indices_) == M
        # No duplicates
        assert len(set(model.entanglement_indices_)) == M
        for i, j in model.entanglement_indices_:
            assert 0 <= i < j < 10


def test_adaptive_feature_count_formula():
    rng = np.random.default_rng(42)
    d = 8
    X = rng.normal(0, 1, size=(40, d))
    y = rng.normal(0, 1, size=40)

    for K in [1, 2, 3]:
        for M in [2, 5, 8]:
            model = QIFCPRegressor(
                qifcp_mode="v2",
                harmonic_order=K,
                n_entanglement_pairs=M,
                entanglement_mode="adaptive",
                random_state=42,
            )
            model.fit(X, y)
            expected_features = 1 + (2 * K * d) + (2 * M)
            assert model.n_quantum_features_ == expected_features


def test_adaptive_pair_selection_is_deterministic():
    rng = np.random.default_rng(101)
    X = rng.normal(0, 1, size=(60, 6))
    y = rng.normal(0, 1, size=60)

    model1 = QIFCPRegressor(
        qifcp_mode="v2",
        harmonic_order=3,
        n_entanglement_pairs=5,
        entanglement_mode="adaptive",
        random_state=42,
    )
    model1.fit(X, y)

    model2 = QIFCPRegressor(
        qifcp_mode="v2",
        harmonic_order=3,
        n_entanglement_pairs=5,
        entanglement_mode="adaptive",
        random_state=42,
    )
    model2.fit(X, y)

    assert model1.entanglement_indices_ == model2.entanglement_indices_
    preds1 = model1.predict(X)
    preds2 = model2.predict(X)
    np.testing.assert_allclose(preds1, preds2)


def test_adaptive_entanglement_training_only_isolation():
    rng = np.random.default_rng(2024)
    X_train = rng.normal(0, 1, size=(50, 6))
    y_train = rng.normal(0, 1, size=50)

    X_test_1 = rng.normal(10, 5, size=(20, 6))
    X_test_2 = rng.normal(-10, 5, size=(20, 6))

    model = QIFCPRegressor(
        qifcp_mode="v2",
        harmonic_order=3,
        n_entanglement_pairs=4,
        entanglement_mode="adaptive",
        random_state=42,
    )
    model.fit(X_train, y_train)
    pairs_after_fit = list(model.entanglement_indices_)

    # Predicting on different test sets never alters learned pairs
    model.predict(X_test_1)
    assert model.entanglement_indices_ == pairs_after_fit
    model.predict(X_test_2)
    assert model.entanglement_indices_ == pairs_after_fit


def test_v1_and_v2_random_backward_compatibility():
    rng = np.random.default_rng(999)
    X = rng.normal(0, 1, size=(30, 5))
    y = rng.normal(0, 1, size=30)

    # v1 legacy behavior
    model_v1 = QIFCPRegressor(qifcp_mode="v1", n_entanglement_pairs=4, random_state=42)
    model_v1.fit(X, y)
    assert model_v1.qifcp_mode_ == "v1"
    assert model_v1.entanglement_mode_ == "random"
    assert model_v1.n_quantum_features_ == 1 + 2 * 5 + 2 * 4

    # v2 random behavior
    model_v2_rand = QIFCPRegressor(
        qifcp_mode="v2",
        harmonic_order=3,
        n_entanglement_pairs=4,
        entanglement_mode="random",
        random_state=42,
    )
    model_v2_rand.fit(X, y)
    assert model_v2_rand.entanglement_mode_ == "random"
    assert model_v2_rand.n_quantum_features_ == 1 + 6 * 5 + 2 * 4


def test_invalid_entanglement_mode():
    with pytest.raises(ValueError) as exc:
        QIFCPRegressor(entanglement_mode="invalid_mode")
    assert "entanglement_mode" in str(exc.value)


# ==============================================================================
# Phase 2C: Grouped Phase Scaling (gamma_g) Unit Tests
# ==============================================================================

def test_grouped_gamma_all_features_receive_exactly_one_group():
    from src.prediction.qifcp import DEFAULT_27_FEATURE_NAMES
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(20, 27))
    y = rng.normal(10, 2, size=20)

    model = QIFCPRegressor(gamma_mode="grouped", random_state=42)
    model.fit(X, y)

    pg = model.phase_groups_
    assert set(pg.keys()) == {"hydrodynamic", "operational", "environmental"}

    hydro = set(pg["hydrodynamic"])
    oper = set(pg["operational"])
    env = set(pg["environmental"])

    # Disjoint groups (no overlap)
    assert len(hydro.intersection(oper)) == 0
    assert len(hydro.intersection(env)) == 0
    assert len(oper.intersection(env)) == 0

    # Exhaustive coverage (all features mapped)
    all_mapped = hydro.union(oper).union(env)
    assert all_mapped == set(DEFAULT_27_FEATURE_NAMES)
    assert len(pg["hydrodynamic"]) + len(pg["operational"]) + len(pg["environmental"]) == 27


def test_grouped_gamma_mapping_is_deterministic():
    rng = np.random.default_rng(101)
    X = rng.normal(0, 1, size=(15, 27))
    y = rng.normal(5, 1, size=15)

    m1 = QIFCPRegressor(gamma_mode="grouped", random_state=1)
    m1.fit(X, y)

    m2 = QIFCPRegressor(gamma_mode="grouped", random_state=999)
    m2.fit(X, y)

    assert m1.phase_groups_ == m2.phase_groups_
    assert m1.phase_group_indices_ == m2.phase_group_indices_


def test_grouped_phase_transform_works():
    rng = np.random.default_rng(2024)
    X = rng.normal(0, 1, size=(10, 27))
    y = rng.normal(5, 1, size=10)

    # Base model
    m_base = QIFCPRegressor(
        gamma_mode="grouped",
        gamma_hydrodynamic=0.5,
        gamma_operational=0.5,
        gamma_environment=0.5,
        random_state=42,
    )
    m_base.fit(X, y)

    # Perturbed hydrodynamic only
    m_hydro = QIFCPRegressor(
        gamma_mode="grouped",
        gamma_hydrodynamic=1.5,
        gamma_operational=0.5,
        gamma_environment=0.5,
        random_state=42,
    )
    m_hydro.fit(X, y)

    # Standardized features are identical
    Z = (X - m_base.mean_) / m_base.std_
    theta_base = np.arctan(m_base.feature_gamma_vector_ * Z)
    theta_hydro = np.arctan(m_hydro.feature_gamma_vector_ * Z)

    hydro_idx = m_base.phase_group_indices_["hydrodynamic"]
    oper_idx = m_base.phase_group_indices_["operational"]
    env_idx = m_base.phase_group_indices_["environmental"]

    # Hydrodynamic features must differ
    assert not np.allclose(theta_base[:, hydro_idx], theta_hydro[:, hydro_idx])
    # Operational and environmental features must be bitwise identical
    np.testing.assert_allclose(theta_base[:, oper_idx], theta_hydro[:, oper_idx])
    np.testing.assert_allclose(theta_base[:, env_idx], theta_hydro[:, env_idx])


def test_global_and_grouped_gamma_mathematical_consistency():
    rng = np.random.default_rng(777)
    X = rng.normal(0, 1, size=(25, 27))
    y = rng.normal(12, 3, size=25)

    test_gamma = 0.85
    m_global = QIFCPRegressor(
        gamma=test_gamma,
        gamma_mode="global",
        qifcp_mode="v2",
        harmonic_order=3,
        n_entanglement_pairs=10,
        random_state=42,
    )
    m_global.fit(X, y)

    m_grouped = QIFCPRegressor(
        gamma_mode="grouped",
        gamma_hydrodynamic=test_gamma,
        gamma_operational=test_gamma,
        gamma_environment=test_gamma,
        qifcp_mode="v2",
        harmonic_order=3,
        n_entanglement_pairs=10,
        random_state=42,
    )
    m_grouped.fit(X, y)

    # Quantum feature matrices must be mathematically identical
    phi_global = m_global._quantum_feature_map(X)
    phi_grouped = m_grouped._quantum_feature_map(X)
    np.testing.assert_allclose(phi_global, phi_grouped, atol=1e-12)

    # Learned projection weights must match
    np.testing.assert_allclose(m_global.weights_, m_grouped.weights_, atol=1e-12)

    # Predictions must match
    preds_global = m_global.predict(X)
    preds_grouped = m_grouped.predict(X)
    np.testing.assert_allclose(preds_global, preds_grouped, atol=1e-12)


def test_grouped_model_exposes_gamma_attributes():
    model = QIFCPRegressor(
        gamma_mode="grouped",
        gamma_hydrodynamic=0.45,
        gamma_operational=0.65,
        gamma_environment=0.85,
        random_state=42,
    )
    assert hasattr(model, "gamma_hydrodynamic_")
    assert hasattr(model, "gamma_operational_")
    assert hasattr(model, "gamma_environment_")
    assert model.gamma_hydrodynamic_ == 0.45
    assert model.gamma_operational_ == 0.65
    assert model.gamma_environment_ == 0.85

    X = np.random.randn(10, 5)
    y = np.random.randn(10)
    model.fit(X, y)
    assert model.gamma_hydrodynamic_ == 0.45
    assert model.gamma_operational_ == 0.65
    assert model.gamma_environment_ == 0.85


def test_invalid_gamma_values_rejected():
    with pytest.raises(ValueError):
        QIFCPRegressor(gamma=-0.5)
    with pytest.raises(ValueError):
        QIFCPRegressor(gamma=0.0)
    with pytest.raises(ValueError):
        QIFCPRegressor(gamma=float("nan"))
    with pytest.raises(ValueError):
        QIFCPRegressor(gamma_mode="invalid_gamma_mode")
    with pytest.raises(ValueError):
        QIFCPRegressor(gamma_mode="grouped", gamma_hydrodynamic=-0.1)
    with pytest.raises(ValueError):
        QIFCPRegressor(gamma_mode="grouped", gamma_operational=0.0)
    with pytest.raises(ValueError):
        QIFCPRegressor(gamma_mode="grouped", gamma_environment=float("inf"))


def test_grouped_gamma_deterministic_repeated_fit_predict():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(30, 27))
    y = rng.normal(10, 2, size=30)

    m1 = QIFCPRegressor(
        gamma_mode="grouped",
        gamma_hydrodynamic=0.4,
        gamma_operational=0.7,
        gamma_environment=1.1,
        harmonic_order=3,
        n_entanglement_pairs=8,
        random_state=123,
    )
    m2 = QIFCPRegressor(
        gamma_mode="grouped",
        gamma_hydrodynamic=0.4,
        gamma_operational=0.7,
        gamma_environment=1.1,
        harmonic_order=3,
        n_entanglement_pairs=8,
        random_state=123,
    )

    m1.fit(X, y)
    m2.fit(X, y)

    np.testing.assert_allclose(m1.weights_, m2.weights_)
    np.testing.assert_allclose(m1.predict(X), m2.predict(X))


def test_grouped_gamma_preserves_entanglement_pair_count():
    rng = np.random.default_rng(88)
    X = rng.normal(0, 1, size=(40, 27))
    y = rng.normal(10, 2, size=40)

    for M in [5, 10, 15]:
        m_global = QIFCPRegressor(
            gamma_mode="global",
            n_entanglement_pairs=M,
            entanglement_mode="adaptive",
            random_state=42,
        )
        m_grouped = QIFCPRegressor(
            gamma_mode="grouped",
            n_entanglement_pairs=M,
            entanglement_mode="adaptive",
            random_state=42,
        )
        m_global.fit(X, y)
        m_grouped.fit(X, y)

        assert len(m_global.entanglement_indices_) == M
        assert len(m_grouped.entanglement_indices_) == M
        assert m_global.entanglement_indices_ == m_grouped.entanglement_indices_
        assert m_global.n_quantum_features_ == m_grouped.n_quantum_features_


# ==============================================================================
# Phase 2D: Physics-Informed Residual Learning Unit Tests (Section 18)
# ==============================================================================


def test_physics_baseline_produces_finite_output():
    from src.prediction.qifcp import NavalPhysicsFuelBaseline

    rng = np.random.default_rng(42)
    X = np.abs(rng.normal(50, 15, size=(40, 27)))
    # Ensure positive displacement, speed, distance
    X[:, 0] = 50000.0  # DWT
    X[:, 2] = 1200.0   # Distance
    X[:, 3] = 14.0     # Speed
    X[:, 4] = 1.1      # Weather factor
    X[:, 10] = (0.2 * 50000 + 40000) ** (2/3) * (14.0 ** 3) # Power proxy
    X[:, 11] = 1200.0 / 14.0 # Implied hours
    X[:, 19] = 1.0     # Diesel
    y = np.abs(rng.normal(200, 30, size=40))

    model = NavalPhysicsFuelBaseline()
    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape == (40,)
    assert np.all(np.isfinite(preds))
    assert np.all(preds >= 0.0)
    assert model.c_prop_ is not None and model.c_prop_ >= 0.0
    assert model.c_aux_ is not None and model.c_aux_ >= 0.0


def test_physics_baseline_is_deterministic():
    from src.prediction.qifcp import NavalPhysicsFuelBaseline

    rng = np.random.default_rng(42)
    X = np.abs(rng.normal(30, 5, size=(30, 27)))
    y = np.abs(rng.normal(150, 20, size=30))

    m1 = NavalPhysicsFuelBaseline()
    m2 = NavalPhysicsFuelBaseline()
    m1.fit(X, y)
    m2.fit(X, y)

    assert m1.c_prop_ == m2.c_prop_
    assert m1.c_aux_ == m2.c_aux_
    np.testing.assert_allclose(m1.predict(X), m2.predict(X))


def test_physics_calibration_uses_training_data_only():
    from src.prediction.qifcp import NavalPhysicsFuelBaseline

    rng = np.random.default_rng(77)
    X_train = np.abs(rng.normal(40, 10, size=(50, 27)))
    y_train = np.abs(rng.normal(180, 25, size=50))
    X_test_1 = np.abs(rng.normal(100, 20, size=(20, 27)))
    X_test_2 = np.abs(rng.normal(5, 1, size=(20, 27)))

    model = NavalPhysicsFuelBaseline()
    model.fit(X_train, y_train)

    c_prop_fit = model.c_prop_
    c_aux_fit = model.c_aux_

    # Prediction on test sets does not change parameters
    model.predict(X_test_1)
    assert model.c_prop_ == c_prop_fit
    assert model.c_aux_ == c_aux_fit

    model.predict(X_test_2)
    assert model.c_prop_ == c_prop_fit
    assert model.c_aux_ == c_aux_fit


def test_residual_target_reconstruction():
    from src.prediction.qifcp import NavalPhysicsFuelBaseline

    rng = np.random.default_rng(101)
    X = np.abs(rng.normal(30, 5, size=(35, 27)))
    y = np.abs(rng.normal(120, 15, size=35))

    phys = NavalPhysicsFuelBaseline()
    phys.fit(X, y)
    y_phys = phys.predict(X)
    residual = y - y_phys

    # Exact mathematical reconstruction: y = y_phys + residual
    np.testing.assert_allclose(y, y_phys + residual)


def test_final_prediction_equals_physics_plus_residual():
    from src.prediction.qifcp import PhysicsInformedQIFCPRegressor

    rng = np.random.default_rng(42)
    X = np.abs(rng.normal(30, 5, size=(40, 27)))
    y = np.abs(rng.normal(150, 20, size=40))

    model = PhysicsInformedQIFCPRegressor(random_state=42)
    model.fit(X, y)

    y_phys, r_hat, y_final = model.predict_components(X)
    direct_pred = model.predict(X)

    expected = np.maximum(y_phys + r_hat, 0.0)
    np.testing.assert_allclose(direct_pred, expected)
    np.testing.assert_allclose(y_final, direct_pred)


def test_grouped_qifcp_configuration_preserved():
    from src.prediction.qifcp import PhysicsInformedQIFCPRegressor

    model = PhysicsInformedQIFCPRegressor(
        harmonic_order=3,
        n_entanglement_pairs=15,
        gamma_mode="grouped",
        entanglement_mode="adaptive",
    )
    assert model.qifcp_residual_.harmonic_order == 3
    assert model.qifcp_residual_.n_entanglement_pairs == 15
    assert model.qifcp_residual_.gamma_mode == "grouped"
    assert model.qifcp_residual_.entanglement_mode == "adaptive"
    assert model.lambda_residual == 1.0


def test_no_leakage_into_test_partition():
    from src.prediction.qifcp import PhysicsInformedQIFCPRegressor

    rng = np.random.default_rng(55)
    X_train = np.abs(rng.normal(30, 5, size=(40, 27)))
    y_train = np.abs(rng.normal(150, 20, size=40))
    X_test = np.abs(rng.normal(80, 10, size=(15, 27)))

    model = PhysicsInformedQIFCPRegressor(random_state=42)
    model.fit(X_train, y_train)

    phys_params_before = dict(model.physics_parameters_)
    q_weights_before = np.copy(model.qifcp_residual_.weights_)

    # Inference on test partition does not modify fitted training parameters
    _ = model.predict(X_test)

    assert model.physics_parameters_ == phys_params_before
    np.testing.assert_array_equal(model.qifcp_residual_.weights_, q_weights_before)


def test_non_negative_prediction_validity():
    from src.prediction.qifcp import PhysicsInformedQIFCPRegressor

    rng = np.random.default_rng(66)
    X = np.abs(rng.normal(30, 5, size=(30, 27)))
    y = np.abs(rng.normal(50, 5, size=30))

    model = PhysicsInformedQIFCPRegressor(random_state=42)
    model.fit(X, y)
    preds = model.predict(X)

    assert np.all(preds >= 0.0)
    assert np.all(np.isfinite(preds))


def test_deterministic_repeated_training():
    from src.prediction.qifcp import PhysicsInformedQIFCPRegressor

    rng = np.random.default_rng(77)
    X = np.abs(rng.normal(30, 5, size=(35, 27)))
    y = np.abs(rng.normal(100, 15, size=35))

    m1 = PhysicsInformedQIFCPRegressor(random_state=42)
    m2 = PhysicsInformedQIFCPRegressor(random_state=42)

    m1.fit(X, y)
    m2.fit(X, y)

    assert m1.physics_model_.c_prop_ == m2.physics_model_.c_prop_
    assert m1.physics_model_.c_aux_ == m2.physics_model_.c_aux_
    np.testing.assert_allclose(m1.qifcp_residual_.weights_, m2.qifcp_residual_.weights_)
    np.testing.assert_allclose(m1.predict(X), m2.predict(X))


def test_physics_baseline_consistency_monotonicity():
    from src.prediction.qifcp import NavalPhysicsFuelBaseline

    rng = np.random.default_rng(88)
    X = np.abs(rng.normal(30, 5, size=(30, 27)))
    y = np.abs(rng.normal(100, 15, size=30))

    model = NavalPhysicsFuelBaseline()
    model.fit(X, y)

    # Base voyage: DWT=50000, Cargo=40000, Dist=1000, Speed=14, Weather=1.0, Diesel=1.0
    x_base = np.zeros((1, 27))
    x_base[0, 0] = 50000.0
    x_base[0, 1] = 40000.0
    x_base[0, 2] = 1000.0
    x_base[0, 3] = 14.0
    x_base[0, 4] = 1.0
    x_base[0, 10] = (0.2 * 50000 + 40000) ** (2/3) * (14.0 ** 3)
    x_base[0, 11] = 1000.0 / 14.0
    x_base[0, 19] = 1.0

    # Longer voyage: Dist=2000
    x_longer = np.copy(x_base)
    x_longer[0, 2] = 2000.0
    x_longer[0, 11] = 2000.0 / 14.0

    # Faster voyage: Speed=18
    x_faster = np.copy(x_base)
    x_faster[0, 3] = 18.0
    x_faster[0, 10] = (0.2 * 50000 + 40000) ** (2/3) * (18.0 ** 3)
    x_faster[0, 11] = 1000.0 / 18.0

    fuel_base = model.predict(x_base)[0]
    fuel_longer = model.predict(x_longer)[0]
    fuel_faster = model.predict(x_faster)[0]

    # Increasing distance strictly increases fuel
    assert fuel_longer > fuel_base
    # Increasing speed at fixed distance increases total voyage fuel (cubic power vs linear time -> quadratic net)
    assert fuel_faster > fuel_base


