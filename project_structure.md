# Project Structure & Architecture Ownership

**Problem ID:** SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization  
**Phase:** Production Implementation & Validated Benchmark Platform (Deliverables D1–D10 Complete)  
**Target Runtime:** Python 3.12+

---

## 1. Directory Tree

```
c:/HACKATHONS/SIH_TRY3/
│
├── src/                                    # Core production domain implementations
│   ├── compliance/                         # Statutory compliance evaluation & penalty estimators
│   │   ├── __init__.py
│   │   └── compliance_engine.py            # MaritimeComplianceEngine (IMO CII MEPC.353(78) G2, MEPC.354(78) G4, MEPC.400(83); FuelEU (EU) 2023/1805)
│   ├── ingestion/                          # Data ingestion, synthetic generation & observational adapters
│   │   ├── __init__.py
│   │   ├── dataset_loader.py               # LocalDatasetLoader, schema validator, and streaming reader
│   │   ├── mock_generator.py               # Synthetic hydrodynamic voyage dataset generator (10k records)
│   │   ├── real_data_adapter.py            # THETIS-MRV & FuelCast observational ingestion with proxy-leakage neutralization
│   │   └── validators.py                   # Physical boundaries, type verification, and schema integrity guards
│   ├── optimization/                       # Quantum-behaved & evolutionary green fleet optimizers
│   │   ├── __init__.py
│   │   ├── fleet_objective.py              # Unified multi-engine scalar objective J, strict fuel routing, dynamic spec merging
│   │   ├── fleet_optimizer.py              # FleetOptimizationRunner, equal-budget QPSO vs PSO, problem generator
│   │   ├── nsga2_pareto.py                 # ParetoFleetOptimizer, hybrid DE-NSGA-II solver with Deb et al. (2002) crowding distance
│   │   ├── qpso.py                         # Quantum-behaved Particle Swarm Optimization (mean best position & wave collapse)
│   │   └── scenario_analysis.py            # ScenarioAnalysisEngine, 6-fuel macro transition simulation (Diesel, LNG, Methanol, H2, NH3, Shore)
│   ├── physics/                            # Naval architecture & first-principles thermodynamic energy conversion
│   │   ├── __init__.py
│   │   └── fuel_physics_engine.py          # MaritimeFuelPhysicsEngine (Admiralty resistance, novel fuel LHV, ShorePower electrical MWh)
│   ├── prediction/                         # Machine learning, quantum-inspired modeling & lifecycle emissions
│   │   ├── __init__.py
│   │   ├── emission_engine.py              # MaritimeEmissionEngine (TTW direct combustion & WTW lifecycle GHG: CO2, CH4, N2O, CO2e)
│   │   ├── explainability.py               # Permutation importance and feature attribution analytics
│   │   ├── features.py                     # Hydrodynamic feature engineering pipeline, rate formulation (t/h), one-hot categoricals
│   │   ├── metrics.py                      # Regression metrics (R2, RMSE, MAE, MAPE, NRMSE_mean)
│   │   ├── model_manager.py                # ProductionModelManager, registry persistence, model promotion
│   │   ├── models.py                       # Classical regression models (LinearRegression, RandomForest, HistGradientBoosting)
│   │   ├── predictor.py                    # Verified inference engine conforming to contracts.interfaces.PredictionEngine
│   │   ├── qifcp.py                        # Quantum-Inspired Fuel Consumption Prediction with nested vessel-disjoint QPSO tuning
│   │   └── trainer.py                      # Model training orchestrator and cross-validation pipelines
│   └── scheduler/                          # Fleet timetable generation and commercial order matching
│       ├── __init__.py
│       └── fleet_scheduler.py              # FleetScheduler (greedy feasible-first vessel-cargo allocator, laycan/draft auditor)
│
├── app/                                    # Presentation & dashboard layer
│   ├── __init__.py
│   └── dashboard/                          # Streamlit analytical modules
│       ├── __init__.py
│       ├── page_compliance.py              # Page 4: IMO CII dynamic rating audit & FuelEU deficit penalty calculator
│       ├── page_optimization.py            # Page 3: Swarm scalability sweeps & interactive Pareto frontier trade-offs
│       ├── page_overview.py                # Page 1: Fleet telemetry, feature correlation heatmaps, raw voyage explorer
│       ├── page_prediction.py              # Page 2: 4-Model leaderboard, per-source normalized error, live inference sandbox
│       └── page_scenarios.py               # Page 5: Macro decarbonization multi-fuel transition simulation
│
├── contracts/                              # Canonical API contract package (frozen interfaces and schemas)
│   ├── __init__.py                         # Exposes all schemas, interfaces, constants, exceptions, version
│   ├── constants.py                        # Strict string enums (FuelType, OptimizerType, ModelType) and supported tuples
│   ├── exceptions.py                       # Granular domain exception hierarchy (MaritimeSystemError base)
│   ├── interfaces.py                       # Pure Abstract Base Classes (ABCs) defining execution signatures
│   ├── schemas.py                          # Immutable dataclasses (VoyageRecord, PredictionResult, CIIResult, FuelEUResult, etc.)
│   └── version.py                          # Canonical contract semantic version (CONTRACT_VERSION = "1.0.0")
│
├── models/                                 # Quantum circuit architectures
│   └── fuel_predictor.py                   # QuantumKernelPredictor (PennyLane default.qubit, ZZFeatureMap, KernelRidge)
│
├── scripts/                                # Reproducible automation and benchmarking scripts
│   ├── benchmark_models.py                 # Multi-model comparative training & leaderboard generator
│   ├── benchmark_optimization.py           # Multi-seed swarm optimization scalability benchmark (QPSO vs PSO)
│   ├── build_for_deploy.sh                 # Cloud build script for Render deployment
│   ├── make_mock_dataset.py                # 10,000 synthetic voyage dataset synthesizer
│   ├── run_full_pipeline.py                # End-to-end multi-stage pipeline runner (synthetic & blended real data modes)
│   └── run_scenario_comparison.py          # 6-fuel macro transition scenario benchmark
│
├── tests/                                  # Comprehensive deterministic test suites (225 passing tests)
│   ├── __init__.py
│   ├── test_benchmark_script.py            # Benchmark script execution and schema tests
│   ├── test_compliance_engine.py           # IMO CII (G2/G4/MEPC.400), FuelEU Article 23(2) penalties, metric validation
│   ├── test_contract_version.py            # Contract semantic version compatibility
│   ├── test_contracts.py                   # Dataclass serialization, immutability, and ABC invariants
│   ├── test_dataset_loader.py              # Ingestion, validation, and error-handling tests
│   ├── test_emission_contract.py           # Emission result contract conformity
│   ├── test_emission_engine.py             # TTW combustion & WTW lifecycle emission calculations
│   ├── test_explainability.py              # Feature importance and explainability checks
│   ├── test_feature_pipeline.py            # Feature engineering, scaling, and categorical encoders
│   ├── test_fleet_objective.py             # Objective J arithmetic, strict fuel routing, dynamic spec merging
│   ├── test_fleet_scheduler.py             # Greedy allocation, laycan constraints, draft capacity matching
│   ├── test_fuel_physics.py                # Admiralty drag equations, LHV conversions, ShorePower MWh
│   ├── test_import_consistency.py          # Root shim and package import integrity
│   ├── test_mock_dataset.py                # Synthetic dataset distribution checks
│   ├── test_model_manager.py               # Model registry, caching, and model persistence
│   ├── test_no_source_leakage.py           # Duration proxy-leakage adversarial classification invariance (<= 60%)
│   ├── test_nsga2.py                       # Fast non-dominated sorting, Deb 2002 crowding distance, DE-NSGA-II Pareto
│   ├── test_prediction_metrics.py          # R2, RMSE, MAE, MAPE, NRMSE metric derivations
│   ├── test_prediction_models.py           # Classical ML regression models (LR, RF, HistGBDT)
│   ├── test_prediction_pipeline.py         # End-to-end model training, validation, and inference
│   ├── test_qifcp.py                       # Quantum-inspired neural regressor, nested vessel-disjoint QPSO tuning
│   ├── test_qpso_pso.py                    # Classical PSO vs Quantum-behaved PSO convergence
│   ├── test_real_data_adapter.py           # THETIS-MRV & FuelCast observational adapters
│   ├── test_recommendation_engine.py       # Fuel and operational recommendations
│   ├── test_scalability.py                 # Multi-tier swarm scalability tests (Small, Medium, Large)
│   ├── test_scenario_analysis.py           # 6-fuel macro transition scenario analysis
│   └── test_validators.py                  # Schema input validation and error raising
│
├── data/                                   # Persistent data store
│   ├── raw/                                # Raw synthetic and observational datasets
│   └── processed/                          # Sanitized, normalized, and feature-engineered datasets
│
├── artifacts/                              # Generated model binaries and metric manifests
│   ├── models/                             # Serialized model weights (*.pkl)
│   └── metrics/                            # Baseline and model registry manifests (*.json)
│
├── outputs/                                # Runtime reports, figures, and execution logs
│   ├── figures/                            # Exported charts and plots
│   ├── logs/                               # Rotating execution logs
│   └── reports/                            # Benchmark summaries and pipeline run manifests
│
├── app.py                                  # Main Streamlit executive dashboard entry point
├── config.py                               # Centralized configuration dataclasses (SystemConfig)
├── constants.py                            # Root backward-compatible alias to contracts.constants
├── exceptions.py                           # Root backward-compatible alias to contracts.exceptions
├── interfaces.py                           # Root backward-compatible alias to contracts.interfaces
├── schemas.py                              # Root backward-compatible alias to contracts.schemas
├── logging_config.py                       # Centralized console and rotating file logger
├── project_structure.md                    # Structural specification and ownership documentation
├── README.md                               # System design, regulatory methodologies, and user guide
├── render.yaml                             # Cloud deployment blueprint specification
├── requirements.txt                        # Pinned dependencies for Python 3.12+
└── run_demo.py                             # Verification script confirming system health and imports
```

---

## 2. Subsystem Ownership & Layer Responsibilities

### `src/physics/`
- **Ownership:** Naval Architecture & Hydrodynamics Engineer
- **Contract:** Implements [`contracts.interfaces.FuelPhysicsEngine`](contracts/interfaces.py)
- **Core Focus:** Admiralty hydrodynamic propulsion formula ($P \propto \Delta^{2/3} V^3 / C_{\text{adm}}$), lower heating value (LHV) fuel mass conversions, and cold-ironing shore power electrical energy (MWh) tracking. First-principles engine for unobserved zero-emission novel fuels (**Hydrogen**, **Ammonia**, **ShorePower**).

### `src/prediction/`
- **Ownership:** Machine Learning & Quantum Algorithm Engineer
- **Contract:** Implements [`contracts.interfaces.PredictionEngine`](contracts/interfaces.py) and [`contracts.interfaces.EmissionEngine`](contracts/interfaces.py)
- **Core Focus:**
  - Hydrodynamic rate formulation ($\text{metric tons/hour}$) to eliminate voyage duration bias.
  - Multi-architecture regression: Linear Regression, Random Forest, `HistGradientBoostingRegressor` (standardized GBDT), and Quantum-Inspired Neural Regressor (`QIFCP`).
  - Strict nested vessel-disjoint cross-validation protocol (`GroupShuffleSplit` across vessel IDs during both outer test and inner QPSO hyperparameter tuning).
  - Tank-to-Wake (TTW) combustion emissions and Well-to-Wake (WTW) lifecycle emissions across $\text{CO}_2$, $\text{CH}_4$, $\text{N}_2\text{O}$, and $\text{CO}_2\text{e}$ factors.

### `src/compliance/`
- **Ownership:** Maritime Regulatory & Environmental Policy Specialist
- **Contract:** Implements [`contracts.interfaces.ComplianceEngine`](contracts/interfaces.py)
- **Core Focus:**
  - IMO Carbon Intensity Indicator (CII) operational rating and attained metric estimator under Resolution MEPC.353(78) (G2 reference lines with exact DWT/GT effective capacity caps), Resolution MEPC.354(78) (G4 ship-type-specific rating boundary vectors $d_1$--$d_4$), and Resolution MEPC.400(83) ($Z$-factor reduction trajectory through 2030).
  - Strict statutory capacity metric validation: enforces Gross Tonnage (GT) for Ro-Ro/passenger vessels and Deadweight Tonnage (DWT) for cargo vessels, raising `DataValidationError` on unit mismatch.
  - EU FuelEU Maritime GHG-intensity compliance and deficit penalty estimator under Regulation (EU) 2023/1805 Articles 4, 23, and Annex IV, including Article 23(2) consecutive-period multiplier scaling ($1 + \frac{n-1}{10}$).

### `src/scheduler/`
- **Ownership:** Maritime Operations & Fleet Logistics Engineer
- **Contract:** Implements [`contracts.interfaces.SchedulerEngine`](contracts/interfaces.py)
- **Core Focus:** Greedy feasible-first vessel-to-cargo scheduling enforcing laycan transit windows, port draft depth thresholds, and cargo payload deadweight capacity matching.

### `src/optimization/`
- **Ownership:** Operations Research & Metaheuristics Specialist
- **Contract:** Implements [`contracts.interfaces.OptimizationEngine`](contracts/interfaces.py) and [`contracts.interfaces.ScenarioEngine`](contracts/interfaces.py)
- **Core Focus:**
  - Single-objective scalar fleet optimization ($J = w_1 \cdot \text{Cost} + w_2 \cdot \text{CO}_2\text{e} + w_3 \cdot \text{Delay}$) via equal-budget Classical PSO and Quantum-behaved PSO (`QPSO`).
  - Multi-objective Pareto optimization via hybrid DE-NSGA-II (`ParetoFleetOptimizer`), implementing canonical Deb et al. (2002) fast non-dominated sorting and crowding-distance environmental selection.
  - Unified system-wide dual-route fuel architecture: strictly routes conventional fuels (`Diesel`, `LNG`, `Methanol`) to ML models and novel alternative pathways (`Hydrogen`, `Ammonia`, `ShorePower`) to naval physics.
  - Dynamic hierarchical parameterization: extracts `vessel_type`, `vessel_dwt`, `cargo_tons`, `weather_factor`, and `sea_state` directly from `voyage_specs` and `vessel_specs`.

### `src/ingestion/`
- **Ownership:** Data Engineering & Pipeline Specialist
- **Contract:** Implements [`contracts.interfaces.DatasetLoader`](contracts/interfaces.py)
- **Core Focus:** Synthetic hydrodynamic dataset synthesis (`10,000` records), heterogeneous observational ingestion (EU THETIS-MRV and FuelCast sensor streams), duration proxy-leakage neutralization, and physical sanity boundaries.

### `app/dashboard/` & `app.py`
- **Ownership:** Full-Stack & Maritime Visualization Engineer
- **Core Focus:** Interactive Streamlit web interface presenting live fleet telemetry, model benchmark leaderboards with per-source normalized metrics ($R^2$, RMSE, MAE, MAPE, NRMSE), interactive voyage fuel prediction sandbox, QPSO vs PSO scalability sweeps, Pareto tradeoff frontiers, statutory IMO CII/FuelEU audits, and 6-fuel macro transition simulators.

---

## 3. Architecture Invariants & Standards

1. **Strict Contract Boundaries:** All inter-layer data transfer occurs strictly through frozen dataclasses defined in [`contracts/schemas.py`](contracts/schemas.py). Subsystems never pass untyped dictionaries or mutable global objects across module boundaries.
2. **Dual-Route Fuel Boundary:** Statistical regression models are never evaluated on novel zero-emission fuels lacking historical empirical distributions; first-principles naval architecture (`MaritimeFuelPhysicsEngine`) governs Hydrogen, Ammonia, and ShorePower across all engines.
3. **Zero-Leakage Rate Formulation:** Fuel consumption is trained and evaluated in rate mode ($\text{t/h}$) on standardized voyage legs, neutralizing voyage duration proxy leakage (adversarial classifier accuracy $\le 60\%$).
4. **Decoupled Compliance Architecture:** Statutory compliance evaluations produce discrete, domain-specific assessments ([`CIIResult`](contracts/schemas.py) and [`FuelEUResult`](contracts/schemas.py)) unified under [`ComplianceAssessment`](contracts/schemas.py), ensuring clear separation between statutory operational metrics.
