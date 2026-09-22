# Project Structure & Architecture Ownership

**Problem ID:** SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization  
**Phase:** Phase 0 Architecture Contract

---

## 1. Directory Tree

```
c:/HACKATHONS/SIH_TRY3/
│
├── app/                                # Core domain package
│   ├── __init__.py                     # Package marker for application module
│   ├── prediction/                     # Fuel consumption prediction engines
│   │   └── __init__.py
│   ├── optimization/                   # Fleet optimization algorithms (PSO, QPSO, NSGA-II)
│   │   └── __init__.py
│   ├── scheduler/                      # Vessel assignment and constraint engines
│   │   └── __init__.py
│   ├── emissions/                      # IMO and FuelEU GHG lifecycle calculators
│   │   └── __init__.py
│   ├── compliance/                     # Regulatory compliance (IMO CII & FuelEU Maritime)
│   │   └── __init__.py
│   ├── physics/                        # Naval architecture and hydrodynamic resistance
│   │   └── __init__.py
│   └── dashboard/                      # Web UI, analytics, and scenario explorer
│       └── __init__.py
│
├── data/                               # Persistent data store
│   ├── raw/                            # Ingested immutable voyage telemetry files
│   │   └── .gitkeep
│   └── processed/                      # Cleaned, normalized, and feature-engineered datasets
│       └── .gitkeep
│
├── tests/                              # Unit, integration, and contract test suites
│   ├── __init__.py
│   ├── test_contract_version.py        # Semantic contract version and schema compatibility tests
│   └── test_contracts.py               # Phase 0 contract, serialization, and ABC invariants
│
├── outputs/                            # Runtime execution artifacts
│   ├── figures/                        # Exported plots and charts
│   │   └── .gitkeep
│   ├── logs/                           # Rotating log outputs
│   │   └── .gitkeep
│   ├── models/                         # Serialized ML model weights and checkpoints
│   │   └── .gitkeep
│   └── reports/                        # Exported compliance audits and benchmark summaries
│       └── .gitkeep
│
├── contracts/                          # Canonical API contract package
│   ├── __init__.py                     # Exposes all schemas, interfaces, constants, exceptions, version
│   ├── constants.py                    # Enumerations for fuels, optimizers, and models
│   ├── exceptions.py                   # Custom domain exception hierarchy
│   ├── interfaces.py                   # Pure Abstract Base Classes for all layers
│   ├── schemas.py                      # Strongly typed dataclasses with JSON serialization
│   └── version.py                      # Canonical contract semantic version (CONTRACT_VERSION)
│
├── config.py                           # Centralized configuration dataclasses
├── constants.py                        # Backward-compatible root alias to contracts.constants
├── exceptions.py                       # Backward-compatible root alias to contracts.exceptions
├── interfaces.py                       # Backward-compatible root alias to contracts.interfaces
├── schemas.py                          # Backward-compatible root alias to contracts.schemas
├── logging_config.py                   # Centralized console and rotating file logger
├── project_structure.md                # Directory ownership and future phase mapping
├── README.md                           # Architecture and onboarding guide
├── requirements.txt                    # Project dependency specification
└── run_demo.py                         # Phase 0 repository health verification script
```

---

## 2. File & Directory Inventory

### Root Contract Files

| File | Primary Responsibility | Interface / Schema Exports |
| :--- | :--- | :--- |
| `config.py` | Centralized parameter storage using dataclasses. Defines paths, random seeds, optimizer defaults, and dashboard defaults. | `DataPaths`, `OutputPaths`, `OptimizerDefaults`, `DashboardDefaults`, `SystemConfig`, `get_default_config()` |
| `constants.py` | Strict string enums and tuples. Prevents hardcoding magic strings across modules. | `FuelType`, `OptimizerType`, `ModelType`, `SUPPORTED_FUELS`, `SUPPORTED_OPTIMIZERS`, `SUPPORTED_PREDICTION_MODELS` |
| `schemas.py` | Strongly typed immutable dataclasses defining data contracts exchanged between layers. | `VoyageRecord`, `PredictionResult`, `EmissionResult`, `ComplianceResult`, `FleetAssignment`, `OptimizationResult`, `ScenarioResult` |
| `contracts/interfaces.py` | Pure Abstract Base Classes (ABCs) defining execution signatures without business logic. | `DatasetLoader`, `PredictionEngine`, `FuelPhysicsEngine`, `EmissionEngine`, `ComplianceEngine`, `SchedulerEngine`, `OptimizationEngine`, `ScenarioEngine` |
| `contracts/version.py` | Semantic contract version indicator. Protects downstream modules from breaking schema changes. | `CONTRACT_VERSION` |
| `exceptions.py` | Granular domain exception hierarchy inheriting from a common base class. | `MaritimeSystemError`, `DataValidationError`, `PredictionError`, `OptimizationError`, `SchedulerError`, `ComplianceError`, `IntegrationError` |
| `logging_config.py`| Centralized logger configuration supporting stdout and rotating file handlers (`RotatingFileHandler`). | `configure_logging()` |
| `run_demo.py` | Standalone verification script confirming environment health, imports, and contracts without executing business logic. | `main()` |
| `requirements.txt`| Fixed dependency specifications spanning data, ML, optimization, UI, and QA tooling. | Dependency list |
| `README.md` | Comprehensive system design, architecture diagrams, module breakdown, and verification steps. | System documentation |
| `project_structure.md` | Full directory tree, ownership mapping, and phased implementation schedule. | Structural specification |

---

## 3. Subsystem Ownership & Layer Responsibilities

### `app/physics/`
- **Ownership:** Naval Architecture & Hydrodynamics Engineer
- **Contract:** Implements `FuelPhysicsEngine`
- **Core Focus:** Computes resistance coefficients, wave encounter frequencies, admiralty formulae, and converts brake power requirements to fuel mass.

### `app/prediction/`
- **Ownership:** Machine Learning Engineer
- **Contract:** Implements `DatasetLoader` and `PredictionEngine`
- **Core Focus:** Trains and performs inference on voyage data using Linear Regression, XGBoost, and Quantum-Inspired Fuel Consumption Prediction (`QIFCP`).

### `app/emissions/`
- **Ownership:** Sustainability & Lifecycle Assessment Specialist
- **Contract:** Implements `EmissionEngine`
- **Core Focus:** Evaluates Tank-to-Wake (TtW) direct combustion and Well-to-Wake (WtW) lifecycle carbon, methane, and nitrous oxide balances.

### `app/compliance/`
- **Ownership:** Regulatory & Maritime Policy Engineer
- **Contract:** Implements `ComplianceEngine`
- **Core Focus:** Validates operational voyages against IMO Carbon Intensity Indicator (CII) rating thresholds and EU FuelEU Maritime compliance penalties.

### `app/scheduler/`
- **Ownership:** Fleet Operations & Logistics Engineer
- **Contract:** Implements `SchedulerEngine`
- **Core Focus:** Enforces vessel laycan windows, port draft restrictions, cargo compatibility, and generates discrete vessel assignments.

### `app/optimization/`
- **Ownership:** Operations Research & Optimization Specialist
- **Contract:** Implements `OptimizationEngine` and `ScenarioEngine`
- **Core Focus:** Multi-objective heuristic searches (PSO, QPSO, NSGA-II) balancing bunker fuel costs against greenhouse gas emissions and transit delays.

### `app/dashboard/`
- **Ownership:** Full-Stack & UI/UX Engineer
- **Contract:** Consumes `ScenarioResult`, `OptimizationResult`, and `ComplianceResult`
- **Core Focus:** Interactive Streamlit web interface presenting scenario comparisons, Pareto frontiers, and vessel tracking views.

---

## 4. Future Phase Mapping

| Phase | Title | Modules Activated | Target Deliverables |
| :--- | :--- | :--- | :--- |
| **Phase 0** | **Foundation & Contracts** *(Current)* | Root files, test suite, scaffolding | Interfaces, schemas, configs, exceptions, logging, repo health test. |
| **Phase 1** | **Data & Hydrodynamic Physics** | `data/`, `app/physics/`, `app/prediction/loader.py` | Admiralty resistance model, synthetic voyage generator, data loaders. |
| **Phase 2** | **Predictive Fuel Modeling** | `app/prediction/` | Baseline Linear Regression, XGBoost, and Quantum-Inspired (QIFCP) regression. |
| **Phase 3** | **Emissions & Compliance Accounting** | `app/emissions/`, `app/compliance/` | WtW GHG coefficients, IMO CII grading formulas, FuelEU penalty calculators. |
| **Phase 4** | **Fleet Assignment & Scheduling** | `app/scheduler/` | Mixed-integer / constraint-based cargo-to-vessel matching engine. |
| **Phase 5** | **Quantum-Inspired Fleet Optimization** | `app/optimization/` | Particle Swarm Optimization (PSO), QPSO, and NSGA-II multi-objective solvers. |
| **Phase 6** | **Scenario Simulation & Benchmarking** | `app/optimization/scenario.py` | Multi-fuel scenario comparisons (Diesel vs. LNG vs. Methanol/Ammonia). |
| **Phase 7** | **Executive Dashboard & Visual UI** | `app/dashboard/` | Interactive Streamlit dashboard, Pareto frontier plots, and report exports. |
