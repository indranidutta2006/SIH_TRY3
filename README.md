# Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization

**Problem ID:** SIH26138  
**Phase:** Production Implementation & Validated Benchmark Platform (Deliverables D1–D10 Complete)  
**Target Runtime:** Python 3.12+

---

## 1. Project Overview

The **Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization** system is an enterprise-grade platform designed to assist commercial vessel operators, charterers, and maritime authorities in decarbonizing maritime logistics.

The objective of the platform is twofold:
1. **Accurately Predict Marine Fuel Consumption:** Capture nonlinear hydrodynamic resistance, adverse weather dynamics, and cargo payload constraints using classical regression (Linear Regression, Random Forest, HistGradientBoosting) and Quantum-Inspired Fuel Consumption Prediction (`QIFCP`) algorithms.
2. **Optimize Green Fleet Operations:** Allocate vessels to cargo orders and schedule routes to minimize total Well-to-Wake (WtW) greenhouse gas emissions, fuel expenditures, and operational delays while ensuring statutory compliance with International Maritime Organization (IMO) Carbon Intensity Indicator (CII) ratings and estimating European Union FuelEU Maritime GHG intensity limits and financial penalty exposures.

This repository contains the **complete production implementation and architectural foundation**. It integrates all mathematical models, quantum metaheuristic and hybrid evolutionary optimizers, machine learning pipelines, statutory compliance estimators, and the executive Streamlit dashboard required by Problem ID SIH26138.

---

## 2. Architectural Design & Boundaries

The system is decoupled into eight functional layers communicating strictly via immutable schemas and abstract interfaces:

```
+-----------------------------------------------------------------------------+
|                           Layer 8: Presentation Layer                       |
|                     (Streamlit / Plotly Scenario Explorer)                  |
+-----------------------------------------------------------------------------+
                                       |
+-----------------------------------------------------------------------------+
|                         Layer 7: Optimization Engine                        |
|                     (PSO / QPSO / NSGA-II Fleet Schedulers)                 |
+-----------------------------------------------------------------------------+
        |                              |                              |
        v                              v                              v
+------------------+         +-------------------+         +------------------+
| Layer 6:         |         | Layer 5:          |         | Layer 4:         |
| Scheduler Layer  |         | Compliance Layer  |         | Emissions Layer  |
| (Vessel-Cargo)   |         | (IMO CII, FuelEU) |         | (TtW & WtW GHG)  |
+------------------+         +-------------------+         +------------------+
                                       |
                                       v
                             +--------------------+
                             | Layer 3:           |
                             | Fuel Physics Layer |
                             | (Admiralty / Hydro)|
                             +--------------------+
                                       |
                                       v
                             +--------------------+
                             | Layer 2:           |
                             | Prediction Layer   |
                             | (ML / QIFCP)       |
                             +--------------------+
                                       |
                                       v
                             +--------------------+
                             | Layer 1:           |
                             | Data Layer         |
                             | (ETL & Validation) |
                             +--------------------+
```

### Module Responsibilities

| Module | Location | Primary Responsibility |
| :--- | :--- | :--- |
| **Data Layer** | `src/ingestion/`, `data/` | Ingestion, sanitization, schema validation, and storage of historical voyage records. |
| **Prediction Layer** | `src/prediction/` | Model training, inference, and benchmarking across Linear Regression, Random Forest, HistGradientBoosting, and QIFCP models. |
| **Fuel Physics Layer** | `src/physics/` | Hydrodynamic drag equations, admiralty coefficients, and propulsion energy conversion. |
| **Emissions Layer** | `src/prediction/emission_engine.py` | Tank-to-Wake (combustion) and Well-to-Wake (lifecycle) GHG quantification for diverse marine fuels. |
| **Compliance Layer** | `src/compliance/` | IMO CII rating computation (grades A to E) and EU FuelEU Maritime penalty evaluation. |
| **Scheduler Layer** | `src/scheduler/` | Timetable generation, laycan matching, port draft constraints, and cargo allocation. |
| **Optimization Layer**| `src/optimization/` | Heuristic and quantum-inspired search (PSO, QPSO, NSGA-II) for pareto-optimal trade-offs. |
| **Dashboard Layer** | `app/dashboard/`, `app.py` | Interactive visualization, tradeoff frontier plots, and what-if scenario simulators. |

---

## Deliverables

| # | Deliverable | Status | Output Path |
|---|-------------|--------|-------------|
| D1 | Mathematical Model | ✅ Complete | `contracts/schemas.py`, `src/physics/` |
| D2 | Fuel Consumption Prediction Module (QIFCP & QKP) | ✅ Complete | `src/prediction/qifcp.py`, `models/fuel_predictor.py` |
| D3 | Quantum Metaheuristic Optimizer (QPSO) | ✅ Complete | `src/optimization/qpso.py` |
| D4 | Alternative Fuel Scenario Analyser | ✅ Complete | `src/optimization/scenario_analysis.py` |
| D5 | Multi-Objective Optimization (DE-NSGA-II Pareto) | ✅ Complete | `src/optimization/nsga2_pareto.py` |
| D6 | Constraint Handler (FuelEU / IMO CII MEPC.400(83)) | ✅ Complete | `src/compliance/compliance_engine.py` |
| D7 | Benchmarking Suite | ✅ Complete | `scripts/benchmark_*.py`, `outputs/reports/` |
| D8 | Case Studies & Data Pipelines | ✅ Complete (Synthetic & Real Data) | `data/pipeline.py`, `outputs/reports/full_pipeline_run.json` |
| D9 | Integrated Software Platform (Streamlit) | ✅ Complete | `app.py` |
| D10 | Documentation & User Guide | ✅ Complete | `README.md` |

> [!NOTE] Empirical Methodology & Real-Data Boundary (Deliverable D8)
> In the European Union THETIS-MRV dataset, `fuel_consumption` is reported as an annual aggregate (total metric tons per year per vessel) rather than discrete per-voyage telemetry. The real-data adapter decomposes this figure across representative voyage legs using annual operational hours and average fuel burn per nautical mile, which introduces natural observational variance and explains why real-data RMSE is higher than synthetic benchmarks. Conversely, for genuinely novel zero-emission fuels (**Hydrogen**, **Ammonia**, and **ShorePower**) where commercial operational voyage records do not yet exist at scale, the physics engine (`src/physics/fuel_physics.py` / `MaritimeFuelPhysicsEngine`) computes consumption strictly via first-principles hydrodynamic resistance, specific energy densities, and engine thermal efficiencies rather than learned statistical approximations.

---

## 3. Integration Contracts

All subsequent modules plug directly into the defined abstract base classes in [`contracts/interfaces.py`](contracts/interfaces.py) and consume or produce the immutable dataclasses defined in [`contracts/schemas.py`](contracts/schemas.py):

1. **Plugging into Prediction:** Subclass `PredictionEngine` to implement `train()`, `predict()`, and `evaluate()`. Feed standard `VoyageRecord` sequences and return `PredictionResult` objects.
2. **Plugging into Physics:** Subclass `FuelPhysicsEngine` to implement `calculate_fuel_use()` and `calculate_energy()`.
3. **Plugging into Emissions:** Subclass `EmissionEngine` to implement `calculate_ttw()` and `calculate_wtw()`, outputting `EmissionResult`.
4. **Plugging into Compliance:** Subclass `ComplianceEngine` to implement `evaluate_cii()` and `evaluate_fueleu()`, outputting `ComplianceResult`.
5. **Plugging into Schedulers:** Subclass `SchedulerEngine` to implement `assign_vessels()` and `validate_constraints()`, returning `FleetAssignment`.
6. **Plugging into Optimizers:** Subclass `OptimizationEngine` to implement `optimize()` and `benchmark()`, producing `OptimizationResult`.
7. **Plugging into Scenarios:** Subclass `ScenarioEngine` to implement `run_scenario()` and `compare_scenarios()`, yielding `ScenarioResult`.

```python
# Canonical import pattern for all future modules:
from contracts import (
    VoyageRecord,
    PredictionResult,
    EmissionResult,
    ComplianceResult,
    CIIResult,
    FuelEUResult,
    ComplianceAssessment,
    FleetAssignment,
    OptimizationResult,
    ScenarioResult,
    PredictionEngine,
    FuelPhysicsEngine,
    EmissionEngine,
    ComplianceEngine,
    SchedulerEngine,
    OptimizationEngine,
    ScenarioEngine,
    FuelType,
    OptimizerType,
    ModelType,
    DEFAULT_EUR_TO_USD_FX_RATE,
    DataValidationError,
    PredictionError,
    ComplianceError,
    OptimizationError,
)
```

### Canonical Compliance Contract (`ComplianceResult`)

The compliance subsystem evaluates fleet voyages against statutory **IMO Carbon Intensity Indicator (CII)** ratings and **EU FuelEU Maritime** greenhouse gas intensity caps. The canonical dataclass contracts decouple independent statutory regimes while maintaining backwards compatibility:

| Field | Type | Description |
|:---|:---:|:---|
| `cii_rating` | `str` | Operational letter rating (`A` through `E` for CII, or `N/A` for FuelEU). |
| `attained_cii` | `float` | Attained annual operational CII in $\text{gCO}_2 / (\text{Capacity\_metric} \cdot \text{nm})$. |
| `required_cii` | `float` | Target required CII under IMO MEPC.337(76) & MEPC.400(83). |
| `cii_ratio` | `float` | Attained-to-Required CII ratio ($< 1.0$ indicates outperforming statutory target). |
| `fueleu_pass` | `Optional[bool]` | Statutory pass/fail against EU FuelEU limit (`None` during standalone CII assessment). |
| `fueleu_target` | `float` | Maximum permitted Well-to-Wake GHG intensity ($\text{gCO}_2\text{eq/MJ}$) for assessment year. |
| `ghg_intensity` | `float` | Attained Well-to-Wake GHG intensity ($\text{gCO}_2\text{eq/MJ}$). |
| `penalty_eur` | `float` | Statutory financial penalty in Euros (€) under Regulation (EU) 2023/1805 Article 23. |
| `compliance_status` | `str` | Standardized status string (`COMPLIANT` or `NON_COMPLIANT`). |
| `compliance_score` | `float` | *(Deprecated)* Legacy compatibility field. Use `cii_ratio` or `penalty_eur` directly. |
| `capacity_metric` | `str` | Statutory capacity metric basis (`DWT` or `GT`) under IMO Resolution MEPC.353(78) G2. |
| `reference_line_a` | `float` | Statutory reference curve regression parameter $a$ under IMO MEPC.353(78). |
| `reference_line_c` | `float` | Statutory reference curve regression exponent $c$ under IMO MEPC.353(78). |
| `rating_boundaries`| `tuple` | Statutory boundary vector $(d_1, d_2, d_3, d_4)$ under IMO Resolution MEPC.354(78) G4. |

### Decoupled Regulatory Architectures (`contracts.schemas`)
To prevent semantic overloading between IMO and EU regulations, the platform introduces dedicated schemas accessible directly or via `ComplianceResult` view properties:
- **`CIIResult`:** Encapsulates pure IMO operational carbon intensity metrics (`cii_rating`, `attained_cii`, `required_cii`, `cii_ratio`, `compliance_status`, `rating_boundaries`). Exposes `.is_compliant`.
- **`FuelEUResult`:** Encapsulates pure EU FuelEU Maritime compliance (`fueleu_pass`, `fueleu_target`, `ghg_intensity`, `penalty_eur`, `compliance_status`). Exposes `.is_compliant`.
- **`ComplianceAssessment`:** Unified multi-regulatory container (`cii: Optional[CIIResult]`, `fueleu: Optional[FuelEUResult]`).
- **Semantic Integrity:** When evaluating standalone CII, `fueleu_pass` is strictly `None` (rather than copying the CII pass status), guaranteeing that an operational CII downgrade (e.g. Grade E) never falsely flags an EU FuelEU deficit.

> [!WARNING] Deprecation Notice: `compliance_score`
> The field `compliance_score` is deprecated and preserved strictly for backwards compatibility with legacy tests. It previously functioned as a polymorphic alias (`cii_ratio` for CII, `penalty_eur` for FuelEU). All downstream optimization engines (`FleetObjective`, `nsga2_pareto`, `ScenarioAnalysis`) and dashboards consume the explicit canonical fields `penalty_eur` and `cii_ratio` directly.

### Canonical Prediction Contracts & Model Registry Alignment (`ModelType`)

The central architecture contracts in [`contracts/constants.py`](contracts/constants.py) strictly align with the estimators implemented in [`src/prediction/model_registry.py`](src/prediction/model_registry.py) via `ModelType`:

| `ModelType` Enum | Registry Identifier | Estimator Class | Implementation Status | Description |
|:---|:---:|:---:|:---:|:---|
| `ModelType.LINEAR_REGRESSION` | `linear_regression` | `LinearRegression` | ✅ Active | Ordinary Least Squares hydrodynamic baseline. |
| `ModelType.RANDOM_FOREST` | `random_forest` | `RandomForestRegressor` | ✅ Active | Bagged ensemble of decision trees ($N=100$, max depth 15). |
| `ModelType.HIST_GRADIENT_BOOSTING` | `hist_gradient_boosting` | `HistGradientBoostingRegressor` | ✅ Active | Scikit-learn native histogram-based gradient boosted decision tree regressor. |
| `ModelType.QIFCP` | `qifcp` | `QIFCPRegressor` | ✅ Active | Quantum-Inspired Fuel Consumption Predictor with nested vessel-disjoint QPSO tuning. |
| `ModelType.XGBOOST` *(Alias)* | `hist_gradient_boosting` | `HistGradientBoostingRegressor` | 🔄 Architectural Alias | Contract alias mapped to native histogram GBDT (eliminates unpinned external C++ binaries). |

> [!NOTE] Architectural Clarification: XGBoost & Native GBDT Alignment
> Early Phase 0 contract specifications referenced `ModelType.XGBOOST`. However, to ensure deterministic cross-platform execution (Windows, Linux, Docker, Render cloud containers) without unpinned native C++ dynamic library dependencies (`libxgboost`, OpenMP runtime mismatches), the implementation standardizes on `HistGradientBoostingRegressor` (`scikit-learn`). `HistGradientBoosting` belongs to the same broad class of histogram-based gradient-boosted decision tree methods as XGBoost and LightGBM (inspired by LightGBM's binning paradigm). For architectural compatibility, `ModelType.XGBOOST` is preserved as a valid contract alias and transparently resolves to `hist_gradient_boosting` in `normalize_model_name()`.

### Canonical Alternative Fuel & Electrification Contract (`ScenarioResult`)

The multi-fuel scenario analysis and green fleet optimization modules output standardized results conforming to `ScenarioResult` in [`contracts/schemas.py`](contracts/schemas.py). The schema cleanly decouples bunker fuel expenditures ($USD) from statutory regulatory penalties (€EUR) while providing native electrical energy tracking for cold-ironing electrification:

| Field | Type | Description |
|:---|:---:|:---|
| `scenario_name` | `str` | Descriptive identifier for the operational scenario (e.g., `"Methanol Transition"`). |
| `fuel_type` | `str` | Evaluated marine fuel or power source (`Diesel`, `LNG`, `Methanol`, `Hydrogen`, `Ammonia`, `ShorePower`). |
| `total_cost` | `float` | Unified financial cost in $USD ($\text{fuel\_cost\_usd} + \text{fueleu\_penalty\_usd}$). |
| `total_emissions` | `float` | Aggregate lifecycle Well-to-Wake (WTW) $\text{CO}_2\text{e}$ emissions in metric tons. |
| `fuel_consumption` | `float` | Direct combustion fuel mass in metric tons ($0.0\text{ t}$ for cold-ironing `ShorePower`). |
| `energy_consumption_mwh` | `float` | Total energy consumed in megawatt-hours (MWh), providing direct thermodynamic comparability between grid electricity and chemical fuels. |
| `fuel_cost_usd` | `float` | Direct bunker fuel or electricity expenditure in US Dollars ($USD). |
| `fueleu_penalty_eur` | `float` | Statutory FuelEU Maritime penalty exposure in Euros (€EUR) under Regulation (EU) 2023/1805. |
| `fueleu_penalty_usd` | `float` | Converted FuelEU Maritime penalty exposure in US Dollars ($USD) at the evaluation FX rate. |
| `exchange_rate_eur_to_usd`| `float` | Configurable foreign exchange conversion multiplier (baseline `1.08` USD per EUR). |

---

## 4. Quick Start

A judge or evaluator can reproduce the entire platform from a fresh `git clone` to the running interactive dashboard by following these five sequential steps:

### Step 1: Environment Setup & Installation
Clone the repository and install required production dependencies within a clean virtual environment:
```bash
git clone https://github.com/indranidutta2006/SIH_TRY3.git
cd SIH_TRY3
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
pip install -r requirements.txt
```
* **Produces:** Isolated Python 3.12+ execution environment equipped with all scientific, machine learning, quantum optimization, and dashboard visualization dependencies.

---

### Step 2: Generate the Synthetic Voyage Dataset
Synthesize the canonical 10,000-record historical maritime voyage dataset:
```bash
python scripts/make_mock_dataset.py --rows 10000 --output data/raw/voyages_sample.csv
```
* **Produces:** `data/raw/voyages_sample.csv` (10,000 typed, physically validated voyage records containing hydrodynamic resistance, adverse weather factors, and cargo payload constraints).

---

### Step 3: Run the Full End-to-End Pipeline
Execute the full multi-stage analytical and training pipeline:
```bash
python scripts/run_full_pipeline.py
```
> [!TIP]
> To train on heterogeneous real-world data (THETIS-MRV + FuelCast sensor telemetry blended 50/50 with synthetic records under rate formulation), add `--use-real-data`:
> ```bash
> python scripts/run_full_pipeline.py --use-real-data
> ```

* **Produces:**
  - Trained baseline model binaries (`artifacts/models/linear_regression.pkl`, `artifacts/models/random_forest.pkl`, `artifacts/models/hist_gradient_boosting.pkl`)
  - Model registry metrics manifest (`artifacts/metrics/baseline_metrics.json`)
  - Prediction benchmark leaderboard (`outputs/reports/prediction_benchmark.json`)
  - Swarm optimization scalability benchmarks (`outputs/reports/optimization_benchmark.json`)
  - 6-Fuel macro decarbonization scenario comparisons (`outputs/reports/scenario_comparison.json`)
  - Full execution pipeline run manifest (`outputs/reports/full_pipeline_run.json`)

---

### Step 4: Verify System Integrity via Test Suite
Verify that all architectural contracts, physics derivations, statutory emissions rules, and zero-leakage guards pass:
```bash
python -m pytest tests/ -v
```
* **Produces:** 250/250 passing deterministic tests (**0 failures, 0 errors**) confirming strict contract adherence, proxy-leakage neutralization, granular statutory MEPC.353(78) G2 baseline curves (including 279k DWT Bulk, 57.7k GT Ro-Ro Vehicle carrier, and 65k DWT LNG carrier effective capacity caps), strict statutory capacity unit validation (GT vs DWT), strict statutory vessel category validation (rejection of unsupported vessel categories via `ComplianceError`), dimensional currency consistency and explicit foreign exchange conversion (EUR to USD) across all optimization objectives and scenario simulations, ShorePower explicit electrical energy accounting (`energy_consumption_mwh`) with per-vessel energy tracking in heterogeneous fleets, decoupled bunker pricing (`FUEL_PRICES_USD_PER_TON`) and electricity tariff (`SHORE_POWER_PRICE_USD_PER_MWH`), MEPC.400(83) compliance targets, MEPC.354(78) G4 ship-type-specific rating boundaries, decoupled statutory compliance schemas, FuelEU Article 23(2) consecutive-deficit penalty scaling, canonical Deb et al. (2002) NSGA-II crowding-distance environmental selection, nested vessel-disjoint inner validation for QPSO-tuned QIFCP, unified system-wide dual-route fuel architecture, dynamic vessel-weather parameterization across fleet optimization routines, and full ModelType central contract & ModelRegistry estimator alignment.

---

### Step 5: Launch the Executive Dashboard
Start the interactive Streamlit presentation application:
```bash
streamlit run app.py
```
* **Produces:** Live web dashboard served at `http://localhost:8501` featuring:
  1. **Fleet Telemetry & Explorer:** Distribution histograms, correlation heatmaps, and raw voyage inspection.
  2. **Fuel Prediction & QIFCP:** 4-model accuracy leaderboard, normalized per-source error breakdowns, and live what-if voyage inference sandbox.
  3. **Swarm Optimization (QPSO):** Scalability comparisons across Small, Medium, and Large fleet tiers with interactive bi-objective Pareto frontier trade-offs.
  4. **Statutory Compliance (IMO/EU):** Automated vessel-level IMO CII ratings (A–E) across all MEPC.353(78) G2 reference curves (with dynamic DWT/GT capacity metrics) and FuelEU Maritime penalty audits.
  5. **Scenario Analysis & Alternative Fuels:** Full Well-to-Wake (WTW) lifecycle emissions, operational costs, and balanced multicriteria rankings across 6 alternative marine fuels.

---

### Pipeline Execution Stages & Output Artifacts

| Stage | Name | Pipeline Responsibility | Primary Output Artifacts |
|:---:|---|---|---|
| **1** | **Data Ingestion & Validation** | Ingests, type-checks, and validates voyage records against schema boundaries | `data/raw/voyages_sample.csv`, `outputs/logs/dataset_generator.log` |
| **2** | **Prediction Layer & QIFCP** | Extracts 29 hydrodynamic features, trains baselines, and tunes QIFCP via QPSO | `artifacts/models/*.pkl`, `artifacts/metrics/baseline_metrics.json`, `outputs/reports/prediction_benchmark.json` |
| **3** | **Emissions Compliance** | Computes Tank-to-Wake (TTW) & Well-to-Wake (WTW) GHG against IMO CII & FuelEU limits | Audited in-memory via `ComplianceEngine` |
| **4** | **Fleet Scheduler** | Validates laycan windows, draft constraints, and cargo capacity matching | Audited in-memory via `FleetScheduler` |
| **5** | **Optimization Swarm & Pareto** | Executes equal-budget QPSO vs. PSO across fleet tiers and solves bi-objective Pareto front | `outputs/reports/optimization_benchmark.json` |
| **6** | **Scenario Analysis** | Evaluates fleet-wide transitions across Diesel, LNG, Methanol, Hydrogen, Ammonia, and ShorePower | `outputs/reports/scenario_comparison.json`, `outputs/reports/full_pipeline_run.json` |

---

## 5. Lifecycle Emissions, Statutory Compliance & Quantum Prediction Modeling

The platform models maritime decarbonization through three mathematically grounded engines:

### 5.1 Lifecycle Greenhouse Gas Accounting (TTW & WTW)
Implemented in `src/prediction/emission_engine.py` conforming to [`contracts.interfaces.EmissionEngine`](contracts/interfaces.py):
- **Tank-to-Wake (TTW):** Direct operational combustion emissions ($\text{CO}_2$, $\text{CH}_4$, $\text{N}_2\text{O}$, and $\text{CO}_2\text{e}$) calculated using statutory IMO MEPC factors and IPCC AR5 100-year Global Warming Potentials ($\text{GWP}_{\text{CH}_4} = 28.0, \text{GWP}_{\text{N}_2\text{O}} = 265.0$).
- **Well-to-Wake (WTW):** Total lifecycle climate footprint ($\text{WTW} = \text{TTW} + \text{WTT}$), capturing upstream fuel production, refining, liquefaction, transport, and bunkering. Direct combustion $\text{CO}_2$ is maintained separately from $\text{CO}_2\text{e}$ to avoid mixing physical combustion units with overall lifecycle climate impact.

### 5.2 IMO Carbon Intensity Indicator (CII), Resolution MEPC.353(78) (G2) & MEPC.400(83)
Implemented in `src/compliance/compliance_engine.py` conforming to [`contracts.interfaces.ComplianceEngine`](contracts/interfaces.py):
- **Attained CII:** Calculated per voyage or annually as:
  $$\text{CII}_{\text{attained}} = \frac{\text{CO}_2 \times 10^6}{\text{Capacity} \times \text{Distance}}$$
  where $\text{Capacity}$ is evaluated strictly in **Gross Tonnage (GT)** for Ro-Ro and passenger vessels, and **Deadweight Tonnage (DWT)** for cargo, container, tanker, gas, and bulk vessels.
- **Granular Baseline Reference Lines (IMO Resolution MEPC.353(78) G2):** Superseding the single-curve baseline in MEPC.337(76), the engine implements the full multi-branch statutory lookup:
  $$\text{Vessel Type} + \text{Capacity Metric} + \text{Capacity Value} \longrightarrow (a, c, \text{Effective Capacity})$$
  $$\text{CII}_{\text{ref}} = a \cdot \text{Capacity}^{-c} \quad (\text{or } a \text{ if } c = 0)$$

| Vessel Category | Capacity Sub-tier | Capacity Metric | Parameter $a$ | Exponent $c$ |
|:---|:---|:---:|:---:|:---:|
| **General Cargo Ship** | $\ge 20,000$ DWT | DWT | 31,948.0 | 0.7920 |
| **General Cargo Ship** | $< 20,000$ DWT | DWT | 588.0 | 0.3885 |
| **LNG Carrier** | $\ge 100,000$ DWT | DWT | 9.827 | 0.0000 |
| **LNG Carrier** | $65,000 \le \text{DWT} < 100,000$ | DWT | $1.4479 \times 10^{14}$ | 2.6730 |
| **LNG Carrier** | $< 65,000$ DWT (statutory fixed effective capacity at 65,000 DWT) | DWT | $1.4779 \times 10^{14}$ | 2.6730 |
| **Ro-Ro Vehicle Carrier** | $\ge 57,700$ GT (cap at 57,700 GT) | **GT** | 3,627.0 | 0.5900 |
| **Ro-Ro Vehicle Carrier** | $30,000 \le \text{GT} < 57,700$ | **GT** | 3,627.0 | 0.5900 |
| **Ro-Ro Vehicle Carrier** | $< 30,000$ GT | **GT** | 330.0 | 0.3290 |
| **Ro-Ro Cargo Ship** | All sizes | **GT** | 1,967.0 | 0.4850 |
| **Ro-Ro Passenger Ship** | All sizes | **GT** | 2,023.0 | 0.4600 |
| **Cruise Passenger Ship** | All sizes | **GT** | 930.0 | 0.3830 |
| **Bulk Carrier** | All sizes (cap at 279,000 DWT) | DWT | 4,745.0 | 0.6220 |
| **Tanker** | All sizes | DWT | 5,247.0 | 0.6100 |
| **Containership** | All sizes | DWT | 1,984.0 | 0.4890 |
| **Gas Carrier** | $\ge 65,000$ DWT | DWT | $1.4405 \times 10^{11}$ | 2.0710 |
| **Gas Carrier** | $< 65,000$ DWT | DWT | 8,104.0 | 0.6390 |
| **Refrigerated Cargo** | All sizes | DWT | 4,600.0 | 0.5570 |
| **Combination Carrier**| All sizes | DWT | 5,119.0 | 0.6220 |

- **Statutory Reduction Factor $Z$ (MEPC.400(83)):** On 11 April 2025, IMO MEPC 83 adopted updated annual reduction trajectories:

| Year | Statutory $Z$ Factor | Governing Instrument |
|:---:|:---:|:---|
| **2023** | 5.000% (`0.05000`) | IMO Resolution MEPC.337(76) |
| **2024** | 7.000% (`0.07000`) | IMO Resolution MEPC.337(76) |
| **2025** | 9.000% (`0.09000`) | IMO Resolution MEPC.337(76) |
| **2026** | 11.000% (`0.11000`) | IMO Resolution MEPC.337(76) |
| **2027** | **13.625%** (`0.13625`) | **IMO Resolution MEPC.400(83)** |
| **2028** | **16.250%** (`0.16250`) | **IMO Resolution MEPC.400(83)** |
| **2029** | **18.875%** (`0.18875`) | **IMO Resolution MEPC.400(83)** |
| **2030** | **21.500%** (`0.21500`) | **IMO Resolution MEPC.400(83)** |

- **Statutory Lookup & Regulatory Range Enforcement:** Implemented via immutable lookup table `CII_Z_FACTORS`. Both `get_cii_z_factor(year)` and `evaluate_cii(..., year=year)` strictly reject calendar years outside the statutory regulatory range ($[2023, 2030]$) by raising `ComplianceError`, preventing erroneous extrapolations or arbitrary linear drift.
- **Strict Statutory Capacity Metric Enforcement (`DataValidationError`):** Gross Tonnage (GT) and Deadweight Tonnage (DWT) measure fundamentally distinct physical quantities (internal enclosed volume vs cargo deadweight carrying capacity) and cannot be legally interchanged. The engine strictly validates supplied capacity units against Table 1 of IMO Resolution MEPC.353(78): Ro-Ro and passenger vessels mandate **GT**, while bulk, tanker, container, general cargo, and gas vessels mandate **DWT**. Supplying an incompatible capacity metric (such as DWT for a Ro-Ro cargo ship or GT for a bulk carrier) immediately raises `DataValidationError`.
- **Strict Statutory Vessel Category Validation (`ComplianceError`):** A regulatory calculator must never silently default or re-assign unmodeled or unsupported vessel categories to an arbitrary fallback type. Both `resolve_cii_reference_line` and `resolve_cii_rating_boundaries` strictly validate the input vessel category against the 12 statutory ship types defined in IMO Resolutions MEPC.353(78) and MEPC.354(78). Unrecognized or unsupported vessel types (such as `"Oil Service Vessel"`, `"Tugboat"`, offshore supply vessels, or unmodeled craft) explicitly raise `ComplianceError` with structured details, preventing the silent acquisition of Bulk Carrier statutory parameters.
- **Ship-Type-Specific Letter Rating Boundaries (IMO Resolution MEPC.354(78) G4):** Superseding simplified universal thresholds (`0.83, 0.94, 1.06, 1.19`), the engine resolves statutory rating boundary vectors $(d_1, d_2, d_3, d_4)$ conforming to Table 1 of Resolution MEPC.354(78). Operational rating (A–E) is evaluated against the statutory ratio $r = \text{CII}_{\text{attained}} / \text{CII}_{\text{required}}$:
  - **Grade A (Superior):** $r \le \exp(d_1)$
  - **Grade B (Minor Superior):** $\exp(d_1) < r \le \exp(d_2)$
  - **Grade C (Moderate):** $\exp(d_2) < r \le \exp(d_3)$
  - **Grade D (Inferior):** $\exp(d_3) < r \le \exp(d_4)$
  - **Grade E (Inferior / Non-compliant):** $r > \exp(d_4)$

| Statutory Vessel Category | Capacity Sub-tier | $\exp(d_1)$ (A/B) | $\exp(d_2)$ (B/C) | $\exp(d_3)$ (C/D) | $\exp(d_4)$ (D/E) |
|:---|:---|:---:|:---:|:---:|:---:|
| **Bulk Carrier** | All sizes | 0.86 | 0.94 | 1.06 | 1.18 |
| **Tanker** | All sizes | 0.82 | 0.93 | 1.08 | 1.28 |
| **Containership** | All sizes | 0.83 | 0.94 | 1.07 | 1.19 |
| **General Cargo Ship** | All sizes | 0.83 | 0.94 | 1.06 | 1.19 |
| **LNG Carrier** | $\ge 100,000$ DWT | 0.89 | 0.98 | 1.06 | 1.13 |
| **LNG Carrier** | $< 100,000$ DWT | 0.78 | 0.92 | 1.10 | 1.37 |
| **Ro-Ro Vehicle Carrier** | All sizes | 0.86 | 0.94 | 1.06 | 1.16 |
| **Ro-Ro Cargo Ship** | All sizes | 0.76 | 0.89 | 1.08 | 1.27 |
| **Ro-Ro Passenger Ship** | All sizes | 0.76 | 0.92 | 1.14 | 1.30 |
| **Cruise Passenger Ship** | All sizes | 0.87 | 0.95 | 1.06 | 1.16 |
| **Gas Carrier** | $\ge 65,000$ DWT | 0.81 | 0.91 | 1.12 | 1.44 |
| **Gas Carrier** | $< 65,000$ DWT | 0.85 | 0.95 | 1.06 | 1.25 |
| **Refrigerated Cargo** | All sizes | 0.78 | 0.91 | 1.07 | 1.20 |
| **Combination Carrier** | All sizes | 0.87 | 0.96 | 1.06 | 1.14 |

> [!NOTE] Defensible Regulatory Scope Notice: IMO CII Operational Estimator
> This module functions specifically as an **IMO Carbon Intensity Indicator (CII) Rating and Attained Metric Estimator** conforming to the geometric foundations of IMO Resolution MEPC.353(78) (G2 reference curves), MEPC.354(78) (G4 rating boundaries), and MEPC.400(83) (annual reduction trajectories). It does not claim full statutory flag-state certification as it intentionally evaluates standard uncorrected operational intensity without simulating the optional voyage adjustments and correction factors specified under **IMO Resolution MEPC.355(78) (G5)** (such as ice-class navigation allowances, refrigerated container electrical loads, cargo heating, shuttle tanker dynamic positioning, or severe weather search-and-rescue voyage exclusions).

### 5.3 EU FuelEU Maritime GHG-Intensity Compliance & Penalty Estimator (Regulation (EU) 2023/1805)
Implemented in `src/compliance/compliance_engine.py` conforming to Article 4, Article 23, and Annex IV:
$$\text{Base Penalty (EUR)} = \frac{|\text{Compliance Balance (gCO}_2\text{eq)}|}{\text{GHGIE}_{\text{actual}} \times 41000\text{ MJ/t}} \times 2400\text{ EUR/t}$$
$$\text{Total Penalty (EUR)} = \text{Base Penalty} \times \left(1 + \frac{n - 1}{10}\right)$$
where:
- $\text{Compliance Balance} = (\text{GHGIE}_{\text{target}} - \text{GHGIE}_{\text{actual}}) \times \text{Energy Consumed (MJ)}$
- **Consecutive Deficit Multiplier (Article 23(2)):** When a vessel incurs a compliance deficit across $n \ge 2$ consecutive reporting periods, penalties scale by $1 + \frac{n-1}{10}$ ($1.1\times$ for $n=2$, $1.2\times$ for $n=3$, etc.).
- **Dimensional Foreign Exchange Alignment (`DEFAULT_EUR_TO_USD_FX_RATE`):** Statutory FuelEU penalties are legally denominated in Euros (€) under Regulation (EU) 2023/1805 Article 23, whereas international bunker fuel markets quote in US Dollars ($/t). Downstream optimization routines (`fleet_objective`, `nsga2_pareto`, and `scenario_analysis`) resolve this dimensional mismatch by converting statutory EUR penalties to USD using an explicit configurable exchange rate (`eur_to_usd_rate`, default `1.08` USD per EUR) before combining them with bunker fuel expenditures:
  $$\text{Total Financial Cost (USD)} = \text{Bunker Fuel Cost (USD)} + \left(\text{FuelEU Penalty (EUR)} \times \text{FX}_{\text{EUR}\to\text{USD}}\right)$$
  `ScenarioResult` explicitly tracks `fuel_cost_usd`, `fueleu_penalty_eur`, `fueleu_penalty_usd`, and `exchange_rate_eur_to_usd` to guarantee transparent financial accounting.

> [!NOTE] Defensible Regulatory Scope Notice
> This component functions specifically as a **FuelEU GHG-intensity compliance and penalty estimator**. It evaluates operational Well-to-Wake GHG intensity targets (Article 4) and financial penalty exposure (Article 23 & Annex IV). It does not claim full statutory compliance certification as it intentionally does not simulate:
> 1. Renewable Fuels of Non-Biological Origin (**RFNBO**) subtargets (Article 5)
> 2. Onshore Power Supply (**OPS**) zero-emission berth requirements for container and passenger ships (Article 6)
> 3. Fleet compliance **pooling** mechanisms (Article 21) or surplus **banking/borrowing** (Article 20)
> 4. Port-call geographic exemptions (outermost regions, islands, or ice-class provisions).

### 5.4 Quantum-Inspired & Quantum Kernel Modeling
- **QIFCP (`src/prediction/qifcp.py`):** Quantum-inspired hydrodynamic neural regressor tuned with Quantum-behaved Particle Swarm Optimization (`QPSO`). Encodes standardized hydro-meteorological features into quantum phase angles $\theta_j = \arctan(\gamma \cdot z_j)$ and pairwise quantum superposition-entanglement correlation states.
- **Nested Vessel-Disjoint Validation Protocol:** To eliminate data leakage and guard against optimistic generalization bias across all evaluation stages, the QIFCP hyperparameter tuning and model evaluation pipeline enforces strict nested vessel-disjoint splitting:
  $$\text{Outer Vessel-Disjoint Split (Test Evaluation)} \longrightarrow \text{Training Partition} \longrightarrow \text{Inner Vessel-Disjoint Split (QPSO Hyperparameter Search)}$$
  In `QIFCPRegressor.tune_with_qpso(X, y, groups=vessel_ids)`, the inner validation partition is constructed using `GroupShuffleSplit` over vessel identities. This ensures that no vessel present in the inner validation fold appears in the inner training fold while QPSO searches for optimal phase scaling ($\gamma$) and L2 regularization ($\alpha_{\text{reg}}$).
- **QuantumKernelPredictor (`models/fuel_predictor.py`):** PennyLane simulation (`default.qubit`) utilizing `AngleEmbedding` on 4 normalized hydrodynamic features (`speed_kn`, `load_factor`, `wave_ht_m`, `wind_bft`) combined with a 2-repetition `ZZFeatureMap` entangling circuit. The resulting quantum state transition kernel matrix is fed into `KernelRidge(alpha=0.1)`, achieving $R^2 > 0.88$ on cross-class synthetic voyages.
  > *Quantum principle used:* Interference between feature-encoded states produces a similarity metric unavailable to classical RBF kernels.

> [!NOTE] Empirical Validation Metrics vs. Nominal Confidence Scores
> In compliance with rigorous statistical standards, model evaluation on the executive dashboard and benchmark reports is grounded exclusively in verifiable empirical metrics ($R^2$, RMSE, MAE, MAPE, NRMSE, fit times, and cross-source generalization spreads). Downstream schemas retain `confidence_score` solely as an immutable contract compatibility field, intentionally excluding synthetic scalar confidence percentages from user-facing presentations in favor of defensible statistical metrics.

> [!NOTE] Alternative Fuel Baseline Assumptions
> In this baseline implementation, alternative power pathways (**Hydrogen**, **Ammonia**, and **ShorePower**) reflect certified green renewable supply chains (e.g., green hydrogen from water electrolysis powered by renewables, green ammonia synthesized with zero-carbon energy, and zero-emission shore grid connections).
> 
> In industrial operations:
> - *Green hydrogen $\ne$ all hydrogen* (fossil steam methane reforming entails significant upstream WTT emissions).
> - *Green ammonia $\ne$ all ammonia* (Haber-Bosch with fossil feedstock has high carbon intensity).
> - *Grid shore power $\ne$ zero lifecycle emissions* (grid carbon intensity varies widely by regional port grid mix).
> 
> The architecture explicitly provides configurable `wtt_factors` in `MaritimeEmissionEngine`, allowing operators to model grey, blue, or green fuel pathways dynamically.

### 5.5 Multi-Objective Green Fleet Optimization & Hybrid DE-NSGA-II Pareto Solver
Implemented in `src/optimization/nsga2_pareto.py` (`ParetoFleetOptimizer`):
- **Bi-Objective Tradeoff Formulation:** Solves for Pareto-optimal decision vectors $x = [v_1, f_1, v_2, f_2, \dots, v_n, f_n]$ across scheduled fleet voyages ($v_i \in [10, 20]\text{ knots}$, $f_i \in \{\text{Diesel, LNG, Methanol, Hydrogen, Ammonia, ShorePower}\}$):
  $$f_1(x) = \text{Bunker Fuel Cost (USD)} + \left(\text{FuelEU Penalties (EUR)} \times \text{FX}_{\text{EUR}\to\text{USD}}\right)$$
  $$f_2(x) = \text{Total Lifecycle Well-to-Wake CO}_2\text{e Emissions (metric tons)}$$
- **Fast Non-Dominated Sorting:** Partitions the combined $2N$ parent and offspring population ($R_t = P_t \cup Q_t$) into sequential domination fronts $\mathcal{F}_1, \mathcal{F}_2, \dots$ in $\mathcal{O}(M N^2)$ time based on strict Pareto dominance ($p \prec q \iff \forall m: f_m(p) \le f_m(q) \land \exists m: f_m(p) < f_m(q)$).
- **Canonical Crowding-Distance Environmental Selection (Deb et al., 2002):**
  When transitioning between generations, successive non-dominated fronts are admitted until the next front $\mathcal{F}_l$ exceeds remaining population capacity ($|P_{t+1}| + |\mathcal{F}_l| > N$). Rather than applying naive truncation (`front[:needed]`), the solver computes the canonical crowding distance for all individuals in $\mathcal{F}_l$:
  $$d_i = \sum_{m=1}^{M} \frac{f_m(i+1) - f_m(i-1)}{f_m^{\max} - f_m^{\min}}$$
  where boundary solutions with extreme minimal and maximal values along each objective are explicitly assigned infinite distance ($d_i = \infty$).
  The front is sorted in descending order of crowding distance, and the top $N - |P_{t+1}|$ individuals are selected. This canonical survival mechanism guarantees the preservation of extreme boundary solutions (minimum cost and minimum emissions configurations) and maximizes the diversity and uniform spread of solutions along the Pareto frontier.
- **Offspring Generation Nuance (DE-NSGA-II / DEMO Architecture):**
  While the environmental survival mechanism strictly implements Deb et al. (2002) non-dominated sorting and crowding distance, offspring generation uses continuous **differential-evolution variation** (DE/rand/1/bin mutation: $v = x_{r1} + F(x_{r2} - x_{r3})$ with differential weight $F=0.8$, combined with binomial crossover at $CR=0.7$) rather than textbook genetic algorithm simulated binary crossover (SBX) and polynomial mutation. In multi-objective evolutionary computation literature, this hybrid paradigm is formally designated as **DE-NSGA-II** or **DEMO** (Differential Evolution for Multiobjective Optimization), leveraging DE's continuous step adaptation while preserving NSGA-II's elitist frontier geometry.

### 5.6 Unified System-Wide Fuel-Routing Architecture (ML vs. First-Principles Physics Boundary)
To maintain strict scientific integrity across both comparative scenario simulations and automated fleet optimization routines, the platform enforces a **unified dual-route fuel calculation architecture**:

```
                       Scheduled Voyage / Fuel Selection
                                      |
                 +--------------------+--------------------+
                 |                                         |
     ML-Routed Conventional /                   First-Principles Novel
     Transitional Fuels                         Zero-Emission Pathways
    (Diesel, LNG, Methanol)                    (Hydrogen, Ammonia, ShorePower)
                 |                                         |
                 v                                         v
   +----------------------------+            +----------------------------+
   |  ProductionModelManager    |            | MaritimeFuelPhysicsEngine  |
   | (Vectorized Batch Predict) |            | (Admiralty / Hydrodynamics)|
   +----------------------------+            +----------------------------+
                 |                                         |
                 |      +----------------------------------+
                 |      |
                 v      v
   +------------------------------------------------------+
   |         Downstream Statutory & Cost Engines          |
   | - MaritimeEmissionEngine (Lifecycle WTW CO2e)        |
   | - MaritimeComplianceEngine (FuelEU Penalty Estimator)|
   | - Scheduled Fuel & Electricity Cost Evaluation       |
   +------------------------------------------------------+
```

- **ML-Trained Fuels (`Diesel`, `LNG`, `Methanol`):** Dispatched in vectorized batches through `ProductionModelManager` (Linear Regression, Random Forest, HistGradientBoosting, or QIFCP). These fuels possess empirical training distributions derived from sensor telemetry and MRV logbooks.
- **Novel Zero-Emission Pathways (`Hydrogen`, `Ammonia`, `ShorePower`):** Dispatched exclusively through `MaritimeFuelPhysicsEngine`. Because commercial operational voyage records for deep-sea hydrogen or ammonia carriers do not yet exist at historical scale, the statistical ML regressors are **never called** for these fuels. Instead, fuel consumption is derived from first-principles naval architecture:
  $$P_{\text{propulsion}} = \frac{\Delta^{2/3} \cdot V^3}{C_{\text{adm}}}, \quad E_{\text{mech}} = (P_{\text{propulsion}} + P_{\text{aux}}) \times t_{\text{sea}} \times w_{\text{weather}}$$
  $$\text{Fuel Mass (t)} = \frac{E_{\text{mech}} / \eta_{\text{thermal}}}{\text{LHV}_{\text{fuel}}}$$
- **ShorePower Direct Grid Accounting & Energy Representation:** For vessels drawing cold-ironing shore power (`ShorePower`), operational combustion fuel mass is $0.0\text{ metric tons}$, operational $\text{CO}_2\text{e}$ emissions are $0.0\text{ metric tons}$, and FuelEU penalty exposure is $0.0\text{ EUR}$. To avoid the misleading representation of "zero consumption" when cold-ironing replaces bunker fuel, `ScenarioResult` explicitly reports electrical energy consumption in megawatt-hours (`energy_consumption_mwh`):
  $$\text{Cost} = E_{\text{electrical}}\text{ (MWh)} \times \text{Electricity Tariff (USD/MWh)}$$
  For all thermal combustion fuels, `energy_consumption_mwh` reports thermodynamic fuel energy ($\text{Fuel Mass} \times \text{LCV} / 3600$), ensuring direct dimensional comparability between electrified grid power and alternative fuels.
- **System-Wide Uniformity:** This dispatch boundary is identically enforced across `ScenarioAnalysisEngine` (`src/optimization/scenario_analysis.py`), scalar fleet objective evaluation for QPSO/PSO (`src/optimization/fleet_objective.py`), and bi-objective Pareto optimization (`src/optimization/nsga2_pareto.py`).

### 5.7 Dynamic Vessel & Weather Parameterization in Fleet Optimization (`voyage_specs` & `vessel_specs`)
To guarantee physical realism and ensure that scheduling and routing decisions reflect authentic vessel heterogeneity and environmental conditions, both [`fleet_objective.py`](src/optimization/fleet_objective.py) and [`nsga2_pareto.py`](src/optimization/nsga2_pareto.py) resolve vessel and meteorological characteristics dynamically rather than substituting hardcoded constants:

```
                  Context Dictionary (context)
                                |
        +-----------------------+-----------------------+
        |                                               |
  vessel_specs                                     voyage_specs
(per-vessel defaults:                            (per-cargo / per-voyage:
 vessel_type, vessel_dwt)                         cargo_tons, weather_factor, sea_state)
        |                                               |
        +-----------------------+-----------------------+
                                |
                                v
               Hierarchical Specification Merging:
             spec = {**v_entry, **c_entry, **pair_entry}
                                |
        +-----------------------+-----------------------+
        |                                               |
        v                                               v
   ML Inference                                First-Principles Physics
  (VoyageRecord populated with                 (calculate_fuel_use invoked with
   exact vessel_type, dwt,                      exact cargo_tons, weather_factor,
   cargo_tons, weather, sea_state)              and vessel_dwt)
```

- **Dynamic Attributes Resolved:**
  - `vessel_type`: Populated dynamically (e.g., `Container Ship`, `Tanker`, `Gas Carrier`, `General Cargo`, `Bulk Carrier`) rather than assuming every vessel is a generic bulk carrier.
  - `vessel_dwt`: Extracted from vessel capacity / deadweight specifications rather than arbitrary fallback tonnages.
  - `cargo_tons`: Extracted directly from scheduled cargo consignment weights.
  - `weather_factor`: Captures route-specific environmental resistance multipliers ($w \ge 1.0$) rather than calm-sea neutral conditions ($1.0$).
  - `sea_state`: Evaluates Douglas sea state scale integers ($0 \le \text{sea\_state} \le 9$) reflecting actual wave action.
- **Hierarchical Precedence:** The objective evaluation merges context dictionaries with graceful fallback precedence:
  $$\text{pair\_entry (vessel-cargo pair)} \succ \text{c\_entry (cargo/route)} \succ \text{v\_entry (vessel)} \succ \text{physical fallback}$$
- **Fleet Benchmark Alignment:** In `generate_fleet_problem()`, generated problem instances deterministically populate diverse vessel categories (`vessel_type`, `vessel_dwt`) and cargo meteorological profiles (`weather_factor`, `sea_state`), allowing the optimizers to discover genuine operational distinctions between vessel assignments.

---

## 6. Cloud Deployment & Live Demo (Render)

The SIH26138 executive dashboard is configured for deployment on [Render](https://render.com) as a Python web service using the root [`render.yaml`](render.yaml) blueprint and build script [`scripts/build_for_deploy.sh`](scripts/build_for_deploy.sh).

### Automated Build Pipeline
Because git ignores generated runtime datasets, model binaries, and evaluation reports, Render automatically regenerates all required assets on each deploy via `scripts/build_for_deploy.sh`:
1. Synthesizes 10,000 voyage records (`scripts/make_mock_dataset.py`).
2. Trains all baseline models, tunes `QIFCP`, and exports metrics (`src/prediction/`).
3. Runs multi-fuel scenario simulations across 6 marine fuels (`scripts/run_scenario_comparison.py`).
4. Executes quantum vs classical swarm optimization scalability sweeps (`scripts/benchmark_optimization.py`).

### Deployment Steps on Render
1. Push this repository to GitHub or GitLab.
2. In the Render Dashboard, click **New +** -> **Blueprint**.
3. Connect your repository. Render detects `render.yaml` and provisions the web service automatically.
4. The service will execute `buildCommand` (pip install + artifact generation) and launch via `startCommand` on the dynamic `$PORT`.

### Cold-Start Mitigation & Judging Protocol
> [!IMPORTANT] Free-Tier Sleep & Cold-Start Protocol
> Free-tier Render services sleep after inactivity; the first request after idle takes 30-50s. Before a live demo, open the URL 5 minutes ahead of time to warm it up, and keep `streamlit run app.py` working locally as a fallback if the live link is slow to respond during judging.

To run the executive dashboard locally:
```bash
streamlit run app.py
```

---

## 7. Real-World Observational Data Integration & Canonical Target Formulation

### Heterogeneous Telemetry Sources
The system integrates observational maritime data alongside synthetic hydrodynamic simulations:
1. **EU THETIS-MRV:** Annual mandatory reporting data covering commercial cargo vessels, container ships, and tankers operating in European waters (macro-scale operational reporting).
2. **FuelCast:** High-frequency, sensor-level underway telemetry capturing second-by-second and hourly engine fuel rates and sea state conditions across commercial container and cargo vessels.

### Neutralizing Duration Proxy Leakage
Prior to mitigation, an adversarial 3-class classifier trained solely on `hours_at_sea` achieved **95.25% accuracy** in fingerprinting dataset source (`mock`, `thetis_mrv`, `fuelcast`). This occurred because high-frequency sensor streams operated at 1.0-hour intervals while annual MRV logs aggregated thousands of hours.

To eliminate this proxy leakage:
- **Uniform Leg Sampling:** Observational records are decomposed into standard commercial voyage legs ($D \in [300, 7500]\text{ nm}$), standardizing voyage duration distributions ($\mu \approx 280\text{ h}$).
- **Feature Pipeline Disconnection:** Raw `hours_at_sea` is completely removed from model feature matrix $X$ and replaced with `implied_hours = distance_nm / speed_knots` derived identically across all sources, augmented by one-hot `source_group` duration bins (`short` $<100\text{ h}$, `medium` $100\text{--}400\text{ h}$, `long` $>400\text{ h}$).
- **Leakage Invariant Test:** Verified via `tests/test_no_source_leakage.py`, single-feature classifier accuracy drops to **52.4%** on blended data and **37.3%** on balanced classes (near the 33.3% random baseline), strictly satisfying the required $\le 60\%$ safety threshold.

### Canonical Target Framing: Rate Formulation ($t/h$)
The platform designates **fuel burn rate** ($\text{metric tons per hour}$, $t/h$) as its canonical target framing:
$$\text{Target: } y_{\text{rate}} = \frac{\text{fuel\_consumption}}{\text{hours\_at\_sea}}$$
Absolute fuel consumption for any voyage is reconstructed via:
$$\hat{y}_{\text{abs}} = \hat{y}_{\text{rate}} \times \text{hours\_at\_sea}$$

**Why Rate Framing is the Scientifically Rigorous Choice:**
1. **Scale Invariance Across Temporal Resolutions:** Absolute fuel consumption scales directly with voyage duration ($R^2$ dominated by voyage length). Rate framing normalizes across 1-hour sensor intervals, multi-day coastal legs, and trans-oceanic voyages without introducing multi-order-of-magnitude variance.
2. **Alignment with Naval Propulsion Hydrodynamics:** The instantaneous rate of fuel burn reflects the ship's instantaneous power requirement ($P \propto \Delta^{2/3} V^3 / C_{\text{adm}}$) and engine specific fuel oil consumption (SFOC). Predicting rate directly forces the regressors to learn physical hull resistance and payload efficiencies rather than simple duration multiplication.
3. **Prevention of Source Dominance:** In absolute mode, large voyages dominate the mean squared loss gradient. Rate formulation balances the learning signal equally across long-haul and regional routes.

### Execution Modes
- **Blended Real Data Pipeline (Canonical):**
  ```bash
  python scripts/run_full_pipeline.py --use-real-data
  ```
  Blends 50/50 real and synthetic data, trains on leakage-free features, evaluates rate-formulated regressors, and updates `artifacts/metrics/baseline_metrics.json` and `outputs/reports/prediction_benchmark.json`.
- **Synthetic Fallback Mode:**
  ```bash
  python scripts/run_full_pipeline.py
  ```
  Runs exclusively on synthetic baseline voyages without external data dependencies.

---

## 8. Summary of Recent Architectural Hardening & Production Audits

This section provides an explicit audit trace of all recent methodology, compliance, optimization, and presentation hardenings implemented across the repository:

### 8.1 Machine Learning & Validation Rigor (P1 Methodology Hardening)
1. **Nested Vessel-Disjoint Validation for QIFCP:**
   - *Previous state:* QPSO hyperparameter search used ordinary positional array splitting (`X_train = X_arr[:split_idx]`).
   - *Hardened state:* `QIFCPRegressor.tune_with_qpso(X, y, groups=vessel_ids)` enforces strict vessel-disjoint `GroupShuffleSplit` across inner validation folds:
     $$\text{Outer Vessel-Disjoint Test Split} \longrightarrow \text{Training Partition} \longrightarrow \text{Inner Vessel-Disjoint Validation Fold} \longrightarrow \text{QPSO Parameter Search}$$
     Guarantees that no vessel appearing in the inner validation fold is present in the inner training partition, preventing data leakage during phase-scale ($\gamma$) and L2 regularization ($\alpha_{\text{reg}}$) search.
2. **Standardized Native GBDT Architecture (`HistGradientBoostingRegressor`):**
   - Implemented using Scikit-Learn's native histogram-based gradient-boosted decision tree regressor (`hist_gradient_boosting`), eliminating unpinned external C++ dynamic library dependencies (`libxgboost`, OpenMP runtime incompatibilities) across Windows, Linux, and cloud containers.
   - Belongs to the same broad class of histogram-based gradient-boosting tree methods as XGBoost and LightGBM (inspired by LightGBM's binning paradigm). `ModelType.XGBOOST` is preserved as a backward-compatible contract alias that resolves to `hist_gradient_boosting`.
3. **Empirical Metrics vs. Nominal Confidence Scores:**
   - The scalar field `confidence_score=0.95` is retained strictly as an immutable backward-compatible schema field.
   - User-facing dashboards, leadership leaderboards, and evaluation benchmarks strictly present empirical validation metrics ($R^2$, RMSE, MAE, MAPE, NRMSE, fit times, and cross-source generalization spreads) rather than synthetic scalar confidence percentages.

### 8.2 Green Fleet Optimization & Physics Dispatch Architecture
4. **System-Wide Dual-Route Fuel Architecture Enforcement:**
   - *Previous state:* `fleet_objective.py` routed all fuels (including Hydrogen, Ammonia, and ShorePower) through machine learning predictors, creating an inconsistency with `scenario_analysis.py`.
   - *Hardened state:* Both `fleet_objective.py` (scalar PSO/QPSO) and `nsga2_pareto.py` (bi-objective NSGA-II) strictly enforce the system-wide fuel architecture:
     - **Conventional / Transitional fuels (`Diesel`, `LNG`, `Methanol`):** Dispatched to ML models (`ProductionModelManager`) which possess empirical training distributions.
     - **Novel zero-emission pathways (`Hydrogen`, `Ammonia`, `ShorePower`):** Dispatched exclusively to first-principles naval architecture physics (`MaritimeFuelPhysicsEngine`). Statistical models are never called for unobserved fuel pathways.
5. **Dynamic Vessel and Weather Parameterization:**
   - *Previous state:* Optimization routines evaluated every voyage assuming a generic Bulk Carrier with neutral weather ($w=1.0$) and calm sea state ($3$).
   - *Hardened state:* Hierarchical context resolution (`voyage_specs` and `vessel_specs`) dynamically injects authentic vessel types (`vessel_type`), deadweight carrying capacities (`vessel_dwt`), cargo tonnages (`cargo_tons`), route weather resistance factors (`weather_factor`), and Douglas sea state scales (`sea_state`) into both ML inference and physical drag computations.
6. **Canonical Deb et al. (2002) Crowding-Distance Environmental Selection:**
   - *Previous state:* Truncation of overflowing Pareto fronts used naive array slicing (`front[:needed]`).
   - *Hardened state:* `calculate_crowding_distance()` assigns infinite distance ($d_i = \infty$) to boundary extrema and accumulates normalized objective spread across intermediate neighbors. The solver sorts the boundary front in descending crowding distance, preserving Pareto frontier spread and boundary solutions.
   - *Offspring Variation Nuance:* Paired with continuous differential-evolution variation (DE/rand/1/bin, $F=0.8, CR=0.7$), forming a hybrid **DE-NSGA-II / DEMO** architecture.
7. **Dimensional Currency Consistency (`DEFAULT_EUR_TO_USD_FX_RATE`):**
   - *Previous state:* FuelEU regulatory penalties (€EUR) were directly summed with bunker fuel costs ($USD) without unit conversion.
   - *Hardened state:* Introduced explicit configurable exchange rate (`DEFAULT_EUR_TO_USD_FX_RATE = 1.08`) across `fleet_objective.py`, `nsga2_pareto.py`, and `scenario_analysis.py`. Financial objectives cleanly convert penalties to USD before computing total operational cost.

### 8.3 Statutory Compliance Rigor & Scope Boundaries
8. **Strict Statutory Capacity Metric Validation (`DataValidationError`):**
   - Enforces IMO Resolution MEPC.353(78) Table 1 statutory capacity types. Supplying DWT for passenger/Ro-Ro vessels or GT for cargo/tanker/bulk carriers immediately raises `DataValidationError`.
9. **Strict Statutory Vessel Category Validation (`ComplianceError`):**
   - Replaced silent fallback to Bulk Carrier parameters with explicit `raise ComplianceError` for unmodeled vessel types in `resolve_cii_reference_line` and `resolve_cii_rating_boundaries`.
10. **Statutory Capacity Boundary Enforcements (MEPC.353(78) G2):**
    - LNG carriers below 65,000 DWT: Enforces statutory fixed effective capacity floor of $65{,}000\text{ DWT}$.
    - Ro-Ro vehicle carriers: Enforces statutory capacity cap of $57{,}700\text{ GT}$.
    - Bulk carriers: Enforces statutory capacity cap of $279{,}000\text{ DWT}$.
11. **Defensible Operational Estimator Scope Notices:**
    - IMO CII engine positioned strictly as an operational CII estimator (MEPC.353(78) G2, MEPC.354(78) G4, MEPC.400(83) Z-factors) without simulating optional MEPC.355(78) G5 voyage correction factors.
    - FuelEU Maritime engine positioned strictly as a GHG-intensity compliance and penalty estimator (Articles 4 & 23, Annex IV, with Article 23(2) consecutive-deficit penalty scaling) without simulating RFNBO subtargets, OPS mandates, or pooling/banking mechanisms.

### 8.4 Cold-Ironing Electrification & Scenario Transparency
12. **Native Electrical Energy Reporting (`energy_consumption_mwh`):**
    - `ScenarioResult` explicitly reports `energy_consumption_mwh` alongside `fuel_cost_usd`, `fueleu_penalty_eur`, `fueleu_penalty_usd`, and `exchange_rate_eur_to_usd`.
    - Eliminates the misleading representation of "zero consumption" when cold-ironing shore power replaces bunker fuel, while enabling thermodynamic energy parity comparisons across all 6 alternative marine fuels.
13. **Per-Vessel Electrical Energy Storage in Multi-Vessel Fleets:**
    - *Previous state:* `ScenarioAnalysisEngine` calculated per-vessel physics but only saved `fuel_per_vessel`, and subsequently sampled `self.physics_engine.last_energy_mwh` in a second loop. In heterogeneous fleets with differing speeds, distances, payloads, and DWTs, this erroneously overwrote earlier vessels with the final vessel's energy ($E_{\text{total}} = N \times E_N$).
    - *Hardened state:* `ScenarioAnalysisEngine` immediately captures and stores each vessel's specific `(fuel_tons, energy_mwh)` tuple upon physics calculation (`vessel_consumptions`), guaranteeing exact linear energy aggregation ($\sum_{i=1}^N E_i$) regardless of vessel heterogeneity.
14. **Dimensional Separation of Bunker Fuel & Shore Power Electricity Pricing:**
    - *Previous state:* Baseline prices were housed in a monolithic dictionary `STANDARD_FUEL_PRICES_USD` described as "USD per metric ton", and ShorePower calculations multiplied `energy_mwh * price_per_ton` with inline comments noting that the value acted as USD/MWh.
    - *Hardened state:* The pricing contract explicitly separates commercial bunker fuels from electrical tariffs:
      - `FUEL_PRICES_USD_PER_TON`: Dedicated dictionary for chemical fuels (`Diesel`, `LNG`, `Methanol`, `Hydrogen`, `Ammonia`) strictly denominated in $\$ / \text{metric ton}$.
      - `SHORE_POWER_PRICE_USD_PER_MWH`: Dedicated constant ($300.0\text{ USD/MWh}$) strictly denominated in $\$ / \text{MWh}$.
      - Optimization objectives (`fleet_objective`, `nsga2_pareto`) and scenario simulations evaluate $\text{Cost}_{\text{electric}} = E_{\text{mwh}} \times \text{Tariff}_{\text{USD/MWh}}$ and $\text{Cost}_{\text{fuel}} = \text{FuelMass}_{\text{tons}} \times \text{Price}_{\text{USD/ton}}$, eliminating dimensional and variable-naming ambiguity.




