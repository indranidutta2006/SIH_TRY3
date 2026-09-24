# Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization

**Problem ID:** SIH26138  
**Phase:** Phase 0 (Architecture Contract & Project Foundation)  
**Target Runtime:** Python 3.12+

---

## 1. Project Overview

The **Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization** system is an enterprise-grade platform designed to assist commercial vessel operators, charterers, and maritime authorities in decarbonizing maritime logistics.

The objective of the platform is twofold:
1. **Accurately Predict Marine Fuel Consumption:** Capture nonlinear hydrodynamic resistance, adverse weather dynamics, and cargo payload constraints using classical regression (Linear Regression, Random Forest, HistGradientBoosting) and Quantum-Inspired Fuel Consumption Prediction (`QIFCP`) algorithms.
2. **Optimize Green Fleet Operations:** Allocate vessels to cargo orders and schedule routes to minimize total Well-to-Wake (WtW) greenhouse gas emissions, fuel expenditures, and operational delays while guaranteeing full compliance with the International Maritime Organization (IMO) Carbon Intensity Indicator (CII) ratings and European Union FuelEU Maritime standards.

This repository contains the **Phase 0 architectural foundation**. It explicitly defines the schemas, abstract contracts, centralized configurations, exception hierarchies, and logging facilities required for subsequent development phases.

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
| **Data Layer** | `app/prediction/`, `data/` | Ingestion, sanitization, schema validation, and storage of historical voyage records. |
| **Prediction Layer** | `app/prediction/` | Model training, inference, and benchmarking across Linear Regression, Random Forest, HistGradientBoosting, and QIFCP models. |
| **Fuel Physics Layer** | `app/physics/` | Hydrodynamic drag equations, admiralty coefficients, and propulsion energy conversion. |
| **Emissions Layer** | `app/emissions/` | Tank-to-Wake (combustion) and Well-to-Wake (lifecycle) GHG quantification for diverse marine fuels. |
| **Compliance Layer** | `app/compliance/` | IMO CII rating computation (grades A to E) and EU FuelEU Maritime penalty evaluation. |
| **Scheduler Layer** | `app/scheduler/` | Timetable generation, laycan matching, port draft constraints, and cargo allocation. |
| **Optimization Layer**| `app/optimization/` | Heuristic and quantum-inspired search (PSO, QPSO, NSGA-II) for pareto-optimal trade-offs. |
| **Dashboard Layer** | `app/dashboard/` | Interactive visualization, tradeoff frontier plots, and what-if scenario simulators. |

---

## Deliverables

| # | Deliverable | Status | Output Path |
|---|-------------|--------|-------------|
| D1 | Mathematical Model | ✅ Complete | `contracts/schemas.py`, `src/physics/` |
| D2 | Fuel Consumption Prediction Module (QIFCP & QKP) | ✅ Complete | `src/prediction/qifcp.py`, `models/fuel_predictor.py` |
| D3 | Quantum Metaheuristic Optimizer (QPSO) | ✅ Complete | `src/optimization/qpso.py` |
| D4 | Alternative Fuel Scenario Analyser | ✅ Complete | `src/optimization/scenario_analysis.py` |
| D5 | Multi-Objective Optimization (NSGA-II Pareto) | ✅ Complete | `src/optimization/nsga2_pareto.py` |
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
    DataValidationError,
    PredictionError,
    OptimizationError,
)
```

### Canonical Compliance Contract (`ComplianceResult`)

The compliance subsystem evaluates fleet voyages against statutory **IMO Carbon Intensity Indicator (CII)** ratings and **EU FuelEU Maritime** greenhouse gas intensity caps. The canonical dataclass contract is:

| Field | Type | Description |
|:---|:---:|:---|
| `cii_rating` | `str` | Operational letter rating (`A` through `E` for CII, or `N/A` for FuelEU). |
| `attained_cii` | `float` | Attained annual operational CII in $\text{gCO}_2 / (\text{DWT} \cdot \text{nm})$. |
| `required_cii` | `float` | Target required CII under IMO MEPC.337(76) & MEPC.400(83). |
| `cii_ratio` | `float` | Attained-to-Required CII ratio ($< 1.0$ indicates outperforming statutory target). |
| `fueleu_pass` | `bool` | Statutory pass/fail against EU FuelEU Maritime limit. |
| `fueleu_target` | `float` | Maximum permitted Well-to-Wake GHG intensity ($\text{gCO}_2\text{eq/MJ}$) for assessment year. |
| `ghg_intensity` | `float` | Attained Well-to-Wake GHG intensity ($\text{gCO}_2\text{eq/MJ}$). |
| `penalty_eur` | `float` | Statutory financial penalty in Euros (€) under Regulation (EU) 2023/1805 Article 23. |
| `compliance_status` | `str` | Standardized status string (`COMPLIANT` or `NON_COMPLIANT`). |
| `compliance_score` | `float` | *(Deprecated)* Legacy compatibility field. Use `cii_ratio` or `penalty_eur`. |

> [!WARNING] Deprecation Notice: `compliance_score`
> The field `compliance_score` is deprecated and preserved strictly for backwards compatibility with legacy tests. It previously functioned as a polymorphic alias (`cii_ratio` for CII, `penalty_eur` for FuelEU). All downstream optimization engines (`FleetObjective`, `nsga2_pareto`, `ScenarioAnalysis`) and dashboards consume the explicit canonical fields `penalty_eur` and `cii_ratio`.

---

## 4. Quick Start

A judge or evaluator can reproduce the entire platform from a fresh `git clone` to the running interactive dashboard by following these five sequential steps:

### Step 1: Environment Setup & Installation
Clone the repository and install required production dependencies within a clean virtual environment:
```bash
git clone https://github.com/indranidutta2006/SIH26138.git
cd SIH26138
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
* **Produces:** 183/183 passing deterministic tests (**0 failures, 0 errors**) confirming strict contract adherence, proxy-leakage neutralization, MEPC.400(83) compliance targets, and algorithm correctness.

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
  4. **Statutory Compliance (IMO/EU):** Automated vessel-level IMO CII ratings (A–E) and FuelEU Maritime penalty audits.
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

### 5.2 IMO Carbon Intensity Indicator (CII) & Resolution MEPC.400(83)
Implemented in `src/compliance/compliance_engine.py` conforming to [`contracts.interfaces.ComplianceEngine`](contracts/interfaces.py):
- **Attained CII:** Calculated per voyage or annually as $\text{CII}_{\text{attained}} = \frac{\text{CO}_2 \times 10^6}{\text{DWT} \times \text{Distance}}$.
- **Baseline Reference Curve:** $\text{CII}_{\text{ref}} = a \cdot \text{DWT}^{-c}$ with coefficients defined under IMO Resolution MEPC.337(76) across Bulk Carriers ($a=4745, c=0.622$), Tankers ($a=5247, c=0.610$), Containers ($a=1984, c=0.489$), General Cargo ($a=3196, c=0.540$), and RoRo ($a=1686, c=0.388$).
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

- **Statutory Lookup & Regulatory Range Enforcement:** Implemented via immutable lookup table `CII_Z_FACTORS`. Any compliance evaluation for calendar years outside the statutory regulatory range ($[2023, 2030]$) strictly raises `ComplianceError`, preventing erroneous extrapolations or arbitrary linear drift.
- **Letter Rating Bands (A–E):** Based on ratio $r = \text{CII}_{\text{attained}} / \text{CII}_{\text{required}}$:
  $r \le 0.83 \implies \text{A}$, $r \le 0.94 \implies \text{B}$, $r \le 1.06 \implies \text{C}$, $r \le 1.19 \implies \text{D}$, else $\text{E}$.

### 5.3 EU FuelEU Maritime Statutory Penalties (Regulation (EU) 2023/1805)
Implemented in `src/compliance/compliance_engine.py` following Article 23 & Annex IV:
$$\text{Penalty (EUR)} = \frac{|\text{Compliance Balance (gCO}_2\text{eq)}|}{\text{GHGIE}_{\text{actual}} \times 41000\text{ MJ/t}} \times 2400\text{ EUR/t}$$
where:
- $\text{Compliance Balance} = (\text{GHGIE}_{\text{target}} - \text{GHGIE}_{\text{actual}}) \times \text{Energy Consumed (MJ)}$
- Reference baseline: $91.16\text{ gCO}_2\text{eq/MJ}$ with phased reductions ($2025 = -2\%, 2030 = -6\%, 2035 = -14.5\%, 2040 = -31\%, 2045 = -62\%, 2050 = -80\%$).
- Downstream optimization routines directly consume canonical `comp.penalty_eur`.

### 5.4 Quantum-Inspired & Quantum Kernel Modeling
- **QIFCP (`src/prediction/qifcp.py`):** Quantum-inspired hydrodynamic neural regressor tuned with Quantum-behaved Particle Swarm Optimization (`QPSO`).
- **QuantumKernelPredictor (`models/fuel_predictor.py`):** PennyLane simulation (`default.qubit`) utilizing `AngleEmbedding` on 4 normalized hydrodynamic features (`speed_kn`, `load_factor`, `wave_ht_m`, `wind_bft`) combined with a 2-repetition `ZZFeatureMap` entangling circuit. The resulting quantum state transition kernel matrix is fed into `KernelRidge(alpha=0.1)`, achieving $R^2 > 0.88$ on cross-class synthetic voyages.
  > *Quantum principle used:* Interference between feature-encoded states produces a similarity metric unavailable to classical RBF kernels.

> [!NOTE] Alternative Fuel Baseline Assumptions
> In this baseline implementation, alternative power pathways (**Hydrogen**, **Ammonia**, and **ShorePower**) reflect certified green renewable supply chains (e.g., green hydrogen from water electrolysis powered by renewables, green ammonia synthesized with zero-carbon energy, and zero-emission shore grid connections).
> 
> In industrial operations:
> - *Green hydrogen $\ne$ all hydrogen* (fossil steam methane reforming entails significant upstream WTT emissions).
> - *Green ammonia $\ne$ all ammonia* (Haber-Bosch with fossil feedstock has high carbon intensity).
> - *Grid shore power $\ne$ zero lifecycle emissions* (grid carbon intensity varies widely by regional port grid mix).
> 
> The architecture explicitly provides configurable `wtt_factors` in `MaritimeEmissionEngine`, allowing operators to model grey, blue, or green fuel pathways dynamically.

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

