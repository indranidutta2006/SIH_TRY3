# Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization

**Problem ID:** SIH26138  
**Phase:** Phase 0 (Architecture Contract & Project Foundation)  
**Target Runtime:** Python 3.12+

---

## 1. Project Overview

The **Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization** system is an enterprise-grade platform designed to assist commercial vessel operators, charterers, and maritime authorities in decarbonizing maritime logistics.

The objective of the platform is twofold:
1. **Accurately Predict Marine Fuel Consumption:** Capture nonlinear hydrodynamic resistance, adverse weather dynamics, and cargo payload constraints using classical regression (Linear Regression, XGBoost) and Quantum-Inspired Fuel Consumption Prediction (`QIFCP`) algorithms.
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
| **Prediction Layer** | `app/prediction/` | Model training, inference, and benchmarking across Linear, XGBoost, and QIFCP models. |
| **Fuel Physics Layer** | `app/physics/` | Hydrodynamic drag equations, admiralty coefficients, and propulsion energy conversion. |
| **Emissions Layer** | `app/emissions/` | Tank-to-Wake (combustion) and Well-to-Wake (lifecycle) GHG quantification for diverse marine fuels. |
| **Compliance Layer** | `app/compliance/` | IMO CII rating computation (grades A to E) and EU FuelEU Maritime penalty evaluation. |
| **Scheduler Layer** | `app/scheduler/` | Timetable generation, laycan matching, port draft constraints, and cargo allocation. |
| **Optimization Layer**| `app/optimization/` | Heuristic and quantum-inspired search (PSO, QPSO, NSGA-II) for pareto-optimal trade-offs. |
| **Dashboard Layer** | `app/dashboard/` | Interactive visualization, tradeoff frontier plots, and what-if scenario simulators. |

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

---

## 4. Setup and Verification

### Prerequisites
- Python 3.12 or higher
- Git

### Installation

1. Clone or navigate to the repository root:
   ```bash
   cd c:/HACKATHONS/SIH_TRY3
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running Phase 0 Verification

Execute the standalone health check script:
```bash
python run_demo.py
```

Expected output:
```text
YYYY-MM-DD HH:MM:SS | INFO     | maritime_system:configure_logging:... - Logging initialized at level 'INFO'. Output file: .../outputs/logs/phase0_verification.log
YYYY-MM-DD HH:MM:SS | INFO     | maritime_system:main:... - Initializing Phase 0 Repository Health Verification
YYYY-MM-DD HH:MM:SS | INFO     | maritime_system:main:... - All architectural interfaces and schemas successfully verified.
System initialized successfully
```

### Running Unit Tests

Verify all contract invariants via pytest:
```bash
pytest -v
```

---

## 5. Lifecycle Emissions & Alternative Fuel Modeling Boundaries

The emissions subsystem (`src/prediction/emission_engine.py`) implements full compliance with the [`contracts.interfaces.EmissionEngine`](contracts/interfaces.py) contract:
- **Tank-to-Wake (TTW):** Direct operational combustion emissions ($\text{CO}_2$, $\text{CH}_4$, $\text{N}_2\text{O}$, and $\text{CO}_2\text{e}$) calculated using statutory IMO MEPC factors and IPCC AR5 100-year Global Warming Potentials ($\text{GWP}_{\text{CH}_4} = 28.0, \text{GWP}_{\text{N}_2\text{O}} = 265.0$).
- **Well-to-Wake (WTW):** Total lifecycle climate footprint ($\text{WTW} = \text{TTW} + \text{WTT}$), capturing upstream fuel production, refining, liquefaction, transport, and bunkering. Direct combustion $\text{CO}_2$ is maintained separately from $\text{CO}_2\text{e}$ to avoid mixing physical combustion units with overall lifecycle climate impact.

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

