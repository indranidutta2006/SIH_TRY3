# SIH26138: Architecture Baseline & System Dataflow Specification

**Problem ID:** SIH26138 — Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization  
**Repository:** `SIH_TRY3`  
**Branch:** `main`  
**Baseline Git Commit:** `fe6c358fd476ec8af2e27d9746f4f6eb5de40903`  
**Audit Date:** 2026-09-26  
**Document Status:** Architecture Frozen (Phase 0 Baseline)

---

## 1. Executive Summary & Verification Environment

This document establishes the verified architectural baseline for the SIH26138 maritime software platform. All subsequent feature expansions, algorithmic enhancements, and quantum swarm tuning must preserve backwards-compatibility with this baseline and measure improvements against the empirical runtime metrics recorded herein.

### Hardware & Environment Specifications
- **Python Version:** `3.12.10`
- **Operating System:** Windows 11 Enterprise (Build 10.0.26200)
- **Processor:** Intel64 Family 6 Model 186 Stepping 3, GenuineIntel (5 Physical Cores, 6 Logical Cores)
- **Installed System RAM:** 7.69 GB
- **Primary Package Root:** `src/`
- **Canonical Contract Root:** `contracts/`
- **Current Test Suite:** 329 collected, 329 passing, 0 failing, 0 skipped (Runtime: 29.40s)

---

## 2. Canonical Contracts & Schema Integrity

The `contracts/` package is the **sole canonical source of truth** for all schemas, interfaces, physical constants, custom exceptions, and versioning. Root-level compatibility modules (`schemas.py`, `constants.py`, `exceptions.py`, `interfaces.py`) are pure re-export shims that point directly to `contracts/`.

### Class Breakdown in `contracts.schemas`
An automated structural audit verifies exactly **34 classes** declared in `contracts/schemas.py`:
- **32 Strongly-Typed Dataclasses** (all enforced with `frozen=True, slots=True`):
  - Telemetry & Physics: `VoyageRecord`, `PredictionResult`, `EmissionResult`
  - Statutory Compliance: `CIIResult`, `FuelEUResult`, `ComplianceAssessment`, `ComplianceResult`
  - Operations & Reliability: `DemandSatisfactionMetrics`, `ReliabilityMetrics`, `FleetAssignment`
  - Optimization Results: `OptimizationResult`, `FleetCompositionResult`, `CapacityOptimizationResult`, `SpeedOptimizationResult`, `FleetStrategyRecommendation`, `OptimizationScenario`
  - Benchmarking & Validation: `BenchmarkResult`, `BenchmarkSuiteResult`, `PredictionBenchmarkResult`, `ConvergenceAnalysisResult`, `ParetoResult`, `StatisticalStabilityResult`, `WorkflowBenchmarkResult`
  - Lifecycle & Energy: `FuelLifecycleProfile`, `LifecycleAssessmentResult`
  - Decarbonization Roadmap: `TransitionMilestone`, `TransitionRoadmap`
  - Strategic Scenarios: `RegulatoryForecastResult`, `ScenarioDefinition`, `ScenarioComparisonResult`, `ScenarioResult`
  - Executive Layer: `ExecutiveRecommendation`
- **2 Enums** (`StrEnum` / `Enum`):
  - `EvidenceCategory`
  - `OptimizationStatus`
- **Serialization Uniformity:** 100% of the 32 dataclasses implement `to_dict()` and `to_json()`, guaranteeing JSON-safe serialization across disk storage, API boundaries, and dashboard state.

### Root Shims & Object Identity
Object identity (`obj1 is obj2`) is strictly maintained across all import conventions:
```python
from contracts.schemas import OptimizationScenario
from schemas import OptimizationScenario
assert contracts.schemas.OptimizationScenario is schemas.OptimizationScenario
```
Root-level package bridges (`optimization`, `operations`, `benchmarking`) mirror their source packages under `src/` with zero class duplication.

---

## 3. End-to-End System Dataflow

The platform implements an 11-stage pipeline linking real-world telemetry to C-suite executive decision intelligence:

```
[1. Data Ingestion] 
       │ (VoyageRecord: THETIS, MRV, FuelCast)
       ▼
[2. Fuel Prediction] 
       │ (QIFCP / HistGBDT: Distance, Speed, Cargo, Weather, Sea State)
       ▼
[3. Hydrodynamic Physics] 
       │ (Admiralty Coefficient, Weather Resistance, Cubic Displacement)
       ▼
[4. GHG Emissions Engine] 
       │ (WTW: WTT upstream + TTW combustion for CO2, CH4, N2O, CO2e)
       ▼
[5. Fleet Strategy Optimization] 
       │ (Dual Engine: Combinatorial MIP or 8-D QPSO Swarm)
       ▼
[6. Operational Reliability] 
       │ (Schedule Adherence, Port Delays, Buffer Margins, Service Level)
       ▼
[7. Solver Benchmarking] 
       │ (Apples-to-apples: QPSO, Classical PSO, GA, SA, LP, Greedy)
       ▼
[8. Lifecycle Assessment (LCA)] 
       │ (Well-to-Wake Feedstock: e-fuels, bio-pathways, fossil baselines)
       ▼
[9. Regulatory Forecasting] 
       │ (Statutory IMO CII 2026-2030 + Projected Scenario CII 2031-2040)
       ▼
[10. Executive Decision Support] 
       │ (Signed ROI Waterfalls, TCO, Carbon Exposure, Transition Roadmap)
       ▼
[11. Interactive Dashboard] 
         (Streamlit 9-Page Executive Command Center)
```

### Pipeline Description:
1. **Data Ingestion (`src.ingestion`):** Ingests raw telemetry from IMO DCS, EU THETIS, and simulated synthetic datasets into strongly-typed `VoyageRecord` instances with strict schema validation and target-leakage guardrails.
2. **Prediction Engine (`src.prediction`):** Deploys the production `hist_gradient_boosting` regressor (or Quantum-Inspired `QIFCP`) to estimate baseline voyage fuel consumption.
3. **Hydrodynamic Physics (`src.physics`):** Computes vessel power and fuel demand using naval architecture equations (Admiralty coefficient law, cargo loading ratios, and weather drag penalties).
4. **Emissions Engine (`src.prediction.emission_engine`):** Evaluates Well-to-Wake (WtW) lifecycle greenhouse gas emissions including methane slip ($CH_4$) and nitrous oxide ($N_2O$) according to IMO Resolution MEPC.376(80).
5. **Fleet Strategy Optimization (`src.optimization`):** Slices fleet configuration across discrete vessel classes (Feeder, Panamax, Capesize) and 5 fuels (Diesel, LNG, Methanol, Hydrogen, Ammonia). Offers dual-engine architecture:
   - `deterministic`: Exact combinatorial integer MIP with branch-and-bound lower-bound pruning.
   - `qpso`: 8-dimensional continuous particle swarm with quantum delta-potential-well updates and largest-remainder integer projection.
6. **Operational Reliability (`src.operations`):** Assesses schedule reliability scores (0–100) using 1-to-1 observation-to-voyage cardinality, penalizing weather delays and port congestion against deadline constraints.
7. **Solver Benchmarking (`src.benchmarking`):** Unified multi-solver benchmark suite evaluating 6 distinct optimization algorithms on standardized operational scenarios.
8. **Lifecycle Assessment (`src.lifecycle`):** Evaluates upstream Well-to-Tank (WTT) feedstock pathways (e.g. bio-methanol, e-diesel, green hydrogen) versus fossil baselines.
9. **Regulatory Forecasting (`src.strategy` / `src.compliance`):** Evaluates statutory compliance (IMO CII & FuelEU Maritime) through 2030 and projected scenario trajectories (2031–2040).
10. **Executive Decision Support (`src.decision_support`):** Compiles signed economic deltas (unclamped fuel, carbon, and penalty savings), net investment capex, and multi-year transition roadmaps.
11. **Dashboard Interface (`app/`):** Streamlit analytical dashboard presenting interactive views for engineers and maritime executives.

---

## 4. Frozen Runtime Performance Baseline

The baseline runtime figures below represent actual measured empirical values on the reference environment under standardized operational conditions.

| Component / Subsystem | Benchmark Metric | Measured Baseline | Measurement Conditions |
| :--- | :--- | :--- | :--- |
| **Core Prediction Engine** | Single-voyage inference latency | **20.19 ms** | Production `hist_gradient_boosting` artifact, batch=1, $N=100$ repetitions, 1 warm-up call |
| **Deterministic Fleet Optimizer** | End-to-end strategy solve | **10.74 ms** | Scenario: `SCEN-BASE-BENCHMARK` ($D=250\text{k t}, B=\$120\text{M}$), exact combinatorial MIP |
| **QPSO Fleet Optimizer** | Continuous swarm solve | **78.13 ms** | Scenario: `SCEN-BASE-BENCHMARK`, 8 dimensions, Population=25, Iterations=40, $\alpha \in [1.0, 0.5]$ |
| **Benchmark Suite** | 6-solver comparative suite | **0.3708 s** | 6 solvers (QPSO, PSO, GA, SA, LP, Greedy), 50 iterations each, seed=42 |
| **Streamlit Startup** | Headless module import latency | **2.6094 s** | Cold import of `streamlit` and `app.dashboard` submodules |

---

## 5. QPSO 8-Dimensional Formulation

The Quantum-Behaved Particle Swarm Optimization (`QPSO`) implementation maps discrete fleet decisions into an 8-dimensional bounded continuous space $\mathbb{R}^8$:

| Dimension Index | Variable Name | Physical Meaning | Search Bounds |
| :---: | :--- | :--- | :---: |
| `vec[0]` | $x_{\text{feeder}}$ | Continuous count of Feeder class vessels | $[0.0, \min(20, N_{\max})]$ |
| `vec[1]` | $x_{\text{medium}}$ | Continuous count of Medium/Panamax vessels | $[0.0, \min(15, N_{\max})]$ |
| `vec[2]` | $x_{\text{large}}$ | Continuous count of Large/Capesize vessels | $[0.0, \min(10, N_{\max})]$ |
| `vec[3]` | $n_{\text{alt}}$ | Continuous count of alternative green fuel vessels | $[0.0, N_{\max}]$ |
| `vec[4]` | $w_{\text{lng}}$ | Relative allocation preference for LNG | $[0.0, 1.0]$ |
| `vec[5]` | $w_{\text{methanol}}$ | Relative allocation preference for Methanol | $[0.0, 1.0]$ |
| `vec[6]` | $w_{\text{hydrogen}}$ | Relative allocation preference for Hydrogen | $[0.0, 1.0]$ |
| `vec[7]` | $w_{\text{ammonia}}$ | Relative allocation preference for Ammonia | $[0.0, 1.0]$ |

### Quantum Position Update Law:
$$p_{i,d} = \phi \cdot pbest_{i,d} + (1 - \phi) \cdot gbest_d, \quad \phi \sim U(0, 1)$$
$$mbest_d = \frac{1}{M} \sum_{i=1}^M pbest_{i,d}$$
$$X_{i,d}(t+1) = p_{i,d} \pm \alpha(t) \cdot |mbest_d - X_{i,d}(t)| \cdot \ln(1 / u), \quad u \sim U(0, 1)$$
$$\alpha(t) = \alpha_{\text{start}} - \frac{t}{T_{\max}} (\alpha_{\text{start}} - \alpha_{\text{end}}), \quad \alpha_{\text{start}}=1.0, \, \alpha_{\text{end}}=0.5$$

---

## 6. Legacy Scaffolding Directory Status

The following directories in `app/` contain minimal package stub `__init__.py` files from early project scaffolding:
- `app/compliance/`
- `app/emissions/`
- `app/optimization/`
- `app/physics/`
- `app/prediction/`
- `app/scheduler/`

**Architectural Status:** **`LEGACY / COMPATIBILITY SCAFFOLD`**  
These stubs are retained to prevent breaking any legacy import paths or third-party entrypoints, but are strictly passive. Active implementations reside exclusively in `src/` and active UI views reside in `app/dashboard/`.

---

## 7. Phase 0 Acceptance Criteria Checklist

| Criterion | Requirement | Verification Method | Status |
| :--- | :--- | :--- | :---: |
| **1. Commit SHA Recorded** | Current commit tied to baseline | Recorded in JSON and MD (`fe6c358`) | **VERIFIED** |
| **2. Contract Schema Counts** | Exact 34 classes (32 dataclasses, 2 enums) | Automated Python AST audit | **VERIFIED** |
| **3. Milestone Serialization** | Complete `to_json` / `from_dict` / `from_json` | Unit test in `test_import_consistency.py` | **VERIFIED** |
| **4. Contract & Package Identity** | Root shims resolve to identical `src` objects | 8 identity tests in `test_import_consistency.py` | **VERIFIED** |
| **5. Full Test Suite Clean** | 100% pass across all tests with 0 regressions | `python -m pytest -q --tb=short` (329 tests) | **VERIFIED** |
| **6. Empirical Runtime Baseline** | Measured under documented conditions | `scratch/measure_baseline.py` recorded | **VERIFIED** |
| **7. QPSO Dimension Freeze** | Explicit 8-D definition and hyperparameters | Documented in Section 5 | **VERIFIED** |
| **8. Production Code Untouched** | Zero refactoring of working logic in Phase 0 | Strict freeze enforced | **VERIFIED** |
| **9. Legacy Stubs Documented** | Labeled as `LEGACY / COMPATIBILITY SCAFFOLD` | Explicitly marked in Section 6 | **VERIFIED** |
| **10. Machine-Readable Baseline** | `docs/architecture_baseline.json` created | Validated JSON structure | **VERIFIED** |
