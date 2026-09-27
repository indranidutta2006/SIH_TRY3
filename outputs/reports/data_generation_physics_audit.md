# SIH26138: Dataset Generation, Target Formula & Physics Baseline Audit Report

**Phase:** Phase 2D Scientific Audit & Target Generation Validation  
**Date:** 2026-09-27  
**Canonical Dataset:** `data/raw/voyages_sample.csv`  
**Dataset SHA-256:** `069dfc97f9c3b75ad5e903a772095151692aa6501d07c2c648175cfe27ddcd2c`  
**Circularity Classification:** **LEVEL 3 (Target generated primarily from same physical equations as baseline)**  

---

## 1. Dataset Identity & Provenance

- **File Path:** `data/raw/voyages_sample.csv`
- **Record Count:** 500 rows
- **Vessel Pool:** 60 unique commercial vessels across 4 classes (`VSL-BC-*`, `VSL-CS-*`, `VSL-OT-*`, `VSL-GC-*`)
- **Categorical Columns:**
  - `vessel_type`: Bulk Carrier (31.0%), Container Ship (25.8%), Oil Tanker (24.8%), General Cargo (18.4%)
  - `fuel_type`: Diesel (69.2%), LNG (21.6%), Methanol (9.2%)
  - `data_source`: 100% `mock`
  - `is_synthetic`: 100% `True`

---

## 2. Generator Source & Execution Trace

The authoritative generator responsible for creating `data/raw/voyages_sample.csv` is:
- **File:** `scripts/make_mock_dataset.py`
- **Entrypoints:** `generate_synthetic_voyages(num_rows=500, seed=42)` and `calculate_physics_fuel()`
- **Pipeline Invocation:** `scripts/ci_generate_fixtures.py` (Stage 1) calls `generate_synthetic_voyages` and serializes records via `write_csv(records, data_file)`.
- **Random Seeds:**
  - `py_rng = random.Random(42)` (fleet sampling, vessel selection, port buffer hours)
  - `np_rng = np.random.default_rng(42)` (speed variance, voyage distance, sea state, stochastic factor)

---

## 3. Authoritative Target-Generation Formula

In `scripts/make_mock_dataset.py`, the target `fuel_consumption` is calculated via `calculate_physics_fuel()`:

$$F_{\text{target}} = \text{round}\left( \text{nominal\_fuel\_tons} \times \text{stochastic\_factor}, 2 \right)$$

where:
1. **Total Hydrodynamic Displacement ($\Delta$):**
   $$\Delta = (0.20 \times \text{vessel\_dwt}) + \text{cargo\_tons} \quad [\text{metric tons}]$$
2. **Admiralty Hull Coefficient ($C_{\text{adm}}$):**
   Evaluated per vessel type as the midpoint of vessel design bounds:
   - Bulk Carrier: $(480 + 560)/2 = 520.0$
   - Container Ship: $(550 + 640)/2 = 595.0$
   - Oil Tanker: $(460 + 540)/2 = 500.0$
   - General Cargo: $(420 + 500)/2 = 460.0$
3. **Propulsion Power ($P_{\text{prop}}$):**
   $$P_{\text{prop}} = \frac{\Delta^{2/3} \cdot V^3}{C_{\text{adm}}(\text{vessel\_type})} \quad [\text{kW}]$$
4. **Auxiliary and Hotel Service Load ($P_{\text{aux}}$):**
   $$P_{\text{aux}} = 0.05 \cdot (\text{vessel\_dwt})^{0.6} \cdot 100.0 \quad [\text{kW}]$$
5. **Total Power Demand ($P_{\text{total}}$):**
   $$P_{\text{total}} = P_{\text{prop}} + P_{\text{aux}} \quad [\text{kW}]$$
6. **Specific Fuel Oil Consumption ($\text{SFOC}$):**
   Lookup table based on fuel type:
   - Diesel: $175.0 \text{ g/kWh}$
   - LNG: $145.0 \text{ g/kWh}$
   - Methanol: $345.0 \text{ g/kWh}$
7. **Environmental Multiplier ($E_{\text{env}}$):**
   $$E_{\text{env}} = \text{weather\_factor} \times (1.0 + 0.025 \times \text{sea\_state})$$
8. **Voyage Duration ($H$):**
   $$H = \text{round}\left( \frac{\text{distance\_nm}}{\text{speed\_knots}} + U(0.5, 2.5), 2 \right) \quad [\text{hours}]$$
9. **Nominal Fuel Consumption:**
   $$\text{nominal\_fuel\_tons} = \left( \frac{P_{\text{total}} \times \text{SFOC} \times H}{1,000,000.0} \right) \times E_{\text{env}}$$
10. **Stochastic Perturbation:**
    $$\text{stochastic\_factor} \sim \mathcal{N}(\mu=1.0, \sigma=0.02)$$

---

## 4. Phase 2D Naval Physics Baseline Formula

In `src/prediction/qifcp.py`, `NavalPhysicsFuelBaseline` models:

$$y_{\text{phys}} = c_{\text{prop}} \cdot e_{\text{prop}}(X) + c_{\text{aux}} \cdot e_{\text{aux}}(X)$$

where:
1. **Propulsion Theoretical Fuel:**
   $$e_{\text{prop}} = \frac{\Delta^{2/3} \cdot V^3}{500.0} \times \frac{3600}{\eta_{\text{th}} \cdot \text{LHV}(\text{fuel})} \times \left( \frac{D}{V} \right) \times \max(W, 1.0) \times \frac{1}{1000}$$
2. **Auxiliary Theoretical Fuel:**
   $$e_{\text{aux}} = 0.05 \cdot (\text{DWT})^{0.6} \cdot 100.0 \times \frac{3600}{\eta_{\text{th}} \cdot \text{LHV}(\text{fuel})} \times \left( \frac{D}{V} \right) \times \max(W, 1.0) \times \frac{1}{1000}$$
3. **Calibration:**
   $c_{\text{prop}}$ and $c_{\text{aux}}$ are estimated strictly on $X_{\text{train}}, y_{\text{train}}$ via ordinary least squares without an intercept. Mean fitted values across the 5 outer folds:
   - $c_{\text{prop}} = 0.8194 \pm 0.0088$
   - $c_{\text{aux}} = 1.2771 \pm 0.0274$
   - Effective Admiralty $C_{\text{adm, eff}} = 500 / c_{\text{prop}} \approx 610.2$

---

## 5. Generator vs Physics Baseline Component Comparison

| Component | Dataset Generator (`scripts/make_mock_dataset.py`) | Physics Baseline (`NavalPhysicsFuelBaseline`) | Relationship Classification | Notes / Differences |
| :--- | :--- | :--- | :--- | :--- |
| **Displacement ($\Delta$)** | $(0.20 \times \text{DWT}) + \text{Cargo}$ | $(0.20 \times \text{DWT}) + \text{Cargo}$ | **DIRECTLY SAME** | Identical 20% lightweight displacement assumption. |
| **Speed Relationship ($V$)** | Cubic power law ($V^3$) | Cubic power law ($V^3$) | **DIRECTLY SAME** | Identical hydrodynamic power law. |
| **Admiralty Coeff ($C_{\text{adm}}$)**| 460 to 595 (by vessel class) | Nominal 500, calibrated via $c_{\text{prop}}$ to $\approx 610$ | **MATHEMATICALLY RELATED** | Baseline uses single global calibrated $C_{\text{adm}}$; class differences left to residual. |
| **Auxiliary Load ($P_{\text{aux}}$)** | $0.05 \times \text{DWT}^{0.6} \times 100$ kW | $0.05 \times \text{DWT}^{0.6} \times 100$ kW | **DIRECTLY SAME** | Identical power curve formulation. |
| **Voyage Duration ($H$)** | $(D / V) + U(0.5, 2.5)$ | Implied hours $D / V$ | **MATHEMATICALLY RELATED** | Baseline omits unobservable port buffer ($U(0.5, 2.5)$) to prevent leakage. |
| **Fuel Energy Conversion** | Empirical SFOC lookup (145–345 g/kWh) | Thermodynamic LHV / $\eta_{\text{th}}$ ($45\%$ efficiency) | **MATHEMATICALLY RELATED** | $\text{SFOC}_{\text{eff}} = 3600 / (\eta \cdot \text{LHV})$ has ratio $\approx 0.86\text{--}0.93$ to SFOC table. |
| **Weather Multiplier** | $W \times (1.0 + 0.025 \times \text{Sea State})$ | $\max(W, 1.0)$ | **PARTIALLY RELATED** | Baseline omits sea state term; sea state interaction left to QIFCP residual. |
| **Stochastic Noise** | $\mathcal{N}(\mu=1.0, \sigma=0.02)$ | None (deterministic model) | **INDEPENDENT** | Baseline is deterministic; residual model fits unmodeled variance. |

---

## 6. Feature Leakage Audit (All 27 Predictors)

Every feature in the 27-dimensional feature matrix produced by `FeatureEngineeringPipeline` was audited against five leakage categories:
- **Category A:** Directly observed prior to voyage execution.
- **Category B:** Derived strictly from legitimate pre-voyage input variables.
- **Category C:** Derived from the target (`fuel_consumption`).
- **Category D:** Derived from post-voyage outcome proxies not available at inference.
- **Category E:** Ambiguous.

| # | Feature Name | Classification | Source / Derivation | Leakage Assessment |
|:---:|:---|:---:|:---|:---|
| 1 | `vessel_dwt` | **Category A** | Vessel registry structural capacity | **Clean** |
| 2 | `cargo_tons` | **Category A** | Pre-departure bill of lading manifest | **Clean** |
| 3 | `distance_nm` | **Category A** | Navigational waypoint planned route distance | **Clean** |
| 4 | `speed_knots` | **Category A** | Ordered service cruising speed | **Clean** |
| 5 | `weather_factor` | **Category A** | Metocean forecast weather severity index | **Clean** |
| 6 | `sea_state` | **Category A** | Douglas sea state forecast | **Clean** |
| 7 | `cargo_ratio` | **Category B** | `cargo_tons / vessel_dwt` | **Clean** |
| 8 | `cargo_utilization_pct` | **Category B** | `cargo_ratio * 100.0` | **Clean** |
| 9 | `transport_work` | **Category B** | `cargo_tons * distance_nm` | **Clean** |
| 10 | `ton_nautical_miles` | **Category B** | `cargo_tons * distance_nm` | **Clean** |
| 11 | `power_proxy` | **Category B** | `(0.20 * dwt + cargo)^(2/3) * speed^3` | **Clean** (Legitimate pre-voyage kinematic proxy) |
| 12 | `implied_hours` | **Category B** | `distance_nm / speed_knots` | **Clean** (Inference-time kinematic calculation) |
| 13 | `weather_speed_interaction` | **Category B** | `speed_knots * weather_factor` | **Clean** |
| 14 | `weather_sea_interaction` | **Category B** | `weather_factor * (sea_state + 1.0)` | **Clean** |
| 15–18 | `vessel_type_*` (4 types) | **Category A** | Vessel registry categorical one-hots | **Clean** |
| 19–24 | `fuel_type_*` (6 types) | **Category A** | Bunkered fuel specification one-hots | **Clean** |
| 25–27 | `source_group_*` (3 buckets)| **Category B** | Binned purely from `implied_hours` ($D/V$) | **Clean** |

**Summary of Leakage Guardrails:**
- `hours_at_sea` (which contained the random post-departure buffer $U(0.5, 2.5)$) is **explicitly excluded** from $X$ by `FeatureEngineeringPipeline.check_for_target_leakage()`.
- `co2_emissions` is **strictly excluded**.
- Zero Category C, D, or E features exist in the predictive matrix.

---

## 7. Physics Calibration & Residual Target Leakage Audit

1. **Calibration Leakage Check:**
   - `NavalPhysicsFuelBaseline.fit(X_train, y_train)`: Verified that calibration parameters ($c_{\text{prop}}, c_{\text{aux}}$) are estimated strictly on the current training fold using ordinary least squares without intercept.
   - Out-of-sample test vessels ($X_{\text{test}}, y_{\text{test}}$) are never passed to `.fit()` or referenced in global statistics.
2. **Residual Formation Leakage Check:**
   - Training residual is defined as $r_{\text{train}} = y_{\text{train}} - \hat{y}_{\text{phys, train}}$.
   - Test inference computes $\hat{y}_{\text{final}} = \hat{y}_{\text{phys, test}} + \hat{r}_{\text{qifcp, test}}$ using $X_{\text{test}}$ alone.
   - $y_{\text{test}}$ is used strictly for final error metric computation.

---

## 8. Quantitative Structural-Overlap Measurements (Training Data Only)

Evaluated on the training partition ($N=394$) for Seed 42:

### Regression A: Fit of Physics Baseline Against Training Target
$$\text{fuel\_consumption} \sim \text{NavalPhysicsFuelBaseline}(X_{\text{train}})$$
- **$R^2$:** **0.9905**
- **MAE:** **51.13 metric tons**
- **RMSE:** **89.31 metric tons**

### Regression B: Fit of Generator-Equivalent Physics Terms Against Training Target
$$\text{fuel\_consumption} \sim \beta_1 f_{\text{prop, generator}} + \beta_2 f_{\text{aux, generator}}$$
- **$R^2$:** **0.9991**
- **MAE:** **17.72 metric tons**
- **RMSE:** **28.07 metric tons**
- **Fitted Coefficients:** $\beta_1 = 0.9964$, $\beta_2 = 1.0208$ (virtually $1.000$)

**Interpretation:**
The target $\text{fuel\_consumption}$ in `voyages_sample.csv` can be reconstructed to $R^2 = 0.9991$ using the generator's physical terms. The remaining unexplained error ($\text{MAE} = 17.72\text{ t}$) corresponds almost entirely to the 2% Gaussian noise $\mathcal{N}(1.0, 0.02)$ and the random port buffer $U(0.5, 2.5)\text{ h}$.

---

## 9. Synthetic Data Realism Statistics

Audit of the 500 voyage records in `data/raw/voyages_sample.csv`:

| Continuous Variable | Unit | Min | Median | Mean | Max | Std Dev | Realism Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Fuel Consumption** | t | 10.95 | 839.66 | 1,073.43 | 6,840.87 | 907.09 | Realistic commercial voyage range (short feeder to intercontinental). |
| **CO2 Emissions** | t | 35.11 | 2,523.91 | 3,071.28 | 13,407.33 | 2,519.10 | Conforms to IMO fuel carbon factors. |
| **Vessel DWT** | t | 5,100.0 | 106,100.0 | 111,248.2 | 292,600.0 | 77,251.48 | Spans Handysize to VLCC/Capesize. |
| **Cargo Tons** | t | 2,926.3 | 78,106.3 | 86,725.44 | 270,779.1 | 64,478.40 | Enforces $\text{cargo} \le \text{DWT}$ physically. |
| **Distance** | nm | 303.7 | 3,937.55 | 3,924.22 | 7,462.70 | 2,042.29 | Matches regional to transoceanic trade lanes. |
| **Speed** | knots | 9.80 | 13.30 | 14.31 | 20.50 | 2.63 | Reflects slow steaming and container service speeds. |
| **Hours at Sea** | h | 17.97 | 272.94 | 283.95 | 634.42 | 156.83 | Consistent with distance / speed kinematics. |
| **Weather Factor** | — | 1.00 | 1.12 | 1.12 | 1.31 | 0.06 | Moderate adverse weather penalty. |
| **Sea State** | Douglas | 1.00 | 3.00 | 3.12 | 7.00 | 1.52 | Typical North Atlantic / Pacific sea states. |

---

## 10. Circularity Classification

### Assigned Level: **LEVEL 3**
> **"Target is generated primarily from the same physical equations used by the baseline."**

**Technical Justification:**
1. Both `scripts/make_mock_dataset.py` and `NavalPhysicsFuelBaseline` use identical naval architectural core mechanics:
   - Displacement exponent $2/3$ applied to $(0.20 \times \text{DWT} + \text{Cargo})$
   - Velocity exponent $3.0$
   - Auxiliary power proportional to $\text{DWT}^{0.6} \times 100$
   - Proportional scaling with voyage duration and weather severity
2. The Phase 2D physics baseline was **not** an accidental discovery of an empirical natural phenomenon; it successfully **recovers the intentional mathematical formulation used by the project's own synthetic dataset generator**.
3. The structural differences between generator and baseline (SFOC lookup vs LHV, class-specific $C_{\text{adm}}$ vs calibrated scalar $C_{\text{adm}}$, sea-state multiplier terms) are exactly the residual errors that the multi-harmonic QIFCP model learns.

---

## 11. What the Current Phase 2D Benchmark Legitimately Demonstrates

1. **Recovery and Correction of a Known Synthetic Physical Target Structure:**  
   The experiment proves that an additive physics-informed quantum surrogate ($\hat{y} = y_{\text{phys}} + \hat{r}_{\text{qifcp}}$) can rapidly recover a complex synthetic target structure where pure machine learning models (Random Forest, HistGradientBoosting, Direct QIFCP) struggle with vessel-disjoint out-of-distribution generalization.
2. **QIFCP Residual Compensation Capability:**  
   When the gross physical trend is removed, QIFCP's multi-harmonic feature basis and adaptive entanglement graph effectively model the remaining secondary effects (vessel-type efficiency offsets, sea-state penalties, fuel-density deviations), reducing residual variance by an additional $35\%\text{--}55\%$.
3. **Architectural Pipeline Integrity:**  
   The model demonstrates strict leakage guardrails, sub-millisecond inference latency (0.68 ms / 100 samples), and 100% deterministic repeatability across seeds.

---

## 12. What the Current Phase 2D Benchmark DOES NOT Demonstrate

1. **Does NOT Demonstrate Performance on Independent Real-World Telemetry:**  
   Real-world vessel fuel consumption is governed by complex hydrodynamics including dynamic trim, wave diffraction, fouling degradation, engine load-tuning curves, and varying sea currents that do not follow an exact closed-form Admiralty equation.
2. **Does NOT Prove "Quantum Advantage" or Universal Superiority:**  
   The high accuracy ($R^2 > 0.99$, MAE $\approx 40\text{ t}$) is primarily a direct consequence of the physics baseline aligning with the synthetic data generator's equations.
3. **Does NOT Establish Causal Physical Validity:**  
   Calibrated multipliers ($c_{\text{prop}} \approx 0.82, c_{\text{aux}} \approx 1.28$) reflect empirical parameter fitting against synthetic data, not measured shaft dynamometer readings.

---

## 13. Recommended Wording for the Final SIH Report & Presentation

To ensure uncompromising academic rigor, scientific credibility, and compliance with hackathon evaluation standards, use the following wording guidelines:

### Approved Framing:
> *"The Phase 2D hybrid architecture integrates a first-principles naval architecture baseline (Admiralty resistance and hotel auxiliary power) with a Quantum-Inspired Feature-Encoding (QIFCP) residual compensator. On the controlled synthetic benchmark dataset (`voyages_sample.csv`), this physics-informed decomposition enables the model to capture the primary hydrodynamic power relationship while QIFCP learns secondary nonlinear interactions (sea-state resistance and class offsets), reducing out-of-sample MAE from 110.88 t to 40.07 t across unseen vessels. This demonstrates the viability of hybrid quantum-classical residual learning on physically structured targets."*

### Explicit Disclaimers Required:
> *"Note on Benchmark Dataset: The current evaluation is conducted on a synthetic maritime voyage dataset generated using naval architectural propulsion principles. Consequently, the physics baseline's high explanatory power reflects recovery of this underlying synthetic formulation. Validation against independent, measured high-frequency noon-report or sensor telemetry is required before asserting real-world operational accuracy."*

### Prohibited Phrases:
- Do NOT claim: *"universal superiority over classical ML in real-world shipping"*.
- Do NOT claim: *"proven quantum advantage in maritime fuel forecasting"*.
- Do NOT claim: *"empirical discovery of naval propulsion laws"*.
