# Dataset Sanity & Quality Audit Report

**Problem ID:** SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization  
**Dataset File:** `voyages_sample.csv`  
**Dataset Records:** 10,000  
**Audit Status:** APPROVED FOR ML PIPELINE  

---

## Executive Summary

A comprehensive data sanity audit was conducted on the synthetic voyage telemetry dataset (`10,000` records). The dataset was evaluated across three formal verification criteria:
1. **Data Completeness and Boundary Integrity** (zero missing, infinite, or negative values).
2. **Maritime Physics & Correlation Significance** (hydrodynamic resistance laws and feature sensitivity).
3. **Target Spread & Dispersion** (absence of point-mass spikes or degenerate uniformity).

All checks passed successfully.

---

## Check 1: Data Completeness & Boundary Verification

| Integrity Metric | Observed Count | Threshold / Rule | Result |
| :--- | :--- | :--- | :--- |
| **Missing Values (NaN / Null)** | `0` | Must be `0` | **PASS** |
| **Infinite Values (Inf / -Inf)** | `0` | Must be `0` | **PASS** |
| **Negative Numeric Values** | `0` | Must be `0` | **PASS** |
| **Synthetic Markers (`is_synthetic`)** | `10,000` / `10,000` | Must be 100% `True` | **PASS** |
| **Lineage Source (`data_source`)** | `10,000` / `10,000` | Must be 100% `'mock'` | **PASS** |

### Numerical Summary Statistics

| Feature | Min | 25% | Median | 75% | Max | Skewness |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `vessel_dwt` | 5,100 | 47,500 | 103,200 | 150,200 | 292,600 | 0.59 |
| `cargo_tons` | 2,806.2 | 33,721.3 | 75,190.1 | 116,684.6 | 277,164.8 | 0.80 |
| `distance_nm` | 300.1 | 2,143.8 | 3,942.6 | 5,750.4 | 7,499.7 | -0.03 |
| `speed_knots` | 9.3 | 12.5 | 13.4 | 17.3 | 22.5 | 0.76 |
| `hours_at_sea` | 17.1 | 149.7 | 277.0 | 401.2 | 776.9 | 0.21 |
| `weather_factor` | 1.00 | 1.08 | 1.12 | 1.16 | 1.33 | 0.46 |
| `sea_state` | 1 | 2 | 3 | 4 | 7 | 0.55 |
| **`fuel_consumption`** | **10.95** | **359.44** | **811.17** | **1519.25** | **8065.55** | **1.74** |
| `co2_emissions` | 35.11 | 1024.54 | 2352.68 | 4321.75 | 14424.47 | 1.12 |

---

## Check 2: Correlation & Physical Plausibility Analysis

### Feature Correlations with `fuel_consumption`

| Feature | Pearson Correlation (r) | Physical Direction | Maritime Physics Rationale |
| :--- | :--- | :--- | :--- |
| **`distance_nm`** | `+0.589` | Positive (Increases) | Direct voyage length driving propulsion duration. |
| **`vessel_dwt`** | `+0.558` | Positive (Increases) | Displacement hull resistance (Power proportional to Displacement^(2/3)). |
| **`cargo_tons`** | `+0.537` | Positive (Increases) | Payload mass directly increases vessel draft and displacement. |
| **`hours_at_sea`** | `+0.450` | Positive (Increases) | Cumulative engine operational hours. |
| **`speed_knots`** | `+0.249` | Positive (Increases) | Governed by cubic power law (Power proportional to Speed^3). |
| **`weather_factor`**| `+0.108` | Positive (Increases) | Wave and aerodynamic added resistance. |
| **`sea_state`** | `+0.105` | Positive (Increases) | Douglas sea scale wave encounter severity. |

### Operational Rate Sensitivity

Because total voyage fuel is the product of hourly fuel burn and transit time (Fuel = Rate * Hours), examining the hourly consumption rate (MT/hour) isolates the pure hydrodynamic power signal:

Fuel Rate = fuel_consumption / hours_at_sea proportional to Displacement^(2/3) * Speed^3

* **Correlation (`fuel_per_hour` vs `speed_knots`):** `+0.558` (Strong cubic speed dependence)
* **Correlation (`fuel_per_hour` vs `vessel_dwt`):** `+0.665` (Strong displacement hull scale dependence)
* **Correlation (`fuel_per_hour` vs `weather_factor`):** `+0.148` (Environmental resistance increase)

---

## Check 3: Target Distribution (`fuel_consumption`)

* **Mean:** `1074.85` MT
* **Standard Deviation:** `947.41` MT
* **Median:** `811.17` MT
* **Interquartile Range (IQR):** `359.44` MT to `1519.25` MT
* **95th Percentile:** `2849.25` MT
* **Skewness:** `+1.74` (Healthy positive / right-skewed operational distribution)
* **Kurtosis:** `+4.76` (No degenerate point-mass spikes or uniform flatlines)

The target exhibits the classical unimodal, right-skewed distribution observed in real maritime commercial operations: small vessels / short feeder voyages form the bulk of operational volume, while large laden VLCC / Capesize intercontinental voyages form the long upper tail.

---

## Audit A: Duplicate Voyage & Fleet Entity Integrity

* **Total Recorded Voyages:** `10,000`
* **Unique `voyage_id` Count:** `10,000` (100.0% Unique, Zero Duplicates)
* **Unique `vessel_id` Entities:** `60` distinct commercial vessels
* **Average Voyages per Vessel:** `166.7` voyages
* **Verdict:** **PASS** (Voyage IDs are strictly unique primary keys; multi-voyage vessel reuse mirrors real AIS commercial operations).

---

## Audit B: Fuel-Type Balance & Multi-Fuel Representation

| Marine Fuel Type | Record Count | Representation (%) | Role in Decarbonization Pipeline |
| :--- | :--- | :--- | :--- |
| **Diesel (MDO/MGO)** | `6,860` | `68.60%` | Dominant commercial baseline fuel |
| **LNG (Liquefied Natural Gas)** | `2,123` | `21.23%` | Transitional lower-carbon hydrocarbon |
| **Methanol (CH3OH)** | `1,017` | `10.17%` | Emerging green alternative electro-fuel |

* **Verdict:** **PASS** (Diesel is dominant as expected in world fleet data, while LNG and Methanol provide substantial statistical mass for multi-fuel comparative analysis).

---

## Audit C: Vessel-Class Coverage & Stratification

| Vessel Classification | Voyage Count | Share (%) | Mean DWT | Mean Fuel (MT) | Mean Speed (kn) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Bulk Carrier** | `2,905` | `29.05%` | 102,229 MT | 947.3 MT | 12.7 kn |
| **Container Ship** | `2,677` | `26.77%` | 100,582 MT | 1,304.9 MT | 18.5 kn |
| **Oil Tanker** | `2,573` | `25.73%` | 193,541 MT | 1,547.9 MT | 13.7 kn |
| **General Cargo** | `1,845` | `18.45%` | 17,054 MT | 282.3 MT | 12.0 kn |

* **Verdict:** **PASS** (All 4 vessel classes are robustly sampled without class collapse; differences in operational profiles faithfully reflect maritime naval architecture).

---

## Audit D: Target Leakage & Architectural Guardrails (CRITICAL)

* **Observed Pearson Correlation (`fuel_consumption` vs `co2_emissions`):** `+0.8836`
* **Deterministic Mechanism:**
  CO2 = fuel_consumption * C_F(fuel_type)
  Where stoichiometric emission factor C_F is strictly constant per fuel (Diesel: 3.206, LNG: 2.750, Methanol: 1.375).

### Mandatory Architectural Constraint for Phase 2 Prediction Engine:
> **WARNING: TARGET LEAKAGE GUARD**  
> Under no circumstances may `co2_emissions` be admitted into the predictive feature set X.  
> Because CO2 is a deterministic scalar multiple of fuel consumption, exposing it to any supervised model would cause the model to bypass hydrodynamic and kinematic features and trivially solve fuel approx CO2 / C_F. At inference time, voyage CO2 is unobserved until predicted fuel is known.

* **Approved Feature Vector (X):**
  X = [vessel_type, vessel_dwt, cargo_tons, distance_nm, speed_knots, hours_at_sea, fuel_type, weather_factor, sea_state]
* **Prediction Target (y):**
  y = fuel_consumption
* **Downstream Compliance & Emissions Pipeline:**
  y_hat_co2 = EmissionEngine.calculate_ttw(y_hat_fuel, fuel_type)

---

## Machine Learning Suitability Benchmark

To confirm that the dataset is ready for supervised learning before building the ingestion pipeline:

1. **Standard Linear Regression (Raw Features):** R2 = 0.725
2. **Log-Linear Regression (ln(y) ~ ln(X)):** R2 = 0.949
3. **Gradient Boosted Decision Trees (HistGBDT):** R2 = 0.982, MAPE = 8.67%

The strong physical relationships allow predictive engines to converge smoothly while retaining sufficient stochastic variance (2% hull condition variance) to prevent trivial overfitting.

---

## Generated Artifacts

1. [`correlation_matrix.png`](../figures/correlation_matrix.png): Complete Pearson feature correlation heatmap.
2. [`fuel_distribution.png`](../figures/fuel_distribution.png): Histogram and density profiling of target fuel consumption.
3. [`vessel_distribution.png`](../figures/vessel_distribution.png): Distribution of voyages and mean consumption across vessel classes.
