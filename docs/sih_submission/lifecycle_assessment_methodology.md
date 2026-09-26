# SIH26138: Lifecycle Assessment (LCA) Methodology & Fuel Pathways

**Problem ID:** SIH26138 — Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization  

---

## 1. Well-to-Wake (WTW) Boundary Definition

The platform models maritime emissions under strict Well-to-Wake (WTW) boundaries conforming to Regulation (EU) 2023/1805 (FuelEU Maritime) and IMO lifecycle guidelines:

$$\text{WTW Emissions} = \text{Well-to-Tank (WTT)} + \text{Tank-to-Wake (TTW)}$$

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             WELL-TO-WAKE BOUNDARY                                │
├────────────────────────────────────────┬─────────────────────────────────────────┤
│         WELL-TO-TANK (WTT)             │          TANK-TO-WAKE (TTW)             │
│  - Primary Resource Extraction         │  - Onboard Storage & Boil-Off Slip      │
│  - Refining / Chemical Synthesis       │  - Internal Combustion Engine / Fuel Cell│
│  - Liquefaction & Compression          │  - Direct Exhaust GHGs (CO2, CH4, N2O)  │
│  - Maritime Transport & Bunkering      │                                         │
└────────────────────────────────────────┴─────────────────────────────────────────┘
```

---

## 2. Granular Feedstock Pathways (`FuelLifecycleProfile`)

To prevent simplistic "zero-carbon" assumptions, the platform models explicit production pathways:

| Fuel Name | Production Pathway | Production EF ($t\text{CO}_2\text{e}/t$) | Transport EF ($t\text{CO}_2\text{e}/t$) | TTW Factor ($t\text{CO}_2\text{e}/t$) | LHV Energy ($MJ/t$) | Renewable Share |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Diesel** | Fossil (MGO/HFO) | 0.450 | 0.140 | 3.206 | 42,700 | 0.0 |
| **Diesel** | Biodiesel (HVO) | 0.250 | 0.100 | 3.206* | 37,200 | 1.0 |
| **LNG** | Fossil LNG | 0.500 | 0.150 | 2.750 + slip | 49,100 | 0.0 |
| **LNG** | Bio-LNG | 0.180 | 0.070 | 2.750 + slip | 49,100 | 1.0 |
| **Methanol** | Fossil (Natural Gas) | 0.300 | 0.100 | 1.375 | 19,900 | 0.0 |
| **Methanol** | Bio-Methanol | 0.120 | 0.080 | 1.375* | 19,900 | 1.0 |
| **Methanol** | E-Methanol (DAC+H2) | 0.040 | 0.060 | 1.375* | 19,900 | 1.0 |
| **Hydrogen** | Grey (SMR) | 9.000 | 1.500 | 0.000 | 120,000 | 0.0 |
| **Hydrogen** | Blue (SMR + CCS) | 1.500 | 1.000 | 0.000 | 120,000 | 0.0 |
| **Hydrogen** | Green (Electrolysis) | 0.250 | 0.100 | 0.000 | 120,000 | 1.0 |
| **Ammonia** | Grey (Fossil Haber-Bosch) | 1.900 | 0.300 | 0.000 | 18,600 | 0.0 |
| **Ammonia** | Blue (Haber-Bosch + CCS) | 0.450 | 0.200 | 0.000 | 18,600 | 0.0 |
| **Ammonia** | Green (Renewable H2) | 0.080 | 0.120 | 0.000 | 18,600 | 1.0 |

*\*Note: Under FuelEU crediting, biogenic and e-fuel carbon in TTW is counted as atmospheric neutral when certified sustainable.*
