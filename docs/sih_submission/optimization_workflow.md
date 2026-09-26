# SIH26138: Multi-Tier Green Fleet Optimization Workflow

**Problem ID:** SIH26138 — Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization  

---

## 1. Co-Optimization Pipeline

Rather than optimizing fleet mix, ship sizing, and sailing speed independently, the platform executes a **unified multi-tier co-optimization**:

```mermaid
sequenceDiagram
    participant User as Fleet Operator / Executive
    participant Strategy as FleetStrategyOptimizer
    participant Composition as FleetCompositionOptimizer
    participant Capacity as VesselCapacityOptimizer
    participant Speed as EcoSpeedOptimizer
    participant Reliability as ReliabilityEngine
    participant Compliance as ComplianceEngine

    User->>Strategy: Submit Operational Scenario (Demand, Distance, Deadline, Budget)
    Strategy->>Composition: Determine Optimal Mix (Feeder, Panamax, Capesize, Alt-Fuel Share)
    Composition->>Capacity: Check Class Sizing Bounds & Port Restrictions
    Capacity-->>Composition: Verified DWT / TEU Capacity Constraints
    Composition-->>Strategy: Sized Fleet Candidate Vector
    Strategy->>Speed: Optimize Cruising Speed (Admiralty Wave-Making Law)
    Speed->>Compliance: Verify FuelEU Penalties & IMO CII Trajectory
    Compliance-->>Speed: Compliance Cost Multipliers
    Speed-->>Strategy: Eco-Speed & Arrival Buffer Allocation
    Strategy->>Reliability: Audit Cargo Demand & Schedule Reliability
    Reliability-->>Strategy: Decomposed Reliability Score & Route Feasibility
    Strategy-->>User: Holistic Fleet Strategy Recommendation
```

---

## 2. Objective Function Formulation

The co-optimization minimizes composite multi-criteria cost:

$$\min_{x} J(x) = w_{\text{cost}} \cdot \frac{C(x)}{C_0} + w_{\text{fuel}} \cdot \frac{F(x)}{F_0} + w_{\text{emiss}} \cdot \frac{E(x)}{E_0} + \mathcal{P}_{\text{penalties}}(x)$$

where:
- $C(x) = \text{Capex} + \text{Charter} + \text{Bunker Cost} + \text{Carbon Tax} + \text{FuelEU Penalties} + \text{Demurrage}$
- $F(x) = \sum_{v} \Delta_v^{2/3} V_v^3 \cdot \frac{D_v}{24 V_v \cdot C_{\text{adm}}}$
- $E(x) = \text{WTW Lifecycle CO}_2\text{e Emissions}$
- $\mathcal{P}_{\text{penalties}}(x)$: Severe quadratic barrier penalties for capacity shortfall, missed delivery deadlines, or budget violations.
