import os
from pathlib import Path
from typing import Dict, List, Literal, Tuple
import numpy as np
import pandas as pd


class VesselDataset:
    """Maritime operational dataset pipeline supporting synthetic generation and AIS ingestion.

    Physics Grounding:
    Fuel consumption obeys the Admiralty Coefficient empirical formula:
        P_brake = (Disp^(2/3) * V^3) / C_adm
    where:
        - Disp is vessel displacement (metric tons), proportional to DWT and load factor.
        - V is vessel speed over ground (knots), establishing a cubic power-speed law.
        - C_adm is the dimensionless Admiralty coefficient (ranging 450 - 650 depending on hull design).
        - Fuel consumption (MT/day) = (P_brake * SFOC * 24) / 10^6, with Specific Fuel Oil Consumption (SFOC) ~ 170-205 g/kWh.
        - Hydrodynamic weather penalties scale resistance via Kwon's empirical wave-wind resistance formulation.
    """

    def __init__(self, raw_data_path: str = "data/ais_raw.csv") -> None:
        self.raw_data_path = Path(raw_data_path)
        self.df: pd.DataFrame = pd.DataFrame()

    @staticmethod
    def emission_factors() -> Dict[str, float]:
        """Well-to-Wake (WtW) lifecycle greenhouse gas emission factors in gCO2eq/MJ.

        Values represent lifecycle carbon intensity conforming to statutory FuelEU Maritime
        and IMO MEPC.308(73) regulatory baselines.
        """
        return {
            "HFO": 94.0,
            "LNG": 75.0,
            "methanol_green": 15.0,
            "H2_green": 9.0,
            "NH3_green": 4.0,
        }

    def load(self, source: Literal["synthetic", "ais"] = "synthetic", n_per_class: int = 500, seed: int = 42) -> pd.DataFrame:
        """Loads dataset either via deterministic synthetic simulation or local AIS file ingestion."""
        if source == "synthetic":
            self.df = self._generate_synthetic(n_per_class=n_per_class, seed=seed)
        elif source == "ais":
            self.df = self._load_ais()
        else:
            raise ValueError(f"Unsupported source '{source}'. Expected 'synthetic' or 'ais'.")
        return self.df

    def _generate_synthetic(self, n_per_class: int = 500, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)

        # Baseline parameters per vessel class
        # class_name: (dwt_nominal, speed_min, speed_max, c_adm, sfoc_base_g_kwh, disp_dwt_ratio)
        class_configs = {
            "container": (50000.0, 14.0, 24.0, 580.0, 175.0, 1.30),
            "tanker": (100000.0, 10.0, 16.0, 480.0, 185.0, 1.20),
            "bulk_carrier": (70000.0, 10.0, 15.5, 490.0, 180.0, 1.22),
            "LNG_carrier": (75000.0, 14.0, 20.0, 540.0, 170.0, 1.25),
        }

        fuel_options = ["HFO", "LNG", "methanol_green", "H2_green", "NH3_green"]
        fuel_probabilities = [0.55, 0.25, 0.10, 0.05, 0.05]

        # Fuel energy density relative to standard HFO (LHV ~ 40.2 MJ/kg)
        # Higher LHV requires fewer metric tons for equivalent propulsion work
        fuel_mass_ratios = {
            "HFO": 1.00,
            "LNG": 0.82,            # LHV ~ 49.2 MJ/kg
            "methanol_green": 2.02, # LHV ~ 19.9 MJ/kg
            "H2_green": 0.33,       # LHV ~ 120.0 MJ/kg
            "NH3_green": 2.16,      # LHV ~ 18.6 MJ/kg
        }

        routes = [f"RT_{idx:02d}" for idx in range(1, 16)]
        route_distances = {r: float(rng.integers(350, 4500)) for r in routes}

        records: List[Dict] = []

        for class_name, (dwt_nom, spd_min, spd_max, c_adm, sfoc, disp_ratio) in class_configs.items():
            for i in range(n_per_class):
                vessel_id = f"{class_name[:3].upper()}_{i + 1:04d}"
                dwt = float(dwt_nom * rng.uniform(0.96, 1.04))
                speed_kn = float(rng.uniform(spd_min, spd_max))
                load_factor = float(rng.uniform(0.35, 0.95))
                wind_bft = int(rng.integers(0, 9))  # Beaufort scale 0-8
                wave_ht_m = float(np.clip(rng.gamma(shape=2.0, scale=0.8), 0.1, 7.5))
                fuel_type = str(rng.choice(fuel_options, p=fuel_probabilities))
                route_id = str(rng.choice(routes))
                distance_nm = float(route_distances[route_id] + rng.normal(0, 15))
                distance_nm = max(distance_nm, 50.0)

                # Displacement calculation: Lightship weight + carried cargo
                cargo_carried = dwt * load_factor
                displacement = (dwt * (disp_ratio - 1.0)) + cargo_carried

                # Hydrodynamic Propulsion Power (Admiralty Formula):
                # P_prop (kW) = (Displacement^(2/3) * Speed^3) / C_adm
                propulsion_power_kw = (np.power(displacement, 2.0 / 3.0) * np.power(speed_kn, 3.0)) / c_adm

                # Weather resistance penalty factor based on wind force and wave height:
                # delta_R/R = 1 + alpha * (wave_ht / 2) + beta * (wind_bft / 6)^2
                weather_factor = 1.0 + (0.045 * wave_ht_m) + (0.025 * (wind_bft ** 1.5))
                total_power_kw = propulsion_power_kw * weather_factor

                # Daily fuel consumption (MT/day) baseline under Admiralty cube law
                # Fuel_MT_day = (P_kW * SFOC_g_kWh * 24 hours) / 1,000,000 g_per_MT
                fuel_consumption_mt_day = float((total_power_kw * sfoc * 24.0) / 1e6)

                records.append({
                    "vessel_id": vessel_id,
                    "class": class_name,
                    "DWT": round(dwt, 1),
                    "speed_kn": round(speed_kn, 2),
                    "load_factor": round(load_factor, 3),
                    "wind_bft": wind_bft,
                    "wave_ht_m": round(wave_ht_m, 2),
                    "fuel_type": fuel_type,
                    "fuel_consumption_mt_day": round(fuel_consumption_mt_day, 3),
                    "distance_nm": round(distance_nm, 1),
                    "route_id": route_id,
                })

        return pd.DataFrame(records)

    def _load_ais(self) -> pd.DataFrame:
        if not self.raw_data_path.exists():
            raise FileNotFoundError(
                f"AIS data source file not found at: {self.raw_data_path.resolve()}"
            )
        df_ais = pd.read_csv(self.raw_data_path)
        required_columns = [
            "vessel_id", "class", "DWT", "speed_kn", "load_factor",
            "wind_bft", "wave_ht_m", "fuel_type", "fuel_consumption_mt_day",
            "distance_nm", "route_id"
        ]
        missing = [c for c in required_columns if c not in df_ais.columns]
        if missing:
            raise KeyError(f"AIS file missing required schema columns: {missing}")

        return df_ais[required_columns].copy()

    def split(self, test_size: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Performs a reproducible train-test split stratified across vessel classes."""
        if self.df.empty:
            raise ValueError("Dataset is empty. Call load() prior to splitting.")

        rng = np.random.default_rng(seed)
        train_indices: List[int] = []
        test_indices: List[int] = []

        # Stratified sampling per vessel class to maintain distribution integrity
        for _, class_group in self.df.groupby("class"):
            indices = class_group.index.to_numpy().copy()
            rng.shuffle(indices)
            split_idx = int(len(indices) * (1.0 - test_size))
            train_indices.extend(indices[:split_idx])
            test_indices.extend(indices[split_idx:])

        train_df = self.df.loc[train_indices].sample(frac=1.0, random_state=seed).reset_index(drop=True)
        test_df = self.df.loc[test_indices].sample(frac=1.0, random_state=seed).reset_index(drop=True)

        return train_df, test_df
