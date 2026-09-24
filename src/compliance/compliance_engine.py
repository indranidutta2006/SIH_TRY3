"""Maritime statutory compliance engine for IMO CII and EU FuelEU Maritime regulations.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Fulfills contracts.interfaces.ComplianceEngine, evaluating Carbon Intensity Indicator (CII)
operational rating curves (A to E) and EU FuelEU Maritime statutory GHG deficit penalties.
"""

import logging
from typing import Any, Final, Optional

from contracts.exceptions import ComplianceError, DataValidationError
from contracts.interfaces import ComplianceEngine
from contracts.schemas import (
    CIIResult,
    ComplianceAssessment,
    ComplianceResult,
    FuelEUResult,
)

logger = logging.getLogger("maritime_system")

# FuelEU Maritime 2020 reference baseline (Regulation (EU) 2023/1805)
FUELEU_REFERENCE_GHG_INTENSITY: Final[float] = 91.16  # gCO2eq/MJ
VLSFO_ENERGY_DENSITY_MJ_PER_TON: Final[float] = 41000.0  # MJ/t
FUELEU_STATUTORY_PENALTY_RATE_EUR: Final[float] = 2400.0  # EUR/ton VLSFO equivalent

# IMO Resolution MEPC.400(83) statutory annual reduction factor Z (adopted 11 April 2025)
CII_Z_FACTORS: Final[dict[int, float]] = {
    2023: 0.05000,
    2024: 0.07000,
    2025: 0.09000,
    2026: 0.11000,
    2027: 0.13625,
    2028: 0.16250,
    2029: 0.18875,
    2030: 0.21500,
}


class MaritimeComplianceEngine(ComplianceEngine):
    """Statutory maritime environmental compliance accounting engine."""

    def __init__(self) -> None:
        """Initialize compliance engine."""
        self.logger = logger

    def get_cii_z_factor(self, year: int) -> float:
        """Retrieve statutory IMO CII reduction factor Z conforming to MEPC.400(83).

        Args:
            year: Compliance assessment reporting year (supported range: 2023–2030).

        Returns:
            Statutory reduction factor Z as a float.

        Raises:
            ComplianceError: If reporting year is outside statutory range (2023–2030).
        """
        if year not in CII_Z_FACTORS:
            raise ComplianceError(
                f"Statutory IMO CII reduction factor Z is only defined for years 2023–2030 "
                f"under IMO Resolution MEPC.400(83), got: {year}.",
                details={"year": year, "supported_years": list(CII_Z_FACTORS.keys())},
            )
        return CII_Z_FACTORS[year]

    def get_fueleu_target(self, year: int) -> float:
        """Retrieve statutory FuelEU Maritime maximum GHG intensity target for given year."""
        if year < 2020:
            raise ComplianceError(
                f"Invalid compliance reporting year: {year}. Must be >= 2020.",
                details={"year": year},
            )
        if year <= 2024:
            return FUELEU_REFERENCE_GHG_INTENSITY
        elif year <= 2029:
            return round(FUELEU_REFERENCE_GHG_INTENSITY * 0.98, 4)  # -2%
        elif year <= 2034:
            return round(FUELEU_REFERENCE_GHG_INTENSITY * 0.94, 4)  # -6%
        elif year <= 2039:
            return round(FUELEU_REFERENCE_GHG_INTENSITY * 0.855, 4)  # -14.5%
        elif year <= 2044:
            return round(FUELEU_REFERENCE_GHG_INTENSITY * 0.69, 4)  # -31%
        elif year <= 2049:
            return round(FUELEU_REFERENCE_GHG_INTENSITY * 0.38, 4)  # -62%
        else:
            return round(FUELEU_REFERENCE_GHG_INTENSITY * 0.20, 4)  # -80%

    def resolve_cii_reference_line(
        self,
        vessel_type: str,
        capacity: float,
        capacity_type: Optional[str] = None,
    ) -> tuple[float, float, float, str]:
        """Resolve IMO Resolution MEPC.353(78) (G2) reference line parameters.

        Form: CII_ref = a * (Capacity ** -c)
        Where Capacity is DWT or GT depending on the statutory category.

        Statutory branches conforming to Table 1 of Resolution MEPC.353(78):
        1. Bulk carrier:
           - Capped at 279,000 DWT: a=4745.0, c=0.622, capacity_metric="DWT"
        2. Tanker:
           - a=5247.0, c=0.610, capacity_metric="DWT"
        3. Containership / Container:
           - a=1984.0, c=0.489, capacity_metric="DWT"
        4. General cargo ship:
           - >= 20,000 DWT: a=31948.0, c=0.792, capacity_metric="DWT"
           - < 20,000 DWT: a=588.0, c=0.3885, capacity_metric="DWT"
        5. LNG carrier:
           - >= 100,000 DWT: a=9.827, c=0.0 (CII_ref = 9.827), capacity_metric="DWT"
           - 65,000 to < 100,000 DWT: a=1.4479e14, c=2.673, capacity_metric="DWT"
           - < 65,000 DWT: a=1.4779e14, c=2.673, capacity_metric="DWT"
        6. Ro-ro cargo ship (vehicle carrier):
           - >= 30,000 GT: a=3627.0, c=0.590, capacity_metric="GT"
           - < 30,000 GT: a=330.0, c=0.329, capacity_metric="GT"
        7. Ro-ro cargo ship:
           - All sizes: a=1967.0, c=0.485, capacity_metric="GT"
        8. Ro-ro passenger ship:
           - All sizes: a=2023.0, c=0.460, capacity_metric="GT"
        9. Cruise passenger ship:
           - All sizes: a=930.0, c=0.383, capacity_metric="GT"
        10. Gas carrier:
           - >= 65,000 DWT: a=1.4405e11, c=2.071, capacity_metric="DWT"
           - < 65,000 DWT: a=8104.0, c=0.639, capacity_metric="DWT"
        11. Refrigerated cargo carrier:
           - All sizes: a=4600.0, c=0.557, capacity_metric="DWT"
        12. Combination carrier:
           - All sizes: a=5119.0, c=0.622, capacity_metric="DWT"

        Returns:
            Tuple of (a, c, effective_capacity, statutory_capacity_metric).
        """
        v_norm = (vessel_type or "").strip().lower()

        # 1. Ro-Ro categories (Statutory capacity is Gross Tonnage - GT)
        if "vehicle" in v_norm or "car carrier" in v_norm:
            metric = "GT"
            if capacity >= 30000.0:
                a, c = 3627.0, 0.590
            else:
                a, c = 330.0, 0.329
            eff_cap = capacity
        elif "passenger" in v_norm and ("ro" in v_norm or "ferry" in v_norm):
            metric = "GT"
            a, c = 2023.0, 0.460
            eff_cap = capacity
        elif "roro" in v_norm or "ro-ro" in v_norm:
            metric = "GT"
            a, c = 1967.0, 0.485
            eff_cap = capacity
        elif "cruise" in v_norm:
            metric = "GT"
            a, c = 930.0, 0.383
            eff_cap = capacity

        # 2. General Cargo Ship (Statutory capacity is DWT with threshold at 20,000 DWT)
        elif "general cargo" in v_norm:
            metric = "DWT"
            if capacity >= 20000.0:
                a, c = 31948.0, 0.792
            else:
                a, c = 588.0, 0.3885
            eff_cap = capacity

        # 3. LNG Carrier (Statutory capacity is DWT with multi-tier curves)
        elif "lng" in v_norm:
            metric = "DWT"
            if capacity >= 100000.0:
                a, c = 9.827, 0.0
            elif capacity >= 65000.0:
                a, c = 1.4479e14, 2.673
            else:
                a, c = 1.4779e14, 2.673
            eff_cap = capacity

        # 4. Gas Carrier
        elif "gas" in v_norm:
            metric = "DWT"
            if capacity >= 65000.0:
                a, c = 1.4405e11, 2.071
            else:
                a, c = 8104.0, 0.639
            eff_cap = capacity

        # 5. Container Ship
        elif "container" in v_norm:
            metric = "DWT"
            a, c = 1984.0, 0.489
            eff_cap = capacity

        # 6. Tanker
        elif "tanker" in v_norm:
            metric = "DWT"
            a, c = 5247.0, 0.610
            eff_cap = capacity

        # 7. Refrigerated Cargo
        elif "refrigerated" in v_norm:
            metric = "DWT"
            a, c = 4600.0, 0.557
            eff_cap = capacity

        # 8. Combination Carrier
        elif "combination" in v_norm:
            metric = "DWT"
            a, c = 5119.0, 0.622
            eff_cap = capacity

        # 9. Bulk Carrier & Default Fallback
        else:
            metric = "DWT"
            a, c = 4745.0, 0.622
            eff_cap = min(capacity, 279000.0)

        # Notify if user provided capacity_type contradicts statutory standard
        if capacity_type and capacity_type.strip().upper() != metric:
            self.logger.warning(
                "Statutory reference line for '%s' under IMO MEPC.353(78) uses %s, "
                "but capacity_type='%s' was specified.",
                vessel_type,
                metric,
                capacity_type,
            )

        return a, c, eff_cap, metric

    def resolve_cii_rating_boundaries(
        self,
        vessel_type: str,
        capacity: float,
    ) -> tuple[float, float, float, float]:
        """Resolve IMO Resolution MEPC.354(78) (G4) ship-type-specific rating boundaries.

        Table 1 dd vectors transformed into exponential boundary multipliers:
        exp(d1), exp(d2), exp(d3), exp(d4):
        - Grade A (Superior):  ratio <= exp(d1)
        - Grade B (Minor):     exp(d1) < ratio <= exp(d2)
        - Grade C (Moderate):  exp(d2) < ratio <= exp(d3)
        - Grade D (Inferior):  exp(d3) < ratio <= exp(d4)
        - Grade E (Poor):      ratio > exp(d4)

        Statutory branches conforming to Table 1 of Resolution MEPC.354(78):
        1. Bulk carrier:               (0.86, 0.94, 1.06, 1.18)
        2. Tanker:                     (0.82, 0.93, 1.08, 1.28)
        3. Containership:              (0.83, 0.94, 1.07, 1.19)
        4. General cargo ship:         (0.83, 0.94, 1.06, 1.19)
        5. LNG carrier (>= 100k DWT):  (0.89, 0.98, 1.06, 1.13)
        6. LNG carrier (< 100k DWT):   (0.78, 0.92, 1.10, 1.37)
        7. Ro-ro cargo (vehicle):      (0.86, 0.94, 1.06, 1.16)
        8. Ro-ro cargo ship:           (0.76, 0.89, 1.08, 1.27)
        9. Ro-ro passenger ship:       (0.76, 0.92, 1.14, 1.30)
        10. Cruise passenger ship:     (0.87, 0.95, 1.06, 1.16)
        11. Gas carrier (>= 65k DWT):  (0.81, 0.91, 1.12, 1.44)
        12. Gas carrier (< 65k DWT):   (0.85, 0.95, 1.06, 1.25)
        13. Refrigerated cargo:        (0.78, 0.91, 1.07, 1.20)
        14. Combination carrier:       (0.87, 0.96, 1.06, 1.14)

        Returns:
            Tuple of (exp_d1, exp_d2, exp_d3, exp_d4).
        """
        v_norm = (vessel_type or "").strip().lower()

        # 1. Ro-Ro categories
        if "vehicle" in v_norm or "car carrier" in v_norm:
            return 0.86, 0.94, 1.06, 1.16
        elif "passenger" in v_norm and ("ro" in v_norm or "ferry" in v_norm):
            return 0.76, 0.92, 1.14, 1.30
        elif "roro" in v_norm or "ro-ro" in v_norm:
            return 0.76, 0.89, 1.08, 1.27
        elif "cruise" in v_norm:
            return 0.87, 0.95, 1.06, 1.16

        # 2. Tanker
        elif "tanker" in v_norm:
            return 0.82, 0.93, 1.08, 1.28

        # 3. Bulk Carrier
        elif "bulk" in v_norm:
            return 0.86, 0.94, 1.06, 1.18

        # 4. Container Ship
        elif "container" in v_norm:
            return 0.83, 0.94, 1.07, 1.19

        # 5. General Cargo
        elif "general cargo" in v_norm:
            return 0.83, 0.94, 1.06, 1.19

        # 6. LNG Carrier
        elif "lng" in v_norm:
            if capacity >= 100000.0:
                return 0.89, 0.98, 1.06, 1.13
            else:
                return 0.78, 0.92, 1.10, 1.37

        # 7. Gas Carrier
        elif "gas" in v_norm:
            if capacity >= 65000.0:
                return 0.81, 0.91, 1.12, 1.44
            else:
                return 0.85, 0.95, 1.06, 1.25

        # 8. Refrigerated Cargo
        elif "refrigerated" in v_norm:
            return 0.78, 0.91, 1.07, 1.20

        # 9. Combination Carrier
        elif "combination" in v_norm:
            return 0.87, 0.96, 1.06, 1.14

        # Fallback default (bulk carrier standard)
        return 0.86, 0.94, 1.06, 1.18

    def evaluate_cii(
        self,
        co2_emissions: Optional[float] = None,
        cargo_tons: Optional[float] = None,
        distance_nm: Optional[float] = None,
        year: int = 2024,
        *,
        vessel_type: Optional[str] = None,
        vessel_dwt: Optional[float] = None,
        vessel_gt: Optional[float] = None,
        capacity: Optional[float] = None,
        capacity_type: Optional[str] = None,
        annual_distance_nm: Optional[float] = None,
        annual_co2_tons: Optional[float] = None,
        **kwargs: Any,
    ) -> ComplianceResult:
        """Calculate IMO Carbon Intensity Indicator (CII) and operational letter rating.

        Supports standard physical metrics (co2_emissions, cargo_tons, distance_nm)
        and dashboard/statutory reporting aliases (vessel_type, capacity, capacity_type, vessel_dwt, vessel_gt).

        Args:
            co2_emissions: Total operational direct CO2 emissions in metric tons.
            cargo_tons: Vessel deadweight capacity in metric tons (legacy alias for capacity).
            distance_nm: Total distance navigated in nautical miles.
            year: Compliance reporting calendar year (2023–2030).
            vessel_type: Optional vessel classification (e.g. 'Bulk Carrier', 'General Cargo', 'Ro-Ro').
            vessel_dwt: Alias for cargo_tons (metric tons DWT).
            vessel_gt: Gross tonnage for Ro-Ro / passenger categories.
            capacity: Generic vessel capacity value.
            capacity_type: Unit basis of capacity ('DWT' or 'GT').
            annual_distance_nm: Alias for distance_nm.
            annual_co2_tons: Alias for co2_emissions.
            **kwargs: Additional contextual metadata.

        Returns:
            ComplianceResult detailing letter rating (A-E), attained/required CII, ratio, status, and G2/G4 parameters.

        Raises:
            DataValidationError: If inputs are negative, zero, or missing.
            ComplianceError: If reporting year is outside statutory range (2023–2030).
        """
        resolved_co2 = annual_co2_tons if annual_co2_tons is not None else co2_emissions
        resolved_dist = annual_distance_nm if annual_distance_nm is not None else distance_nm

        # Resolve capacity value
        resolved_cap = capacity
        if resolved_cap is None:
            resolved_cap = vessel_gt if vessel_gt is not None else (vessel_dwt if vessel_dwt is not None else cargo_tons)

        resolved_cap_type = capacity_type
        if resolved_cap_type is None:
            if vessel_gt is not None:
                resolved_cap_type = "GT"
            elif vessel_dwt is not None or cargo_tons is not None:
                resolved_cap_type = "DWT"

        if resolved_co2 is None:
            raise DataValidationError(
                "Direct CO2 emissions must be provided (co2_emissions or annual_co2_tons)."
            )
        if resolved_cap is None:
            raise DataValidationError(
                "Vessel capacity must be provided (capacity, vessel_dwt, vessel_gt, or cargo_tons)."
            )
        if resolved_dist is None:
            raise DataValidationError(
                "Navigated distance must be provided (distance_nm or annual_distance_nm)."
            )

        if resolved_co2 < 0.0:
            raise DataValidationError(
                f"CO2 emissions cannot be negative: {resolved_co2}",
                details={"co2_emissions": resolved_co2},
            )
        if resolved_cap <= 0.0:
            raise DataValidationError(
                f"Vessel capacity must be positive: {resolved_cap}",
                details={"capacity": resolved_cap},
            )
        if resolved_dist <= 0.0:
            raise DataValidationError(
                f"Distance must be positive: {resolved_dist}",
                details={"distance_nm": resolved_dist},
            )
        if year not in CII_Z_FACTORS:
            raise ComplianceError(
                f"CII reporting year {year} is outside the supported statutory regulatory range (2023–2030) "
                f"under IMO Resolution MEPC.400(83).",
                details={"year": year, "supported_years": list(CII_Z_FACTORS.keys())},
            )

        # Resolve IMO Resolution MEPC.353(78) G2 Reference Line Branch
        vtype = vessel_type or "Bulk Carrier"
        a, c, eff_cap, statutory_metric = self.resolve_cii_reference_line(
            vessel_type=vtype,
            capacity=resolved_cap,
            capacity_type=resolved_cap_type,
        )

        # Baseline Reference CII (MEPC.353(78))
        baseline_cii = a * (eff_cap ** -c) if c != 0.0 else a

        # Attained CII (gCO2 / (capacity * nm)) on the statutory capacity basis
        attained_cii = (resolved_co2 * 1e6) / (resolved_cap * resolved_dist)

        # Statutory Annual Reduction Factor Z (IMO Resolution MEPC.400(83))
        z_factor = self.get_cii_z_factor(year)

        required_cii = baseline_cii * (1.0 - z_factor)
        ratio = attained_cii / required_cii if required_cii > 0.0 else 1.0

        # IMO Resolution MEPC.354(78) G4 Ship-Type-Specific Rating Boundaries
        d1, d2, d3, d4 = self.resolve_cii_rating_boundaries(vessel_type=vtype, capacity=resolved_cap)

        if ratio <= d1:
            rating = "A"
        elif ratio <= d2:
            rating = "B"
        elif ratio <= d3:
            rating = "C"
        elif ratio <= d4:
            rating = "D"
        else:
            rating = "E"

        is_compliant = rating in ("A", "B", "C")
        compliance_status = "COMPLIANT" if is_compliant else "NON_COMPLIANT"

        return ComplianceResult(
            cii_rating=rating,
            attained_cii=round(attained_cii, 4),
            required_cii=round(required_cii, 4),
            cii_ratio=round(ratio, 4),
            fueleu_pass=None,  # FuelEU is not evaluated during CII assessment
            fueleu_target=0.0,
            ghg_intensity=0.0,
            penalty_eur=0.0,
            compliance_status=compliance_status,
            compliance_score=round(ratio, 4),
            capacity_metric=statutory_metric,
            reference_line_a=round(a, 4),
            reference_line_c=round(c, 4),
            rating_boundaries=(d1, d2, d3, d4),
        )

    def evaluate_fueleu(
        self,
        ghg_intensity: float,
        energy_used_mj: float,
        year: int,
    ) -> ComplianceResult:
        """Evaluate penalty and compliance status against European Union FuelEU Maritime targets.

        Args:
            ghg_intensity: Attained Well-to-Wake GHG intensity (gCO2eq/MJ).
            energy_used_mj: Total energy consumed on covered voyages in megajoules.
            year: Compliance reporting calendar year.

        Returns:
            ComplianceResult specifying pass/fail status and financial penalty in EUR.

        Raises:
            DataValidationError: If inputs are negative.
            ComplianceError: If compliance year is invalid.
        """
        if ghg_intensity < 0.0:
            raise DataValidationError(
                f"GHG intensity cannot be negative: {ghg_intensity}",
                details={"ghg_intensity": ghg_intensity},
            )
        if energy_used_mj < 0.0:
            raise DataValidationError(
                f"Energy used cannot be negative: {energy_used_mj}",
                details={"energy_used_mj": energy_used_mj},
            )

        target_limit = self.get_fueleu_target(year)

        if ghg_intensity <= target_limit or energy_used_mj == 0.0:
            # Full compliance — zero penalty
            return ComplianceResult(
                cii_rating="N/A",
                attained_cii=0.0,
                required_cii=0.0,
                cii_ratio=0.0,
                fueleu_pass=True,
                fueleu_target=round(target_limit, 4),
                ghg_intensity=round(ghg_intensity, 4),
                penalty_eur=0.0,
                compliance_status="COMPLIANT",
                compliance_score=0.0,
            )

        # Statutory FuelEU Deficit & Financial Penalty (Article 23)
        ghg_deficit_g = (ghg_intensity - target_limit) * energy_used_mj
        # Tons of VLSFO equivalent energy deficit
        equivalent_fuel_deficit_tons = ghg_deficit_g / (ghg_intensity * VLSFO_ENERGY_DENSITY_MJ_PER_TON)
        penalty_eur = equivalent_fuel_deficit_tons * FUELEU_STATUTORY_PENALTY_RATE_EUR

        return ComplianceResult(
            cii_rating="N/A",
            attained_cii=0.0,
            required_cii=0.0,
            cii_ratio=0.0,
            fueleu_pass=False,
            fueleu_target=round(target_limit, 4),
            ghg_intensity=round(ghg_intensity, 4),
            penalty_eur=round(penalty_eur, 2),
            compliance_status="NON_COMPLIANT",
            compliance_score=round(penalty_eur, 2),
        )

    def assess_cii(
        self,
        co2_emissions: Optional[float] = None,
        cargo_tons: Optional[float] = None,
        distance_nm: Optional[float] = None,
        year: Optional[int] = None,
        *,
        vessel_type: Optional[str] = None,
        vessel_dwt: Optional[float] = None,
        annual_distance_nm: Optional[float] = None,
        annual_co2_tons: Optional[float] = None,
        capacity_type: Optional[str] = None,
    ) -> CIIResult:
        """Evaluate IMO Carbon Intensity Indicator returning a clean, decoupled CIIResult."""
        res = self.evaluate_cii(
            co2_emissions=co2_emissions,
            cargo_tons=cargo_tons,
            distance_nm=distance_nm,
            year=year,
            vessel_type=vessel_type,
            vessel_dwt=vessel_dwt,
            annual_distance_nm=annual_distance_nm,
            annual_co2_tons=annual_co2_tons,
            capacity_type=capacity_type,
        )
        return res.cii_result

    def assess_fueleu(
        self,
        ghg_intensity: float,
        energy_used_mj: float,
        year: int,
    ) -> FuelEUResult:
        """Evaluate EU FuelEU Maritime returning a clean, decoupled FuelEUResult."""
        res = self.evaluate_fueleu(
            ghg_intensity=ghg_intensity,
            energy_used_mj=energy_used_mj,
            year=year,
        )
        return res.fueleu_result

    def assess_compliance(
        self,
        cii_params: Optional[dict[str, Any]] = None,
        fueleu_params: Optional[dict[str, Any]] = None,
    ) -> ComplianceAssessment:
        """Evaluate both or either regulations returning a unified ComplianceAssessment container."""
        cii_res = self.assess_cii(**cii_params) if cii_params else None
        fueleu_res = self.assess_fueleu(**fueleu_params) if fueleu_params else None
        return ComplianceAssessment(cii=cii_res, fueleu=fueleu_res)
