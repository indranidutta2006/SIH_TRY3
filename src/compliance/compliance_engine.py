"""Maritime statutory compliance engine for IMO CII and EU FuelEU Maritime regulations.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Fulfills contracts.interfaces.ComplianceEngine, evaluating Carbon Intensity Indicator (CII)
operational rating curves (A to E) and EU FuelEU Maritime statutory GHG deficit penalties.
"""

import logging
from typing import Any, Final, Optional

from contracts.exceptions import ComplianceError, DataValidationError
from contracts.interfaces import ComplianceEngine
from contracts.schemas import ComplianceResult

logger = logging.getLogger("maritime_system")

# FuelEU Maritime 2020 reference baseline (Regulation (EU) 2023/1805)
FUELEU_REFERENCE_GHG_INTENSITY: Final[float] = 91.16  # gCO2eq/MJ
VLSFO_ENERGY_DENSITY_MJ_PER_TON: Final[float] = 41000.0  # MJ/t
FUELEU_STATUTORY_PENALTY_RATE_EUR: Final[float] = 2400.0  # EUR/ton VLSFO equivalent


class MaritimeComplianceEngine(ComplianceEngine):
    """Statutory maritime environmental compliance accounting engine."""

    def __init__(self) -> None:
        """Initialize compliance engine."""
        self.logger = logger

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

    def evaluate_cii(
        self,
        co2_emissions: Optional[float] = None,
        cargo_tons: Optional[float] = None,
        distance_nm: Optional[float] = None,
        year: int = 2024,
        *,
        vessel_type: Optional[str] = None,
        vessel_dwt: Optional[float] = None,
        annual_distance_nm: Optional[float] = None,
        annual_co2_tons: Optional[float] = None,
        **kwargs: Any,
    ) -> ComplianceResult:
        """Calculate IMO Carbon Intensity Indicator (CII) and operational letter rating.

        Supports standard physical metrics (co2_emissions, cargo_tons, distance_nm)
        and dashboard/statutory reporting aliases (annual_co2_tons, vessel_dwt, annual_distance_nm, vessel_type).

        Args:
            co2_emissions: Total operational direct CO2 emissions in metric tons.
            cargo_tons: Vessel deadweight capacity in metric tons.
            distance_nm: Total distance navigated in nautical miles.
            year: Compliance reporting calendar year (>= 2020).
            vessel_type: Optional vessel classification (e.g. 'Bulk Carrier', 'Container', 'Tanker').
            vessel_dwt: Alias for cargo_tons (metric tons DWT).
            annual_distance_nm: Alias for distance_nm.
            annual_co2_tons: Alias for co2_emissions.
            **kwargs: Additional contextual metadata.

        Returns:
            ComplianceResult detailing letter rating (A-E), attained/required CII, ratio, and status.

        Raises:
            DataValidationError: If inputs are negative, zero, or missing.
            ComplianceError: If reporting year is invalid (< 2020).
        """
        resolved_co2 = annual_co2_tons if annual_co2_tons is not None else co2_emissions
        resolved_dwt = vessel_dwt if vessel_dwt is not None else cargo_tons
        resolved_dist = annual_distance_nm if annual_distance_nm is not None else distance_nm

        if resolved_co2 is None:
            raise DataValidationError(
                "Direct CO2 emissions must be provided (co2_emissions or annual_co2_tons)."
            )
        if resolved_dwt is None:
            raise DataValidationError(
                "Vessel capacity / deadweight must be provided (cargo_tons or vessel_dwt)."
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
        if resolved_dwt <= 0.0:
            raise DataValidationError(
                f"Cargo tons/capacity must be positive: {resolved_dwt}",
                details={"cargo_tons": resolved_dwt},
            )
        if resolved_dist <= 0.0:
            raise DataValidationError(
                f"Distance must be positive: {resolved_dist}",
                details={"distance_nm": resolved_dist},
            )
        if year < 2020:
            raise ComplianceError(
                f"CII reporting year must be >= 2020, got: {year}",
                details={"year": year},
            )

        # Attained CII (gCO2 / (dwt * nm))
        attained_cii = (resolved_co2 * 1e6) / (resolved_dwt * resolved_dist)

        # Statutory IMO Baseline CII (MEPC.337(76))
        # Form: Baseline_CII = a * (DWT ** -c)
        vessel_curves = {
            "bulk carrier": (4745.0, 0.622),
            "tanker": (5247.0, 0.610),
            "container": (1984.0, 0.489),
            "general cargo": (3196.0, 0.540),
            "roro": (1686.0, 0.388),
            "lng carrier": (9.827, 0.0),
        }
        curve_key = vessel_type.strip().lower() if vessel_type else ""
        if curve_key in vessel_curves:
            a, c = vessel_curves[curve_key]
        else:
            # Default to bulk carrier baseline reference curve (a=4745, c=0.622)
            a, c = 4745.0, 0.622

        baseline_cii = a * (resolved_dwt ** -c) if c != 0.0 else a

        # Statutory Annual Reduction Factor Z
        if year <= 2022:
            z_factor = 0.0
        elif year == 2023:
            z_factor = 0.05
        elif year == 2024:
            z_factor = 0.07
        elif year == 2025:
            z_factor = 0.09
        elif year == 2026:
            z_factor = 0.11
        else:
            # 2% annual reduction increment through 2030 (up to 21%)
            z_factor = min(0.21, 0.11 + 0.02 * (year - 2026))

        required_cii = baseline_cii * (1.0 - z_factor)
        ratio = attained_cii / required_cii if required_cii > 0.0 else 1.0

        # IMO Standard Rating Boundaries
        if ratio <= 0.83:
            rating = "A"
        elif ratio <= 0.94:
            rating = "B"
        elif ratio <= 1.06:
            rating = "C"
        elif ratio <= 1.19:
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
            fueleu_pass=is_compliant,
            fueleu_target=0.0,
            ghg_intensity=0.0,
            penalty_eur=0.0,
            compliance_status=compliance_status,
            compliance_score=round(ratio, 4),
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
