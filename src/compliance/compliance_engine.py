"""Maritime statutory compliance engine for IMO CII and EU FuelEU Maritime regulations.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Fulfills contracts.interfaces.ComplianceEngine, evaluating Carbon Intensity Indicator (CII)
operational rating curves (A to E) and EU FuelEU Maritime statutory GHG deficit penalties.
"""

import logging
from typing import Final

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
        co2_emissions: float,
        cargo_tons: float,
        distance_nm: float,
        year: int,
    ) -> ComplianceResult:
        """Calculate IMO Carbon Intensity Indicator (CII) and operational letter rating.

        Args:
            co2_emissions: Total operational direct CO2 emissions in metric tons.
            cargo_tons: Vessel deadweight capacity in metric tons.
            distance_nm: Total distance navigated in nautical miles.
            year: Compliance reporting calendar year (>= 2020).

        Returns:
            ComplianceResult detailing letter rating (A-E) and ratio score.

        Raises:
            DataValidationError: If inputs are negative or zero.
            ComplianceError: If required baselines cannot be resolved.
        """
        if co2_emissions < 0.0:
            raise DataValidationError(
                f"CO2 emissions cannot be negative: {co2_emissions}",
                details={"co2_emissions": co2_emissions},
            )
        if cargo_tons <= 0.0:
            raise DataValidationError(
                f"Cargo tons/capacity must be positive: {cargo_tons}",
                details={"cargo_tons": cargo_tons},
            )
        if distance_nm <= 0.0:
            raise DataValidationError(
                f"Distance must be positive: {distance_nm}",
                details={"distance_nm": distance_nm},
            )
        if year < 2020:
            raise ComplianceError(
                f"CII reporting year must be >= 2020, got: {year}",
                details={"year": year},
            )

        # Attained CII (gCO2 / (dwt * nm))
        attained_cii = (co2_emissions * 1e6) / (cargo_tons * distance_nm)

        # Statutory IMO Baseline CII (MEPC.337(76) bulk carrier reference curve: a=4745, c=0.622)
        baseline_cii = 4745.0 * (cargo_tons ** -0.622)

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

        return ComplianceResult(
            cii_rating=rating,
            fueleu_pass=is_compliant,
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
                fueleu_pass=True,
                compliance_score=0.0,
            )

        # Statutory FuelEU Deficit & Financial Penalty (Article 23)
        ghg_deficit_g = (ghg_intensity - target_limit) * energy_used_mj
        # Tons of VLSFO equivalent energy deficit
        equivalent_fuel_deficit_tons = ghg_deficit_g / (ghg_intensity * VLSFO_ENERGY_DENSITY_MJ_PER_TON)
        penalty_eur = equivalent_fuel_deficit_tons * FUELEU_STATUTORY_PENALTY_RATE_EUR

        return ComplianceResult(
            cii_rating="N/A",
            fueleu_pass=False,
            compliance_score=round(penalty_eur, 2),
        )
