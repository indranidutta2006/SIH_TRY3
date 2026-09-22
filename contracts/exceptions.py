"""Custom exception hierarchy for the maritime fuel prediction and fleet optimization system.

Defines standardized domain exceptions raised across data, prediction, physics,
emissions, compliance, scheduling, optimization, and scenario evaluation layers.
"""

from typing import Any


class MaritimeSystemError(Exception):
    """Base exception class for all custom domain errors in the maritime system."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialize the base domain exception.

        Args:
            message: Human-readable error description.
            details: Optional contextual metadata relevant to the failure.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DataValidationError(MaritimeSystemError):
    """Raised when incoming raw records or feature matrices fail schema/boundary validations."""


class PredictionError(MaritimeSystemError):
    """Raised when model training, inference, feature extraction, or evaluation fails."""


class OptimizationError(MaritimeSystemError):
    """Raised when an optimization algorithm fails to converge, encounters invalid bounds, or faults."""


class SchedulerError(MaritimeSystemError):
    """Raised when voyage-to-vessel matching encounters impossible constraints or invalid schedules."""


class ComplianceError(MaritimeSystemError):
    """Raised when regulatory indices (IMO CII, FuelEU Maritime) fail evaluation or miss parameters."""


class IntegrationError(MaritimeSystemError):
    """Raised when interconnecting layer contracts or external pipeline adapters fail."""
