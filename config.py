"""Centralized application and experiment configuration module.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Structures all runtime hyperparameters, input/output filesystem paths, numerical seeds,
optimizer defaults, and visualization parameters using type-annotated dataclasses.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from contracts.constants import OptimizerType

# Determine base project root relative to this configuration file
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent


@dataclass(frozen=True, slots=True)
class DataPaths:
    """Filesystem locations for persistent data input sources."""

    base_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    raw_data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw")
    processed_data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed")
    sample_voyages_file: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "raw" / "voyages_sample.csv"
    )


@dataclass(frozen=True, slots=True)
class OutputPaths:
    """Filesystem locations for logging, model serialization, and visual artifacts."""

    base_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")
    logs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs" / "logs")
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs" / "models")
    reports_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs" / "reports")
    figures_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs" / "figures")


@dataclass(frozen=True, slots=True)
class OptimizerDefaults:
    """Default hyperparameter configuration for numerical and quantum-inspired optimizers."""

    default_optimizer: str = OptimizerType.QPSO.value
    max_iterations: int = 200
    population_size: int = 50
    convergence_tolerance: float = 1e-5
    inertia_weight: float = 0.729
    cognitive_coefficient: float = 1.49445
    social_coefficient: float = 1.49445
    contraction_expansion_coefficient: float = 0.75


@dataclass(frozen=True, slots=True)
class DashboardDefaults:
    """Default configuration parameters for web presentation and dashboard layers."""

    app_title: str = "Quantum Green Fleet Navigator"
    host: str = "0.0.0.0"
    port: int = 8501
    theme: str = "dark"
    refresh_rate_seconds: int = 5


@dataclass(frozen=True, slots=True)
class SystemConfig:
    """Central configuration container for the maritime optimization project."""

    random_seed: int = 42
    cross_validation_folds: int = 5
    logging_level: str = "INFO"
    data_paths: DataPaths = field(default_factory=DataPaths)
    output_paths: OutputPaths = field(default_factory=OutputPaths)
    optimizer_defaults: OptimizerDefaults = field(default_factory=OptimizerDefaults)
    dashboard_defaults: DashboardDefaults = field(default_factory=DashboardDefaults)


def get_default_config() -> SystemConfig:
    """Instantiate and return the canonical system configuration.

    Returns:
        SystemConfig instance populated with default architectural parameters.
    """
    return SystemConfig()
