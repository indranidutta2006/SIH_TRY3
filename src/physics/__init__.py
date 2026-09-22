"""Naval architectural physics and alternative fuel conversion package."""

from src.physics.fuel_physics_engine import (
    LHV_MJ_PER_KG,
    MaritimeFuelPhysicsEngine,
    PhysicsCalculationError,
    normalize_fuel_name,
)

__all__ = [
    "MaritimeFuelPhysicsEngine",
    "PhysicsCalculationError",
    "LHV_MJ_PER_KG",
    "normalize_fuel_name",
]
