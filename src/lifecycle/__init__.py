"""Maritime Lifecycle Assessment (LCA) package.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 4 Deliverable: Evaluates Well-to-Wake (WTW), Well-to-Tank (WTT), and Tank-to-Wake (TTW) emissions
with granular feedstock pathways (Grey/Blue/Green/Bio/E-fuels).
"""

from src.lifecycle.lifecycle_assessment_engine import (
    DEFAULT_LIFECYCLE_PROFILES,
    MaritimeLifecycleAssessmentEngine,
)

__all__ = [
    "DEFAULT_LIFECYCLE_PROFILES",
    "MaritimeLifecycleAssessmentEngine",
]
