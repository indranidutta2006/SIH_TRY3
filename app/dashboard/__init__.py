"""Dashboard module package for visual interaction, scenario testing, and analytics."""

from app.dashboard.page_compliance import render_compliance_page
from app.dashboard.page_optimization import render_optimization_page
from app.dashboard.page_overview import render_overview_page
from app.dashboard.page_prediction import render_prediction_page
from app.dashboard.page_scenarios import render_scenarios_page

__all__ = [
    "render_overview_page",
    "render_prediction_page",
    "render_optimization_page",
    "render_compliance_page",
    "render_scenarios_page",
]
