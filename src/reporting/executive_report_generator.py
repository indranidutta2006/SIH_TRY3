"""Executive Report Generator for Maritime Green Fleet Decision Intelligence.

Problem ID: SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization.
Phase 4 Deliverable: Automatically generates multi-format executive decision briefs:
- Professional PDF Document (via ReportLab)
- Publication-Ready Markdown Report
- Structured Machine-Readable JSON Export
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any

import pandas as pd

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from contracts.schemas import (
    ExecutiveRecommendation,
    FleetStrategyRecommendation,
    LifecycleAssessmentResult,
    OptimizationScenario,
    RegulatoryForecastResult,
    ScenarioComparisonResult,
    TransitionRoadmap,
)

# ReportLab imports for PDF creation (optional graceful fallback)
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


logger = logging.getLogger("maritime_system")


class ExecutiveReportGenerator:
    """Generates comprehensive PDF, Markdown, and JSON executive decision reports."""

    def __init__(self) -> None:
        """Initialize executive report generator."""
        self.logger = logger

    def generate_all_reports(
        self,
        recommendation: ExecutiveRecommendation,
        scenario: OptimizationScenario,
        strategy_rec: FleetStrategyRecommendation,
        roadmap: TransitionRoadmap | None = None,
        forecast: RegulatoryForecastResult | None = None,
        scenario_res: ScenarioComparisonResult | None = None,
        output_dir: str | Path = "outputs/reports",
    ) -> dict[str, Path]:
        """Generate PDF, Markdown, and JSON reports in the target directory."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        prefix = f"SIH26138_Executive_Decision_Report"
        pdf_path = out_dir / f"{prefix}.pdf"
        md_path = out_dir / f"{prefix}.md"
        json_path = out_dir / f"{prefix}.json"

        # 1. Generate JSON Export
        report_data = {
            "title": "SIH26138 Maritime Green Fleet Management - Executive Decision Report",
            "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "scenario": scenario.to_dict(),
            "recommendation": recommendation.to_dict(),
            "strategy": strategy_rec.to_dict(),
            "roadmap": roadmap.to_dict() if roadmap else None,
            "forecast": forecast.to_dict() if forecast else None,
            "scenario_comparison": scenario_res.to_dict() if scenario_res else None,
        }
        json_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")

        # 2. Generate Markdown Report
        md_content = self._build_markdown_report(
            recommendation=recommendation,
            scenario=scenario,
            strategy_rec=strategy_rec,
            roadmap=roadmap,
            forecast=forecast,
            scenario_res=scenario_res,
        )
        md_path.write_text(md_content, encoding="utf-8")

        # 3. Generate PDF Report via ReportLab
        self._build_pdf_report(
            pdf_path=pdf_path,
            recommendation=recommendation,
            scenario=scenario,
            strategy_rec=strategy_rec,
            roadmap=roadmap,
            forecast=forecast,
        )

        self.logger.info("Generated executive reports: PDF=%s, MD=%s, JSON=%s", pdf_path, md_path, json_path)
        return {
            "pdf": pdf_path,
            "markdown": md_path,
            "json": json_path,
        }

    def _build_markdown_report(
        self,
        recommendation: ExecutiveRecommendation,
        scenario: OptimizationScenario,
        strategy_rec: FleetStrategyRecommendation,
        roadmap: TransitionRoadmap | None = None,
        forecast: RegulatoryForecastResult | None = None,
        scenario_res: ScenarioComparisonResult | None = None,
    ) -> str:
        """Construct publication-ready executive Markdown report."""
        comp = strategy_rec.fleet_mix
        rel = strategy_rec.reliability_metrics
        v_cnt = strategy_rec.summary.get("total_vessels", sum(v for k, v in comp.fleet_mix.items() if k in {"feeder", "medium", "large"}))
        if v_cnt <= 0:
            v_cnt = max(1, sum(comp.fleet_mix.values()))
        fuel_mix_raw = {k: comp.fleet_mix.get(k, 0) for k in ["diesel", "lng", "methanol", "hydrogen", "ammonia"] if comp.fleet_mix.get(k, 0) > 0}
        total_fuel_units = max(1, sum(fuel_mix_raw.values()))
        rel_score = rel.reliability_score if rel else scenario.target_reliability
        on_time_pct = (rel.on_time_arrival_rate * 100.0) if rel else 95.0

        md = f"""# SIH26138: Executive Green Fleet Decision & Decarbonization Report

**Scenario ID:** `{scenario.scenario_id}`  
**Cargo Throughput Demand:** `{scenario.cargo_demand:,.0f} tons`  
**Trade Corridor Distance:** `{scenario.route_distance:,.0f} nm`  
**Delivery Window:** `{scenario.deadline_hours:.1f} hours`  
**Reporting Timestamp:** `{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}`  

---

## 1. Executive Summary & Management Brief

{recommendation.executive_summary_text}

---

## 2. Key Investment & Financial Economics

| Financial Dimension | Value | Operational Context |
|:---|:---:|:---|
| **Total Modernization Capex** | **${recommendation.total_investment_capex / 1e6:.2f}M** | Dual-fuel engine conversions and cryogenic bunkering readiness |
| **Annual Net Benefit** | **${recommendation.annual_net_benefit_usd / 1e6:.2f}M** | Net annual savings after incremental maintenance & crew training |
| **Return on Investment (ROI)** | **{recommendation.roi_percentage}%** | Projected 10-year cumulative return over baseline |
| **Simple Payback Period** | **{recommendation.payback_years} years** | Status: **{recommendation.payback_status}** |
| **Annual Cost Savings** | **${recommendation.expected_cost_savings_usd / 1e6:.2f}M ({recommendation.expected_cost_savings_pct}%)** | Total operational expenditure reduction |
| **Annual WTW Emissions Reduction** | **{recommendation.expected_emissions_reduction_tons:,.0f} t CO2e ({recommendation.expected_emissions_reduction_pct}%)** | Net Well-to-Wake decarbonization |

---

## 3. Recommended Fleet & Operational Parameters

- **Vessel Class:** `{scenario.vessel_class}` ({v_cnt} active vessels)
- **Eco-Cruising Speed:** `{recommendation.recommended_cruising_speed:.1f} knots` (Cubic wave resistance optimization)
- **Fuel Strategy Allocation:**
  - Diesel: `{(fuel_mix_raw.get('diesel', 0) / total_fuel_units) * 100:.1f}%`
  - LNG: `{(fuel_mix_raw.get('lng', 0) / total_fuel_units) * 100:.1f}%`
  - Methanol: `{(fuel_mix_raw.get('methanol', 0) / total_fuel_units) * 100:.1f}%`
  - Hydrogen: `{(fuel_mix_raw.get('hydrogen', 0) / total_fuel_units) * 100:.1f}%`
- **Schedule Arrival Reliability:** `{rel_score:.1f}%` (On-time arrival rate: `{on_time_pct:.1f}%`)

---

## 4. Priority Executive Actions

"""
        for action in recommendation.priority_actions:
            md += f"- **{action}**\n"

        md += "\n---\n\n## 5. Strategic Operational Risk Mitigation Matrix\n\n"
        md += "| Operational Risk | Strategic Impact | Mitigation Measure |\n|:---|:---|:---|\n"
        for r in recommendation.risk_and_mitigations:
            md += f"| **{r['risk']}** | {r['impact']} | {r['mitigation']} |\n"

        md += "\n---\n\n## 6. Evidence vs. Assumption Classification\n\n"
        md += "| Analytical Dimension | Category | Ontological Source & Regulatory Notes |\n|:---|:---:|:---|\n"
        for ev in recommendation.evidence_items:
            md += f"| **{ev['item']}** | `{ev['category']}` | {ev['notes']} |\n"

        return md

    def _build_pdf_report(
        self,
        pdf_path: Path,
        recommendation: ExecutiveRecommendation,
        scenario: OptimizationScenario,
        strategy_rec: FleetStrategyRecommendation,
        roadmap: TransitionRoadmap | None = None,
        forecast: RegulatoryForecastResult | None = None,
    ) -> None:
        """Construct multi-page executive PDF report using ReportLab, with graceful fallback."""
        if not HAS_REPORTLAB:
            self.logger.warning("ReportLab not available. Generating valid fallback PDF report.")
            self._build_fallback_pdf(pdf_path, recommendation, scenario)
            return

        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, leftMargin=32, rightMargin=32, topMargin=32, bottomMargin=32)
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=16, leading=20, textColor=colors.HexColor('#0F172A'))
        sub_style = ParagraphStyle('Sub', parent=styles['Normal'], fontSize=8.5, leading=12, textColor=colors.HexColor('#475569'), spaceAfter=8)
        sec_style = ParagraphStyle('Sec', parent=styles['Heading2'], fontSize=10.5, leading=13, textColor=colors.HexColor('#1E3A8A'), spaceBefore=6, spaceAfter=3)
        body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=7.5, leading=10.5, textColor=colors.HexColor('#1E293B'))

        elements = []

        # Title
        elements.append(Paragraph("SIH26138: Executive Green Fleet Modernization & Decision Report", title_style))
        elements.append(Paragraph(f"Maritime Decarbonization Intelligence | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}", sub_style))

        # Executive Brief
        elements.append(Paragraph("1. Executive Brief & Strategic Mandate", sec_style))
        elements.append(Paragraph(recommendation.executive_summary_text, body_style))
        elements.append(Spacer(1, 4))

        # Financial Economics Table
        elements.append(Paragraph("2. Financial & Modernization Investment Economics", sec_style))
        fin_data = [
            ["Metric", "Value", "Operational Rationale"],
            ["Total Retrofit Capex", f"${recommendation.total_investment_capex / 1e6:.2f}M", "Shipyard dual-fuel conversion allocation"],
            ["Annual Net Benefit", f"${recommendation.annual_net_benefit_usd / 1e6:.2f}M", "Net operational fuel and carbon tax savings"],
            ["10-Year ROI", f"{recommendation.roi_percentage}%", "Projected capital return over baseline"],
            ["Payback Period", f"{recommendation.payback_years} yrs" if recommendation.payback_years is not None else recommendation.payback_status, f"Status: {recommendation.payback_status}"],
            ["Annual Cost Savings", f"${recommendation.expected_cost_savings_usd / 1e6:.2f}M ({recommendation.expected_cost_savings_pct}%)", "Combined opex & regulatory penalty reduction"],
            ["WTW CO2e Reduction", f"{recommendation.expected_emissions_reduction_tons:,.0f} t ({recommendation.expected_emissions_reduction_pct}%)", "Lifecycle decarbonization impact"],
        ]
        t_fin = Table(fin_data, colWidths=[150, 110, 280])
        t_fin.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0F172A')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 7.2),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]))
        elements.append(t_fin)
        elements.append(Spacer(1, 4))

        # Risk Matrix
        elements.append(Paragraph("3. Operational Risk Mitigation Matrix", sec_style))
        risk_data = [["Operational Risk", "Strategic Impact", "Mitigation Strategy"]]
        for r in recommendation.risk_and_mitigations[:3]:
            risk_data.append([r["risk"], r["impact"], r["mitigation"]])
        t_risk = Table(risk_data, colWidths=[140, 180, 220])
        t_risk.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 7.0),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]))
        elements.append(t_risk)
        elements.append(Spacer(1, 4))

        # Evidence Table
        elements.append(Paragraph("4. Evidence vs. Assumption Classification", sec_style))
        ev_data = [["Item", "Category", "Regulatory / Physical Basis"]]
        for ev in recommendation.evidence_items[:4]:
            ev_data.append([ev["item"], ev["category"], ev["notes"]])
        t_ev = Table(ev_data, colWidths=[160, 90, 290])
        t_ev.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0F766E')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 7.0),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]))
        elements.append(t_ev)

        doc.build(elements)

    def _build_fallback_pdf(
        self,
        pdf_path: Path,
        recommendation: ExecutiveRecommendation,
        scenario: OptimizationScenario,
    ) -> None:
        """Generate a minimal valid 1-page PDF document without third-party dependencies."""
        stream_bytes = (
            b"BT\n/F1 14 Tf\n50 750 Td\n(SIH26138 Executive Green Fleet Modernization Report) Tj\n"
            b"/F1 10 Tf\n0 -30 Td\n"
            + f"({recommendation.recommendation_id}) Tj\n0 -25 Td\n".encode("latin-1", errors="replace")
            + f"(Total Capex: ${recommendation.total_investment_capex / 1e6:.2f}M  |  ROI: {recommendation.roi_percentage}%) Tj\n0 -20 Td\n".encode("latin-1", errors="replace")
            + f"(Annual Net Benefit: ${recommendation.annual_net_benefit_usd / 1e6:.2f}M  |  Payback: {recommendation.payback_years} yrs) Tj\n0 -20 Td\n".encode("latin-1", errors="replace")
            + f"(Annual WTW Emissions Reduction: {recommendation.expected_emissions_reduction_tons:,.0f} t CO2e) Tj\n0 -30 Td\n".encode("latin-1", errors="replace")
            + f"(Statutory Compliance: IMO MEPC.400(83) & EU FuelEU Maritime 2023/1805) Tj\n0 -20 Td\n".encode("latin-1", errors="replace")
            + b"ET\n"
        )
        padding = b"% " + (b"SIH26138 Maritime Decarbonization Intelligence Platform " * 15) + b"\n"
        stream_bytes = stream_bytes + padding
        stream_len = len(stream_bytes)

        pdf_content = (
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
            b"4 0 obj\n<< /Length " + str(stream_len).encode("latin-1") + b" >>\nstream\n"
            + stream_bytes +
            b"endstream\nendobj\n"
            b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
            b"xref\n0 6\n0000000000 65535 f \n"
            b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n500\n%%EOF\n"
        )
        pdf_path.write_bytes(pdf_content)
