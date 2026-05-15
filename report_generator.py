from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def generate_inspection_report(
    report_data: dict[str, Any],
    output_path: Path,
    visual_assets: list[dict[str, Any]] | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50,
        title="Aircraft Defect Detection Report",
        author=report_data.get("author", "OpenAI Codex"),
    )

    styles = _build_styles()
    story: list[Any] = []
    assets = visual_assets or []

    story.extend(_build_title_page(report_data, styles))
    story.append(PageBreak())
    story.extend(_build_summary_section(report_data, styles))
    story.extend(_build_results_section(report_data, styles))
    story.extend(_build_maintenance_section(report_data, styles))
    story.extend(_build_insights_section(report_data, styles))
    story.extend(_build_visualization_section(assets, styles, "Advanced Analytics"))
    story.extend(_build_visualization_section(assets, styles, "Visual Evidence"))

    doc.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)
    return output_path


def _build_styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=sample["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#0F172A"),
            spaceAfter=16,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=sample["Heading2"],
            fontName="Helvetica",
            fontSize=12,
            leading=18,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#475569"),
            spaceAfter=8,
        ),
        "heading": ParagraphStyle(
            "Heading",
            parent=sample["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=22,
            textColor=colors.HexColor("#0F172A"),
            spaceBefore=10,
            spaceAfter=8,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=16,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor("#1E293B"),
            spaceAfter=8,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=sample["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#475569"),
            spaceAfter=12,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#334155"),
        ),
        "section_intro": ParagraphStyle(
            "SectionIntro",
            parent=sample["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=16,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor("#334155"),
            spaceAfter=6,
        ),
    }


def _build_title_page(report_data: dict[str, Any], styles: dict[str, ParagraphStyle]) -> list[Any]:
    project_name = report_data.get("project_name", "Aircraft Defect Detection")
    created_on = report_data.get("timestamp", datetime.utcnow().isoformat())
    return [
        Spacer(1, 1.6 * inch),
        Paragraph(project_name, styles["title"]),
        Paragraph("AI-Powered Aircraft Inspection & Maintenance Report", styles["subtitle"]),
        Spacer(1, 0.45 * inch),
        Paragraph(f"Generated on: {created_on}", styles["small"]),
        Spacer(1, 2.8 * inch),
        Paragraph(
            "This report consolidates defect detection, segmentation quantification, maintenance intelligence, and risk interpretation for the uploaded aircraft component image.",
            styles["body"],
        ),
    ]


def _build_summary_section(report_data: dict[str, Any], styles: dict[str, ParagraphStyle]) -> list[Any]:
    return [
        Paragraph("Executive Summary", styles["heading"]),
        Paragraph(report_data.get("summary", "No summary available."), styles["body"]),
    ]


def _build_results_section(report_data: dict[str, Any], styles: dict[str, ParagraphStyle]) -> list[Any]:
    table_data = [
        ["Metric", "Value"],
        ["Timestamp", report_data.get("timestamp", "N/A")],
        ["Detected Defects", str(report_data.get("defects", 0))],
        ["Severity", report_data.get("severity", "N/A")],
        ["Risk Level", report_data.get("risk", "N/A")],
        ["Detected Component", report_data.get("component", "N/A")],
        ["Failure Type", report_data.get("failure_type", "N/A")],
        ["Surface Occupancy", f"{report_data.get('occupancy', 0.0):.2f}%"],
        ["Spread", f"{report_data.get('spread', 0.0):.4f}"],
        ["Defect Area", f"{report_data.get('area', 0.0):,.0f} px"],
        ["Average Confidence", f"{report_data.get('confidence', 0.0) * 100.0:.1f}%"],
    ]

    table = Table(table_data, colWidths=[2.2 * inch, 3.4 * inch], hAlign="LEFT")
    table.setStyle(_table_style("#0F766E"))
    return [
        Paragraph("Inspection Metrics", styles["heading"]),
        Paragraph(
            "The metrics below capture the current inspection state, quantified surface damage, and model confidence used by the maintenance intelligence engine.",
            styles["body"],
        ),
        table,
        Spacer(1, 0.18 * inch),
    ]


def _build_maintenance_section(report_data: dict[str, Any], styles: dict[str, ParagraphStyle]) -> list[Any]:
    table_data = [
        ["Maintenance Field", "Value"],
        ["Recommended Action", report_data.get("action", report_data.get("recommendation", "N/A"))],
        ["Schedule", report_data.get("schedule", "N/A")],
        ["Priority Level", report_data.get("priority", "N/A")],
        ["Maintenance Type", report_data.get("maintenance_type", "N/A")],
        ["Next Inspection Date", report_data.get("next_inspection_date", "N/A")],
    ]
    table = Table(table_data, colWidths=[2.2 * inch, 3.4 * inch], hAlign="LEFT")
    table.setStyle(_table_style("#1D4ED8"))

    content: list[Any] = [
        Paragraph("Maintenance Recommendation & Schedule", styles["heading"]),
        Paragraph(
            "This section converts the current defect severity into an actionable maintenance workflow for operations and engineering teams.",
            styles["body"],
        ),
        table,
        Spacer(1, 0.16 * inch),
    ]

    notes = report_data.get("notes", [])
    if notes:
        content.append(Paragraph("Maintenance Notes", styles["section_intro"]))
        for note in notes:
            content.append(Paragraph(f"• {note}", styles["body"]))
    return content


def _build_insights_section(report_data: dict[str, Any], styles: dict[str, ParagraphStyle]) -> list[Any]:
    insights = report_data.get("insights", {})
    return [
        Paragraph("Risk Interpretation", styles["heading"]),
        Paragraph(f"<b>Severity Reasoning:</b> {report_data.get('severity_reasoning', 'N/A')}", styles["body"]),
        Paragraph(f"<b>Root Cause Suggestion:</b> {insights.get('root_cause_suggestion', 'N/A')}", styles["body"]),
        Paragraph(f"<b>Damage Interpretation:</b> {insights.get('damage_interpretation', 'N/A')}", styles["body"]),
        Paragraph(f"<b>Risk Explanation:</b> {insights.get('risk_explanation', 'N/A')}", styles["body"]),
    ]


def _build_visualization_section(
    visual_assets: list[dict[str, Any]],
    styles: dict[str, ParagraphStyle],
    section_name: str,
) -> list[Any]:
    section_assets = [asset for asset in visual_assets if asset.get("section") == section_name]
    if not section_assets:
        return []

    content: list[Any] = [Paragraph(section_name, styles["heading"])]
    for asset in section_assets:
        title = asset.get("title", "Visualization")
        caption = asset.get("caption", "")
        image_path = asset.get("image_path")
        if not image_path:
            continue
        content.append(Paragraph(title, styles["section_intro"]))
        report_image = _create_reportlab_image(Path(image_path))
        if report_image is not None:
            content.append(report_image)
            content.append(Spacer(1, 0.06 * inch))
            if caption:
                content.append(Paragraph(caption, styles["caption"]))
            for explanation in asset.get("explanations", []):
                content.append(Paragraph(f"• {explanation}", styles["body"]))
            content.append(Spacer(1, 0.12 * inch))
    return content


def _table_style(header_color: str) -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_color)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#F8FAFC")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]
    )


def _create_reportlab_image(image_path: Path) -> RLImage | None:
    try:
        from PIL import Image

        image = Image.open(image_path)
        image.load()
        image_width, image_height = image.size
        max_width = 6.7 * inch
        max_height = 4.7 * inch
        scale = min(max_width / image_width, max_height / image_height, 1.0)
        rendered = RLImage(str(image_path))
        rendered.drawWidth = image_width * scale
        rendered.drawHeight = image_height * scale
        rendered.hAlign = "CENTER"
        return rendered
    except Exception:
        return None


def _draw_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(doc.leftMargin, 24, "Aircraft Defect Detection Report")
    canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 24, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()
