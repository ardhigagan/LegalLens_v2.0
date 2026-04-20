"""
report.py — LegalLens AI v2: PDF Report Export
Generates a professional PDF risk report using ReportLab.
"""

import io
from datetime import datetime

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("ReportLab not installed. Run: pip install reportlab")


# Brand colors
COLOR_PRIMARY   = colors.HexColor("#4A90E2")
COLOR_DANGER    = colors.HexColor("#C62828")
COLOR_SAFE      = colors.HexColor("#2E7D32")
COLOR_WARN      = colors.HexColor("#F57F17")
COLOR_BG_RISK   = colors.HexColor("#FFEBEE")
COLOR_BG_SAFE   = colors.HexColor("#E8F5E9")
COLOR_LIGHT_GRAY= colors.HexColor("#F5F5F5")
COLOR_MID_GRAY  = colors.HexColor("#BDBDBD")
COLOR_DARK      = colors.HexColor("#212121")
COLOR_MUTED     = colors.HexColor("#757575")


def _risk_color(score: float) -> colors.Color:
    if score >= 0.80:
        return COLOR_DANGER
    elif score >= 0.65:
        return COLOR_WARN
    return colors.HexColor("#1565C0")


def generate_pdf_report(
    filename: str,
    summary: str,
    risks: list[dict],
    document_name: str = "Uploaded Contract",
) -> bytes:
    """
    Generates a PDF risk report and returns it as bytes.

    Args:
        filename: Logical filename (used in header).
        summary: Executive summary text.
        risks: List of risk dicts from analyze_document().
        document_name: Display name of the contract.

    Returns:
        PDF bytes ready for st.download_button.
    """
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("ReportLab is not installed. Run: pip install reportlab")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    story = []

    # --- STYLES ---
    title_style = ParagraphStyle(
        "Title", parent=styles["Title"],
        fontSize=22, textColor=COLOR_PRIMARY,
        spaceAfter=4, fontName="Helvetica-Bold",
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"],
        fontSize=11, textColor=COLOR_MUTED,
        spaceAfter=2,
    )
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading2"],
        fontSize=13, textColor=COLOR_DARK,
        fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6,
        borderPad=4,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10, textColor=COLOR_DARK,
        leading=15, alignment=TA_JUSTIFY,
    )
    snippet_style = ParagraphStyle(
        "Snippet", parent=styles["Normal"],
        fontSize=9, textColor=COLOR_MUTED,
        leading=13, leftIndent=10, fontName="Helvetica-Oblique",
    )
    caption_style = ParagraphStyle(
        "Caption", parent=styles["Normal"],
        fontSize=8, textColor=COLOR_MUTED,
    )

    # --- HEADER ---
    story.append(Paragraph("LegalLens AI", title_style))
    story.append(Paragraph("Contract Risk Analysis Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=COLOR_PRIMARY, spaceAfter=6))

    # Meta info table
    now = datetime.now().strftime("%d %B %Y, %H:%M")
    meta_data = [
        ["Document:", document_name],
        ["Generated:", now],
        ["Risks Found:", str(len(risks)) if risks else "None"],
        ["Risk Level:", _overall_risk_label(risks)],
    ]
    meta_table = Table(meta_data, colWidths=[3.5 * cm, 13 * cm])
    meta_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), COLOR_MUTED),
        ("TEXTCOLOR", (1, 0), (1, -1), COLOR_DARK),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 16))

    # --- EXECUTIVE SUMMARY ---
    story.append(Paragraph("Executive Summary", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=COLOR_MID_GRAY, spaceAfter=8))
    story.append(Paragraph(summary or "No summary available.", body_style))
    story.append(Spacer(1, 16))

    # --- RISK OVERVIEW TABLE ---
    story.append(Paragraph("Risk Overview", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=COLOR_MID_GRAY, spaceAfter=8))

    if not risks:
        story.append(Paragraph(
            "✓ No high-risk clauses were detected in this document.",
            ParagraphStyle("Safe", parent=body_style, textColor=COLOR_SAFE, fontName="Helvetica-Bold"),
        ))
    else:
        # Summary table
        table_data = [["#", "Risk Type", "Confidence", "Severity"]]
        for i, risk in enumerate(risks, 1):
            score = risk["score"]
            severity = "High" if score >= 0.80 else ("Medium" if score >= 0.65 else "Low")
            table_data.append([
                str(i),
                risk["type"],
                f"{int(score * 100)}%",
                severity,
            ])

        risk_table = Table(table_data, colWidths=[1 * cm, 9.5 * cm, 3 * cm, 3 * cm])
        risk_table.setStyle(TableStyle([
            # Header
            ("BACKGROUND", (0, 0), (-1, 0), COLOR_PRIMARY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            # Body
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOR_LIGHT_GRAY]),
            ("ALIGN", (2, 1), (3, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.25, COLOR_MID_GRAY),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(risk_table)

    story.append(Spacer(1, 20))

    # --- DETAILED RISK FINDINGS ---
    if risks:
        story.append(Paragraph("Detailed Risk Findings", section_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=COLOR_MID_GRAY, spaceAfter=8))

        for i, risk in enumerate(risks, 1):
            score = risk["score"]
            rc = _risk_color(score)
            severity = "High" if score >= 0.80 else ("Medium" if score >= 0.65 else "Low")

            risk_block = []

            # Risk title row
            title_data = [[
                Paragraph(f"{i}. {risk['type']}", ParagraphStyle(
                    "RiskTitle", parent=body_style,
                    fontName="Helvetica-Bold", fontSize=10, textColor=rc,
                )),
                Paragraph(f"Confidence: {int(score*100)}% | {severity}", ParagraphStyle(
                    "RiskMeta", parent=caption_style, alignment=1,
                )),
            ]]
            title_table = Table(title_data, colWidths=[11 * cm, 5.5 * cm])
            title_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), COLOR_BG_RISK),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("LINEBELOW", (0, 0), (-1, -1), 0.5, rc),
            ]))
            risk_block.append(title_table)

            # Snippet
            snippet = risk.get("text_snippet", "")
            if snippet:
                risk_block.append(Spacer(1, 4))
                risk_block.append(Paragraph(f'"{snippet}"', snippet_style))

            risk_block.append(Spacer(1, 10))
            story.append(KeepTogether(risk_block))

    # --- DISCLAIMER ---
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=COLOR_MID_GRAY, spaceAfter=6))
    story.append(Paragraph(
        "Disclaimer: This report is generated by an AI model and is intended for informational purposes only. "
        "It does not constitute legal advice. Always consult a qualified legal professional before making decisions "
        "based on this analysis.",
        ParagraphStyle("Disclaimer", parent=caption_style, alignment=TA_CENTER, textColor=COLOR_MUTED),
    ))

    doc.build(story)
    return buffer.getvalue()


def _overall_risk_label(risks: list[dict]) -> str:
    if not risks:
        return "Clean"
    max_score = max(r["score"] for r in risks)
    count = len(risks)
    if max_score >= 0.80 or count >= 5:
        return "HIGH RISK"
    elif max_score >= 0.65 or count >= 3:
        return "MEDIUM RISK"
    return "LOW RISK"
