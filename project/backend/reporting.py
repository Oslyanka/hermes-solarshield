from __future__ import annotations

from pathlib import Path
import textwrap
from typing import Any

from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from database.db import create_report, utc_now
from genai.copilot import generate_sections

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
LOGO_PATH = BASE_DIR.parent / "frontend" / "public" / "hermes-logo.png"

SECTION_TITLES = {
    "Resultado da Visao Computacional",
    "Explicacao da IA Generativa",
    "Recomendacoes Operacionais",
    "Relatorio executivo",
}


def build_report_text(analysis: dict[str, Any], genai_sections: dict[str, str] | None = None) -> str:
    sections = genai_sections or generate_sections(analysis)
    recommendations = sections["mitigation_plan"]
    return f"""Hermes SolarShield - Relatorio Operacional
Data da analise: {analysis.get("created_at", utc_now())}

Resultado da Visao Computacional
- Nivel de risco: {analysis["risk_level"]}
- Score: {float(analysis["score"]):.3f}
- Confianca do modelo: {float(analysis["confidence"]):.3f}
- Explicacao curta: {analysis["explanation_short"]}

Explicacao da IA Generativa
Resumo tecnico: {sections["technical_summary"]}

Resumo para leigos: {sections["plain_summary"]}

Recomendacoes Operacionais
{recommendations}

Relatorio executivo
{sections["executive_report"]}
"""


def _write_pdf(path: Path, content: str) -> None:
    pdf = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    margin = 48

    def draw_header(page_number: int) -> float:
        pdf.setFillColorRGB(0.012, 0.024, 0.043)
        pdf.rect(0, height - 102, width, 102, fill=1, stroke=0)
        if LOGO_PATH.exists():
            pdf.setFillColorRGB(0.96, 0.98, 1.0)
            pdf.roundRect(margin, height - 86, 56, 56, 8, fill=1, stroke=0)
            pdf.drawImage(ImageReader(str(LOGO_PATH)), margin + 5, height - 81, width=46, height=46, mask="auto")
        pdf.setFillColorRGB(0.97, 0.98, 1.0)
        pdf.setFont("Helvetica-Bold", 17)
        pdf.drawString(margin + 62, height - 48, "Hermes SolarShield")
        pdf.setFont("Helvetica", 9)
        pdf.setFillColorRGB(0.63, 0.71, 0.80)
        pdf.drawString(margin + 62, height - 64, "Relatorio Operacional")
        pdf.setFillColorRGB(0.72, 0.76, 0.82)
        pdf.rect(margin, height - 104, width - (margin * 2), 2, fill=1, stroke=0)
        pdf.setFont("Helvetica", 8)
        pdf.setFillColorRGB(0.48, 0.54, 0.63)
        pdf.drawRightString(width - margin, 30, f"Hermes SolarShield | Pagina {page_number}")
        return height - 132

    page_number = 1
    y = draw_header(page_number)

    for paragraph in content.splitlines():
        text = paragraph.strip()
        if not text:
            y -= 8
            continue
        is_section = text in SECTION_TITLES or text.startswith("Hermes SolarShield -")
        if y < 72:
            pdf.showPage()
            page_number += 1
            y = draw_header(page_number)
        if text.startswith("Hermes SolarShield -"):
            pdf.setFillColorRGB(0.97, 0.98, 1.0)
            pdf.setFont("Helvetica-Bold", 13)
            pdf.drawString(margin, y, text)
            y -= 18
            continue
        if is_section:
            y -= 6
            pdf.setFillColorRGB(0.08, 0.11, 0.16)
            pdf.setFont("Helvetica-Bold", 10)
            pdf.drawString(margin, y, text)
            y -= 16
            continue
        pdf.setFillColorRGB(0.13, 0.16, 0.21)
        pdf.setFont("Helvetica", 9)
        for line in textwrap.wrap(text, width=96, break_long_words=False):
            if y < 58:
                pdf.showPage()
                page_number += 1
                y = draw_header(page_number)
                pdf.setFillColorRGB(0.13, 0.16, 0.21)
                pdf.setFont("Helvetica", 9)
            pdf.drawString(margin, y, line)
            y -= 13
    pdf.save()


def generate_report_files(analysis: dict[str, Any], genai_sections: dict[str, str] | None = None) -> dict[str, Any]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    content = build_report_text(analysis, genai_sections)
    stamp = utc_now().replace(":", "-")
    base_name = f"hermes_report_{analysis.get('id', 'demo')}_{stamp}"
    txt_path = REPORTS_DIR / f"{base_name}.txt"
    pdf_path = REPORTS_DIR / f"{base_name}.pdf"
    txt_path.write_text(content, encoding="utf-8")
    _write_pdf(pdf_path, content)
    return create_report(
        analysis_id=analysis.get("id"),
        title=f"Relatorio Hermes SolarShield #{analysis.get('id', 'demo')}",
        content=content,
        txt_path=str(txt_path),
        pdf_path=str(pdf_path),
    )
