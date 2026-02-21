"""Convert provisional_patent_specification.txt to PDF for USPTO upload.

Uses UbuntuMono (Unicode-capable) so math symbols render correctly.
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

_DIR  = os.path.dirname(os.path.abspath(__file__))
src   = os.path.join(_DIR, "provisional_patent_specification.txt")
dst   = os.path.join(_DIR, "provisional_patent_specification.pdf")

# Register Unicode-capable monospace font
FONT_PATH = "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf"
pdfmetrics.registerFont(TTFont("UbuntuMono", FONT_PATH))

doc = SimpleDocTemplate(
    dst,
    pagesize=letter,
    rightMargin=inch,
    leftMargin=inch,
    topMargin=inch,
    bottomMargin=inch,
    title="Provisional Patent Application — Nikolas Yoo",
)

style = ParagraphStyle(
    "mono",
    fontName="UbuntuMono",
    fontSize=9,
    leading=12,
    leftIndent=0,
    spaceAfter=0,
    spaceBefore=0,
    wordWrap="CJK",
)

# Substitutions for glyphs missing from UbuntuMono
_SUBS = {
    "→": "->",
    "←": "<-",
    "↔": "<->",
    "∇": "nabla",
    "‖": "||",
}

story = []
with open(src, encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        for ch, rep in _SUBS.items():
            line = line.replace(ch, rep)
        # Escape XML special chars required by Paragraph
        line = (line
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))
        story.append(Paragraph(line if line.strip() else "&nbsp;", style))

doc.build(story)
size_kb = os.path.getsize(dst) // 1024
print(f"Saved  → {dst}")
print(f"Size   → {size_kb} KB")
