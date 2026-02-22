"""
Convert provisional_patent_specification.txt to USPTO-compliant DOCX.

Font:    Courier New, 12 pt
Margins: 1.25 inch all sides (37 CFR 1.52)
Spacing: Double-spaced body text (USPTO requirement)
"""
import os
from docx import Document
from docx.shared import Inches, Pt
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

_DIR = os.path.dirname(os.path.abspath(__file__))
src  = os.path.join(_DIR, "provisional_patent_specification.txt")
dst  = os.path.join(_DIR, "provisional_patent_specification.docx")

# ── Document setup ────────────────────────────────────────────────────────────
doc = Document()

# Page margins (1.25" all sides — 37 CFR 1.52)
MARGIN = Inches(1.25)
for section in doc.sections:
    section.top_margin    = MARGIN
    section.bottom_margin = MARGIN
    section.left_margin   = MARGIN
    section.right_margin  = MARGIN

# Default paragraph style
style = doc.styles["Normal"]
font  = style.font
font.name = "Courier New"
font.size = Pt(12)

# Set double spacing on default style
from docx.shared import Pt
from docx.oxml.ns import qn
pPr = style.element.get_or_add_pPr()
spacing = OxmlElement("w:spacing")
spacing.set(qn("w:line"),    "480")   # 480 twips = double-space (240 = single)
spacing.set(qn("w:lineRule"), "auto")
spacing.set(qn("w:before"), "0")
spacing.set(qn("w:after"),  "0")
pPr.append(spacing)

# ── Build document ────────────────────────────────────────────────────────────
with open(src, encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        p = doc.add_paragraph(line if line else "")
        # Ensure font carries through on each paragraph
        for run in p.runs:
            run.font.name = "Courier New"
            run.font.size = Pt(12)

doc.save(dst)
size_kb = os.path.getsize(dst) // 1024
print(f"Saved  -> {dst}")
print(f"Size   -> {size_kb} KB")
print("Done.")
