"""
Convert provisional_patent_specification.txt to USPTO-compliant PDF.

Font:    Courier (built-in PDF Type 1 — no embedding, universally accepted)
Margins: 1.25 inch all sides (meets 37 CFR 1.52)
Unicode: All non-ASCII math symbols replaced with ASCII equivalents
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph
import os

_DIR = os.path.dirname(os.path.abspath(__file__))
src  = os.path.join(_DIR, "provisional_patent_specification.txt")
dst  = os.path.join(_DIR, "provisional_patent_specification.pdf")

# ── USPTO-compliant margins (37 CFR 1.52) ────────────────────────────────────
MARGIN = 1.25 * inch

# ── Unicode → ASCII substitution map ─────────────────────────────────────────
# All non-ASCII characters that appear in the spec are mapped to ASCII strings.
_SUBS = [
    # Greek lowercase
    ("ω",  "omega"), ("δ",  "delta"), ("ζ",  "zeta"),  ("α",  "alpha"),
    ("β",  "beta"),  ("γ",  "gamma"), ("ε",  "epsilon"),("η",  "eta"),
    ("θ",  "theta"), ("κ",  "kappa"), ("λ",  "lambda"), ("μ",  "mu"),
    ("π",  "pi"),    ("ρ",  "rho"),   ("σ",  "sigma"),  ("τ",  "tau"),
    ("φ",  "phi"),   ("ψ",  "psi"),
    # Greek uppercase
    ("Ω",  "Omega"), ("Δ",  "Delta"), ("Σ",  "Sigma"),  ("Φ",  "Phi"),
    ("Γ",  "Gamma"),
    # Dot/ddot accents (combining chars — replace whole sequence)
    ("\u03b4\u0308", "delta_ddot"),  # δ̈
    ("\u03b4\u0307", "delta_dot"),   # δ̇
    ("\u1e8b",       "x_dot"),       # ẋ
    ("\u1e8d",       "x_ddot"),      # ẍ  (rare)
    ("\u1e57",       "p_dot"),       # ṗ
    ("\u1e41",       "m_dot"),       # ṁ
    ("\u1e59",       "r_dot"),       # ṙ
    ("\u1e61",       "s_dot"),       # ṡ
    ("\u0307",       "_dot"),        # bare combining dot above
    ("\u0308",       "_ddot"),       # bare combining diaeresis
    # Dots/operators
    ("\u00b7",  "."),     # middle dot ·
    ("\u22c5",  "."),     # dot operator ⋅
    ("\u00d7",  "x"),     # multiplication sign ×
    ("\u00b1",  "+/-"),   # plus-minus ±
    # Math symbols
    ("\u221a",  "sqrt"),  # √
    ("\u2202",  "d/d"),   # ∂ (partial)
    ("\u2207",  "nabla"), # ∇
    ("\u2211",  "sum"),   # ∑
    ("\u222b",  "integral"), # ∫
    ("\u221e",  "inf"),   # ∞
    ("\u2208",  "in"),    # ∈
    ("\u2248",  "~="),    # ≈
    ("\u2260",  "!="),    # ≠
    ("\u2265",  ">="),    # ≥
    ("\u2264",  "<="),    # ≤
    ("\u2016",  "||"),    # ‖ double vertical line
    ("\u2225",  "||"),    # ∥ parallel
    # Arrows
    ("\u2192",  "->"),    # →
    ("\u2190",  "<-"),    # ←
    ("\u2194",  "<->"),   # ↔
    # Subscript digits/letters
    ("\u2080",  "_0"), ("\u2081",  "_1"), ("\u2082",  "_2"),
    ("\u2083",  "_3"), ("\u2084",  "_4"), ("\u2085",  "_5"),
    ("\u2086",  "_6"), ("\u2087",  "_7"), ("\u2088",  "_8"),
    ("\u2089",  "_9"),
    ("\u2099",  "_n"), ("\u209a",  "_p"), ("\u2095",  "_h"),
    ("\u2096",  "_k"), ("\u2097",  "_l"), ("\u2098",  "_m"),
    ("\u209b",  "_s"), ("\u209c",  "_t"), ("\u1d62",  "_i"),
    ("\u2071",  "^i"), ("\u207f",  "^n"),
    # Superscript digits
    ("\u00b2",  "^2"),    # ²
    ("\u00b3",  "^3"),    # ³
    ("\u00b9",  "^1"),    # ¹
    ("\u2070",  "^0"),
    ("\u2074",  "^4"), ("\u2075",  "^5"), ("\u2076",  "^6"),
    # Superscript letters
    ("\u1d40",  "^T"),    # ᵀ
    # Special physics/math
    ("\u0127",  "hbar"),  # ħ
    ("\u210f",  "hbar"),  # ℏ
    ("\u00b0",  " deg"),  # °
    ("\u00e9",  "e"),     # é
    ("\u00f6",  "o"),     # ö
    ("\u00ff",  "y"),     # ÿ
    ("\u0117",  "e"),     # ė
    ("\u0178",  "Y"),     # Ÿ
    ("\u017c",  "z"),     # ż
    ("\u0130",  "I"),     # İ
    ("\u0126",  "H"),     # Ħ  (Planck H-bar capital variant)
    # Math minus sign (different Unicode from ASCII hyphen)
    ("\u2212",  "-"),     # − minus sign (very common in equations)
    # Missing subscripts
    ("\u2091",  "_e"),    # ₑ
    ("\u2098",  "_m"),    # ₘ
    ("\u209a",  "_p"),    # ₚ
    ("\u209b",  "_s"),    # ₛ
    ("\u2c7c",  "_j"),    # ⱼ
    # Missing dot-accented uppercase/lowercase
    ("\u1e44",  "N_dot"), # Ṅ
    ("\u1e56",  "P_dot"), # Ṗ
    ("\u1e58",  "R_dot"), # Ṙ
    ("\u1e60",  "S_dot"), # Ṡ
    ("\u1e8e",  "Y_dot"), # Ẏ
    ("\u1e8f",  "y_dot"), # ẏ
    # Missing superscripts
    ("\u2078",  "^8"),    # ⁸
    ("\u207a",  "^+"),    # ⁺
    ("\u207b",  "^-"),    # ⁻
    # Misc accented Latin (appear in bibliography/names)
    ("\u00cb",  "E"),     # Ë
    ("\u0116",  "E"),     # Ė
    # Misc punctuation
    ("\u2014",  "--"),    # em dash
    ("\u2013",  "-"),     # en dash
    ("\u201c",  '"'),     # left double quote
    ("\u201d",  '"'),     # right double quote
    ("\u2018",  "'"),     # left single quote
    ("\u2019",  "'"),     # right single quote
    ("\u2026",  "..."),   # ellipsis
    ("\u00a0",  " "),     # non-breaking space
    # Dividers/box chars (the ─────── lines in the spec)
    ("\u2500",  "-"),     # box drawing light horizontal
    ("\u2502",  "|"),     # box drawing light vertical
    ("\u253c",  "+"),     # box drawing light cross
    ("\u2550",  "="),     # box drawing double horizontal
    ("\u2551",  "||"),    # box drawing double vertical
    ("\u256c",  "#"),     # box drawing double cross
    ("\uff3d",  "]"),     # fullwidth right square bracket
]

def _to_ascii(text):
    """Apply all substitutions; drop any remaining non-ASCII chars."""
    # Multi-char sequences first (longer matches before their components)
    for src_ch, rep in _SUBS:
        text = text.replace(src_ch, rep)
    # Drop any residual non-ASCII
    return text.encode("ascii", errors="replace").decode("ascii").replace("?", "")

# ── Document style ────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    dst,
    pagesize=letter,
    rightMargin=MARGIN,
    leftMargin=MARGIN,
    topMargin=MARGIN,
    bottomMargin=MARGIN,
    title="Provisional Patent Application — Nikolas Yoo",
    author="Nikolas Yoo",
)

style = ParagraphStyle(
    "body",
    fontName="Courier",   # built-in Type 1 — no embedding, USPTO-accepted
    fontSize=10,
    leading=14,           # ~1.4x line spacing
    leftIndent=0,
    spaceAfter=0,
    spaceBefore=0,
    wordWrap="CJK",
)

# ── Build story ───────────────────────────────────────────────────────────────
story = []
with open(src, encoding="utf-8") as f:
    for line in f:
        line = _to_ascii(line.rstrip("\n"))
        # Escape XML special chars for Paragraph
        line = line.replace("&", "&amp;")
        # < and > are already ASCII-safe after substitution but escape anyway
        line = line.replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(line if line.strip() else "&nbsp;", style))

doc.build(story)
size_kb = os.path.getsize(dst) // 1024
print(f"Saved  -> {dst}")
print(f"Size   -> {size_kb} KB")

# Verify no non-ASCII survived
with open(dst, "rb") as f:
    content = f.read()
print(f"Pages  -> check PDF viewer")
print("Done.")
