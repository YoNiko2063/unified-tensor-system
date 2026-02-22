#!/usr/bin/env bash
# Build whitepaper: generate figures, then compile LaTeX.
#
# Usage:
#   cd whitepaper && ./build.sh
#
# Requirements:
#   - Python with numpy, matplotlib, scipy
#   - pdflatex (texlive-base)

set -euo pipefail
cd "$(dirname "$0")"

echo "=== Step 1: Generate figures ==="
python figures.py

echo ""
echo "=== Step 2: Compile LaTeX ==="
if command -v pdflatex &>/dev/null; then
    pdflatex -interaction=nonstopmode whitepaper.tex
    pdflatex -interaction=nonstopmode whitepaper.tex   # second pass for refs
    echo ""
    echo "=== Done: whitepaper.pdf ==="
else
    echo "pdflatex not found. Figures generated in figures/."
    echo "Install texlive-base to compile: sudo apt install texlive-base texlive-latex-extra"
fi
