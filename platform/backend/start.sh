#!/bin/bash
# Start the Unified Tensor System FastAPI backend
# Requires: conda env 'tensor' with fastapi + uvicorn installed
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ECEMATH="${PROJ_ROOT}/ecemath"

# Pre-set PYTHONPATH so project root resolves before backend dir
# This avoids the '' (cwd) ambiguity when uvicorn loads main.py
export PYTHONPATH="${PROJ_ROOT}:${ECEMATH}:${PYTHONPATH:-}"

cd "${SCRIPT_DIR}"
conda run -n tensor --no-capture-output \
    uvicorn main:app --reload --port 8000 --host 0.0.0.0
