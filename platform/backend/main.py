"""
Unified Tensor System — FastAPI backend
All intelligence is local-only (no external API calls).

sys.path strategy:
  Project root and ecemath are inserted at position 0 (project root first),
  then the backend directory is appended for local router imports.
  After each batch of imports, project root is re-asserted at position 0
  to prevent ecemath/src from shadowing project-level packages.
"""
import sys
import os

_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
_ECEMATH = os.path.join(_PROJ_ROOT, 'ecemath')
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))


def _ensure_proj_root_first():
    """Re-insert project_root at position 0 so it wins over any ecemath/src shadowing."""
    for _p in [_ECEMATH, _PROJ_ROOT]:
        if _p in sys.path:
            sys.path.remove(_p)
        sys.path.insert(0, _p)
    if _BACKEND_DIR not in sys.path:
        sys.path.append(_BACKEND_DIR)


_ensure_proj_root_first()

from routers import regime, calendar  # noqa: E402
_ensure_proj_root_first()
from routers import codegen, hdv, physics  # noqa: E402
_ensure_proj_root_first()
from routers.circuit import router as circuit_router  # noqa: E402
_ensure_proj_root_first()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Unified Tensor System",
    version="1.0",
    description="Local-only intelligence platform: Koopman evaluators, BorrowVector classifier, "
                "calendar regime encoder, and HDV cross-domain discovery.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(regime.router, prefix="/api/v1/regime", tags=["regime"])
app.include_router(calendar.router, prefix="/api/v1/calendar", tags=["calendar"])
app.include_router(codegen.router, prefix="/api/v1/codegen", tags=["codegen"])
app.include_router(hdv.router, prefix="/api/v1/hdv", tags=["hdv"])
app.include_router(physics.router, prefix="/api/v1/physics", tags=["physics"])
app.include_router(circuit_router)


@app.get("/")
def root():
    return {"status": "ok", "platform": "Unified Tensor System", "version": "1.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}
