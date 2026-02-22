"""
POST /api/v1/hdv/encode    body: {text, domain}
GET  /api/v1/hdv/universals -> cross-domain universals
"""
from typing import Any, Dict, List, Optional
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tensor.integrated_hdv import IntegratedHDVSystem
from tensor.cross_dimensional_discovery import CrossDimensionalDiscovery

router = APIRouter()

VALID_DOMAINS = ["math", "physics", "behavioral", "language", "visual"]

_hdv: IntegratedHDVSystem | None = None
_discovery: CrossDimensionalDiscovery | None = None


def _get_hdv() -> IntegratedHDVSystem:
    global _hdv
    if _hdv is None:
        _hdv = IntegratedHDVSystem()
    return _hdv


def _get_discovery() -> CrossDimensionalDiscovery:
    global _discovery
    if _discovery is None:
        _discovery = CrossDimensionalDiscovery(_get_hdv())
    return _discovery


# In-memory store of encoded vectors for PCA projection
_encoded_vectors: List[Dict[str, Any]] = []


def _pca_2d(vec: np.ndarray, reference_pool: List[np.ndarray]) -> List[float]:
    """Project vec into 2D using PCA over a pool + vec itself."""
    pool = reference_pool + [vec]
    X = np.stack(pool, axis=0)  # (n, d)

    # Center
    mean = X.mean(axis=0)
    X_c = X - mean

    # Thin SVD — use only the last vector's projection
    try:
        from numpy.linalg import svd
        # Use a subset of dims to keep it fast (first 512 dims)
        X_sub = X_c[:, :512]
        U, S, Vt = svd(X_sub, full_matrices=False)
        coords = U[:, :2] * S[:2]  # (n, 2)
        last = coords[-1].tolist()
    except Exception:
        last = [0.0, 0.0]

    return last


class EncodeRequest(BaseModel):
    text: str
    domain: str


class EncodeResponse(BaseModel):
    pca_2d: List[float]
    domain: str
    norm: float
    dim: int


class UniversalEntry(BaseModel):
    dimension: int
    domains: List[str]
    pattern: str
    confidence: float


class UniversalsResponse(BaseModel):
    universals: List[UniversalEntry]
    count: int
    domains_active: List[str]


@router.post("/encode", response_model=EncodeResponse)
def encode_hdv(req: EncodeRequest) -> EncodeResponse:
    """Encode text into a hyperdimensional vector and return its 2D PCA projection."""
    if req.domain not in VALID_DOMAINS:
        raise HTTPException(
            status_code=422,
            detail=f"domain must be one of {VALID_DOMAINS}, got {req.domain!r}",
        )
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")

    hdv = _get_hdv()
    vec = hdv.structural_encode(req.text, req.domain)

    # Accumulate a small reference pool (max 20 vectors) for stable PCA
    _encoded_vectors.append({"domain": req.domain, "vec": vec})
    if len(_encoded_vectors) > 20:
        _encoded_vectors.pop(0)

    ref_pool = [e["vec"] for e in _encoded_vectors[:-1]]
    pca2d = _pca_2d(vec, ref_pool) if len(ref_pool) >= 2 else [0.0, 0.0]

    return EncodeResponse(
        pca_2d=pca2d,
        domain=req.domain,
        norm=float(np.linalg.norm(vec)),
        dim=int(vec.shape[0]),
    )


@router.get("/universals", response_model=UniversalsResponse)
def get_universals() -> UniversalsResponse:
    """Return current cross-domain universal patterns discovered."""
    disc = _get_discovery()
    raw = disc.find_universals()

    entries: List[UniversalEntry] = []
    for u in raw:
        entries.append(UniversalEntry(
            dimension=int(u.get("dimension", 0)),
            domains=list(u.get("domains", [])),
            pattern=str(u.get("pattern", "")),
            confidence=float(u.get("confidence", 0.0)),
        ))

    # Report which domains have been encoded so far
    domains_active = list({e["domain"] for e in _encoded_vectors})

    return UniversalsResponse(
        universals=entries,
        count=len(entries),
        domains_active=domains_active,
    )
