"""
POST /api/v1/codegen/generate  body: {domain, operation, parameters}
GET  /api/v1/codegen/templates -> list all registered templates
"""
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from codegen.pipeline import CodeGenPipeline
from codegen.intent_spec import IntentSpec

router = APIRouter()

_pipeline: CodeGenPipeline | None = None

D_SEP = 0.43  # borrow-vector separation boundary


def _get_pipeline() -> CodeGenPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = CodeGenPipeline()
    return _pipeline


class GenerateRequest(BaseModel):
    domain: str
    operation: str
    parameters: Dict[str, Any] = {}


class GenerateResponse(BaseModel):
    rust_source: str
    e_borrow: float
    predicted_compiles: bool
    template_name: str
    borrow_vector: List[float]
    probability: float
    success: bool
    error: Optional[str] = None


class TemplateInfo(BaseModel):
    name: str
    domain: str
    operation: str
    borrow_profile: str
    e_borrow: float
    description: str


@router.post("/generate", response_model=GenerateResponse)
def generate_code(req: GenerateRequest) -> GenerateResponse:
    """Run the CodeGen pipeline for the given domain + operation + parameters."""
    pipeline = _get_pipeline()

    spec = IntentSpec(
        domain=req.domain,
        operation=req.operation,
        parameters=req.parameters,
    )

    result = pipeline.generate(spec)

    # Pull metrics from whichever gate ran
    gate = result.post_gate or result.pre_gate
    e_borrow = float(gate.e_borrow) if gate else 0.0
    predicted = bool(gate.predicted_compile) if gate else True
    bv = list(gate.borrow_vector) if gate else [0.0] * 6
    prob = float(gate.probability) if gate else 1.0

    rust_source = result.rust_source or ""
    error_msg = None
    if not result.success and result.post_gate and result.post_gate.compile_stderr:
        # Trim to first 300 chars so the API stays readable
        error_msg = result.post_gate.compile_stderr[:300]

    return GenerateResponse(
        rust_source=rust_source,
        e_borrow=e_borrow,
        predicted_compiles=predicted,
        template_name=result.template_name or "",
        borrow_vector=bv,
        probability=prob,
        success=result.success,
        error=error_msg,
    )


@router.get("/templates", response_model=List[TemplateInfo])
def list_templates() -> List[TemplateInfo]:
    """Return all registered Rust code templates."""
    pipeline = _get_pipeline()
    templates = pipeline.registry.all_templates()
    return [
        TemplateInfo(
            name=t.name,
            domain=t.domain,
            operation=t.operation,
            borrow_profile=t.borrow_profile.value if hasattr(t.borrow_profile, 'value') else str(t.borrow_profile),
            e_borrow=float(sum(w * b for w, b in zip(
                [0.25, 0.18, 0.15, 0.17, 0.15, 0.10], list(t.design_bv)
            ))),
            description=t.description,
        )
        for t in templates
    ]
