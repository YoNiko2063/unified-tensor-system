"""Rust code generation pipeline — IntentSpec → compiled Rust with BorrowVector validation."""

from codegen.intent_spec import IntentSpec, BorrowProfile
from codegen.template_registry import TemplateRegistry, RustTemplate
from codegen.borrow_predictor import BorrowPredictor
from codegen.pipeline import CodeGenPipeline, GenerationResult

__all__ = [
    "IntentSpec",
    "BorrowProfile",
    "TemplateRegistry",
    "RustTemplate",
    "BorrowPredictor",
    "CodeGenPipeline",
    "GenerationResult",
]
