"""CodeGenPipeline — orchestrator: IntentSpec → compiled Rust.

Flow:
  1. Pre-gate: predict from intent's expected BV
  2. Template lookup: best_match from registry
  3. Render: template.render(intent.parameters) → Rust source
  4. Post-gate: AST-extract BV, predict, try_compile
  5. Feedback: record (BV, compile_result) to feedback store

For crate-dependent templates (requires_cargo=True), uses Cargo-based
validation instead of standalone rustc.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Dict, Optional

from codegen.borrow_predictor import BorrowPredictor, PredictionResult
from codegen.feedback_store import FeedbackStore
from codegen.intent_spec import IntentSpec
from codegen.template_registry import RustTemplate, TemplateRegistry
from codegen.templates import numeric_kernel, market_model, api_handler, text_parser, html_navigator, physics_sim, trading_sim


@dataclass
class GenerationResult:
    """Result of a code generation pipeline run."""
    success: bool
    intent: IntentSpec
    template_name: str = ""
    rust_source: str = ""
    cargo_toml: str = ""
    pre_gate: Optional[PredictionResult] = None
    post_gate: Optional[PredictionResult] = None
    error: str = ""


class CodeGenPipeline:
    """IntentSpec → compiled Rust code generation pipeline."""

    def __init__(
        self,
        registry: Optional[TemplateRegistry] = None,
        predictor: Optional[BorrowPredictor] = None,
        feedback: Optional[FeedbackStore] = None,
    ) -> None:
        self._registry = registry or self._default_registry()
        self._predictor = predictor or BorrowPredictor()
        self._feedback = feedback or FeedbackStore()

    @staticmethod
    def _default_registry() -> TemplateRegistry:
        """Build registry with all built-in templates."""
        reg = TemplateRegistry()
        numeric_kernel.register_all(reg)
        market_model.register_all(reg)
        api_handler.register_all(reg)
        text_parser.register_all(reg)
        html_navigator.register_all(reg)
        physics_sim.register_all(reg)
        trading_sim.register_all(reg)
        return reg

    def generate(self, intent: IntentSpec) -> GenerationResult:
        """Full pipeline: intent → pre-gate → render → post-gate → feedback."""
        result = GenerationResult(success=False, intent=intent)

        # 1. Pre-gate
        pre = self._predictor.from_intent(intent)
        result.pre_gate = pre
        if not pre.predicted_compile:
            result.error = (
                f"Pre-gate FAIL: E_borrow={pre.e_borrow:.3f} > D_SEP, "
                f"prob={pre.probability:.3f}"
            )
            return result

        # 2. Template lookup
        template = self._registry.best_match(intent)
        if template is None:
            result.error = (
                f"No template found for domain={intent.domain}, "
                f"operation={intent.operation}"
            )
            return result
        result.template_name = template.name

        # 3. Render
        try:
            rust_source = template.render(intent.parameters)
        except Exception as e:
            result.error = f"Template render failed: {e}"
            return result
        result.rust_source = rust_source

        # 4. Generate Cargo.toml if needed
        if template.requires_cargo:
            try:
                from codegen.templates.api_handler import cargo_toml_for
                result.cargo_toml = cargo_toml_for(template.name, intent.parameters)
            except ValueError:
                result.cargo_toml = ""

        # 5. Post-gate
        if template.requires_cargo:
            post = self._cargo_validate(rust_source, result.cargo_toml)
        else:
            post = self._predictor.validate_generated(rust_source)
        result.post_gate = post

        # 6. Feedback
        if post.actual_compile is not None:
            self._feedback.record_from_result(post, template_name=template.name)

        # 7. Result
        if post.actual_compile is True:
            result.success = True
        elif post.actual_compile is None:
            # rustc not available — trust the prediction
            result.success = post.predicted_compile
            result.error = "rustc unavailable — using prediction only"
        else:
            result.success = False
            result.error = f"Compile FAIL: {post.compile_stderr[:200]}"

        return result

    def _cargo_validate(self, rust_source: str, cargo_toml: str) -> PredictionResult:
        """Validate crate-dependent template with Cargo."""
        from codegen.borrow_predictor import PredictionResult
        from optimization.code_gen_experiment import extract_borrow_vector, e_borrow

        # AST extraction (independent of Cargo)
        from optimization.code_gen_experiment import predict as _predict
        ast_bv = extract_borrow_vector(rust_source)
        bv = ast_bv if ast_bv is not None else (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        source = "ast" if ast_bv is not None else "ast_fallback"
        eb = e_borrow(bv)

        self._predictor._ensure_classifier()
        pred, prob = _predict(self._predictor._clf, self._predictor._scaler, bv)

        # Try cargo check
        actual, stderr = self._try_cargo_check(rust_source, cargo_toml)

        return PredictionResult(
            borrow_vector=bv,
            e_borrow=eb,
            predicted_compile=pred,
            probability=prob,
            actual_compile=actual,
            compile_stderr=stderr,
            source=source,
        )

    @staticmethod
    def _try_cargo_check(rust_source: str, cargo_toml: str) -> tuple:
        """Run cargo check in a temp directory."""
        cargo = shutil.which("cargo")
        if cargo is None:
            return None, ""

        tmpdir = tempfile.mkdtemp(prefix="codegen_")
        try:
            src_dir = os.path.join(tmpdir, "src")
            os.makedirs(src_dir)

            with open(os.path.join(tmpdir, "Cargo.toml"), "w") as f:
                f.write(cargo_toml)
            with open(os.path.join(src_dir, "lib.rs"), "w") as f:
                f.write(rust_source)

            result = subprocess.run(
                [cargo, "check"],
                cwd=tmpdir,
                capture_output=True, text=True, timeout=120,
            )
            return result.returncode == 0, result.stderr
        except Exception as e:
            return None, str(e)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @property
    def registry(self) -> TemplateRegistry:
        return self._registry

    @property
    def predictor(self) -> BorrowPredictor:
        return self._predictor
