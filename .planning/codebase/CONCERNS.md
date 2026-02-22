# Codebase Concerns

**Analysis Date:** 2026-02-22

## Tech Debt

### Missing Error Logging in Silent Exception Handlers
- **Issue:** Multiple modules use bare `except Exception: pass` blocks that silently swallow errors without logging
- **Files:** `codegen/spectral_validator.py` (lines 132, 161), `tensor/deepwiki_navigator.py` (multiple locations), `tensor/financial_ingestion.py`
- **Impact:** Difficult to diagnose failures in production; operators unaware when data acquisition fails
- **Fix approach:** Convert silent except blocks to at minimum log.warning(); optionally surface to monitoring systems

### Binary Dependency on rust-borrow-extractor
- **Issue:** BorrowPredictor falls back to zero-vector (B=0,0,0,0,0,0) when AST extractor binary unavailable (`codegen/borrow_predictor.py:96-102`)
- **Files:** `codegen/borrow_predictor.py`, `optimization/code_gen_experiment.py` (EXTRACTOR_BIN path)
- **Impact:** Prediction accuracy degrades silently; classifier trained on real BVs but post-gate gets synthetic zero-BV; false compile predictions possible
- **Fix approach:**
  1. Require explicit error propagation when binary missing (don't hide under fallback)
  2. Add integration test verifying extractor works end-to-end
  3. Consider vendoring AST extraction as pure Python fallback

### Asymmetric Classifier Training vs AST Reality Gap
- **Issue:** BorrowPredictor classifier trained on design-time BVs from metrics.jsonl, but post-gate uses AST-extracted BVs; no continuous calibration when divergence detected
- **Files:** `codegen/feedback_store.py`, `codegen/borrow_predictor.py`, `optimization/code_gen_experiment.py`
- **Impact:** As codebase evolves, AST-extracted vectors drift from training distribution; classifier accuracy may degrade undetected
- **Fix approach:**
  1. Add feedback validation gate: reject feedback with suspicious BV deviations
  2. Implement drift detection in retrain() (compute Mahalanobis distance vs training centroid)
  3. Add confidence threshold to feedback acceptance

### Hard-Coded Path Dependencies
- **Issue:** METRICS_JSONL, EXTRACTOR_BIN use hardcoded expanduser() paths (`optimization/code_gen_experiment.py:49-55`)
- **Files:** `optimization/code_gen_experiment.py`, `codegen/spectral_validator.py:55`
- **Impact:** Build breaks if paths differ across environments; no validation that paths exist
- **Fix approach:** Use environment variables with sensible defaults; add validation in __init__ or module load

### Unsafe Subprocess Calls Without Timeout Edge Cases
- **Issue:** Pipeline uses subprocess.run with timeout=120s for cargo check, but maturin_build has 300s timeout with no graceful degradation
- **Files:** `codegen/pipeline.py:178-182` (cargo), `codegen/pipeline.py:250` (maturin)
- **Impact:** Long-running Rust compilations can block CI; no configurable timeout; resource exhaustion possible
- **Fix approach:** Make timeouts configurable; add resource limits (memory, CPU); implement retry-with-smaller-config fallback

## Known Bugs

### Borrow Energy Fallback Masking AST Extraction Failure
- **Symptoms:** When rust-borrow-extractor unavailable, post-gate uses E_borrow=0.025 (pure functional profile), predicting 100% compile success regardless of actual borrow complexity
- **Files:** `codegen/borrow_predictor.py:101-103`, `optimization/code_gen_experiment.py:63-65` (e_borrow function)
- **Trigger:** rust-borrow-extractor binary not in PATH or fails silently
- **Workaround:** Manually validate critical code with rustc before deployment

### Gram Matrix Ill-Conditioning Not Surfaced
- **Symptoms:** EDMDKoopman computes gram_cond but does not warn when G becomes near-singular (cond > 1e10); fitting proceeds with numerically unstable K
- **Files:** `tensor/koopman_edmd.py:144-147` (stores cond but no check)
- **Trigger:** Trajectory with near-collinear points in observable basis; typically polynomial degree ≥ 2
- **Workaround:** Pre-filter trajectory data; use lower observable_degree

### Spectral Validator Recovers Silently from Missing Kernel
- **Symptoms:** When rust_physics_kernel import fails, SpectralValidator falls back to Python RK4, potentially giving 1000× slower results without warning
- **Files:** `codegen/spectral_validator.py:46-62`
- **Trigger:** rust-physics-kernel not installed or build failed
- **Workaround:** Explicitly test `python -m pytest tests/test_spectral_validator.py` in CI

## Security Considerations

### Untrusted JSONL Parsing
- **Risk:** FeedbackStore and analysis modules use json.loads() on JSONL without schema validation
- **Files:** `codegen/feedback_store.py:82-91`, `codegen/analysis.py:234-239`
- **Current mitigation:** Files are internal JSONL written only by pipeline; not user-facing
- **Recommendations:**
  1. Add pydantic schema validation in FeedbackSample deserialization
  2. Wrap json.loads() in try/except with explicit error logging
  3. Consider append-only ACLs on feedback.jsonl

### Cargo.toml Generation Lacks Validation
- **Risk:** pipeline.py:104 calls cargo_toml_for() with user-supplied template names; generated toml written without inspection
- **Files:** `codegen/pipeline.py:104-107`, `codegen/templates/api_handler.py` (assumes cargo_toml_for exists)
- **Current mitigation:** Templates are built-in, not user-supplied; inline validation missing
- **Recommendations:**
  1. Validate generated Cargo.toml syntax before writing
  2. Restrict dependencies to whitelisted crates
  3. Add hash verification of template registry

### Thread Safety Not Documented
- **Risk:** AutonomousLearningSystem spawns 6+ threads accessing IntegratedHDVSystem concurrently; no locking strategy documented
- **Files:** `tensor/autonomous_training.py:78-265` (multiple threads), `tensor/integrated_hdv.py` (shared data structures)
- **Current mitigation:** IntegratedHDVSystem appears to use numpy arrays (thread-safe for reads); write conflicts unstated
- **Recommendations:**
  1. Document which operations are thread-safe (read-only vs mutation)
  2. Add explicit locks around HDV writes (domain_masks, domain_dim_usage)
  3. Add race-condition test in CI

## Performance Bottlenecks

### Polynomial Observable Basis Explosion
- **Problem:** EDMDKoopman with degree=2 on 2-state system creates 6-dim observable; degree=3 becomes 9-dim. For 10-state systems, degree=2 → 66-dim (O(n²) explosion)
- **Files:** `tensor/koopman_edmd.py:76-107`
- **Cause:** All degree-2 monomials xᵢxⱼ; no pruning or sparsity
- **Improvement path:**
  1. Add observable_sparsity parameter (select top-k monomials by variance)
  2. Use randomized observable basis for large systems
  3. Benchmark: current O(n²) → O(n log n) with sparse basis

### EDMD Gram Matrix Inversion on Full Data
- **Problem:** Large trajectories (T > 5000) create k×k Gram matrices; even degree=2 with 100-dim state → 5151×5151 matrix
- **Files:** `tensor/koopman_edmd.py:139-150` (np.linalg.lstsq on full Gram)
- **Cause:** No mini-batching or streaming EDMD
- **Improvement path:**
  1. Implement incremental EDMD (Sherman-Morrison update for online fitting)
  2. Add batched EDMD option (fit on trajectory windows)
  3. Benchmark: 10000-step trajectory currently takes ~2s; streaming should be <100ms

### IntegratedHDVSystem Lazy Initialization Fragility
- **Problem:** network, function_library, hdv_mapper created in __init__ but not validated; first encode() call may fail if import chain broken
- **Files:** `tensor/integrated_hdv.py:45-77`
- **Cause:** No eager validation; errors only surface at encode time
- **Improvement path:**
  1. Add eager health-check in __init__
  2. Implement fallback to structural_encode if network missing
  3. Add unit test for missing dependency recovery

## Fragile Areas

### Spectral Validator Relies on Unfenced Imports
- **Files:** `codegen/spectral_validator.py:50-62`
- **Why fragile:**
  - Tries local import first, then searches ~/projects/rust-physics-kernel, then soft-fails with pass
  - No indication to caller whether it succeeded; returns None K module on silent failure
  - Fallback Python RK4 is 1000× slower but unlogged
- **Safe modification:**
  1. Always check _rk_module is not None before use
  2. Add explicit warning if fallback activated
  3. Test both paths (kernel available vs unavailable) in CI
- **Test coverage:** codegen/test_spectral_validator.py exists but may not cover missing-kernel case

### FeedbackStore Zero-Vector Rejection Heuristic
- **Files:** `codegen/feedback_store.py:58-61`
- **Why fragile:** Rejects all-zeros BV assuming AST extractor failed, but legitimate zero-BV (pure functional) exists
- **Safe modification:**
  1. Track AST extraction failure separately (e_g., bool field in FeedbackSample)
  2. Don't use BV values to infer extraction success
  3. Test: add feedback with legitimate (0.05, 0, 0, 0, 0, 0) and verify not rejected
- **Test coverage:** No explicit test for zero-BV edge case

### Pipeline Cargo Validation Ignores Crate Errors
- **Files:** `codegen/pipeline.py:133-159` (cargo_validate)
- **Why fragile:** Silently converts Cargo build failures to post.actual_compile=False; no distinction between compilation errors vs build system errors
- **Safe modification:**
  1. Parse cargo error output to distinguish error types
  2. Retry on transient errors (network, disk)
  3. Log full stderr for debugging
- **Test coverage:** Integration tests with intentionally-broken cargo projects missing

### Validation Bridge Assumes Patch.metadata Consistency
- **Files:** `tensor/validation_bridge.py:50-101`
- **Why fragile:** make_merged_patch() trusts trust_a, trust_b, K_a, K_b without validation; invalid matrices passed by caller silently produce wrong merged spectrum
- **Safe modification:**
  1. Add input validation: check matrices are square, same shape, K eigenvalues sensible
  2. Verify trust ∈ [0, 1]
  3. Add assertion: merged trust should be between min/max of inputs
- **Test coverage:** Unit test for make_merged_patch exists but edge cases may be missing

## Scaling Limits

### HDV Dimension Explosion Under Multi-Domain Growth
- **Current capacity:** IntegratedHDVSystem hdv_dim=10000 default; 150 modes (Layer 9)
- **Limit:** AutonomousLearningSystem adds new domains continuously (arxiv math, GitHub code, behavioral, etc.); no growth cap documented
- **Scaling path:**
  1. Implement Fibonacci growth cap (CRITICAL-4 constraint ~500 active dims at once)
  2. Add dimension pruning for low-variance domains
  3. Monitor active_dim_count; warn if > 5000

### EDMD Memory Scaling
- **Current capacity:** Observable degree 2 on 10-state system = ~66-dim observable; K matrix = 4.3 KB
- **Limit:** Degree 3 on 20-state system would be ~2200-dim observable; K = ~38 MB. Ill-conditioned G impossible to invert.
- **Scaling path:**
  1. Implement observable basis truncation (keep top-k by variance)
  2. Switch to randomized/sketched EDMD for large systems (see NIPS 2020 methods)
  3. Add automatic degree reduction if observable_dim > 500

### Thread Contention on IntegratedHDVSystem
- **Current capacity:** AutonomousLearningSystem spawns 6-10 threads reading/writing domain_masks, domain_dim_usage
- **Limit:** No explicit locking; concurrent mask updates could cause numpy array corruption under GIL release
- **Scaling path:**
  1. Use explicit threading.RLock for mask updates
  2. Implement copy-on-write for domain_masks (atomic swap)
  3. Add contention monitoring (measure lock wait times)

## Dependencies at Risk

### Optional torch/tensorflow Not Gracefully Handled
- **Risk:** Several modules conditionally import torch (dual_geometry.py, integrated_hdv.py) but don't always fail clearly
- **Files:** `tensor/dual_geometry.py:126-137` (raises RuntimeError), `tensor/integrated_hdv.py:27` (hard import)
- **Impact:** Some features silently break if torch unavailable; inconsistent error messages
- **Migration plan:**
  1. Standardize to ImportError → sensible fallback or explicit NotImplementedError
  2. Add module-level decorator: @requires_torch with clear error message
  3. Mock torch in tests to verify fallback paths work

### Classifier Model Not Versioned
- **Risk:** BorrowPredictor loads classifier from metrics.jsonl; no version check if training procedure changed
- **Files:** `optimization/code_gen_experiment.py:144-169`
- **Impact:** Old metrics.jsonl may be incompatible with current training logic; silent performance degradation
- **Migration plan:**
  1. Add version field to metrics.jsonl (e.g., "classifier_version": 2)
  2. Raise error if loading metrics_version < current_code_version
  3. Add migration script for old metrics

## Missing Critical Features

### No Recovery for Partial Feedback Corruption
- **Problem:** FeedbackStore.load_feedback() skips malformed JSONL lines silently; over time, historical feedback becomes inconsistent
- **Blocks:** Reproducible retrain results; debugging feedback quality issues
- **Fix approach:**
  1. Track line numbers of corrupted samples; log warnings
  2. Add repair mode: scan for and report corruption
  3. Implement append-only validation on write

### No Monitoring of Classifier Drift
- **Problem:** Retrain() checks AUC > min_auc but doesn't warn if AUC dropping over time; model could be slowly degrading undetected
- **Blocks:** Proactive model updates; operator visibility into prediction quality
- **Fix approach:**
  1. Add historical AUC tracking (store retrain results)
  2. Implement drift detector: flag if AUC_new < AUC_old - 0.05
  3. Add metrics export for monitoring dashboards

### No Fallback Template When best_match() Returns None
- **Problem:** CodeGenPipeline.generate() returns error when no template found; no graceful degradation
- **Files:** `codegen/pipeline.py:84-90`
- **Blocks:** Handling new intent domains; graceful failure modes
- **Fix approach:**
  1. Implement generic fallback template (identity function, minimal Rust)
  2. Add template suggestion system (find closest existing template)
  3. Log new intent for later template creation

## Test Coverage Gaps

### AST Extractor Unavailability Not Tested
- **What's not tested:** Post-gate behavior when rust-borrow-extractor missing; verification that fallback doesn't silently accept broken code
- **Files:** `codegen/borrow_predictor.py`, `optimization/code_gen_experiment.py`
- **Risk:** Zero-BV fallback could pass clearly-broken code as compilable
- **Priority:** High — affects production safety

### Gram Matrix Conditioning Edge Cases
- **What's not tested:** EDMD fitting with near-singular Gram matrices; verification that cond warning (or error) is surfaced
- **Files:** `tensor/koopman_edmd.py`
- **Risk:** Fitting proceeds with K matrix containing large errors; eigenvalues unreliable
- **Priority:** High — affects all EDMD-based analysis

### Concurrent Access to Shared HDV State
- **What's not tested:** Race conditions in IntegratedHDVSystem under simultaneous domain_mask writes from multiple threads
- **Files:** `tensor/integrated_hdv.py`, `tensor/autonomous_training.py` (spawns threads)
- **Risk:** Data corruption; unpredictable behavior under load
- **Priority:** High — affects production stability

### Cargo Compilation Failure Modes
- **What's not tested:** Distinction between Rust compile errors vs build system failures vs timeout in cargo check
- **Files:** `codegen/pipeline.py:178-187`
- **Risk:** Transient build failures treated as code generation failures; real bugs hidden
- **Priority:** Medium — affects pipeline reliability

---

*Concerns audit: 2026-02-22*
