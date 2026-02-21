"""
optimization/ — constrained optimization over HDV-parameterized physical systems.

Modules:
  koopman_signature  — KoopmanInvariantDescriptor + compute_invariants()
  koopman_memory     — KoopmanExperienceMemory + OptimizationExperience
  rlc_parameterization — RLCDesignMapper (HDV → R, L, C)
  rlc_evaluator      — RLCEvaluator (physics + simulation verification)
  hdv_optimizer      — ConstrainedHDVOptimizer (search + memory integration)
  demo_rlc_optimization — CLI entry point
"""
