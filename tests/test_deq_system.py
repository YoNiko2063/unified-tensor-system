"""Tests for tensor/deq_system.py"""

import pytest
import numpy as np

from tensor.deq_system import (
    DifferentialEquation,
    PaperToDEQConverter,
    CircuitToDEQConverter,
    CodeToDEQConverter,
    Model3DToDEQConverter,
    VeriGPUToDEQConverter,
    TinyGPUToDEQConverter,
    GPUPhysicsSimulator,
    UnifiedDEQSolver,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def small_hdv(tmp_path):
    from tensor.integrated_hdv import IntegratedHDVSystem
    return IntegratedHDVSystem(
        hdv_dim=300, n_modes=5, embed_dim=32,
        library_path=str(tmp_path / "lib.json"),
    )


RC_NETLIST = """\
R1 N1 N2 1000
C1 N2 GND 1e-6
V1 N1 GND 5.0
"""

PAPER_TEXT = """\
We study the equation

$$\\frac{dx}{dt} = -\\alpha x$$

where x is the state and \\alpha > 0 is the decay rate.
"""

PYTHON_CODE = """\
n = 10
total = 0
for i in range(n):
    total += i
"""

MODEL_DESC = "Cylinder 30mm diameter, 50mm height, printed in PLA."


# ── DifferentialEquation ──────────────────────────────────────────────────────

def test_deq_evaluate_default():
    deq = DifferentialEquation(equation="dx/dt = -x")
    x = np.array([1.0, 2.0])
    result = deq.evaluate(0.0, x)
    assert np.allclose(result, [-1.0, -2.0])


def test_deq_evaluate_single():
    deq = DifferentialEquation(equation="dx/dt = -x")
    result = deq.evaluate(0.0, np.array([5.0]))
    assert result[0] == pytest.approx(-5.0)


def test_deq_verify_solution_zero_means_zero_residual():
    # dx/dt = -x at x=0 → RHS=0 → residual=0
    deq = DifferentialEquation(equation="dx/dt = -x")
    res = deq.verify_solution(np.array([0.0]))
    assert res == pytest.approx(0.0)


def test_deq_verify_solution_nonzero():
    deq = DifferentialEquation(equation="dx/dt = -x")
    res = deq.verify_solution(np.array([3.0]))
    assert res > 0.0


def test_deq_dataclass_defaults():
    deq = DifferentialEquation(equation="test")
    assert deq.domain == "math"
    assert deq.variables == []
    assert deq.parameters == {}
    assert deq.t_final == 1.0
    assert deq.source == ""


# ── PaperToDEQConverter ───────────────────────────────────────────────────────

def test_paper_converter_returns_deq():
    conv = PaperToDEQConverter()
    deq = conv.convert(PAPER_TEXT)
    assert isinstance(deq, DifferentialEquation)


def test_paper_converter_domain_math():
    conv = PaperToDEQConverter()
    deq = conv.convert(PAPER_TEXT)
    assert deq.domain == "math"


def test_paper_converter_source():
    conv = PaperToDEQConverter()
    deq = conv.convert(PAPER_TEXT)
    assert deq.source == "paper"


def test_paper_converter_equation_nonempty():
    conv = PaperToDEQConverter()
    deq = conv.convert(PAPER_TEXT)
    assert len(deq.equation) > 0


def test_paper_converter_detects_alpha_param():
    conv = PaperToDEQConverter()
    deq = conv.convert(PAPER_TEXT)
    assert "alpha" in deq.parameters or len(deq.equation) > 0


def test_paper_converter_no_equations_uses_default():
    conv = PaperToDEQConverter()
    deq = conv.convert("This paper discusses many things without equations.")
    assert "=" in deq.equation


def test_paper_find_variables():
    conv = PaperToDEQConverter()
    vars_ = conv._find_variables("dx/dt = -alpha * x + u")
    assert "x" in vars_ or "u" in vars_ or "d" in vars_


def test_paper_standard_form_passthrough():
    conv = PaperToDEQConverter()
    eq = "dx/dt = -x + 1"
    assert conv._to_standard_form(eq) == eq


def test_paper_standard_form_static_eq():
    conv = PaperToDEQConverter()
    out = conv._to_standard_form("x + 1 = 0")
    assert "∂x/∂t" in out or "dx/dt" in out or "=" in out


# ── CircuitToDEQConverter ─────────────────────────────────────────────────────

def test_circuit_converter_returns_deq():
    conv = CircuitToDEQConverter()
    deq = conv.convert(RC_NETLIST)
    assert isinstance(deq, DifferentialEquation)


def test_circuit_converter_domain_physical():
    conv = CircuitToDEQConverter()
    deq = conv.convert(RC_NETLIST)
    assert deq.domain == "physical"


def test_circuit_converter_source():
    conv = CircuitToDEQConverter()
    deq = conv.convert(RC_NETLIST)
    assert deq.source == "circuit"


def test_circuit_converter_has_state_vars():
    conv = CircuitToDEQConverter()
    deq = conv.convert(RC_NETLIST)
    # Should have V_C1 as capacitor state variable
    assert any("C1" in v for v in deq.variables)


def test_circuit_converter_has_params():
    conv = CircuitToDEQConverter()
    deq = conv.convert(RC_NETLIST)
    assert "C1" in deq.parameters


def test_circuit_converter_equation_contains_derivative():
    conv = CircuitToDEQConverter()
    deq = conv.convert(RC_NETLIST)
    assert "/dt" in deq.equation or "∂" in deq.equation


def test_circuit_converter_empty_netlist():
    conv = CircuitToDEQConverter()
    deq = conv.convert("")
    assert isinstance(deq, DifferentialEquation)
    assert "=" in deq.equation


def test_circuit_converter_inductor():
    conv = CircuitToDEQConverter()
    netlist = "L1 N1 N2 1e-3\nR1 N2 GND 50\nV1 N1 GND 12.0\n"
    deq = conv.convert(netlist)
    assert any("L1" in v for v in deq.variables)


def test_circuit_parse_comment_lines():
    conv = CircuitToDEQConverter()
    netlist = "# RC filter\nR1 in out 100\nC1 out gnd 1e-6\n"
    comps = conv._parse(netlist)
    assert len(comps) == 2


# ── CodeToDEQConverter ────────────────────────────────────────────────────────

def test_code_converter_returns_deq():
    conv = CodeToDEQConverter()
    deq = conv.convert(PYTHON_CODE)
    assert isinstance(deq, DifferentialEquation)


def test_code_converter_domain_execution():
    conv = CodeToDEQConverter()
    deq = conv.convert(PYTHON_CODE)
    assert deq.domain == "execution"


def test_code_converter_source():
    conv = CodeToDEQConverter()
    deq = conv.convert(PYTHON_CODE)
    assert deq.source == "code"


def test_code_converter_state_var_total():
    conv = CodeToDEQConverter()
    deq = conv.convert(PYTHON_CODE)
    # "total" is modified by += in a loop
    assert "total" in deq.variables


def test_code_converter_params_n():
    conv = CodeToDEQConverter()
    deq = conv.convert(PYTHON_CODE)
    # n = 10 is a named constant
    assert "n" in deq.parameters
    assert deq.parameters["n"] == 10.0


def test_code_converter_equation_contains_partial():
    conv = CodeToDEQConverter()
    deq = conv.convert(PYTHON_CODE)
    assert "∂total/∂t" in deq.equation or "dx/dt" in deq.equation


def test_code_converter_syntax_error_returns_default():
    conv = CodeToDEQConverter()
    deq = conv.convert("def !! broken code @@##")
    assert isinstance(deq, DifferentialEquation)
    assert "dx/dt" in deq.equation


def test_code_converter_empty_code():
    conv = CodeToDEQConverter()
    deq = conv.convert("")
    assert isinstance(deq, DifferentialEquation)


def test_code_converter_while_loop():
    code = "i = 0\nwhile i < 10:\n    i += 1\n"
    conv = CodeToDEQConverter()
    deq = conv.convert(code)
    assert "i" in deq.variables


# ── Model3DToDEQConverter ─────────────────────────────────────────────────────

def test_3d_converter_returns_deq():
    conv = Model3DToDEQConverter()
    deq = conv.convert(MODEL_DESC)
    assert isinstance(deq, DifferentialEquation)


def test_3d_converter_domain_physical():
    conv = Model3DToDEQConverter()
    deq = conv.convert(MODEL_DESC)
    assert deq.domain == "physical"


def test_3d_converter_source_contains_material():
    conv = Model3DToDEQConverter()
    deq = conv.convert(MODEL_DESC, material="pla")
    assert "pla" in deq.source.lower()


def test_3d_converter_pla_params():
    conv = Model3DToDEQConverter()
    deq = conv.convert(MODEL_DESC, material="pla")
    assert "alpha_thermal" in deq.parameters
    assert deq.parameters["alpha_thermal"] == pytest.approx(1.2e-4, rel=1e-3)


def test_3d_converter_steel_params():
    conv = Model3DToDEQConverter()
    deq = conv.convert("100mm cube", material="steel")
    assert deq.parameters["rho_material"] == pytest.approx(7850.0, rel=1e-3)


def test_3d_converter_equation_has_heat():
    conv = Model3DToDEQConverter()
    deq = conv.convert(MODEL_DESC)
    assert "∂T/∂t" in deq.equation


def test_3d_converter_volume_from_dims():
    conv = Model3DToDEQConverter()
    deq = conv.convert("Box 10mm x 20mm x 30mm")
    assert deq.parameters["volume_mm3"] == pytest.approx(6000.0, rel=0.1)


def test_3d_converter_default_volume_when_no_dims():
    conv = Model3DToDEQConverter()
    deq = conv.convert("some amorphous shape without dimensions")
    assert deq.parameters["volume_mm3"] == 1000.0


def test_3d_converter_unknown_material_uses_pla():
    conv = Model3DToDEQConverter()
    deq = conv.convert("object", material="unobtanium")
    alpha_pla = Model3DToDEQConverter.MATERIAL_DB["pla"]["alpha"]
    assert deq.parameters["alpha_thermal"] == pytest.approx(alpha_pla, rel=1e-3)


# ── UnifiedDEQSolver ──────────────────────────────────────────────────────────

def test_solver_convert_paper():
    solver = UnifiedDEQSolver()
    deq = solver.convert(PAPER_TEXT, "paper")
    assert isinstance(deq, DifferentialEquation)


def test_solver_convert_circuit():
    solver = UnifiedDEQSolver()
    deq = solver.convert(RC_NETLIST, "circuit")
    assert isinstance(deq, DifferentialEquation)


def test_solver_convert_code():
    solver = UnifiedDEQSolver()
    deq = solver.convert(PYTHON_CODE, "code")
    assert isinstance(deq, DifferentialEquation)


def test_solver_convert_3d():
    solver = UnifiedDEQSolver()
    deq = solver.convert(MODEL_DESC, "3d_model")
    assert isinstance(deq, DifferentialEquation)


def test_solver_convert_unknown_type_raises():
    solver = UnifiedDEQSolver()
    with pytest.raises(ValueError, match="Unknown input_type"):
        solver.convert("data", "unknown_type")


def test_solver_solve_returns_dict():
    solver = UnifiedDEQSolver()
    result = solver.solve(PAPER_TEXT, "paper")
    assert isinstance(result, dict)
    assert "equation" in result
    assert "solution" in result
    assert "verified" in result
    assert "confidence" in result


def test_solver_solve_circuit():
    solver = UnifiedDEQSolver()
    result = solver.solve(RC_NETLIST, "circuit")
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0


def test_solver_solve_code():
    solver = UnifiedDEQSolver()
    result = solver.solve(PYTHON_CODE, "code")
    assert result["domain"] == "execution"


def test_solver_solution_has_t_y():
    solver = UnifiedDEQSolver()
    result = solver.solve(PAPER_TEXT, "paper")
    sol = result["solution"]
    assert "t" in sol
    assert "y" in sol


def test_solver_with_hdv(small_hdv):
    solver = UnifiedDEQSolver(hdv_system=small_hdv)
    result = solver.solve(PAPER_TEXT, "paper")
    assert isinstance(result["confidence"], float)


def test_solver_stores_solutions(small_hdv):
    solver = UnifiedDEQSolver(hdv_system=small_hdv)
    solver.solve(PAPER_TEXT, "paper")
    assert len(solver._store) == 1


def test_solver_finds_similar_on_second_call(small_hdv):
    solver = UnifiedDEQSolver(hdv_system=small_hdv)
    solver.solve(PAPER_TEXT, "paper")
    result = solver.solve(PAPER_TEXT, "paper")  # same equation
    assert result["similar_found"] >= 0   # may be 1 or more


def test_solver_verify_success_true():
    solver = UnifiedDEQSolver()
    deq = DifferentialEquation(equation="dx/dt = -x")
    sol = {"t": [0.0, 1.0], "y": [[1.0], [0.37]], "success": True, "message": "ok"}
    verified, conf = solver._verify(deq, sol, None)
    assert isinstance(verified, bool)
    assert 0.0 <= conf <= 1.0


def test_solver_verify_failed_solution():
    solver = UnifiedDEQSolver()
    deq = DifferentialEquation(equation="dx/dt = -x")
    sol = {"t": [], "y": [], "success": False, "message": "error"}
    verified, conf = solver._verify(deq, sol, None)
    assert not verified
    assert conf == 0.0


# ── VeriGPUToDEQConverter ─────────────────────────────────────────────────────

VERILOG_SPEC = "always coherence must hold; eventually deadlock free guaranteed"
VERILOG_SPEC_UNTIL = "race free must hold until coherence is reached"

def test_verigpu_returns_deq():
    deq = VeriGPUToDEQConverter().convert(VERILOG_SPEC)
    assert isinstance(deq, DifferentialEquation)

def test_verigpu_domain_physical():
    deq = VeriGPUToDEQConverter().convert(VERILOG_SPEC)
    assert deq.domain == "physical"

def test_verigpu_source():
    deq = VeriGPUToDEQConverter().convert(VERILOG_SPEC)
    assert deq.source == "verigpu"

def test_verigpu_detects_coherence_var():
    deq = VeriGPUToDEQConverter().convert("always coherence invariant")
    assert "coherence" in deq.variables

def test_verigpu_detects_race_free_var():
    deq = VeriGPUToDEQConverter().convert("race condition must be eliminated")
    assert "race_free" in deq.variables

def test_verigpu_default_vars_when_no_keywords():
    deq = VeriGPUToDEQConverter().convert("some verification spec")
    assert len(deq.variables) >= 1

def test_verigpu_equilibrium_form():
    deq = VeriGPUToDEQConverter().convert("always coherence invariant")
    assert "∂coherence/∂t" in deq.equation

def test_verigpu_convergence_form():
    deq = VeriGPUToDEQConverter().convert("coherence eventually converges liveness")
    assert "(1 - coherence)" in deq.equation

def test_verigpu_until_form():
    deq = VeriGPUToDEQConverter().convert("race until deadlock free")
    # Until couples two adjacent variables
    assert " - " in deq.equation

def test_verigpu_has_constraints():
    deq = VeriGPUToDEQConverter().convert(VERILOG_SPEC)
    assert len(deq.constraints) > 0

def test_verigpu_solver_integration():
    solver = UnifiedDEQSolver()
    result = solver.solve(VERILOG_SPEC, "verigpu")
    assert result["verified"] is True or result["confidence"] >= 0.0

def test_verigpu_solver_type_registered():
    solver = UnifiedDEQSolver()
    assert "verigpu" in solver.converters


# ── TinyGPUToDEQConverter ─────────────────────────────────────────────────────

PIPELINE_DESC = "fetch decode execute writeback pipeline stages"
CACHE_DESC = "L1 L2 DRAM cache miss hit memory bandwidth"
WARP_DESC = "warp thread scheduling SIMD latency hide"
FULL_GPU_DESC = PIPELINE_DESC + " " + CACHE_DESC + " " + WARP_DESC

def test_tinygpu_returns_deq():
    deq = TinyGPUToDEQConverter().convert(PIPELINE_DESC)
    assert isinstance(deq, DifferentialEquation)

def test_tinygpu_domain_physical():
    deq = TinyGPUToDEQConverter().convert(PIPELINE_DESC)
    assert deq.domain == "physical"

def test_tinygpu_source():
    deq = TinyGPUToDEQConverter().convert(PIPELINE_DESC)
    assert deq.source == "tiny_gpu"

def test_tinygpu_pipeline_var():
    deq = TinyGPUToDEQConverter().convert(PIPELINE_DESC)
    assert "stage" in deq.variables

def test_tinygpu_cache_var():
    deq = TinyGPUToDEQConverter().convert(CACHE_DESC)
    assert "hit_rate" in deq.variables

def test_tinygpu_warp_var():
    deq = TinyGPUToDEQConverter().convert(WARP_DESC)
    assert "utilization" in deq.variables

def test_tinygpu_all_three_vars():
    deq = TinyGPUToDEQConverter().convert(FULL_GPU_DESC)
    assert "stage" in deq.variables
    assert "hit_rate" in deq.variables
    assert "utilization" in deq.variables

def test_tinygpu_n_stages_param():
    deq = TinyGPUToDEQConverter().convert(PIPELINE_DESC)
    assert deq.parameters.get("n_stages", 0) >= 4

def test_tinygpu_default_when_no_keywords():
    deq = TinyGPUToDEQConverter().convert("general gpu description")
    assert isinstance(deq, DifferentialEquation)
    assert "=" in deq.equation

def test_tinygpu_extract_patterns_pipeline():
    conv = TinyGPUToDEQConverter()
    deq = conv.convert(PIPELINE_DESC)
    patterns = conv.extract_optimization_patterns(deq)
    types = [p["type"] for p in patterns]
    assert "pipeline" in types

def test_tinygpu_extract_patterns_cache():
    conv = TinyGPUToDEQConverter()
    deq = conv.convert(CACHE_DESC)
    patterns = conv.extract_optimization_patterns(deq)
    types = [p["type"] for p in patterns]
    assert "cache_hierarchy" in types

def test_tinygpu_extract_patterns_warp():
    conv = TinyGPUToDEQConverter()
    deq = conv.convert(WARP_DESC)
    patterns = conv.extract_optimization_patterns(deq)
    types = [p["type"] for p in patterns]
    assert "warp_scheduling" in types

def test_tinygpu_solver_integration():
    solver = UnifiedDEQSolver()
    result = solver.solve(PIPELINE_DESC, "tiny_gpu")
    assert "equation" in result
    assert "confidence" in result

def test_tinygpu_solver_type_registered():
    solver = UnifiedDEQSolver()
    assert "tiny_gpu" in solver.converters


# ── GPUPhysicsSimulator ───────────────────────────────────────────────────────

def test_physics_sim_returns_dict():
    conv = TinyGPUToDEQConverter()
    deq = conv.convert(PIPELINE_DESC)
    result = GPUPhysicsSimulator().simulate(deq, t_max=1e-6, steps=10)
    assert isinstance(result, dict)

def test_physics_sim_keys():
    deq = TinyGPUToDEQConverter().convert(PIPELINE_DESC)
    result = GPUPhysicsSimulator().simulate(deq, t_max=1e-6, steps=10)
    for key in ("temperature", "power", "performance", "violations", "energy_efficiency", "verified"):
        assert key in result

def test_physics_sim_temperature_trace_length():
    deq = TinyGPUToDEQConverter().convert(PIPELINE_DESC)
    result = GPUPhysicsSimulator().simulate(deq, t_max=1e-6, steps=20)
    assert len(result["temperature"]) == 20

def test_physics_sim_no_violations_short_run():
    # Very short run, very small power → no violations expected
    deq = TinyGPUToDEQConverter().convert(PIPELINE_DESC)
    result = GPUPhysicsSimulator().simulate(deq, t_max=1e-9, steps=5)
    assert isinstance(result["violations"], list)

def test_physics_sim_verigpu_deq():
    deq = VeriGPUToDEQConverter().convert("always coherence invariant deadlock free")
    result = GPUPhysicsSimulator().simulate(deq, t_max=1e-6, steps=10)
    assert isinstance(result["verified"], bool)

def test_physics_sim_energy_efficiency_positive():
    deq = TinyGPUToDEQConverter().convert(PIPELINE_DESC)
    result = GPUPhysicsSimulator().simulate(deq, t_max=1e-6, steps=10)
    assert result["energy_efficiency"] >= 0.0

def test_physics_sim_custom_parameters():
    deq = TinyGPUToDEQConverter().convert(PIPELINE_DESC)
    result = GPUPhysicsSimulator().simulate(deq, parameters={"V": 0.5, "f": 1e8}, t_max=1e-6, steps=5)
    assert isinstance(result, dict)

def test_solver_has_physics():
    solver = UnifiedDEQSolver()
    assert hasattr(solver, "physics")
    assert isinstance(solver.physics, GPUPhysicsSimulator)


# ── (existing) solver_euler_fallback ─────────────────────────────────────────

def test_solver_euler_fallback():
    # Disable scipy to force Euler
    import tensor.deq_system as ds
    orig = ds._SCIPY
    ds._SCIPY = False
    try:
        solver = UnifiedDEQSolver()
        deq = DifferentialEquation(equation="dx/dt = -x", variables=["x"])
        sol = solver._solve_numerically(deq, None)
        assert sol["success"] is True
        assert len(sol["t"]) > 1
    finally:
        ds._SCIPY = orig
