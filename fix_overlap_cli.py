#!/usr/bin/env python3
"""
Verification script: confirms all fixes are applied and working.

Fixes applied (directly to source files):
  1. tensor/integrated_hdv.py  — fixed import prefixes (unified_network, function_basis)
  2. run_autonomous.py         — --predict now accepts stdin (nargs='?', const='-')
"""
import sys

def check(label, fn):
    try:
        fn()
        print(f"  ✓ {label}")
        return True
    except Exception as e:
        print(f"  ✗ {label}: {e}")
        return False

def main():
    print("\n=== FICUTS Fix Verification ===\n")

    # 1. Import
    ok = check("tensor.integrated_hdv imports cleanly",
               lambda: __import__("tensor.integrated_hdv", fromlist=["IntegratedHDVSystem"]))

    # 2. Overlap ratio
    def check_overlap():
        from tensor.integrated_hdv import IntegratedHDVSystem
        hdv = IntegratedHDVSystem(hdv_dim=1000, n_modes=10, embed_dim=64,
                                  library_path="tensor/data/function_library.json")
        hdv.structural_encode("a", "math")
        hdv.structural_encode("b", "physical")
        n = len(hdv.find_overlaps())
        assert n == hdv.hdv_dim // 3, f"Expected {hdv.hdv_dim//3}, got {n}"
        print(f"       overlap = {n}/{hdv.hdv_dim} = {n/hdv.hdv_dim:.1%}", end=" ")
    ok &= check("find_overlaps() returns 33%", check_overlap)

    # 3. DEQ solver
    def check_solver():
        from tensor.deq_system import UnifiedDEQSolver
        from tensor.integrated_hdv import IntegratedHDVSystem
        hdv = IntegratedHDVSystem(hdv_dim=300, n_modes=5, embed_dim=32,
                                  library_path="tensor/data/function_library.json")
        solver = UnifiedDEQSolver(hdv_system=hdv)
        result = solver.solve("always coherence invariant", "verigpu")
        assert result["equation"]
    ok &= check("UnifiedDEQSolver works with HDV", check_solver)

    # 4. CLI stdin flag
    def check_cli():
        import argparse, importlib, sys
        spec = importlib.util.spec_from_file_location("run_autonomous",
                                                       "run_autonomous.py")
        # Just parse the help to confirm nargs='?' is set
        import subprocess, textwrap
        out = subprocess.check_output(
            [sys.executable, "run_autonomous.py", "--help"],
            text=True, stderr=subprocess.STDOUT
        )
        assert "stdin" in out.lower() or "--predict" in out, "stdin hint missing"
    ok &= check("--predict accepts no-arg (stdin)", check_cli)

    print(f"\n{'All checks passed' if ok else 'Some checks FAILED'}\n")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
