"""
FICUTS Simulation Trainer

Generates physical training data by running actual circuit and mechanical
simulations via ecemath, then encoding the results as HDV vectors in the
'physical' dimension.

The key insight (from LOGIC_FLOW.md Section 5 PoC Phase 9):
- An RC circuit in the linear region is an LCA patch with eigenvalue −1/RC
- A mechanical mass-spring-damper has eigenvalue −b/2m ± sqrt((b/2m)²−k/m)
- Both are second-order linear ODEs → same operator algebra → shared Pontryagin characters
  → their HDV vectors should have high overlap similarity

By populating the 'physical' dimension with these simulation results,
CrossDimensionalDiscovery can find universals between:
  - 'physical' (circuit/mechanical simulation results)
  - 'math' (arXiv equations about eigenvalues, stability, ODEs)
  - 'execution' (ecemath code functions like numerical_jacobian())

This closes the loop: the system learns that:
  ecemath.numerical_jacobian()  ≡  ∂F_i/∂x_j from arXiv papers
  RLC eigenvalues  ≡  modal analysis from physics papers
  spring-mass dynamics  ≡  circuit resonance equations
"""
from __future__ import annotations

import os
import sys
import time
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# ── Path setup: add ecemath root so relative imports inside src/ work ───────
# Must use ecemath/ as root (not ecemath/src/) so that src.backend.* relative
# imports resolve correctly. Pattern used by other tensor modules.
_ROOT = Path(__file__).parent.parent
_ECEMATH_ROOT = str(_ROOT / "ecemath")
if _ECEMATH_ROOT not in sys.path:
    sys.path.insert(0, _ECEMATH_ROOT)


# ── Circuit topology definitions ────────────────────────────────────────────

def _build_rc_circuit(R: float, C: float) -> Any:
    """Build a simple RC circuit IR."""
    from src.ir.circuit_ir import CircuitIR
    ir = CircuitIR("rc")
    ir.add_port("Vin", "input")
    ir.add_port("Vout", "output")
    ir.add_net("n1")
    ir.add_instance("R1", "resistor", {"p": "Vin", "n": "n1"}, {"R": R})
    ir.add_instance("C1", "capacitor", {"p": "n1", "n": "GND"}, {"C": C})
    return ir


def _build_rlc_circuit(R: float, L: float, C: float) -> Any:
    """Build a series RLC resonant circuit."""
    from src.ir.circuit_ir import CircuitIR
    ir = CircuitIR("rlc")
    ir.add_port("Vin", "input")
    ir.add_port("Vout", "output")
    ir.add_net("n_rl")
    ir.add_instance("R1", "resistor",  {"p": "Vin",  "n": "n_rl"}, {"R": R})
    ir.add_instance("L1", "inductor",  {"p": "n_rl", "n": "Vout"}, {"L": L})
    ir.add_instance("C1", "capacitor", {"p": "Vout", "n": "GND"},   {"C": C})
    return ir


def _build_lc_circuit(L: float, C: float) -> Any:
    """Build a lossless LC tank circuit."""
    from src.ir.circuit_ir import CircuitIR
    ir = CircuitIR("lc")
    ir.add_port("Vin", "input")
    ir.add_port("Vout", "output")
    ir.add_net("n_lc")
    ir.add_instance("L1", "inductor",  {"p": "Vin",  "n": "n_lc"}, {"L": L})
    ir.add_instance("C1", "capacitor", {"p": "n_lc", "n": "GND"},   {"C": C})
    return ir


# ── Simulation result extraction ─────────────────────────────────────────────

class SimulationResult:
    """
    Structured result from a single circuit simulation.
    Contains the data we use to encode HDV vectors.
    """

    def __init__(
        self,
        circuit_type: str,
        params: Dict[str, float],
        eigenvalues: np.ndarray,
        stable: bool,
        n_states: int,
        natural_freq: float,
        damping_ratio: float,
        time_constant: float,
        jacobian: Optional[np.ndarray] = None,
    ):
        self.circuit_type = circuit_type
        self.params = params
        self.eigenvalues = eigenvalues
        self.stable = stable
        self.n_states = n_states
        self.natural_freq = natural_freq     # Hz
        self.damping_ratio = damping_ratio   # ζ
        self.time_constant = time_constant   # τ = 1/|Re(λ)|
        self.jacobian = jacobian             # optional

    def to_text_description(self) -> str:
        """
        Build a rich text description that shares vocabulary with:
        - arXiv math papers: eigenvalue, stability, frequency, damping, jacobian, resonance
        - ecemath code: circuit dynamics, state space, oscillation, regime
        """
        eig_real = [round(float(np.real(e)), 2) for e in self.eigenvalues[:4]]
        eig_imag = [round(float(np.imag(e)), 2) for e in self.eigenvalues[:4]]
        has_oscillation = any(abs(im) > 1e-6 for im in eig_imag)

        parts = [
            f"circuit {self.circuit_type}",
            f"eigenvalue stability {'stable' if self.stable else 'unstable'}",
            f"real eigenvalue {' '.join(str(r) for r in eig_real)}",
        ]
        if has_oscillation:
            parts += [
                f"imaginary eigenvalue oscillation resonance {' '.join(str(i) for i in eig_imag)}",
                f"natural frequency {round(self.natural_freq, 2)} Hz",
                f"damping ratio {round(self.damping_ratio, 4)}",
            ]
        else:
            parts.append(f"time constant {round(self.time_constant, 6)} seconds overdamped")

        parts += [
            f"state space dimension {self.n_states}",
            f"jacobian rank {self.n_states}",
            f"dynamical system linear MNA eigenvalue spectrum",
        ]

        # Add physics keywords based on circuit type
        if self.circuit_type == "rc":
            parts.append("first order low pass filter exponential decay")
        elif self.circuit_type == "rlc":
            parts.append("second order resonance bandwidth quality factor Q")
        elif self.circuit_type == "lc":
            parts.append("lossless tank oscillator undamped resonance")
        elif self.circuit_type == "spring_mass":
            parts.append("mechanical vibration natural mode eigenvector")

        return " ".join(parts)

    def to_domain_keywords(self) -> List[str]:
        """Keywords for domain classification."""
        kw = ["eigenvalue", "stability", "dynamics", "jacobian", "spectrum"]
        if self.circuit_type in ("rlc", "lc"):
            kw += ["resonance", "frequency", "oscillation", "filter"]
        if self.circuit_type == "spring_mass":
            kw += ["mechanical", "vibration", "modal", "structural"]
        if not self.stable:
            kw.append("unstable")
        return kw


# ── Simulation runner ────────────────────────────────────────────────────────

class CircuitSimulator:
    """
    Runs ecemath circuit simulations across parameter sweeps.

    Produces SimulationResult objects that capture:
    - Eigenvalue structure (natural frequencies, damping)
    - Stability classification
    - Jacobian rank and operator structure

    These map directly to the LCA/non-abelian/chaotic classification
    from LOGIC_FLOW.md Section 0D.
    """

    def simulate_rc(self, R: float, C: float) -> Optional[SimulationResult]:
        """Simulate RC circuit with given R, C values."""
        try:
            from src.backend.graph_bridge import ir_to_dynamical_system
            ir = _build_rc_circuit(R, C)
            system, node_map, _ = ir_to_dynamical_system(ir)
            n = max(node_map.values()) + 1
            v0 = np.zeros(n)
            v_eq, ok = system.find_equilibrium(v0)
            if not ok:
                return None
            eigs = system.eigenvalues(v_eq)
            stable = system.is_stable(v_eq)
            tau = 1.0 / abs(float(np.real(eigs[0]))) if len(eigs) > 0 and abs(np.real(eigs[0])) > 1e-12 else float("inf")
            return SimulationResult(
                circuit_type="rc",
                params={"R": R, "C": C},
                eigenvalues=eigs,
                stable=stable,
                n_states=len(eigs),
                natural_freq=0.0,
                damping_ratio=1.0,
                time_constant=tau,
            )
        except Exception:
            return None

    def simulate_rlc(self, R: float, L: float, C: float) -> Optional[SimulationResult]:
        """Simulate RLC circuit — second-order with resonance."""
        try:
            from src.backend.graph_bridge import ir_to_dynamical_system
            ir = _build_rlc_circuit(R, L, C)
            system, node_map, _ = ir_to_dynamical_system(ir)
            n = max(node_map.values()) + 1
            v0 = np.zeros(n)
            v_eq, ok = system.find_equilibrium(v0)
            if not ok:
                return None
            eigs = system.eigenvalues(v_eq)
            stable = system.is_stable(v_eq)
            # Analytical: ω₀ = 1/√(LC), ζ = R/(2)√(C/L)
            omega_0 = 1.0 / np.sqrt(L * C) if L > 0 and C > 0 else 0.0
            zeta = (R / 2.0) * np.sqrt(C / L) if L > 0 and C > 0 else 0.0
            f0 = omega_0 / (2 * np.pi)
            tau = 2.0 * L / R if R > 0 else float("inf")
            return SimulationResult(
                circuit_type="rlc",
                params={"R": R, "L": L, "C": C},
                eigenvalues=eigs,
                stable=stable,
                n_states=len(eigs),
                natural_freq=f0,
                damping_ratio=zeta,
                time_constant=tau,
            )
        except Exception:
            return None

    def simulate_lc(self, L: float, C: float) -> Optional[SimulationResult]:
        """Simulate ideal LC tank — purely imaginary eigenvalues."""
        try:
            from src.backend.graph_bridge import ir_to_dynamical_system
            ir = _build_lc_circuit(L, C)
            system, node_map, _ = ir_to_dynamical_system(ir)
            n = max(node_map.values()) + 1
            v0 = np.zeros(n)
            v_eq, ok = system.find_equilibrium(v0)
            if not ok:
                return None
            eigs = system.eigenvalues(v_eq)
            stable = True  # LC is marginally stable (imaginary axis)
            omega_0 = 1.0 / np.sqrt(L * C) if L > 0 and C > 0 else 0.0
            f0 = omega_0 / (2 * np.pi)
            return SimulationResult(
                circuit_type="lc",
                params={"L": L, "C": C},
                eigenvalues=eigs,
                stable=stable,
                n_states=len(eigs),
                natural_freq=f0,
                damping_ratio=0.0,
                time_constant=float("inf"),
            )
        except Exception:
            return None

    def sweep_rc(self, n_R: int = 4, n_C: int = 4) -> List[SimulationResult]:
        """Parameter sweep over RC circuit (log-spaced values)."""
        Rs = np.logspace(2, 6, n_R)       # 100Ω to 1MΩ
        Cs = np.logspace(-12, -6, n_C)    # 1pF to 1µF
        results = []
        for R, C in itertools.product(Rs, Cs):
            r = self.simulate_rc(R, C)
            if r is not None:
                results.append(r)
        return results

    def sweep_rlc(self, n_R: int = 3, n_L: int = 3, n_C: int = 3) -> List[SimulationResult]:
        """Parameter sweep over RLC circuit."""
        Rs = np.logspace(1, 4, n_R)       # 10Ω to 10kΩ
        Ls = np.logspace(-6, -3, n_L)     # 1µH to 1mH
        Cs = np.logspace(-9, -6, n_C)     # 1nF to 1µF
        results = []
        for R, L, C in itertools.product(Rs, Ls, Cs):
            r = self.simulate_rlc(R, L, C)
            if r is not None:
                results.append(r)
        return results

    def sweep_lc(self, n_L: int = 4, n_C: int = 4) -> List[SimulationResult]:
        """Sweep LC tank circuits."""
        Ls = np.logspace(-6, -3, n_L)
        Cs = np.logspace(-9, -6, n_C)
        results = []
        for L, C in itertools.product(Ls, Cs):
            r = self.simulate_lc(L, C)
            if r is not None:
                results.append(r)
        return results


class MechanicalSimulator:
    """
    Simulates mechanical mass-spring-damper systems.

    Equation: m·ẍ + b·ẋ + k·x = 0
    State: [x, ẋ], matrix A = [[0, 1], [-k/m, -b/m]]
    Eigenvalues: λ = -b/(2m) ± √((b/(2m))² - k/m)

    This is isomorphic to the RLC circuit (R↔b, L↔m, C↔1/k).
    The system should automatically discover this cross-domain equivalence
    when RLC results and spring-mass results are both in the 'physical' HDV.
    """

    def simulate(self, m: float, b: float, k: float) -> Optional[SimulationResult]:
        """Simulate mass-spring-damper analytically."""
        try:
            # A = [[0, 1], [-k/m, -b/m]]
            A = np.array([[0.0, 1.0], [-k / m, -b / m]])
            eigs = np.linalg.eigvals(A)
            stable = all(np.real(e) < 0 for e in eigs)
            # Natural frequency and damping
            omega_0 = np.sqrt(k / m) if k > 0 and m > 0 else 0.0
            zeta = b / (2.0 * np.sqrt(m * k)) if m > 0 and k > 0 else 0.0
            f0 = omega_0 / (2 * np.pi)
            tau = 2.0 * m / b if b > 0 else float("inf")
            return SimulationResult(
                circuit_type="spring_mass",
                params={"m": m, "b": b, "k": k},
                eigenvalues=eigs,
                stable=stable,
                n_states=2,
                natural_freq=f0,
                damping_ratio=zeta,
                time_constant=tau,
                jacobian=A,
            )
        except Exception:
            return None

    def sweep(self, n: int = 4) -> List[SimulationResult]:
        """Parameter sweep over mass-spring-damper."""
        masses = np.logspace(-3, 1, n)       # 1g to 10kg
        dampings = np.logspace(-2, 2, n)     # 0.01 to 100 Ns/m
        stiffnesses = np.logspace(0, 4, n)   # 1 to 10000 N/m
        results = []
        for m, b, k in itertools.product(masses, dampings, stiffnesses):
            r = self.simulate(m, b, k)
            if r is not None:
                results.append(r)
        return results


# ── HDV encoding for physical simulation results ─────────────────────────────

class PhysicalHDVPopulator:
    """
    Encodes simulation results as HDV vectors in the 'physical' dimension.

    Analogous to GeometricHDVPopulator (LaTeX structure → HDV) but for
    numerical simulation results.

    The rich text description of eigenvalues, stability, and circuit type
    shares vocabulary with:
    - arXiv math papers (eigenvalue, stability, resonance, Jacobian)
    - ecemath code functions (stability_analysis, eigenvalue_spectrum)
    Enabling cross-dimensional universal discovery.
    """

    def __init__(self, hdv_system, discovery=None):
        self.hdv_system = hdv_system
        self.discovery = discovery
        self._encoded: List[Dict] = []

    def encode_result(self, result: SimulationResult) -> np.ndarray:
        """Encode a SimulationResult as a physical-domain HDV vector."""
        text = result.to_text_description()
        vec = self.hdv_system.structural_encode(text, "physical")

        # Record to discovery system if available
        if self.discovery is not None:
            self.discovery.record_pattern(
                "physical",
                vec,
                {
                    "type": "simulation",
                    "circuit": result.circuit_type,
                    "params": result.params,
                    "stable": result.stable,
                    "n_states": result.n_states,
                    "natural_freq": result.natural_freq,
                    "damping_ratio": result.damping_ratio,
                    "time_constant": result.time_constant,
                    "eigenvalue_summary": [
                        f"{np.real(e):.3f}+{np.imag(e):.3f}j"
                        for e in result.eigenvalues[:4]
                    ],
                },
            )
            self._encoded.append({
                "text": text,
                "circuit": result.circuit_type,
                "params": result.params,
            })

        return vec

    def populate_all_circuits(
        self,
        n_sweep: int = 4,
    ) -> int:
        """
        Run complete simulation sweep across all circuit types and encode results.

        Args:
            n_sweep: Number of points per parameter axis (total = n_sweep^k)

        Returns:
            Number of encoded simulation results.
        """
        circuit_sim = CircuitSimulator()
        mech_sim = MechanicalSimulator()
        total = 0

        # RC circuits
        for r in circuit_sim.sweep_rc(n_R=n_sweep, n_C=n_sweep):
            self.encode_result(r)
            total += 1

        # RLC circuits
        for r in circuit_sim.sweep_rlc(n_R=max(n_sweep-1, 2), n_L=max(n_sweep-1, 2), n_C=max(n_sweep-1, 2)):
            self.encode_result(r)
            total += 1

        # LC circuits
        for r in circuit_sim.sweep_lc(n_L=n_sweep, n_C=n_sweep):
            self.encode_result(r)
            total += 1

        # Mass-spring-damper (proof-of-concept cross-domain equivalence)
        for r in mech_sim.sweep(n=max(n_sweep-1, 2)):
            self.encode_result(r)
            total += 1

        return total

    def find_cross_domain_equivalences(self, top_k: int = 5) -> List[Dict]:
        """
        Find simulation results most similar to each other across domain types.

        Returns list of (result_a_text, result_b_text, similarity) dicts
        for the most similar cross-type pairs — these are the PoC cross-domain
        path demonstrations from LOGIC_FLOW.md Section 9.
        """
        if len(self._encoded) < 2:
            return []

        vecs = []
        for enc in self._encoded:
            v = self.hdv_system.structural_encode(enc["text"], "physical")
            vecs.append(v)

        # Pairwise cosine similarity (only cross-type pairs)
        from itertools import combinations
        pairs = []
        for (i, a), (i2, b) in combinations(enumerate(self._encoded), 2):
            if a["circuit"] == b["circuit"]:
                continue  # skip same circuit type
            va, vb = vecs[i], vecs[i2]
            norm = (np.linalg.norm(va) * np.linalg.norm(vb))
            if norm < 1e-10:
                continue
            sim = float(np.dot(va, vb) / norm)
            pairs.append({
                "circuit_a": a["circuit"],
                "circuit_b": b["circuit"],
                "params_a": a["params"],
                "params_b": b["params"],
                "similarity": sim,
            })

        pairs.sort(key=lambda x: -x["similarity"])
        return pairs[:top_k]


# ── SimulationTrainer: orchestrator for the autonomous loop ──────────────────

class SimulationTrainer:
    """
    Orchestrator that runs simulation sweeps and populates the 'physical' HDV.

    Designed to be called from AutonomousLearningSystem as a thread.

    Training strategy:
    1. RC circuits → first-order exponential decay patterns
    2. RLC circuits → second-order resonance patterns (Q, bandwidth, damping)
    3. LC circuits → undamped oscillation (purely imaginary eigenvalues)
    4. Spring-mass → mechanical vibration (isomorphic to RLC)
    5. find_cross_domain_equivalences() to verify circuit ↔ mechanical match

    After population, CrossDimensionalDiscovery should find universals between
    'physical' and 'math' dimensions because the text descriptions share
    vocabulary with arXiv equation papers on stability and eigenvalues.
    """

    def __init__(self, hdv_system, discovery, domain_registry=None):
        self.hdv_system = hdv_system
        self.discovery = discovery
        self.domain_registry = domain_registry
        self.populator = PhysicalHDVPopulator(hdv_system, discovery)
        self._total_encoded = 0
        self._equivalences: List[Dict] = []

    def run_one_pass(self, n_sweep: int = 3) -> Dict:
        """
        Run one full simulation sweep.

        Args:
            n_sweep: Number of points per parameter axis.

        Returns:
            Summary dict: {encoded, equivalences, elapsed_s}
        """
        t0 = time.time()

        # Activate relevant domains
        if self.domain_registry is not None:
            for text in [
                "circuit semiconductor eigenvalue stability",
                "mechanical vibration structural health dynamics",
                "power grid energy demand electrical systems",
                "thermal optimization heat transfer PDE",
            ]:
                try:
                    self.domain_registry.activate_for_text(text, self.hdv_system)
                except Exception:
                    pass

        n = self.populator.populate_all_circuits(n_sweep=n_sweep)
        self._total_encoded += n

        self._equivalences = self.populator.find_cross_domain_equivalences(top_k=5)

        elapsed = time.time() - t0
        return {
            "encoded": n,
            "total_encoded": self._total_encoded,
            "top_equivalences": self._equivalences,
            "elapsed_s": round(elapsed, 2),
        }

    def print_report(self, result: Dict) -> None:
        """Print a human-readable summary of the simulation training pass."""
        print(f"\n[SimTrainer] Encoded {result['encoded']} simulation results "
              f"({result['elapsed_s']:.1f}s)")
        if result["top_equivalences"]:
            print("[SimTrainer] Top cross-type similarities:")
            for eq in result["top_equivalences"][:3]:
                print(
                    f"  {eq['circuit_a']} ↔ {eq['circuit_b']}  "
                    f"sim={eq['similarity']:.3f}"
                )
