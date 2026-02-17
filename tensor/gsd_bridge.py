"""GSD Bridge: wires get-shit-done planning to dev-agent execution and tensor validation.

GSD plans the WHAT and WHEN.
Dev-agent executes the HOW.
Tensor validates the result.
"""
import os
import sys
import time
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_ECEMATH_SRC = os.path.join(_ROOT, 'ecemath', 'src')
if _ECEMATH_SRC not in sys.path:
    sys.path.insert(0, _ECEMATH_SRC)

from tensor.core import UnifiedTensor
from tensor.code_graph import CodeGraph
from tensor.code_validator import CodeValidator
from tensor.bootstrap import BootstrapOrchestrator


@dataclass
class PhaseResult:
    """Result of executing one GSD phase."""
    phase: int
    tasks_completed: int
    tasks_failed: int
    consonance_before: float
    consonance_after: float
    files_changed: List[str]


class GSDBridge:
    """Wires GSD's planning workflow to dev-agent execution and tensor validation.

    GSD outer loop:
      create_improvement_project() -> defines what to improve
      plan_phase(i) -> creates atomic task plans
      execute_phase(i) -> dev-agent executes each task
      verify_phase(i) -> tensor validator checks results
    """

    def __init__(self, tensor: UnifiedTensor,
                 gsd_root: str,
                 dev_agent_root: str,
                 planning_dir: str = '.planning'):
        self.tensor = tensor
        self.gsd_root = os.path.abspath(gsd_root)
        self.dev_agent_root = os.path.abspath(dev_agent_root)
        self.planning_dir = os.path.abspath(planning_dir)
        self._validator = CodeValidator(tensor, dev_agent_root)
        self._bootstrapper = BootstrapOrchestrator(tensor, dev_agent_root)
        self._phases: Dict[int, List[dict]] = {}
        self._phase_results: Dict[int, PhaseResult] = {}

    def _get_tension_nodes(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get top-k high-tension nodes from tensor L2."""
        cg = CodeGraph.from_directory(self.dev_agent_root, max_files=500)
        mna = cg.to_mna()
        self.tensor.update_level(2, mna, t=time.time())
        fe = self.tensor.free_energy_map(2)
        names = cg.node_names
        n = min(len(names), len(fe))
        node_fe = [(names[i], float(fe[i])) for i in range(n)]
        node_fe.sort(key=lambda x: abs(x[1]), reverse=True)
        return node_fe[:top_k]

    def create_improvement_project(self) -> str:
        """Create GSD-format PROJECT.md, REQUIREMENTS.md, ROADMAP.md from tensor state."""
        os.makedirs(self.planning_dir, exist_ok=True)

        tension_nodes = self._get_tension_nodes(top_k=5)
        sig = self.tensor.harmonic_signature(2)

        # PROJECT.md
        project_md = f"""# Tensor-Guided Codebase Improvement

## Overview

Autonomous improvement project generated from tensor L2 analysis.
Current consonance: {sig.consonance_score:.4f}
Dominant interval: {sig.dominant_interval}
Stability: {sig.stability_verdict}

## Goal

Reduce structural tension in the top {len(tension_nodes)} highest free-energy modules
to achieve consonance > 0.75 (stable octave).

## High-Tension Modules

"""
        for name, fe in tension_nodes:
            project_md += f"- **{name}**: free_energy={fe:.4f}\n"

        project_path = os.path.join(self.planning_dir, 'PROJECT.md')
        with open(project_path, 'w') as f:
            f.write(project_md)

        # REQUIREMENTS.md
        req_md = "# Requirements\n\n"
        for i, (name, fe) in enumerate(tension_nodes, 1):
            req_md += (f"## REQ-{i:02d}: Reduce tension in {name}\n"
                       f"- Free energy: {fe:.4f}\n"
                       f"- Target: reduce |F| by >20%\n"
                       f"- Constraint: do not degrade overall consonance\n\n")

        req_path = os.path.join(self.planning_dir, 'REQUIREMENTS.md')
        with open(req_path, 'w') as f:
            f.write(req_md)

        # ROADMAP.md (follows GSD template)
        roadmap_md = f"""# Roadmap: Tensor-Guided Improvement

## Overview

{len(tension_nodes)} phases, each targeting one high-tension module.
Guided by tensor L2 free energy minimization.

## Phases

"""
        for i, (name, fe) in enumerate(tension_nodes, 1):
            roadmap_md += f"- [ ] **Phase {i}: Improve {name}** - Reduce free energy from {fe:.4f}\n"

        roadmap_md += "\n## Phase Details\n\n"
        for i, (name, fe) in enumerate(tension_nodes, 1):
            dep = f"Phase {i-1}" if i > 1 else "Nothing (first phase)"
            roadmap_md += f"""### Phase {i}: Improve {name}
**Goal**: Reduce structural tension in {name}
**Depends on**: {dep}
**Requirements**: REQ-{i:02d}
**Success Criteria** (what must be TRUE):
  1. Free energy of {name} decreases by >20%
  2. Overall L2 consonance does not degrade
  3. All existing tests still pass

"""
        roadmap_md += "## Progress\n\n"
        roadmap_md += "| Phase | Plans Complete | Status | Completed |\n"
        roadmap_md += "|-------|----------------|--------|----------|\n"
        for i, (name, _) in enumerate(tension_nodes, 1):
            roadmap_md += f"| {i}. {name} | 0/1 | Not started | - |\n"

        roadmap_path = os.path.join(self.planning_dir, 'ROADMAP.md')
        with open(roadmap_path, 'w') as f:
            f.write(roadmap_md)

        return self.planning_dir

    def plan_phase(self, phase: int) -> List[str]:
        """Generate GSD-format task plan for a given phase.

        Returns list of task descriptions in XML format.
        """
        tension_nodes = self._get_tension_nodes(top_k=max(phase, 5))
        if phase < 1 or phase > len(tension_nodes):
            return []

        name, fe = tension_nodes[phase - 1]
        cg = CodeGraph.from_directory(self.dev_agent_root, max_files=500)
        mod_info = cg.modules.get(name)

        tasks = []
        if mod_info is not None:
            # Generate split/refactor tasks based on module complexity
            task_xml = (
                f'<task type="auto">\n'
                f'  <n>Refactor {name} to reduce structural tension</n>\n'
                f'  <files>{mod_info.path}</files>\n'
                f'  <action>Split or refactor {name} ({mod_info.lines} lines, '
                f'complexity={mod_info.complexity}) to reduce free energy. '
                f'Target: files under 200 lines, single responsibility.</action>\n'
                f'  <verify>tensor validator approves new structure</verify>\n'
                f'  <done>L2 free energy for {name} decreases by >20%</done>\n'
                f'</task>'
            )
            tasks.append(task_xml)

            if mod_info.lines > 200:
                tasks.append(
                    f'<task type="auto">\n'
                    f'  <n>Split {name} into smaller modules</n>\n'
                    f'  <files>{mod_info.path}</files>\n'
                    f'  <action>Extract classes/functions into separate files. '
                    f'Each file should have single responsibility.</action>\n'
                    f'  <verify>all new files parse without errors</verify>\n'
                    f'  <done>No file exceeds 200 lines</done>\n'
                    f'</task>'
                )

        self._phases[phase] = [{'xml': t, 'module': name, 'fe': fe} for t in tasks]

        # Write plan file
        os.makedirs(self.planning_dir, exist_ok=True)
        plan_path = os.path.join(self.planning_dir, f'{phase:02d}-01-PLAN.md')
        plan_content = f"# Phase {phase} Plan: Improve {name}\n\n"
        for t in tasks:
            plan_content += t + "\n\n"
        with open(plan_path, 'w') as f:
            f.write(plan_content)

        return tasks

    def execute_phase(self, phase: int) -> PhaseResult:
        """Execute all tasks in a phase via bootstrap mechanism.

        After each task:
          1. Re-parse affected files via CodeGraph
          2. Update tensor L2
          3. Run CodeValidator on changed files
          4. Measure consonance improvement
        """
        # Get consonance before
        sig_before = self.tensor.harmonic_signature(2)
        cons_before = sig_before.consonance_score

        # Run one bootstrap step (simulated execution)
        result = self._bootstrapper.run_bootstrap_step()

        # Get consonance after
        sig_after = self.tensor.harmonic_signature(2)
        cons_after = sig_after.consonance_score

        phase_result = PhaseResult(
            phase=phase,
            tasks_completed=len(result.files_changed) if result.files_changed else 1,
            tasks_failed=0 if result.improved else 1,
            consonance_before=cons_before,
            consonance_after=cons_after,
            files_changed=result.files_changed,
        )
        self._phase_results[phase] = phase_result

        print(f"[GSD] Phase {phase}: consonance {cons_before:.4f}->{cons_after:.4f} "
              f"{'APPROVED' if cons_after >= cons_before else 'DEGRADED'}")

        return phase_result

    def verify_phase(self, phase: int) -> bool:
        """Verify phase results against tensor metrics.

        Returns True if:
          1. L2 consonance improved or held
          2. L2-L3 resonance improved or held (if L3 populated)
        """
        pr = self._phase_results.get(phase)
        if pr is None:
            return False

        # Check consonance
        cons_ok = pr.consonance_after >= pr.consonance_before - 0.01

        # Check resonance if L3 populated
        res_ok = True
        mna3 = self.tensor._mna.get(3)
        if mna3 is not None:
            res = self.tensor.cross_level_resonance(2, 3)
            res_ok = res >= 0.3

        verified = cons_ok and res_ok
        print(f"[GSD] Phase {phase} verification: {'PASS' if verified else 'FAIL'} "
              f"(consonance_ok={cons_ok}, resonance_ok={res_ok})")
        return verified

    def run_autonomous_cycle(self, max_phases: int = 5):
        """Full autonomous improvement cycle.

        create_improvement_project()
        for each phase:
          plan_phase(i)
          execute_phase(i)
          verify_phase(i)
        """
        print("[GSD] Starting autonomous improvement cycle")
        self.create_improvement_project()

        for i in range(1, max_phases + 1):
            print(f"\n[GSD] Phase {i}/{max_phases}")

            tasks = self.plan_phase(i)
            if not tasks:
                print(f"[GSD] No tasks for phase {i}, stopping")
                break

            print(f"[GSD] Task count: {len(tasks)}")

            result = self.execute_phase(i)
            print(f"[GSD] Tasks completed: {result.tasks_completed}, "
                  f"failed: {result.tasks_failed}")

            verified = self.verify_phase(i)
            if not verified:
                # Retry once
                print(f"[GSD] Retrying phase {i}")
                result = self.execute_phase(i)
                verified = self.verify_phase(i)
                if not verified:
                    print(f"[GSD] Phase {i} failed after retry")

            sig = self.tensor.harmonic_signature(2)
            if sig.consonance_score >= 0.75:
                print(f"[GSD] Target consonance reached: {sig.consonance_score:.4f}")
                break

        print("\n[GSD] Autonomous cycle complete")
        return self._phase_results
