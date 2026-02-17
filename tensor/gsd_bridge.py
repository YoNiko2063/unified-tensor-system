"""GSD Bridge: wires get-shit-done planning to dev-agent execution and tensor validation.

GSD plans the WHAT and WHEN.
Dev-agent executes the HOW.
Tensor validates the result.
"""
import os
import sys
import time
import json
import uuid
import subprocess
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_ECEMATH_SRC = os.path.join(_ROOT, 'ecemath', 'src')
if _ECEMATH_SRC not in sys.path:
    sys.path.insert(0, _ECEMATH_SRC)

_DEV_AGENT_SRC = Path(_ROOT, 'dev-agent/src').resolve()
if str(_DEV_AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(_DEV_AGENT_SRC))

from tensor.core import UnifiedTensor
from tensor.code_graph import CodeGraph
from tensor.code_validator import CodeValidator
from tensor.bootstrap import BootstrapOrchestrator
from tensor.math_connections import fisher_guided_planning


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

        # Compute φ-weight for this phase's priority index
        guidance = fisher_guided_planning(self.tensor, level=2)
        task_index = phase - 1
        phi_weight = float(guidance.phi_weights[task_index]) if (
            guidance.phi_weights is not None and task_index < len(guidance.phi_weights)
        ) else 0.0

        self._phases[phase] = [{'xml': t, 'module': name, 'fe': fe,
                                 'phi_weight': phi_weight} for t in tasks]

        # Write plan file
        os.makedirs(self.planning_dir, exist_ok=True)
        plan_path = os.path.join(self.planning_dir, f'{phase:02d}-01-PLAN.md')
        plan_content = f"# Phase {phase} Plan: Improve {name}\n\n"
        for t in tasks:
            plan_content += t + "\n\n"
        with open(plan_path, 'w') as f:
            f.write(plan_content)

        return tasks

    def _find_dev_agent_cli(self) -> Optional[str]:
        """Locate the dev-agent CLI executable."""
        candidates = [
            os.path.join(self.dev_agent_root, 'run.py'),
            os.path.join(self.dev_agent_root, 'dev-agent', 'run.py'),
            os.path.join(self.dev_agent_root, 'app.py'),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
        return None

    def execute_phase(self, phase: int) -> PhaseResult:
        """Execute all tasks in a phase via dev-agent CLI.

        For each task:
          1. Snapshot consonance as plain float BEFORE mutation
          2. Invoke dev-agent via subprocess with task JSON
          3. Re-parse affected files via CodeGraph
          4. Update tensor L2
          5. Snapshot consonance as plain float AFTER mutation
          6. If consonance degraded: git checkout modified path
        """
        # Snapshot phase-level consonance before any task runs
        cons_phase_start = float(self.tensor.harmonic_signature(2).consonance_score)

        tasks = self._phases.get(phase, [])
        files_changed = []
        tasks_completed = 0
        tasks_failed = 0

        for task_idx, task_info in enumerate(tasks):
            # Snapshot per-task consonance as plain float before execution
            cons_before = float(self.tensor.harmonic_signature(2).consonance_score)

            module_name = task_info.get('module', '')
            phi_weight = task_info.get('phi_weight', 0.0)

            # Build autonomy proposal for dev-agent
            from dev_agent.autonomy.autonomy_loop import run_autonomy_tick

            proposal_id = f"tensor-gsd-{uuid.uuid4().hex[:8]}"
            proposal = {
                "artifact_type": "proposal",
                "schema_version": "1.1",
                "proposal_id": proposal_id,
                "proposal_class": "adaptive",
                "title": f"Tensor-guided improvement: {module_name}",
                "derived_from": {
                    "intent_version": "tensor_fisher_guided",
                    "failure_aggregate": "tensor_gsd_bridge"
                },
                "justification": {
                    "evidence": (f"FIM priority module: {module_name}, "
                                 f"phi_weight: {phi_weight:.4f}, "
                                 f"consonance: {cons_before:.4f}"),
                    "aggregate_refs": ["tensor:l2_eigenvalue", "tensor:fisher_fim"]
                },
                "scope": {
                    "modules": [module_name.replace('.', '/')],
                    "invariants_unchanged": [
                        "no_execution_without_approval",
                        "read_only_introspection"
                    ]
                },
                "risk": {
                    "level": "low",
                    "rollback": "git revert"
                },
                "status": "approved"
            }
            proposal_path = Path('dev-agent/logs/proposals') / f"{proposal_id}.json"
            proposal_path.write_text(json.dumps(proposal, indent=2))

            try:
                result = run_autonomy_tick(project_root=Path('dev-agent').resolve())
                success = result.get('status') not in ('error', 'failed')
            except Exception as e:
                print(f"[GSD] autonomy_tick error: {e}")
                success = False

            # Re-parse codebase and update L2
            cg = CodeGraph.from_directory(self.dev_agent_root, max_files=500)
            new_mna = cg.to_mna()
            self.tensor.update_level(2, new_mna, t=time.time())

            # Snapshot per-task consonance as plain float after update_level
            cons_after = float(self.tensor.harmonic_signature(2).consonance_score)
            cons_delta = cons_after - cons_before

            # Track growth regime from L2 harmonic signature
            sig_after = self.tensor.harmonic_signature(2)
            growth_count = getattr(sig_after, 'growth_regime_count', 0)

            if cons_delta > 0:
                tasks_completed += 1
                files_changed.append(module_name)
                print(f"[GSD] Task '{module_name}': consonance "
                      f"{cons_before:.4f}->{cons_after:.4f} (+{cons_delta:.4f}) KEPT "
                      f"growth_regime={growth_count}")
            else:
                # Revert: git checkout modified path
                mod_info = cg.modules.get(module_name)
                if mod_info is not None:
                    try:
                        subprocess.run(
                            ['git', 'checkout', mod_info.path],
                            capture_output=True, timeout=30,
                            cwd=self.dev_agent_root,
                        )
                        print(f"[GSD] Task '{module_name}': consonance "
                              f"{cons_before:.4f}->{cons_after:.4f} ({cons_delta:.4f}) REVERTED "
                              f"growth_regime={growth_count}")
                    except (subprocess.TimeoutExpired, FileNotFoundError):
                        pass
                tasks_failed += 1

        # Final consonance and growth regime snapshot as plain floats
        cons_final = float(self.tensor.harmonic_signature(2).consonance_score)
        sig_final = self.tensor.harmonic_signature(2)
        growth_final = getattr(sig_final, 'growth_regime_count', 0)

        phase_result = PhaseResult(
            phase=phase,
            tasks_completed=tasks_completed,
            tasks_failed=tasks_failed,
            consonance_before=cons_phase_start,
            consonance_after=cons_final,
            files_changed=files_changed,
        )
        self._phase_results[phase] = phase_result

        print(f"[GSD] Phase {phase}: consonance {cons_phase_start:.4f}->{cons_final:.4f} "
              f"{'APPROVED' if cons_final >= cons_phase_start else 'DEGRADED'} "
              f"growth_regime={growth_final}")

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
