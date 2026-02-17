"""Skill writer: documents successful improvements as reusable SKILL.md files.

Follows the obsidian-skills / Agent Skills specification:
  - YAML frontmatter with name + description
  - Markdown body with trigger conditions, intervention, measurements
"""
import os
import sys
import json
import time
from typing import Dict, List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tensor.bootstrap import BootstrapResult
from tensor.gsd_bridge import PhaseResult


class SkillWriter:
    """Documents successful improvements as reusable skills.

    Skills are stored in: skills/tensor-learned/
    Each skill = one pattern that reliably reduces free energy.
    """

    def __init__(self, skills_dir: str = 'skills/tensor-learned',
                 log_dir: str = 'tensor/logs'):
        self.skills_dir = os.path.abspath(skills_dir)
        self.log_dir = os.path.abspath(log_dir)
        self._skill_results: Dict[str, List[bool]] = {}

    def write_skill(self, improvement: BootstrapResult,
                    phase_result: Optional[PhaseResult] = None) -> str:
        """Write a SKILL.md after a successful improvement.

        Returns path to the created skill file.
        """
        if not improvement.improved:
            return ''

        # Derive pattern name from high-tension nodes
        if improvement.high_tension_nodes:
            primary = improvement.high_tension_nodes[0]
            pattern_name = primary.replace('.', '_').replace('/', '_')
        else:
            pattern_name = f'improvement_step_{improvement.step}'

        skill_dir = os.path.join(self.skills_dir, pattern_name)
        os.makedirs(skill_dir, exist_ok=True)

        # Determine intervention type
        if improvement.files_changed:
            intervention = 'file-split'
            intervention_desc = (
                f'Split/refactored files to reduce structural tension. '
                f'Changed {len(improvement.files_changed)} file(s).'
            )
        else:
            intervention = 'structural-rebalance'
            intervention_desc = 'Rebalanced module dependencies to reduce coupling.'

        # Build SKILL.md in obsidian-skills format
        skill_md = f"""---
name: {pattern_name}
description: >
  Tensor-learned pattern: {intervention} applied to reduce L2 free energy.
  Consonance improvement: {improvement.consonance_delta:+.4f}
  (from {improvement.consonance_before:.4f} to {improvement.consonance_after:.4f}).
---

# Skill: {pattern_name}

## Overview

This skill was learned by the tensor bootstrap system at step {improvement.step}.
It documents a successful structural improvement pattern.

## Trigger Conditions

Apply this skill when:
- L2 free energy is high for modules matching: {', '.join(improvement.high_tension_nodes[:3])}
- Module has similar structural characteristics (coupling pattern, file size)
- Current consonance is below {improvement.consonance_after:.4f}

## Intervention

**Type**: {intervention}

{intervention_desc}

### Files Changed

"""
        for f in improvement.files_changed[:10]:
            skill_md += f'- `{f}`\n'

        skill_md += f"""
## Measurements

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Consonance | {improvement.consonance_before:.4f} | {improvement.consonance_after:.4f} | {improvement.consonance_delta:+.4f} |
"""

        if phase_result:
            skill_md += f"""
### Phase Execution

- Tasks completed: {phase_result.tasks_completed}
- Tasks failed: {phase_result.tasks_failed}
- Phase consonance: {phase_result.consonance_before:.4f} -> {phase_result.consonance_after:.4f}
"""

        skill_md += f"""
## When NOT to Apply

- If the module is already below average free energy
- If consonance is already above 0.75 (stable octave)
- If the change would increase circular dependencies

## References

- Tensor level: L2 (code structure)
- Bootstrap step: {improvement.step}
- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}
"""

        skill_path = os.path.join(skill_dir, 'SKILL.md')
        with open(skill_path, 'w') as f:
            f.write(skill_md)

        # Track success
        self._skill_results.setdefault(pattern_name, []).append(True)

        # Log to JSONL
        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(self.log_dir, 'skills_log.jsonl')
        log_entry = {
            'pattern': pattern_name,
            'intervention': intervention,
            'consonance_delta': improvement.consonance_delta,
            'step': improvement.step,
            'files_changed': improvement.files_changed,
            'timestamp': time.time(),
        }
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        return skill_path

    def record_failure(self, pattern_name: str):
        """Record that applying a skill did NOT reduce free energy."""
        self._skill_results.setdefault(pattern_name, []).append(False)

    def skill_library(self) -> List[dict]:
        """List all learned skills with their success rates."""
        skills = []

        # From tracked results
        for pattern, results in self._skill_results.items():
            total = len(results)
            successes = sum(results)
            rate = successes / total if total > 0 else 0.0
            skills.append({
                'pattern': pattern,
                'total_applications': total,
                'successes': successes,
                'success_rate': rate,
                'flagged': rate < 0.5,
            })

        # Also scan skills directory for any on-disk skills not in memory
        if os.path.isdir(self.skills_dir):
            for entry in os.listdir(self.skills_dir):
                skill_path = os.path.join(self.skills_dir, entry, 'SKILL.md')
                if os.path.exists(skill_path) and entry not in self._skill_results:
                    skills.append({
                        'pattern': entry,
                        'total_applications': 0,
                        'successes': 0,
                        'success_rate': 0.0,
                        'flagged': False,
                    })

        return skills
