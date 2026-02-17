"""Observable tensor state for humans and AI agents.

Single interface to see the full tensor state as markdown
or compressed agent context.
"""
import os
import sys
import json
import time
from datetime import datetime, timezone
from typing import Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tensor.core import UnifiedTensor, LEVEL_NAMES


class TensorObserver:
    """Observe the full tensor state in human/AI-readable formats."""

    def __init__(self, tensor: UnifiedTensor,
                 log_dir: str = 'tensor/logs'):
        self.tensor = tensor
        self.log_dir = log_dir

    def snapshot_markdown(self, priority_map: list = None) -> str:
        """Full tensor state as markdown."""
        snap = self.tensor.tensor_snapshot()
        ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        lines = [f'# Tensor State — {ts}', '']

        # System health
        populated = [l for l in range(self.tensor.n_levels)
                     if snap['levels'][l].get('populated')]
        if populated:
            cons_scores = [snap['levels'][l]['harmonic_signature']['consonance_score']
                           for l in populated]
            avg_cons = sum(cons_scores) / len(cons_scores)
            risks = [snap['levels'][l]['phase_transition_risk'] for l in populated]
            max_risk = max(risks)
            # Dominant key from most populated level
            biggest = max(populated, key=lambda l: snap['levels'][l].get('n_nodes', 0))
            dom_key = snap['levels'][biggest]['harmonic_signature']['dominant_interval']

            if max_risk > 0.8:
                risk_label = 'CRITICAL'
            elif max_risk > 0.6:
                risk_label = 'HIGH'
            elif max_risk > 0.4:
                risk_label = 'MEDIUM'
            else:
                risk_label = 'LOW'

            lines.append('## System Health')
            lines.append(f'- Overall consonance: {avg_cons:.4f}')
            lines.append(f'- Dominant key: {dom_key}')
            lines.append(f'- Phase risk: **{risk_label}** (max={max_risk:.2f})')
            lines.append('')

        # Level status table
        lines.append('## Level Status')
        lines.append('| Level | Nodes | Gap | Risk | Key | Verdict |')
        lines.append('|-------|-------|-----|------|-----|---------|')
        for l in range(self.tensor.n_levels):
            info = snap['levels'][l]
            name = info['name']
            if not info.get('populated'):
                lines.append(f'| L{l} {name} | — | — | — | — | empty |')
            else:
                sig = info['harmonic_signature']
                lines.append(
                    f"| L{l} {name} | {info['n_nodes']} | "
                    f"{info['eigenvalue_gap']:.2f} | "
                    f"{info['phase_transition_risk']:.2f} | "
                    f"{sig['dominant_interval']} | "
                    f"{sig['stability_verdict']} |"
                )
        lines.append('')

        # Cross-level resonance
        pairs = []
        for l in populated:
            for r in info.get('cross_level_resonance', {}).items():
                pass  # handled below
        lines.append('## Cross-Level Resonance')
        lines.append('| Pair | Resonance | Interpretation |')
        lines.append('|------|-----------|----------------|')
        seen = set()
        for l in populated:
            res_map = snap['levels'][l].get('cross_level_resonance', {})
            for other_name, res_val in res_map.items():
                pair_key = tuple(sorted([LEVEL_NAMES.get(l, f'L{l}'), other_name]))
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                if res_val > 0.8:
                    interp = 'Strong alignment'
                elif res_val > 0.5:
                    interp = 'Moderate alignment'
                else:
                    interp = 'Structural mismatch'
                lines.append(
                    f'| {pair_key[0]} ↔ {pair_key[1]} | {res_val:.4f} | {interp} |'
                )
        lines.append('')

        # Improvement priorities (L2)
        if priority_map:
            lines.append('## Improvement Priorities (L2 Code)')
            for i, item in enumerate(priority_map[:10], 1):
                lines.append(
                    f"{i}. **{item['module']}** — "
                    f"weight={item['weight']:.2f}, "
                    f"F={item['free_energy']:.3f}, "
                    f"H={item['harmonic_tension']:.3f}  ")
                lines.append(f"   _{item['reason']}_")
            lines.append('')

        # Active signals
        lines.append('## Active Signals')
        for l in populated:
            info = snap['levels'][l]
            risk = info['phase_transition_risk']
            sig = info['harmonic_signature']
            name = info['name']
            if risk > 0.7:
                lines.append(
                    f"- **{name.upper()} phase risk {risk:.2f}**: "
                    f"{sig['dominant_interval']} dominant, "
                    f"verdict={sig['stability_verdict']}")
            gap = info['eigenvalue_gap']
            if gap < 0.2:
                lines.append(
                    f"- {name} eigenvalue gap narrowing ({gap:.3f}): "
                    f"phase transition imminent")
        # Cross-level signals
        for pair_key in seen:
            a, b = pair_key
            # Find resonance value
            for l in populated:
                res_map = snap['levels'][l].get('cross_level_resonance', {})
                if a in res_map:
                    val = res_map[a]
                elif b in res_map:
                    val = res_map[b]
                else:
                    continue
                if val < 0.5:
                    lines.append(
                        f"- {a}↔{b} resonance LOW ({val:.2f}): structural mismatch")
                break

        if not any(line.startswith('- ') for line in lines[-5:]):
            lines.append('- No critical signals at this time')

        lines.append('')
        return '\n'.join(lines)

    def to_agent_context(self) -> str:
        """Compressed tensor state under 500 tokens for agent context."""
        snap = self.tensor.tensor_snapshot()
        parts = ['[TENSOR STATE]']

        populated = [l for l in range(self.tensor.n_levels)
                     if snap['levels'][l].get('populated')]

        for l in populated:
            info = snap['levels'][l]
            sig = info['harmonic_signature']
            parts.append(
                f"L{l}({info['name']}): {info['n_nodes']}n "
                f"gap={info['eigenvalue_gap']:.2f} "
                f"risk={info['phase_transition_risk']:.2f} "
                f"key={sig['dominant_interval']} "
                f"verdict={sig['stability_verdict']}"
            )

        # Cross-level
        seen = set()
        for l in populated:
            for other, val in snap['levels'][l].get('cross_level_resonance', {}).items():
                pair = tuple(sorted([LEVEL_NAMES.get(l, f'L{l}'), other]))
                if pair not in seen:
                    seen.add(pair)
                    parts.append(f"resonance({pair[0]},{pair[1]})={val:.2f}")

        # Signals
        signals = []
        for l in populated:
            info = snap['levels'][l]
            if info['phase_transition_risk'] > 0.7:
                signals.append(f"WARN:{info['name']} risk={info['phase_transition_risk']:.2f}")
        if signals:
            parts.append('SIGNALS: ' + '; '.join(signals))
        else:
            parts.append('SIGNALS: none')

        return ' | '.join(parts)

    def log_snapshot(self):
        """Write snapshot to JSONL log."""
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, 'tensor_state.jsonl')
        snap = self.tensor.tensor_snapshot()
        snap['timestamp'] = datetime.now(timezone.utc).isoformat()
        with open(path, 'a') as f:
            f.write(json.dumps(snap, default=str) + '\n')
