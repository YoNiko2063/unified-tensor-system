"""Observable tensor state for humans and AI agents.

Single interface to see the full tensor state as markdown
or compressed agent context.
"""
import os
import sys
import json
import time
import numpy as np
from datetime import datetime, timezone
from typing import Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tensor.core import UnifiedTensor, LEVEL_NAMES


class TensorObserver:
    """Observe the full tensor state in human/AI-readable formats."""

    def __init__(self, tensor: UnifiedTensor,
                 log_dir: str = 'tensor/logs',
                 trajectory=None,
                 predictive=None,
                 fiber_bundle=None):
        self.tensor = tensor
        self.log_dir = log_dir
        self.trajectory = trajectory
        self.predictive = predictive
        self.fiber_bundle = fiber_bundle

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
        lines.append('| Level | Nodes | Gap | Risk | Key | Verdict | φ-Growth |')
        lines.append('|-------|-------|-----|------|-----|---------|----------|')
        for l in range(self.tensor.n_levels):
            info = snap['levels'][l]
            name = info['name']
            if not info.get('populated'):
                lines.append(f'| L{l} {name} | — | — | — | — | empty | — |')
            else:
                sig = info['harmonic_signature']
                growth_count = sig.get('growth_regime_count', 0)
                lines.append(
                    f"| L{l} {name} | {info['n_nodes']} | "
                    f"{info['eigenvalue_gap']:.2f} | "
                    f"{info['phase_transition_risk']:.2f} | "
                    f"{sig['dominant_interval']} | "
                    f"{sig['stability_verdict']} | "
                    f"{growth_count} |"
                )
                # WARNING: eigenvalue gap collapse detection
                if info['eigenvalue_gap'] < 0.05:
                    mna = self.tensor._mna.get(l)
                    if mna is not None:
                        G_active = mna.G[:mna.n_total, :mna.n_total]
                        eigvals = np.sort(np.abs(np.linalg.eigvalsh(G_active)))[::-1]
                        # Find contributing node indices (highest diagonal G)
                        diag = np.abs(np.diag(G_active))
                        contrib_indices = np.argsort(-diag)[:5].tolist()
                        lines.append(
                            f"| | **WARNING**: L{l} gap={info['eigenvalue_gap']:.4f} < 0.05 | "
                            f"Contributing nodes: {contrib_indices} | | | |"
                        )
        lines.append('')

        # Cross-level resonance (golden angle based when subspace data available)
        lines.append('## Cross-Level Resonance')
        golden_mat = self.tensor.golden_resonance_matrix()
        use_golden = golden_mat.size > 0 and golden_mat.sum() > 0
        if use_golden:
            lines.append('| Pair | Golden Resonance | Harmonic Resonance | Interpretation |')
            lines.append('|------|-----------------|-------------------|----------------|')
        else:
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
                if use_golden:
                    # Find other level index
                    other_idx = None
                    for idx, name in LEVEL_NAMES.items():
                        if name == other_name:
                            other_idx = idx
                            break
                    golden_val = golden_mat[l, other_idx] if (
                        other_idx is not None and l < golden_mat.shape[0]
                        and other_idx < golden_mat.shape[1]) else 0.0
                    if golden_val > 0.8:
                        interp = 'Golden angle aligned'
                    lines.append(
                        f'| {pair_key[0]} ↔ {pair_key[1]} | {golden_val:.4f} | '
                        f'{res_val:.4f} | {interp} |'
                    )
                else:
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

        # Hardware-code alignment signal
        code_hw_resonance = None
        if 2 in populated and 3 in populated:
            code_hw_resonance = self.tensor.cross_level_resonance(2, 3)
            if code_hw_resonance < 0.5:
                lines.append(
                    f"- SIGNAL: code structure misaligned with hardware geometry "
                    f"(resonance={code_hw_resonance:.2f}). "
                    f"Recommend: run bootstrap to restructure toward "
                    f"hardware-consonant patterns.")
            elif code_hw_resonance > 0.8:
                lines.append(
                    f"- SIGNAL: code is hardware-optimal for this machine "
                    f"(resonance={code_hw_resonance:.2f}).")

        if not any(line.startswith('- ') for line in lines[-5:]):
            lines.append('- No critical signals at this time')

        lines.append('')

        # Learning trajectory (if available)
        if self.trajectory is not None and self.trajectory.points:
            lines.append('## Learning Trajectory')
            for l_name in sorted(set(
                k for p in self.trajectory.points
                for k in p.consonance)):
                vel = self.trajectory.consonance_velocity(l_name)
                acc = self.trajectory.consonance_acceleration(l_name)
                lines.append(
                    f"- {l_name}: velocity={vel:.6f}, acceleration={acc:.6f}")
            cs = self.trajectory.compounding_subspaces()
            if cs:
                lines.append(f"- Compounding: {', '.join(cs)}")
            ml = self.trajectory.meta_loss()
            lines.append(f"- Meta-loss: {ml:.6f}")
            lines.append('')

        # Ignorance map (if available)
        if self.predictive is not None and self.predictive.error_history:
            lines.append('## Ignorance Map')
            ig = self.predictive.ignorance_map()
            for l_name, val in sorted(ig.items()):
                lines.append(f"- {l_name}: {val:.4f}")
            prio = self.predictive.learning_priority()
            if prio:
                lines.append(f"- Learning priority: {', '.join(prio)}")
            lines.append('')

        # Universal patterns (if available)
        if self.fiber_bundle is not None and self.fiber_bundle.fibers:
            n_universal = len(self.fiber_bundle.universal_patterns())
            lines.append(f'## Domain Fibers')
            lines.append(f"- Universal patterns found: {n_universal}")
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
