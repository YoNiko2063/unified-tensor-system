"""
FICUTS Geometric HDV Population

Encodes LaTeX equation STRUCTURE into HDV without semantic understanding.

Key insight: the topology of an expression tree (depth, branching, operator
types, variable positions) is information-bearing even before any semantic
interpretation. Similar structures → similar DEQs → HDV clustering.

Structural features captured:
  - Tree depth       (complexity proxy)
  - Branching factor (operator arity)
  - Node-type ratios (var / num / op / cmd proportions)
  - Positional hashes (left-subtree vs right-subtree patterns)

These create an HDV "geometric dimension" that the cross-dimensional
discovery system can later compare against math/physical/execution dims.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── Expression tree ───────────────────────────────────────────────────────────

class ExprNode:
    """Node in a structural parse tree of a LaTeX expression."""

    __slots__ = ("kind", "value", "children")

    def __init__(self, kind: str, value: str = "", children: Optional[List] = None):
        self.kind = kind       # 'var' | 'num' | 'op' | 'cmd' | 'group'
        self.value = value
        self.children: List["ExprNode"] = children or []

    @property
    def depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(c.depth for c in self.children)

    @property
    def size(self) -> int:
        return 1 + sum(c.size for c in self.children)

    @property
    def branching(self) -> float:
        if not self.children:
            return 0.0
        return float(len(self.children)) + sum(
            c.branching for c in self.children
        ) / len(self.children)


# ── Structural LaTeX parser ───────────────────────────────────────────────────

class LatexTreeParser:
    """
    Minimal structural parser for LaTeX math.

    Produces an ExprNode tree capturing topology without full semantics.
    Handles: fractions, subscripts, superscripts, Greek letters, parentheses.
    """

    _BINARY_OPS = frozenset({"+", "-", "*", "/", "=", "<", ">", "^", "_"})
    _BINARY_CMDS = frozenset({"\\frac", "\\binom", "\\dfrac", "\\tfrac"})
    _UNARY_CMDS = frozenset({
        "\\partial", "\\nabla", "\\sqrt", "\\sum", "\\int", "\\prod",
        "\\sin", "\\cos", "\\exp", "\\log", "\\ln", "\\lim", "\\det",
    })

    def parse(self, latex: str) -> ExprNode:
        tokens = self._tokenize(latex)
        node, _ = self._parse_seq(tokens, 0)
        return node

    def _tokenize(self, latex: str) -> List[str]:
        latex = re.sub(r'\s+', ' ', latex.strip())
        return re.findall(
            r'(\\[a-zA-Z]+|[{}()\[\]]|[+\-*/=<>^_,;]|[a-zA-Z0-9]+|\.)',
            latex,
        )

    def _parse_seq(self, toks: List[str], pos: int) -> Tuple[ExprNode, int]:
        children: List[ExprNode] = []
        while pos < len(toks):
            tok = toks[pos]
            if tok in ("}", ")", "]"):
                break
            if tok in ("{", "(", "["):
                close = {"(": ")", "[": "]", "{": "}"}[tok]
                child, pos = self._parse_seq(toks, pos + 1)
                if pos < len(toks) and toks[pos] == close:
                    pos += 1
                children.append(ExprNode("group", tok, [child] if child.kind != "group" or child.children else child.children))
            elif tok in self._BINARY_CMDS:
                pos += 1
                a1, pos = self._parse_group(toks, pos)
                a2, pos = self._parse_group(toks, pos)
                children.append(ExprNode("cmd", tok, [a1, a2]))
            elif tok in self._UNARY_CMDS:
                children.append(ExprNode("cmd", tok))
                pos += 1
            elif tok in self._BINARY_OPS:
                children.append(ExprNode("op", tok))
                pos += 1
            elif re.fullmatch(r'[0-9]+(\.[0-9]*)?', tok):
                children.append(ExprNode("num", tok))
                pos += 1
            else:
                children.append(ExprNode("var", tok))
                pos += 1

        if len(children) == 1:
            return children[0], pos
        return ExprNode("group", "", children), pos

    def _parse_group(self, toks: List[str], pos: int) -> Tuple[ExprNode, int]:
        if pos < len(toks) and toks[pos] == "{":
            child, pos = self._parse_seq(toks, pos + 1)
            if pos < len(toks) and toks[pos] == "}":
                pos += 1
            return child, pos
        if pos < len(toks):
            tok = toks[pos]
            pos += 1
            if re.fullmatch(r'[0-9]+(\.[0-9]*)?', tok):
                return ExprNode("num", tok), pos
            return ExprNode("var", tok), pos
        return ExprNode("group", ""), pos


# ── Geometric HDV Populator ───────────────────────────────────────────────────

class GeometricHDVPopulator:
    """
    Populate the HDV "geometric" dimension from raw LaTeX structure.

    Works even with no semantic understanding of equations.
    Similar expression structures hash to nearby HDV regions.

    Usage:
      pop = GeometricHDVPopulator(hdv_system=hdv)
      vec = pop.populate_from_latex(r'\\frac{\\partial u}{\\partial t} = \\alpha \\nabla^2 u')
      similar = pop.find_similar_structure(r'\\frac{dT}{dt} = k(T_0 - T)', top_k=3)
    """

    def __init__(self, hdv_system=None):
        self.hdv = hdv_system
        self.parser = LatexTreeParser()
        self._stored: List[Dict[str, Any]] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def populate_from_latex(self, latex: str) -> np.ndarray:
        """
        Encode equation structure to HDV and store in 'geometric' domain.

        Returns float32 [hdv_dim] vector.
        Registration in self.hdv is thread-safe (done via _register_domain_dims).
        """
        tree = self.parser.parse(latex)
        features = self._extract_features(tree)
        vec = self._features_to_hdv(features)

        if self.hdv is not None:
            self.hdv._register_domain_dims("geometric", vec)

        self._stored.append({"latex": latex, "features": features, "vec": vec})
        return vec

    def populate_batch(self, latex_list: List[str]) -> List[np.ndarray]:
        """Encode a list of LaTeX strings; returns list of HDV vectors."""
        return [self.populate_from_latex(eq) for eq in latex_list]

    def find_similar_structure(
        self, latex: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find stored equations with the most similar geometric structure.

        Does NOT use semantics — purely topological similarity.
        """
        q_features = self._extract_features(self.parser.parse(latex))
        q_vec = self._features_to_hdv(q_features)
        n_q = np.linalg.norm(q_vec)

        results = []
        for entry in self._stored:
            sv = entry["vec"]
            n_s = np.linalg.norm(sv)
            if n_q < 1e-9 or n_s < 1e-9:
                continue
            sim = float(np.dot(q_vec, sv) / (n_q * n_s))
            results.append({"latex": entry["latex"], "similarity": sim,
                             "features": entry["features"]})

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    @property
    def n_stored(self) -> int:
        return len(self._stored)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _extract_features(self, tree: ExprNode) -> Dict[str, float]:
        """Compute scalar structural features from expression tree."""
        nodes = _all_nodes(tree)
        total = max(len(nodes), 1)
        counts: Dict[str, int] = {}
        for n in nodes:
            counts[n.kind] = counts.get(n.kind, 0) + 1

        return {
            "depth":      float(tree.depth),
            "size":       float(tree.size),
            "branching":  tree.branching,
            "var_ratio":  counts.get("var", 0) / total,
            "num_ratio":  counts.get("num", 0) / total,
            "op_ratio":   counts.get("op", 0) / total,
            "cmd_ratio":  counts.get("cmd", 0) / total,
        }

    def _features_to_hdv(self, features: Dict[str, float]) -> np.ndarray:
        """
        Map structural feature dict → HDV via hash projection.

        Each (feature_name, quantised_bucket) pair hashes to two HDV dims.
        The magnitude of the second dim carries the raw feature value,
        giving the vector both structural and magnitude information.
        """
        hdv_dim = self.hdv.hdv_dim if self.hdv else 10000
        vec = np.zeros(hdv_dim, dtype=np.float32)

        for name, val in features.items():
            # Quantise into 64 buckets for structural routing
            bucket = int(abs(val) * 64) % 64
            key = f"{name}:{bucket}".encode()
            h1 = int(hashlib.md5(key).hexdigest(), 16) % hdv_dim
            h2 = int(hashlib.sha1(key).hexdigest(), 16) % hdv_dim
            vec[h1] = 1.0
            # Second dim carries fractional magnitude in [0,1]
            vec[h2] = min(float(val) / (float(val) + 1.0 + 1e-9), 1.0)

        # Universal dims (first 33%) — allow cross-domain comparison
        universal_end = hdv_dim // 3
        fingerprint = "_".join(f"{k}:{v:.2f}" for k, v in sorted(features.items()))
        h_u = int(hashlib.sha256(fingerprint.encode()).hexdigest(), 16) % universal_end
        vec[h_u] = 1.0

        return np.clip(vec, 0.0, 1.0)


def _all_nodes(tree: ExprNode) -> List[ExprNode]:
    out = [tree]
    for c in tree.children:
        out.extend(_all_nodes(c))
    return out
