"""
FICUTS Prediction-Driven Learning System

Implements the 6-stage autonomous learning loop defined in
PREDICTION_DRIVEN_LEARNING.md:

  1. StructuredTextExtractor  — mutual-info concept graph from text
  2. PredictiveConceptLearner — entropy-based next-concept ordering
  3. ProblemGenerator         — test problems for each concept
  4. KnowledgeBasedProblemSolver — HDV-based retrieval + synthesis
  5. PredictionVerifier       — Lyapunov tracking + gap detection
  6. ContinuousLearningLoop   — full autonomous loop

Mathematical foundations:
  - Mutual Information: I(c1;c2) = p12·log2(p12/(p1·p2))
  - Entropy: H = -Σ p·log2(p)
  - Information gain: ΔI = H_before - H_after
  - MDL: confidence = 1/(1+description_length)
  - Lyapunov energy: E = fraction of unlearned concepts (should decrease)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


# ── Concept + ConceptGraph ────────────────────────────────────────────────────

@dataclass
class Concept:
    name: str
    concept_type: str = "general"   # operation | theorem | algorithm | general
    section: str = ""
    prerequisites: List[str] = field(default_factory=list)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Concept) and self.name == other.name


class ConceptGraph:
    """Directed graph of Concept nodes with MI-weighted edges."""

    def __init__(self):
        self._concepts: Dict[str, Concept] = {}
        self._edges: Dict[str, List[Tuple[str, float]]] = {}
        self._learned: Set[str] = set()

    # ── Mutation ───────────────────────────────────────────────────────────────

    def add_concept(self, concept: Concept):
        self._concepts[concept.name] = concept
        self._edges.setdefault(concept.name, [])

    def add_edge(self, from_name: str, to_name: str, weight: float = 1.0):
        self._edges.setdefault(from_name, []).append((to_name, weight))

    def mark_learned(self, name: str):
        self._learned.add(name)

    # ── Query ──────────────────────────────────────────────────────────────────

    def get_concept(self, name: str) -> Optional[Concept]:
        return self._concepts.get(name)

    def all_concepts(self) -> List[Concept]:
        return list(self._concepts.values())

    def get_prerequisites(self, concept: Concept) -> List[Concept]:
        return [self._concepts[p] for p in concept.prerequisites
                if p in self._concepts]

    def out_degree(self, concept: Concept) -> int:
        return len(self._edges.get(concept.name, []))

    def is_learned(self, name: str) -> bool:
        return name in self._learned

    def get_unknown_concepts(
        self, learned: Optional[Set[str]] = None
    ) -> List[Concept]:
        check = learned if learned is not None else self._learned
        return [c for c in self._concepts.values() if c.name not in check]

    def prerequisites_met(
        self, concept: Concept, learned: Optional[Set[str]] = None
    ) -> bool:
        check = learned if learned is not None else self._learned
        return all(p in check for p in concept.prerequisites)

    def all_concepts_learned(self) -> bool:
        return all(c.name in self._learned for c in self._concepts.values())

    def __len__(self) -> int:
        return len(self._concepts)


# ── Stage 1: StructuredTextExtractor ─────────────────────────────────────────

class StructuredTextExtractor:
    """
    Extract a ConceptGraph from raw text using mutual information.

    Process:
      1. Chunk text into sections (headers / double-newlines)
      2. Extract candidate concept names per section via regex patterns
      3. Build MI-weighted dependency edges between co-occurring concepts
    """

    _PATTERNS = [
        r'\*\*([^*]{3,40})\*\*',              # **bold**
        r'`([^`]{2,30})`',                    # `code`
        r'\b([A-Z][a-z]{2,}(?:[ _][A-Z][a-z]+)*)\b',  # CamelCase term
        r'\b([\w_]{3,20})(?=\s+(?:is|refers|means|denotes)\b)',  # X is ...
    ]

    def __init__(self, mi_threshold: float = 0.005):
        self.mi_threshold = mi_threshold

    def extract_concepts(self, text: str) -> ConceptGraph:
        sections = self._chunk(text)
        seen: Dict[str, Concept] = {}

        for section in sections:
            for concept in self._extract_from_section(section):
                if concept.name not in seen:
                    seen[concept.name] = concept

        graph = ConceptGraph()
        concepts = list(seen.values())
        for c in concepts:
            graph.add_concept(c)

        # Build MI edges
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        n_para = max(len(paragraphs), 1)
        words = text.lower().split()
        n_words = max(len(words), 1)

        for i, c1 in enumerate(concepts):
            for c2 in concepts[i + 1:]:
                mi = self._mutual_info(c1, c2, text, paragraphs, n_words, n_para)
                if mi > self.mi_threshold:
                    graph.add_edge(c1.name, c2.name, weight=mi)
                    graph.add_edge(c2.name, c1.name, weight=mi)

        return graph

    def _chunk(self, text: str) -> List[str]:
        parts = re.split(r'\n#{1,4}\s|\n-{3,}\n|\n{3,}', text)
        return [p.strip() for p in parts if p.strip()]

    def _extract_from_section(self, section: str) -> List[Concept]:
        results, seen_names = [], set()
        sl = section.lower()

        ctype = "general"
        if "theorem" in sl or "lemma" in sl or "proof" in sl:
            ctype = "theorem"
        elif "algorithm" in sl or "procedure" in sl or "pseudocode" in sl:
            ctype = "algorithm"
        elif any(k in sl for k in ("compute", "calculate", "apply", "evaluate")):
            ctype = "operation"

        for pattern in self._PATTERNS:
            for m in re.finditer(pattern, section):
                raw = m.group(1).strip()
                name = re.sub(r'\s+', '_', raw.lower())
                if 2 < len(name) < 40 and name not in seen_names:
                    seen_names.add(name)
                    results.append(Concept(
                        name=name,
                        concept_type=ctype,
                        section=section[:80],
                    ))
        return results

    def _mutual_info(
        self,
        c1: Concept,
        c2: Concept,
        text: str,
        paragraphs: List[str],
        n_words: int,
        n_para: int,
    ) -> float:
        t1 = c1.name.replace("_", " ")
        t2 = c2.name.replace("_", " ")
        tl = text.lower()

        p1 = tl.count(t1) / n_words
        p2 = tl.count(t2) / n_words
        joint = sum(1 for p in paragraphs if t1 in p.lower() and t2 in p.lower())
        p12 = joint / n_para

        if p12 > 0 and p1 > 0 and p2 > 0:
            return p12 * math.log2(p12 / (p1 * p2 + 1e-12))
        return 0.0


# ── Stage 2: PredictiveConceptLearner ─────────────────────────────────────────

class PredictiveConceptLearner:
    """
    Choose next concept to maximise information gain ΔI = H_before - H_after.

    H = entropy over the distribution of out-degrees of unknown concepts.
    Higher out-degree → more other concepts depend on this one → higher priority.
    """

    def __init__(self, graph: ConceptGraph, hdv_system=None):
        self.graph = graph
        self.hdv_system = hdv_system
        self.learned_concepts: Set[str] = set()

    def predict_next_concept(self) -> Optional[Concept]:
        candidates = [
            c for c in self.graph.all_concepts()
            if c.name not in self.learned_concepts
            and self.graph.prerequisites_met(c, self.learned_concepts)
        ]
        if not candidates:
            return None
        return max(candidates, key=self._information_gain)

    def _information_gain(self, concept: Concept) -> float:
        unknown_before = self.graph.get_unknown_concepts(self.learned_concepts)
        h_before = self._entropy(unknown_before)
        temp = self.learned_concepts | {concept.name}
        unknown_after = self.graph.get_unknown_concepts(temp)
        h_after = self._entropy(unknown_after)
        return h_before - h_after

    def _entropy(self, concepts: List[Concept]) -> float:
        if not concepts:
            return 0.0
        total = sum(self.graph.out_degree(c) for c in concepts)
        n = len(concepts)
        entropy = 0.0
        for c in concepts:
            p = self.graph.out_degree(c) / total if total > 0 else 1.0 / n
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def mark_learned(self, name: str):
        self.learned_concepts.add(name)
        self.graph.mark_learned(name)


# ── Stage 3: ProblemGenerator ─────────────────────────────────────────────────

@dataclass
class Problem:
    concept: Concept
    text: str
    difficulty: int
    ground_truth: Optional[Any] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


class ProblemGenerator:
    """
    Generate test problems to verify understanding of each concept.

    Difficulty ∝ |prerequisites|. Verification uses MDL proxy:
      confidence = 1 / (1 + |description_length - expected_length|/expected)
    """

    _TEMPLATES: Dict[str, List[str]] = {
        "operation": [
            "Apply {name} to a simple input and describe the result.",
            "Show how {name} transforms an input step-by-step.",
            "Chain {name} with a prerequisite and explain the combined effect.",
        ],
        "theorem": [
            "State the conditions under which {name} holds.",
            "Give a counterexample where {name} fails.",
            "Prove {name} for the simplest non-trivial case.",
        ],
        "algorithm": [
            "Trace {name} on a small concrete input.",
            "What is the time complexity of {name}? Justify.",
            "Implement the core loop of {name} in pseudocode.",
        ],
        "general": [
            "Explain {name} in one sentence.",
            "Give a concrete example of {name}.",
            "How does {name} relate to the concepts that depend on it?",
        ],
    }

    def __init__(self, graph: ConceptGraph):
        self.graph = graph

    def generate_problem(self, concept: Concept) -> Problem:
        prereqs = self.graph.get_prerequisites(concept)
        difficulty = len(prereqs)
        templates = self._TEMPLATES.get(concept.concept_type, self._TEMPLATES["general"])
        idx = min(difficulty, len(templates) - 1)
        text = templates[idx].format(name=concept.name.replace("_", " "))
        return Problem(
            concept=concept,
            text=text,
            difficulty=difficulty,
            parameters={"name": concept.name, "difficulty": difficulty},
        )

    def verify_solution(
        self, problem: Problem, solution: Any
    ) -> Tuple[bool, float]:
        if problem.ground_truth is not None:
            ok = solution == problem.ground_truth
            return ok, 1.0 if ok else 0.0

        sol_str = str(solution) if solution is not None else ""
        if not sol_str.strip():
            return False, 0.0

        # MDL proxy: length near expected → confident
        n_words = len(sol_str.split())
        expected = 15 + problem.difficulty * 10
        confidence = 1.0 / (1.0 + abs(n_words - expected) / max(expected, 1))
        return confidence > 0.4, confidence


# ── Stage 4: KnowledgeBasedProblemSolver ─────────────────────────────────────

class KnowledgeBasedProblemSolver:
    """
    Solve problems using HDV-encoded knowledge.

    Encodes the problem text as an HDV query, retrieves the most similar
    stored solution, and synthesises a response.
    """

    def __init__(self, hdv_system=None):
        self.hdv = hdv_system
        self._store: List[Dict] = []   # {concept_name, solution, hdv}

    def solve(self, problem: Problem) -> Tuple[Optional[str], float]:
        if self.hdv is None:
            return None, 0.0

        q_hdv = self.hdv.structural_encode(problem.text, "execution")
        similar = self._find_similar(q_hdv, threshold=0.2)

        if not similar:
            return None, 0.0

        best = similar[0]
        return (
            f"Based on {best['concept_name']}: {best['solution']}",
            best["similarity"],
        )

    def store_solution(self, concept_name: str, solution: str):
        if self.hdv is None:
            return
        hdv_vec = self.hdv.structural_encode(
            concept_name + " " + solution, "execution"
        )
        self._store.append({
            "concept_name": concept_name,
            "solution": solution,
            "hdv": hdv_vec,
        })

    def _find_similar(
        self, query: np.ndarray, threshold: float = 0.2
    ) -> List[Dict]:
        results = []
        for entry in self._store:
            v = entry["hdv"]
            n1, n2 = np.linalg.norm(query), np.linalg.norm(v)
            if n1 < 1e-9 or n2 < 1e-9:
                continue
            sim = float(np.dot(query, v) / (n1 * n2))
            if sim >= threshold:
                results.append({**entry, "similarity": sim})
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results


# ── Stage 5: PredictionVerifier ───────────────────────────────────────────────

class PredictionVerifier:
    """
    Verify predictions and update HDV space via Lyapunov tracking.

    Correct → E decreases (stable). Wrong → gap detected, E rises.
    E = fraction of unlearned concepts.
    """

    def __init__(
        self,
        hdv_system,
        graph: ConceptGraph,
        solver: KnowledgeBasedProblemSolver,
    ):
        self.hdv = hdv_system
        self.graph = graph
        self.solver = solver
        self._energy: float = 1.0
        self.energy_history: List[float] = []

    def verify_and_update(
        self,
        concept: Concept,
        problem: Problem,
        solution: Optional[str],
        is_correct: bool,
    ) -> Dict:
        result: Dict[str, Any] = {
            "concept": concept.name,
            "correct": is_correct,
        }

        if is_correct:
            if solution:
                self.solver.store_solution(concept.name, solution)
            new_e = self._compute_energy(correct=True)
            self._energy = min(self._energy, new_e)
            result["action"] = "reinforce"
        else:
            gap = self._identify_gap(concept, problem)
            new_e = self._compute_energy(correct=False)
            self._energy = new_e
            result["action"] = f"relearn:{gap['type']}"
            result["gap"] = gap

        self.energy_history.append(self._energy)
        return result

    def _compute_energy(self, correct: bool) -> float:
        all_c = self.graph.all_concepts()
        if not all_c:
            return 0.0
        learned = sum(1 for c in all_c if self.graph.is_learned(c.name))
        base = 1.0 - learned / len(all_c)
        return base * (0.9 if correct else 1.05)

    def _identify_gap(self, concept: Concept, problem: Problem) -> Dict:
        for p in self.graph.get_prerequisites(concept):
            if not self.graph.is_learned(p.name):
                return {"type": "missing_prerequisite", "concept": p.name}

        if self.hdv:
            q = self.hdv.structural_encode(problem.text, "execution")
            if not self.solver._find_similar(q, threshold=0.1):
                return {"type": "no_relevant_patterns", "concept": concept.name}

        return {"type": "synthesis_error", "concept": concept.name}

    def lyapunov_stable(self) -> bool:
        if len(self.energy_history) < 2:
            return True
        return self.energy_history[-1] <= self.energy_history[-2]


# ── Stage 6: ContinuousLearningLoop ──────────────────────────────────────────

class ContinuousLearningLoop:
    """
    Full prediction-driven learning loop:

      Extract concepts → Predict next → Encode to HDV → Generate problem
      → Solve → Verify → Update → Repeat until all learned or max_iterations.
    """

    def __init__(self, hdv_system=None, max_iterations: int = 200):
        self.hdv = hdv_system
        self.max_iterations = max_iterations
        self.extractor = StructuredTextExtractor()
        self.log: List[Dict] = []

        # Set after run() is called
        self.graph: Optional[ConceptGraph] = None
        self.learner: Optional[PredictiveConceptLearner] = None
        self.verifier: Optional[PredictionVerifier] = None

    def run(self, text: str, verbose: bool = False) -> Dict:
        """
        Learn autonomously from text.

        Returns a summary dict with:
          concepts_learned, total_concepts, iterations,
          lyapunov_stable, energy_history
        """
        self.graph = self.extractor.extract_concepts(text)

        if verbose:
            print(f"[Learning] {len(self.graph)} concepts extracted")

        self.learner = PredictiveConceptLearner(self.graph, self.hdv)
        gen = ProblemGenerator(self.graph)
        solver = KnowledgeBasedProblemSolver(self.hdv)
        self.verifier = PredictionVerifier(self.hdv, self.graph, solver)

        for iteration in range(1, self.max_iterations + 1):
            if self.graph.all_concepts_learned():
                break

            concept = self.learner.predict_next_concept()
            if concept is None:
                break

            gain = self.learner._information_gain(concept)
            if verbose:
                print(f"[{iteration}] {concept.name} (gain={gain:.4f})")

            # Encode to HDV
            if self.hdv is not None:
                self.hdv.structural_encode(concept.name, "math")

            problem = gen.generate_problem(concept)
            solution, conf = solver.solve(problem)
            is_correct, _ = gen.verify_solution(problem, solution)
            result = self.verifier.verify_and_update(
                concept, problem, solution, is_correct
            )

            self.learner.mark_learned(concept.name)

            self.log.append({
                "iteration": iteration,
                "concept": concept.name,
                "correct": is_correct,
                "confidence": conf,
                "action": result.get("action"),
                "energy": self.verifier._energy,
            })

        return {
            "concepts_learned": len(self.learner.learned_concepts),
            "total_concepts": len(self.graph),
            "iterations": len(self.log),
            "lyapunov_stable": self.verifier.lyapunov_stable(),
            "energy_history": self.verifier.energy_history,
        }
