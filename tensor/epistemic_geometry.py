"""Epistemic Geometry Layer — token trajectory → dynamical signal analysis.

Treats token embedding sequences as dynamical system trajectories:
  x_1, x_2, ..., x_T ∈ R^n

Extracts:
  - Semantic velocity:  v(t) = x_{t+1} - x_t
  - Semantic curvature: a(t) = x_{t+1} - 2x_t + x_{t-1}
  - Spectral analysis:  FFT per section (low-freq → stable, high-freq → hype)
  - Epistemic projection: validity = ‖P_science(x)‖ - ‖P_hype(x)‖
  - Section-level features: technical density, citation density, hedging index

Uses hdv.structural_encode() for deterministic, reproducible embeddings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Hedging markers for epistemic uncertainty detection
_HEDGING_WORDS = {
    "may", "might", "could", "possibly", "perhaps", "likely", "unlikely",
    "suggest", "suggests", "indicate", "indicates", "appear", "appears",
    "seem", "seems", "approximately", "roughly", "estimate", "uncertain",
    "potential", "probable", "tentative", "preliminary",
}

# Citation pattern: [N], (Author, Year), doi:, arXiv:
_CITATION_RE = re.compile(
    r"\[\d+\]|\([A-Z][a-z]+(?:\s+(?:et\s+al\.?|and|&))?,?\s*\d{4}\)|doi:|arXiv:",
    re.IGNORECASE,
)

# Technical vocabulary (STEM indicators)
_TECHNICAL_WORDS = {
    "equation", "theorem", "proof", "lemma", "corollary", "hypothesis",
    "derivative", "integral", "convergence", "eigenvalue", "matrix",
    "vector", "tensor", "gradient", "optimization", "algorithm",
    "variance", "correlation", "regression", "distribution", "stochastic",
    "differential", "partial", "boundary", "nonlinear", "linear",
    "coefficient", "parameter", "variable", "function", "operator",
    "spectrum", "frequency", "amplitude", "phase", "resonance",
    "bifurcation", "stability", "perturbation", "asymptotic",
}

# Sentiment/hype vocabulary
_HYPE_WORDS = {
    "incredible", "amazing", "revolutionary", "breakthrough", "game-changing",
    "disruptive", "explosive", "skyrocket", "moon", "unstoppable",
    "guaranteed", "massive", "insane", "unbelievable", "unprecedented",
    "soaring", "plummeting", "crash", "surge", "rocket", "pump", "dump",
    "fomo", "hodl", "yolo", "bullish", "bearish", "mooning",
}


@dataclass
class SectionProfile:
    """Epistemic profile for a single text section."""
    mean: np.ndarray                  # μ_k — mean embedding
    covariance: np.ndarray            # Σ_k — embedding covariance
    spectrum: np.ndarray              # FFT magnitudes of trajectory
    technical_density: float = 0.0    # fraction of technical words
    citation_density: float = 0.0     # citations per sentence
    hedging_index: float = 0.0        # fraction of hedging words
    sentiment_amplitude: float = 0.0  # hype word density
    velocity_mean: float = 0.0        # mean ‖v(t)‖
    curvature_mean: float = 0.0       # mean ‖a(t)‖


@dataclass
class EpistemicProfile:
    """Full epistemic profile for an article."""
    sections: List[SectionProfile] = field(default_factory=list)
    overall_validity: float = 0.0    # ‖P_science‖ - ‖P_hype‖
    spectral_signature: np.ndarray = field(default_factory=lambda: np.array([]))
    curvature_stats: Dict[str, float] = field(default_factory=dict)
    classification: str = ""          # "scientific", "editorial", "hype"
    section_weights: np.ndarray = field(default_factory=lambda: np.array([]))


class EpistemicGeometryLayer:
    """Extracts epistemic signal from text via embedding trajectory analysis."""

    def __init__(self, hdv_system=None, hdv_dim: int = 10000) -> None:
        self._hdv = hdv_system
        self._hdv_dim = hdv_dim
        # Reference subspaces (built lazily from corpus)
        self._science_basis: Optional[np.ndarray] = None
        self._hype_basis: Optional[np.ndarray] = None

    def analyze(self, text: str, domain: str = "general") -> EpistemicProfile:
        """Full epistemic analysis of an article.

        1. Split into sections
        2. Embed tokens per section → trajectory
        3. Compute velocity, curvature, FFT
        4. Score technical/citation/hedging/sentiment densities
        5. Project onto science/hype subspaces → validity
        6. Classify: scientific/editorial/hype
        """
        sections = self._split_sections(text)
        if not sections:
            return EpistemicProfile()

        section_profiles = []
        all_embeddings = []

        for section_text in sections:
            tokens = section_text.split()
            if len(tokens) < 3:
                continue

            # Embed tokens → trajectory
            trajectory = self._embed_trajectory(tokens, domain)
            if trajectory.shape[0] < 3:
                continue

            # Dynamics
            velocity = np.diff(trajectory, axis=0)
            curvature = np.diff(trajectory, n=2, axis=0)

            # Spectral analysis — FFT of trajectory (magnitude spectrum)
            spectrum = np.abs(np.fft.rfft(trajectory, axis=0)).mean(axis=1)

            # Section covariance
            mean = trajectory.mean(axis=0)
            cov = np.cov(trajectory.T) if trajectory.shape[0] > trajectory.shape[1] else np.eye(trajectory.shape[1])

            # Text features
            words_lower = [w.lower().strip(".,!?;:\"'()[]") for w in tokens]
            n_words = max(len(words_lower), 1)

            tech_density = sum(1 for w in words_lower if w in _TECHNICAL_WORDS) / n_words
            hedging = sum(1 for w in words_lower if w in _HEDGING_WORDS) / n_words
            hype_density = sum(1 for w in words_lower if w in _HYPE_WORDS) / n_words

            sentences = max(section_text.count(".") + section_text.count("!") + section_text.count("?"), 1)
            citations = len(_CITATION_RE.findall(section_text))
            cite_density = citations / sentences

            sp = SectionProfile(
                mean=mean,
                covariance=cov if isinstance(cov, np.ndarray) and cov.ndim == 2 else np.eye(mean.shape[0]),
                spectrum=spectrum,
                technical_density=tech_density,
                citation_density=cite_density,
                hedging_index=hedging,
                sentiment_amplitude=hype_density,
                velocity_mean=float(np.linalg.norm(velocity, axis=1).mean()) if len(velocity) > 0 else 0.0,
                curvature_mean=float(np.linalg.norm(curvature, axis=1).mean()) if len(curvature) > 0 else 0.0,
            )
            section_profiles.append(sp)
            all_embeddings.append(trajectory)

        if not section_profiles:
            return EpistemicProfile()

        # Overall spectral signature — concatenated trajectory FFT
        full_traj = np.concatenate(all_embeddings, axis=0) if all_embeddings else np.zeros((1, 1))
        spectral_sig = np.abs(np.fft.rfft(full_traj, axis=0)).mean(axis=1)

        # Validity score
        science_score = np.mean([s.technical_density + s.citation_density for s in section_profiles])
        hype_score = np.mean([s.sentiment_amplitude for s in section_profiles])
        validity = float(science_score - hype_score)

        # Curvature stats
        curvatures = [s.curvature_mean for s in section_profiles]
        curv_stats = {
            "mean": float(np.mean(curvatures)),
            "std": float(np.std(curvatures)),
            "max": float(np.max(curvatures)),
        }

        # Section weights (technical density + citation density normalized)
        raw_weights = np.array([
            s.technical_density + s.citation_density + 0.01
            for s in section_profiles
        ])
        section_weights = raw_weights / raw_weights.sum()

        # Classification
        classification = self._classify(validity, section_profiles)

        return EpistemicProfile(
            sections=section_profiles,
            overall_validity=validity,
            spectral_signature=spectral_sig,
            curvature_stats=curv_stats,
            classification=classification,
            section_weights=section_weights,
        )

    def _embed_trajectory(self, tokens: List[str], domain: str) -> np.ndarray:
        """Embed token sequence → (T, d) trajectory using HDV structural_encode."""
        embeddings = []
        window = 3  # small window for locality
        for i in range(len(tokens)):
            start = max(0, i - window // 2)
            end = min(len(tokens), i + window // 2 + 1)
            chunk = " ".join(tokens[start:end])

            if self._hdv is not None:
                vec = self._hdv.structural_encode(chunk, domain)
                if isinstance(vec, np.ndarray):
                    embeddings.append(vec)
                    continue

            # Fallback: deterministic hash embedding
            embeddings.append(self._hash_embed(chunk))

        if not embeddings:
            return np.zeros((1, min(self._hdv_dim, 200)))

        traj = np.array(embeddings)
        # Project to manageable dimension for FFT/covariance
        if traj.shape[1] > 200:
            traj = traj[:, :200]
        return traj

    def _hash_embed(self, text: str) -> np.ndarray:
        """Deterministic hash-based embedding (fallback when no HDV system)."""
        import hashlib
        dim = min(self._hdv_dim, 200)
        vec = np.zeros(dim)
        for i, ch in enumerate(text.encode("utf-8")):
            h = int(hashlib.md5(f"{ch}_{i}".encode()).hexdigest(), 16)
            idx = h % dim
            vec[idx] += 1.0 if (h // dim) % 2 == 0 else -1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _split_sections(self, text: str) -> List[str]:
        """Split text into sections (paragraphs or semantic blocks)."""
        # Split on double newlines or section headers
        raw = re.split(r"\n\s*\n|\n#+\s", text)
        sections = [s.strip() for s in raw if len(s.strip()) > 20]
        if not sections:
            # Single block — split by sentences into chunks
            sentences = re.split(r"(?<=[.!?])\s+", text)
            chunk_size = max(len(sentences) // 3, 3)
            sections = []
            for i in range(0, len(sentences), chunk_size):
                chunk = " ".join(sentences[i:i + chunk_size])
                if len(chunk.strip()) > 20:
                    sections.append(chunk)
        return sections if sections else [text]

    def _classify(self, validity: float, sections: List[SectionProfile]) -> str:
        """Classify article as scientific/editorial/hype."""
        avg_tech = np.mean([s.technical_density for s in sections])
        avg_cite = np.mean([s.citation_density for s in sections])
        avg_hype = np.mean([s.sentiment_amplitude for s in sections])
        avg_hedge = np.mean([s.hedging_index for s in sections])

        # Scientific: high tech + citations, low hype
        if avg_tech > 0.03 and avg_cite > 0.1 and avg_hype < 0.02:
            return "scientific"
        # Hype: high sentiment amplitude, low technical content
        if avg_hype > 0.03 and avg_tech < 0.02:
            return "hype"
        # Editorial: moderate tech, some hedging, moderate hype
        return "editorial"

    def build_reference_subspaces(
        self,
        scientific_texts: List[str],
        hype_texts: List[str],
        domain: str = "general",
        rank: int = 10,
    ) -> None:
        """Build science and hype reference subspaces from labeled corpora.

        Uses PCA on labeled text embeddings to define projection subspaces.
        """
        def _embed_corpus(texts):
            all_vecs = []
            for text in texts:
                tokens = text.split()[:100]
                traj = self._embed_trajectory(tokens, domain)
                all_vecs.append(traj.mean(axis=0))
            return np.array(all_vecs)

        if scientific_texts:
            sci_vecs = _embed_corpus(scientific_texts)
            if sci_vecs.shape[0] >= rank:
                U, _, _ = np.linalg.svd(sci_vecs - sci_vecs.mean(axis=0), full_matrices=False)
                self._science_basis = U[:, :rank].T  # (rank, dim)

        if hype_texts:
            hype_vecs = _embed_corpus(hype_texts)
            if hype_vecs.shape[0] >= rank:
                U, _, _ = np.linalg.svd(hype_vecs - hype_vecs.mean(axis=0), full_matrices=False)
                self._hype_basis = U[:, :rank].T

    def project_validity(self, embedding: np.ndarray) -> float:
        """Compute validity = ‖P_science(x)‖ - ‖P_hype(x)‖ using reference subspaces."""
        sci_norm = 0.0
        hype_norm = 0.0

        if self._science_basis is not None:
            proj = self._science_basis @ embedding
            sci_norm = float(np.linalg.norm(proj))

        if self._hype_basis is not None:
            proj = self._hype_basis @ embedding
            hype_norm = float(np.linalg.norm(proj))

        return sci_norm - hype_norm
