"""Orthogonal subspace learning for the unified tensor.

Each level and each regime-coherent node cluster within a level gets its
own orthogonal basis in a higher-dimensional space. Cross-level resonance
is the cosine of the angle between subspaces. The target angle between
any two well-coupled subspaces is cos⁻¹(1/φ) ≈ 51.8° — the golden angle.

Not 0° (identical = no independent information).
Not 90° (orthogonal = no coupling).
51.8° = maximum information sharing without collapse.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict

PHI = 1.6180339887
GOLDEN_ANGLE_COS = 1.0 / PHI  # ≈ 0.618


@dataclass
class SubspaceBasis:
    """Orthogonal basis for a cluster of nodes within a level."""
    labels: np.ndarray        # (n,) cluster assignment per node
    n_clusters: int
    basis_vectors: Dict[int, np.ndarray]  # cluster_id → (n, k) basis matrix
    cluster_sizes: Dict[int, int]


def assign_subspace(G: np.ndarray, n_clusters: int) -> SubspaceBasis:
    """Spectral clustering on G matrix → cluster labels.

    Each cluster = one orthogonal subspace.
    Basis vectors from cluster-restricted eigenvectors.

    Args:
        G: (n, n) conductance matrix.
        n_clusters: Number of clusters to form.

    Returns:
        SubspaceBasis with cluster assignments and per-cluster basis.
    """
    n = G.shape[0]
    n_clusters = max(1, min(n_clusters, n))

    if n <= 1 or n_clusters <= 1:
        return SubspaceBasis(
            labels=np.zeros(n, dtype=int),
            n_clusters=1,
            basis_vectors={0: np.eye(n) if n > 0 else np.zeros((0, 0))},
            cluster_sizes={0: n},
        )

    # Compute Laplacian: L = D - A where A = |G| with zero diagonal
    A = np.abs(G.copy())
    np.fill_diagonal(A, 0)
    D = np.diag(A.sum(axis=1))
    L = D - A

    # Spectral embedding: use bottom k eigenvectors of L (skip first = constant)
    eigvals, eigvecs = np.linalg.eigh(L)
    # Use eigenvectors 1..n_clusters for embedding
    k = min(n_clusters, n - 1)
    embedding = eigvecs[:, 1:k + 1]  # (n, k)

    if embedding.shape[1] == 0:
        return SubspaceBasis(
            labels=np.zeros(n, dtype=int),
            n_clusters=1,
            basis_vectors={0: np.eye(n)},
            cluster_sizes={0: n},
        )

    # Simple k-means clustering on the embedding (no sklearn dependency)
    labels = _kmeans(embedding, n_clusters, max_iter=50, seed=42)

    # Build per-cluster basis from cluster-restricted eigenvectors
    basis_vectors = {}
    cluster_sizes = {}
    for c in range(n_clusters):
        mask = labels == c
        cluster_sizes[c] = int(mask.sum())
        if cluster_sizes[c] == 0:
            basis_vectors[c] = np.zeros((n, 0))
            continue

        # Restrict G to cluster nodes, compute eigenvectors
        indices = np.where(mask)[0]
        G_sub = G[np.ix_(indices, indices)]
        if G_sub.shape[0] < 2:
            basis = np.zeros((n, 1))
            basis[indices[0], 0] = 1.0
        else:
            _, evecs = np.linalg.eigh(G_sub)
            k_basis = min(G_sub.shape[0], max(1, n_clusters))
            # Lift back to full space
            basis = np.zeros((n, k_basis))
            basis[indices, :] = evecs[:, :k_basis]

        basis_vectors[c] = basis

    return SubspaceBasis(
        labels=labels,
        n_clusters=n_clusters,
        basis_vectors=basis_vectors,
        cluster_sizes=cluster_sizes,
    )


def _kmeans(X: np.ndarray, k: int, max_iter: int = 50,
            seed: int = 42) -> np.ndarray:
    """Simple k-means clustering. Returns (n,) label array."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    k = min(k, n)

    # Initialize centroids via k-means++
    centroids = np.zeros((k, d))
    centroids[0] = X[rng.integers(n)]
    for i in range(1, k):
        dists = np.min([np.sum((X - centroids[j]) ** 2, axis=1)
                        for j in range(i)], axis=0)
        probs = dists / (dists.sum() + 1e-30)
        centroids[i] = X[rng.choice(n, p=probs)]

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # Assign
        dists = np.array([np.sum((X - c) ** 2, axis=1) for c in centroids])
        new_labels = np.argmin(dists, axis=0)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # Update centroids
        for i in range(k):
            mask = labels == i
            if mask.sum() > 0:
                centroids[i] = X[mask].mean(axis=0)

    return labels


def golden_resonance(subspace_a: SubspaceBasis, cluster_a: int,
                     subspace_b: SubspaceBasis, cluster_b: int) -> float:
    """Compute how close the angle between two subspaces is to the golden angle.

    Returns score in [0, 1] where 1.0 = exactly at golden angle (cos⁻¹(1/φ) ≈ 51.8°).
    Handles different-sized basis vectors from different levels by comparing
    eigenvalue spectra rather than raw dot products.
    """
    basis_a = subspace_a.basis_vectors.get(cluster_a)
    basis_b = subspace_b.basis_vectors.get(cluster_b)

    if basis_a is None or basis_b is None:
        return 0.0
    if basis_a.shape[1] == 0 or basis_b.shape[1] == 0:
        return 0.0

    n_a, k_a = basis_a.shape
    n_b, k_b = basis_b.shape

    if n_a == n_b:
        # Same-size levels: direct subspace angle via SVD
        M = basis_a.T @ basis_b
    else:
        # Different-size levels: compare via spectral fingerprints
        # Use singular value spectra of each basis as fingerprints
        s_a = np.linalg.svd(basis_a, compute_uv=False)
        s_b = np.linalg.svd(basis_b, compute_uv=False)
        # Pad to same length
        k_max = max(len(s_a), len(s_b))
        sa_pad = np.zeros(k_max)
        sb_pad = np.zeros(k_max)
        sa_pad[:len(s_a)] = s_a
        sb_pad[:len(s_b)] = s_b
        # Cosine similarity of spectral fingerprints
        na = np.linalg.norm(sa_pad)
        nb = np.linalg.norm(sb_pad)
        if na > 1e-30 and nb > 1e-30:
            mean_cos = float(np.dot(sa_pad, sb_pad) / (na * nb))
        else:
            mean_cos = 0.0
        return float(1.0 - abs(mean_cos - GOLDEN_ANGLE_COS))

    if M.size == 0:
        return 0.0

    svd_vals = np.linalg.svd(M, compute_uv=False)
    # Clamp to valid cosine range
    cos_angles = np.clip(svd_vals, -1.0, 1.0)
    mean_cos = float(np.mean(cos_angles))

    # Score: how close mean cosine is to golden angle cosine (0.618)
    return float(1.0 - abs(mean_cos - GOLDEN_ANGLE_COS))


def golden_resonance_matrix(subspace_bases: Dict[int, SubspaceBasis]) -> np.ndarray:
    """Compute L×L pairwise golden resonance scores between levels.

    For each pair of levels, averages golden_resonance across all
    cluster pairs.

    Args:
        subspace_bases: {level: SubspaceBasis}

    Returns:
        (L, L) matrix of golden resonance scores.
    """
    levels = sorted(subspace_bases.keys())
    L = max(levels) + 1 if levels else 0
    matrix = np.zeros((L, L))

    for i in levels:
        for j in levels:
            if i == j:
                matrix[i, j] = 1.0
                continue
            sb_i = subspace_bases[i]
            sb_j = subspace_bases[j]

            scores = []
            for ci in range(sb_i.n_clusters):
                for cj in range(sb_j.n_clusters):
                    if sb_i.cluster_sizes.get(ci, 0) > 0 and sb_j.cluster_sizes.get(cj, 0) > 0:
                        scores.append(golden_resonance(sb_i, ci, sb_j, cj))

            matrix[i, j] = float(np.mean(scores)) if scores else 0.0

    return matrix
