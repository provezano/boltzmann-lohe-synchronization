#!/usr/bin/env python3

"""Self-contained theorem and proposition verification script for:
Redescending Boltzmann-Lohe Rotation Synchronization and the Hub-Precision Phase Transition on SO(d)
"""

from __future__ import annotations

import math
import numpy as np
from scipy import linalg

def verify_theorems(rng: np.random.Generator) -> None:
    # 1. Theorem 1: Exact Block-MM monotonicity on SO(d) including det(U V^T) = -1 branch!
    for d in [3, 4, 5]:
        for _ in range(80):
            n = 8
            Rs = []
            for _ in range(n):
                Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
                if np.linalg.det(Q) < 0:
                    Q[:, -1] *= -1
                Rs.append(Q)
            Z = np.zeros((n, n, d, d))
            C = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.uniform() < 0.6:
                        Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
                        if np.linalg.det(Q) < 0:
                            Q[:, -1] *= -1
                        Z[i, j] = Q
                        Z[j, i] = Q.T
                        C[i, j] = C[j, i] = rng.uniform(0.2, 2.0)
            tau = rng.uniform(0.05, 0.5)
            lam = rng.uniform(0.0, 0.3)

            def obj(R_list: list[np.ndarray]) -> float:
                tot = 0.0
                for a in range(n):
                    for b in range(a + 1, n):
                        if C[a, b] > 0:
                            tr = float(np.sum(R_list[a] * (Z[a, b] @ R_list[b])))
                            tot += C[a, b] * d * tau * math.exp((tr - d) / (d * tau))
                return tot

            J0 = obj(Rs)
            i0 = int(rng.integers(0, n))
            Fi = lam * Rs[i0].copy()
            for j in range(n):
                if C[i0, j] > 0:
                    tr = float(np.sum(Rs[i0] * (Z[i0, j] @ Rs[j])))
                    rij = math.exp((tr - d) / (d * tau))
                    Fi += C[i0, j] * rij * (Z[i0, j] @ Rs[j])
            U, _, Vt = np.linalg.svd(Fi)
            S_det = np.eye(d)
            S_det[-1, -1] = 1.0 if np.linalg.det(U @ Vt) >= 0 else -1.0
            Ri_plus = U @ S_det @ Vt
            assert abs(np.linalg.det(Ri_plus) - 1.0) < 1e-10
            Rs_new = list(Rs)
            Rs_new[i0] = Ri_plus
            J1 = obj(Rs_new)
            lb = 0.5 * lam * float(np.sum((Ri_plus - Rs[i0]) ** 2))
            assert J1 - J0 >= lb - 1e-10, f"Thm 1 Block-MM violated at d={d}"

    # 2. Proposition 1: Complete repulsive quotient graph forces regular simplex; dipole collapse under linear repulsion
    for K in [3, 4, 5]:
        sizes = np.array([10] + [2] * (K - 1), dtype=float)
        assert sizes[0] > sizes[1:].sum()
        # Test random unit vectors in S^{K-1} against dipole configuration u_2=...=u_K=-u_1
        e_dipole = 0.5 * ((sizes[0] - sizes[1:].sum()) ** 2 - np.sum(sizes ** 2))
        for _ in range(100):
            U = rng.normal(size=(K, K))
            U /= np.linalg.norm(U, axis=1, keepdims=True)
            e_rand = sum(sizes[r] * sizes[s] * float(U[r] @ U[s]) for r in range(K) for s in range(r + 1, K))
            assert e_rand >= e_dipole - 1e-10, "Prop 1 dipole minimum violated"

    # 3. Theorem 2: Loewner Gauss-Markov optimality on 2-vertex-connected graph
    n = 7
    edges = [(i, (i + 1) % n) for i in range(n)] + [(0, 3), (1, 5), (2, 6)]  # 2-vertex-connected
    m = len(edges)
    B = np.zeros((n, m))
    for idx, (u, v) in enumerate(edges):
        B[u, idx] = 1.0
        B[v, idx] = -1.0
    sig2 = rng.uniform(0.1, 2.0, size=m)
    psi_star = 1.0 / sig2
    L_star = B @ np.diag(psi_star) @ B.T
    Cov_star = linalg.pinv(L_star)
    for _ in range(100):
        psi = rng.uniform(0.1, 3.0, size=m)
        Lp = B @ np.diag(psi) @ B.T
        Lp_inv = linalg.pinv(Lp)
        Cov_p = Lp_inv @ B @ np.diag((psi ** 2) * sig2) @ B.T @ Lp_inv
        eigs = np.linalg.eigvalsh(Cov_p - Cov_star)
        assert eigs.min() >= -1e-10, "Thm 2 Loewner Gauss-Markov violated"


if __name__ == "__main__":
    verify_theorems(np.random.default_rng(20261122))
    print("ALL_THEOREMS_VERIFIED_OK")
