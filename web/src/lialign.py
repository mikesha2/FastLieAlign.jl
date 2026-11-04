"""Minimal Python implementation of the Lie algebra alignment method and its
Gauss--Newton refinement, used for the application examples in the paper.
The reference implementation is the Julia package in the Zenodo archive."""
import numpy as np
from scipy.linalg import expm, logm, polar


def orthonormal_basis(proj, n, field=float):
    """Frobenius-orthonormal real basis of the image of `proj` (a real subspace)."""
    gens = []
    for i in range(n):
        for j in range(n):
            E = np.zeros((n, n), dtype=complex if field is complex else float)
            E[i, j] = 1
            gens.append(E)
            if field is complex:
                gens.append(1j * E)
    basis = []
    for E in gens:
        P = proj(E)
        for b in basis:
            P = P - np.real(np.vdot(b, P)) * b
        nrm = np.linalg.norm(P)
        if nrm > 1e-10:
            basis.append(P / nrm)
    return basis


def lie_algebra_estimate(Y, X, proj, real=True):
    g0 = Y @ np.linalg.pinv(X)
    L = logm(g0)
    if real:
        L = L.real
    return expm(proj(L))


def gauss_newton(Y, X, g, basis, tol=1e-13, maxit=12):
    for _ in range(maxit):
        R = Y - g @ X
        cols = [(b @ g @ X).ravel() for b in basis]
        D = np.stack(cols, 1)
        r = R.ravel()
        if np.iscomplexobj(D) or np.iscomplexobj(r):
            D = np.vstack([D.real, D.imag]); r = np.concatenate([r.real, r.imag])
        c = np.linalg.lstsq(D, r, rcond=None)[0]
        delta = sum(ck * b for ck, b in zip(c, basis))
        if np.linalg.norm(delta) <= tol * max(1.0, np.linalg.norm(g)):
            break
        g = expm(delta) @ g
    return g
