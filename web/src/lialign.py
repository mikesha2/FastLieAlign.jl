"""Minimal Python implementation of the Lie algebra alignment method: the naive
method (Algorithm 1 of the paper, `lie_algebra_estimate`), the branch-cut aware
method (Algorithm 2, `branch_cut_align`) with its involution computed from the
Schur form (Algorithm 3, `negative_involution` and `even_reflection`), and the
Gauss--Newton refinement (Algorithm 5, `gauss_newton`).  The spin groups
(Algorithm 4) are not covered here.  The reference implementation is the Julia
package FastLieAlign.jl (https://github.com/mikesha2/FastLieAlign.jl)."""
import numpy as np
from scipy.linalg import expm, logm, polar, schur
from scipy.linalg.lapack import ztrsen


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


def negative_involution(g0, M=None, sesquilinear=False, block=None):
    """The involution r = I - 2P of Algorithm 3 for the groups preserving a form M
    (bilinear, star = transpose; or sesquilinear, star = conjugate transpose), or for
    GL(n), SL(n, R) with M = None (then M = I and r is a unitary reflection).
    P projects onto N, the invariant subspace of g0 for the eigenvalues with
    Re lambda < 0, along its complement, and r negates exactly those eigenvalues.
    For the Euclidean and similarity groups pass block=m, the size of the rotation
    or Lorentz block; r is then computed from that block and extended by 1.
    Returns the identity when no eigenvalue has negative real part."""
    n = g0.shape[0]
    if block is not None:
        r = np.eye(n, dtype=g0.dtype)
        r[:block, :block] = negative_involution(g0[:block, :block], M, sesquilinear)
        return r
    real = not np.iscomplexobj(g0)
    # the Schur form with the eigenvalues of negative real part ordered first;
    # the real Schur form keeps conjugate pairs together in 2x2 blocks
    T, Z, sdim = schur(g0, output='real' if real else 'complex', sort='lhp')
    if sdim == 0:
        return np.eye(n, dtype=g0.dtype)
    if sdim == n:
        return -np.eye(n, dtype=g0.dtype)
    Q = Z[:, :sdim]
    Mm = np.eye(n) if M is None else M
    Qs = Q.conj().T if sesquilinear else Q.T
    P = Q @ np.linalg.solve(Qs @ Mm @ Q, Qs @ Mm)
    return np.eye(n) - 2 * P


def even_reflection(g0):
    """The involution of Algorithm 3 for SU(n) and SL(n, C): a unitary reflection
    r = I - 2 Q_S Q_S^H of the invariant subspace of an even number of eigenvalues,
    chosen among those of largest argument so that the principal logarithm of r g0
    is traceless (Eq. (flipcount) of the paper)."""
    n = g0.shape[0]
    T, Z = schur(np.asarray(g0, dtype=complex), output='complex')
    theta = np.angle(np.diag(T))
    k = int(np.rint(theta.sum() / (2 * np.pi)))
    neg = sorted((i for i in range(n) if theta[i] < 0), key=lambda i: -abs(theta[i]))  # s_i = +1
    pos = sorted((i for i in range(n) if theta[i] > 0), key=lambda i: -abs(theta[i]))  # s_i = -1
    cost = lambda i: np.pi * (np.pi - 2 * abs(theta[i]))
    cneg = np.concatenate([[0.0], np.cumsum([cost(i) for i in neg])])
    cpos = np.concatenate([[0.0], np.cumsum([cost(i) for i in pos])])
    best = None
    for a in range(len(neg) + 1):          # a flips from neg, b from pos, a - b = -2k
        b = a + 2 * k
        if 0 <= b <= len(pos):
            c = cneg[a] + cpos[b]
            if best is None or c < best[0]:
                best = (c, a, b)
    if best is None:
        raise ValueError("no admissible set of eigenvalues to reflect")
    S = neg[:best[1]] + pos[:best[2]]
    if not S:
        return np.eye(n, dtype=complex)
    select = np.zeros(n, dtype=np.int32); select[S] = 1
    T2, Z2, *_ = ztrsen(select, T, Z, job=b'N')   # reorder the Schur form so that S comes first
    Q = Z2[:, :len(S)]
    return np.eye(n) - 2 * Q @ Q.conj().T


def branch_cut_align(Y, X, proj, M=None, sesquilinear=False, real=True,
                     traceless_complex=False, block=None):
    """Algorithm 2 of the paper: g = r exp(proj(log(r^{-1} g0))) with g0 = Y X^+ and
    r the involution of Algorithm 3.  Pass the form M and whether it is sesquilinear
    for the groups preserving a form (M=None for GL(n), SL(n, R)), real=False for
    complex groups, traceless_complex=True for SU(n) and SL(n, C), and block=m for
    the Euclidean and similarity groups in homogeneous coordinates.  Exact on the
    whole group for noiseless data (Theorem 2.3 of the paper)."""
    g0 = Y @ np.linalg.pinv(X)
    r = even_reflection(g0) if traceless_complex else negative_involution(g0, M, sesquilinear, block)
    L = logm(np.linalg.solve(r, g0))
    if real:
        L = L.real
    return r @ expm(proj(L))
