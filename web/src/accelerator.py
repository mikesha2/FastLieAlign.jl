"""Application 2: one-turn transfer map of a storage ring from turn-by-turn data.
Linear transverse motion is symplectic, so the one-turn map lies in Sp(4,R)."""
import numpy as np
from scipy.linalg import expm
from lialign import orthonormal_basis, lie_algebra_estimate, gauss_newton

rng = np.random.default_rng(2)
J = np.block([[np.zeros((2, 2)), np.eye(2)], [-np.eye(2), np.zeros((2, 2))]])
# coordinates ordered (x, y, px, py) so that J is the standard form
def proj(L):
    return 0.5 * (L + J @ L.T @ J)

def one_turn_map(Qx, Qy, coupling):
    def rot(mu): return np.array([[np.cos(mu), np.sin(mu)], [-np.sin(mu), np.cos(mu)]])
    R = np.zeros((4, 4)); mx, my = 2 * np.pi * Qx, 2 * np.pi * Qy
    Rx, Ry = rot(mx), rot(my)
    R[np.ix_([0, 2], [0, 2])] = Rx; R[np.ix_([1, 3], [1, 3])] = Ry
    # Twiss normalization (beta in m, alpha) and a small random coupling
    bx, ax, by, ay = 12.0, -1.1, 7.0, 0.6
    N = np.zeros((4, 4))
    N[np.ix_([0, 2], [0, 2])] = [[np.sqrt(bx), 0], [-ax / np.sqrt(bx), 1 / np.sqrt(bx)]]
    N[np.ix_([1, 3], [1, 3])] = [[np.sqrt(by), 0], [-ay / np.sqrt(by), 1 / np.sqrt(by)]]
    C = expm(coupling * proj(rng.standard_normal((4, 4))))
    Aff = C @ N
    return Aff @ R @ np.linalg.inv(Aff)

def tunes(M):
    lam = np.linalg.eigvals(M)
    up = lam[np.imag(lam) > 0]
    return np.sort(np.angle(up) / (2 * np.pi)), np.max(np.abs(np.abs(lam) - 1))

basis = orthonormal_basis(proj, 4)
print("dimension:", len(basis))
M = one_turn_map(0.31, 0.27, 0.05)
assert np.allclose(M.T @ J @ M, J)
true_tunes, _ = tunes(M)
print("true tunes", true_tunes)
turns = 256
for sigma_rel in [1e-3, 1e-2, 3e-2]:
    out = {k: [] for k in ["ls", "la", "gn"]}
    for _ in range(300):
        z = np.zeros((4, turns + 1)); z[:, 0] = [1e-3, 0.7e-3, 0.0, 0.0]  # 1 mm, 0.7 mm kick
        for k in range(turns):
            z[:, k + 1] = M @ z[:, k]
        scale = np.std(z, axis=1, keepdims=True)
        zn = z + sigma_rel * scale * rng.standard_normal(z.shape)
        X, Y = zn[:, :-1], zn[:, 1:]
        M0 = Y @ np.linalg.pinv(X)
        Mla = lie_algebra_estimate(Y, X, proj)
        Mgn = gauss_newton(Y, X, Mla, basis)
        for k, E in zip(out, [M0, Mla, Mgn]):
            t, dl = tunes(E)
            out[k].append((np.linalg.norm(E - M) / np.linalg.norm(M), np.max(np.abs(t - true_tunes)), dl,
                           np.linalg.norm(E.T @ J @ E - J)))
    for k in out:
        a = np.median(np.array(out[k]), 0)
        print(f"noise={sigma_rel:g} {k}: rel err {a[0]:.2e}  tune err {a[1]:.2e}  max||lambda|-1| {a[2]:.2e}  symplectic defect {a[3]:.1e}")
