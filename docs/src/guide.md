# User guide

## Data layout

The vectors of frame A are the columns of an ``n \times m`` matrix `X`, those of frame B the
columns of `Y`, with the ``i``-th column of each corresponding. `X` must have full row rank,
which needs ``m \ge n`` vectors in general position. For the Euclidean and similarity groups,
`X` and `Y` hold homogeneous coordinates, with a last row of ones.

## Groups

Every group is a value of a subtype of [`LieGroup`](@ref). The constructors are

| Group | Constructor | Notes |
|:--|:--|:--|
| ``GL(n,\mathbb R)``, ``GL(n,\mathbb C)`` | `GeneralLinear(n)`, `GeneralLinear(n; field = :complex)` | |
| ``SL(n,\mathbb R)``, ``SL(n,\mathbb C)`` | `SpecialLinear(n)`, `SpecialLinear(n; field = :complex)` | |
| ``SO(n)``, ``O(n)`` | `Orthogonal(n)`, `Orthogonal(n; special = false)` | |
| ``SO(n,\mathbb C)``, ``O(n,\mathbb C)`` | `Orthogonal(n; field = :complex)`, with `special = false` | |
| ``SU(n)``, ``U(n)`` | `Unitary(n)`, `Unitary(n; special = false)` | |
| ``SO(p,q)`` | `IndefiniteOrthogonal(p, q)` | metric ``\eta = \mathrm{diag}(I_p, -I_q)`` |
| ``SO(3,1)`` | `Lorentz()` | the same as `IndefiniteOrthogonal(1, 3)` |
| ``U(p,q)`` | `IndefiniteUnitary(p, q)` | |
| ``Sp(2m,\mathbb R)``, ``Sp(2m,\mathbb C)`` | `Symplectic(m)`, `Symplectic(m; field = :complex)` | `Symplectic(2)` is ``Sp(4)`` |
| ``\mathrm{Spin}(n)``, ``\mathrm{Pin}(n)`` | `Spin(n)`, `Pin(n)` | spinor representation of dimension ``2^{\lfloor n/2\rfloor}``, resp. ``2^{\lfloor (n+1)/2\rfloor}`` |
| ``SE(p,q)``, ``SE(n)`` | `SpecialEuclidean(p, q)`, `SpecialEuclidean(0, n)` | homogeneous coordinates |
| Poincaré ``SE(3,1)`` | `PoincareSE()` | |
| ``\mathrm{Sim}(n)`` | `Similarity(n)` | scalings, rotations and translations |

Each group knows its matrix size, `matsize(G)`, its Lie algebra projection,
`algebra_projection(G, L)`, an orthonormal basis of the algebra, `algebra_basis(G)`, and
membership tests, `in_group(G, g)` and `in_algebra(G, A)`. `random_group(G; scale)` and
`random_algebra(G)` draw test elements.

## Alignment methods

| Function | Paper | What it does |
|:--|:--|:--|
| [`naive_align`](@ref) | Algorithm 1 | projected principal logarithm of ``Y X^+``; throws when ``Y X^+`` has no real logarithm |
| [`branch_cut_align`](@ref) | Algorithm 2 | factors out the involution ``r`` first; exact on the whole group |
| [`negative_involution`](@ref), [`even_reflection`](@ref) | Algorithm 3 | the involution ``r`` from the Schur form of ``Y X^+`` |
| [`spin_align`](@ref) | Algorithm 4 | the ``\mathrm{Spin}(n)`` and ``\mathrm{Pin}(n)`` case, through the covering map |
| [`gauss_newton_refine`](@ref) | Algorithm 5 | Gauss–Newton on the group from the Lie algebra estimate |
| [`align_newton`](@ref), [`align_newton_full`](@ref), [`align_direct`](@ref) | §4 baselines | Newton with a finite difference or exact Hessian, and BFGS, through the exponential map |
| [`align_components`](@ref) | | the naive method on every component, with the data choosing the best |
| [`align_se`](@ref), [`align_se_centroid`](@ref) | §5.2 | rigid registration in homogeneous coordinates, and the centroid route |

`branch_cut_align` is the recommended estimator. It reduces to `naive_align` when
``Y X^+`` has no eigenvalue with negative real part, and otherwise it negates those
eigenvalues with an involution of the group before taking the logarithm, so that elements
of the other connected components, rotations by ``\pi``, and elements of ``SU(n)`` and
``SL(n,\mathbb C)`` whose principal logarithm is not traceless are all recovered exactly from
noiseless data.

## Accuracy

The Lie algebra estimate has a first order error in the noise (Theorem 3.1 of the paper).
For the compact groups ``SO(n)``, ``U(n)`` and ``SU(n)`` with isotropic inputs,
``X X^\dagger = c I``, it agrees with the least squares solution over the group to first
order (Corollary 3.3). Otherwise `gauss_newton_refine` reaches that solution, at a cost of
a few times the closed form. The refinement is not recommended when the inputs `X` are
themselves noisy, since the least squares objective is then biased.

`align_error(g_est, g_true)` returns the Frobenius and maximum entry errors.

## Complex groups

Complex matrices are handled directly; the projections use the real inner product
``\langle A, B\rangle = \mathrm{Re}\,\mathrm{tr}(A^\dagger B)``, because the Lie algebras of
the complex groups are real vector spaces.
