# FastLieAlign.jl

Vector alignment in matrix Lie groups: given pairs of vectors ``v_{B,i} \approx g\, v_{A,i}``
related by an unknown element ``g`` of a classical matrix Lie group, recover ``g``.
The package implements the Lie algebra method of the paper

> C. M. Sha, *Vector alignment in matrix Lie groups*, Computer Physics Communications (submitted),
> code and reproduction scripts on [Zenodo](https://doi.org/10.5281/zenodo.21045016),

which generalizes [Optimal alignment of Lorentz orientation and generalization to matrix Lie
groups](https://arxiv.org/abs/2506.14994), J. Comput. Appl. Math. 485 (2026) 117482.

The estimate is a group element by construction and is computed in closed form:

1. ``g_0 = Y X^+``, the unconstrained least squares matrix,
2. ``r``, an involution of the group which moves the eigenvalues of ``g_0`` off the branch cut of the logarithm,
3. ``L = \log(r^{-1} g_0)``, the principal matrix logarithm,
4. ``g = r \exp(\mathrm{proj}_{\mathfrak g} L)``, the orthogonal projection onto the Lie algebra, exponentiated.

A Gauss–Newton refinement on the group reaches the least squares solution from this start in a few steps.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/USERNAME/FastLieAlign.jl")
```

The package needs Julia 1.10 or newer and depends on the standard library `LinearAlgebra`;
`ForwardDiff` and `ReverseDiff` are used only by the Newton and BFGS comparison methods.

## Quick start

```@example quickstart
using FastLieAlign

G = Lorentz()                                 # SO(3,1), the proper orthochronous Lorentz group
g_true = random_group(G; scale = 0.5)         # a random boost and rotation
X = randn(4, 12)                              # twelve four-vectors in frame A
Y = g_true * X .+ 0.01 .* randn(4, 12)        # the same vectors in frame B, with noise

g = branch_cut_align(G, X, Y)                 # Algorithm 2 of the paper
in_group(G, g), align_error(g, g_true)        # membership, and (Frobenius, max entry) errors
```

```@example quickstart
g_refined = gauss_newton_refine(G, X, Y; init = g)   # Algorithm 5: the least squares element of G
align_error(g_refined, g_true)
```

The naive method of the earlier paper, `naive_align`, is the same computation without the
involution ``r``. It fails when ``g_0`` has an eigenvalue on the negative real axis, for example
for a rotation by ``\pi``, which `branch_cut_align` handles:

```@example quickstart
G = Orthogonal(3)
g_true = exp(π * [0 -1 0; 1 0 0; 0 0 0])      # rotation by π about the z axis
X = randn(3, 12); Y = g_true * X
g = branch_cut_align(G, X, Y)
align_error(g, g_true)
```

## Contents

```@contents
Pages = ["guide.md", "reproduce.md", "demo.md", "api.md"]
Depth = 2
```

## License

MIT License, Copyright (c) 2026 Congzhou M. Sha.
