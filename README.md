# FastLieAlign.jl — vector alignment in matrix Lie groups

[![CI](https://github.com/USERNAME/FastLieAlign.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/USERNAME/FastLieAlign.jl/actions/workflows/CI.yml)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://USERNAME.github.io/FastLieAlign.jl/dev/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21045016.svg)](https://doi.org/10.5281/zenodo.21045016)

Code and reproduction scripts for the manuscript ***Vector alignment in matrix Lie
groups*** (Congzhou M. Sha). This repository contains a Julia package implementing
the alignment methods, the scripts which reproduce the figures and tables of the
manuscript, the documentation site and an interactive browser demo.

```julia
using Pkg; Pkg.add(url = "https://github.com/USERNAME/FastLieAlign.jl")
using FastLieAlign
G = Lorentz(); g = branch_cut_align(G, X, Y)       # X, Y: 4×m matrices of four-vectors
```

The documentation, with a user guide and the API reference, is at
<https://USERNAME.github.io/FastLieAlign.jl/dev/>, and the browser demo at
<https://USERNAME.github.io/FastLieAlign.jl/dev/demo/>.

The work generalizes the earlier study

> Congzhou M. Sha, *Optimal alignment of Lorentz orientation and generalization
> to matrix Lie groups*, J. Comput. Appl. Math. **485** (2026) 117482
> ([arXiv:2506.14994](https://arxiv.org/abs/2506.14994))

to every classical matrix Lie group over both the real and complex fields and adds
a Gauss–Newton refinement on the group.

**Version 2 (October 2026).** This version accompanies the revised manuscript.
The Julia code and figures are unchanged from version 1.

**Version 3 (October 2026).** The Julia module gains `branch_cut_align`, the extension of the method which is exact on the whole
group (an involution computed from the negative eigenvalues for the real groups; a
reflection of an even number of eigenvalues for SL(n,C) and SU(n), which makes the
logarithm traceless at the cost of one Schur form; and for Spin(n) the involution and
the branch of the logarithm taken from the covered rotation `spin_covering(g0)` in
SO(n), lifted by `spin_lift` and `spin_lift_involution`, with the central sign chosen
by the data; Theorem 2.3 of the manuscript),
`align_components`, an `init` keyword for `gauss_newton_refine`, and an informative error
when the least squares matrix has no real logarithm. It also adds the similarity group
`Similarity(n)` of scalings, rotations and translations, Sim(n), with its block projection,
the complex orthogonal groups (`Orthogonal(n; field = :complex)`), the indefinite unitary
groups `IndefiniteUnitary(p, q)`, and `Pin(n)`, the double cover of O(n), realized with the
generators `ε_a = i γ_a γ_{n+1}` and the grading `Γ = γ_{n+1}` of a Clifford representation
with n+1 generators, for which `spin_covering` is the twisted adjoint representation onto O(n)
and `spin_align` lifts reflections of odd-dimensional subspaces,
and the scripts `reproduce/worked_example.jl` and `reproduce/worked_example_su3.jl`
behind the worked examples of Appendix C of the manuscript. `reproduce/reproduce.jl` had a timing bug
in the SO(4) study (the baseline timed a precomputed matrix, and warm starts
omitted the cost of the Lie algebra estimate), which affected the SO(4) timings
quoted in earlier versions of the manuscript; it is fixed, the SO(4) table now
includes the Kabsch (SVD) solution, the polar factor of the least squares estimate and a cold-started Gauss-Newton run, the
medians behind the timing figure are printed, and the timings in the manuscript
were re-measured on an Apple M5 Max MacBook Pro. Seven groups were added to the error histograms (Figures S12–S18).

Copyright (c) 2026 Congzhou M. Sha, consha@sas.upenn.edu. Released under the MIT License; see [`LICENSE`](LICENSE).

---

## The problem

Given corresponding vectors `{v_{A,i}}` (frame *A*) and `{v_{B,i}}` (frame *B*)
sharing a common origin, recover the matrix Lie group element `g ∈ G` with
`g v_{A,i} ≈ v_{B,i}`. The **Lie algebra method** is a four-step closed-form
algorithm that generalizes to *any* matrix Lie group (`X`, `Y` stack the vectors
as columns; only `proj_𝔤` is group-specific):

1. `g₀ = Y X⁺` — unconstrained least squares (Moore–Penrose pseudoinverse)
2. `l₀ = log g₀` — principal matrix logarithm
3. `l  = proj_𝔤(l₀)` — orthogonal (Frobenius) projection onto the Lie algebra
4. `g  = exp l` — matrix exponential

Six alignment methods are implemented: `naive_align` (Lie algebra, closed form; Algorithm 1), `branch_cut_align` (the same, extended to the whole group; Algorithm 2), `spin_align` (the `Spin(n)` case through the covering map; Algorithm 4),
`align_direct` (direct BFGS optimization), `align_newton` /
`align_newton_full` (Newton optimization, finite-difference or exact
forward-over-reverse Hessian), and `gauss_newton_refine` (Gauss–Newton refinement
on the group, warm-started from `naive_align`; Algorithm 5). The involution `r` of Algorithm 2 is computed from the Schur decomposition of `Y X⁺` by `negative_involution` and `even_reflection` (Algorithm 3). On noiseless data satisfying the
exactness condition of Section 2.3 of the manuscript (full-row-rank data; ground
truth `exp(A)` with `A ∈ 𝔤` and spectrum in `|Im λ| < π`), all agree to machine
precision. Under noise, the refinement reaches the same least-squares optimum
as the direct optimizers at a fraction of their cost.

---

## Contents

This archive contains:

| Path | Contents |
|------|----------|
| `src/FastLieAlign.jl` | the Julia module: group types, projections, and the alignment methods |
| `reproduce/reproduce.jl` | **single script** that regenerates every figure and the SO(4) table |
| `reproduce/extra_experiments.jl` | the additional checks quoted in the text (cold starts, unrefined SE(3), Lorentz error against boost size) |
| `reproduce/worked_example.jl`, `reproduce/worked_example_su3.jl` | the SO(3) and SU(3) worked examples of Appendix C, standard library only |
| `test/runtests.jl` | Julia test suite |
| `Project.toml`, `Manifest.toml` | pinned Julia environment |
| `reproduce/qubits.jl` | the eight-qubit circuit example of Section 5.1 (Table 8) |
| `README.md`, `LICENSE` | this file and the license |
| `docs/` | the Documenter.jl source of the documentation site (GitHub repository only) |
| `web/` | the browser demo, WebAssembly and WebGPU (GitHub repository only) |
| `lean/` | Lean 4 formalizations of the projections and of Theorem 2.3(a) (GitHub repository only) |

`reproduce/reproduce.jl` is self-contained.

---

## Julia: methods, tests, and reproduction

### Install Julia

Install Julia 1.10 or newer via [`juliaup`](https://github.com/JuliaLang/juliaup):

```bash
curl -fsSL https://install.julialang.org | sh    # installs juliaup + Julia
```

### Set up the environment

From the repository root, instantiate the pinned environment (downloads the
exact dependency versions in `Manifest.toml`):

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Run the tests

```bash
julia --project=. test/runtests.jl
```

This checks the projections, group membership, exact recovery on noiseless
data, and noisy robustness across all groups.

### Reproduce every figure and finding

```bash
julia --project=. reproduce/reproduce.jl
```

This single script regenerates, into `reproduce/figures/`:

| Output | Manuscript item |
|--------|-----------------|
| printed SO(4) table | Table 6(a) (Kabsch, polar factor of the least squares estimate, Lie algebra method, Gauss–Newton refinement, AD-Newton timing) |
| `gr3.pdf` | noiseless cross-group recovery error (reported in the text of Section 4) |
| `gr4.pdf` | Figure 2 (cross-group timing under noise) |
| `gr5.pdf` | SE(3) homogeneous vs. centroid route (reported in the text of Section 5.2) |
| `pergroup_<tag>.pdf` | Figures S1–S18 of the supplement (per-group error histograms) |
| printed whole-group table (from `reproduce/whole_group_benchmark.jl`) | Table 7 (Algorithm 2 on elements beyond the naive method) |
| `branch_error.pdf` (from `reproduce/figure_branch.jl`) | Figure 1(c) (error near a rotation by π; panels (a) and (b) are drawn in the manuscript source) |
| printed Lorentz table | Table 6(b) (Lorentz error against the scale of the transformation) |
| printed circuit table (from `reproduce/qubits.jl`) | Table 8 (eight-qubit circuits in U(256)) |

Default trial counts match the manuscript (1000 per cell for the histograms and
the SO(4) and Lorentz studies; 100 for the cross-group bars). They can be
lowered for a quick run via environment variables:

```bash
CROSS_TRIALS=20 PERGROUP_TRIALS=100 SO4_TRIALS=100 LORENTZ_TRIALS=100 \
  julia --project=. reproduce/reproduce.jl
```

The random seed is set once at the start of the SO(4) study, so the figures are
reproduced exactly only when `main()` runs the studies in order. A full run takes
a few minutes on a modern laptop; the manuscript timings were
measured on an Apple M5 Max MacBook Pro (version 3) with single-shot wall-clock
timers, and vary by about 10% between runs.

---

## Repository layout

- `src/FastLieAlign.jl` — the Julia module.
- `reproduce/` — the single reproduction script and its output figures.
- `test/` — the Julia test suite.
