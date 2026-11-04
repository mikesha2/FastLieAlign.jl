# Reproducing the paper

## Install Julia

Install Julia 1.10 or newer via [`juliaup`](https://github.com/JuliaLang/juliaup):

```bash
curl -fsSL https://install.julialang.org | sh    # installs juliaup + Julia
```

## Set up the environment

From the repository root, instantiate the pinned environment (downloads the
exact dependency versions in `Manifest.toml`):

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Run the tests

```bash
julia --project=. test/runtests.jl
```

This checks the projections, group membership, exact recovery on noiseless
data, and noisy robustness across all groups.

## Reproduce every figure and finding

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
