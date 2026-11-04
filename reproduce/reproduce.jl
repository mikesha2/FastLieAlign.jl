#!/usr/bin/env julia
# =============================================================================
# reproduce.jl  --  one-shot reproduction of every figure and numerical finding
# in "Vector alignment in matrix Lie groups" (Congzhou M. Sha).
#
# This single script regenerates, from scratch:
#   * Table 6(a): SO(4) cost of the correction vs. automatic-differentiation
#                Newton (printed to stdout).
#   * gr3.pdf  : cross-group recovery error, noiseless (quoted in Section 4).
#   * Figure 2 : cross-group timing under noise (gr4.pdf).
#   * gr5.pdf  : SE(3) homogeneous vs. centroid route, both corrected (quoted in Section 5.2).
#   * Figures S1-S18 of the supplement : per-group recovery-error histograms (pergroup_<tag>.pdf).
#   Figure 1 (branch.pdf) is produced by reproduce/figure_branch.jl.
#   * the Lorentz accuracy findings (printed to stdout).
#
# All figures are written to reproduce/figures/.  Run from the repository root:
#
#     julia --project=. reproduce/reproduce.jl
#
# Environment variables (optional) trade accuracy for speed:
#     CROSS_TRIALS   trials for Figs 1-2  (default 100)
#     PERGROUP_TRIALS trials for the per group histograms (Figs. S1-S11) (default 1000)
#     SO4_TRIALS     trials for Table 6(a) (default 1000)
#     LORENTZ_TRIALS trials for the Lorentz study (default 1000)
#
# Dependencies (see the repository Project.toml): LinearAlgebra, Statistics,
# Random, Printf, Plots, StatsPlots, ReverseDiff, ForwardDiff, HypothesisTests,
# LaTeXStrings.  The figure pipeline is pure Julia (no Python).
# =============================================================================

using LinearAlgebra, Statistics, Random, Printf
using Plots, StatsPlots, LaTeXStrings
using HypothesisTests
gr()

const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "src", "FastLieAlign.jl"))
using .FastLieAlign

const OUT = joinpath(@__DIR__, "figures")
isdir(OUT) || mkpath(OUT)

cross_trials()    = parse(Int, get(ENV, "CROSS_TRIALS",    "100"))
pergroup_trials() = parse(Int, get(ENV, "PERGROUP_TRIALS", "1000"))
so4_trials()      = parse(Int, get(ENV, "SO4_TRIALS",      "1000"))
lorentz_trials()  = parse(Int, get(ENV, "LORENTZ_TRIALS",  "1000"))

const NOISE = 0.05   # additive-noise standard deviation for the timing studies

# -----------------------------------------------------------------------------
# The seven methods compared throughout, in plotting order: the closed-form Lie
# algebra estimate, the data-weighted correction, the direct BFGS optimizer, and
# automatic-differentiation Newton in four configurations (finite-difference vs.
# exact forward-over-reverse Hessian, each warm- or cold-started).
# -----------------------------------------------------------------------------
const SERIES_LABELS = ["Lie algebra" "Corrected" "BFGS" "Newton FD, warm" "Newton full, warm" "Newton FD, cold" "Newton full, cold"]
const SERIES_COLORS = [RGBA(0.27,0.47,0.67,0.9) RGBA(0.67,0.20,0.47,0.9) RGBA(0.40,0.40,0.40,0.9) RGBA(0.97,0.70,0.30,0.9) RGBA(0.45,0.75,0.35,0.9) RGBA(0.93,0.47,0.20,0.9) RGBA(0.13,0.53,0.20,0.9)]
# In the bar charts the slow BFGS optimizer (column 3) is drawn last.
const BAR_PERM = [1, 2, 4, 5, 6, 7, 3]

# Returns the seven recovered group elements for one instance (X, Y).
function bench_methods(G, X, Y)
    Λl = naive_align(G, X, Y)
    (Λl,
     gauss_newton_refine(G, X, Y),
     align_direct(G, X, Y),
     align_newton(G, X, Y; init = Λl),
     align_newton_full(G, X, Y; init = Λl),
     align_newton(G, X, Y),
     align_newton_full(G, X, Y))
end

iscomplex(G) = eltype(FastLieAlign.identity_element(G)) <: Complex
randmat(m, cplx) = cplx ? randn(m, 3m) + im*randn(m, 3m) : randn(m, 3m)

# Groups for the cross-group bar charts (Figs 1-2).
const BAR_GROUPS = [
    ("GL(4,ℝ)", GeneralLinear(4)), ("SL(4,ℝ)", SpecialLinear(4)),
    ("SO(5)", Orthogonal(5)), ("SU(3)", Unitary(3; special = true)),
    ("SO(3,1)", Lorentz()), ("SO(2,3)", IndefiniteOrthogonal(2, 3)),
    ("Sp(4,ℝ)", Symplectic(2)),
]

# Groups for the per-group histograms (Figs. S1-S18 of the supplement).
const PG_GROUPS = [
    ("GL(4,R)", "gl4r", GeneralLinear(4)), ("SL(4,R)", "sl4r", SpecialLinear(4)),
    ("SO(5)", "so5", Orthogonal(5)), ("U(3)", "u3", Unitary(3)),
    ("SU(3)", "su3", Unitary(3; special = true)), ("SO(3,1)", "so31", Lorentz()),
    ("SO(2,3)", "so23", IndefiniteOrthogonal(2, 3)), ("Sp(4,R)", "sp4r", Symplectic(2)),
    ("Sp(4,C)", "sp4c", Symplectic(2; field = :complex)), ("Spin(5)", "spin5", Spin(5)),
    ("SE(3,1)", "se31", PoincareSE()),
    ("GL(3,C)", "gl3c", GeneralLinear(3; field = :complex)),
    ("SL(3,C)", "sl3c", SpecialLinear(3; field = :complex)),
    ("U(2,1)", "u21", IndefiniteUnitary(2, 1)),
    ("SO(3,C)", "so3c", Orthogonal(3; field = :complex)),
    ("SE(3)", "se3", SpecialEuclidean(0, 3)),
    ("Sim(3)", "sim3", Similarity(3)),
    ("Pin(3)", "pin3", Pin(3)),
]

# -----------------------------------------------------------------------------
# Table 6(a) of the manuscript: SO(4) cost of the correction vs. AD-Newton.
# -----------------------------------------------------------------------------
function so4_table(; trials = so4_trials())
    G = Orthogonal(4); m = matsize(G); ε = NOISE
    labels = ["Kabsch (SVD)", "Polar factor of Y X^+", "Lie algebra (baseline)", "Gauss-Newton, warm start", "Gauss-Newton, cold start",
              "AD-Newton, FD-Hess, warm", "AD-Newton, full-Hess, warm",
              "AD-Newton, FD-Hess, cold", "AD-Newton, full-Hess, cold"]
    kabsch(X, Y) = (F = svd(Y * X'); U = F.U; V = F.Vt; U * Diagonal([ones(size(X, 1) - 1); sign(det(U * V))]) * V)
    # nearest rotation to the unconstrained least squares estimate g0 = Y X^+
    polar(X, Y) = (F = svd(Y / X); U = F.U; V = F.Vt; U * Diagonal([ones(size(X, 1) - 1); sign(det(U * V))]) * V)
    eye = Matrix{Float64}(I, m, m)
    # warmup
    let X = randn(m, 3m), Λ = random_group(G; scale = 0.3), Y = Λ*X .+ ε.*randn(m, 3m)
        bench_methods(G, X, Y)
    end
    T = [Float64[] for _ in 1:9]; E = [Float64[] for _ in 1:9]
    Random.seed!(0); GC.gc()
    for _ in 1:trials
        X = randn(m, 3m); Λ = random_group(G; scale = 0.3); Y = Λ*X .+ ε.*randn(m, 3m)
        # every run is timed from the raw data; warm starts include the cost of the Lie algebra estimate
        runs = ((() -> kabsch(X, Y)),
                (() -> polar(X, Y)),
                (() -> naive_align(G, X, Y)),
                (() -> gauss_newton_refine(G, X, Y)),
                (() -> gauss_newton_refine(G, X, Y; init = eye)),
                (() -> (a = naive_align(G, X, Y); align_newton(G, X, Y; init = a))),
                (() -> (a = naive_align(G, X, Y); align_newton_full(G, X, Y; init = a))),
                (() -> align_newton(G, X, Y)),
                (() -> align_newton_full(G, X, Y)))
        for (s, f) in enumerate(runs)
            t0 = time_ns(); r = f(); push!(T[s], (time_ns()-t0)/1e3); push!(E[s], norm(r - Λ))
        end
    end
    tL = median(T[3])
    println("\n=== Table 6(a) of the manuscript: SO(4), $trials trials, noise ε=$ε ===")
    @printf("  %-28s %12s %9s %14s\n", "method", "median time", "vs Lie", "recovery err")
    for s in 1:9
        @printf("  %-28s %9.1f μs %7.1fx %14.3e\n",
                labels[s], median(T[s]), median(T[s])/tL, median(E[s]))
    end
end

# Draw a compact legend with small square color swatches.  Plots.jl has no
# swatch-size attribute for bar series, so we suppress the automatic legend and
# add invisible scatter series whose square markers serve as the swatches.
function square_legend!(fig, labels, colors; loc, fontsize = 6, ms = 3)
    for s in eachindex(labels)
        scatter!(fig, [NaN], [NaN]; label = labels[s], color = colors[s],
                 markershape = :rect, markersize = ms, markerstrokewidth = 0.3)
    end
    plot!(fig; legend = loc, legendfontsize = fontsize)
    fig
end

# -----------------------------------------------------------------------------
# gr3: cross-group recovery error, noiseless.
# -----------------------------------------------------------------------------
function figure1_accuracy(; trials = cross_trials())
    names = String[]; med = [Float64[] for _ in 1:7]
    for (name, G) in BAR_GROUPS
        m = matsize(G); cplx = iscomplex(G)
        r = [Float64[] for _ in 1:7]
        for _ in 1:trials
            X = randmat(m, cplx)
            Λ = random_group(G; scale = 0.3); nΛ = norm(Λ)
            Y = Λ * X                       # noiseless
            for (s, g) in enumerate(bench_methods(G, X, Y))
                push!(r[s], norm(g - Λ)/nΛ)
            end
        end
        push!(names, name)
        for s in 1:7; push!(med[s], median(r[s])); end
    end
    M = hcat(med...)[:, BAR_PERM]
    ylo = minimum(M)/2; yhi = maximum(M)*8
    fig = groupedbar(names, M; yscale = :log10, label = "", legend = false,
        bar_position = :dodge, color = SERIES_COLORS[:, BAR_PERM],
        ylabel = "median relative error", xlabel = "group", xrotation = 45,
        size = (900, 470), ylims = (ylo, yhi), tickfontsize = 7,
        left_margin = 8Plots.mm, bottom_margin = 12Plots.mm)
    square_legend!(fig, vec(SERIES_LABELS[:, BAR_PERM]), vec(SERIES_COLORS[:, BAR_PERM]); loc = :topright)
    savefig(fig, joinpath(OUT, "gr3.pdf")); println("wrote ", joinpath(OUT, "gr3.pdf"))
end

# -----------------------------------------------------------------------------
# Figure 2 (gr4): cross-group timing under noise (BFGS drawn last).
# -----------------------------------------------------------------------------
function figure2_timing(; trials = cross_trials(), noise = NOISE)
    names = String[]; med = [Float64[] for _ in 1:7]
    for (name, G) in BAR_GROUPS
        m = matsize(G); cplx = iscomplex(G)
        mkX() = randmat(m, cplx)
        mkE() = cplx ? noise*(randn(m, 3m) + im*randn(m, 3m)) : noise*randn(m, 3m)
        let X = mkX(); Λ = random_group(G; scale = 0.3); Y = Λ*X .+ mkE()
            bench_methods(G, X, Y)
        end
        t = [Float64[] for _ in 1:7]; GC.gc()
        for _ in 1:trials
            X = mkX(); Λ = random_group(G; scale = 0.3); Y = Λ*X .+ mkE()
            t0 = time_ns(); naive_align(G, X, Y);                push!(t[1], (time_ns()-t0)/1e3)
            t0 = time_ns(); gauss_newton_refine(G, X, Y);      push!(t[2], (time_ns()-t0)/1e3)
            t0 = time_ns(); align_direct(G, X, Y);         push!(t[3], (time_ns()-t0)/1e3)
            t0 = time_ns(); a = naive_align(G, X, Y); align_newton(G, X, Y; init = a);      push!(t[4], (time_ns()-t0)/1e3)
            t0 = time_ns(); a = naive_align(G, X, Y); align_newton_full(G, X, Y; init = a); push!(t[5], (time_ns()-t0)/1e3)
            t0 = time_ns(); align_newton(G, X, Y);         push!(t[6], (time_ns()-t0)/1e3)
            t0 = time_ns(); align_newton_full(G, X, Y);    push!(t[7], (time_ns()-t0)/1e3)
        end
        push!(names, name)
        for s in 1:7; push!(med[s], median(t[s])); end
    end
    # print the medians behind Fig. 2 (μs per alignment)
    println("\n=== Fig. 2: median time per alignment (μs), $trials trials, noise ε=$noise ===")
    @printf("  %-10s", "group"); for l in vec(SERIES_LABELS); @printf(" %12s", l); end; println()
    for (i, name) in enumerate(names)
        @printf("  %-10s", name); for s in 1:7; @printf(" %12.1f", med[s][i]); end; println()
    end
    M = hcat(med...)[:, BAR_PERM]
    ylo = minimum(M)/2; yhi = 1e6
    decades = [10.0^k for k in floor(Int, log10(ylo)):6]
    fig = groupedbar(names, M; yscale = :log10, label = "", legend = false,
        bar_position = :dodge, color = SERIES_COLORS[:, BAR_PERM],
        ylabel = "time per alignment (μs)", xlabel = "group", xrotation = 45,
        size = (900, 470), ylims = (ylo, yhi), yticks = decades,
        tickfontsize = 7, left_margin = 8Plots.mm, bottom_margin = 12Plots.mm)
    square_legend!(fig, vec(SERIES_LABELS[:, BAR_PERM]), vec(SERIES_COLORS[:, BAR_PERM]); loc = :topleft)
    savefig(fig, joinpath(OUT, "gr4.pdf")); println("wrote ", joinpath(OUT, "gr4.pdf"))
end

# -----------------------------------------------------------------------------
# gr5: SE(3) homogeneous vs. centroid route, both corrected.
# -----------------------------------------------------------------------------
function se_centroid_corrected(G::SpecialEuclidean, PA, PB)
    n = G.p + G.q; m = size(PA, 2)
    cA = vec(sum(PA, dims = 2))./m; cB = vec(sum(PB, dims = 2))./m
    QA = Matrix(PA) .- cA; QB = Matrix(PB) .- cB
    Λ = gauss_newton_refine(rotation_group(G), QA, QB)
    M = zeros(n+1, n+1); M[1:n, 1:n] = Λ; M[1:n, n+1] = cB - Λ*cA; M[n+1, n+1] = 1
    M
end

function figure3_se(; trials = lorentz_trials())
    G = SpecialEuclidean(0, 3); εs = [0.0, 0.01, 0.05]
    labels = ["noiseless", "ε = 0.01", "ε = 0.05"]
    HOM = RGBA(0.27, 0.47, 0.67, 0.6); ALT = RGBA(0.93, 0.47, 0.20, 0.6)
    homog(P) = vcat(Matrix(P), ones(1, size(P, 2)))
    panels = Plots.Plot[]
    for (k, ε) in enumerate(εs)
        eh = Float64[]; ec = Float64[]
        for _ in 1:trials
            R = random_group(Orthogonal(3); scale = 0.6); t = randn(3)
            m = 15; PA = randn(3, m); PB = R*PA .+ t .+ (ε == 0 ? 0 : ε*randn(3, m))
            Mtrue = [R t; zeros(1, 3) 1]
            push!(eh, norm(gauss_newton_refine(G, homog(PA), homog(PB)) - Mtrue))
            push!(ec, norm(se_centroid_corrected(G, PA, PB) - Mtrue))
        end
        both = vcat(filter(>(0), eh), filter(>(0), ec))
        lo, hi = quantile(both, [0.01, 0.99]); hi <= lo && (hi = lo*10)
        edges = 10.0 .^ range(log10(lo), log10(hi); length = 26)
        first = k == 1; last = k == length(εs)
        p = histogram(eh; bins = edges, xscale = :log10, color = HOM, lw = 0,
            label = last ? "SE(3) corrected" : "", legend = last ? :topright : false)
        histogram!(p, ec; bins = edges, xscale = :log10, color = ALT, lw = 0,
            label = last ? "centroid + SO(3) corrected" : "")
        plot!(p; title = labels[k], titlefontsize = 8, xlims = (lo, hi),
            xlabel = L"\|G_{\mathrm{est}} - G_{\mathrm{GT}}\|_F", xguidefontsize = 8,
            tickfontsize = 6, ylabel = first ? "count" : "", legendfontsize = 7)
        push!(panels, p)
    end
    fig = plot(panels...; layout = (1, 3), size = (820, 240), link = :y,
        bottom_margin = 4Plots.mm, left_margin = 3Plots.mm)
    savefig(fig, joinpath(OUT, "gr5.pdf")); println("wrote ", joinpath(OUT, "gr5.pdf"))
end

# -----------------------------------------------------------------------------
# Figures S1-S11 of the supplement: per-group recovery-error histograms.  Rows are displayed in the
# order Lie, corrected, Newton, BFGS; columns are the three noise levels.
# -----------------------------------------------------------------------------
const PG_METHODS = ["Lie", "corrected", "Newton", "BFGS"]
const PG_COLORS = [RGBA(0.27,0.47,0.67,0.85), RGBA(0.67,0.20,0.47,0.85),
                   RGBA(0.13,0.53,0.20,0.85), RGBA(0.40,0.40,0.40,0.85)]
const PG_XLABEL = L"\|G_{\mathrm{method}} - G_{\mathrm{GT}}\|_F"

# row-displayed order maps to (Lie, BFGS/direct, Newton, corrected) raw series.
pergroup_series(G, X, Y) =
    (naive_align(G, X, Y), gauss_newton_refine(G, X, Y), align_newton(G, X, Y), align_direct(G, X, Y))

function maxbin(data, edges)
    counts = zeros(Int, length(edges)-1)
    nb = length(edges) - 1
    for x in data
        for b in 1:nb
            if edges[b] <= x < edges[b+1] || (b == nb && x == edges[end])
                counts[b] += 1
                break           # leaves only the inner loop
            end
        end
    end
    isempty(counts) ? 1 : maximum(counts)
end

function figures_pergroup(; trials = pergroup_trials())
    εs = [0.0, 0.01, 0.05]; εlabels = ["noiseless", "ε = 0.01", "ε = 0.05"]
    for (name, tag, G) in PG_GROUPS
        m = matsize(G); cplx = iscomplex(G)
        cols = []   # per noise level: (series4, lo, hi, edges, ymax)
        for ε in εs
            series = [Float64[] for _ in 1:4]   # rows: Lie, corrected, Newton, BFGS
            for _ in 1:trials
                X = G isa SpecialEuclidean || G isa Similarity ? (V = randn(m, 3m); V[end, :] .= 1; V) :
                    randmat(m, cplx)
                Λ = random_group(G; scale = 0.3)
                Y = ε == 0 ? Λ*X :
                    Λ*X .+ ε.*(cplx ? randn(m, 3m) + im*randn(m, 3m) : randn(m, 3m))
                for (s, g) in enumerate(pergroup_series(G, X, Y))
                    push!(series[s], norm(g - Λ))
                end
            end
            both = vcat((filter(>(0), s) for s in series)...)   # zero errors cannot be shown on a log axis
            lo, hi = extrema(both); hi <= lo && (hi = lo*10)      # full range of the data
            edges = 10.0 .^ range(log10(lo), log10(hi); length = 26)
            ymax = maximum(maxbin(s, edges) for s in series)
            push!(cols, (series, lo, hi, edges, ymax))
        end
        panels = Plots.Plot[]
        for s in 1:4, (ei, col) in enumerate(cols)
            series, lo, hi, edges, ymax = col
            p = histogram(series[s]; bins = edges, xscale = :log10, color = PG_COLORS[s],
                linewidth = 0, legend = false, xlims = (lo, hi), ylims = (0, ymax*1.08))
            plot!(p; title = s == 1 ? εlabels[ei] : "", titlefontsize = 7,
                ylabel = ei == 1 ? PG_METHODS[s] : "", yguidefontsize = 8,
                xlabel = s == 4 ? PG_XLABEL : "", xguidefontsize = 8, tickfontsize = 5,
                xformatter = s == 4 ? :auto : (_ -> ""),
                yformatter = ei == 1 ? :auto : (_ -> ""))
            push!(panels, p)
        end
        fig = plot(panels...; layout = (4, 3), size = (760, 680),
            plot_title = name, plot_titlefontsize = 11,
            left_margin = 3Plots.mm, bottom_margin = 3Plots.mm)
        out = joinpath(OUT, "pergroup_$tag.pdf")
        savefig(fig, out); println("wrote ", out)
    end
end

# -----------------------------------------------------------------------------
# Lorentz accuracy findings (printed): Lie vs. direct median L2 error per cell.
# -----------------------------------------------------------------------------
timelike_vectors(n) = (V = zeros(4, n);
    for i in 1:n; s = 0.3*randn(3); V[2:4, i] = s; V[1, i] = sqrt(1 + sum(abs2, s)); end; V)

function lorentz_findings(; trials = lorentz_trials())
    G = Lorentz(); εs = [0.0, 0.01, 0.02, 0.03]; ns = [5, 10, 25, 50]
    println("\n=== Lorentz accuracy: median L2 recovery error (Lie | direct) ===")
    @printf("  %-6s", "ε\\n"); for n in ns; @printf(" %12d", n); end; println()
    ratios = Float64[]
    for ε in εs
        @printf("  %-6.2f", ε)
        for n in ns
            el = Float64[]; ed = Float64[]
            for _ in 1:trials
                Λ = random_group(G; scale = 0.3); X = timelike_vectors(n)
                Y = Λ*X .+ ε.*randn(4, n)
                push!(el, first(align_error(naive_align(G, X, Y), Λ)))
                push!(ed, first(align_error(align_direct(G, X, Y), Λ)))
            end
            ml, md = median(el), median(ed)
            ε > 0 && push!(ratios, ml/md)
            @printf(" %5.0e|%5.0e", ml, md)
        end
        println()
    end
    @printf("  under noise, Lie/direct error ratio spans %.1f-%.1f across cells\n",
            minimum(ratios), maximum(ratios))
end

# -----------------------------------------------------------------------------
function main()
    t0 = time()
    println("Reproducing all figures and findings into ", OUT)
    so4_table()
    figure1_accuracy()
    figure2_timing()
    figure3_se()
    figures_pergroup()
    lorentz_findings()
    @printf("\nDone in %.0f s.  All figures are in %s\n", time() - t0, OUT)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
