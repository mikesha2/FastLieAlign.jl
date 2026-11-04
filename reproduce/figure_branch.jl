# Figure 1(c): median error of the naive Lie algebra method (projection of the complex principal
# logarithm, i.e. `naive_align` with its DomainError guard removed) and of Algorithm 2
# (`branch_cut_align`) in SO(3) as the rotation angle approaches π, where the two
# eigenvalues near −1 of g0 = Y X⁺ split under noise either into a complex pair (a real
# principal logarithm exists) or into two negative reals (no real logarithm exists).
# Figure 1(a) and (b), the schematics of the Riemann surface of the logarithm, are drawn in
# the manuscript source with pgfplots.
# Run from the package root:  julia --project=. reproduce/figure_branch.jl
using LinearAlgebra, Statistics, Random, Printf
using Plots, LaTeXStrings
gr()

const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "src", "FastLieAlign.jl"))
using .FastLieAlign

const OUT = joinpath(@__DIR__, "figures")
isdir(OUT) || mkpath(OUT)

skew(u) = [0 -u[3] u[2]; u[3] 0 -u[1]; -u[2] u[1] 0]
rotation(θ, u) = exp(θ * skew(u / norm(u)))
const NOISE = 0.01
const M = 12

function basic_unguarded(G, X, Y)
    g0 = Y * pinv(X)
    exp(algebra_projection(G, log(complex(g0))))
end

function sample(rng, θ)
    u = randn(rng, 3); g = rotation(θ, u)
    X = randn(rng, 3, M); Y = g * X + NOISE * randn(rng, 3, M)
    g, X, Y
end

G = Orthogonal(3)
rng = MersenneTwister(7)
θs = range(2.6, π; length = 15)
nC = 200
medB = Float64[]; loB = Float64[]; hiB = Float64[]
medG = Float64[]; loG = Float64[]; hiG = Float64[]
nreal = 0
for θ in θs
    eB = Float64[]; eG = Float64[]
    for _ in 1:nC
        g, X, Y = sample(rng, θ)
        if θ == θs[end]
            λ = eigvals(Y * pinv(X))
            global nreal += any(l -> abs(imag(l)) < 1e-12 && real(l) < 0, λ)
        end
        push!(eB, norm(basic_unguarded(G, X, Y) - g))
        push!(eG, norm(branch_cut_align(G, X, Y) - g))
    end
    push!(medB, median(eB)); push!(loB, quantile(eB, 0.1)); push!(hiB, quantile(eB, 0.9))
    push!(medG, median(eG)); push!(loG, quantile(eG, 0.1)); push!(hiG, quantile(eG, 0.9))
end
println(@sprintf("at θ = π: naive median error %.3g, Algorithm 2 median error %.3g, fraction of g0 with a negative real eigenvalue %.2f",
    medB[end], medG[end], nreal / nC))
fig = plot(; yscale = :log10, xlabel = "rotation angle " * L"\theta_\ast",
    ylabel = L"\|g-g_\ast\|_F", title = "(c) " * L"SO(3)" * ", 12 vectors, noise 0.01", titlefontsize = 10,
    legend = :topleft, legendfontsize = 8, tickfontsize = 8, guidefontsize = 9,
    xticks = ([2.6, 2.8, 3.0, π], ["2.6", "2.8", "3.0", L"\pi"]), size = (520, 400),
    left_margin = 4Plots.mm, bottom_margin = 4Plots.mm)
plot!(fig, θs, medB; ribbon = (medB .- loB, hiB .- medB), fillalpha = 0.2, color = :crimson,
    lw = 2, label = "naive Lie algebra method (Algorithm 1), projected complex logarithm")
plot!(fig, θs, medG; ribbon = (medG .- loG, hiG .- medG), fillalpha = 0.2, color = :royalblue,
    lw = 2, label = "branch-cut aware method (Algorithm 2)")
out = joinpath(OUT, "branch_error.pdf")
savefig(fig, out); println("wrote ", out)
