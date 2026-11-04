# extra_experiments.jl: the checks quoted in the text of the manuscript which are not part of the
# figures: Kabsch and cold-started Gauss-Newton on SO(4), cold starts on noncompact groups with a
# raised step cap, the unrefined homogeneous SE(3) estimate against the centroid route, and the
# Lorentz error against the size of the boost (Table 6(b)). Run from the archive root:
#     julia --project=. reproduce/extra_experiments.jl
# This is the script as it was run for the manuscript; timings are single wall clock measurements.
using LinearAlgebra, Random, Statistics, Printf
include(joinpath(@__DIR__, "..", "src", "FastLieAlign.jl")); using .FastLieAlign
FLA = FastLieAlign
kabsch(X, Y) = (F = svd(Y * X'); d = sign(det(F.U * F.Vt)); F.U * Diagonal([ones(size(X,1)-1); d]) * F.Vt)
# --- (i),(ii) SO(4): Kabsch and GN cold start, same setup as Table 6(a)
G = Orthogonal(4); m = 4; ε = 0.05
function gn_from(G, X, Y, g; iters = 12, tol = 1e-13)
    basis = FLA.algebra_basis(G); d = length(basis); n = 0
    for it in 1:iters
        R = Y - g * X; D = hcat([vec(b * g * X) for b in basis]...)
        c = D \ vec(R); δ = sum(c[k] * basis[k] for k in 1:d); n += 1
        norm(δ) <= tol * max(1, norm(g)) && break
        g = exp(δ) * g
    end
    g, n
end
Random.seed!(0)
let X = randn(m, 3m), Λ = FLA.random_group(G; scale = 0.3), Y = Λ*X .+ ε*randn(m, 3m)
    kabsch(X, Y); naive_align(G, X, Y); gauss_newton_refine(G, X, Y); gn_from(G, X, Y, Matrix(1.0I, 4, 4))
end
tk = Float64[]; ek = Float64[]; tc = Float64[]; ec = Float64[]; nc = Int[]; tw = Float64[]; nw = Int[]; tl = Float64[]
Random.seed!(0)
for _ in 1:1000
    X = randn(m, 3m); Λ = FLA.random_group(G; scale = 0.3); Y = Λ*X .+ ε*randn(m, 3m)
    t0 = time_ns(); gk = kabsch(X, Y); push!(tk, (time_ns()-t0)/1e3); push!(ek, norm(gk - Λ))
    t0 = time_ns(); gl = naive_align(G, X, Y); push!(tl, (time_ns()-t0)/1e3)
    t0 = time_ns(); gw, n1 = gn_from(G, X, Y, gl); push!(tw, (time_ns()-t0)/1e3); push!(nw, n1)
    t0 = time_ns(); gc, n2 = gn_from(G, X, Y, Matrix(1.0I, 4, 4)); push!(tc, (time_ns()-t0)/1e3); push!(ec, norm(gc - Λ)); push!(nc, n2)
end
@printf("SO(4): Kabsch time %.1f us, err %.3e | LA time %.1f | GN warm (excluding LA) time %.1f, steps median %d max %d | GN cold from I: time %.1f us, err %.3e, steps median %d max %d\n",
        median(tk), median(ek), median(tl), median(tw), median(nw), maximum(nw), median(tc), median(ec), median(nc), maximum(nc))
# --- cold start GN on noncompact groups with larger scale
for (name, G2, s) in [("SO(3,1)", Lorentz(), 0.3), ("SO(3,1)", Lorentz(), 1.0), ("Sp(4)", Symplectic(2), 0.3), ("Sp(4)", Symplectic(2), 1.0)]
    n = FLA.matsize(G2); nw2 = Int[]; nc2 = Int[]; ew = Float64[]; ec2 = Float64[]; el = Float64[]
    Random.seed!(1)
    for _ in 1:300
        X = randn(n, 3n); Λ = FLA.random_group(G2; scale = s); Y = Λ*X .+ 0.01*randn(n, 3n)
        gl = naive_align(G2, X, Y); gw, a = gn_from(G2, X, Y, gl); gc, b = gn_from(G2, X, Y, Matrix(1.0I, n, n); iters = 50)
        push!(nw2, a); push!(nc2, b); push!(ew, norm(gw - Λ)); push!(ec2, norm(gc - Λ)); push!(el, norm(gl - Λ))
    end
    @printf("%s s=%.1f eps=0.01: LA err %.3e | GN warm steps med %d max %d err %.3e | GN cold steps med %d max %d err %.3e, fraction cold err>2x warm: %.3f\n",
            name, s, median(el), median(nw2), maximum(nw2), median(ew), median(nc2), maximum(nc2), median(ec2), mean(ec2 .> 2 .* ew))
end
# --- (iii) SE(3): unrefined homogeneous LA vs centroid+Kabsch (exact LS), 1000 motions, s=0.6, 15 points
Gse = SpecialEuclidean(0, 3)
for ε in (0.01, 0.05)
    Random.seed!(2); e_la = Float64[]; e_kab = Float64[]; e_gn = Float64[]
    for _ in 1:1000
        R = FLA.random_group(Orthogonal(3); scale = 0.6); t = randn(3); P = randn(3, 15)
        Q = R * P .+ t .+ ε * randn(3, 15)
        X = [P; ones(1, 15)]; Y = [Q; ones(1, 15)]
        gl = naive_align(Gse, X, Y); gg = gauss_newton_refine(Gse, X, Y)
        pc = mean(P, dims = 2); qc = mean(Q, dims = 2); Rk = kabsch(P .- pc, Q .- qc); tk_ = vec(qc - Rk * pc)
        gk = [Rk tk_; 0 0 0 1]; gt = [R t; 0 0 0 1]
        push!(e_la, norm(gl - gt)); push!(e_kab, norm(gk - gt)); push!(e_gn, norm(gg - gt))
    end
    @printf("SE(3) eps=%.2f: median err LA (unrefined) %.3e, centroid+Kabsch %.3e, LA+GN %.3e; ratio LA/Kabsch %.3f\n", ε, median(e_la), median(e_kab), median(e_gn), median(e_la ./ e_kab))
end
# --- (iv) Lorentz: error against size of the boost
for s in (0.3, 1.0, 2.0, 3.0)
    Random.seed!(3); el = Float64[]; eg = Float64[]; bound = Float64[]; fails = 0
    for _ in 1:300
        A = 0.0; Λ = FLA.random_group(Lorentz(); scale = s); X = randn(4, 12); E = 0.01*randn(4, 12); Y = Λ*X .+ E
        gl = try naive_align(Lorentz(), X, Y) catch; fails += 1; continue end
        gg = gauss_newton_refine(Lorentz(), X, Y)
        push!(el, norm(gl - Λ)); push!(eg, norm(gg - Λ))
        push!(bound, exp(opnorm(log(Λ))) * norm(E) / svdvals(X)[end])
    end
    @printf("SO(3,1) s=%.1f eps=0.01: median LA err %.3e, GN err %.3e, ratio %.2f, median first-order bound without kappa %.3e, failures %d\n", s, median(el), median(eg), median(el ./ eg), median(bound), fails)
end
