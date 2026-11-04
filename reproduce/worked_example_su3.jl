# The SU(3) worked example of Section 2.4 of the manuscript: g* = ωI with ω = exp(2πi/3),
# four complex Gaussian vectors, complex Gaussian noise of standard deviation 0.05, data
# rounded to two decimals (seed 1 of MersenneTwister). Prints X, Y, g0 = Y X⁺, the
# eigenvalue arguments and the trace count k, the reflection r = I - 2 Q_S Q_S† of the two
# eigenvalues of largest argument, r⁻¹ g0, its projected logarithm, the output of Algorithm 2
# and its error, and the naive output. Standard library only:
#   julia reproduce/worked_example_su3.jl

using LinearAlgebra, Random, Printf
ω = exp(2π*im/3); gs = ω * Matrix{ComplexF64}(I, 3, 3)
cfmt(z) = @sprintf("%5.2f%+5.2fi", real(z), imag(z))
fmt(A) = join([join([cfmt(x) for x in A[i, :]], "  ") for i in 1:size(A, 1)], "\n")
rnd(A) = round.(real(A); digits=2) .+ im .* round.(imag(A); digits=2)
function run(seed; show=false, m=3)
    rng = MersenneTwister(seed)
    X = rnd(randn(rng, ComplexF64, 3, m)); Y = rnd(gs * X + 0.05 * randn(rng, ComplexF64, 3, m))
    g0 = Y * pinv(X)
    λ = eigvals(g0); θ = angle.(λ); k = round(Int, sum(θ) / (2π))
    all(θ .> 0) && k == 1 || return false
    F = schur(g0); ord = sortperm(abs.(angle.(F.values)); rev=true)      # largest |θ| first
    sel = falses(3); sel[ord[1:2]] .= true                               # two with θ>0 (all are)
    Fo = ordschur(F, sel); Q = Fo.Z[:, 1:2]
    r = I - 2Q*Q'; h = r \ g0; L = log(h); S = (L - L') / 2; A = S - (tr(S)/3) * I; g = r * exp(A)
    L0 = log(g0); S0 = (L0 - L0')/2; gn = exp(S0 - (tr(S0)/3) * I)
    if show
        println("X=\n", fmt(X)); println("Y=\n", fmt(Y)); println("g0=\n", fmt(g0))
        println("eig g0: ", join(cfmt.(λ), ", "), "   args/π: ", round.(θ ./ π; digits=3), "  sum/2π=", round(sum(θ)/(2π); digits=3), " k=", k)
        println("r=\n", fmt(r)); println("eig r: ", round.(real(eigvals(r)); digits=3), "  det r=", round(det(r); digits=6), " |r'r-I|=", norm(r'r-I))
        println("r^-1 g0=\n", fmt(h)); println("eig r^-1 g0 args/π: ", round.(angle.(eigvals(h)) ./ π; digits=3), " sum/π=", round(sum(angle.(eigvals(h)))/π; digits=3))
        println("log(r^-1 g0)=\n", fmt(L)); println("tr log=", cfmt(tr(L)), "  |L+L'|=", round(norm(L+L'); digits=3))
        println("proj A=\n", fmt(A)); println("g=\n", fmt(g))
        @printf("|g-ωI|=%.4f |g0-ωI|=%.4f naive |gn-ωI|=%.4f |g'g-I|=%.1e det g=%s |gX-Y|=%.4f |ωX-Y|=%.4f\n", norm(g-gs), norm(g0-gs), norm(gn-gs), norm(g'g-I), cfmt(det(g)), norm(g*X-Y), norm(gs*X-Y))
        println("naive gn=\n", fmt(gn))
    end
    true
end
run(1; show=true, m=4)
