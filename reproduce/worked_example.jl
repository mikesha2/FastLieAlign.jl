# The worked example of Section 2.4 of the manuscript: a rotation by π about (1,1,1)/√3,
# four Gaussian points, Gaussian noise of standard deviation 0.05, data rounded to two
# decimals (seed 1 of MersenneTwister). Prints X, Y, g0 = Y X⁺, its eigenvalues, the
# reflection r = I - 2 Q_N Q_Nᵀ across the invariant subspace N of the negative eigenvalues,
# r⁻¹ g0, the projected logarithm, the output of Algorithm 2 and its error. Standard library
# only, so it can be run as  julia reproduce/worked_example.jl  from the package root.

using LinearAlgebra, Random, Printf
u = [1.0, 1.0, 1.0] / sqrt(3)
gs = 2 * u * u' - I                          # rotation by pi about u
fmt(A) = join([join([@sprintf("%7.3f", x) for x in A[i, :]], " ") for i in 1:size(A, 1)], "\n")
function run(seed; show=false)
    rng = MersenneTwister(seed)
    X = round.(randn(rng, 3, 4); digits=2)
    Y = round.(gs * X + 0.05 * randn(rng, 3, 4); digits=2)
    g0 = Y * pinv(X)
    ev = eigvals(g0)
    negreal = [l for l in ev if abs(imag(l)) < 1e-12 && real(l) < 0]
    length(negreal) == 2 || return false
    F = schur(g0); mask = [real(l) <= 0 for l in F.values]; k = count(mask)
    ordschur!(F, mask); Q = F.Z[:, 1:k]
    P = Q * Q'; r = I - 2P
    h = r \ g0
    L = real(log(h)); A = (L - L') / 2; g = r * exp(A)
    Lc = log(complex(g0)); gn = exp(real((Lc - Lc') / 2))
    if show
        println("X=\n", fmt(X)); println("Y=\n", fmt(Y)); println("g0=\n", fmt(g0)); println("eig g0: ", round.(sort(ev; by=real); digits=4))
        println("Q_N=\n", fmt(Q)); println("r=\n", fmt(r)); println("r^-1 g0=\n", fmt(h)); println("eig r^-1 g0: ", round.(eigvals(h); digits=4))
        println("log=\n", fmt(L)); println("proj=\n", fmt(A)); println("output g=\n", fmt(g)); println("g*=\n", fmt(Matrix(gs)))
        @printf("|g-g*|=%.4f |g0-g*|=%.4f naive |gn-g*|=%.4f  |gX-Y|=%.4f |g*X-Y|=%.4f |g0X-Y|=%.4f  r'r-I=%.1e  g'g-I=%.1e det g=%.6f\n",
            norm(g - gs), norm(g0 - gs), norm(gn - gs), norm(g * X - Y), norm(gs * X - Y), norm(g0 * X - Y), norm(r'r - I), norm(g'g - I), det(g))
        println("naive gn=\n", fmt(gn)); println("Lc real=\n", fmt(real(Lc))); println("Lc imag=\n", fmt(imag(Lc)))
        println("rotation angle of g: ", acos((tr(g) - 1) / 2), "  axis: ", round.(normalize([g[3,2]-g[2,3], g[1,3]-g[3,1], g[2,1]-g[1,2]]); digits=3))
    end
    return true
end
run(1; show=true)
