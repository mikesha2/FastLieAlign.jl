# Benchmark of the branch-cut aware method (Algorithm 2, `branch_cut_align`) on true elements
# which the naive Lie algebra method (Algorithm 1, `naive_align`) cannot recover: elements of the
# other connected components, elements of the identity component which are not exponentials,
# rotations by π, elements of SU(n) whose principal logarithm has nonzero trace, and elements
# of Spin(n) whose principal logarithm in the spinor representation is not a bivector.
# For each case: m = 3n Gaussian input vectors, Gaussian noise of standard deviation 0.01 in
# every component of Y, TRIALS trials. The naive column uses `naive_align` where it is defined and
# otherwise the real projection of the complex principal logarithm (what `naive_align` would return
# without its DomainError guard); the "undefined" column is the fraction of trials in which
# `naive_align` throws. Times are medians of single wall clock measurements; the naive time is
# that of the unguarded computation, so it excludes the cost of the thrown exception.
# Run from the package root:  julia --project=. reproduce/whole_group_benchmark.jl
using LinearAlgebra, Statistics, Random, Printf
const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "src", "FastLieAlign.jl"))
using .FastLieAlign

const TRIALS = parse(Int, get(ENV, "WG_TRIALS", "200"))
const NOISE = 0.01

skew(u) = [0 -u[3] u[2]; u[3] 0 -u[1]; -u[2] u[1] 0]
rotation(θ, u) = exp(θ * skew(u / norm(u)))
isrealgroup(G) = FastLieAlign.eltype_of(G) <: Real

# The naive method without its guard: the real projection of the complex principal logarithm.
# This is what `naive_align` returns where it is defined and what the naive column reports
# otherwise, and it is what the naive time measures (the guard's exception is not timed).
function naive_unguarded(G, X, Y)
    g0 = Y * pinv(X)
    L0 = log(Matrix{ComplexF64}(g0))
    return exp(algebra_projection(G, isrealgroup(G) ? real(L0) : L0))
end

function naive_fallback(G, X, Y)
    try
        return naive_align(G, X, Y), false
    catch e
        (e isa DomainError || e isa ArgumentError || e isa LinearAlgebra.SingularException) || rethrow()
        return naive_unguarded(G, X, Y), true
    end
end

N2 = [0.0 1.0; 0.0 0.0]
ω = exp(2π * im / 3)
T4 = Diagonal([-1.0, 1, 1, 1]) |> Matrix     # time reversal, η = diag(-1, 1, 1, 1)
P4 = Diagonal([1.0, -1, -1, -1]) |> Matrix   # parity
boost(rng) = random_group(IndefiniteOrthogonal(1, 3); scale = 0.3, rng = rng)
planes3(n, θ) = (A = zeros(n, n); for j in 1:3; A[2j-1, 2j] = θ; A[2j, 2j-1] = -θ; end; A)

# (label, group, generator of the true element)
cases = [
    ("SO(3), rotation by π",            Orthogonal(3),                       rng -> rotation(π, randn(rng, 3))),
    ("O(3), reflection times rotation", Orthogonal(3; special = false),      rng -> (u = normalize(randn(rng, 3)); (I - 2u * u') * random_group(Orthogonal(3); scale = 0.3, rng = rng))),
    ("O(3,1), time reversal times boost", IndefiniteOrthogonal(1, 3; special = false), rng -> T4 * boost(rng)),
    ("O(3,1), parity times boost",      IndefiniteOrthogonal(1, 3; special = false), rng -> P4 * boost(rng)),
    ("SO(3,1), PT times boost",         IndefiniteOrthogonal(1, 3),          rng -> P4 * T4 * boost(rng)),
    ("SL(2,R), -(I+N) times exp",       SpecialLinear(2),                    rng -> -(I + N2) * random_group(SpecialLinear(2); scale = 0.3, rng = rng)),
    ("GL(2,R), diag(-1,-2) times exp",  GeneralLinear(2),                    rng -> Diagonal([-1.0, -2.0]) * random_group(GeneralLinear(2); scale = 0.3, rng = rng)),
    ("Sp(4,R), -(I+N) on one pair",     Symplectic(2),                       rng -> (g = Matrix(1.0I, 4, 4); g[1,1] = -1; g[3,3] = -1; g[1,3] = -1; g * random_group(Symplectic(2); scale = 0.3, rng = rng))),
    ("SE(3), rotation by π with translation", SpecialEuclidean(0, 3),        rng -> (g = Matrix(1.0I, 4, 4); g[1:3,1:3] = rotation(π, randn(rng, 3)); g[1:3,4] = randn(rng, 3); g)),
    ("Sim(3), scaled rotation by π with translation", Similarity(3),         rng -> (g = Matrix(1.0I, 4, 4); g[1:3,1:3] = exp(0.5 * randn(rng)) * rotation(π, randn(rng, 3)); g[1:3,4] = randn(rng, 3); g)),
    ("SU(3), ω times exp",              Unitary(3; special = true),          rng -> ω * random_group(Unitary(3; special = true); scale = 0.3, rng = rng)),
    ("SL(3,C), ω times exp",            SpecialLinear(3; field = :complex),  rng -> ω * random_group(SpecialLinear(3; field = :complex); scale = 0.3, rng = rng)),
    ("SU(10), ω₁₀³ times exp",           Unitary(10; special = true),         rng -> exp(2π * im * 3 / 10) * random_group(Unitary(10; special = true); scale = 0.3, rng = rng)),
    ("SL(10,C), ω₁₀³ times exp",         SpecialLinear(10; field = :complex), rng -> exp(2π * im * 3 / 10) * random_group(SpecialLinear(10; field = :complex); scale = 0.3, rng = rng)),
    ("Spin(6), 0.8π in three planes times exp", Spin(6),                     rng -> exp(spin_lift(Spin(6), planes3(6, 0.8π))) * random_group(Spin(6); scale = 0.3, rng = rng)),
    ("Spin(7), 0.8π in three planes times exp", Spin(7),                     rng -> exp(spin_lift(Spin(7), planes3(7, 0.8π))) * random_group(Spin(7); scale = 0.3, rng = rng)),
    ("Pin(3), reflection times exp",    Pin(3),                              rng -> (ε = FastLieAlign.clifford_generators(Pin(3)); u = normalize(randn(rng, 3)); sum(u[a] .* ε[a] for a in 1:3) * random_group(Pin(3); scale = 0.3, rng = rng))),
    ("Pin(4), reflection times exp",    Pin(4),                              rng -> (ε = FastLieAlign.clifford_generators(Pin(4)); u = normalize(randn(rng, 4)); sum(u[a] .* ε[a] for a in 1:4) * random_group(Pin(4); scale = 0.3, rng = rng))),
    ("SO(3,C), rotation by π",          Orthogonal(3; field = :complex),     rng -> conj_by(Orthogonal(3; field = :complex), ComplexF64.(Diagonal([-1, -1, 1])), rng)),
    ("O(3,C), reflection times exp",    Orthogonal(3; special = false, field = :complex), rng -> ComplexF64.(Diagonal([-1, 1, 1])) * random_group(Orthogonal(3; field = :complex); scale = 0.3, rng = rng)),
]
conj_by(G, g, rng) = (q = random_group(G; scale = 0.4, rng = rng); q * g * inv(q))

function timeit(f)
    f(); t = @elapsed f(); t
end

println(@sprintf("%-40s %9s %9s %9s %9s %8s %8s", "case", "naive", "undef", "Alg. 2", "Alg.2+GN", "t naive", "t Alg.2"))
rows = String[]
for (label, G, gen) in cases
    rng = MersenneTwister(11)
    n = matsize(G); cplx = !isrealgroup(G)
    eN = Float64[]; eG = Float64[]; eC = Float64[]; tN = Float64[]; tG = Float64[]; undef = 0
    for _ in 1:TRIALS
        g = gen(rng)
        X = cplx ? randn(rng, ComplexF64, n, 3n) : randn(rng, n, 3n)
        Y = g * X + NOISE * (cplx ? randn(rng, ComplexF64, n, 3n) : randn(rng, n, 3n))
        gN, failed = naive_fallback(G, X, Y); undef += failed
        gG = branch_cut_align(G, X, Y)
        gC = gauss_newton_refine(G, X, Y; init = gG)
        push!(eN, norm(gN - g)); push!(eG, norm(gG - g)); push!(eC, norm(gC - g))
        push!(tN, timeit(() -> naive_unguarded(G, X, Y))); push!(tG, timeit(() -> branch_cut_align(G, X, Y)))
    end
    row = @sprintf("%-40s %9.3g %9.2f %9.3g %9.3g %8.0f %8.0f", label, median(eN), undef / TRIALS, median(eG), median(eC), 1e6 * median(tN), 1e6 * median(tG))
    println(row); push!(rows, row)
end
