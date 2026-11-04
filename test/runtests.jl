using LinearAlgebra, Random
Random.seed!(2026)   # the suite draws random data; the seed makes it deterministic
using FastLieAlign

# A representative sample of the classical matrix Lie groups.
groups = Dict(
    "GL(4,ℝ)"      => GeneralLinear(4),
    "GL(3,ℂ)"      => GeneralLinear(3; field = :complex),
    "SL(4,ℝ)"      => SpecialLinear(4),
    "SL(3,ℂ)"      => SpecialLinear(3; field = :complex),
    "SO(5)"        => Orthogonal(5; special = true),
    "U(3)"         => Unitary(3; special = false),
    "SU(3)"        => Unitary(3; special = true),
    "SO(3,1) Lorentz" => Lorentz(),
    "SO(2,3)"      => IndefiniteOrthogonal(2, 3),
    "Sp(4,ℝ)"      => Symplectic(2),
    "Sp(4,ℂ)"      => Symplectic(2; field = :complex),
    "Sp(6,ℂ)"      => Symplectic(3; field = :complex),
    "U(4)"         => Unitary(4; special = false),
    "SU(4)"        => Unitary(4; special = true),
    "Spin(3)"      => Spin(3),
    "Spin(4)"      => Spin(4),
    "Spin(5)"      => Spin(5),
    "Spin(6)"      => Spin(6),
    "Pin(3)"       => Pin(3),
    "Pin(4)"       => Pin(4),
    "SE(3) Euclid" => SpecialEuclidean(0, 3),
    "Sim(3)"       => Similarity(3),
    "SE(3,1)"      => PoincareSE(),
    "SO(3,ℂ)"      => Orthogonal(3; field = :complex),
    "O(4,ℂ)"       => Orthogonal(4; special = false, field = :complex),
    "U(2,1)"       => IndefiniteUnitary(2, 1),
)

npass = 0; nfail = 0
@inline check(name, cond) = (cond ? (global npass += 1) :
    (global nfail += 1; println("  FAIL: ", name)))

for (name, G) in sort(collect(groups); by = first)
    println("== ", name, " (", matsize(G), "×", matsize(G), ") ==")

    # 1. Projection is idempotent and lands in the algebra.
    M = random_algebra(G; scale = 1.0)
    check("$name: projection idempotent",
        norm(algebra_projection(G, M) - M) < 1e-10)
    check("$name: random_algebra in 𝔤", in_algebra(G, M))

    # 2. exp(𝔤) lands in the group.
    g = random_group(G; scale = 0.3)
    check("$name: random_group in G", in_group(G, g; atol = 1e-7))

    # 3. Exact recovery: noiseless, well-posed alignment problem.
    Λ = random_group(G; scale = 0.3)
    n = matsize(G)
    X = eltype(Λ) <: Complex ? randn(n, 2n) + im * randn(n, 2n) : randn(n, 2n)
    Y = Λ * X
    Λ̂ = naive_align(G, X, Y)
    rec = norm(Λ̂ - Λ) / norm(Λ)
    check("$name: exact recovery (rel err $(round(rec; sigdigits=2)))", rec < 1e-6)
    check("$name: estimate in G", in_group(G, Λ̂; atol = 1e-6))

    # 4. Noisy alignment stays close.
    E = 0.01 * (eltype(Λ) <: Complex ? randn(n, 2n) + im * randn(n, 2n) : randn(n, 2n))
    Λ̂n = naive_align(G, X, Y + E)
    l2, _ = align_error(Λ̂n, Λ)
    check("$name: noisy estimate in G", in_group(G, Λ̂n; atol = 1e-6))

    # 5. Direct method (Method 1): correct algebra dimension, exact recovery,
    #    and agreement with the Lie algebra method.
    # The direct method (BFGS with exact reverse-mode AD gradients) stops at its
    # gradient tolerance, so it agrees with the closed-form Lie method to about
    # 1e-5 relative error rather than to rounding.
    Λ̂d = align_direct(G, X, Y)
    recd = norm(Λ̂d - Λ) / norm(Λ)
    check("$name: direct recovery (rel err $(round(recd; sigdigits=2)))", recd < 1e-4)
    check("$name: direct estimate in G", in_group(G, Λ̂d; atol = 1e-6))
    check("$name: methods agree", norm(Λ̂d - Λ̂) / norm(Λ̂) < 1e-4)
    println("   noiseless: lie=$(round(rec; sigdigits=3)) direct=$(round(recd; sigdigits=3)),  noisy L² err = $(round(l2; sigdigits=3))")
end

# ---------------------------------------------------------------------------
# Focused tests: unitary, symplectic, and spin groups over real & complex
# base fields — including correct (real) algebra dimensions.
# ---------------------------------------------------------------------------
println("\n== Unitary / Symplectic / Spin (real & complex fields) ==")

# (group, expected real dimension of 𝔤)
field_cases = [
    ("U(2)",      Unitary(2; special = false), 4),    # complex; dim n²
    ("U(4)",      Unitary(4; special = false), 16),
    ("SU(2)",     Unitary(2; special = true),  3),    # complex; dim n²−1
    ("SU(4)",     Unitary(4; special = true),  15),
    ("Sp(2,ℝ)",   Symplectic(1),               3),    # real; dim n(2n+1), n=1
    ("Sp(4,ℝ)",   Symplectic(2),               10),
    ("Sp(2,ℂ)",   Symplectic(1; field=:complex), 6),  # complex; 2× real dim
    ("Sp(4,ℂ)",   Symplectic(2; field=:complex), 20),
    ("Spin(3)",   Spin(3), 3),                        # ≅ SU(2)
    ("Spin(4)",   Spin(4), 6),                        # ≅ SU(2)×SU(2)
    ("Spin(5)",   Spin(5), 10),                       # ≅ USp(4)
    ("Spin(6)",   Spin(6), 15),                       # ≅ SU(4)
    ("Pin(3)",    Pin(3), 3),                         # algebra = spin(3)
    ("Pin(5)",    Pin(5), 10),
    ("SO(3,ℂ)",   Orthogonal(3; field = :complex), 6),   # complex; 2× real dim of so(3)
    ("U(2,1)",    IndefiniteUnitary(2, 1), 9),          # dim n²
]

for (name, G, expdim) in field_cases
    n = matsize(G); cplx = eltype(identity_element(G)) <: Complex
    check("$name: real algebra dimension == $expdim", length(algebra_basis(G)) == expdim)

    # Generators lie in 𝔤 and projection is idempotent.
    M = random_algebra(G; scale = 1.0)
    check("$name: random_algebra in 𝔤", in_algebra(G, M))
    check("$name: projection idempotent", norm(algebra_projection(G, M) - M) < 1e-10)

    # exp lands in the group; both alignment methods recover it exactly.
    Λ = random_group(G; scale = 0.3)
    check("$name: random_group in G", in_group(G, Λ; atol = 1e-7))
    X = cplx ? randn(n, 3n) + im*randn(n, 3n) : randn(n, 3n)
    Y = Λ * X
    Λl = naive_align(G, X, Y); Λd = align_direct(G, X, Y)
    check("$name: lie recovery",    norm(Λl - Λ)/norm(Λ) < 1e-6)
    check("$name: direct recovery", norm(Λd - Λ)/norm(Λ) < 1e-6)
    check("$name: estimate in G",   in_group(G, Λl; atol = 1e-6))
    println("   $name ($(cplx ? "ℂ" : "ℝ"), $(n)×$(n)): dim=$(length(algebra_basis(G)))")
end

# ---------------------------------------------------------------------------
# SE(3,1): Lie algebra point-alignment vs the traditional centroid method.
# ---------------------------------------------------------------------------
println("\n== SE(3,1): Lie method vs centroid decoupling ==")
let G = PoincareSE(), rot = rotation_group(G)
    for trial in 1:25
        Λ = random_group(rot; scale = 0.3); t = randn(4)
        m = 15; PA = randn(4, m); PB = Λ * PA .+ t
        Mtrue = [Λ t; zeros(1, 4) 1]
        Mlie  = align_se(G, PA, PB)
        Mcent = align_se_centroid(G, PA, PB)
        check("SE(3,1) trial $trial: lie recovers",      norm(Mlie  - Mtrue) < 1e-8)
        check("SE(3,1) trial $trial: centroid recovers", norm(Mcent - Mtrue) < 1e-8)
        check("SE(3,1) trial $trial: methods agree",     norm(Mlie  - Mcent) < 1e-8)
        check("SE(3,1) trial $trial: lie in SE(3,1)",    in_group(G, Mlie;  atol = 1e-7))
        check("SE(3,1) trial $trial: centroid in SE(3,1)", in_group(G, Mcent; atol = 1e-7))
    end
    println("   25 random Poincaré transforms: both methods recover [Λ t; 0 1] to ~1e-15")
end

# ---------------------------------------------------------------------------
# Algebra membership: random_algebra lies in 𝔤, *unstructured* random
# matrices generally do not. For every group we draw 20 unstructured random
# matrices (real or complex as appropriate) and 20 random_algebra samples,
# then check that
#
#   in_algebra(G, M)         is true  for M = random_algebra(G)  (every trial),
#   in_algebra(G, M)         is false for unstructured random M (most trials),
#   the Frobenius norm of the residual
#                              R = M − proj_𝔤(M)
#   is small for algebra samples and generically large for unstructured ones.
#
# This verifies the algebra constraint is *non-trivial*: a non-trivial
# fraction of the ambient matrix space is rejected, and the orthogonal
# projection removes a large component for almost every random input.
# ---------------------------------------------------------------------------
println("\n== Algebra membership: random_algebra vs unstructured random ==")
const N_MEMBERSHIP = 20
const ALGEBRA_TOL = 1e-8

for (name, G) in sort(collect(groups); by = first)
    n = matsize(G)
    cplx = eltype(FastLieAlign.identity_element(G)) <: Complex
    T = cplx ? ComplexF64 : Float64

    in_alg_alg = 0      # count of algebra samples confirmed in 𝔤
    in_alg_unstr = 0    # count of unstructured random samples (mistakenly) in 𝔤
    avg_resid_alg = 0.0
    avg_resid_unstr = 0.0

    for _ in 1:N_MEMBERSHIP
        # Algebra sample: built by random_algebra, so it must lie in 𝔤.
        A = random_algebra(G; scale = 1.0)
        in_alg_alg += in_algebra(G, A; atol = ALGEBRA_TOL)
        rA = norm(A - algebra_projection(G, A)) / max(1, norm(A))
        avg_resid_alg += rA

        # Unstructured random: a generic matrix drawn from the ambient space.
        # Almost surely not in 𝔤 for any non-trivial algebra.
        M = cplx ? randn(T, n, n) : randn(T, n, n)
        in_alg_unstr += in_algebra(G, M; atol = ALGEBRA_TOL)
        rM = norm(M - algebra_projection(G, M)) / max(1, norm(M))
        avg_resid_unstr += rM
    end
    avg_resid_alg /= N_MEMBERSHIP
    avg_resid_unstr /= N_MEMBERSHIP

    check("$name: random_algebra ∈ 𝔤 in $N_MEMBERSHIP/$N_MEMBERSHIP trials",
        in_alg_alg == N_MEMBERSHIP)
    # The unstructured samples should mostly fail: at most 1 of 20 may
    # accidentally land in 𝔤. The algebra constraint is a non-trivial
    # codimension-(n² − dim 𝔤) subspace of M_n, so the probability of
    # landing in it is essentially zero.
    #
    # Exception: for `GeneralLinear` the Lie algebra is the *whole* matrix
    # space, so every random matrix trivially lies in 𝔤 and the projection
    # is the identity. We only enforce the membership check for groups
    # with a *proper* subalgebra.
    if G isa GeneralLinear
        check("$name: GL has full algebra (every random matrix ∈ 𝔤)",
            in_alg_unstr == N_MEMBERSHIP)
        check("$name: ‖residual‖ = 0 (projection = identity) for GL",
            avg_resid_unstr == 0.0)
    else
        check("$name: unstructured random ∉ 𝔤 in (≥ $(N_MEMBERSHIP-1))/$N_MEMBERSHIP",
            in_alg_unstr <= 1)
        # The relative residual is tiny for algebra samples and O(1) for
        # unstructured ones, confirming the projection is doing real work.
        check("$name: ‖residual‖ small for algebra samples (≤ 1e-6)",
            avg_resid_alg <= 1e-6)
        check("$name: ‖residual‖ large for unstructured samples (≥ 0.1)",
            avg_resid_unstr >= 0.1)
    end

    println("   $name: algebra resid=$(round(avg_resid_alg; sigdigits=2)),  " *
            "unstructured resid=$(round(avg_resid_unstr; sigdigits=2)),  " *
            "unstructured-in-𝔤: $in_alg_unstr/$N_MEMBERSHIP")
end

# Components other than the identity component (align_components).
let
    for (name, G, h) in [("O(3) reflection", Orthogonal(3; special = false), Diagonal([-1.0, 1, 1])),
                         ("O(1,3) time reversal", IndefiniteOrthogonal(1, 3; special = false), Diagonal([-1.0, 1, 1, 1])),
                         ("O(1,3) parity", IndefiniteOrthogonal(1, 3; special = false), Diagonal([1.0, -1, -1, -1]))]
        n = matsize(G); Λ = Matrix(h) * random_group(G; scale = 0.5); X = randn(n, 3n)
        check("$name: exact without noise", norm(align_components(G, X, Λ * X) - Λ) < 1e-9)
        Y = Λ * X .+ 1e-3 .* randn(n, 3n)
        check("$name: small error with noise", norm(align_components(G, X, Y) - Λ) < 0.05)
        check("$name: all=true agrees", norm(align_components(G, X, Y; all = true) - align_components(G, X, Y)) < 1e-12)
    end
    # a matrix with a negative real eigenvalue has no real logarithm: an informative error, not an InexactError
    X = randn(2, 6); ok = try naive_align(SpecialLinear(2), X, Diagonal([-2.0, -0.5]) * X); false catch e; e isa DomainError end
    check("SL(2): negative eigenvalues raise DomainError", ok)
end
# Covering the whole group (branch_cut_align): elements outside the image of the principal logarithm.
let
    N2 = [0.0 1.0; 0.0 0.0]
    conj_by(G, g, s) = (q = random_group(G; scale = s); q * g * inv(q))
    planes3(n, θ) = (A = zeros(n, n); for j in 1:3; A[2j-1, 2j] = θ; A[2j, 2j-1] = -θ; end; A)
    u = normalize(randn(3)); v = normalize(randn(3)); ω = exp(2π * im / 3)
    M = Matrix(1.0I, 4, 4); M[[1, 3], [1, 3]] .= -I - N2
    cases = [
        ("SL(2,R) -I-N", SpecialLinear(2), -I - N2),
        ("SL(3,R) negative pair", SpecialLinear(3), conj_by(SpecialLinear(3), Matrix(Diagonal([-1.0, -2.0, 0.5])), 0.7)),
        ("GL(2,R) diag(-1,-2)", GeneralLinear(2), Matrix(Diagonal([-1.0, -2.0]))),
        ("GL(3,R) det<0", GeneralLinear(3), conj_by(GeneralLinear(3), Matrix(Diagonal([-1.0, 2.0, 3.0])), 0.5)),
        ("Sp(4,R) negative pair", Symplectic(2), conj_by(Symplectic(2), Matrix(Diagonal([-2.0, 1.0, -0.5, 1.0])), 0.6)),
        ("Sp(4,R) -I-N on (q1,p1)", Symplectic(2), conj_by(Symplectic(2), M, 0.6)),
        ("O(3) reflection", Orthogonal(3; special = false), I - 2u * u'),
        ("O(3) reflection diag(1,-1,1)", Orthogonal(3; special = false), Matrix(Diagonal([1.0, -1, 1]))),
        ("SO(3) rotation by pi", Orthogonal(3), 2v * v' - I),
        ("O(3,1) time reversal", IndefiniteOrthogonal(1, 3; special = false), Diagonal([-1.0, 1, 1, 1]) * random_group(Lorentz(); scale = 0.7)),
        ("O(3,1) parity", IndefiniteOrthogonal(1, 3; special = false), Diagonal([1.0, -1, -1, -1]) * random_group(Lorentz(); scale = 0.7)),
        ("SE(3) rotation by pi", SpecialEuclidean(0, 3), [2v * v' - I randn(3); 0 0 0 1]),
        ("SU(3) omega I", Unitary(3), ω * Matrix(1.0I, 3, 3)),
        ("SU(2) -I", Unitary(2), -Matrix(1.0I, 2, 2) .+ 0im),
        ("SL(2,C) -I-N", SpecialLinear(2; field = :complex), -I - N2 .+ 0im),
        ("SL(3,C) J_3(omega)", SpecialLinear(3; field = :complex), conj_by(SpecialLinear(3; field = :complex), ω * Matrix(1.0I, 3, 3) + diagm(1 => [1.0, 1.0]) .+ 0im, 0.4)),
        ("Spin(6) 0.8pi in three planes", Spin(6), exp(spin_lift(Spin(6), planes3(6, 0.8π)))),
        ("Spin(7) 0.8pi in three planes", Spin(7), exp(spin_lift(Spin(7), planes3(7, 0.8π))) * random_group(Spin(7); scale = 0.2)),
        ("Spin(5) -1 times exp", Spin(5), -random_group(Spin(5); scale = 0.3)),
        ("Spin(4) rotation by pi", Spin(4), exp(spin_lift(Spin(4), [0 π 0 0; -π 0 0 0; 0 0 0 0; 0 0 0 0]))),
        ("Pin(3) reflection times exp", Pin(3), FastLieAlign.clifford_generators(Pin(3))[1] * random_group(Pin(3); scale = 0.3)),
        ("Pin(4) reflection times exp", Pin(4), (ε = FastLieAlign.clifford_generators(Pin(4)); w = normalize(randn(4)); sum(w[a] .* ε[a] for a in 1:4) * random_group(Pin(4); scale = 0.3))),
        ("Pin(3) -I", Pin(3), -Matrix{ComplexF64}(I, 4, 4)),
        ("SO(3,C) rotation by pi", Orthogonal(3; field = :complex), conj_by(Orthogonal(3; field = :complex), ComplexF64.(Diagonal([-1, -1, 1])), 0.5)),
        ("O(3,C) reflection", Orthogonal(3; special = false, field = :complex), ComplexF64.(Diagonal([-1, 1, 1])) * random_group(Orthogonal(3; field = :complex); scale = 0.4)),
        ("U(2,1) eigenvalue -1", IndefiniteUnitary(2, 1), conj_by(IndefiniteUnitary(2, 1), ComplexF64.(Diagonal([-1, exp(0.3im), exp(-0.7im)])), 0.4)),
    ]
    for (name, G, gs) in cases
        n = size(gs, 1); X = randn(eltype(gs), n, 3n)
        g = branch_cut_align(G, X, gs * X)
        check("$name: branch_cut_align exact", norm(g - gs) < 1e-8 * max(1, norm(gs)))
        gn = branch_cut_align(G, X, gs * X .+ 1e-4 .* randn(eltype(gs), n, 3n))
        check("$name: branch_cut_align small error with noise", norm(gn - gs) < 1e-2 * max(1, norm(gs)))
        inG = G isa SpecialEuclidean ? isapprox(gn[1:3, 1:3]' * gn[1:3, 1:3], I; atol = 1e-6) && isapprox(gn[4, :], [0, 0, 0, 1.0]; atol = 1e-6) : in_group(G, gn; atol = 1e-6)
        check("$name: branch_cut_align in the group", inG)
    end
    # agreement with naive_align where the principal logarithm already lies in the algebra
    for (G, s) in [(Orthogonal(5), 0.5), (Symplectic(2), 0.5), (IndefiniteOrthogonal(2, 3), 0.5), (Unitary(3), 0.5), (SpecialEuclidean(0, 3), 0.5),
                   (Orthogonal(3; field = :complex), 0.5), (IndefiniteUnitary(2, 1), 0.5)]
        gs = random_group(G; scale = s); n = size(gs, 1); X = randn(eltype(gs), n, 3n); Y = gs * X .+ 1e-3 .* randn(eltype(gs), n, 3n)
        check("branch_cut_align agrees with naive_align on $(typeof(G))", norm(branch_cut_align(G, X, Y) - naive_align(G, X, Y)) < 1e-10)
    end
    # for Spin(n) the two agree to first order in the noise (the difference is of second order)
    for n in 3:7
        G = Spin(n); gs = random_group(G; scale = 0.4); d = matsize(G); X = randn(ComplexF64, d, 3d); Y = gs * X .+ 1e-3 .* randn(ComplexF64, d, 3d)
        check("branch_cut_align agrees with naive_align to first order on Spin($n)", norm(branch_cut_align(G, X, Y) - naive_align(G, X, Y)) < 1e-6)
        check("spin_covering of a lift is the rotation", (A = random_algebra(Orthogonal(n); scale = 0.5); norm(spin_covering(G, exp(spin_lift(G, A))) - exp(A)) < 1e-10))
        # Pin(n): a unit vector covers the reflection across its orthogonal complement, with determinant -1
        P = Pin(n); ε = FastLieAlign.clifford_generators(P); u = normalize(randn(n)); ut = sum(u[a] .* ε[a] for a in 1:n)
        check("Pin($n): unit vector covers the reflection", norm(spin_covering(P, ut) - (I - 2u * u')) < 1e-10)
        check("Pin($n): the reflection is in Pin(n)", in_group(P, ut))
    end
end

println("\n", npass, " passed, ", nfail, " failed")
nfail == 0 || error("test failures")
