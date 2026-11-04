# qubits.jl: the eight-qubit circuit example of the Discussion (Section 5.1 of the manuscript).
# The unknown unitary U ∈ U(256) acts on the 256 computational basis states, X = I, and the
# output amplitudes Y = U + E are read with complex Gaussian noise of standard deviation ε per
# entry.  Three circuits: (i) a short Ising evolution exp(-iHt), (ii) a random circuit of depth
# 20, (iii) the GHZ preparation circuit (a Hadamard layer followed by a CNOT ladder), whose
# eigenvalue -1 has multiplicity 128.  Methods: the polar factor of Y (the least squares
# unitary, since X = I), Algorithm 1 and Algorithm 2.  Run from the archive root:
#     julia --project=. reproduce/qubits.jl
using LinearAlgebra, Random, Statistics, Printf
include(joinpath(@__DIR__, "..", "src", "FastLieAlign.jl")); using .FastLieAlign

const nq = 8
const d = 2^nq
const I2 = Matrix{ComplexF64}(I, 2, 2)
const σx = ComplexF64[0 1; 1 0]
const σz = ComplexF64[1 0; 0 -1]
const Hd = ComplexF64[1 1; 1 -1] / sqrt(2)
const CNOT = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
kron_all(ops) = reduce(kron, ops)
op_on(q, A) = kron_all([i == q ? A : I2 for i in 1:nq])
cnot_on(q) = kron_all(vcat([I2 for _ in 1:q-1], [CNOT], [I2 for _ in q+2:nq]))

function haar_su2(rng)
    A = randn(rng, ComplexF64, 2, 2); Q, R = qr(A); Q = Matrix(Q) * Diagonal(sign.(diag(R)))
    Q / sqrt(det(Q))
end
function random_circuit(rng, depth)
    U = Matrix{ComplexF64}(I, d, d)
    for layer in 1:depth
        U = kron_all([haar_su2(rng) for _ in 1:nq]) * U
        for q in (isodd(layer) ? (1:2:nq-1) : (2:2:nq-1))
            U = cnot_on(q) * U
        end
    end
    U
end
function ising_hamiltonian(h)
    H = zeros(ComplexF64, d, d)
    for q in 1:nq-1; H -= op_on(q, σz) * op_on(q+1, σz); end
    for q in 1:nq;   H -= h * op_on(q, σx); end
    Hermitian(H)
end
ghz_circuit() = (U = kron_all([Hd for _ in 1:nq]); for q in 1:nq-1; U = cnot_on(q) * U; end; U)

polar_unitary(Y) = (F = svd(Y); F.U * F.Vt)

const G = Unitary(d; special = false)
const X = Matrix{ComplexF64}(I, d, d)
const t = 0.1
const Hising = ising_hamiltonian(1.0)
rng = MersenneTwister(2026)
circuits = [("exp(-iHt), Ising, t = 0.1", exp(-im * t * Matrix(Hising))),
            ("random circuit, depth 20", random_circuit(rng, 20)),
            ("GHZ circuit (H layer, CNOT ladder)", ghz_circuit())]
for (name, U) in circuits
    λ = eigvals(U)
    @printf("%-36s  eigenvalues within 1e-6 of -1: %3d   min |arg λ - π| otherwise: %.4f\n",
            name, count(abs.(λ .+ 1) .< 1e-6), minimum(abs.(abs.(angle.(λ)) .- π)[abs.(λ .+ 1) .>= 1e-6]))
end
println()
ntrial = 50
@printf("%-36s %6s %10s %10s %10s %10s %10s %8s %8s\n", "circuit", "ε", "polar", "Alg. 1", "Alg. 2", "A2-polar", "H error", "t1 ms", "t2 ms")
for (name, U) in circuits, ε in (1e-3, 1e-2)
    ep = Float64[]; e1 = Float64[]; e2 = Float64[]; e21 = Float64[]; eh = Float64[]; t1 = Float64[]; t2 = Float64[]
    for trial in 1:ntrial
        Y = U + ε * (randn(rng, d, d) + im * randn(rng, d, d)) / sqrt(2)
        Up = polar_unitary(Y)
        s = time_ns(); U1 = naive_align(G, X, Y); push!(t1, (time_ns() - s) / 1e6)
        s = time_ns(); U2 = branch_cut_align(G, X, Y); push!(t2, (time_ns() - s) / 1e6)
        push!(ep, norm(Up - U) / norm(U)); push!(e1, norm(U1 - U) / norm(U)); push!(e2, norm(U2 - U) / norm(U))
        push!(e21, norm(U2 - Up) / norm(U))
        if name[1:3] == "exp"
            Hhat = im * log(U2) / t
            push!(eh, norm(Hhat - Hising) / norm(Hising))
        end
    end
    @printf("%-36s %6.0e %10.2e %10.2e %10.2e %10.2e %10s %8.0f %8.0f\n", name, ε, median(ep), median(e1), median(e2), median(e21),
            isempty(eh) ? "" : @sprintf("%.2e", median(eh)), median(t1), median(t2))
end
