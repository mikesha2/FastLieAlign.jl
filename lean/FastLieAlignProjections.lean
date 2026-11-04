/-
  Formalization of the projections onto the classical Lie algebras from the manuscript
  "Vector alignment in matrix Lie groups": Theorem 2.1 (projection for groups preserving
  a form), Table 2 (the closed forms), Proposition S1.1 of the Supplement (the spin
  algebra) and Appendix A.4 (the real inner product and the real embedding of complex
  matrices).

  For each Lie algebra listed in Table 2, we state the closed-form formula
  and prove the two defining properties of the orthogonal projection:
    (1) membership in the Lie algebra, and
    (2) orthogonality against every element of the algebra using the
        Frobenius (Hilbert–Schmidt) inner product.

  Status: All sections build with no `sorry` and depend only on the standard axioms
  `propext`, `Classical.choice` and `Quot.sound`.  Every membership and
  orthogonality theorem is proved, including the symplectic cases
  (`proj_Sp_mem` via the matrix identity `(l₀ᵀ + J l₀ J) J + J(l₀ + Jᵀ l₀ᵀ Jᵀ)
  = 0`, and `proj_Sp_orthogonal` via the trace-cycle through `J v = -vᵀ J`).

  The five error bound theorems (`proj_SL/SO/U/Opq/Sp_error_bound`) are proved:
  the deviation `l₀ - proj_G(l₀)` is `(trace l₀ / n) • 1` for SL and
  `(1/2) • (l₀ ± conjugate)` for the symmetric-part projections, giving the
  Frobenius squared-norm bounds.

  Spin(n), Proposition S1.1 of the Supplement: for Hermitian Euclidean gamma
  matrices (`IsCliffordHerm`), the
  bivectors `½γᵢγⱼ` satisfy `Re tr(Sᵢⱼᴴ Sₖₗ) = (d/4)δ` (`bivector_gram_diag`,
  `bivector_gram_offdiag`), so `(2/√d)Sᵢⱼ` is orthonormal
  (`spin_basis_orthonormal`); projection onto the real span of an orthonormal
  family is the unique closest point (`onb_proj_orthogonal`, `onb_proj_optimal`).

  SE(n), SE(p,q): blockwise projections in homogeneous coordinates
  (`proj_SE_mem/orthogonal/optimal`, `proj_SEpq_mem/orthogonal/optimal`).

  The Frobenius norm bound `frobenius_inner_add_sq_eq_add_sq_of_orthogonal`
  (Pythagorean theorem for the Frobenius inner product) is proved.

  Appendix A.4 (`realEmbedding_is_homomorphism`): the real embedding
  `φ : Matrix n n ℂ → Matrix (n ⊕ n) (n ⊕ n) ℝ`,
  `φ(M) = [[Re M, -Im M], [Im M, Re M]]`, is an injective unital
  real-algebra homomorphism (additivity, real homogeneity, identity,
  multiplicativity), and (`exp_commutes_realEmbedding`)
  `exp(φ M) = φ(exp M)`, proved from the ring homomorphism
  plus continuity via `NormedSpace.map_exp`.

  Strategy:
    - Membership: proved entry-wise via `Matrix.ext` followed by `ring`.
    - Orthogonality: proved by rewriting `l₀ - P` to a single matrix using
      `Matrix.ext` + `ring`, then rewriting the resulting trace using
      Mathlib's `Matrix.trace_*` lemmas (`trace_transpose`, `trace_mul_comm`,
      `trace_smul`, `trace_add`, `trace_sub`, `trace_neg`, `trace_transpose_mul`).
-/

import Mathlib.LinearAlgebra.Matrix.Trace
import Mathlib.LinearAlgebra.Matrix.Nondegenerate
import Mathlib.LinearAlgebra.Matrix.Symmetric
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.InnerProductSpace.Projection.Basic
import Mathlib.Analysis.Normed.Algebra.MatrixExponential
import Mathlib.LinearAlgebra.FiniteDimensional.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Complex.Basic

open Matrix
open scoped InnerProductSpace

namespace FastLieAlignProjections

variable {n : Type*} [DecidableEq n] [Fintype n] [Nonempty n]

/-! ## The substitution principle -/

section SubstitutionPrinciple

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- **Substitution principle**: if `P ∈ g` and `⟪l₀ - P, v⟫ = 0` for every
    `v ∈ g`, then `P` is the (star) projection of `l₀` onto `g`. -/
theorem starProjection_eq_of_mem_orthogonal
    {g : Submodule ℝ E} [g.HasOrthogonalProjection]
    {l₀ P : E}
    (hP : P ∈ g)
    (hOrth : ∀ v ∈ g, ⟪l₀ - P, v⟫_ℝ = 0) :
    g.starProjection l₀ = P :=
  Submodule.eq_starProjection_of_mem_of_inner_eq_zero hP hOrth

end SubstitutionPrinciple

/-! ## GL(n): projection is the identity

`gl(n) = 𝔤𝔩(n) = Mat(n, R)` is the entire ambient space, so the projection
of any matrix onto it is the matrix itself. -/

section GLSection

variable {R : Type*} [CommRing R]

/-- The projection of `l₀` onto `gl(n) = Mat(n, R)` is `l₀` itself. -/
def proj_GL (l₀ : Matrix n n R) : Matrix n n R := l₀

end GLSection

/-! ## SL(n): projection is `l₀ - (tr l₀ / n) • I`

The Lie algebra `sl(n)` consists of traceless matrices. -/

section SL

variable {R : Type*} [Field R] [CharZero R]

/-- Projection onto `sl(n)`: `P := l₀ - (tr l₀ / Fintype.card n) • 1`. -/
def proj_SL (l₀ : Matrix n n R) : Matrix n n R :=
  l₀ - (Matrix.trace l₀ / (Fintype.card n : R)) • (1 : Matrix n n R)

/-- **Membership**: `trace P = 0`. -/
theorem proj_SL_mem (l₀ : Matrix n n R) :
    Matrix.trace (proj_SL l₀) = 0 := by
  unfold proj_SL
  rw [Matrix.trace_sub, Matrix.trace_smul, Matrix.trace_one]
  show Matrix.trace l₀ - (Matrix.trace l₀ / ↑(Fintype.card n)) *
        ↑(Fintype.card n) = 0
  rw [div_mul_cancel₀ _ (Nat.cast_ne_zero.mpr Fintype.card_ne_zero)]
  ring

/-- **Orthogonality**: `trace ((l₀ - P)ᵀ v) = 0` for every traceless `v`.

    The deviation `l₀ - P = (tr l₀ / n) • 1` is a scalar matrix, and
    `trace ((c • 1)ᵀ v) = c • trace v = 0` when `trace v = 0`. -/
theorem proj_SL_orthogonal (l₀ v : Matrix n n R) (hv : Matrix.trace v = 0) :
    Matrix.trace ((l₀ - proj_SL l₀)ᵀ * v) = 0 := by
  unfold proj_SL
  have hkey : l₀ - (l₀ - (Matrix.trace l₀ / (Fintype.card n : R)) •
        (1 : Matrix n n R)) =
      (Matrix.trace l₀ / (Fintype.card n : R)) • (1 : Matrix n n R) := by
    rw [sub_sub_cancel]
  rw [hkey]
  rw [transpose_smul, transpose_one, Matrix.smul_mul, Matrix.one_mul,
      Matrix.trace_smul]
  show (Matrix.trace l₀ / ↑(Fintype.card n)) * Matrix.trace v = 0
  rw [hv, mul_zero]

end SL

/-! ## SO(n): projection is the antisymmetric part `(l₀ - l₀ᵀ) / 2`

The Lie algebra `so(n)` consists of skew-symmetric matrices: `Aᵀ = -A`. -/

section SO

variable {R : Type*} [Field R] [Invertible (2 : R)] [NeZero (2 : R)]

/-- `1/2` as a scalar in `R`. -/
local notation "half" => (2 : R)⁻¹

/-- Projection onto `so(n)`: `P := (1/2) • (l₀ - l₀ᵀ)`. -/
def proj_SO (l₀ : Matrix n n R) : Matrix n n R :=
  half • (l₀ - l₀ᵀ)

/-- **Membership**: `Pᵀ = -P`. -/
theorem proj_SO_mem (l₀ : Matrix n n R) :
    (proj_SO l₀)ᵀ = - proj_SO l₀ := by
  unfold proj_SO
  rw [transpose_smul, transpose_sub, transpose_transpose, ← smul_neg, neg_sub]

/-- **Helper identity**: for any `A, B` real matrices with `Bᵀ = -B`
    (skew-symmetric), `trace (Aᵀ * B) = -trace (A * B)`.

    Proof chain:
    `trace (Aᵀ * B) = trace (B * Aᵀ)`                     (trace_mul_comm)
                   `= trace ((B * Aᵀ)ᵀ)`                     (trace_transpose)
                   `= trace (Aᵀᵀ * Bᵀ)`                       (transpose_mul)
                   `= trace (A * (-B))`                       (transpose_transpose, hv)
                   `= -trace (A * B)`.                       (trace_neg) -/
theorem trace_skew_orthogonal_symmetric
    (A B : Matrix n n R) (hv : Bᵀ = -B) :
    Matrix.trace (Aᵀ * B) = -Matrix.trace (A * B) := by
  -- Chain:
  --   `trace (Aᵀ * B) = trace ((Aᵀ * B)ᵀ)` = `trace (Bᵀ * A)`  (transpose)
  --                  `= trace ((-B) * A)`   (hv)
  --                  `= -trace (A * B)`.    (trace_mul_comm, trace_neg)
  -- First: use `show` to make the LHS be `trace ((Aᵀ * B)ᵀ)`.  This
  -- is provable by `Matrix.transpose_transpose` (since (Xᵀ)ᵀ = X).
  have h_eq : (Aᵀ * B) = (Aᵀ * B) := rfl
  -- We do the rewrite in reverse: turn the LHS into a transpose.
  -- Use `change` to replace the LHS with `trace ((Aᵀ * B)ᵀ)`:
  change Matrix.trace (Aᵀ * B) = -Matrix.trace (A * B)
  rw [show Matrix.trace (Aᵀ * B) = Matrix.trace ((Aᵀ * B)ᵀ) from by
        rw [Matrix.trace_transpose]]
  rw [Matrix.transpose_mul, Matrix.transpose_transpose,
      hv, Matrix.neg_mul, Matrix.trace_mul_comm, Matrix.trace_neg]

/-- **Orthogonality**: `trace ((l₀ - P)ᵀ v) = 0` for every skew-symmetric `v`. -/
theorem proj_SO_orthogonal (l₀ v : Matrix n n R) (hv : vᵀ = -v) :
    Matrix.trace ((l₀ - proj_SO l₀)ᵀ * v) = 0 := by
  unfold proj_SO
  -- Reduce `l₀ - half • (l₀ - l₀ᵀ)` to `half • (l₀ + l₀ᵀ)` entry-wise:
  have hkey : l₀ - half • (l₀ - l₀ᵀ) = half • (l₀ + l₀ᵀ) := by
    have hhalf : (half : R) + half = 1 := by
      show ((2 : R)⁻¹ + (2 : R)⁻¹ : R) = 1
      rw [← two_mul, mul_inv_cancel₀ two_ne_zero]
    have hone_half : (1 : R) - half = half := by
      -- `hhalf : half + half = 1`, so `1 - half = half`.  Prove via
      -- `sub_eq_iff_eq_add : a - b = c ↔ a = c + b`:
      rw [sub_eq_iff_eq_add, hhalf]
    ext i j
    simp only [Matrix.sub_apply, Matrix.smul_apply, Matrix.add_apply,
      transpose_apply]
    linear_combination -hhalf * (l₀ i j)
  rw [hkey]
  rw [transpose_smul, transpose_add, Matrix.smul_mul, Matrix.add_mul,
      Matrix.trace_smul, Matrix.trace_add, Matrix.transpose_transpose]
  -- Apply the helper to `trace (l₀ᵀ * v)`:
  rw [trace_skew_orthogonal_symmetric l₀ v hv]
  rw [neg_add_cancel, smul_zero]

end SO

/-! ## U(n): projection is the skew-Hermitian part `(l₀ - l₀†) / 2`

The Lie algebra `u(n)` consists of skew-Hermitian matrices: `A† = -A`. -/

section U

variable {n : Type*} [DecidableEq n] [Fintype n] [Nonempty n]

/-- `1/2` as a complex scalar. -/
local notation "half" => (2 : ℂ)⁻¹

/-- Projection onto `u(n)`: `P := (1/2) • (l₀ - l₀ᴴ)`. -/
noncomputable def proj_U (l₀ : Matrix n n ℂ) : Matrix n n ℂ :=
  half • (l₀ - l₀ᴴ)

/-- **Membership**: `Pᴴ = -P`. -/
theorem proj_U_mem (l₀ : Matrix n n ℂ) :
    (proj_U l₀)ᴴ = - proj_U l₀ := by
  unfold proj_U
  rw [conjTranspose_smul, conjTranspose_sub, conjTranspose_conjTranspose]
  -- Goal: `star half • (l₀ᴴ - l₀) = - (half • (l₀ - l₀ᴴ))`.
  -- Since `2 : ℂ` is real, `star half = half`.  Use `RCLike.star_inv`
  -- and `Complex.conj_ofReal`:
  rw [show (star half : ℂ) = half from by
        show (star ((2 : ℂ)⁻¹) : ℂ) = (2 : ℂ)⁻¹
        simp [star_inv, Complex.conj_ofReal, Complex.ofReal_inv]]
  -- Goal: `2⁻¹ • (l₀ᴴ - l₀) = -(2⁻¹ • (l₀ - l₀ᴴ))`.
  -- Prove entry-wise.  Use `Matrix.ext`:
  ext i j
  -- Reduce to entry-wise algebraic identity:
  simp only [Matrix.smul_apply, Matrix.sub_apply, Matrix.neg_apply]
  -- Goal: `2⁻¹ * (l₀ᴴ i j - l₀ i j) = -(2⁻¹ * (l₀ i j - l₀ᴴ i j))`.
  ring

/-- **Auxiliary identity**: for `H` Hermitian (`Hᴴ = H`) and `v`
    skew-Hermitian (`vᴴ = -v`), `trace (H v)` is purely imaginary:
    `trace (H v) + star (trace (H v)) = 0`. -/
lemma trace_Hermitian_smul_skewHermelian_im
    (H v : Matrix n n ℂ) (hH : Hᴴ = H) (hv : vᴴ = -v) :
    Matrix.trace (H * v) + star (Matrix.trace (H * v)) = 0 := by
  rw [← Matrix.trace_conjTranspose, conjTranspose_mul, hH, hv]
  rw [neg_mul, Matrix.trace_neg, Matrix.trace_mul_comm]
  ring

/-- **Orthogonality** in the *real* Frobenius inner product on complex
    matrices: `Re (trace ((l₀ - P)ᴴ v)) = 0`, equivalently
    `trace ((l₀ - P)ᴴ v) + star (trace ((l₀ - P)ᴴ v)) = 0`. -/
theorem proj_U_orthogonal (l₀ v : Matrix n n ℂ) (hv : vᴴ = -v) :
    Matrix.trace ((l₀ - proj_U l₀)ᴴ * v) +
      star (Matrix.trace ((l₀ - proj_U l₀)ᴴ * v)) = 0 := by
  unfold proj_U
  -- Reduce `l₀ - half • (l₀ - l₀ᴴ)` to `half • (l₀ + l₀ᴴ)` entry-wise:
  have hkey : l₀ - half • (l₀ - l₀ᴴ) = half • (l₀ + l₀ᴴ) := by
    have hhalf : (half : ℂ) + half = 1 := by
      show ((2 : ℂ)⁻¹ + (2 : ℂ)⁻¹ : ℂ) = 1
      rw [← two_mul, mul_inv_cancel₀ two_ne_zero]
    have hone_half : (1 : ℂ) - half = half := by
      rw [sub_eq_iff_eq_add, hhalf]
    ext i j
    simp only [Matrix.sub_apply, Matrix.smul_apply, Matrix.add_apply,
      conjTranspose_apply]
    linear_combination -hhalf * (l₀ i j)
  rw [hkey]
  rw [conjTranspose_smul, conjTranspose_add, conjTranspose_conjTranspose]
  rw [show (star half : ℂ) = half from by
        show (star ((2 : ℂ)⁻¹) : ℂ) = (2 : ℂ)⁻¹
        simp [star_inv, Complex.conj_ofReal, Complex.ofReal_inv]]
  rw [Matrix.smul_mul, Matrix.trace_smul, star_smul]
  rw [show (star half : ℂ) = half from by
        show (star ((2 : ℂ)⁻¹) : ℂ) = (2 : ℂ)⁻¹
        simp [star_inv, Complex.conj_ofReal, Complex.ofReal_inv]]
  rw [← smul_add]
  have hH : (l₀ᴴ + l₀)ᴴ = l₀ᴴ + l₀ := by
    rw [conjTranspose_add, conjTranspose_conjTranspose, add_comm]
  rw [trace_Hermitian_smul_skewHermelian_im (l₀ᴴ + l₀) v hH hv]
  rw [smul_zero]

end U

/-! ## O(p, q): projection is `(l₀ - η l₀ᵀ η) / 2`

The Lie algebra `o(p, q)` consists of matrices `A` with `Aᵀ η + η A = 0`. -/

section O_pq

variable {R : Type*} [Field R] [Invertible (2 : R)] [NeZero (2 : R)]

/-- `1/2` as a scalar in `R`. -/
local notation "half" => (2 : R)⁻¹

/-- Projection onto `o(p, q)`: `P := (1/2) • (l₀ - η l₀ᵀ η)`. -/
noncomputable def proj_Opq (η : Matrix n n R) (l₀ : Matrix n n R) :
    Matrix n n R :=
  half • (l₀ - η * l₀ᵀ * η)

/-- **Membership**: `Pᵀ η + η P = 0`. -/
theorem proj_Opq_mem (η l₀ : Matrix n n R)
    (hη_symm : ηᵀ = η) (hη_sq : η * η = 1) :
    (proj_Opq η l₀)ᵀ * η + η * proj_Opq η l₀ = 0 := by
  unfold proj_Opq
  rw [transpose_smul, transpose_sub, Matrix.transpose_mul, Matrix.transpose_mul,
      transpose_transpose, hη_symm]
  rw [Matrix.smul_mul, Matrix.sub_mul]
  rw [Matrix.mul_assoc, Matrix.mul_assoc, hη_sq, Matrix.mul_one]
  rw [Matrix.mul_smul, Matrix.mul_sub]
  rw [show η * ((η * l₀ᵀ) * η) = l₀ᵀ * η from by
        simp only [← Matrix.mul_assoc, hη_sq, Matrix.one_mul]]
  rw [← smul_add]
  abel
  simp

/-- **Orthogonality**: `trace ((l₀ - P)ᵀ v) = 0` for every `v ∈ o(p, q)`
    (i.e. `vᵀ η + η v = 0`).

    The deviation is `d = ½ • (l₀ + η l₀ᵀ η)`, which satisfies `η dᵀ η = d`,
    while every `v ∈ o(p,q)` satisfies `v = -(η vᵀ η)`.  Then
    `trace (dᵀ v) = -trace (η dᵀ η vᵀ) = -trace (d vᵀ) = -trace (dᵀ v)` by
    cyclicity, forcing `trace (dᵀ v) = 0`. -/
theorem proj_Opq_orthogonal (η l₀ v : Matrix n n R)
    (hη_symm : ηᵀ = η) (hη_sq : η * η = 1)
    (hv : vᵀ * η + η * v = 0) :
    Matrix.trace ((l₀ - proj_Opq η l₀)ᵀ * v) = 0 := by
  -- `v = -(η vᵀ η)`:
  have hηv : η * v = -(vᵀ * η) := eq_neg_of_add_eq_zero_right hv
  have hvconj : v = -(η * vᵀ * η) := by
    have h2 : η * (η * v) = η * -(vᵀ * η) := by rw [hηv]
    rw [Matrix.mul_neg, ← Matrix.mul_assoc, hη_sq, Matrix.one_mul] at h2
    rw [← Matrix.mul_assoc] at h2
    exact h2
  -- deviation `l₀ - P = ½ • (l₀ + η l₀ᵀ η)`:
  have hdev : l₀ - proj_Opq η l₀ = (2⁻¹ : R) • (l₀ + η * l₀ᵀ * η) := by
    unfold proj_Opq
    have key : (2⁻¹ : R) • (l₀ + η * l₀ᵀ * η)
        + (2⁻¹ : R) • (l₀ - η * l₀ᵀ * η) = l₀ := by
      rw [← smul_add]
      rw [show (l₀ + η * l₀ᵀ * η) + (l₀ - η * l₀ᵀ * η) = (2 : R) • l₀ by
        rw [two_smul]; abel]
      rw [smul_smul, inv_mul_cancel₀ (NeZero.ne (2 : R)), one_smul]
    exact (eq_sub_of_add_eq key).symm
  rw [hdev, transpose_smul, Matrix.smul_mul, Matrix.trace_smul]
  -- reduce to `trace ((l₀ + η l₀ᵀ η)ᵀ v) = 0`:
  have hmain : Matrix.trace ((l₀ + η * l₀ᵀ * η)ᵀ * v) = 0 := by
    -- `η (l₀ + η l₀ᵀ η)ᵀ η = l₀ + η l₀ᵀ η`:
    have hAconj : η * (l₀ + η * l₀ᵀ * η)ᵀ * η = l₀ + η * l₀ᵀ * η := by
      rw [transpose_add, Matrix.transpose_mul, Matrix.transpose_mul,
          transpose_transpose, hη_symm]
      rw [Matrix.mul_add, Matrix.add_mul]
      rw [show (η * (η * (l₀ * η))) * η = l₀ by
        simp only [← Matrix.mul_assoc, hη_sq, Matrix.one_mul]
        rw [Matrix.mul_assoc, hη_sq, Matrix.mul_one]]
      abel
    -- doubling: `t = -t`:
    have hdouble : Matrix.trace ((l₀ + η * l₀ᵀ * η)ᵀ * v)
        = - Matrix.trace ((l₀ + η * l₀ᵀ * η)ᵀ * v) := by
      conv_lhs => rw [hvconj]
      rw [Matrix.mul_neg, Matrix.trace_neg]
      congr 1
      rw [show (l₀ + η * l₀ᵀ * η)ᵀ * (η * vᵀ * η)
            = ((l₀ + η * l₀ᵀ * η)ᵀ * (η * vᵀ)) * η by simp [Matrix.mul_assoc]]
      rw [Matrix.trace_mul_comm]
      rw [show η * ((l₀ + η * l₀ᵀ * η)ᵀ * (η * vᵀ))
            = (η * (l₀ + η * l₀ᵀ * η)ᵀ * η) * vᵀ by simp [Matrix.mul_assoc]]
      rw [hAconj]
      rw [← Matrix.trace_transpose ((l₀ + η * l₀ᵀ * η) * vᵀ),
          Matrix.transpose_mul, transpose_transpose, Matrix.trace_mul_comm]
    -- `t = -t` over a field of characteristic ≠ 2 forces `t = 0`:
    have htt : Matrix.trace ((l₀ + η * l₀ᵀ * η)ᵀ * v)
        + Matrix.trace ((l₀ + η * l₀ᵀ * η)ᵀ * v) = 0 := by
      nth_rewrite 1 [hdouble]; ring
    have h2 : (2 : R) ≠ 0 := NeZero.ne 2
    rw [← two_mul] at htt
    exact (mul_eq_zero.mp htt).resolve_left h2
  rw [hmain, smul_zero]

end O_pq

/-! ## Sp(2m): projection is `(l₀ + Jᵀ l₀ᵀ Jᵀ) / 2`

The symplectic Lie algebra `sp(2m)` consists of matrices `A` with
`Aᵀ J + J A = 0`, where `J` is the standard symplectic form with
`Jᵀ = -J` and `J Jᵀ = 1`. -/

section Sp

variable {R : Type*} [Field R] [Invertible (2 : R)] [NeZero (2 : R)]

/-- `1/2` as a scalar in `R`. -/
local notation "half" => (2 : R)⁻¹

/-- Projection onto `sp(2m)`: `P := (1/2) • (l₀ + Jᵀ l₀ᵀ Jᵀ)`. -/
noncomputable def proj_Sp (J : Matrix n n R) (l₀ : Matrix n n R) :
    Matrix n n R :=
  half • (l₀ + Jᵀ * l₀ᵀ * Jᵀ)

/-- **Membership**: `Pᵀ J + J P = 0`. -/
theorem proj_Sp_mem (J l₀ : Matrix n n R)
    (hJ : Jᵀ = -J) (hJJ : J * Jᵀ = 1) :
    (proj_Sp J l₀)ᵀ * J + J * proj_Sp J l₀ = 0 := by
  -- Derive `J² = -1` from `hJ : Jᵀ = -J` and `hJJ : J * Jᵀ = 1`:
  have hJJ2 : J * J = -1 := by
    have e1 : J * (-J) = 1 := by rw [← hJ]; exact hJJ
    rw [Matrix.mul_neg] at e1
    rwa [neg_eq_iff_eq_neg] at e1
  -- Strategy: rewrite `(proj_Sp)ᵀ J + J (proj_Sp)` entry-wise.
  --   `(proj_Sp)ᵀ = half • (l₀ᵀ + (Jᵀ l₀ᵀ Jᵀ)ᵀ) = half • (l₀ᵀ + J l₀ J)` (transpose_*)
  --   `(Pᵀ) J = half • (l₀ᵀ J + J l₀ J²) = half • (l₀ᵀ J - J l₀)`     (hJJ2)
  --   `J P = half • (J l₀ + J Jᵀ l₀ᵀ Jᵀ) = half • (J l₀ + l₀ᵀ Jᵀ)`      (hJJ)
  --              `= half • (J l₀ - l₀ᵀ J)`                                (hJ)
  --   Sum: `half • (l₀ᵀ J - J l₀ + J l₀ - l₀ᵀ J) = 0`.  Pairwise cancellation.
  -- The bracketed matrix identity (independent of the `1/2` scalar):
  --   `(l₀ᵀ + J(l₀ J))J + J(l₀ + Jᵀ l₀ᵀ Jᵀ) = 0`.
  have hX : (l₀ᵀ + J * (l₀ * J)) * J + J * (l₀ + Jᵀ * l₀ᵀ * Jᵀ) = 0 := by
    rw [Matrix.add_mul, Matrix.mul_add]
    rw [show J * (l₀ * J) * J = J * l₀ * (J * J) by simp [Matrix.mul_assoc]]
    rw [hJJ2]
    rw [show J * (Jᵀ * l₀ᵀ * Jᵀ) = (J * Jᵀ) * l₀ᵀ * Jᵀ by simp [Matrix.mul_assoc]]
    rw [hJJ, Matrix.one_mul, hJ]
    simp only [Matrix.mul_neg, Matrix.mul_one]
    abel
  unfold proj_Sp
  rw [transpose_smul, transpose_add, Matrix.transpose_mul, Matrix.transpose_mul,
      transpose_transpose, transpose_transpose, Matrix.smul_mul, Matrix.mul_smul,
      ← smul_add, hX, smul_zero]

/-- **Orthogonality**: `trace ((l₀ - P)ᵀ v) = 0` for every `v ∈ sp(2m)`.

    The deviation is `half • (l₀ - Jᵀ l₀ᵀ Jᵀ)`, and `trace ((l₀ - Jᵀ l₀ᵀ Jᵀ)ᵀ v)
    = trace (l₀ᵀ v) - trace (J l₀ J v)`.  We prove `trace (l₀ᵀ v) =
    trace (J l₀ J v)` using the cycle chain in the proof body. -/
theorem proj_Sp_orthogonal (J l₀ v : Matrix n n R)
    (hJ : Jᵀ = -J) (hJJ : J * Jᵀ = 1)
    (hv : vᵀ * J + J * v = 0) :
    Matrix.trace ((l₀ - proj_Sp J l₀)ᵀ * v) = 0 := by
  have hJJ2 : J * J = -1 := by
    have e1 : J * (-J) = 1 := by rw [← hJ]; exact hJJ
    rw [Matrix.mul_neg] at e1
    rwa [neg_eq_iff_eq_neg] at e1
  have hJv : J * v = -(vᵀ * J) := eq_neg_of_add_eq_zero_right hv
  -- The deviation `l₀ - P = (1/2) • (l₀ - Jᵀ l₀ᵀ Jᵀ)`:
  have hdev : l₀ - proj_Sp J l₀ = (2⁻¹ : R) • (l₀ - Jᵀ * l₀ᵀ * Jᵀ) := by
    unfold proj_Sp
    have h2 : (2 : R) ≠ 0 := NeZero.ne 2
    have key : (2⁻¹ : R) • (l₀ - Jᵀ * l₀ᵀ * Jᵀ)
        + (2⁻¹ : R) • (l₀ + Jᵀ * l₀ᵀ * Jᵀ) = l₀ := by
      rw [← smul_add]
      rw [show (l₀ - Jᵀ * l₀ᵀ * Jᵀ) + (l₀ + Jᵀ * l₀ᵀ * Jᵀ) = (2 : R) • l₀ by
        rw [two_smul]; abel]
      rw [smul_smul, inv_mul_cancel₀ h2, one_smul]
    exact (eq_sub_of_add_eq key).symm
  rw [hdev, transpose_smul, Matrix.smul_mul, Matrix.trace_smul,
      transpose_sub, Matrix.sub_mul, Matrix.trace_sub]
  -- Reduce to `trace (l₀ᵀ v) = trace ((Jᵀ l₀ᵀ Jᵀ)ᵀ v)`:
  have htr : Matrix.trace ((Jᵀ * l₀ᵀ * Jᵀ)ᵀ * v) = Matrix.trace (l₀ᵀ * v) := by
    have ht : (Jᵀ * l₀ᵀ * Jᵀ)ᵀ = J * l₀ * J := by
      simp [Matrix.transpose_mul, Matrix.mul_assoc]
    rw [ht]
    rw [show J * l₀ * J * v = J * ((l₀ * J) * v) by simp [Matrix.mul_assoc]]
    rw [Matrix.trace_mul_comm]
    rw [show (l₀ * J) * v * J = l₀ * (J * v) * J by simp [Matrix.mul_assoc]]
    rw [hJv]
    rw [show l₀ * (-(vᵀ * J)) * J = -(l₀ * vᵀ * (J * J)) by
      simp [Matrix.mul_assoc, Matrix.mul_neg, Matrix.neg_mul]]
    rw [hJJ2]
    rw [show -(l₀ * vᵀ * (-1 : Matrix n n R)) = l₀ * vᵀ by
      simp [Matrix.mul_neg, Matrix.mul_one]]
    rw [Matrix.trace_mul_comm,
        ← Matrix.trace_transpose (vᵀ * l₀), Matrix.transpose_mul,
        Matrix.transpose_transpose]
  rw [htr, sub_self, smul_zero]

end Sp

/-! ## Norm bound for the projections

For each Lie algebra `g` and its associated projection `proj_G`, we have the
identity `‖proj_G(l₀)‖_F² = ‖l₀‖_F² - ‖l₀ - proj_G(l₀)‖_F²`, which follows from
the Pythagorean theorem (the deviation `l₀ - P` is orthogonal to `P`). -/

section NormBound

/-- **Pythagorean theorem for matrix Frobenius inner product**: if `⟪P, D⟫ = 0`
    (i.e., `trace (Pᵀ * D) = 0`), then `‖P + D‖² = ‖P‖² + ‖D‖²`.

    Proof.  Write `‖P + D‖² = ⟨P + D, P + D⟩`.  Expand the square:
    ```
      ‖P + D‖²
        = ⟨P + D, P + D⟩
        = ⟨P, P⟩ + ⟨P, D⟩ + ⟨D, P⟩ + ⟨D, D⟩
        = ‖P‖² + ⟨P, D⟩ + ⟨D, P⟩ + ‖D‖².
    ```
    The cross terms cancel: `⟨P, D⟩ = trace(Pᵀ D) = 0` by hypothesis, and
    `⟨D, P⟩ = trace(Dᵀ P) = trace(Pᵀ D) = 0` by cyclicity.  Hence
    `‖P + D‖² = ‖P‖² + ‖D‖²`. -/
theorem frobenius_inner_add_sq_eq_add_sq_of_orthogonal
    (P D : Matrix n n ℝ) (hOrth : Matrix.trace (Pᵀ * D) = 0) :
    Matrix.trace ((P + D)ᵀ * (P + D)) =
      Matrix.trace (Pᵀ * P) + Matrix.trace (Dᵀ * D) := by
  -- `‖P + D‖² = trace((P + D)ᵀ (P + D))`.  Expand the square:
  --   `(P + D)ᵀ = Pᵀ + Dᵀ`
  --   `(Pᵀ + Dᵀ) * (P + D) = Pᵀ * (P + D) + Dᵀ * (P + D)`
  --                         = (Pᵀ * P + Pᵀ * D) + (Dᵀ * P + Dᵀ * D)`.
  -- Taking the trace gives `trace(Pᵀ P) + trace(Pᵀ D) + trace(Dᵀ P) + trace(Dᵀ D)`.
  rw [transpose_add, Matrix.add_mul, Matrix.mul_add, Matrix.mul_add,
      Matrix.trace_add, Matrix.trace_add, Matrix.trace_add]
  -- The two cross terms: `trace(Pᵀ D) = 0` by hypothesis, and
  -- `trace(Dᵀ P) = trace(P Dᵀ) = trace(Pᵀ D) = 0` by cyclicity and the
  -- transpose-of-product identity.
  rw [hOrth]
  -- `trace(Dᵀ P) = trace((Dᵀ P)ᵀ) = trace(Pᵀ D)` (trace-of-transpose and
  -- `trace_mul_comm`), and `trace(Pᵀ D) = 0` by hypothesis.
  rw [← Matrix.trace_transpose (Dᵀ * P), Matrix.transpose_mul,
      Matrix.transpose_transpose, Matrix.trace_mul_comm, hOrth]
  -- `trace(Pᵀ P) + 0 + 0 + trace(Dᵀ D) = trace(Pᵀ P) + trace(Dᵀ D)`:
  ring

/-- The Frobenius squared norm of a real matrix is the trace of `Aᵀ A`. -/
def frobenius_sq_norm (A : Matrix n n ℝ) : ℝ :=
  Matrix.trace (Aᵀ * A)

/-- The Frobenius squared norm of a complex matrix (using transpose, not
    conjugate-transpose, in the inner product). -/
def frobenius_sq_norm_complex (A : Matrix n n ℂ) : ℂ :=
  Matrix.trace (Aᵀ * A)

/-- For `sl(n)`, the deviation `l₀ - proj_SL(l₀)` is a scalar matrix
    `(trace l₀ / n) • 1`.  Hence its Frobenius squared norm is
    `(trace l₀)² / n`. -/
theorem proj_SL_error_bound (l₀ : Matrix n n ℝ) :
    frobenius_sq_norm (l₀ - proj_SL l₀) =
      (Matrix.trace l₀) ^ 2 / (Fintype.card n : ℝ) := by
  unfold proj_SL frobenius_sq_norm
  have hkey : l₀ - (l₀ - (Matrix.trace l₀ / (Fintype.card n : ℝ)) • 1) =
      (Matrix.trace l₀ / (Fintype.card n : ℝ)) • 1 := by
    rw [sub_sub_cancel]
  rw [hkey]
  -- `(c • 1)ᵀ = c • 1`.  Then `(c • 1)ᵀ * (c • 1) = (c • 1) * (c • 1) = c² • 1`.
  -- Trace: `c² * trace 1 = c² * n`.
  simp [transpose_smul, transpose_one, Matrix.smul_mul, Matrix.one_mul,
    Matrix.trace_smul, Matrix.trace_one, Matrix.mul_smul]
  -- Now: `(trace/n) * trace = trace² / n`.  Use `field_simp` to reduce to:
  -- `trace² = trace²`.  This requires the field axioms.
  field_simp [Fintype.card_ne_zero]

/-- For `so(n)`, the deviation is `(1/2) • (l₀ + l₀ᵀ)`.  Its Frobenius squared
    norm equals `(1/4) • trace ((l₀ + l₀ᵀ)ᵀ (l₀ + l₀ᵀ))`. -/
theorem proj_SO_error_bound (l₀ : Matrix n n ℝ) :
    frobenius_sq_norm (l₀ - proj_SO l₀) =
      (1 / 4 : ℝ) * Matrix.trace ((l₀ + l₀ᵀ)ᵀ * (l₀ + l₀ᵀ)) := by
  -- `proj_SO` defines `(1/2 : ℝ) • (l₀ - l₀ᵀ)`.  After unfolding:
  -- `l₀ - proj_SO l₀ = l₀ - (1/2) • (l₀ - l₀ᵀ) = (1/2) • (l₀ + l₀ᵀ)`.
  have hkey : l₀ - (1 / 2 : ℝ) • (l₀ - l₀ᵀ) = (1 / 2 : ℝ) • (l₀ + l₀ᵀ) := by
    have hhalf : (1 / 2 : ℝ) + (1 / 2 : ℝ) = 1 := by ring
    have hone_half : (1 : ℝ) - 1 / 2 = 1 / 2 := by linarith
    ext i j
    simp only [Matrix.sub_apply, Matrix.smul_apply, Matrix.add_apply,
      transpose_apply]
    linear_combination -hhalf * l₀ i j
  unfold proj_SO frobenius_sq_norm
  -- `proj_SO l₀` is `2⁻¹ • (l₀ - l₀ᵀ)`, and `(2⁻¹ : ℝ) = 1/2`:
  have hhalf_eq : (2⁻¹ : ℝ) • (l₀ - l₀ᵀ) = (1 / 2 : ℝ) • (l₀ - l₀ᵀ) := by
    congr 1
    rw [inv_eq_one_div]
  rw [hhalf_eq]
  rw [hkey]
  -- Now compute `frobenius_sq_norm ((1/2) • (l₀ + l₀ᵀ))`:
  simp [transpose_smul, transpose_add, transpose_transpose,  Matrix.add_mul, Matrix.mul_smul, Matrix.one_mul,
    Matrix.trace_smul, Matrix.trace_add, Matrix.mul_add]
  ring

/-- For `u(n)`, the deviation is `(1/2) • (l₀ + l₀ᴴ)`.  Same form as SO. -/
theorem proj_U_error_bound (l₀ : Matrix n n ℂ) :
    frobenius_sq_norm_complex (l₀ - proj_U l₀) =
      (1 / 4 : ℂ) * Matrix.trace ((l₀ + l₀ᴴ)ᵀ * (l₀ + l₀ᴴ)) := by
  unfold proj_U frobenius_sq_norm_complex
  -- `l₀ - (1/2) • (l₀ - l₀ᴴ) = (1/2) • (l₀ + l₀ᴴ)`:
  have hkey : l₀ - (1 / 2 : ℂ) • (l₀ - l₀ᴴ) = (1 / 2 : ℂ) • (l₀ + l₀ᴴ) := by
    have hhalf : (1 / 2 : ℂ) + (1 / 2 : ℂ) = 1 := by ring
    have hone_half : (1 : ℂ) - 1 / 2 = 1 / 2 := by rw [sub_eq_iff_eq_add]; ring
    ext i j
    simp only [Matrix.sub_apply, Matrix.smul_apply, Matrix.add_apply,
      conjTranspose_apply]
    linear_combination -hhalf * l₀ i j
  have hhalf_eq : (2⁻¹ : ℂ) • (l₀ - l₀ᴴ) = (1 / 2 : ℂ) • (l₀ - l₀ᴴ) := by
    congr 1
    rw [inv_eq_one_div]
  rw [hhalf_eq]
  rw [hkey]
  -- Compute the norm of `(1/2) • (l₀ + l₀ᴴ)`:
  simp [transpose_smul, transpose_add, Matrix.add_mul,
    Matrix.trace_smul, Matrix.trace_add, Matrix.mul_add]
  ring

/-- For `o(p, q)`, the deviation is `(1/2) • (l₀ + η l₀ᵀ η)`. -/
theorem proj_Opq_error_bound (η l₀ : Matrix n n ℝ)
    (_ : ηᵀ = η) (_ : η * η = 1) :
    frobenius_sq_norm (l₀ - proj_Opq η l₀) =
      (1 / 4 : ℝ) * Matrix.trace ((l₀ + η * l₀ᵀ * η)ᵀ * (l₀ + η * l₀ᵀ * η)) := by
  unfold proj_Opq frobenius_sq_norm
  -- `l₀ - (1/2) • (l₀ - η l₀ᵀ η) = (1/2) • (l₀ + η l₀ᵀ η)` (using η² = 1):
  have hkey : l₀ - (1 / 2 : ℝ) • (l₀ - η * l₀ᵀ * η) =
      (1 / 2 : ℝ) • (l₀ + η * l₀ᵀ * η) := by
    have hhalf : (1 / 2 : ℝ) + (1 / 2 : ℝ) = 1 := by ring
    have hone_half : (1 : ℝ) - 1 / 2 = 1 / 2 := by linarith
    ext i j
    simp only [Matrix.sub_apply, Matrix.smul_apply, Matrix.add_apply,
      Matrix.mul_apply]
    linear_combination -hhalf * l₀ i j
  have hhalf_eq : (2⁻¹ : ℝ) • (l₀ - η * l₀ᵀ * η) = (1 / 2 : ℝ) • (l₀ - η * l₀ᵀ * η) := by
    congr 1
    rw [inv_eq_one_div]
  rw [hhalf_eq]
  rw [hkey]
  -- `(½•A)ᵀ * (½•A) = ¼•(Aᵀ*A)`, so the trace is `¼ * trace(Aᵀ*A)`:
  simp only [transpose_smul, Matrix.smul_mul, Matrix.mul_smul,
    Matrix.trace_smul, smul_eq_mul]
  ring

/-- For `sp(2m)`, the deviation is `(1/2) • (l₀ - Jᵀ l₀ᵀ Jᵀ)`. -/
theorem proj_Sp_error_bound (J l₀ : Matrix n n ℝ)
    (hJ : Jᵀ = -J) (hJJ : J * Jᵀ = 1) :
    frobenius_sq_norm (l₀ - proj_Sp J l₀) =
      (1 / 4 : ℝ) * Matrix.trace ((l₀ - Jᵀ * l₀ᵀ * Jᵀ)ᵀ * (l₀ - Jᵀ * l₀ᵀ * Jᵀ)) := by
  unfold proj_Sp frobenius_sq_norm
  -- `l₀ - (1/2) • (l₀ + Jᵀ l₀ᵀ Jᵀ) = (1/2) • (l₀ - Jᵀ l₀ᵀ Jᵀ)`:
  have hkey : l₀ - (1 / 2 : ℝ) • (l₀ + Jᵀ * l₀ᵀ * Jᵀ) =
      (1 / 2 : ℝ) • (l₀ - Jᵀ * l₀ᵀ * Jᵀ) := by
    have hhalf : (1 / 2 : ℝ) + (1 / 2 : ℝ) = 1 := by ring
    have hone_half : (1 : ℝ) - 1 / 2 = 1 / 2 := by linarith
    ext i j
    simp only [Matrix.sub_apply, Matrix.smul_apply, Matrix.add_apply,
      Matrix.mul_apply]
    linear_combination (hhalf - 1) * l₀ i j
  have hhalf_eq : (2⁻¹ : ℝ) • (l₀ + Jᵀ * l₀ᵀ * Jᵀ) = (1 / 2 : ℝ) • (l₀ + Jᵀ * l₀ᵀ * Jᵀ) := by
    congr 1
    rw [inv_eq_one_div]
  rw [hhalf_eq]
  rw [hkey]
  -- `(½•A)ᵀ * (½•A) = ¼•(Aᵀ*A)`, so the trace is `¼ * trace(Aᵀ*A)`:
  simp only [transpose_smul, Matrix.smul_mul, Matrix.mul_smul,
    Matrix.trace_smul, smul_eq_mul]
  ring

/-- The Frobenius squared norm is nonnegative (it is a sum of squares). -/
theorem frobenius_sq_norm_nonneg (M : Matrix n n ℝ) : 0 ≤ frobenius_sq_norm M := by
  unfold frobenius_sq_norm Matrix.trace
  simp only [Matrix.diag_apply, Matrix.mul_apply, Matrix.transpose_apply]
  exact Finset.sum_nonneg fun i _ =>
    Finset.sum_nonneg fun j _ => mul_self_nonneg _

/-- The Frobenius form is positive definite: `‖M‖²_F = 0 ↔ M = 0`. -/
theorem frobenius_sq_norm_eq_zero_iff (M : Matrix n n ℝ) :
    frobenius_sq_norm M = 0 ↔ M = 0 := by
  constructor
  · intro h
    unfold frobenius_sq_norm Matrix.trace at h
    simp only [Matrix.diag_apply, Matrix.mul_apply, Matrix.transpose_apply] at h
    ext i j
    have hcol := (Finset.sum_eq_zero_iff_of_nonneg fun a _ =>
      Finset.sum_nonneg fun b _ => mul_self_nonneg _).mp h j (Finset.mem_univ j)
    have hentry := (Finset.sum_eq_zero_iff_of_nonneg fun b _ =>
      mul_self_nonneg _).mp hcol i (Finset.mem_univ i)
    simpa using mul_self_eq_zero.mp hentry
  · intro h
    rw [h]
    unfold frobenius_sq_norm
    simp

/-- **Minimality and uniqueness of the projection.**  If the deviation
    `l₀ - P` is Frobenius-orthogonal to `P - g` (which holds for every `g` in
    the Lie algebra, since `P - g` lies in the algebra and `l₀ - P ⟂ 𝔤`), then
    `P` is at least as close to `l₀` as `g`, with equality **iff** `g = P`.
    Hence `P` is the unique global minimizer of `g ↦ ‖l₀ - g‖_F` over `𝔤`. -/
theorem frobenius_proj_unique_min (l₀ P g : Matrix n n ℝ)
    (horth : Matrix.trace ((l₀ - P)ᵀ * (P - g)) = 0) :
    frobenius_sq_norm (l₀ - P) ≤ frobenius_sq_norm (l₀ - g) ∧
      (frobenius_sq_norm (l₀ - g) = frobenius_sq_norm (l₀ - P) ↔ g = P) := by
  -- Write `‖l₀ - g‖² = ⟨l₀ - g, l₀ - g⟩` and decompose
  -- `l₀ - g = (l₀ - P) + (P - g)`.  Expanding the square:
  --   `‖l₀ - g‖²
  --      = ‖l₀ - P‖² + 2⟨l₀ - P, P - g⟩ + ‖P - g‖²`
  -- with the cross term vanishing by `horth` (the hypothesis that the
  -- residual `l₀ - P` is Frobenius-orthogonal to every element of the
  -- algebra, here specialized to `P - g`).
  have hpyth : frobenius_sq_norm (l₀ - g)
      = frobenius_sq_norm (l₀ - P) + frobenius_sq_norm (P - g) := by
    -- `frobenius_sq_norm M = trace(Mᵀ M)`.
    simp only [frobenius_sq_norm]
    -- `l₀ - g = (l₀ - P) + (P - g)`.
    rw [show l₀ - g = (l₀ - P) + (P - g) by abel]
    -- Apply the Pythagoras theorem: `trace((P+D)ᵀ(P+D)) = trace(PᵀP) + trace(DᵀD)`.
    exact frobenius_inner_add_sq_eq_add_sq_of_orthogonal (l₀ - P) (P - g) horth
  have hnn : 0 ≤ frobenius_sq_norm (P - g) := frobenius_sq_norm_nonneg _
  refine ⟨by rw [hpyth]; linarith, ?_⟩
  rw [hpyth]
  constructor
  · intro h
    have hzero : frobenius_sq_norm (P - g) = 0 := by linarith
    have : P - g = 0 := (frobenius_sq_norm_eq_zero_iff _).mp hzero
    rw [sub_eq_zero] at this
    exact this.symm
  · intro h; rw [h]; simp [frobenius_sq_norm_eq_zero_iff]

/-- **The projection is a Frobenius contraction.**  If `l₀ - P` is
    Frobenius-orthogonal to `P` (which holds because `P ∈ 𝔤` and `l₀ - P ⟂ 𝔤`),
    then `‖P‖²_F ≤ ‖l₀‖²_F`, with equality **iff** `l₀ = P`, i.e. iff `l₀` is
    already in the Lie algebra. -/
theorem frobenius_proj_contraction (l₀ P : Matrix n n ℝ)
    (horth : Matrix.trace (Pᵀ * (l₀ - P)) = 0) :
    frobenius_sq_norm P ≤ frobenius_sq_norm l₀ ∧
      (frobenius_sq_norm P = frobenius_sq_norm l₀ ↔ l₀ = P) := by
  -- Pythagoras with `l₀ = P + (l₀ - P)`:
  have hpyth : frobenius_sq_norm l₀
      = frobenius_sq_norm P + frobenius_sq_norm (l₀ - P) := by
    have hsplit := frobenius_inner_add_sq_eq_add_sq_of_orthogonal P (l₀ - P) horth
    rw [show P + (l₀ - P) = l₀ by abel] at hsplit
    simpa only [frobenius_sq_norm] using hsplit
  have hnn : 0 ≤ frobenius_sq_norm (l₀ - P) := frobenius_sq_norm_nonneg _
  refine ⟨by rw [hpyth]; linarith, ?_⟩
  rw [hpyth]
  constructor
  · intro h
    have hzero : frobenius_sq_norm (l₀ - P) = 0 := by linarith
    have : l₀ - P = 0 := (frobenius_sq_norm_eq_zero_iff _).mp hzero
    rwa [sub_eq_zero] at this
  · intro h; rw [h]; simp [frobenius_sq_norm_eq_zero_iff]

end NormBound

/-! ## Global optimality of the projections

Combining each `proj_G_mem`/`proj_G_orthogonal` pair with the generic
minimizer lemma `frobenius_proj_unique_min`, every projection is the
**unique** global minimizer of `g ↦ ‖l₀ − g‖_F` over its Lie algebra. -/

section Optimality

/-- **SL(n) optimality**: `proj_SL l₀` is the unique closest traceless matrix. -/
theorem proj_SL_optimal (l₀ g : Matrix n n ℝ) (hg : Matrix.trace g = 0) :
    frobenius_sq_norm (l₀ - proj_SL l₀) ≤ frobenius_sq_norm (l₀ - g) ∧
      (frobenius_sq_norm (l₀ - g) = frobenius_sq_norm (l₀ - proj_SL l₀) ↔
        g = proj_SL l₀) := by
  apply frobenius_proj_unique_min
  apply proj_SL_orthogonal
  rw [Matrix.trace_sub, proj_SL_mem, hg, sub_zero]

/-- **SO(n) optimality**: `proj_SO l₀` is the unique closest skew-symmetric
    matrix. -/
theorem proj_SO_optimal (l₀ g : Matrix n n ℝ) (hg : gᵀ = -g) :
    frobenius_sq_norm (l₀ - proj_SO l₀) ≤ frobenius_sq_norm (l₀ - g) ∧
      (frobenius_sq_norm (l₀ - g) = frobenius_sq_norm (l₀ - proj_SO l₀) ↔
        g = proj_SO l₀) := by
  apply frobenius_proj_unique_min
  apply proj_SO_orthogonal
  rw [transpose_sub, proj_SO_mem, hg]; abel

/-- **O(p, q) optimality**: `proj_Opq η l₀` is the unique closest element of
    `o(p, q)`. -/
theorem proj_Opq_optimal (η l₀ g : Matrix n n ℝ)
    (hη_symm : ηᵀ = η) (hη_sq : η * η = 1)
    (hg : gᵀ * η + η * g = 0) :
    frobenius_sq_norm (l₀ - proj_Opq η l₀) ≤ frobenius_sq_norm (l₀ - g) ∧
      (frobenius_sq_norm (l₀ - g) = frobenius_sq_norm (l₀ - proj_Opq η l₀) ↔
        g = proj_Opq η l₀) := by
  apply frobenius_proj_unique_min
  apply proj_Opq_orthogonal η l₀ _ hη_symm hη_sq
  have hm : (proj_Opq η l₀)ᵀ * η + η * proj_Opq η l₀ = 0 :=
    proj_Opq_mem η l₀ hη_symm hη_sq
  rw [transpose_sub, Matrix.sub_mul, Matrix.mul_sub]
  rw [show (proj_Opq η l₀)ᵀ * η - gᵀ * η + (η * proj_Opq η l₀ - η * g)
        = ((proj_Opq η l₀)ᵀ * η + η * proj_Opq η l₀) - (gᵀ * η + η * g) by abel,
      hm, hg, sub_zero]

/-- **Sp(2m) optimality**: `proj_Sp J l₀` is the unique closest element of
    `sp(2m)`. -/
theorem proj_Sp_optimal (J l₀ g : Matrix n n ℝ)
    (hJ : Jᵀ = -J) (hJJ : J * Jᵀ = 1)
    (hg : gᵀ * J + J * g = 0) :
    frobenius_sq_norm (l₀ - proj_Sp J l₀) ≤ frobenius_sq_norm (l₀ - g) ∧
      (frobenius_sq_norm (l₀ - g) = frobenius_sq_norm (l₀ - proj_Sp J l₀) ↔
        g = proj_Sp J l₀) := by
  apply frobenius_proj_unique_min
  apply proj_Sp_orthogonal J l₀ _ hJ hJJ
  have hm : (proj_Sp J l₀)ᵀ * J + J * proj_Sp J l₀ = 0 :=
    proj_Sp_mem J l₀ hJ hJJ
  rw [transpose_sub, Matrix.sub_mul, Matrix.mul_sub]
  rw [show (proj_Sp J l₀)ᵀ * J - gᵀ * J + (J * proj_Sp J l₀ - J * g)
        = ((proj_Sp J l₀)ᵀ * J + J * proj_Sp J l₀) - (gᵀ * J + J * g) by abel,
      hm, hg, sub_zero]

end Optimality

/-! ## U(n): real Frobenius norm and optimality

For complex matrices the natural objective is the *real* Frobenius squared
norm `‖A‖²_F = Re tr(Aᴴ A) = ∑ |Aᵢⱼ|²`, an ordered quantity admitting `≤`.
The `u(n)` projection is then the unique closest skew-Hermitian matrix. -/

section UOptimality

/-- Real Frobenius squared norm of a complex matrix: `Re tr(Aᴴ A) = ∑ |Aᵢⱼ|²`. -/
noncomputable def frobenius_re_norm (A : Matrix n n ℂ) : ℝ :=
  (Matrix.trace (Aᴴ * A)).re

/-- `‖A‖²_F` is the sum of squared entry moduli. -/
theorem frobenius_re_norm_eq_sum (A : Matrix n n ℂ) :
    frobenius_re_norm A = ∑ i, ∑ k, Complex.normSq (A k i) := by
  unfold frobenius_re_norm Matrix.trace
  rw [Complex.re_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  simp only [Matrix.diag_apply, Matrix.mul_apply, Matrix.conjTranspose_apply]
  rw [Complex.re_sum]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [Complex.mul_re]
  show (A k i).re * (A k i).re - (-(A k i).im) * (A k i).im = Complex.normSq (A k i)
  rw [Complex.normSq_apply]; ring

@[simp] theorem frobenius_re_norm_zero :
    frobenius_re_norm (0 : Matrix n n ℂ) = 0 := by
  rw [frobenius_re_norm_eq_sum]; simp

theorem frobenius_re_norm_nonneg (A : Matrix n n ℂ) : 0 ≤ frobenius_re_norm A := by
  rw [frobenius_re_norm_eq_sum]
  exact Finset.sum_nonneg fun i _ =>
    Finset.sum_nonneg fun k _ => Complex.normSq_nonneg _

theorem frobenius_re_norm_eq_zero_iff (A : Matrix n n ℂ) :
    frobenius_re_norm A = 0 ↔ A = 0 := by
  rw [frobenius_re_norm_eq_sum]
  constructor
  · intro h
    ext i j
    have hcol := (Finset.sum_eq_zero_iff_of_nonneg fun a _ =>
      Finset.sum_nonneg fun b _ => Complex.normSq_nonneg _).mp h j (Finset.mem_univ j)
    have hentry := (Finset.sum_eq_zero_iff_of_nonneg fun b _ =>
      Complex.normSq_nonneg _).mp hcol i (Finset.mem_univ i)
    simpa using Complex.normSq_eq_zero.mp hentry
  · intro h; rw [h]; simp

/-- **Pythagorean theorem for the real Frobenius inner product.**  If
    `Re tr(Pᴴ D) = 0` (i.e. `tr(Pᴴ D) + star (tr(Pᴴ D)) = 0`), then
    `‖P + D‖²_F = ‖P‖²_F + ‖D‖²_F`. -/
theorem frobenius_re_add_sq (P D : Matrix n n ℂ)
    (hOrth : Matrix.trace (Pᴴ * D) + star (Matrix.trace (Pᴴ * D)) = 0) :
    frobenius_re_norm (P + D) = frobenius_re_norm P + frobenius_re_norm D := by
  unfold frobenius_re_norm
  rw [conjTranspose_add, Matrix.add_mul, Matrix.mul_add, Matrix.mul_add,
      Matrix.trace_add, Matrix.trace_add, Matrix.trace_add,
      Complex.add_re, Complex.add_re, Complex.add_re]
  have hcross : Matrix.trace (Dᴴ * P) = star (Matrix.trace (Pᴴ * D)) := by
    rw [show Dᴴ * P = (Pᴴ * D)ᴴ by
          rw [conjTranspose_mul, conjTranspose_conjTranspose],
        Matrix.trace_conjTranspose]
  have hstar_re : (star (Matrix.trace (Pᴴ * D))).re = (Matrix.trace (Pᴴ * D)).re := rfl
  have hre0 : (Matrix.trace (Pᴴ * D)).re = 0 := by
    have h := congrArg Complex.re hOrth
    rw [Complex.add_re, Complex.zero_re, hstar_re] at h
    linarith
  rw [hcross, hstar_re, hre0]; ring

/-- **Minimality + uniqueness** for the real Frobenius norm (complex case). -/
theorem frobenius_re_proj_unique_min (l₀ P g : Matrix n n ℂ)
    (horth : Matrix.trace ((l₀ - P)ᴴ * (P - g))
      + star (Matrix.trace ((l₀ - P)ᴴ * (P - g))) = 0) :
    frobenius_re_norm (l₀ - P) ≤ frobenius_re_norm (l₀ - g) ∧
      (frobenius_re_norm (l₀ - g) = frobenius_re_norm (l₀ - P) ↔ g = P) := by
  have hpyth : frobenius_re_norm (l₀ - g)
      = frobenius_re_norm (l₀ - P) + frobenius_re_norm (P - g) := by
    have hsplit := frobenius_re_add_sq (l₀ - P) (P - g) horth
    rwa [show (l₀ - P) + (P - g) = l₀ - g by abel] at hsplit
  have hnn : 0 ≤ frobenius_re_norm (P - g) := frobenius_re_norm_nonneg _
  refine ⟨by rw [hpyth]; linarith, ?_⟩
  rw [hpyth]
  constructor
  · intro h
    have hzero : frobenius_re_norm (P - g) = 0 := by linarith
    have : P - g = 0 := (frobenius_re_norm_eq_zero_iff _).mp hzero
    rw [sub_eq_zero] at this; exact this.symm
  · intro h; subst h; rw [sub_self]; simp

/-- **U(n) optimality**: `proj_U l₀` is the unique closest skew-Hermitian
    matrix to `l₀` in the real Frobenius norm. -/
theorem proj_U_optimal (l₀ g : Matrix n n ℂ) (hg : gᴴ = -g) :
    frobenius_re_norm (l₀ - proj_U l₀) ≤ frobenius_re_norm (l₀ - g) ∧
      (frobenius_re_norm (l₀ - g) = frobenius_re_norm (l₀ - proj_U l₀) ↔
        g = proj_U l₀) := by
  apply frobenius_re_proj_unique_min
  apply proj_U_orthogonal
  rw [conjTranspose_sub, proj_U_mem, hg]; abel

end UOptimality

/-! ## Error bounds summary

For each Lie algebra, the Frobenius squared norm of the deviation
`l₀ - proj(l₀)` is bounded by:
  - **sl(n)**: `(trace l₀)² / n`
  - **so(n), u(n)**: `(1/4) ‖l₀ + l₀ᵀ‖²_F` (symmetric part)
  - **o(p, q)**: `(1/4) ‖l₀ + η l₀ᵀ η‖²_F`
  - **sp(2m)**: `(1/4) ‖l₀ - Jᵀ l₀ᵀ Jᵀ‖²_F`

The projections are contractions in the Frobenius norm: `‖proj(l₀)‖² ≤ ‖l₀‖²`,
with equality iff `l₀` is already in the Lie algebra. -/


/-! ## The real embedding commutes with the exponential (Appendix A.4)

Complex groups are embedded into real $2n \times 2n$ matrices via the
real-linear map $\varphi(M) = \bigl[\begin{smallmatrix}\mathrm{Re}\,M &
-\mathrm{Im}\,M \\ \mathrm{Im}\,M & \mathrm{Re}\,M\end{smallmatrix}\bigr]$,
which preserves addition, real scaling, the identity, and multiplication.
The exponential of $\varphi(M)$ therefore equals $\varphi(\exp M)$ since
$\varphi(M^k) = \varphi(M)^k$ (by multiplicativity and induction), and
$\varphi$ is continuous on a finite-dimensional space, so it commutes with
the limit in the exponential series. -/

section PhiCommutes

variable {n : Type*} [DecidableEq n] [Fintype n]

/-- **The real embedding of Appendix A.4.** The map
    `φ : Matrix n n ℂ → Matrix (n ⊕ n) (n ⊕ n) ℝ` defined as the
    2×2-block matrix `φ(M) = [[Re M, -Im M], [Im M, Re M]]` (using
    `Sum n n` for the block indices) is an injective unital real-algebra
    homomorphism: it preserves addition, real scaling, the identity, and
    multiplication. -/
def realEmbedding (M : Matrix n n ℂ) : Matrix (n ⊕ n) (n ⊕ n) ℝ :=
  Matrix.fromBlocks (M.map Complex.re) (-M.map Complex.im)
                       (M.map Complex.im)  (M.map Complex.re)

lemma realEmbedding_apply_inl_inl (M : Matrix n n ℂ) (i j : n) :
    realEmbedding M (Sum.inl i) (Sum.inl j) = (M i j).re := by
  show Matrix.fromBlocks (M.map Complex.re) (-M.map Complex.im)
                          (M.map Complex.im) (M.map Complex.re)
                (Sum.inl i) (Sum.inl j) = _
  rw [Matrix.fromBlocks_apply₁₁]
  rfl

lemma realEmbedding_apply_inl_inr (M : Matrix n n ℂ) (i j : n) :
    realEmbedding M (Sum.inl i) (Sum.inr j) = -(M i j).im := by
  show Matrix.fromBlocks (M.map Complex.re) (-M.map Complex.im)
                          (M.map Complex.im) (M.map Complex.re)
                (Sum.inl i) (Sum.inr j) = _
  rw [Matrix.fromBlocks_apply₁₂]
  rfl

lemma realEmbedding_apply_inr_inl (M : Matrix n n ℂ) (i j : n) :
    realEmbedding M (Sum.inr i) (Sum.inl j) = (M i j).im := by
  show Matrix.fromBlocks (M.map Complex.re) (-M.map Complex.im)
                          (M.map Complex.im) (M.map Complex.re)
                (Sum.inr i) (Sum.inl j) = _
  rw [Matrix.fromBlocks_apply₂₁]
  rfl

lemma realEmbedding_apply_inr_inr (M : Matrix n n ℂ) (i j : n) :
    realEmbedding M (Sum.inr i) (Sum.inr j) = (M i j).re := by
  show Matrix.fromBlocks (M.map Complex.re) (-M.map Complex.im)
                          (M.map Complex.im) (M.map Complex.re)
                (Sum.inr i) (Sum.inr j) = _
  rw [Matrix.fromBlocks_apply₂₂]
  rfl

lemma realEmbedding_add (M N : Matrix n n ℂ) :
    realEmbedding (M + N) = realEmbedding M + realEmbedding N := by
  -- Use the `realEmbedding_apply_*` lemmas to expand both sides in terms
  -- of the four 2×2 real blocks, then `simp` with `Complex.add_re`,
  -- `Complex.add_im`, and the linear map lemmas.  Each of the four
  -- resulting goals is closed by `ring` (real parts expand to
  -- `Re(M) + Re(N) = Re(M) + Re(N)`, etc.).
  refine Matrix.ext (fun i j => ?_) <;>
    rcases i with (ri | ri) <;>
    rcases j with (cj | cj) <;>
    simp [realEmbedding_apply_inl_inl, realEmbedding_apply_inl_inr,
          realEmbedding_apply_inr_inl, realEmbedding_apply_inr_inr,
          Matrix.add_apply, Complex.add_re, Complex.add_im,
          add_comm, add_left_comm]

lemma realEmbedding_smul (r : ℝ) (M : Matrix n n ℂ) :
    realEmbedding (r • M) = r • realEmbedding M := by
  refine Matrix.ext (fun i j => ?_) <;>
    rcases i with (ri | ri) <;>
    rcases j with (cj | cj) <;>
    simp [realEmbedding_apply_inl_inl, realEmbedding_apply_inl_inr,
          realEmbedding_apply_inr_inl, realEmbedding_apply_inr_inr,
          Matrix.smul_apply, Complex.smul_re, Complex.smul_im,
          smul_eq_mul, mul_neg]

lemma realEmbedding_one : realEmbedding (1 : Matrix n n ℂ) = 1 := by
  refine Matrix.ext (fun i j => ?_) <;>
    rcases i with (ri | ri) <;>
    rcases j with (cj | cj) <;>
    simp only [realEmbedding_apply_inl_inl, realEmbedding_apply_inl_inr,
          realEmbedding_apply_inr_inl, realEmbedding_apply_inr_inr,
          Matrix.one_apply, Sum.inl.injEq, Sum.inr.injEq,
          reduceCtorEq, apply_ite Complex.re, apply_ite Complex.im,
          Complex.one_re, Complex.one_im, Complex.zero_im,
          Complex.zero_re, neg_zero, if_false] <;>
    split_ifs <;> simp

lemma realEmbedding_mul (M N : Matrix n n ℂ) :
    realEmbedding (M * N) = realEmbedding M * realEmbedding N := by
  -- Each of the four 2×2 real blocks of `realEmbedding (M*N)` equals the
  -- corresponding real-matrix product of the blocks of `realEmbedding M`
  -- and `realEmbedding N`.  Expand the matrix product over `n ⊕ n` with
  -- `Fintype.sum_sum_type`, push `Re`/`Im` through the sum with
  -- `re_sum`/`im_sum`, and use `Re(zw)`/`Im(zw)`; the resulting sums match
  -- termwise by `ring`.
  refine Matrix.ext (fun i j => ?_)
  rcases i with (ri | ri) <;> rcases j with (cj | cj) <;>
    simp only [realEmbedding_apply_inl_inl, realEmbedding_apply_inl_inr,
               realEmbedding_apply_inr_inl, realEmbedding_apply_inr_inr,
               Matrix.mul_apply, Fintype.sum_sum_type, Complex.re_sum,
               Complex.im_sum, Complex.mul_re, Complex.mul_im,
               ← Finset.sum_neg_distrib, ← Finset.sum_add_distrib,
               ← Finset.sum_sub_distrib] <;>
    exact Finset.sum_congr rfl fun k _ => by ring

lemma realEmbedding_injective :
    Function.Injective (realEmbedding (n := n)) := by
  intro M N h
  have hre : ∀ i j, (M i j).re = (N i j).re := fun i j => by
    have h0 : realEmbedding M (Sum.inl i) (Sum.inl j)
            = realEmbedding N (Sum.inl i) (Sum.inl j) := by rw [h]
    rwa [realEmbedding_apply_inl_inl, realEmbedding_apply_inl_inl] at h0
  have him : ∀ i j, (M i j).im = (N i j).im := fun i j => by
    have h0 : realEmbedding M (Sum.inr i) (Sum.inl j)
            = realEmbedding N (Sum.inr i) (Sum.inl j) := by rw [h]
    rwa [realEmbedding_apply_inr_inl, realEmbedding_apply_inr_inl] at h0
  funext i j
  apply Complex.ext <;> first | exact hre i j | exact him i j

/-- **Appendix A.4.** The real embedding
    `realEmbedding : Matrix n n ℂ → Matrix (n ⊕ n) (n ⊕ n) ℝ` is an injective
    unital real-algebra homomorphism. -/
theorem realEmbedding_is_homomorphism :
    Function.Injective (realEmbedding (n := n)) ∧
    (∀ M N : Matrix n n ℂ,
      realEmbedding (M + N) = realEmbedding M + realEmbedding N) ∧
    (∀ (r : ℝ) (M : Matrix n n ℂ),
      realEmbedding (r • M) = r • realEmbedding M) ∧
    realEmbedding (1 : Matrix n n ℂ) = 1 ∧
    (∀ M N : Matrix n n ℂ,
      realEmbedding (M * N) = realEmbedding M * realEmbedding N) := by
  refine ⟨realEmbedding_injective, realEmbedding_add, realEmbedding_smul,
          realEmbedding_one, realEmbedding_mul⟩

lemma realEmbedding_zero : realEmbedding (0 : Matrix n n ℂ) = 0 := by
  refine Matrix.ext (fun i j => ?_) <;>
    rcases i with (ri | ri) <;>
    rcases j with (cj | cj) <;>
    simp [realEmbedding_apply_inl_inl, realEmbedding_apply_inl_inr,
          realEmbedding_apply_inr_inl, realEmbedding_apply_inr_inr,
          Matrix.zero_apply]

/-- `φ` packaged as a real-linear map, used only to obtain continuity from
    finite-dimensionality. -/
def realEmbeddingₗ : Matrix n n ℂ →ₗ[ℝ] Matrix (n ⊕ n) (n ⊕ n) ℝ where
  toFun := realEmbedding
  map_add' := realEmbedding_add
  map_smul' := realEmbedding_smul

/-- `φ` packaged as a ring homomorphism. -/
def realEmbeddingRingHom : Matrix n n ℂ →+* Matrix (n ⊕ n) (n ⊕ n) ℝ where
  toFun := realEmbedding
  map_one' := realEmbedding_one
  map_mul' := realEmbedding_mul
  map_zero' := realEmbedding_zero
  map_add' := realEmbedding_add

open scoped Matrix.Norms.Operator in
lemma realEmbedding_continuous : Continuous (realEmbedding (n := n)) :=
  realEmbeddingₗ.continuous_of_finiteDimensional

open scoped Matrix.Norms.Operator in
/-- **The embedding commutes with the exponential** (Appendix A.4):
    `exp(φ(M)) = φ(exp(M))` for every complex `M`.  Proof: `φ` is a
    continuous unital ring homomorphism, so it commutes with
    `NormedSpace.exp` by `map_exp`:
    absolute convergence of the exponential series together with
    continuity of `φ` gives `φ(Σ Mᵏ/k!) = Σ φ(M)ᵏ/k!`, i.e.
    `φ(exp M) = exp(φ M)`. -/
theorem exp_commutes_realEmbedding (M : Matrix n n ℂ) :
    NormedSpace.exp (realEmbedding M) = realEmbedding (NormedSpace.exp M) :=
  (NormedSpace.map_exp realEmbeddingRingHom realEmbedding_continuous M).symm

end PhiCommutes

/-! ## The data-covariance error analysis

Two identities behind Section 3.2 of the manuscript and the proofs of Section S2 of
the Supplement.  The data-weighted split `‖gX − Y‖² = ‖(g − g₀)X‖² + ‖g₀X − Y‖²`,
used for the Gauss--Newton refinement, reduces to a rectangular Frobenius
Pythagorean theorem once the normal equation `(g₀X − Y)Xᵀ = 0` (satisfied by
the pseudoinverse solution `g₀ = YX⁺`) makes the cross term vanish.  The
isotropy identity behind Corollary 3.3 (`X Xᵀ = c•1`) is that an isotropic
weight `W = σ•1` turns the data-weighted Gram into `σ` times the flat Gram. -/

section DataCovariance

variable {n m : Type*} [Fintype n] [Fintype m] [DecidableEq n]

/-- Rectangular Frobenius Pythagoras: if `trace(Aᵀ B) = 0` then
    `‖A + B‖²_F = ‖A‖²_F + ‖B‖²_F` for `A B : Matrix n m ℝ`. -/
theorem frobenius_add_sq_rect (A B : Matrix n m ℝ)
    (hOrth : Matrix.trace (Aᵀ * B) = 0) :
    Matrix.trace ((A + B)ᵀ * (A + B))
      = Matrix.trace (Aᵀ * A) + Matrix.trace (Bᵀ * B) := by
  have hBA : Matrix.trace (Bᵀ * A) = 0 := by
    rw [← Matrix.trace_transpose (Bᵀ * A), Matrix.transpose_mul,
        Matrix.transpose_transpose, hOrth]
  rw [transpose_add, Matrix.add_mul, Matrix.mul_add, Matrix.mul_add,
      Matrix.trace_add, Matrix.trace_add, Matrix.trace_add, hOrth, hBA]
  ring

/-- **The data-weighted split** (Supplement, Section S2).  If `g₀` satisfies the normal equation
    `(g₀ X − Y) Xᵀ = 0` — which the unconstrained pseudoinverse solution
    `g₀ = Y X⁺` does — then for every `g`,
    `‖gX − Y‖²_F = ‖(g − g₀)X‖²_F + ‖g₀X − Y‖²_F`.  Hence minimizing the
    alignment objective over `g ∈ G` is exactly minimizing the data-weighted
    distance `‖(g − g₀)X‖_F` to the pseudoinverse solution. -/
theorem alignment_objective_split
    (g g₀ : Matrix n n ℝ) (X Y : Matrix n m ℝ)
    (hnormal : (g₀ * X - Y) * Xᵀ = 0) :
    Matrix.trace ((g * X - Y)ᵀ * (g * X - Y))
      = Matrix.trace (((g - g₀) * X)ᵀ * ((g - g₀) * X))
        + Matrix.trace ((g₀ * X - Y)ᵀ * (g₀ * X - Y)) := by
  have hsum : g * X - Y = (g - g₀) * X + (g₀ * X - Y) := by
    rw [Matrix.sub_mul]; abel
  rw [hsum]
  apply frobenius_add_sq_rect
  -- Cross term `trace(((g-g₀)X)ᵀ (g₀X - Y))` vanishes: cycle the trace to expose
  -- the normal-equation factor `(g₀X - Y)Xᵀ = 0`.
  rw [Matrix.transpose_mul, Matrix.trace_mul_comm, ← Matrix.mul_assoc, hnormal,
      Matrix.zero_mul, Matrix.trace_zero]

/-- **Isotropy** (behind Corollary 3.3).  If the weight is isotropic, `W = σ • 1`,
    as for inputs with `X Xᵀ = c • 1`, then the data-weighted Gram of two
    algebra elements is `σ` times the flat Gram,
    `trace(Aᵀ B W) = σ · trace(Aᵀ B)`.  An orthonormal basis of `𝔤`
    (`trace(B_kᵀ B_j) = δ_kj`) therefore stays orthonormal in the data metric,
    which is why the projection and the least squares solution agree to first
    order for isotropic inputs. -/
theorem gram_isotropic (A B W : Matrix n n ℝ) (σ : ℝ)
    (hW : W = σ • (1 : Matrix n n ℝ)) :
    Matrix.trace (Aᵀ * B * W) = σ * Matrix.trace (Aᵀ * B) := by
  rw [hW, Matrix.mul_smul, Matrix.mul_one, Matrix.trace_smul, smul_eq_mul]

end DataCovariance

/-! ## Spin(n): the bivector basis in the spinor representation (Proposition S1.1)

Let `γ : Fin k → Matrix N N ℂ` be Hermitian Euclidean gamma matrices,
`γᵢ² = 1`, `γᵢγⱼ = -γⱼγᵢ` (`i ≠ j`), `γᵢᴴ = γᵢ`, acting on a spinor space of
dimension `d = card N`.  The bivectors `Sᵢⱼ = ½ γᵢγⱼ` (`i < j`) span
`𝔰𝔭𝔦𝔫(k)`.  We prove the Gram identity
`⟨Sᵢⱼ, Sₖₗ⟩ = (d/4) δᵢₖ δⱼₗ` for the real Frobenius product
`⟨A, B⟩ = Re tr(Aᴴ B)`, so `Bᵢⱼ = (2/√d) Sᵢⱼ` is orthonormal, and that
projection onto the real span of any orthonormal family is the unique
Frobenius-closest point. -/

section Spin

variable {N : Type*} [Fintype N] [DecidableEq N] {k : ℕ}

/-- Hermitian Euclidean Clifford relations. -/
structure IsCliffordHerm (γ : Fin k → Matrix N N ℂ) : Prop where
  sq : ∀ i, γ i * γ i = 1
  anti : ∀ i j, i ≠ j → γ i * γ j = -(γ j * γ i)
  herm : ∀ i, (γ i)ᴴ = γ i

variable {γ : Fin k → Matrix N N ℂ}

/-- The bivector generator `Sᵢⱼ = ½ γᵢ γⱼ`. -/
noncomputable def bivector (γ : Fin k → Matrix N N ℂ) (i j : Fin k) :
    Matrix N N ℂ :=
  (2⁻¹ : ℂ) • (γ i * γ j)

/-- Product of two distinct gammas is traceless: `tr(γₐγ_b) = -tr(γ_bγₐ) = -tr(γₐγ_b)`. -/
theorem trace_two_gamma (hγ : IsCliffordHerm γ) {a b : Fin k} (hab : a ≠ b) :
    Matrix.trace (γ a * γ b) = 0 := by
  have h : Matrix.trace (γ a * γ b) = -Matrix.trace (γ a * γ b) := by
    conv_lhs => rw [Matrix.trace_mul_comm, hγ.anti b a (Ne.symm hab), Matrix.trace_neg]
  have h2 : (2 : ℂ) * Matrix.trace (γ a * γ b) = 0 := by linear_combination h
  simpa using h2

/-- Product of four distinct gammas is traceless.  Conjugating by `γₐ` moves it
    past three anticommuting factors, flipping the sign. -/
theorem trace_four_gamma (hγ : IsCliffordHerm γ) {a b c e : Fin k}
    (hab : a ≠ b) (hac : a ≠ c) (hae : a ≠ e) :
    Matrix.trace (γ a * γ b * γ c * γ e) = 0 := by
  -- tr(γₐ γ_b γ_c γ_e) = tr(γ_b γ_c γ_e γₐ)   (cyclicity)
  have hcyc : Matrix.trace (γ a * γ b * γ c * γ e)
      = Matrix.trace (γ b * γ c * γ e * γ a) := by
    rw [show γ a * γ b * γ c * γ e = γ a * (γ b * γ c * γ e) by
          simp only [Matrix.mul_assoc],
        Matrix.trace_mul_comm]
  -- γ_b γ_c γ_e γₐ = -γₐ γ_b γ_c γ_e   (three anticommutations)
  have hmove : γ b * γ c * γ e * γ a = -(γ a * γ b * γ c * γ e) := by
    calc γ b * γ c * γ e * γ a
        = γ b * γ c * (γ e * γ a) := by simp only [Matrix.mul_assoc]
      _ = γ b * γ c * (-(γ a * γ e)) := by rw [hγ.anti e a (Ne.symm hae)]
      _ = -(γ b * (γ c * γ a) * γ e) := by
          simp only [Matrix.mul_neg, Matrix.mul_assoc]
      _ = -(γ b * (-(γ a * γ c)) * γ e) := by rw [hγ.anti c a (Ne.symm hac)]
      _ = (γ b * γ a) * γ c * γ e := by
          simp only [Matrix.mul_neg, Matrix.neg_mul, neg_neg, Matrix.mul_assoc]
      _ = (-(γ a * γ b)) * γ c * γ e := by rw [hγ.anti b a (Ne.symm hab)]
      _ = -(γ a * γ b * γ c * γ e) := by
          simp only [Matrix.neg_mul]
  have h : Matrix.trace (γ a * γ b * γ c * γ e)
      = -Matrix.trace (γ a * γ b * γ c * γ e) := by
    conv_lhs => rw [hcyc, hmove, Matrix.trace_neg]
  have h2 : (2 : ℂ) * Matrix.trace (γ a * γ b * γ c * γ e) = 0 := by
    linear_combination h
  simpa using h2

/-- `Sᵢⱼᴴ Sₖₗ = ¼ γⱼγᵢγₖγₗ`. -/
theorem bivector_conjTranspose_mul (hγ : IsCliffordHerm γ) (i j a b : Fin k) :
    (bivector γ i j)ᴴ * bivector γ a b
      = (4⁻¹ : ℂ) • (γ j * γ i * γ a * γ b) := by
  unfold bivector
  rw [conjTranspose_smul, conjTranspose_mul, hγ.herm i, hγ.herm j,
      Matrix.smul_mul, Matrix.mul_smul, smul_smul]
  congr 1
  · simp [star_inv₀]; norm_num
  · simp only [Matrix.mul_assoc]

/-- **Diagonal Gram entries**: `⟨Sᵢⱼ, Sᵢⱼ⟩ = d/4` with `d = card N`. -/
theorem bivector_gram_diag (hγ : IsCliffordHerm γ) (i j : Fin k) :
    Matrix.trace ((bivector γ i j)ᴴ * bivector γ i j)
      = (Fintype.card N : ℂ) / 4 := by
  rw [bivector_conjTranspose_mul hγ]
  have : γ j * γ i * γ i * γ j = 1 := by
    rw [Matrix.mul_assoc (γ j) (γ i) (γ i), hγ.sq i, Matrix.mul_one, hγ.sq j]
  rw [this, Matrix.trace_smul, Matrix.trace_one, smul_eq_mul]
  ring

/-- **Off-diagonal Gram entries**: for `i < j`, `a < b`, `(i, j) ≠ (a, b)`,
    `tr(Sᵢⱼᴴ S_ab) = 0` (so in particular the real Frobenius product vanishes). -/
theorem bivector_gram_offdiag (hγ : IsCliffordHerm γ) {i j a b : Fin k}
    (hij : i < j) (hab : a < b) (hne : (i, j) ≠ (a, b)) :
    Matrix.trace ((bivector γ i j)ᴴ * bivector γ a b) = 0 := by
  rw [bivector_conjTranspose_mul hγ, Matrix.trace_smul]
  suffices h : Matrix.trace (γ j * γ i * γ a * γ b) = 0 by rw [h, smul_zero]
  have hij' : i ≠ j := ne_of_lt hij
  have hab' : a ≠ b := ne_of_lt hab
  by_cases hia : i = a
  · -- shared first index: γⱼγᵢγᵢγ_b = γⱼγ_b, with j ≠ b
    subst hia
    have hjb : j ≠ b := fun h => hne (by rw [h])
    rw [Matrix.mul_assoc (γ j) (γ i) (γ i), hγ.sq i, Matrix.mul_one]
    exact trace_two_gamma hγ hjb
  by_cases hib : i = b
  · -- i = b, so a < b = i < j, a ≠ j
    subst hib
    have hja : j ≠ a := fun h => by
      subst h; exact absurd (lt_trans hab hij) (lt_irrefl _)
    -- γⱼγᵢγₐγᵢ = -γⱼγₐ
    have : γ j * γ i * γ a * γ i = -(γ j * γ a) := by
      calc γ j * γ i * γ a * γ i
          = γ j * γ i * (γ a * γ i) := by simp only [Matrix.mul_assoc]
        _ = γ j * γ i * (-(γ i * γ a)) := by rw [hγ.anti a i hab']
        _ = -(γ j * (γ i * γ i) * γ a) := by
            simp only [Matrix.mul_neg, Matrix.mul_assoc]
        _ = -(γ j * γ a) := by rw [hγ.sq i, Matrix.mul_one]
    rw [this, Matrix.trace_neg, trace_two_gamma hγ hja, neg_zero]
  by_cases hja : j = a
  · -- i < j = a < b, i ≠ b
    subst hja
    have hib' : i ≠ b := hib
    -- γⱼγᵢγⱼγ_b = -γᵢγⱼγⱼγ_b = -γᵢγ_b
    have : γ j * γ i * γ j * γ b = -(γ i * γ b) := by
      rw [hγ.anti j i (Ne.symm hij'), Matrix.neg_mul, Matrix.neg_mul,
          Matrix.mul_assoc (γ i) (γ j) (γ j), hγ.sq j, Matrix.mul_one]
    rw [this, Matrix.trace_neg, trace_two_gamma hγ hib', neg_zero]
  by_cases hjb : j = b
  · -- shared last index, i ≠ a: tr(γⱼγᵢγₐγⱼ) = tr(γᵢγₐγⱼγⱼ) = tr(γᵢγₐ)
    subst hjb
    have hcyc : Matrix.trace (γ j * γ i * γ a * γ j)
        = Matrix.trace (γ i * γ a * γ j * γ j) := by
      rw [show γ j * γ i * γ a * γ j = γ j * (γ i * γ a * γ j) by
            simp only [Matrix.mul_assoc],
          Matrix.trace_mul_comm]
    rw [hcyc, Matrix.mul_assoc (γ i * γ a) (γ j) (γ j), hγ.sq j, Matrix.mul_one]
    exact trace_two_gamma hγ hia
  · -- four distinct indices
    exact trace_four_gamma hγ (Ne.symm hij') (fun h => hja h) (fun h => hjb h)

/-- **Projection onto the real span of an orthonormal family.**  If
    `{Bₖ}` is orthonormal for `⟨A, B⟩ = Re tr(Aᴴ B)`, the residual of
    `P = Σₖ ⟨Bₖ, l₀⟩ Bₖ` is orthogonal to every real combination `Σⱼ cⱼ Bⱼ`. -/
theorem onb_proj_orthogonal {ι : Type*} [Fintype ι] [DecidableEq ι]
    (B : ι → Matrix N N ℂ)
    (hB : ∀ a b, (Matrix.trace ((B a)ᴴ * B b)).re = if a = b then 1 else 0)
    (l₀ : Matrix N N ℂ) (c : ι → ℝ) :
    (Matrix.trace ((l₀ - ∑ k, ((Matrix.trace ((B k)ᴴ * l₀)).re : ℂ) • B k)ᴴ
        * ∑ j, (c j : ℂ) • B j)).re = 0 := by
  have hsym : ∀ j, (Matrix.trace (l₀ᴴ * B j)).re = (Matrix.trace ((B j)ᴴ * l₀)).re := by
    intro j
    rw [show l₀ᴴ * B j = ((B j)ᴴ * l₀)ᴴ by
          rw [conjTranspose_mul, conjTranspose_conjTranspose],
        Matrix.trace_conjTranspose]
    rfl
  simp only [conjTranspose_sub, conjTranspose_sum, conjTranspose_smul,
    Matrix.sub_mul, Matrix.sum_mul, Matrix.mul_sum, Matrix.smul_mul, Matrix.mul_smul,
    Matrix.trace_sub, Matrix.trace_sum, Matrix.trace_smul, smul_eq_mul,
    Complex.sub_re, Complex.re_sum, Complex.mul_re, Complex.star_def,
    Complex.conj_ofReal, Complex.ofReal_re, Complex.ofReal_im, zero_mul, sub_zero,
    hB, mul_ite, mul_one, mul_zero, hsym]
  simp [Finset.sum_ite_eq']

/-- **Spin projection optimality.**  With `P = Σₖ ⟨Bₖ, l₀⟩ Bₖ` for an
    orthonormal family, `P` is the unique Frobenius-closest element of the
    real span to `l₀`. -/
theorem onb_proj_optimal [Nonempty N] {ι : Type*} [Fintype ι] [DecidableEq ι]
    (B : ι → Matrix N N ℂ)
    (hB : ∀ a b, (Matrix.trace ((B a)ᴴ * B b)).re = if a = b then 1 else 0)
    (l₀ : Matrix N N ℂ) (c : ι → ℝ) :
    let P := ∑ k, ((Matrix.trace ((B k)ᴴ * l₀)).re : ℂ) • B k
    let g := ∑ j, (c j : ℂ) • B j
    frobenius_re_norm (l₀ - P) ≤ frobenius_re_norm (l₀ - g) ∧
      (frobenius_re_norm (l₀ - g) = frobenius_re_norm (l₀ - P) ↔ g = P) := by
  intro P g
  apply frobenius_re_proj_unique_min
  have hdiff : P - g = ∑ j, (((Matrix.trace ((B j)ᴴ * l₀)).re - c j : ℝ) : ℂ) • B j := by
    simp only [P, g, ← Finset.sum_sub_distrib, Complex.ofReal_sub, sub_smul]
  have hre := onb_proj_orthogonal B hB l₀ (fun j => (Matrix.trace ((B j)ᴴ * l₀)).re - c j)
  rw [hdiff]
  apply Complex.ext
  · simp only [Complex.add_re, Complex.star_def, Complex.conj_re, Complex.zero_re]
    linarith
  · simp only [Complex.add_im, Complex.star_def, Complex.conj_im, Complex.zero_im]
    ring

/-- **Orthonormal spin basis** (Proposition S1.1 of the Supplement): `Bᵢⱼ = (2/√d) Sᵢⱼ`
    (`i < j`) is orthonormal for `⟨A, B⟩ = Re tr(Aᴴ B)`, where `d = card N`. -/
theorem spin_basis_orthonormal [Nonempty N] (hγ : IsCliffordHerm γ) {i j a b : Fin k}
    (hij : i < j) (hab : a < b) :
    (Matrix.trace ((((2 / Real.sqrt (Fintype.card N) : ℝ) : ℂ) • bivector γ i j)ᴴ
        * (((2 / Real.sqrt (Fintype.card N) : ℝ) : ℂ) • bivector γ a b))).re
      = if (i, j) = (a, b) then 1 else 0 := by
  rw [conjTranspose_smul, Matrix.smul_mul, Matrix.mul_smul, Matrix.trace_smul,
    Matrix.trace_smul, smul_eq_mul, smul_eq_mul, Complex.star_def, Complex.conj_ofReal]
  split_ifs with h
  · obtain ⟨rfl, rfl⟩ := Prod.mk.inj h
    rw [bivector_gram_diag hγ]
    have hd : (0 : ℝ) < Fintype.card N := by exact_mod_cast Fintype.card_pos
    have hs : Real.sqrt (Fintype.card N) ^ 2 = Fintype.card N := Real.sq_sqrt hd.le
    have hs0 : Real.sqrt (Fintype.card N) ≠ 0 := (Real.sqrt_pos.mpr hd).ne'
    have e : (((2 / Real.sqrt (Fintype.card N) : ℝ) : ℂ)
        * (((2 / Real.sqrt (Fintype.card N) : ℝ) : ℂ) * ((Fintype.card N : ℂ) / 4)))
        = (((2 / Real.sqrt (Fintype.card N)) * ((2 / Real.sqrt (Fintype.card N))
            * ((Fintype.card N : ℝ) / 4)) : ℝ) : ℂ) := by push_cast; ring
    rw [e, Complex.ofReal_re]
    field_simp
    nlinarith [hs]
  · rw [bivector_gram_offdiag hγ hij hab h]; simp

end Spin

/-! ## SE(n) and SE(p,q): blockwise projection in homogeneous coordinates (Table 2)

`𝔰𝔢 = { [a v; 0 0] : a ∈ 𝔰𝔬 }` acting on `n ⊕ o` (with `o` the homogeneous
coordinate).  The projection antisymmetrizes the upper-left block, keeps the
translation column, and zeroes the bottom row. -/

section SE

variable {n o : Type*} [Fintype n] [DecidableEq n] [Nonempty n] [Fintype o] [DecidableEq o]

omit [DecidableEq n] [Nonempty n] [DecidableEq o] in
/-- Trace of a block matrix is the sum of the diagonal-block traces. -/
theorem trace_fromBlocks' {R : Type*} [AddCommMonoid R]
    (A : Matrix n n R) (B : Matrix n o R) (C : Matrix o n R) (D : Matrix o o R) :
    Matrix.trace (fromBlocks A B C D) = Matrix.trace A + Matrix.trace D := by
  simp [Matrix.trace, Fintype.sum_sum_type]

/-- Projection onto `𝔰𝔢(n)`. -/
noncomputable def proj_SE (l₀ : Matrix (n ⊕ o) (n ⊕ o) ℝ) : Matrix (n ⊕ o) (n ⊕ o) ℝ :=
  fromBlocks (proj_SO l₀.toBlocks₁₁) l₀.toBlocks₁₂ 0 0

/-- **Membership**: skew upper-left block, zero bottom row. -/
theorem proj_SE_mem (l₀ : Matrix (n ⊕ o) (n ⊕ o) ℝ) :
    ((proj_SE l₀).toBlocks₁₁)ᵀ = -(proj_SE l₀).toBlocks₁₁ ∧
      (proj_SE l₀).toBlocks₂₁ = 0 ∧ (proj_SE l₀).toBlocks₂₂ = 0 := by
  unfold proj_SE
  simp only [toBlocks_fromBlocks₁₁, toBlocks_fromBlocks₂₁, toBlocks_fromBlocks₂₂,
    and_true]
  exact proj_SO_mem _

/-- **Orthogonality**: the residual is orthogonal to every `[a u; 0 0]`
    with `aᵀ = -a`. -/
theorem proj_SE_orthogonal (l₀ : Matrix (n ⊕ o) (n ⊕ o) ℝ)
    (a : Matrix n n ℝ) (u : Matrix n o ℝ) (ha : aᵀ = -a) :
    Matrix.trace ((l₀ - proj_SE l₀)ᵀ * fromBlocks a u 0 0) = 0 := by
  have hsplit : l₀ - proj_SE l₀
      = fromBlocks (l₀.toBlocks₁₁ - proj_SO l₀.toBlocks₁₁) 0
          l₀.toBlocks₂₁ l₀.toBlocks₂₂ := by
    conv_lhs => rw [← fromBlocks_toBlocks l₀]
    unfold proj_SE
    simp only [toBlocks_fromBlocks₁₁, toBlocks_fromBlocks₁₂]
    ext (i | i) (j | j) <;> simp
  rw [hsplit, fromBlocks_transpose, fromBlocks_multiply, trace_fromBlocks']
  simp only [Matrix.mul_zero, add_zero, Matrix.transpose_zero, Matrix.zero_mul,
    Matrix.trace_zero]
  exact proj_SO_orthogonal _ a ha

/-- **SE(n) optimality**: `proj_SE l₀` is the unique closest element of `𝔰𝔢(n)`. -/
theorem proj_SE_optimal (l₀ : Matrix (n ⊕ o) (n ⊕ o) ℝ)
    (a : Matrix n n ℝ) (u : Matrix n o ℝ) (ha : aᵀ = -a) :
    frobenius_sq_norm (l₀ - proj_SE l₀) ≤ frobenius_sq_norm (l₀ - fromBlocks a u 0 0) ∧
      (frobenius_sq_norm (l₀ - fromBlocks a u 0 0) = frobenius_sq_norm (l₀ - proj_SE l₀) ↔
        fromBlocks a u 0 0 = proj_SE l₀) := by
  apply frobenius_proj_unique_min
  have : proj_SE l₀ - fromBlocks a u 0 0
      = fromBlocks (proj_SO l₀.toBlocks₁₁ - a) (l₀.toBlocks₁₂ - u) 0 0 := by
    unfold proj_SE; ext (i | i) (j | j) <;> simp
  rw [this]
  apply proj_SE_orthogonal
  rw [transpose_sub, proj_SO_mem, ha]; abel

/-- Projection onto `𝔰𝔢(p,q) = { [a v; 0 0] : aᵀη + ηa = 0 }` (e.g. the
    Poincaré algebra for `η = diag(1,-1,-1,-1)`). -/
noncomputable def proj_SEpq (η : Matrix n n ℝ) (l₀ : Matrix (n ⊕ o) (n ⊕ o) ℝ) :
    Matrix (n ⊕ o) (n ⊕ o) ℝ :=
  fromBlocks (proj_Opq η l₀.toBlocks₁₁) l₀.toBlocks₁₂ 0 0

/-- **Membership** for `𝔰𝔢(p,q)`. -/
theorem proj_SEpq_mem (η : Matrix n n ℝ) (hη_symm : ηᵀ = η) (hη_sq : η * η = 1)
    (l₀ : Matrix (n ⊕ o) (n ⊕ o) ℝ) :
    ((proj_SEpq η l₀).toBlocks₁₁)ᵀ * η + η * (proj_SEpq η l₀).toBlocks₁₁ = 0 ∧
      (proj_SEpq η l₀).toBlocks₂₁ = 0 ∧ (proj_SEpq η l₀).toBlocks₂₂ = 0 := by
  unfold proj_SEpq
  simp only [toBlocks_fromBlocks₁₁, toBlocks_fromBlocks₂₁, toBlocks_fromBlocks₂₂,
    and_true]
  exact proj_Opq_mem η _ hη_symm hη_sq

/-- **Orthogonality** for `𝔰𝔢(p,q)`. -/
theorem proj_SEpq_orthogonal (η : Matrix n n ℝ) (hη_symm : ηᵀ = η) (hη_sq : η * η = 1)
    (l₀ : Matrix (n ⊕ o) (n ⊕ o) ℝ)
    (a : Matrix n n ℝ) (u : Matrix n o ℝ) (ha : aᵀ * η + η * a = 0) :
    Matrix.trace ((l₀ - proj_SEpq η l₀)ᵀ * fromBlocks a u 0 0) = 0 := by
  have hsplit : l₀ - proj_SEpq η l₀
      = fromBlocks (l₀.toBlocks₁₁ - proj_Opq η l₀.toBlocks₁₁) 0
          l₀.toBlocks₂₁ l₀.toBlocks₂₂ := by
    conv_lhs => rw [← fromBlocks_toBlocks l₀]
    unfold proj_SEpq
    ext (i | i) (j | j) <;> simp
  rw [hsplit, fromBlocks_transpose, fromBlocks_multiply, trace_fromBlocks']
  simp only [Matrix.mul_zero, add_zero, Matrix.transpose_zero, Matrix.zero_mul,
    Matrix.trace_zero]
  exact proj_Opq_orthogonal η _ a hη_symm hη_sq ha

/-- **SE(p,q) optimality**. -/
theorem proj_SEpq_optimal (η : Matrix n n ℝ) (hη_symm : ηᵀ = η) (hη_sq : η * η = 1)
    (l₀ : Matrix (n ⊕ o) (n ⊕ o) ℝ)
    (a : Matrix n n ℝ) (u : Matrix n o ℝ) (ha : aᵀ * η + η * a = 0) :
    frobenius_sq_norm (l₀ - proj_SEpq η l₀) ≤ frobenius_sq_norm (l₀ - fromBlocks a u 0 0) ∧
      (frobenius_sq_norm (l₀ - fromBlocks a u 0 0) = frobenius_sq_norm (l₀ - proj_SEpq η l₀) ↔
        fromBlocks a u 0 0 = proj_SEpq η l₀) := by
  apply frobenius_proj_unique_min
  have : proj_SEpq η l₀ - fromBlocks a u 0 0
      = fromBlocks (proj_Opq η l₀.toBlocks₁₁ - a) (l₀.toBlocks₁₂ - u) 0 0 := by
    unfold proj_SEpq; ext (i | i) (j | j) <;> simp
  rw [this]
  apply proj_SEpq_orthogonal η hη_symm hη_sq
  have hm := proj_Opq_mem η l₀.toBlocks₁₁ hη_symm hη_sq
  rw [transpose_sub, Matrix.sub_mul, Matrix.mul_sub]
  have hre : (proj_Opq η l₀.toBlocks₁₁)ᵀ * η - aᵀ * η
        + (η * proj_Opq η l₀.toBlocks₁₁ - η * a)
      = ((proj_Opq η l₀.toBlocks₁₁)ᵀ * η + η * proj_Opq η l₀.toBlocks₁₁)
        - (aᵀ * η + η * a) := by abel
  rw [hre, hm, ha, sub_zero]

end SE
