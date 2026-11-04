/-
  Formalization of the branch-cut aware method (Algorithm 2) and of its exactness,
  Theorem 2.3(a) of the manuscript "Vector alignment in matrix Lie groups" (proof in
  Section S1.2 of the Supplement), for the groups which preserve a bilinear or
  sesquilinear form,
    G_M = { g : gᴴ M g = M },      𝔤_M = { A : Aᴴ M + M A = 0 },
  where ᴴ is the conjugate transpose for a `StarRing` scalar field (the plain transpose
  when the star is trivial, as for ℝ).

  What is proved here, with no `sorry` and no axioms:

  * `genEigen_orthogonal`: if g ∈ G_M, x is a generalized eigenvector of g for λ, y one
    for μ, and star λ * μ ≠ 1, then xᴴ M y = 0.  Appendix B of the manuscript states
    this for eigenvectors, Eq. (eigorth); the Supplement's proof of Theorem 2.3(a)
    reaches the same conclusion by writing the spectral projector as a polynomial in g.
    Here it is proved directly by induction on the heights of the Jordan chains.
  * `involution_mem` : if P is an idempotent which is M-self-adjoint (Pᴴ M = M P), then
    r = 1 - 2P satisfies r² = 1 and r ∈ G_M (Theorem 2.2 of the manuscript).
    `genEigen_orthogonal` is what makes the spectral projector onto N, the sum of the
    generalized eigenspaces for Re λ < 0 (Eq. (involution), Section 2.5), M-self-adjoint.
  * `mul_mem` : G_M is closed under products, so r g ∈ G_M.
  * `rg_eigen_flip`, `rg_eigen_keep` : with P commuting with g, an eigenvector of r g in
    the range of P is an eigenvector of g with the opposite eigenvalue, and one in the
    kernel of P has the same eigenvalue; this is why r g has no negative real eigenvalue.
  * `log_mem_algebra` : any logarithm which commutes with the conjugate transpose, with
    similarity and with inversion maps G_M into 𝔤_M.
  * `proj_fixes_algebra` : the projection ½(L - M⁻¹ Lᴴ M) of Theorem 2.1 fixes 𝔤_M.
  * `algorithm_exact` : assembling the above, r · exp(proj(log(r g))) = g for g ∈ G_M.

  What is assumed, and stated as the interface `LogSpec`: a pair (exp, log) with
  exp (log h) = h on a domain D closed under inversion, such that log commutes with the
  conjugate transpose, with similarity and with inversion.  For K = ℂ and D the invertible
  matrices with no eigenvalue on the negative real axis, the principal logarithm has these
  properties (Higham, Functions of Matrices, Theorems 1.13, 1.31 and 11.2); Mathlib has
  no matrix logarithm, so these facts are not formalized.  The decomposition of Kⁿ into
  generalized eigenspaces, which defines the spectral projector P (computed from the
  Schur form by Algorithm 3 of the manuscript), is likewise taken as given through the
  hypotheses `P * P = P`, `Pᴴ * M = M * P` and `g * P = P * g`.
-/

import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.Tactic.NoncommRing
import Mathlib.Tactic.LinearCombination

open Matrix
open scoped Matrix

set_option linter.unusedSectionVars false
set_option linter.unusedSimpArgs false

namespace FastLieAlignGlobal

variable {n : Type*} [Fintype n] [DecidableEq n]
variable {K : Type*} [Field K] [StarRing K]

/-- `g` preserves the form `M`. -/
def InFormGroup (M g : Matrix n n K) : Prop := gᴴ * M * g = M

/-- `A` lies in the Lie algebra of the form group of `M`. -/
def InFormAlgebra (M A : Matrix n n K) : Prop := Aᴴ * M + M * A = 0

/-! ## Products and involutions -/

theorem mul_mem {M r g : Matrix n n K} (hr : InFormGroup M r) (hg : InFormGroup M g) :
    InFormGroup M (r * g) := by
  unfold InFormGroup at *
  rw [conjTranspose_mul]
  calc gᴴ * rᴴ * M * (r * g) = gᴴ * (rᴴ * M * r) * g := by noncomm_ring
    _ = gᴴ * M * g := by rw [hr]
    _ = M := hg

/-- The involution built from an `M`-self-adjoint idempotent `P`: `r = 1 - 2P`. -/
def involution (P : Matrix n n K) : Matrix n n K := 1 - (P + P)

theorem involution_sq (P : Matrix n n K) (hP : P * P = P) :
    involution P * involution P = 1 := by
  unfold involution
  have h : (1 - (P + P)) * (1 - (P + P))
      = 1 - (P + P) - (P + P) + (P * P + P * P + (P * P + P * P)) := by noncomm_ring
  rw [h, hP]; abel

theorem involution_mem (M P : Matrix n n K) (hP : P * P = P) (hadj : Pᴴ * M = M * P) :
    InFormGroup M (involution P) := by
  unfold InFormGroup involution
  have hT : (1 - (P + P))ᴴ = 1 - (Pᴴ + Pᴴ) := by
    simp [conjTranspose_sub, conjTranspose_add, conjTranspose_one]
  rw [hT]
  have h : (1 - (Pᴴ + Pᴴ)) * M * (1 - (P + P))
      = M - (Pᴴ * M + Pᴴ * M) - (M * P + M * P)
        + (Pᴴ * M * P + Pᴴ * M * P + (Pᴴ * M * P + Pᴴ * M * P)) := by noncomm_ring
  rw [h, hadj, Matrix.mul_assoc M P P, hP]; abel

/-! ## Eigenvalues of `r g` -/

theorem involution_comm {P g : Matrix n n K} (hPg : g * P = P * g) :
    g * involution P = involution P * g := by
  unfold involution
  rw [Matrix.mul_sub, Matrix.sub_mul, Matrix.mul_one, Matrix.one_mul, Matrix.mul_add,
    Matrix.add_mul, hPg]

/-- An eigenvector of `r g` in the range of `P` is an eigenvector of `g` for the opposite
eigenvalue. -/
theorem rg_eigen_flip {P g : Matrix n n K} (hPg : g * P = P * g) {x : n → K} {ν : K}
    (hx : P *ᵥ x = x) (hν : (involution P * g) *ᵥ x = ν • x) : g *ᵥ x = (-ν) • x := by
  have hrx : involution P *ᵥ x = -x := by
    unfold involution
    rw [Matrix.sub_mulVec, Matrix.add_mulVec, Matrix.one_mulVec, hx]; abel
  have : (involution P * g) *ᵥ x = -(g *ᵥ x) := by
    rw [← involution_comm hPg, ← Matrix.mulVec_mulVec, hrx, Matrix.mulVec_neg]
  rw [this] at hν
  rw [neg_smul, ← hν, neg_neg]

/-- An eigenvector of `r g` in the kernel of `P` is an eigenvector of `g` for the same
eigenvalue. -/
theorem rg_eigen_keep {P g : Matrix n n K} (hPg : g * P = P * g) {x : n → K} {ν : K}
    (hx : P *ᵥ x = 0) (hν : (involution P * g) *ᵥ x = ν • x) : g *ᵥ x = ν • x := by
  have hrx : involution P *ᵥ x = x := by
    unfold involution
    rw [Matrix.sub_mulVec, Matrix.add_mulVec, Matrix.one_mulVec, hx]; abel
  rw [← involution_comm hPg, ← Matrix.mulVec_mulVec, hrx] at hν
  exact hν

/-! ## Orthogonality of generalized eigenspaces of a form preserving matrix -/

theorem star_smul_vec (a : K) (x : n → K) : star (a • x) = star a • star x := by
  ext i; simp [star_mul']

theorem star_add_vec (x y : n → K) : star (x + y) = star x + star y := by
  ext i; simp

/-- For `g ∈ G_M`, the form is invariant: `(g x)ᴴ M (g y) = xᴴ M y`. -/
theorem form_invariant {M g : Matrix n n K} (hg : InFormGroup M g) (x y : n → K) :
    star (g *ᵥ x) ⬝ᵥ (M *ᵥ (g *ᵥ y)) = star x ⬝ᵥ (M *ᵥ y) := by
  rw [Matrix.star_mulVec, Matrix.mulVec_mulVec, ← Matrix.dotProduct_mulVec,
    Matrix.mulVec_mulVec, ← Matrix.mul_assoc]
  unfold InFormGroup at hg
  rw [hg]

/-- If `g` preserves `M`, `(g - λ)^k x = 0`, `(g - μ)^l y = 0` and `star λ * μ ≠ 1`,
then `xᴴ M y = 0`.  This is Eq. (eigorth) of Appendix B of the manuscript extended from
eigenvectors to Jordan chains, by induction on the sum of the heights `k + l`. -/
theorem genEigen_orthogonal {M g : Matrix n n K} (hg : InFormGroup M g) {lam mu : K}
    (hlm : star lam * mu ≠ 1) :
    ∀ (k l : ℕ) (x y : n → K), ((g - lam • 1) ^ k) *ᵥ x = 0 → ((g - mu • 1) ^ l) *ᵥ y = 0 →
      star x ⬝ᵥ (M *ᵥ y) = 0 := by
  intro k
  induction k with
  | zero =>
    intro l x y hx _
    rw [pow_zero, Matrix.one_mulVec] at hx
    have h0 : star (0 : n → K) = 0 := by ext i; simp
    rw [hx, h0, zero_dotProduct]
  | succ k ihk =>
    intro l
    induction l with
    | zero =>
      intro x y _ hy
      rw [pow_zero, Matrix.one_mulVec] at hy
      rw [hy, Matrix.mulVec_zero, dotProduct_zero]
    | succ l ihl =>
      intro x y hx hy
      -- the next vectors in the Jordan chains
      set x' := (g - lam • 1) *ᵥ x with hx'def
      set y' := (g - mu • 1) *ᵥ y with hy'def
      have hx' : ((g - lam • 1) ^ k) *ᵥ x' = 0 := by
        rw [hx'def, Matrix.mulVec_mulVec, ← pow_succ]; exact hx
      have hy' : ((g - mu • 1) ^ l) *ᵥ y' = 0 := by
        rw [hy'def, Matrix.mulVec_mulVec, ← pow_succ]; exact hy
      have hgx : g *ᵥ x = lam • x + x' := by
        rw [hx'def, Matrix.sub_mulVec, Matrix.smul_mulVec, Matrix.one_mulVec]; abel
      have hgy : g *ᵥ y = mu • y + y' := by
        rw [hy'def, Matrix.sub_mulVec, Matrix.smul_mulVec, Matrix.one_mulVec]; abel
      -- the three inner products with a shorter chain vanish by induction
      have h1 : star x ⬝ᵥ (M *ᵥ y') = 0 := ihl x y' hx hy'
      have h2 : star x' ⬝ᵥ (M *ᵥ y) = 0 := ihk (l + 1) x' y hx' hy
      have h3 : star x' ⬝ᵥ (M *ᵥ y') = 0 := ihk l x' y' hx' hy'
      -- invariance of the form
      have hinv := form_invariant hg x y
      rw [hgx, hgy, star_add_vec, star_smul_vec, Matrix.mulVec_add, Matrix.mulVec_smul] at hinv
      simp only [add_dotProduct, dotProduct_add, smul_dotProduct, dotProduct_smul, h1, h2, h3,
        smul_eq_mul, mul_zero, add_zero, zero_add] at hinv
      -- hinv : star lam * (mu * a) = a  with  a = xᴴ M y
      have hmain : (1 - star lam * mu) * (star x ⬝ᵥ (M *ᵥ y)) = 0 := by
        linear_combination -hinv
      rcases mul_eq_zero.mp hmain with h | h
      · exact absurd (sub_eq_zero.mp h).symm hlm
      · exact h

/-! ## The logarithm interface and the projection -/

/-- A pair `(exp, log)` with the properties of the principal logarithm that the proof of
Theorem 2.3(a) uses: `exp ∘ log = id` on a domain `D` closed under inversion, and `log`
commutes with the conjugate transpose, with similarity and with inversion. -/
structure LogSpec (D : Matrix n n K → Prop) where
  exp : Matrix n n K → Matrix n n K
  log : Matrix n n K → Matrix n n K
  exp_log : ∀ h, D h → exp (log h) = h
  D_inv : ∀ h, D h → D h⁻¹
  log_conjTranspose : ∀ h, D h → log hᴴ = (log h)ᴴ
  log_inv : ∀ h, D h → log h⁻¹ = -log h
  log_similar : ∀ h S, D h → IsUnit S.det → log (S * h * S⁻¹) = S * log h * S⁻¹

/-- For `h ∈ G_M` with `M` invertible: `hᴴ = M h⁻¹ M⁻¹`. -/
theorem conjTranspose_eq_of_mem {M h : Matrix n n K} (hM : IsUnit M.det)
    (hh : IsUnit h.det) (hg : InFormGroup M h) : hᴴ = M * h⁻¹ * M⁻¹ := by
  unfold InFormGroup at hg
  calc hᴴ = hᴴ * M * h * (h⁻¹ * M⁻¹) := by
          rw [Matrix.mul_assoc (hᴴ * M), ← Matrix.mul_assoc h, Matrix.mul_nonsing_inv h hh,
            Matrix.one_mul, Matrix.mul_assoc, Matrix.mul_nonsing_inv M hM, Matrix.mul_one]
    _ = M * h⁻¹ * M⁻¹ := by rw [hg, Matrix.mul_assoc]

/-- A logarithm with the properties of `LogSpec` maps `G_M` into `𝔤_M`. -/
theorem log_mem_algebra {D : Matrix n n K → Prop} (L : LogSpec D) {M h : Matrix n n K}
    (hM : IsUnit M.det) (hh : IsUnit h.det) (hD : D h) (hg : InFormGroup M h) :
    InFormAlgebra M (L.log h) := by
  unfold InFormAlgebra
  have h1 : (L.log h)ᴴ = -(M * L.log h * M⁻¹) := by
    rw [← L.log_conjTranspose h hD, conjTranspose_eq_of_mem hM hh hg,
      L.log_similar h⁻¹ M (L.D_inv h hD) hM, L.log_inv h hD]
    simp [Matrix.mul_neg, Matrix.neg_mul]
  rw [h1, Matrix.neg_mul, Matrix.mul_assoc (M * L.log h), Matrix.nonsing_inv_mul M hM,
    Matrix.mul_one]
  abel

/-- The projection onto `𝔤_M`: `½ (L - M⁻¹ Lᴴ M)`, Theorem 2.1 of the manuscript. -/
noncomputable def proj (M L : Matrix n n K) : Matrix n n K := (2 : K)⁻¹ • (L - M⁻¹ * Lᴴ * M)

theorem proj_fixes_algebra [CharZero K] {M L : Matrix n n K} (hM : IsUnit M.det)
    (hL : InFormAlgebra M L) : proj M L = L := by
  unfold InFormAlgebra at hL
  have h : M⁻¹ * Lᴴ * M = -L := by
    have : Lᴴ * M = -(M * L) := eq_neg_of_add_eq_zero_left hL
    rw [Matrix.mul_assoc, this, Matrix.mul_neg, ← Matrix.mul_assoc,
      Matrix.nonsing_inv_mul M hM, Matrix.one_mul]
  unfold proj
  rw [h, sub_neg_eq_add, ← two_smul K L, smul_smul,
    inv_mul_cancel₀ (two_ne_zero' K), one_smul]

/-! ## Exactness of Algorithm 2 on the form groups (Theorem 2.3(a)) -/

/-- Algorithm 2 on exact data, written for the pair `(exp, log)` of `LogSpec`.  Given
`g ∈ G_M`, an `M`-self-adjoint idempotent `P` whose involution `r = 1 - 2P` makes `r g` fall
in the domain `D` of the logarithm, the output `r · exp(proj(log(r g)))` equals `g`.  The
hypotheses on `P` are supplied by the spectral projector onto `N`, the sum of the
generalized eigenspaces for the eigenvalues with `Re λ < 0` (Eq. (involution)), which is
`M`-self-adjoint by `genEigen_orthogonal`, since `star λ * μ = 1` with `Re λ < 0` forces
`Re μ < 0`. -/
theorem algorithm_exact [CharZero K] {D : Matrix n n K → Prop} (L : LogSpec D)
    {M g P : Matrix n n K} (hM : IsUnit M.det) (hg : InFormGroup M g)
    (hP : P * P = P) (hadj : Pᴴ * M = M * P)
    (hD : D (involution P * g)) (hunit : IsUnit (involution P * g).det) :
    involution P * L.exp (proj M (L.log (involution P * g))) = g := by
  have hrg : InFormGroup M (involution P * g) := mul_mem (involution_mem M P hP hadj) hg
  have hlog : InFormAlgebra M (L.log (involution P * g)) := log_mem_algebra L hM hunit hD hrg
  rw [proj_fixes_algebra hM hlog, L.exp_log _ hD, ← Matrix.mul_assoc, involution_sq P hP,
    Matrix.one_mul]

/-- When `P = 0` the involution is the identity and Algorithm 2 is the naive method,
Algorithm 1: `exp(proj(log g)) = g`, the exactness statement of Section 2.3. -/
theorem basic_exact [CharZero K] {D : Matrix n n K → Prop} (L : LogSpec D)
    {M g : Matrix n n K} (hM : IsUnit M.det) (hg : InFormGroup M g)
    (hD : D g) (hunit : IsUnit g.det) :
    L.exp (proj M (L.log g)) = g := by
  have h := algorithm_exact L hM hg (P := 0) (by simp) (by simp)
    (by simpa [involution] using hD) (by simpa [involution] using hunit)
  simpa [involution] using h

/-- Reassembly with a scalar factor: if `exp A = ζ⁻¹ g` then `ζ · exp A = g` (not used by the final algorithm, whose complex-group factor is a reflection). -/
theorem scalar_reassembly {E : Matrix n n K → Matrix n n K} {A g : Matrix n n K} {ζ : K}
    (hζ : ζ ≠ 0) (h : E A = ζ⁻¹ • g) : ζ • E A = g := by
  rw [h, smul_smul, mul_inv_cancel₀ hζ, one_smul]

end FastLieAlignGlobal
