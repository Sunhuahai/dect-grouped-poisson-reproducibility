import Mathlib

/-!
# Scalar complete-data Poisson term

For incident intensity `I`, latent count `z`, and line integral `p`, the
complete-data contribution is `I * exp (-p) + z * p`. If the balance equation
`I * exp (-p⋆) = z` holds and `z ≥ 0`, then `p⋆` is a global minimizer.
-/

namespace DynamicDualEnergy

noncomputable section

/-- Scalar complete-data negative log-likelihood, up to additive constants. -/
def scalarPoisson (I z p : ℝ) : ℝ := I * Real.exp (-p) + z * p

/-- The balance point is a global minimizer of the scalar Poisson term. -/
theorem scalarPoisson_min_at_balance
    {I z pstar : ℝ} (hz : 0 ≤ z)
    (hbalance : I * Real.exp (-pstar) = z) (p : ℝ) :
    scalarPoisson I z pstar ≤ scalarPoisson I z p := by
  let δ : ℝ := p - pstar
  have hbasic : z * (1 - δ) ≤ z * Real.exp (-δ) :=
    mul_le_mul_of_nonneg_left (Real.one_sub_le_exp_neg δ) hz
  have hexp : Real.exp (-p) = Real.exp (-pstar) * Real.exp (-δ) := by
    rw [← Real.exp_add]
    congr 1
    dsimp [δ]
    ring
  have hIp : I * Real.exp (-p) = z * Real.exp (-δ) := by
    calc
      I * Real.exp (-p) = I * (Real.exp (-pstar) * Real.exp (-δ)) := by rw [hexp]
      _ = (I * Real.exp (-pstar)) * Real.exp (-δ) := by ring
      _ = z * Real.exp (-δ) := by rw [hbalance]
  have hstep : z + z * pstar ≤ z * Real.exp (-δ) + z * p := by
    calc
      z + z * pstar = z * (1 - δ) + z * p := by
        dsimp [δ]
        ring
      _ ≤ z * Real.exp (-δ) + z * p := by
        simpa [add_comm] using add_le_add_right hbasic (z * p)
  calc
    scalarPoisson I z pstar = z + z * pstar := by
      rw [scalarPoisson, hbalance]
    _ ≤ z * Real.exp (-δ) + z * p := hstep
    _ = scalarPoisson I z p := by
      rw [scalarPoisson, hIp]

end

end DynamicDualEnergy
