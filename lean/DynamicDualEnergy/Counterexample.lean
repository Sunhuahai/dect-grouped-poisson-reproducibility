import DynamicDualEnergy.ScalarPoisson
import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# Machine-checked mismatch between log-domain LS and Poisson descent

For the scalar Poisson term with `I = exp 1` and `z = 1`, the exact balance
point is `p⋆ = 1`. A unit-system SIRT step with relaxation `11/6` moves from
`p = 11/5` to `p = 0`. The relaxation is in the standard safe interval
`(0, 2)`, and the step decreases squared distance to `p⋆`, yet strictly
increases the Poisson term. Both projections are nonnegative and the latent
count is smaller than the incident intensity.
-/

namespace DynamicDualEnergy

noncomputable section

private theorem exp_neg_six_fifths_lt_half :
    Real.exp (-6 / 5 : ℝ) < 1 / 2 := by
  have hmono : Real.exp (-6 / 5 : ℝ) < Real.exp (-1) :=
    Real.exp_lt_exp.mpr (by norm_num)
  exact lt_trans hmono Real.exp_neg_one_lt_half

private theorem twenty_seven_tenths_lt_exp_one :
    (27 / 10 : ℝ) < Real.exp 1 := by
  exact lt_trans (by norm_num) Real.exp_one_gt_d9

/-- Explicit counterexample: LS target distance decreases, Poisson cost increases. -/
theorem ls_decreases_but_poisson_increases :
    ((0 : ℝ) - 1) ^ 2 < ((11 / 5 : ℝ) - 1) ^ 2 ∧
      scalarPoisson (Real.exp 1) 1 (11 / 5) <
        scalarPoisson (Real.exp 1) 1 0 := by
  constructor
  · norm_num
  · have hold :
        scalarPoisson (Real.exp 1) 1 (11 / 5) =
          Real.exp (-6 / 5) + 11 / 5 := by
      rw [scalarPoisson]
      norm_num
      rw [← Real.exp_add]
      congr 2
      norm_num
    have hnew : scalarPoisson (Real.exp 1) 1 0 = Real.exp 1 := by
      simp [scalarPoisson]
    rw [hold, hnew]
    nlinarith [exp_neg_six_fifths_lt_half, twenty_seven_tenths_lt_exp_one]

/-- The scalar target `p⋆ = 1` is indeed the exact Poisson balance point. -/
theorem one_is_balance_point :
    Real.exp 1 * Real.exp (-1) = (1 : ℝ) := by
  rw [← Real.exp_add]
  norm_num

/-- The scalar unit-system relaxation `11/6` lies in the safe SIRT interval. -/
theorem scalar_sirt_relaxation_is_safe :
    (0 : ℝ) < 11 / 6 ∧ (11 / 6 : ℝ) < 2 := by
  norm_num

/-- With `A = D_r = D_c = 1`, the safe SIRT step maps `11/5` to `0`. -/
theorem scalar_safe_sirt_step :
    (11 / 5 : ℝ) - (11 / 6) * ((11 / 5) - 1) = 0 := by
  norm_num

end

end DynamicDualEnergy
