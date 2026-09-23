import Mathlib

/-!
# Finite-horizon consequences of sufficient decrease

The infinite-series statement in the manuscript follows by combining these
finite telescoping inequalities with a lower bound on the objective.
-/

namespace DynamicDualEnergy

/-- Telescope a one-step descent inequality over the first `N` iterations. -/
theorem telescoping_descent
    (energy penalty : ℕ → ℝ)
    (hstep : ∀ n, energy (n + 1) + penalty n ≤ energy n) :
    ∀ N, (Finset.range N).sum penalty ≤ energy 0 - energy N := by
  intro N
  induction N with
  | zero => simp
  | succ N ih =>
      rw [Finset.sum_range_succ]
      have hN : energy (Nat.succ N) + penalty N ≤ energy N := by
        simpa [Nat.succ_eq_add_one] using hstep N
      linarith

/-- A lower bound on the energy uniformly bounds every penalty partial sum. -/
theorem penalty_partial_sum_le
    (energy penalty : ℕ → ℝ) (lower : ℝ)
    (hstep : ∀ n, energy (n + 1) + penalty n ≤ energy n)
    (hlower : ∀ n, lower ≤ energy n) (N : ℕ) :
    (Finset.range N).sum penalty ≤ energy 0 - lower := by
  have htel := telescoping_descent energy penalty hstep N
  have hlow := hlower N
  linarith

end DynamicDualEnergy
