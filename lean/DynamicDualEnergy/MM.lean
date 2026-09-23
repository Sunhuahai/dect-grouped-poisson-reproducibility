import Mathlib

/-!
# Abstract majorization--minimization descent

A tangent upper surrogate that is decreased at each step also decreases the
original objective. These lemmas are independent of CT-specific details.
-/

namespace DynamicDualEnergy

/-- `surrogate x anchor` majorizes `objective x` and is tangent at `anchor`. -/
structure Majorizer {α : Type*} (objective : α → ℝ)
    (surrogate : α → α → ℝ) : Prop where
  upper : ∀ x anchor, objective x ≤ surrogate x anchor
  tangent : ∀ anchor, surrogate anchor anchor = objective anchor

/-- Fundamental MM descent: surrogate descent transfers to objective descent. -/
theorem objective_descent_of_surrogate_descent
    {α : Type*} {objective : α → ℝ} {surrogate : α → α → ℝ}
    (hmajor : Majorizer objective surrogate) {current next : α}
    (hstep : surrogate next current ≤ surrogate current current) :
    objective next ≤ objective current := by
  calc
    objective next ≤ surrogate next current := hmajor.upper next current
    _ ≤ surrogate current current := hstep
    _ = objective current := hmajor.tangent current

/-- A quantitative surrogate decrease transfers verbatim to the objective. -/
theorem sufficient_objective_decrease
    {α : Type*} {objective : α → ℝ} {surrogate : α → α → ℝ}
    (hmajor : Majorizer objective surrogate) {current next : α} {δ : ℝ}
    (hstep : surrogate next current + δ ≤ surrogate current current) :
    objective next + δ ≤ objective current := by
  calc
    objective next + δ ≤ surrogate next current + δ := by
      linarith [hmajor.upper next current]
    _ ≤ surrogate current current := hstep
    _ = objective current := hmajor.tangent current

/-- The standard proximal-MM sufficient-decrease implication. -/
theorem proximal_sufficient_decrease
    {E : Type*} [NormedAddCommGroup E]
    {objective : E → ℝ} {surrogate : E → E → ℝ}
    (hmajor : Majorizer objective surrogate) {current next : E} {ρ : ℝ}
    (hstep : surrogate next current + (ρ / 2) * ‖next - current‖ ^ 2 ≤
      surrogate current current) :
    objective next + (ρ / 2) * ‖next - current‖ ^ 2 ≤ objective current := by
  exact sufficient_objective_decrease hmajor hstep

end DynamicDualEnergy
