import Mathlib

/-!
# Grouped-Poisson latent-count allocation

For a finite group of predicted component intensities, the normalized weights
sum to one and the allocated latent counts conserve the observed group count.
-/

namespace DynamicDualEnergy
namespace LatentAllocation

noncomputable section

variable {ι : Type*} [Fintype ι]

/-- Total predicted intensity in one dynamic energy group. -/
def total (q : ι → ℝ) : ℝ := ∑ i, q i

/-- Posterior component weight inside one observed group. -/
def weight (q : ι → ℝ) (i : ι) : ℝ := q i / total q

/-- The posterior weights form a partition of unity if the denominator is nonzero. -/
theorem sum_weight_eq_one (q : ι → ℝ) (htotal : total q ≠ 0) :
    ∑ i, weight q i = 1 := by
  simp only [weight]
  rw [← Finset.sum_div]
  exact div_self htotal

/-- Expected latent component counts add back to the observed grouped count. -/
theorem allocated_count_conservation (q : ι → ℝ) (observed : ℝ)
    (htotal : total q ≠ 0) :
    ∑ i, observed * weight q i = observed := by
  rw [← Finset.mul_sum]
  rw [sum_weight_eq_one q htotal]
  ring

end

end LatentAllocation
end DynamicDualEnergy
