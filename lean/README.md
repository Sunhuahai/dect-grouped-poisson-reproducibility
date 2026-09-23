# Lean 4 verification

This directory contains a deliberately scoped machine-checked companion to the paper. It verifies the algebraic and order-theoretic core of the convergence audit; it does **not** claim that the complete nonsmooth CT convergence theorem has been formalized.

Verified statements:

- tangent majorization plus surrogate descent implies objective descent;
- quantitative/proximal surrogate descent implies the same objective decrease;
- one-step sufficient decrease telescopes over a finite horizon;
- grouped posterior weights sum to one and latent-count allocation conserves counts;
- the scalar complete-data Poisson term is globally minimized at its balance point;
- a positive-count, nonnegative-projection scalar counterexample shows that a safe unit-system SIRT step can decrease log-domain squared distance while increasing the Poisson objective.

There are no `sorry` declarations or custom axioms.

## Build

```bash
cd lean
lake update
lake exe cache get
lake build
```

The toolchain and mathlib tag are pinned to `v4.33.1`. Downloaded dependencies and build outputs in `.lake/` are intentionally excluded from this repository.
