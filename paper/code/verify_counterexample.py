#!/usr/bin/env python3
"""Verify the exact two-bin, two-ray SIRT counterexample in the proof note.

The example uses only Python's standard library.  It checks positive integer
counts, physical count bounds, nonnegative iterates, the standard SIRT
spectral-radius condition, strict LS decrease, and strict Poisson increase.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path


def half_squared_error(value: float, targets: tuple[float, ...]) -> float:
    return 0.5 * sum((value - target) ** 2 for target in targets)


def poisson_objective(value: float, incident: float, counts: tuple[int, ...]) -> float:
    return sum(incident * math.exp(-value) + count * value for count in counts)


def main() -> None:
    # Two singleton energy groups on each of two rays.
    incident_low = 4.0
    incident_high = 1.0
    counts_low = (1, 4)
    counts_high = (1, 1)
    targets_low = (math.log(4.0), 0.0)
    targets_high = (0.0, 0.0)

    x_low_initial = math.log(8.0 / 5.0)
    x_high_initial = 0.0
    relaxation = 1.0

    # A = (1, 1)^T, D_r = I_2, and D_c = 1/2.
    x_low_after = x_low_initial - relaxation * 0.5 * sum(
        x_low_initial - target for target in targets_low
    )
    x_high_after = x_high_initial - relaxation * 0.5 * sum(
        x_high_initial - target for target in targets_high
    )
    lambda_max_h = 1.0

    ls_initial = half_squared_error(x_low_initial, targets_low)
    ls_initial += half_squared_error(x_high_initial, targets_high)
    ls_after = half_squared_error(x_low_after, targets_low)
    ls_after += half_squared_error(x_high_after, targets_high)

    phi_initial = poisson_objective(x_low_initial, incident_low, counts_low)
    phi_initial += poisson_objective(x_high_initial, incident_high, counts_high)
    phi_after = poisson_objective(x_low_after, incident_low, counts_low)
    phi_after += poisson_objective(x_high_after, incident_high, counts_high)

    assert all(isinstance(count, int) and count > 0 for count in counts_low + counts_high)
    assert all(count <= incident_low for count in counts_low)
    assert all(count <= incident_high for count in counts_high)
    assert x_low_initial >= 0.0 and x_high_initial >= 0.0
    assert x_low_after >= 0.0 and x_high_after >= 0.0
    assert math.isclose(x_low_after, math.log(2.0), rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(x_high_after, 0.0, rel_tol=0.0, abs_tol=1e-15)
    assert relaxation < 2.0 / lambda_max_h
    assert ls_after < ls_initial
    assert phi_after > phi_initial

    rows = [
        ("half_squared_error", ls_initial, ls_after, ls_after - ls_initial),
        ("poisson_objective", phi_initial, phi_after, phi_after - phi_initial),
        ("lambda_max_H", lambda_max_h, lambda_max_h, 0.0),
    ]
    output_path = Path(__file__).resolve().parents[1] / "data" / "counterexample.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["metric", "initial", "after_one_sirt", "change"])
        writer.writerows(rows)

    print(f"x_low: {x_low_initial:.12f} -> {x_low_after:.12f}")
    print(f"x_high: {x_high_initial:.12f} -> {x_high_after:.12f}")
    for name, initial, after, change in rows:
        print(f"{name:24s}: {initial:.12f} -> {after:.12f} ({change:+.12f})")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
