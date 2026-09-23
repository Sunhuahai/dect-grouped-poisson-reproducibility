#!/usr/bin/env python3
"""Generate and verify the scalar illustrations used by the manuscript.

Only the Python standard library is required. The script checks the local
Taylor remainder, the uniform- and heterogeneous-count solution regimes, and
monotone descent of exact proximal Poisson steps.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def write_csv(name: str, fieldnames: list[str], rows: list[dict[str, float]]) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def poisson_objective(x: float, incident: tuple[float, ...], counts: tuple[float, ...]) -> float:
    return sum(value * math.exp(-x) for value in incident) + sum(counts) * x


def poisson_gradient(x: float, incident: tuple[float, ...], counts: tuple[float, ...]) -> float:
    return -sum(incident) * math.exp(-x) + sum(counts)


def proximal_poisson_step(
    previous: float,
    incident: tuple[float, ...],
    counts: tuple[float, ...],
    rho: float,
) -> float:
    """Solve the strictly convex scalar proximal M-step by safeguarded Newton."""
    x = previous
    for _ in range(100):
        exp_term = sum(incident) * math.exp(-x)
        gradient = -exp_term + sum(counts) + rho * (x - previous)
        hessian = exp_term + rho
        candidate = x - gradient / hessian
        if candidate < 0.0:
            candidate = 0.5 * x
        if abs(candidate - x) <= 1e-14 * max(1.0, abs(x)):
            return candidate
        x = candidate
    raise RuntimeError("proximal Newton iteration did not converge")


def trajectory(
    incident: tuple[float, float],
    counts: tuple[float, float],
    rho: float,
    iterations: int,
) -> tuple[list[float], float, float, float]:
    targets = tuple(math.log(value / count) for value, count in zip(incident, counts))
    ls_solution = sum(targets) / len(targets)
    weighted_ls_solution = sum(
        count * target for count, target in zip(counts, targets)
    ) / sum(counts)
    poisson_solution = math.log(sum(incident) / sum(counts))
    values = [0.0]
    for _ in range(iterations):
        values.append(proximal_poisson_step(values[-1], incident, counts, rho))
    return values, ls_solution, weighted_ls_solution, poisson_solution


def main() -> None:
    local_rows: list[dict[str, float]] = []
    for index in range(-24, 25):
        delta = index / 20.0
        local_rows.append(
            {
                "delta": delta,
                "exact": math.exp(-delta) - 1.0 + delta,
                "quadratic": 0.5 * delta * delta,
            }
        )
    write_csv("local_quadratic.csv", ["delta", "exact", "quadratic"], local_rows)

    for delta in (1e-1, 5e-2, 2.5e-2):
        remainder = abs(math.exp(-delta) - 1.0 + delta - 0.5 * delta * delta)
        assert remainder / delta**3 < 0.18

    rho = 10.0
    iterations = 12
    uniform = ((1000.0, 1000.0), (900.0, 880.0))
    heterogeneous = ((100.0, 100.0), (90.0, 10.0))
    u_values, u_ls, u_wls, u_star = trajectory(*uniform, rho, iterations)
    h_values, h_ls, h_wls, h_star = trajectory(*heterogeneous, rho, iterations)

    assert abs(u_ls - u_star) < 2e-4
    assert abs(u_wls - u_star) < 1e-4
    assert abs(h_ls - h_star) > 0.4
    assert abs(h_wls - h_star) > 0.3
    assert abs(h_values[-1] - h_star) < 1e-10

    trajectory_rows: list[dict[str, float]] = []
    for iteration in range(iterations + 1):
        trajectory_rows.append(
            {
                "iteration": iteration,
                "ls_uniform": 0.0 if iteration == 0 else u_ls,
                "wls_uniform": 0.0 if iteration == 0 else u_wls,
                "poisson_uniform": u_values[iteration],
                "ls_heterogeneous": 0.0 if iteration == 0 else h_ls,
                "wls_heterogeneous": 0.0 if iteration == 0 else h_wls,
                "poisson_heterogeneous": h_values[iteration],
            }
        )
    write_csv("toy_trajectories.csv", list(trajectory_rows[0]), trajectory_rows)

    h_incident, h_counts = heterogeneous
    optimum = poisson_objective(h_star, h_incident, h_counts)
    diagnostic_rows: list[dict[str, float]] = []
    previous_objective = poisson_objective(h_values[0], h_incident, h_counts)
    for iteration in range(1, iterations + 1):
        value = h_values[iteration]
        objective = poisson_objective(value, h_incident, h_counts)
        assert objective <= previous_objective + 1e-12
        diagnostic_rows.append(
            {
                "iteration": iteration,
                "objective_gap": max(objective - optimum, 1e-16),
                "step": max(abs(value - h_values[iteration - 1]), 1e-16),
                "residual": max(abs(poisson_gradient(value, h_incident, h_counts)), 1e-16),
            }
        )
        previous_objective = objective
    write_csv("poisson_diagnostics.csv", list(diagnostic_rows[0]), diagnostic_rows)

    print("verified local Taylor relation and proximal Poisson descent")
    print(f"uniform limits: LS={u_ls:.9f}, WLS={u_wls:.9f}, Poisson={u_star:.9f}")
    print(f"heterogeneous limits: LS={h_ls:.9f}, WLS={h_wls:.9f}, Poisson={h_star:.9f}")
    for name in ("local_quadratic.csv", "toy_trajectories.csv", "poisson_diagnostics.csv"):
        print(f"wrote {DATA / name}")


if __name__ == "__main__":
    main()
