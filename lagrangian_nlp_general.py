"""
General numeric NLPP solver for **five** decision variables.

Implements equality-constrained smooth optimization via SciPy — the KKT
stationarity conditions are satisfied **implicitly** by the SLSQP / trust-constr
solution of the constrained minimization problem (same mathematical object as
hand-derived Lagrange multiplier systems, without symbolic algebra).

Example: minimize :math:`\\sum_i x_i^2` subject to :math:`\\sum_i x_i = 1` has
solution :math:`x_i = 1/5` (symmetry / KKT).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.optimize import minimize

__all__ = ["solve_lagrangian_nlp_5d", "N_DIM"]

N_DIM = 5


def solve_lagrangian_nlp_5d(
    objective: Callable[[np.ndarray], float],
    equality_constraints: Sequence[Callable[[np.ndarray], float]],
    x0: np.ndarray | Sequence[float],
    *,
    maximize: bool = False,
    bounds: Sequence[tuple[float | None, float | None]] | None = None,
    method: str = "SLSQP",
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Solve a smooth NLPP in **exactly five** variables with **equality** constraints.

    This is the practical “Lagrangian / KKT” solver for problems where you do not
    derive :math:`\\nabla f = \\sum_j \\lambda_j \\nabla g_j` by hand: SciPy finds
    a point satisfying the constrained first-order conditions (up to tolerances).

    Parameters
    ----------
    objective
        Maps ``x`` (length-5 ``float64`` vector) to scalar objective value.
    equality_constraints
        Each callable must return **zero** at feasibility (e.g. ``g(x)==0``).
        Typical NLPP has one constraint; pass ``[g]`` or several ``[g1, g2, ...]``.
    x0
        Initial guess, length 5.
    maximize
        If True, minimizes ``-objective`` (same optimum, flipped sign in report).
    bounds
        Optional length-5 sequence of ``(low, high)`` per coordinate; ``None`` means unbounded.
    method
        ``"SLSQP"`` (default) or ``"trust-constr"`` for constrained problems.
    options
        Passed to ``scipy.optimize.minimize`` (e.g. ``{"maxiter": 500}``).

    Returns
    -------
    dict with keys ``x``, ``objective``, ``success``, ``message``, ``nit``,
    ``constraint_residuals`` (max absolute value per constraint at ``x``),
    ``maximize``, ``method``.
    """
    x0_a = np.asarray(x0, dtype=float).reshape(-1)
    if x0_a.size != N_DIM:
        raise ValueError(f"x0 must have length {N_DIM}, got {x0_a.size}")

    if bounds is not None:
        if len(bounds) != N_DIM:
            raise ValueError(f"bounds must have length {N_DIM}, got {len(bounds)}")

    cons: list[dict[str, Any]] = []
    for j, g in enumerate(equality_constraints):
        cons.append({"type": "eq", "fun": g})

    if not cons:
        raise ValueError("At least one equality constraint is required.")

    def obj_wrapped(x: np.ndarray) -> float:
        v = float(objective(x))
        return -v if maximize else v

    opt = options if options is not None else {"maxiter": 1000, "ftol": 1e-12}

    res = minimize(
        obj_wrapped,
        x0_a,
        method=method,
        bounds=bounds,
        constraints=cons,
        options=opt,
    )

    x_opt = np.asarray(res.x, dtype=float)
    f_raw = float(objective(x_opt))
    f_report = -f_raw if maximize else f_raw

    residuals: list[float] = []
    for g in equality_constraints:
        residuals.append(float(abs(g(x_opt))))

    return {
        "x": x_opt,
        "objective": f_report,
        "success": bool(res.success),
        "message": str(res.message),
        "nit": int(res.nit) if hasattr(res, "nit") else -1,
        "constraint_residuals": residuals,
        "max_constraint_violation": float(max(residuals)) if residuals else 0.0,
        "maximize": maximize,
        "method": method,
    }


def _demo_minimize_sphere_on_simplex() -> None:
    """
    Minimize sum(x_i^2) subject to sum(x_i) = 1, x_i >= 0 (simplex slice).

    Unconstrained KKT on the hyperplane gives x_i = 1/5 (symmetric).
    """
    print("Demo: minimize sum(x_i^2) s.t. sum(x_i) = 1, x_i in [0,1]")

    def f(x: np.ndarray) -> float:
        return float(np.dot(x, x))

    def g_sum(x: np.ndarray) -> float:
        return float(np.sum(x) - 1.0)

    x0 = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=float)
    bnds = ((0.0, 1.0),) * N_DIM
    out = solve_lagrangian_nlp_5d(
        f,
        [g_sum],
        x0,
        maximize=False,
        bounds=bnds,
        method="SLSQP",
    )
    print("  x* =", np.round(out["x"], 6))
    print("  f* =", out["objective"])
    print("  max |g| =", out["max_constraint_violation"])
    print("  success:", out["success"], "|", out["message"])


if __name__ == "__main__":
    _demo_minimize_sphere_on_simplex()
