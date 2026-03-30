"""
General numeric NLPP solvers for 3, 4, 5, ... decision variables.

Implements equality-constrained smooth optimization via SciPy. This is the
practical computational version of Lagrangian/KKT when symbolic derivation is
not convenient.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.optimize import minimize

__all__ = [
    "solve_lagrangian_nlp_nd",
    "solve_lagrangian_nlp_3d",
    "solve_lagrangian_nlp_4d",
    "solve_lagrangian_nlp_5d",
]

def solve_lagrangian_nlp_nd(
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
    Solve a smooth NLPP in N variables with equality constraints.

    SciPy finds a point satisfying constrained first-order conditions (up to
    numeric tolerances), which corresponds to the KKT/Lagrangian optimum.

    Parameters
    ----------
    objective
        Maps ``x`` (length-N ``float64`` vector) to scalar objective value.
    equality_constraints
        Each callable must return **zero** at feasibility (e.g. ``g(x)==0``).
        Typical NLPP has one constraint; pass ``[g]`` or several ``[g1, g2, ...]``.
    x0
        Initial guess; dimension N is inferred from ``x0``.
    maximize
        If True, minimizes ``-objective`` (same optimum, flipped sign in report).
    bounds
        Optional length-N sequence of ``(low, high)`` per coordinate; ``None`` means unbounded.
    method
        ``"SLSQP"`` (default) or ``"trust-constr"`` for constrained problems.
    options
        Passed to ``scipy.optimize.minimize`` (e.g. ``{"maxiter": 500}``).

    Returns
    -------
    dict with keys ``x``, ``n_variables``, ``objective``, ``success``,
    ``message``, ``nit``, ``constraint_residuals`` (absolute per constraint),
    ``maximize``, ``method``.
    """
    x0_a = np.asarray(x0, dtype=float).reshape(-1)
    if x0_a.size < 2:
        raise ValueError("x0 must contain at least 2 variables.")
    n_dim = int(x0_a.size)

    if bounds is not None:
        if len(bounds) != n_dim:
            raise ValueError(f"bounds must have length {n_dim}, got {len(bounds)}")

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
        "n_variables": n_dim,
        "objective": f_report,
        "success": bool(res.success),
        "message": str(res.message),
        "nit": int(res.nit) if hasattr(res, "nit") else -1,
        "constraint_residuals": residuals,
        "max_constraint_violation": float(max(residuals)) if residuals else 0.0,
        "maximize": maximize,
        "method": method,
    }


def solve_lagrangian_nlp_3d(
    objective: Callable[[np.ndarray], float],
    equality_constraints: Sequence[Callable[[np.ndarray], float]],
    x0: np.ndarray | Sequence[float],
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience wrapper enforcing exactly 3 variables."""
    x = np.asarray(x0, dtype=float).reshape(-1)
    if x.size != 3:
        raise ValueError(f"x0 must have length 3, got {x.size}")
    return solve_lagrangian_nlp_nd(objective, equality_constraints, x0, **kwargs)


def solve_lagrangian_nlp_4d(
    objective: Callable[[np.ndarray], float],
    equality_constraints: Sequence[Callable[[np.ndarray], float]],
    x0: np.ndarray | Sequence[float],
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience wrapper enforcing exactly 4 variables."""
    x = np.asarray(x0, dtype=float).reshape(-1)
    if x.size != 4:
        raise ValueError(f"x0 must have length 4, got {x.size}")
    return solve_lagrangian_nlp_nd(objective, equality_constraints, x0, **kwargs)


def solve_lagrangian_nlp_5d(
    objective: Callable[[np.ndarray], float],
    equality_constraints: Sequence[Callable[[np.ndarray], float]],
    x0: np.ndarray | Sequence[float],
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience wrapper enforcing exactly 5 variables."""
    x = np.asarray(x0, dtype=float).reshape(-1)
    if x.size != 5:
        raise ValueError(f"x0 must have length 5, got {x.size}")
    return solve_lagrangian_nlp_nd(objective, equality_constraints, x0, **kwargs)


def _demo_simplex_min_norm(n_dim: int) -> None:
    """
    Minimize sum(x_i^2) subject to sum(x_i)=1 and 0<=x_i<=1.

    By symmetry/KKT, optimum is x_i = 1/n.
    """
    print(f"Demo ({n_dim} vars): minimize sum(x_i^2), sum(x_i)=1, x_i in [0,1]")

    def f(x: np.ndarray) -> float:
        return float(np.dot(x, x))

    def g_sum(x: np.ndarray) -> float:
        return float(np.sum(x) - 1.0)

    x0 = np.full(n_dim, 1.0 / n_dim, dtype=float)
    bnds = ((0.0, 1.0),) * n_dim
    out = solve_lagrangian_nlp_nd(
        f,
        [g_sum],
        x0,
        maximize=False,
        bounds=bnds,
        method="SLSQP",
    )
    print("  x* =", np.round(out["x"], 6))
    print("  expected per variable =", round(1.0 / n_dim, 6))
    print("  f* =", out["objective"])
    print("  max |g| =", out["max_constraint_violation"])
    print("  success:", out["success"], "|", out["message"])


if __name__ == "__main__":
    for n in (3, 4, 5, 8):
        _demo_simplex_min_norm(n)
        print("-" * 60)
