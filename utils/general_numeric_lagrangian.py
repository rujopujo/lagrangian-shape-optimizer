from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.optimize import approx_fprime, minimize


def _grad(fn: Callable[[np.ndarray], float], x: np.ndarray, eps: float) -> np.ndarray:
    return approx_fprime(x, fn, epsilon=eps)


def solve_general_numeric_lagrangian(
    objective: Callable[[np.ndarray], float],
    equality_constraints: Sequence[Callable[[np.ndarray], float]],
    inequality_constraints: Sequence[Callable[[np.ndarray], float]] | None,
    x0: Sequence[float],
    variable_names: Sequence[str] | None = None,
    *,
    maximize: bool = False,
    bounds: Sequence[tuple[float | None, float | None]] | None = None,
    method: str = "SLSQP",
    options: dict[str, Any] | None = None,
    grad_eps: float = 1e-8,
    active_tol: float = 1e-6,
) -> dict[str, Any]:
    """
    Numeric constrained NLPP solver using SciPy SLSQP / trust-constr.

    Equality constraints are in g(x)=0 form.
    Inequality constraints are in h(x)<=0 form.
    """
    x0_arr = np.asarray(x0, dtype=float).reshape(-1)
    n = int(x0_arr.size)
    if n < 2:
        raise ValueError("Need at least 2 variables.")
    if not equality_constraints:
        raise ValueError("At least one equality constraint is required.")

    eq_cons = list(equality_constraints)
    ineq_cons = list(inequality_constraints or [])
    var_names = list(variable_names) if variable_names else [f"x{i+1}" for i in range(n)]
    if len(var_names) != n:
        raise ValueError(f"Expected {n} variable names, got {len(var_names)}.")
    if bounds is not None and len(bounds) != n:
        raise ValueError(f"Expected {n} bounds, got {len(bounds)}.")

    def obj_min(x: np.ndarray) -> float:
        val = float(objective(x))
        return -val if maximize else val

    scipy_constraints: list[dict[str, Any]] = []
    for g in eq_cons:
        scipy_constraints.append({"type": "eq", "fun": g})
    for h in ineq_cons:
        # SciPy expects ineq >= 0, UI uses h(x) <= 0.
        scipy_constraints.append({"type": "ineq", "fun": lambda x, fn=h: -float(fn(x))})

    res = minimize(
        obj_min,
        x0_arr,
        method=method,
        bounds=bounds,
        constraints=scipy_constraints,
        options=options if options is not None else {"ftol": 1e-9, "maxiter": 1000},
    )

    x_star = np.asarray(res.x, dtype=float)
    f_star = float(objective(x_star))
    eq_vals = [float(g(x_star)) for g in eq_cons]
    ineq_vals = [float(h(x_star)) for h in ineq_cons]  # should be <= 0

    lambda_eq: list[float] | None = None
    mu_ineq = [0.0] * len(ineq_cons)

    # 1) Prefer SciPy-reported multipliers when available.
    if hasattr(res, "multipliers"):
        raw = getattr(res, "multipliers")
        if isinstance(raw, dict):
            eq_raw = list(raw.get("eq", []))
            ineq_raw = list(raw.get("ineq", []))
            if eq_raw:
                lambda_eq = [float(v) for v in eq_raw[: len(eq_cons)]]
            for i, v in enumerate(ineq_raw[: len(ineq_cons)]):
                mu_ineq[i] = max(0.0, float(v))
        else:
            arr = np.asarray(raw, dtype=float).reshape(-1)
            if arr.size >= len(eq_cons):
                lambda_eq = [float(v) for v in arr[: len(eq_cons)]]
                tail = arr[len(eq_cons) : len(eq_cons) + len(ineq_cons)]
                for i, v in enumerate(tail):
                    mu_ineq[i] = max(0.0, float(v))

    # 2) Estimate multipliers by least squares if not provided.
    if lambda_eq is None:
        grad_obj = _grad(obj_min, x_star, grad_eps)
        jac_eq = [_grad(g, x_star, grad_eps) for g in eq_cons]
        active_indices = [i for i, hval in enumerate(ineq_vals) if hval >= -active_tol]
        jac_ineq_active = [_grad(ineq_cons[i], x_star, grad_eps) for i in active_indices]

        cols = jac_eq + jac_ineq_active
        if cols:
            a = np.column_stack(cols)
            m_hat, *_ = np.linalg.lstsq(a, -grad_obj, rcond=None)
            m_hat = np.asarray(m_hat, dtype=float).reshape(-1)
            lambda_eq = [float(v) for v in m_hat[: len(eq_cons)]]
            mu_active = m_hat[len(eq_cons) :]
            for slot, idx in enumerate(active_indices):
                mu_ineq[idx] = max(0.0, float(mu_active[slot]))
        else:
            lambda_eq = [0.0] * len(eq_cons)

    # KKT residual checks.
    grad_obj = _grad(obj_min, x_star, grad_eps)
    stationarity = grad_obj.copy()
    for lam, g in zip(lambda_eq, eq_cons):
        stationarity += float(lam) * _grad(g, x_star, grad_eps)
    for mu, h in zip(mu_ineq, ineq_cons):
        stationarity += float(mu) * _grad(h, x_star, grad_eps)

    eq_violation = max((abs(v) for v in eq_vals), default=0.0)
    ineq_violation = max((max(v, 0.0) for v in ineq_vals), default=0.0)
    complementarity = max((abs(mu * hval) for mu, hval in zip(mu_ineq, ineq_vals)), default=0.0)
    stationarity_inf = float(np.linalg.norm(stationarity, ord=np.inf))

    return {
        "x_opt": x_star,
        "variable_names": var_names,
        "optimal_objective": f_star,
        "direction": "maximize" if maximize else "minimize",
        "success": bool(res.success),
        "status": int(getattr(res, "status", -1)),
        "message": str(res.message),
        "iterations": int(getattr(res, "nit", -1)),
        "lambda_eq": [float(v) for v in lambda_eq],
        "mu_ineq": [float(v) for v in mu_ineq],
        "eq_residuals": eq_vals,
        "ineq_values": ineq_vals,
        "kkt_residuals": {
            "stationarity_inf_norm": stationarity_inf,
            "eq_violation_max_abs": float(eq_violation),
            "ineq_violation_max": float(ineq_violation),
            "complementarity_max_abs": float(complementarity),
        },
    }
