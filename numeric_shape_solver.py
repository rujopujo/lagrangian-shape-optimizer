"""
General numeric NLPP solver for the same geometry problems as ``lagrangian_solver``.

Uses SciPy SLSQP (sequential least squares) with **equality constraints** — a
black-box counterpart to the closed-form KKT formulas. Results should match
``LagrangianShapeSolver`` within numerical tolerance for every supported shape.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import minimize

from lagrangian_solver import LagrangianShapeSolver, _triangle_area_heron
from shape_data import VALID_SHAPES

__all__ = ["NumericShapeSolver", "compare_analytic_vs_numeric"]

_TWO_PI = 2.0 * math.pi
_FOUR_THIRDS_PI = (4.0 / 3.0) * math.pi
_TAU = math.tau


def _sphere_surface(r: float) -> float:
    return 4.0 * math.pi * r**2


class NumericShapeSolver:
    """
    Equality-constrained optimization via ``scipy.optimize.minimize`` (SLSQP).

    Same shapes and constraint semantics as :class:`LagrangianShapeSolver`.
    """

    def __init__(self, shape: str, constraint_value: float) -> None:
        s = shape.lower().strip()
        if s not in VALID_SHAPES:
            raise ValueError(f"Unknown shape {s!r}. Valid: {sorted(VALID_SHAPES)}")
        self.shape = s
        self.constraint_value = float(constraint_value)
        self._last_result: dict[str, Any] | None = None

    def solve(self, verbose: bool = False) -> dict[str, Any]:
        """Run SLSQP and return optimal dimensions and objective (same keys as analytic solver)."""
        c = self.constraint_value
        if c <= 0:
            raise ValueError("Constraint value must be positive.")

        if self.shape == "triangle":
            # x = [a, b, c], max area, a+b+c = P
            p = c

            def obj(x: np.ndarray) -> float:
                a, b, cc = float(x[0]), float(x[1]), float(x[2])
                return -_triangle_area_heron(a, b, cc)

            cons = {"type": "eq", "fun": lambda x: float(x[0] + x[1] + x[2]) - p}
            x0 = np.array([p / 3.0, p / 3.0, p / 3.0], dtype=float)
            bnds = ((1e-8, p), (1e-8, p), (1e-8, p))
            res = minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons, options={"ftol": 1e-12})
            a, b, cc = (float(v) for v in res.x)
            area = _triangle_area_heron(a, b, cc)
            optimal_dims = {"side_a": a, "side_b": b, "side_c": cc}
            objective = area

        elif self.shape == "circle":
            p = c

            def obj(x: np.ndarray) -> float:
                r = float(x[0])
                return -math.pi * r * r

            cons = {"type": "eq", "fun": lambda x: _TWO_PI * float(x[0]) - p}
            x0 = np.array([p / _TWO_PI], dtype=float)
            bnds = ((1e-8, p),)
            res = minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons, options={"ftol": 1e-12})
            r = float(res.x[0])
            optimal_dims = {"radius": r}
            objective = math.pi * r * r

        elif self.shape == "rectangle":
            p = c

            def obj(x: np.ndarray) -> float:
                l, w = float(x[0]), float(x[1])
                return -(l * w)

            cons = {"type": "eq", "fun": lambda x: 2.0 * float(x[0]) + 2.0 * float(x[1]) - p}
            x0 = np.array([p / 4.0, p / 4.0], dtype=float)
            bnds = ((1e-8, p), (1e-8, p))
            res = minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons, options={"ftol": 1e-12})
            l, w = float(res.x[0]), float(res.x[1])
            optimal_dims = {"length": l, "width": w}
            objective = l * w

        elif self.shape == "parallelogram":
            p = c

            def obj(x: np.ndarray) -> float:
                a, b, th = float(x[0]), float(x[1]), float(x[2])
                return -(a * b * math.sin(th))

            cons = {"type": "eq", "fun": lambda x: 2.0 * float(x[0]) + 2.0 * float(x[1]) - p}
            x0 = np.array([p / 4.0, p / 4.0, math.pi / 2.0], dtype=float)
            bnds = ((1e-8, p), (1e-8, p), (0.01, math.pi - 0.01))
            res = minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons, options={"ftol": 1e-12})
            a, b, th = float(res.x[0]), float(res.x[1]), float(res.x[2])
            optimal_dims = {"side_a": a, "side_b": b, "angle_rad": th}
            objective = a * b * math.sin(th)

        elif self.shape == "rhombus":
            p = c

            def obj(x: np.ndarray) -> float:
                s, th = float(x[0]), float(x[1])
                return -(s * s * math.sin(th))

            cons = {"type": "eq", "fun": lambda x: 4.0 * float(x[0]) - p}
            x0 = np.array([p / 4.0, math.pi / 2.0], dtype=float)
            bnds = ((1e-8, p), (0.01, math.pi - 0.01))
            res = minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons, options={"ftol": 1e-12})
            s, th = float(res.x[0]), float(res.x[1])
            optimal_dims = {"side": s, "angle_rad": th}
            objective = s * s * math.sin(th)

        elif self.shape == "cylinder":
            v = c

            def obj(x: np.ndarray) -> float:
                r, h = float(x[0]), float(x[1])
                return _TAU * r * r + _TAU * r * h

            cons = {"type": "eq", "fun": lambda x: math.pi * float(x[0]) ** 2 * float(x[1]) - v}
            r0 = (v / _TAU) ** (1.0 / 3.0)
            x0 = np.array([r0, 2.0 * r0], dtype=float)
            bnds = ((1e-8, None), (1e-8, None))
            res = minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons, options={"ftol": 1e-12})
            r, h = float(res.x[0]), float(res.x[1])
            optimal_dims = {"radius": r, "height": h}
            objective = _TAU * r * r + _TAU * r * h

        elif self.shape == "box":
            v = c

            def obj(x: np.ndarray) -> float:
                l, w, h = float(x[0]), float(x[1]), float(x[2])
                return 2.0 * (l * w + l * h + w * h)

            cons = {"type": "eq", "fun": lambda x: float(x[0]) * float(x[1]) * float(x[2]) - v}
            s0 = v ** (1.0 / 3.0)
            x0 = np.array([s0, s0, s0], dtype=float)
            bnds = ((1e-8, None), (1e-8, None), (1e-8, None))
            res = minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons, options={"ftol": 1e-12})
            l, w, h = float(res.x[0]), float(res.x[1]), float(res.x[2])
            optimal_dims = {"length": l, "width": w, "height": h}
            objective = 2.0 * (l * w + l * h + w * h)

        elif self.shape == "sphere":
            v = c

            def obj(x: np.ndarray) -> float:
                r = float(x[0])
                return _sphere_surface(r)

            cons = {"type": "eq", "fun": lambda x: _FOUR_THIRDS_PI * float(x[0]) ** 3 - v}
            r0 = (v / _FOUR_THIRDS_PI) ** (1.0 / 3.0)
            x0 = np.array([r0], dtype=float)
            bnds = ((1e-8, None),)
            res = minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons, options={"ftol": 1e-12})
            r = float(res.x[0])
            optimal_dims = {"radius": r}
            objective = _sphere_surface(r)

        else:
            raise ValueError(f"Unhandled shape: {self.shape}")

        if verbose:
            print(getattr(res, "message", ""))

        out: dict[str, Any] = {
            "optimal_dims": optimal_dims,
            "optimal_objective": float(objective),
            "shape": self.shape,
            "constraint_value": c,
            "success": bool(getattr(res, "success", True)),
        }
        self._last_result = out
        return out


def compare_analytic_vs_numeric(shape: str, constraint_value: float) -> dict[str, Any]:
    """Run closed-form and numeric solvers; return both objectives and relative error on objective."""
    a = LagrangianShapeSolver(shape, constraint_value).solve(verbose=False)
    n = NumericShapeSolver(shape, constraint_value).solve(verbose=False)
    oa = float(a["optimal_objective"])
    on = float(n["optimal_objective"])
    rel = abs(oa - on) / max(abs(oa), 1e-15)
    return {
        "analytic_objective": oa,
        "numeric_objective": on,
        "relative_error_objective": rel,
        "analytic_dims": a["optimal_dims"],
        "numeric_dims": n["optimal_dims"],
        "shape": shape.lower().strip(),
        "constraint_value": float(constraint_value),
    }


if __name__ == "__main__":
    tests = [
        ("triangle", 40.0),
        ("circle", 40.0),
        ("rectangle", 40.0),
        ("parallelogram", 40.0),
        ("rhombus", 40.0),
        ("cylinder", 1000.0),
        ("box", 1000.0),
        ("sphere", 1000.0),
    ]
    for sh, val in tests:
        print("=" * 60)
        cmp = compare_analytic_vs_numeric(sh, val)
        print(f"{sh} @ constraint={val}")
        print(f"  analytic objective: {cmp['analytic_objective']:.12g}")
        print(f"  numeric objective:  {cmp['numeric_objective']:.12g}")
        print(f"  relative |Δ|/|f|:    {cmp['relative_error_objective']:.3e}")
