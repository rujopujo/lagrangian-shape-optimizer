"""
Lagrangian (KKT) solvers for geometric NLPPs with one equality constraint.

No scipy.optimize — closed-form solutions only.
"""

from __future__ import annotations

import math
import sys
from typing import Any

from shape_data import VALID_SHAPES

# Safer printing of Unicode on Windows consoles.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (OSError, ValueError):
        pass

_SQRT3 = math.sqrt(3.0)
_IMPROVEMENT_DENOM_EPS = 1e-15


def _triangle_area_heron(a: float, b: float, c: float) -> float:
    s = 0.5 * (a + b + c)
    x = max(s * (s - a) * (s - b) * (s - c), 0.0)
    return math.sqrt(x)


def _pct_better_maximize(optimal: float, suboptimal: float) -> float:
    """Percent improvement when larger objective is better."""
    return 100.0 * (optimal - suboptimal) / max(suboptimal, _IMPROVEMENT_DENOM_EPS)


def _pct_better_minimize(optimal: float, suboptimal: float) -> float:
    """Percent improvement when smaller objective is better."""
    return 100.0 * (suboptimal - optimal) / max(suboptimal, _IMPROVEMENT_DENOM_EPS)


def _constraint_residual(shape: str, d: dict[str, float], c: float) -> float:
    """Equality constraint g(x) at optimum (should be ≈ 0)."""
    if shape == "triangle":
        return d["side_a"] + d["side_b"] + d["side_c"] - c
    if shape == "rectangle":
        return 2.0 * d["length"] + 2.0 * d["width"] - c
    if shape == "parallelogram":
        return 2.0 * d["side_a"] + 2.0 * d["side_b"] - c
    if shape == "circle":
        return 2.0 * math.pi * d["radius"] - c
    if shape == "rhombus":
        return 4.0 * d["side"] - c
    if shape == "cylinder":
        return math.pi * d["radius"] ** 2 * d["height"] - c
    if shape == "box":
        return d["length"] * d["width"] * d["height"] - c
    if shape == "sphere":
        return (4.0 / 3.0) * math.pi * d["radius"] ** 3 - c
    return 0.0


class LagrangianShapeSolver:
    """Closed-form KKT solutions for supported shapes (see ``shape_data.VALID_SHAPES``)."""

    def __init__(self, shape: str, constraint_value: float) -> None:
        s = shape.lower().strip()
        if s not in VALID_SHAPES:
            raise ValueError(f"Unknown shape {s!r}. Valid: {sorted(VALID_SHAPES)}")
        self.shape = s
        self.constraint_value = float(constraint_value)
        self._last_result: dict[str, Any] | None = None

    def solve(self, verbose: bool = True) -> dict[str, Any]:
        """Apply KKT / symmetry and return optimal dimensions and objective."""
        steps: list[str] = []

        def _step(msg: str) -> None:
            steps.append(msg)
            if verbose:
                print(msg)

        _step("Step 1: Defining the objective function f and constraint g")
        _step("Step 2: Forming the Lagrangian L = f - λg")
        _step("Step 3: Taking partial derivatives ∂L/∂x = 0")
        _step("Step 4: Solving the KKT system analytically")

        val = self.constraint_value
        if val <= 0:
            raise ValueError("Constraint value must be positive.")

        lam: float
        objective: float
        optimal_dims: dict[str, float]

        if self.shape == "triangle":
            # Equilateral: max area for fixed perimeter P (Heron / symmetry)
            p = val
            a = b = c = p / 3.0
            area = _SQRT3 * a**2 / 4.0
            lam = _SQRT3 * p / 36.0
            optimal_dims = {"side_a": a, "side_b": b, "side_c": c}
            objective = area

        elif self.shape == "circle":
            p = val
            r = p / (2.0 * math.pi)
            lam = r
            objective = math.pi * r**2
            optimal_dims = {"radius": r}

        elif self.shape == "rectangle":
            p = val
            side = p / 4.0
            lam = side / 2.0
            objective = side * side
            optimal_dims = {"length": side, "width": side}

        elif self.shape == "parallelogram":
            p = val
            a = b = p / 4.0
            theta = math.pi / 2.0
            lam = a / 2.0
            objective = a * b  # sin(π/2) = 1
            optimal_dims = {"side_a": a, "side_b": b, "angle_rad": theta}

        elif self.shape == "rhombus":
            p = val
            s = p / 4.0
            theta = math.pi / 2.0
            lam = s / 2.0  # sin(π/2) = 1
            objective = s * s
            optimal_dims = {"side": s, "angle_rad": theta}

        elif self.shape == "cylinder":
            v = val
            r_star = (v / (2.0 * math.pi)) ** (1.0 / 3.0)
            h_star = 2.0 * r_star
            lam = 2.0 / r_star
            objective = 2.0 * math.pi * r_star**2 + 2.0 * math.pi * r_star * h_star
            optimal_dims = {"radius": r_star, "height": h_star}

        elif self.shape == "box":
            v = val
            s = v ** (1.0 / 3.0)
            lam = 4.0 / s
            objective = 6.0 * s**2
            optimal_dims = {"length": s, "width": s, "height": s}

        elif self.shape == "sphere":
            v = val
            r = ((3.0 * v) / (4.0 * math.pi)) ** (1.0 / 3.0)
            lam = 2.0 / r
            objective = 4.0 * math.pi * r**2
            optimal_dims = {"radius": r}

        else:
            raise ValueError(f"Unhandled shape: {self.shape}")

        _step("Step 5: Computing optimal dimensions and objective value")

        out: dict[str, Any] = {
            "optimal_dims": optimal_dims,
            "lambda_value": float(lam),
            "optimal_objective": float(objective),
            "shape": self.shape,
            "constraint_value": val,
            "steps": steps,
        }
        self._last_result = out
        return out

    def verify_constraint(self, verbose: bool = True) -> bool:
        """Check g = 0 at optimum within 1e-6."""
        if self._last_result is None:
            self.solve(verbose=False)
        assert self._last_result is not None
        d = self._last_result["optimal_dims"]
        g = _constraint_residual(self.shape, d, self.constraint_value)
        ok = abs(g) <= 1e-6
        if verbose:
            print(f"Constraint verification: g = {g:.3e}  →  {'PASS' if ok else 'FAIL'}")
        return ok

    def get_comparison(self) -> dict[str, Any]:
        """Compare optimal objective to a simple feasible non-optimal competitor."""
        if self._last_result is None:
            self.solve(verbose=False)
        assert self._last_result is not None
        opt_obj = float(self._last_result["optimal_objective"])
        d = self._last_result["optimal_dims"]
        c = self.constraint_value

        if self.shape == "rectangle":
            p = c
            l_sub, w_sub = p / 3.0, p / 6.0
            sub_obj = l_sub * w_sub
            label = "Area"
            improvement_pct = _pct_better_maximize(opt_obj, sub_obj)

        elif self.shape == "parallelogram":
            p = c
            l_sub, w_sub = p / 3.0, p / 6.0
            sub_obj = l_sub * w_sub * math.sin(math.pi / 6.0)
            label = "Area"
            improvement_pct = _pct_better_maximize(opt_obj, sub_obj)

        elif self.shape == "rhombus":
            p = c
            s = p / 4.0
            sub_obj = s**2 * math.sin(math.pi / 6.0)
            label = "Area"
            improvement_pct = _pct_better_maximize(opt_obj, sub_obj)

        elif self.shape == "triangle":
            a, b, k = c / 6.0, 5.0 * c / 12.0, 5.0 * c / 12.0
            sub_obj = _triangle_area_heron(a, b, k)
            label = "Area"
            improvement_pct = _pct_better_maximize(opt_obj, sub_obj)

        elif self.shape == "circle":
            p = c
            l_sub, w_sub = p / 3.0, p / 6.0
            sub_obj = l_sub * w_sub
            label = "Area"
            improvement_pct = _pct_better_maximize(opt_obj, sub_obj)

        elif self.shape == "cylinder":
            v = c
            r_opt = (v / (2.0 * math.pi)) ** (1.0 / 3.0)
            r_sub = 0.7 * r_opt
            h_sub = v / (math.pi * r_sub**2)
            sub_obj = 2.0 * math.pi * r_sub**2 + 2.0 * math.pi * r_sub * h_sub
            label = "Surface Area"
            improvement_pct = _pct_better_minimize(opt_obj, sub_obj)

        elif self.shape == "box":
            v = c
            s = v ** (1.0 / 3.0)
            l_sub = 0.5 * s
            wh = v / l_sub
            h_sub = math.sqrt(wh / 2.0)
            w_sub = 2.0 * h_sub
            sub_obj = 2.0 * (l_sub * w_sub + l_sub * h_sub + w_sub * h_sub)
            label = "Surface Area"
            improvement_pct = _pct_better_minimize(opt_obj, sub_obj)

        elif self.shape == "sphere":
            v = c
            s = v ** (1.0 / 3.0)
            sub_obj = 6.0 * s**2
            label = "Surface Area"
            improvement_pct = _pct_better_minimize(opt_obj, sub_obj)

        else:
            sub_obj = opt_obj
            label = "Objective"
            improvement_pct = 0.0

        return {
            "optimal_objective": opt_obj,
            "suboptimal_objective": float(sub_obj),
            "label": label,
            "improvement_percent": float(improvement_pct),
            "optimal_dims": d,
        }

    def get_summary(self, result: dict[str, Any] | None = None) -> str:
        """Return a formatted multi-line summary string."""
        if result is None:
            result = self.solve(verbose=False)
        lines = [
            f"Shape: {result['shape']}",
            f"Constraint value: {result['constraint_value']}",
            f"Optimal objective: {result['optimal_objective']:.8f}",
            f"λ: {result['lambda_value']:.8f}",
            "Dimensions:",
        ]
        for k, v in result["optimal_dims"].items():
            lines.append(f"  {k}: {v:.8f}")
        return "\n".join(lines)


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
        sol = LagrangianShapeSolver(sh, val)
        r = sol.solve()
        print(sol.get_summary(r))
        print("verify:", sol.verify_constraint())
        print("comparison:", sol.get_comparison())
