"""
Preset shape parameters and LaTeX-friendly metadata for the Lagrangian shape optimizer.

Used by the Streamlit app, ``lagrangian_solver``, and ``visualizer`` so shape keys stay in sync.
"""

from __future__ import annotations

import math
from typing import Any

# All supported shape keys (lowercase).
VALID_SHAPES: frozenset[str] = frozenset(
    {
        "triangle",
        "circle",
        "rectangle",
        "parallelogram",
        "rhombus",
        "cylinder",
        "box",
        "sphere",
    }
)

PERIMETER_SHAPES: frozenset[str] = frozenset(
    {"triangle", "circle", "rectangle", "parallelogram", "rhombus"}
)
VOLUME_SHAPES: frozenset[str] = frozenset({"cylinder", "box", "sphere"})


def uses_perimeter(shape: str) -> bool:
    """True if constraint is perimeter / circumference (length), else volume."""
    return shape.lower().strip() in PERIMETER_SHAPES


def get_sample_data(shape: str) -> dict[str, Any]:
    """Return preset constraint values and labels for the given shape key."""
    s = shape.lower().strip()
    if s in PERIMETER_SHAPES:
        return {
            "shape": s,
            "constraint_value": 40.0,
            "constraint_label": "Fixed Perimeter P",
            "unit_label": "length units",
        }
    if s in VOLUME_SHAPES:
        return {
            "shape": s,
            "constraint_value": 1000.0,
            "constraint_label": "Fixed Volume V",
            "unit_label": "cubic units",
        }
    raise ValueError(f"Unknown shape: {shape!r}")


def get_shape_info(shape: str) -> dict[str, Any]:
    """Return labels, LaTeX, objective mode, and KKT lines for Streamlit."""
    s = shape.lower().strip()

    if s == "triangle":
        return {
            "objective_label": "Maximize Area",
            "constraint_label": "Fixed Perimeter",
            "objective_mode": "maximize",
            "variables": ["side a", "side b", "side c"],
            "latex_objective": r"A(a,b,c) = \sqrt{s(s-a)(s-b)(s-c)},\quad s=\frac{a+b+c}{2}",
            "latex_constraint": r"g = a+b+c-P = 0",
            "latex_lagrangian": r"\mathcal{L} = A - \lambda(a+b+c-P)\ \Rightarrow\ a=b=c=\frac{P}{3}",
            "latex_solution": r"a^\ast=b^\ast=c^\ast=\frac{P}{3},\quad A^\ast=\frac{\sqrt{3}}{36}P^2",
            "kkt_lines": [
                r"\text{Symmetry / KKT} \Rightarrow a=b=c,\quad 3a=P",
            ],
        }

    if s == "circle":
        return {
            "objective_label": "Maximize Area",
            "constraint_label": "Fixed Circumference",
            "objective_mode": "maximize",
            "variables": ["radius r"],
            "latex_objective": r"f(r) = \pi r^2",
            "latex_constraint": r"g(r) = 2\pi r - P = 0",
            "latex_lagrangian": r"\mathcal{L} = \pi r^2 - \lambda(2\pi r - P)",
            "latex_solution": r"r^\ast = \frac{P}{2\pi},\quad A^\ast=\frac{P^2}{4\pi}",
            "kkt_lines": [
                r"\frac{d\mathcal{L}}{dr} = 2\pi r - 2\pi\lambda = 0 \Rightarrow r=\lambda,\quad 2\pi r = P",
            ],
        }

    if s == "rectangle":
        return {
            "objective_label": "Maximize Area",
            "constraint_label": "Fixed Perimeter",
            "objective_mode": "maximize",
            "variables": ["length l", "width w"],
            "latex_objective": r"f(l,w) = l \cdot w",
            "latex_constraint": r"g(l,w) = 2l + 2w - P = 0",
            "latex_lagrangian": r"\mathcal{L}(l,w,\lambda) = lw - \lambda(2l+2w-P)",
            "latex_solution": r"l^\ast = w^\ast = \frac{P}{4},\quad A^\ast = \frac{P^2}{16}",
            "kkt_lines": [
                r"\frac{\partial \mathcal{L}}{\partial l} = w - 2\lambda = 0,\quad "
                r"\frac{\partial \mathcal{L}}{\partial w} = l - 2\lambda = 0,\quad "
                r"\frac{\partial \mathcal{L}}{\partial \lambda} = -(2l+2w-P)=0",
            ],
        }

    if s == "parallelogram":
        return {
            "objective_label": "Maximize Area",
            "constraint_label": "Fixed Perimeter",
            "objective_mode": "maximize",
            "variables": ["side a", "side b", "angle \theta"],
            "latex_objective": r"A = ab\sin\theta",
            "latex_constraint": r"2a+2b = P",
            "latex_lagrangian": r"\mathcal{L} = ab\sin\theta - \lambda(2a+2b-P)",
            "latex_solution": r"\theta^\ast=\frac{\pi}{2},\ a^\ast=b^\ast=\frac{P}{4}\ \text{(rectangle / square)}",
            "kkt_lines": [
                r"\partial\mathcal{L}/\partial\theta = ab\cos\theta = 0 \Rightarrow \theta=\pi/2",
                r"\partial\mathcal{L}/\partial a = b\sin\theta - 2\lambda = 0,\ \partial\mathcal{L}/\partial b = a\sin\theta - 2\lambda = 0 \Rightarrow a=b",
            ],
        }

    if s == "rhombus":
        return {
            "objective_label": "Maximize Area",
            "constraint_label": "Fixed Perimeter",
            "objective_mode": "maximize",
            "variables": ["side s", "angle \theta"],
            "latex_objective": r"A = s^2\sin\theta",
            "latex_constraint": r"4s = P",
            "latex_lagrangian": r"\mathcal{L} = s^2\sin\theta - \lambda(4s-P)",
            "latex_solution": r"s^\ast=\frac{P}{4},\ \theta^\ast=\frac{\pi}{2},\ A^\ast=\frac{P^2}{16}",
            "kkt_lines": [
                r"\partial\mathcal{L}/\partial\theta = s^2\cos\theta = 0 \Rightarrow \theta=\pi/2",
                r"\partial\mathcal{L}/\partial s = 2s\sin\theta - 4\lambda = 0 \Rightarrow s=P/4",
            ],
        }

    if s == "cylinder":
        return {
            "objective_label": "Minimize Surface Area",
            "constraint_label": "Fixed Volume",
            "objective_mode": "minimize",
            "variables": ["radius r", "height h"],
            "latex_objective": r"f(r,h) = 2\pi r^2 + 2\pi r h",
            "latex_constraint": r"g(r,h) = \pi r^2 h - V = 0",
            "latex_lagrangian": r"\mathcal{L}(r,h,\lambda) = 2\pi r^2 + 2\pi r h - \lambda(\pi r^2 h - V)",
            "latex_solution": r"h^\ast = 2r^\ast,\quad r^\ast = \left(\frac{V}{2\pi}\right)^{1/3}",
            "kkt_lines": [
                r"\frac{\partial \mathcal{L}}{\partial h} = 2\pi r - \lambda \pi r^2 = 0 \Rightarrow \lambda = 2/r",
                r"\frac{\partial \mathcal{L}}{\partial r} = 0 \Rightarrow h = 2r,\quad \pi r^2 h = V",
            ],
        }

    if s == "box":
        return {
            "objective_label": "Minimize Surface Area",
            "constraint_label": "Fixed Volume",
            "objective_mode": "minimize",
            "variables": ["length l", "width w", "height h"],
            "latex_objective": r"f(l,w,h) = 2(lw + lh + wh)",
            "latex_constraint": r"g(l,w,h) = lwh - V = 0",
            "latex_lagrangian": r"\mathcal{L} = 2(lw+lh+wh) - \lambda(lwh - V)",
            "latex_solution": r"l^\ast = w^\ast = h^\ast = V^{1/3}",
            "kkt_lines": [
                r"\nabla_{(l,w,h)} f = \lambda \nabla g \Rightarrow l=w=h=V^{1/3}",
            ],
        }

    if s == "sphere":
        return {
            "objective_label": "Minimize Surface Area",
            "constraint_label": "Fixed Volume",
            "objective_mode": "minimize",
            "variables": ["radius r"],
            "latex_objective": r"f(r) = 4\pi r^2",
            "latex_constraint": r"g(r) = \frac{4}{3}\pi r^3 - V = 0",
            "latex_lagrangian": r"\mathcal{L} = 4\pi r^2 - \lambda\left(\frac{4}{3}\pi r^3 - V\right)",
            "latex_solution": r"r^\ast = \left(\frac{3V}{4\pi}\right)^{1/3},\quad S^\ast = 4\pi r^{\ast 2}",
            "kkt_lines": [
                r"\frac{d\mathcal{L}}{dr} = 8\pi r - 4\pi\lambda r^2 = 0 \Rightarrow \lambda = \frac{2}{r},\quad \frac{4}{3}\pi r^3 = V",
            ],
        }

    raise ValueError(f"Unknown shape: {shape!r}")


def validate_input(shape: str, value: float) -> bool:
    """Validate shape name and positive constraint value."""
    s = shape.lower().strip()
    if s not in VALID_SHAPES:
        raise ValueError(f"Unknown shape {shape!r}. Valid: {sorted(VALID_SHAPES)}")
    if value <= 0:
        raise ValueError("Constraint value must be positive (perimeter or volume).")
    if not math.isfinite(value):
        raise ValueError("Constraint value must be a finite positive number.")
    return True
