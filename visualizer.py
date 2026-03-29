"""
Matplotlib figures for geometric Lagrangian optimization (return ``Figure``, never ``plt.show()``).
"""

from __future__ import annotations

import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.patches import Rectangle as MplRectangle


# Dark theme figures to match the Streamlit UI
def _heron(a: float, b: float, c: float) -> float:
    s = 0.5 * (a + b + c)
    x = max(s * (s - a) * (s - b) * (s - c), 0.0)
    return float(np.sqrt(x))


_SQRT3 = math.sqrt(3.0)


plt.rcParams.update(
    {
        "figure.facecolor": "#0f172a",
        "axes.facecolor": "#0f172a",
        "axes.edgecolor": "#475569",
        "axes.labelcolor": "#cbd5e1",
        "axes.titlecolor": "#f1f5f9",
        "text.color": "#e2e8f0",
        "xtick.color": "#94a3b8",
        "ytick.color": "#94a3b8",
        "grid.color": "#334155",
        "grid.linestyle": "-",
        "grid.alpha": 0.4,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "600",
    }
)


def plot_shape_2d(shape: str, optimal_dims: dict[str, float]) -> Figure:
    """Draw a schematic of the optimal shape."""
    sh = shape.lower().strip()
    fig, ax = plt.subplots(figsize=(6, 5))

    if sh == "triangle":
        a = float(optimal_dims["side_a"])
        h = _SQRT3 * a / 2.0
        poly = np.array([[0.0, 0.0], [a, 0.0], [a / 2.0, h]])
        ax.add_patch(Polygon(poly, closed=True, facecolor="#38bdf8", edgecolor="#0c4a6e", linewidth=2))
        ax.set_aspect("equal")
        ax.set_xlim(-0.05 * a, 1.05 * a)
        ax.set_ylim(-0.05 * h, 1.1 * h)
        ax.axis("off")
        ax.set_title("Equilateral Triangle — Maximum Area")

    elif sh == "circle":
        r = float(optimal_dims["radius"])
        circ = plt.Circle((0, 0), r, facecolor="#34d399", edgecolor="#064e3b", linewidth=2)
        ax.add_patch(circ)
        ax.set_aspect("equal")
        ax.set_xlim(-1.1 * r, 1.1 * r)
        ax.set_ylim(-1.1 * r, 1.1 * r)
        ax.axis("off")
        ax.set_title("Circle — Maximum Area (fixed circumference)")

    elif sh == "rectangle":
        l = float(optimal_dims["length"])
        w = float(optimal_dims["width"])
        rect = MplRectangle((0, 0), l, w, facecolor="#a6d8ff", edgecolor="#003366", linewidth=2)
        ax.add_patch(rect)
        ax.set_xlim(-0.05 * max(l, w), 1.05 * max(l, w))
        ax.set_ylim(-0.05 * max(l, w), 1.05 * max(l, w))
        ax.set_aspect("equal")
        ax.set_xlabel("length l")
        ax.set_ylabel("width w")
        ax.text(l / 2, -0.05 * w, f"l = {l:.2f}", ha="center", fontsize=10)
        ax.text(-0.05 * l, w / 2, f"w = {w:.2f}", ha="right", va="center", fontsize=10)
        ax.set_title("Rectangle → Square Optimum")

    elif sh == "parallelogram":
        a = float(optimal_dims["side_a"])
        b = float(optimal_dims["side_b"])
        ang = float(optimal_dims["angle_rad"])
        ox, oy = 0.0, 0.0
        bx, by = a, 0.0
        cx, cy = a + b * math.cos(ang), b * math.sin(ang)
        dx, dy = b * math.cos(ang), b * math.sin(ang)
        poly = np.array([[ox, oy], [bx, by], [cx, cy], [dx, dy]])
        ax.add_patch(Polygon(poly, closed=True, facecolor="#a78bfa", edgecolor="#4c1d95", linewidth=2))
        ax.set_aspect("equal")
        m = max(a, b) * 1.5
        ax.set_xlim(-0.1 * m, m)
        ax.set_ylim(-0.1 * m, m)
        ax.axis("off")
        ax.set_title("Parallelogram → Rectangle Optimum")

    elif sh == "rhombus":
        s = float(optimal_dims["side"])
        ang = float(optimal_dims["angle_rad"])
        poly = np.array(
            [
                [0.0, 0.0],
                [s, 0.0],
                [s + s * math.cos(ang), s * math.sin(ang)],
                [s * math.cos(ang), s * math.sin(ang)],
            ]
        )
        ax.add_patch(Polygon(poly, closed=True, facecolor="#f472b6", edgecolor="#831843", linewidth=2))
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Rhombus → Square Optimum")

    elif sh == "cylinder":
        r = float(optimal_dims["radius"])
        h = float(optimal_dims["height"])
        # Side-view rectangle: width 2r, height h
        w_rect = 2 * r
        x0 = -r
        rect = MplRectangle((x0, 0), w_rect, h, facecolor="#c8e6c9", edgecolor="#1b5e20", linewidth=2)
        ax.add_patch(rect)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.annotate("r", xy=(0, -0.08 * h), ha="center", fontsize=11)
        ax.annotate("", xy=(r, -0.02 * h), xytext=(0, -0.02 * h), arrowprops=dict(arrowstyle="<->", color="black"))
        ax.annotate("h", xy=(r + 0.1 * r, h / 2), fontsize=11)
        ax.annotate("", xy=(r + 0.05 * r, h), xytext=(r + 0.05 * r, 0), arrowprops=dict(arrowstyle="<->", color="black"))
        ax.set_xlim(-1.2 * r, 1.4 * r)
        ax.set_ylim(-0.15 * h, 1.1 * h)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Cylinder Cross-Section")

    elif sh == "sphere":
        r = float(optimal_dims["radius"])
        circ = plt.Circle((0, 0), r, facecolor="#94a3b8", edgecolor="#e2e8f0", linewidth=2)
        ax.add_patch(circ)
        ax.set_aspect("equal")
        ax.set_xlim(-1.15 * r, 1.15 * r)
        ax.set_ylim(-1.15 * r, 1.15 * r)
        ax.axis("off")
        ax.set_title("Sphere (cross-section)")

    else:  # box — isometric-style cube
        s = float(optimal_dims["length"])
        # Simple isometric: three parallelograms for visible faces
        ax.set_aspect("equal")
        orig = np.array([0.0, 0.0])
        ex = np.array([1.0, 0.0]) * s
        ey = np.array([0.35, 0.65]) * s
        ez = np.array([-0.25, 0.75]) * s
        face1 = np.vstack([orig, orig + ex, orig + ex + ey, orig + ey])
        face2 = np.vstack([orig, orig + ey, orig + ey + ez, orig + ez])
        face3 = np.vstack([orig + ex, orig + ex + ey, orig + ex + ey + ez, orig + ex + ez])
        for face, c in [(face1, "#bbdefb"), (face2, "#90caf9"), (face3, "#64b5f6")]:
            ax.add_patch(Polygon(face, closed=True, facecolor=c, edgecolor="#0d47a1", linewidth=1.5))
        ax.set_xlim(-0.3 * s, 1.3 * s)
        ax.set_ylim(-0.2 * s, 1.2 * s)
        ax.axis("off")
        ax.text(0.5 * s, -0.12 * s, f"s = l = w = h = {s:.2f}", ha="center", fontsize=10)
        ax.set_title("Optimal Box (Cube)")

    fig.tight_layout()
    return fig


def plot_objective_vs_dimension(
    shape: str,
    constraint_value: float,
    optimal_dims: dict[str, float],
) -> Figure:
    """Plot objective vs one natural parameter; mark the analytical optimum."""
    sh = shape.lower().strip()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    if sh == "triangle":
        P = constraint_value
        aa = np.linspace(P / 12.0, P * 0.48, 400)
        areas = np.array([_heron(ai, (P - ai) / 2.0, (P - ai) / 2.0) for ai in aa])
        a_opt = float(optimal_dims["side_a"])
        opt = _heron(a_opt, (P - a_opt) / 2.0, (P - a_opt) / 2.0)
        ax.plot(aa, areas, color="#38bdf8", label="Area (isosceles family)")
        ax.scatter([a_opt], [opt], color="#f87171", s=80, zorder=5, label="Equilateral optimum")
        ax.set_xlabel("Base length a (two equal sides)")
        ax.set_ylabel("Area")
        ax.set_title(f"Area vs base — P = {P:g}")

    elif sh == "circle":
        P = constraint_value
        r = np.linspace(P / (4 * math.tau), P / math.tau * 0.99, 400)
        area = math.pi * r**2
        r_opt = float(optimal_dims["radius"])
        ax.plot(r, area, color="#34d399", label=r"$A=\pi r^2$")
        ax.axvline(r_opt, color="#94a3b8", linestyle="--", linewidth=1)
        ax.scatter([r_opt], [math.pi * r_opt**2], color="#f87171", s=80, zorder=5)
        ax.set_xlabel("Radius r")
        ax.set_ylabel("Area")
        ax.set_title(f"Area vs radius — 2πr = {P:g}")

    elif sh == "rectangle":
        P = constraint_value
        l = np.linspace(1e-6, P / 2.0 - 1e-6, 400)
        w = P / 2.0 - l
        area = l * w
        l_opt = float(optimal_dims["length"])
        ax.plot(l, area, color="#1565c0", label="Area = l × w")
        ax.axvline(l_opt, color="gray", linestyle="--", linewidth=1, label="Optimal l")
        ax.scatter([l_opt], [l_opt * (P / 2.0 - l_opt)], color="red", s=80, zorder=5, label="Maximum")
        ax.set_xlabel("Length l")
        ax.set_ylabel("Area = l × w")
        ax.set_title(f"Area vs Length — Perimeter Fixed at P = {P:g}")

    elif sh == "parallelogram":
        P = constraint_value
        l = np.linspace(1e-6, P / 2.0 - 1e-6, 400)
        w = P / 2.0 - l
        area = l * w
        l_opt = float(optimal_dims["side_a"])
        ax.plot(l, area, color="#a78bfa", label=r"$A=ab$ at $\theta=90°$")
        ax.scatter([l_opt], [l_opt * (P / 2.0 - l_opt)], color="#f87171", s=80, zorder=5)
        ax.set_xlabel("Side a (with b = P/2 − a)")
        ax.set_ylabel("Area")
        ax.set_title(f"Parallelogram area — P = {P:g}")

    elif sh == "rhombus":
        P = constraint_value
        s = P / 4.0
        th = np.linspace(0.15, math.pi - 0.15, 400)
        area = s**2 * np.sin(th)
        th_opt = float(optimal_dims["angle_rad"])
        ax.plot(th, area, color="#f472b6", label=r"$A=s^2\sin\theta$")
        ax.scatter([th_opt], [s**2 * math.sin(th_opt)], color="#f87171", s=80, zorder=5)
        ax.set_xlabel(r"Angle $\theta$")
        ax.set_ylabel("Area")
        ax.set_title(f"Rhombus area — side s = P/4 = {s:.2f}")

    elif sh == "cylinder":
        V = constraint_value
        r = np.linspace(0.2, max(3.0 * (V / math.tau) ** (1.0 / 3.0), 1.0), 400)
        h = V / (math.pi * r**2)
        sa = math.tau * r**2 + math.tau * r * h
        r_opt = float(optimal_dims["radius"])
        sa_opt = float(
            math.tau * r_opt**2
            + math.tau * r_opt * float(optimal_dims["height"])
        )
        ax.plot(r, sa, color="#2e7d32", label="Surface area")
        ax.scatter([r_opt], [sa_opt], color="red", s=80, zorder=5, label="Minimum")
        ax.set_xlabel("Radius r")
        ax.set_ylabel("Surface area")
        ax.set_title(f"Surface Area vs Radius — Volume Fixed at V = {V:g}")

    elif sh == "sphere":
        V = constraint_value
        r = np.linspace(0.15, max(2.2 * ((3.0 * V) / (4.0 * math.pi)) ** (1.0 / 3.0), 1.0), 400)
        sa = 4 * math.pi * r**2
        r_opt = float(optimal_dims["radius"])
        sa_opt = 4 * math.pi * r_opt**2
        ax.plot(r, sa, color="#94a3b8", label=r"$S=4\pi r^2$")
        ax.axvline(r_opt, color="#64748b", linestyle="--", linewidth=1)
        ax.scatter([r_opt], [sa_opt], color="#f87171", s=80, zorder=5, label="Minimum (fixed V)")
        ax.set_xlabel("Radius r")
        ax.set_ylabel("Surface area")
        ax.set_title(f"Sphere surface vs r — V = {V:g}")

    elif sh == "box":
        V = constraint_value
        s = np.linspace(V ** (1.0 / 3.0) * 0.4, V ** (1.0 / 3.0) * 1.6, 400)
        hgt = V / (s**2)
        sa = 2 * (s * s + s * hgt + s * hgt)
        s_opt = float(optimal_dims["length"])
        sa_opt = 6 * s_opt**2
        ax.plot(s, sa, color="#6a1b9a", label="Surface area")
        ax.scatter([s_opt], [sa_opt], color="red", s=80, zorder=5, label="Minimum")
        ax.set_xlabel("Side length s (square base l = w = s)")
        ax.set_ylabel("Surface area")
        ax.set_title(f"Surface Area vs Side Length — Volume Fixed at V = {V:g}")

    else:
        ax.text(0.5, 0.5, "Objective plot not defined for this shape.", ha="center", va="center")
        ax.axis("off")

    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_lagrangian_contour(
    shape: str,
    constraint_value: float,
    optimal_dims: dict[str, float],
) -> Figure:
    """Contour of objective / Lagrangian slice."""
    sh = shape.lower().strip()

    if sh == "triangle":
        p = constraint_value
        a = np.linspace(p / 8.0, p * 0.45, 90)
        b = np.linspace(p / 8.0, p * 0.45, 90)
        aa, bb = np.meshgrid(a, b)
        cc = p - aa - bb
        s = 0.5 * (aa + bb + cc)
        heron_inner = s * (s - aa) * (s - bb) * (s - cc)
        valid = cc > 1e-9
        heron_ok = valid & (heron_inner >= 0)
        z = np.full_like(aa, np.nan, dtype=float)
        z[heron_ok] = np.sqrt(heron_inner[heron_ok])
        a_opt = float(optimal_dims["side_a"])
        b_opt = float(optimal_dims["side_b"])
        fig, ax = plt.subplots(figsize=(6.5, 5))
        cs = ax.contourf(aa, bb, z, levels=28, cmap="viridis", alpha=0.9)
        a_line = np.linspace(p / 8.0, p * 0.45, 200)
        ax.plot(a_line, 2.0 * p / 3.0 - a_line, "r-", linewidth=1.5, alpha=0.8, label=r"$a+b+c=P$ with $c=P/3$")
        ax.scatter([a_opt], [b_opt], c="red", s=160, marker="*", edgecolors="white", zorder=6)
        fig.colorbar(cs, ax=ax, label="Area (Heron)")
        ax.set_xlabel("a")
        ax.set_ylabel("b")
        ax.legend(loc="upper right")
        ax.set_title("Triangle area in (a, b) — c = P − a − b")
        fig.tight_layout()
        return fig

    if sh == "circle":
        P = constraint_value
        r_opt = float(optimal_dims["radius"])
        r = np.linspace(r_opt * 0.3, r_opt * 1.7, 80)
        lam = np.linspace(0.1, P / (2 * math.pi), 80)
        Rg, Lam = np.meshgrid(r, lam)
        Z = np.pi * Rg**2 - Lam * (2 * math.pi * Rg - P)
        fig, ax = plt.subplots(figsize=(6.5, 5))
        cs = ax.contourf(Rg, Lam, Z, levels=30, cmap="magma")
        ax.scatter([r_opt], [r_opt], c="cyan", s=120, marker="*", edgecolors="white", zorder=5)
        fig.colorbar(cs, ax=ax, label=r"$\mathcal{L}(r,\lambda)$")
        ax.set_xlabel("r")
        ax.set_ylabel("λ")
        ax.set_title("Circle Lagrangian")
        fig.tight_layout()
        return fig

    if sh in ("rectangle", "parallelogram"):
        P = constraint_value
        if sh == "rectangle":
            l_opt = float(optimal_dims["length"])
            w_opt = float(optimal_dims["width"])
        else:
            # Parallelogram solver stores adjacent sides a, b (optimum: a=b=P/4)
            l_opt = float(optimal_dims["side_a"])
            w_opt = float(optimal_dims["side_b"])
        l = np.linspace(0.01, P / 2.0 - 0.01, 120)
        w = np.linspace(0.01, P / 2.0 - 0.01, 120)
        Lg, Wg = np.meshgrid(l, w)
        Z = Lg * Wg
        fig, ax = plt.subplots(figsize=(6.5, 5))
        cs = ax.contourf(Lg, Wg, Z, levels=30, cmap="viridis", alpha=0.85)
        lc = np.linspace(0.01, P / 2.0 - 0.01, 200)
        wc = P / 2.0 - lc
        ax.plot(lc, wc, "r-", linewidth=2, label=r"$2l+2w=P$")
        ax.scatter([l_opt], [w_opt], c="red", s=200, marker="*", edgecolors="white", zorder=6, label="Optimum")
        # Gradient of lw is (w, l) — a few arrows toward optimum
        for li, wi in [(l_opt * 0.3, w_opt * 0.3), (l_opt * 0.6, w_opt * 0.6)]:
            ax.annotate(
                "",
                xy=(l_opt * 0.95, w_opt * 0.95),
                xytext=(li, wi),
                arrowprops=dict(arrowstyle="->", color="yellow", lw=1.5),
            )
        fig.colorbar(cs, ax=ax, label="f(l,w) = lw")
        ax.set_xlabel("l")
        ax.set_ylabel("w")
        ax.legend(loc="upper right")
        ax.set_title("Lagrangian contour — rectangle / parallelogram slice")
        fig.tight_layout()
        return fig

    if sh == "rhombus":
        P = constraint_value
        s = P / 4.0
        th = np.linspace(0.2, math.pi - 0.2, 80)
        lam = np.linspace(0.1, s, 80)
        Tg, Lam = np.meshgrid(th, lam)
        Z = s**2 * np.sin(Tg) - Lam * (Tg - math.pi / 2.0)
        th_opt = float(optimal_dims["angle_rad"])
        lam_opt = (s**2) * math.cos(th_opt)
        fig, ax = plt.subplots(figsize=(6.5, 5))
        cs = ax.contourf(Tg, Lam, Z, levels=28, cmap="plasma")
        ax.scatter([th_opt], [lam_opt], c="cyan", s=120, marker="*", edgecolors="white", zorder=5)
        fig.colorbar(cs, ax=ax, label=r"$\mathcal{L}(\theta,\lambda)$ (s fixed)")
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel("λ")
        ax.set_title("Rhombus — slice with fixed side")
        fig.tight_layout()
        return fig

    if sh == "cylinder":
        V = constraint_value
        r_opt = float(optimal_dims["radius"])
        h_opt = float(optimal_dims["height"])
        r = np.linspace(r_opt * 0.3, r_opt * 1.8, 80)
        lam = np.linspace(0.1, 3.0 / r_opt, 80)
        Rg, Lam = np.meshgrid(r, lam)
        # L(r,λ) with h fixed at optimal h* (slice): L = 2πr² + 2πr h* - λ(π r² h* - V)
        Z = (
            2 * math.pi * Rg**2
            + 2 * math.pi * Rg * h_opt
            - Lam * (math.pi * Rg**2 * h_opt - V)
        )
        fig, ax = plt.subplots(figsize=(6.5, 5))
        cs = ax.contourf(Rg, Lam, Z, levels=30, cmap="plasma")
        ax.scatter([r_opt], [2.0 / r_opt], c="cyan", s=120, marker="*", edgecolors="black", zorder=5)
        fig.colorbar(cs, ax=ax, label=r"$\mathcal{L}(r,\lambda)$ slice (h fixed at $h^\ast$)")
        ax.set_xlabel("r")
        ax.set_ylabel("λ")
        ax.set_title("Lagrangian cross-section (cylinder)")
        fig.tight_layout()
        return fig

    if sh == "sphere":
        V = constraint_value
        r_opt = float(optimal_dims["radius"])
        r = np.linspace(r_opt * 0.4, r_opt * 1.6, 80)
        lam = np.linspace(0.5, 12.0 / r_opt, 80)
        Rg, Lam = np.meshgrid(r, lam)
        Z = 4 * math.pi * Rg**2 - Lam * ((4.0 / 3.0) * math.pi * Rg**3 - V)
        fig, ax = plt.subplots(figsize=(6.5, 5))
        cs = ax.contourf(Rg, Lam, Z, levels=30, cmap="plasma")
        ax.scatter([r_opt], [2.0 / r_opt], c="orange", s=120, marker="*", edgecolors="white", zorder=5, label=r"$\lambda=2/r$")
        fig.colorbar(cs, ax=ax, label=r"$\mathcal{L}(r,\lambda)$")
        ax.set_xlabel("r")
        ax.set_ylabel("λ")
        ax.set_title("Sphere Lagrangian slice")
        fig.tight_layout()
        return fig

    if sh == "box":
        V = constraint_value
        s_opt = float(optimal_dims["length"])
        l = np.linspace(s_opt * 0.4, s_opt * 1.6, 80)
        lam = np.linspace(0.5, 8.0 / s_opt, 80)
        Lg, Lam = np.meshgrid(l, lam)
        w0, h0 = s_opt, s_opt
        Z = 2 * (Lg * w0 + Lg * h0 + w0 * h0) - Lam * (Lg * w0 * h0 - V)
        fig, ax = plt.subplots(figsize=(6.5, 5))
        cs = ax.contourf(Lg, Lam, Z, levels=30, cmap="cividis")
        ax.scatter([s_opt], [4.0 / s_opt], c="orange", s=120, marker="*", edgecolors="black", zorder=5)
        fig.colorbar(cs, ax=ax, label=r"$\mathcal{L}(l,\lambda)$ slice ($w,h$ fixed at $w^\ast,h^\ast$)")
        ax.set_xlabel("l")
        ax.set_ylabel("λ")
        ax.set_title("Lagrangian cross-section (box)")
        fig.tight_layout()
        return fig

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.text(0.5, 0.5, "Contour not shown for this shape.", ha="center", va="center", color="#94a3b8")
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_comparison_bar(
    shape: str,
    optimal_value: float,
    suboptimal_value: float,
    label: str,
    *,
    maximize: bool = False,
) -> Figure:
    """Bar chart comparing optimal vs non-optimal objective values."""
    fig, ax = plt.subplots(figsize=(6, 4))
    cats = ["Optimal Shape", "Non-Optimal Shape"]
    vals = [optimal_value, suboptimal_value]
    colors = ["#34d399", "#f87171"]
    bars = ax.bar(cats, vals, color=colors, edgecolor="#334155")
    ax.set_ylabel(label)
    pct = 0.0
    if suboptimal_value > 0:
        if maximize:
            pct = 100.0 * (optimal_value - suboptimal_value) / suboptimal_value
        else:
            pct = 100.0 * (suboptimal_value - optimal_value) / suboptimal_value
    ax.set_title(f"Optimal vs Non-Optimal {label} Comparison")
    ax.text(
        0.5,
        0.95,
        f"Improvement: {pct:.2f}%",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        color="#e2e8f0",
        bbox=dict(boxstyle="round", facecolor="#1e293b", alpha=0.95, edgecolor="#475569"),
    )
    for b, v in zip(bars, vals, strict=True):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.tight_layout()
    return fig
