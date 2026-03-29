"""
Streamlit app: geometric shape optimization via Lagrangian / KKT (NLPP).
Dark UI.
"""

from __future__ import annotations

import math

import streamlit as st

from lagrangian_solver import LagrangianShapeSolver
from shape_data import get_sample_data, get_shape_info, uses_perimeter, validate_input
from visualizer import (
    plot_comparison_bar,
    plot_lagrangian_contour,
    plot_objective_vs_dimension,
    plot_shape_2d,
)

st.set_page_config(
    page_title="Shape Optimizer",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apple-inspired typography: system UI fonts (SF on Apple), neutral dark surfaces
st.markdown(
    r"""
<style>
  /* SF Pro on Apple devices; fallbacks match system UI feel */
  html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text",
      "Segoe UI", system-ui, "Helvetica Neue", Helvetica, Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }

  .stApp {
    background: linear-gradient(180deg, #000000 0%, #0d0d0f 45%, #000000 100%);
    color: #f5f5f7;
  }

  [data-testid="stHeader"] { background: transparent; }

  [data-testid="stSidebar"] {
    background: rgba(28, 28, 30, 0.94) !important;
    backdrop-filter: saturate(180%) blur(20px);
    -webkit-backdrop-filter: saturate(180%) blur(20px);
    border-right: 0.5px solid rgba(255, 255, 255, 0.1) !important;
  }

  [data-testid="stSidebar"] * {
    color: #f5f5f7 !important;
  }

  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] .stMarkdown,
  [data-testid="stSidebar"] small,
  [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #a1a1a6 !important;
  }

  [data-testid="stSidebar"] h3 {
    color: #f5f5f7 !important;
    font-weight: 600 !important;
    letter-spacing: -0.025em;
    font-size: 1.125rem !important;
  }

  [data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: rgba(44, 44, 46, 0.9) !important;
    border: 0.5px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 10px !important;
    color: #f5f5f7 !important;
  }

  [data-testid="stSidebar"] input {
    color: #f5f5f7 !important;
  }

  section[data-testid="stMain"] {
    background: transparent;
  }

  section[data-testid="stMain"] h1 {
    color: #f5f5f7 !important;
    font-weight: 600 !important;
    letter-spacing: -0.035em !important;
    font-size: 2.125rem !important;
  }

  section[data-testid="stMain"] h2,
  section[data-testid="stMain"] h3 {
    color: #f5f5f7 !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em;
  }

  /* Hero: Apple-style large title + secondary label */
  .apple-hero-row {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-end;
    justify-content: space-between;
    gap: 1.25rem 2rem;
    padding: 0.5rem 0 1.75rem 0;
    margin-bottom: 0.25rem;
    border-bottom: 0.5px solid rgba(255, 255, 255, 0.08);
  }

  .apple-hero-left {
    flex: 1 1 16rem;
    min-width: 0;
  }

  .apple-title {
    margin: 0 0 0.35rem 0;
    padding: 0;
    font-size: clamp(1.75rem, 4vw, 2.375rem);
    font-weight: 600;
    letter-spacing: -0.045em;
    line-height: 1.08;
    color: #f5f5f7;
    font-family: inherit;
  }

  .apple-subtitle {
    margin: 0;
    padding: 0;
    font-size: 1.0625rem;
    font-weight: 400;
    letter-spacing: -0.015em;
    line-height: 1.45;
    color: #86868b;
    max-width: 36rem;
  }

  .apple-hero-right {
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    padding-bottom: 0.15rem;
  }

  .apple-pill {
    display: inline-block;
    font-size: 0.6875rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    line-height: 1;
    padding: 0.45rem 0.85rem;
    border-radius: 980px;
    color: rgba(255, 255, 255, 0.92);
    background: rgba(255, 255, 255, 0.08);
    border: 0.5px solid rgba(255, 255, 255, 0.14);
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.12);
  }

  .hero-sub {
    color: #86868b !important;
    font-size: 1.0625rem;
    margin: 0 0 1rem 0;
    letter-spacing: -0.015em;
  }

  .pill-tag {
    display: inline-block;
    background: rgba(255, 255, 255, 0.08);
    color: rgba(255, 255, 255, 0.9) !important;
    border: 0.5px solid rgba(255, 255, 255, 0.14);
    font-size: 0.6875rem;
    font-weight: 500;
    padding: 0.4rem 0.8rem;
    border-radius: 980px;
  }

  .panel-dark {
    background: rgba(44, 44, 46, 0.55);
    border: 0.5px solid rgba(255, 255, 255, 0.1);
    border-radius: 18px;
    padding: 1.35rem 1.5rem;
    backdrop-filter: saturate(180%) blur(16px);
  }

  .result-banner {
    background: rgba(48, 209, 88, 0.12);
    border: 0.5px solid rgba(48, 209, 88, 0.35);
    border-radius: 14px;
    padding: 1rem 1.25rem;
    margin-bottom: 1.25rem;
    color: #e8fff0 !important;
    font-size: 0.9375rem;
    letter-spacing: -0.01em;
  }

  .insight-line {
    background: rgba(255, 214, 10, 0.08);
    border-left: 3px solid #ffd60a;
    padding: 0.9rem 1.1rem;
    border-radius: 0 14px 14px 0;
    color: #f5f5f7 !important;
    font-size: 0.9375rem;
    letter-spacing: -0.01em;
  }

  .empty-hint {
    text-align: center;
    padding: 2.25rem 1.5rem;
    color: #a1a1a6 !important;
    font-size: 0.9375rem;
    letter-spacing: -0.01em;
    line-height: 1.5;
  }

  .empty-hint strong {
    color: #f5f5f7 !important;
    font-size: 1.125rem;
    font-weight: 600;
    letter-spacing: -0.025em;
    display: block;
    margin-bottom: 0.5rem;
  }

  div[data-testid="stMetric"] {
    background: rgba(44, 44, 46, 0.65) !important;
    border: 0.5px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 14px !important;
    padding: 0.7rem 0.9rem !important;
  }

  div[data-testid="stMetric"] label {
    color: #86868b !important;
    font-weight: 500 !important;
    font-size: 0.75rem !important;
    letter-spacing: -0.01em;
  }

  div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #f5f5f7 !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em;
  }

  .stButton > button[kind="primary"] {
    background: #0a84ff !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 0.9375rem !important;
    letter-spacing: -0.01em !important;
    border-radius: 980px !important;
    padding: 0.55rem 1.25rem !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
  }

  /* Streamlit nests label text; force solid white on primary buttons */
  .stButton > button[kind="primary"] *,
  .stButton > button[kind="primary"] p,
  .stButton > button[kind="primary"] span,
  .stButton > button[kind="primary"] div {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    opacity: 1 !important;
  }

  .stButton > button[kind="primary"]:hover,
  .stButton > button[kind="primary"]:hover * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
  }

  .stButton > button[kind="primary"]:hover {
    background: #409cff !important;
    filter: none !important;
  }

  [data-testid="stExpander"] {
    background: rgba(44, 44, 46, 0.45);
    border: 0.5px solid rgba(255, 255, 255, 0.1);
    border-radius: 14px;
  }

  .footer-mini {
    text-align: center;
    font-size: 0.8125rem;
    color: #86868b !important;
    padding-top: 1.25rem;
    letter-spacing: -0.01em;
  }

  [data-testid="stCaption"] {
    color: #86868b !important;
  }

  a { color: #64b5f6 !important; }
</style>
""",
    unsafe_allow_html=True,
)

SHAPE_LABEL_TO_KEY: dict[str, str] = {
    "△ Triangle (Equilateral · Max Area)": "triangle",
    "○ Circle (Max Area · Fixed Circumference)": "circle",
    "▭ Rectangle (Max Area)": "rectangle",
    "▱ Parallelogram (Max Area)": "parallelogram",
    "◇ Rhombus (Max Area)": "rhombus",
    "⌭ Cylinder (Min Surface)": "cylinder",
    "▦ Box / Prism (Min Surface)": "box",
    "◯ Sphere (Min Surface)": "sphere",
}

def _sidebar_heading(text: str, *, top_margin: str = "0") -> None:
    """Muted section label in the sidebar."""
    st.markdown(
        f'<p style="color:#86868b;font-size:0.8125rem;font-weight:500;'
        f'margin:{top_margin} 0 0.5rem 0;letter-spacing:-0.01em;">{text}</p>',
        unsafe_allow_html=True,
    )


INSIGHTS: dict[str, str] = {
    "triangle": "Among all triangles with fixed perimeter, the equilateral triangle has the largest area.",
    "circle": "Among all closed curves with fixed perimeter, the circle encloses the largest area (isoperimetric theorem).",
    "rectangle": "The rectangle of maximum area for a fixed perimeter is a square.",
    "parallelogram": "For fixed side lengths, area is maximized when the angle is 90° (a rectangle); with a perimeter constraint this yields a square.",
    "rhombus": "For fixed side length, area is maximized when angles are 90° — a square.",
    "cylinder": "With fixed volume, the minimum surface area occurs when height equals diameter (h = 2r).",
    "box": "With fixed volume, the minimum surface area of a rectangular box occurs for a cube.",
    "sphere": "With fixed volume, the sphere has the smallest surface area of any closed surface (sphere is the 3D isoperimetric solution).",
}

# --- Hero (Apple-style system typography + pill) --------------------------------
st.markdown(
    """
<div class="apple-hero-row">
  <div class="apple-hero-left">
    <h1 class="apple-title">Shape Optimizer</h1>
    <p class="apple-subtitle">Lagrangian And KKT · Closed-Form Classical Geometry.</p>
  </div>
  <div class="apple-hero-right">
    <span class="apple-pill">NLPP · One Equality Constraint</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# --- Sidebar ----------------------------------------------------------------
with st.sidebar:
    st.markdown("### Setup")
    _sidebar_heading("Problem")
    choice = st.selectbox(
        "Problem",
        list(SHAPE_LABEL_TO_KEY.keys()),
        label_visibility="collapsed",
    )
    shape_key = SHAPE_LABEL_TO_KEY[choice]

    _sidebar_heading("Data", top_margin="0.75rem")
    data_mode = st.radio(
        "data_mode",
        ["Sample", "Custom"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if data_mode == "Sample":
        sample = get_sample_data(shape_key)
        cval = float(sample["constraint_value"])
        st.markdown(
            f'<p style="color:#a1a1a6;font-size:0.875rem;margin:0.5rem 0 0 0;letter-spacing:-0.01em;">'
            f"{sample['constraint_label']}: <strong style='color:#f5f5f7;font-weight:600'>{cval:g}</strong></p>",
            unsafe_allow_html=True,
        )
    else:
        if uses_perimeter(shape_key):
            cval = float(st.number_input("Perimeter P", min_value=1.0, value=40.0, step=1.0))
        else:
            cval = float(st.number_input("Volume V", min_value=1.0, value=1000.0, step=10.0))

    st.markdown("")
    run = st.button("Run Optimization", type="primary", use_container_width=True)

# --- Main flow --------------------------------------------------------------
if run:
    try:
        validate_input(shape_key, cval)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    info = get_shape_info(shape_key)
    maximize = info["objective_mode"] == "maximize"

    with st.spinner("Solving…"):
        solver = LagrangianShapeSolver(shape_key, cval)
        result = solver.solve(verbose=False)

    ok = solver.verify_constraint(verbose=False)
    dims = result["optimal_dims"]
    comp = solver.get_comparison()
    imp = comp["improvement_percent"]
    obj_label = "Area" if maximize else "Surface Area"

    if ok:
        st.markdown(
            f'<div class="result-banner"><strong>Constraint Satisfied</strong> — '
            f"g(x) ≈ 0. Versus a simple non-optimal shape: <strong>{imp:.1f}%</strong> better on "
            f"{comp['label'].lower()}.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.warning("Constraint Check Inconclusive — See Metrics.")

    st.markdown("### Results")
    metric_cols = st.columns(len(dims) + 2)
    keys_order = list(dims.keys())
    for i, k in enumerate(keys_order):
        with metric_cols[i]:
            v = float(dims[k])
            if k == "angle_rad":
                st.metric("Angle θ", f"{math.degrees(v):.2f}°")
            else:
                st.metric(k.replace("_", " ").title(), f"{v:.4f}")
    with metric_cols[-2]:
        st.metric("λ", f"{result['lambda_value']:.4f}")
    with metric_cols[-1]:
        st.metric(obj_label, f"{result['optimal_objective']:.4f}")

    st.markdown(
        f'<p class="insight-line"><strong>Insight</strong> — {INSIGHTS.get(shape_key, "")}</p>',
        unsafe_allow_html=True,
    )

    with st.expander("Mathematics", expanded=False):
        st.caption(f"{info['objective_label']} · {info['constraint_label']}")
        st.latex(info["latex_objective"])
        st.latex(info["latex_constraint"])
        st.latex(info["latex_lagrangian"])
        st.markdown("**KKT**")
        for line in info.get("kkt_lines", []):
            st.latex(line)
        st.latex(info["latex_solution"])

    with st.expander("Solver Steps", expanded=False):
        for step in result["steps"]:
            st.write(step)

    st.markdown("### Figures")
    g1, g2 = st.columns(2)
    with g1:
        st.caption("Geometry & Objective")
        fig1 = plot_shape_2d(shape_key, dims)
        st.pyplot(fig1, use_container_width=True)
    with g2:
        st.caption("Objective Vs Parameter")
        fig2 = plot_objective_vs_dimension(shape_key, cval, dims)
        st.pyplot(fig2, use_container_width=True)

    g3, g4 = st.columns(2)
    with g3:
        st.caption("Lagrangian Slice / Contour")
        fig3 = plot_lagrangian_contour(shape_key, cval, dims)
        st.pyplot(fig3, use_container_width=True)
    with g4:
        st.caption("Optimal Vs Non-Optimal")
        fig4 = plot_comparison_bar(
            shape_key,
            comp["optimal_objective"],
            comp["suboptimal_objective"],
            comp["label"],
            maximize=maximize,
        )
        st.pyplot(fig4, use_container_width=True)

    st.markdown(
        '<p class="footer-mini">Shape Optimizer · '
        '<a href="https://github.com/rujopujo/lagrangian-shape-optimizer" style="color:#0a84ff;text-decoration:none;">GitHub</a></p>',
        unsafe_allow_html=True,
    )

else:
    st.markdown(
        """
<div class="panel-dark">
  <div class="empty-hint">
    <strong>Choose A Problem In The Sidebar</strong>
    Use sample or custom constraint values, then <strong>Run Optimization</strong>.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
