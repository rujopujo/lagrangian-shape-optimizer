from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import streamlit as st

from utils.general_numeric_lagrangian import solve_general_numeric_lagrangian

st.set_page_config(page_title="Lagrangian Solver", page_icon="∇", layout="wide", initial_sidebar_state="expanded")

# Match main app (app.py): dark gradient, system fonts, sidebar glass, panels.
st.markdown(
    r"""
<style>
  html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text",
      "Segoe UI", system-ui, "Helvetica Neue", Helvetica, Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased;
  }
  .stApp {
    background: linear-gradient(180deg, #000000 0%, #0d0d0f 45%, #000000 100%);
    color: #f5f5f7;
  }
  [data-testid="stHeader"] { background: transparent; }
  [data-testid="stSidebar"] {
    background: rgba(28, 28, 30, 0.94) !important;
    backdrop-filter: saturate(180%) blur(20px);
    border-right: 0.5px solid rgba(255, 255, 255, 0.1) !important;
  }
  [data-testid="stSidebar"] * { color: #f5f5f7 !important; }
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] .stMarkdown { color: #a1a1a6 !important; }
  section[data-testid="stMain"] { background: transparent; }
  section[data-testid="stMain"] h1, section[data-testid="stMain"] h2, section[data-testid="stMain"] h3 {
    color: #f5f5f7 !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em;
  }
  .apple-hero-row {
    display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between;
    gap: 1.25rem 2rem; padding: 0.5rem 0 1.5rem 0; margin-bottom: 0.5rem;
    border-bottom: 0.5px solid rgba(255, 255, 255, 0.08);
  }
  .apple-title {
    margin: 0 0 0.35rem 0; font-size: clamp(1.65rem, 4vw, 2.25rem);
    font-weight: 600; letter-spacing: -0.04em; line-height: 1.1; color: #f5f5f7;
  }
  .apple-subtitle { margin: 0; font-size: 1.05rem; color: #86868b; max-width: 38rem; line-height: 1.45; }
  .apple-pill {
    display: inline-block; font-size: 0.6875rem; font-weight: 500;
    padding: 0.45rem 0.85rem; border-radius: 980px;
    color: rgba(255, 255, 255, 0.92);
    background: rgba(255, 255, 255, 0.08);
    border: 0.5px solid rgba(255, 255, 255, 0.14);
  }
  .panel-dark {
    background: rgba(44, 44, 46, 0.55);
    border: 0.5px solid rgba(255, 255, 255, 0.1);
    border-radius: 18px;
    padding: 1.15rem 1.35rem;
    backdrop-filter: saturate(180%) blur(16px);
    margin-bottom: 0.75rem;
  }
  .sample-card-title { font-size: 1rem; font-weight: 600; color: #f5f5f7; margin: 0 0 0.35rem 0; letter-spacing: -0.02em; }
  .sample-card-desc { font-size: 0.875rem; color: #a1a1a6; margin: 0 0 0.75rem 0; line-height: 1.45; }
  .section-label { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #86868b; margin: 1.25rem 0 0.65rem 0; }
  .result-banner {
    background: rgba(48, 209, 88, 0.12);
    border: 0.5px solid rgba(48, 209, 88, 0.35);
    border-radius: 14px;
    padding: 0.85rem 1.1rem;
    margin: 0.5rem 0 1rem 0;
    color: #d8ffe6 !important;
    font-size: 0.9375rem;
  }
  [data-testid="stExpander"] {
    background: rgba(44, 44, 46, 0.45);
    border: 0.5px solid rgba(255, 255, 255, 0.1);
    border-radius: 14px;
  }
  div[data-testid="stMetric"] {
    background: rgba(44, 44, 46, 0.65) !important;
    border: 0.5px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 14px !important;
    padding: 0.65rem 0.85rem !important;
  }
  .stButton > button[kind="primary"] {
    background: #0a84ff !important; border: none !important; color: #fff !important;
    font-weight: 600 !important; border-radius: 980px !important;
    padding: 0.5rem 1.15rem !important;
  }
  .footer-mini { text-align: center; font-size: 0.8125rem; color: #86868b !important; padding-top: 1.5rem; }
</style>
""",
    unsafe_allow_html=True,
)

SAFE_GLOBALS = {"__builtins__": {}}
SAFE_LOCALS = {"math": math, "np": np, "abs": abs, "min": min, "max": max}

PRESETS = {
    "Custom": None,
    "3-variable (sum=1)": {
        "n": 3,
        "objective": "x[0]**2 + x[1]**2 + x[2]**2",
        "eq": ["x[0] + x[1] + x[2] - 1"],
        "ineq": [],
        "x0": [1.0, 1.0, 1.0],
    },
    "4-variable (sum=1, x0=x1)": {
        "n": 4,
        "objective": "x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2",
        "eq": ["x[0] + x[1] + x[2] + x[3] - 1", "x[0] - x[1]"],
        "ineq": [],
        "x0": [1.0, 1.0, 1.0, 1.0],
    },
    "5-variable (sum=1)": {
        "n": 5,
        "objective": "x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2",
        "eq": ["x[0] + x[1] + x[2] + x[3] + x[4] - 1"],
        "ineq": [],
        "x0": [1.0, 1.0, 1.0, 1.0, 1.0],
    },
}


if "__pending_n_vars_input" in st.session_state:
    st.session_state["n_vars_input"] = int(st.session_state.pop("__pending_n_vars_input"))


def _apply_preset(name: str) -> None:
    cfg = PRESETS.get(name)
    if not cfg:
        return
    st.session_state["__pending_n_vars_input"] = int(cfg["n"])
    st.session_state["objective_expr"] = cfg["objective"]
    st.session_state["eq_constraints"] = list(cfg["eq"])
    st.session_state["ineq_constraints"] = list(cfg["ineq"])
    st.session_state["x0"] = list(cfg["x0"])
    st.session_state["var_names"] = [f"x{i+1}" for i in range(cfg["n"])]


def _ensure_state(n: int) -> None:
    st.session_state.setdefault("objective_expr", "x[0]**2 + x[1]**2")
    st.session_state.setdefault("eq_constraints", ["x[0] + x[1] - 1"])
    st.session_state.setdefault("ineq_constraints", [])
    st.session_state.setdefault("x0", [1.0] * n)
    st.session_state.setdefault("var_names", [f"x{i+1}" for i in range(n)])


def _resize_list(values: list, n: int, fill):
    if len(values) < n:
        values.extend([fill for _ in range(n - len(values))])
    elif len(values) > n:
        del values[n:]
    return values


def _compile_expr(expr: str) -> Callable[[np.ndarray], float]:
    code = compile(expr, "<user_expression>", "eval")

    def _fn(x: np.ndarray) -> float:
        env = dict(SAFE_LOCALS)
        env["x"] = x
        return float(eval(code, SAFE_GLOBALS, env))  # noqa: S307

    return _fn


def _load_preset_callback() -> None:
    _apply_preset(preset)
    st.session_state["__last_loaded_preset"] = preset


def _sample_3_callback() -> None:
    _apply_preset("3-variable (sum=1)")
    st.session_state["__last_loaded_preset"] = "3-variable (sum=1)"


def _sample_4_callback() -> None:
    _apply_preset("4-variable (sum=1, x0=x1)")
    st.session_state["__last_loaded_preset"] = "4-variable (sum=1, x0=x1)"


def _sample_5_callback() -> None:
    _apply_preset("5-variable (sum=1)")
    st.session_state["__last_loaded_preset"] = "5-variable (sum=1)"


st.markdown(
    """
<div class="apple-hero-row">
  <div>
    <div class="apple-title">Lagrangian Solver</div>
    <p class="apple-subtitle">
      Numeric constrained optimization (SciPy SLSQP). Type expressions with <code>x[0]</code>, <code>x[1]</code>, …
      For class demos, use the quick examples below.
    </p>
  </div>
  <div><span class="apple-pill">EM-IV · local branch</span></div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Settings")
    n_vars = st.number_input(
        "Variables (n)",
        min_value=2,
        max_value=10,
        value=5,
        step=1,
        key="n_vars_input",
        help="How many entries in vector x.",
    )
    direction = st.radio(
        "Goal",
        options=["Minimize", "Maximize"],
        index=0,
        horizontal=True,
    )
    st.divider()
    preset = st.selectbox("Preset from list", options=list(PRESETS.keys()), index=0)
    st.button("Apply preset", on_click=_load_preset_callback, use_container_width=True)

_ensure_state(n_vars)
st.session_state["x0"] = _resize_list(st.session_state["x0"], n_vars, 1.0)
st.session_state["var_names"] = _resize_list(st.session_state["var_names"], n_vars, "")

_last = st.session_state.pop("__last_loaded_preset", None)
if _last:
    st.markdown(
        f'<div class="result-banner">Loaded: <strong>{_last}</strong></div>',
        unsafe_allow_html=True,
    )

st.markdown('<p class="section-label">Quick examples</p>', unsafe_allow_html=True)
q1, q2, q3 = st.columns(3)
with q1:
    st.markdown(
        """
<div class="panel-dark">
  <div class="sample-card-title">3 variables</div>
  <p class="sample-card-desc">Minimize ∑xᵢ² with one equality (sum = 1). Good starter for viva.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.button("Load example", on_click=_sample_3_callback, use_container_width=True, key="btn_s3")
with q2:
    st.markdown(
        """
<div class="panel-dark">
  <div class="sample-card-title">4 variables</div>
  <p class="sample-card-desc">Same idea plus x₀ = x₁ — shows two equalities at once.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.button("Load example", on_click=_sample_4_callback, use_container_width=True, key="btn_s4")
with q3:
    st.markdown(
        """
<div class="panel-dark">
  <div class="sample-card-title">5 variables</div>
  <p class="sample-card-desc">Higher dimension, still one sum constraint — symmetric optimum.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.button("Load example", on_click=_sample_5_callback, use_container_width=True, key="btn_s5")

st.markdown('<p class="section-label">Problem</p>', unsafe_allow_html=True)
st.subheader("Variables & initial guess")
for i in range(0, n_vars, 3):
    cols = st.columns(3)
    for j, col in enumerate(cols):
        idx = i + j
        if idx >= n_vars:
            continue
        with col:
            st.session_state["var_names"][idx] = st.text_input(
                f"Name · x[{idx}]",
                value=st.session_state["var_names"][idx] or f"x{idx+1}",
                key=f"var_name_{idx}",
            )
            st.session_state["x0"][idx] = st.number_input(
                f"Guess · x[{idx}]",
                value=float(st.session_state["x0"][idx]),
                key=f"x0_{idx}",
            )

st.subheader("Objective")
st.session_state["objective_expr"] = st.text_area(
    "f(x) — Python expression",
    value=st.session_state["objective_expr"],
    height=110,
    key="objective_expr_input",
    placeholder="e.g. x[0]**2 + x[1]**2",
)

with st.expander("Equalities · gᵢ(x) = 0", expanded=True):
    c_add, c_rem = st.columns(2)
    with c_add:
        if st.button("+ Add equality", key="add_eq"):
            st.session_state["eq_constraints"].append("0")
    with c_rem:
        if st.session_state["eq_constraints"] and st.button("− Remove last", key="rem_eq"):
            st.session_state["eq_constraints"].pop()
    for i, expr in enumerate(st.session_state["eq_constraints"]):
        st.session_state["eq_constraints"][i] = st.text_input(
            f"g{i+1}(x) = 0",
            value=expr,
            key=f"eq_{i}",
        )

with st.expander("Inequalities · hⱼ(x) ≤ 0 (optional)", expanded=False):
    c_add, c_rem = st.columns(2)
    with c_add:
        if st.button("+ Add inequality", key="add_ineq"):
            st.session_state["ineq_constraints"].append("0")
    with c_rem:
        if st.session_state["ineq_constraints"] and st.button("− Remove last", key="rem_ineq"):
            st.session_state["ineq_constraints"].pop()
    for i, expr in enumerate(st.session_state["ineq_constraints"]):
        st.session_state["ineq_constraints"][i] = st.text_input(
            f"h{i+1}(x) ≤ 0",
            value=expr,
            key=f"ineq_{i}",
        )

st.divider()
if st.button("Run solver", type="primary"):
    try:
        objective_fn = _compile_expr(st.session_state["objective_expr"])
        eq_fns = [_compile_expr(expr) for expr in st.session_state["eq_constraints"] if expr.strip()]
        ineq_fns = [_compile_expr(expr) for expr in st.session_state["ineq_constraints"] if expr.strip()]
        x0 = np.asarray(st.session_state["x0"][:n_vars], dtype=float)

        _ = objective_fn(x0)
        for g in eq_fns:
            _ = g(x0)
        for h in ineq_fns:
            _ = h(x0)

        result = solve_general_numeric_lagrangian(
            objective=objective_fn,
            equality_constraints=eq_fns,
            inequality_constraints=ineq_fns,
            x0=x0,
            variable_names=st.session_state["var_names"][:n_vars],
            maximize=direction == "Maximize",
            method="SLSQP",
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not run: {exc}")
        st.caption("Check syntax (brackets, operators). Try another initial guess if the solver is picky.")
    else:
        if result["success"]:
            st.success(result["message"])
        else:
            st.warning(result["message"])
            st.caption("Often helps: nudge the initial guess or loosen constraints.")

        c1, c2, c3 = st.columns(3)
        c1.metric("f(x*)", f"{result['optimal_objective']:.10g}")
        c2.metric("Iterations", str(result["iterations"]))
        c3.metric("Status", str(result["status"]))

        st.subheader("Solution")
        st.dataframe(
            {
                "variable": result["variable_names"],
                "value": np.asarray(result["x_opt"], dtype=float),
            },
            hide_index=True,
            width="stretch",
        )

        st.subheader("Multipliers (numeric)")
        eq_names = [f"λ{i+1}" for i in range(len(result["lambda_eq"]))]
        ineq_names = [f"μ{i+1}" for i in range(len(result["mu_ineq"]))]
        st.dataframe(
            {
                "symbol": eq_names + ineq_names,
                "value": result["lambda_eq"] + result["mu_ineq"],
                "kind": (["equality"] * len(eq_names)) + (["inequality"] * len(ineq_names)),
            },
            hide_index=True,
            width="stretch",
        )

        st.subheader("Constraints at x*")
        eq_rows = [
            {"name": f"g{i+1}", "value": v, "should be": 0.0, "|error|": abs(v)}
            for i, v in enumerate(result["eq_residuals"])
        ]
        ineq_rows = [
            {"name": f"h{i+1}", "value": v, "should be": "≤ 0", "violation": max(v, 0.0)}
            for i, v in enumerate(result["ineq_values"])
        ]
        st.dataframe(eq_rows + ineq_rows, hide_index=True, width="stretch")

        st.subheader("KKT check (approximate)")
        st.dataframe(
            {
                "metric": list(result["kkt_residuals"].keys()),
                "value": list(result["kkt_residuals"].values()),
            },
            hide_index=True,
            width="stretch",
        )

st.markdown(
    '<p class="footer-mini">SLSQP · numerical gradients · for coursework demos</p>',
    unsafe_allow_html=True,
)
