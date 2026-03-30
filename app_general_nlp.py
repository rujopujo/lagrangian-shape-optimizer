"""
Streamlit demo for general N-variable Lagrangian/KKT-style numeric solving.

This app solves:
    minimize or maximize  sum_i (w_i * x_i^2)
subject to:
    sum_i x_i = S
with optional bounds on each x_i.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from lagrangian_nlp_general import solve_lagrangian_nlp_nd

st.set_page_config(page_title="General N-Variable NLPP Solver", page_icon="∇", layout="wide")

st.title("General N-Variable Lagrangian Solver (Numeric)")
st.caption("Branch playground for 3, 4, 5, ... variables before touching production.")


def _extract_numeric_column(table_obj: object, preferred_key: str, n: int) -> np.ndarray:
    """
    Read edited values from Streamlit data_editor across return formats.

    Streamlit versions may return dict-like objects or DataFrames depending on
    input type/config. This helper accepts either and falls back to the first
    available column.
    """
    # dict-like return
    if isinstance(table_obj, dict):
        if preferred_key in table_obj:
            return np.asarray(table_obj[preferred_key], dtype=float).reshape(-1)
        if table_obj:
            first_val = next(iter(table_obj.values()))
            return np.asarray(first_val, dtype=float).reshape(-1)

    # DataFrame-like return
    if hasattr(table_obj, "columns") and hasattr(table_obj, "__getitem__"):
        cols = list(getattr(table_obj, "columns"))
        if preferred_key in cols:
            return np.asarray(table_obj[preferred_key], dtype=float).reshape(-1)
        if cols:
            return np.asarray(table_obj[cols[0]], dtype=float).reshape(-1)

    # Last-resort: try direct array conversion.
    arr = np.asarray(table_obj, dtype=float).reshape(-1)
    if arr.size == n:
        return arr
    raise ValueError(f"Could not read {preferred_key!r} from editor output.")

with st.sidebar:
    st.header("Problem Setup")
    n_dim = st.slider("Number of variables (N)", min_value=3, max_value=12, value=5, step=1)
    mode = st.selectbox("Objective mode", ["Minimize", "Maximize"], index=0)
    target_sum = st.number_input("Constraint sum S (sum(x_i)=S)", min_value=0.1, value=float(n_dim), step=0.5)
    use_bounds = st.checkbox("Use bounds x_i in [0, upper]", value=True)
    upper = st.number_input("Upper bound", min_value=0.1, value=max(1.0, float(target_sum)), step=0.5)
    solver_method = st.selectbox("Numeric method", ["SLSQP", "trust-constr"], index=0)

st.markdown("### Objective")
st.write("`f(x) = Σ w_i x_i²`  with adjustable positive weights.")

default_weights = np.ones(n_dim, dtype=float)
weights = st.data_editor(
    {"w_i": default_weights},
    use_container_width=False,
    num_rows="fixed",
    key="weights_editor",
)
weights_arr = _extract_numeric_column(weights, "w_i", n_dim)
weights_arr = np.where(weights_arr <= 0.0, 1.0, weights_arr)

st.markdown("### Initial Guess")
x0_default = np.full(n_dim, target_sum / n_dim, dtype=float)
x0_table = st.data_editor(
    {"x0_i": x0_default},
    use_container_width=False,
    num_rows="fixed",
    key="x0_editor",
)
x0 = _extract_numeric_column(x0_table, "x0_i", n_dim)


def objective(x: np.ndarray) -> float:
    return float(np.sum(weights_arr * x * x))


def g_sum(x: np.ndarray) -> float:
    return float(np.sum(x) - target_sum)


bounds = ((0.0, upper),) * n_dim if use_bounds else None
maximize = mode == "Maximize"

if st.button("Solve", type="primary"):
    try:
        out = solve_lagrangian_nlp_nd(
            objective=objective,
            equality_constraints=[g_sum],
            x0=x0,
            maximize=maximize,
            bounds=bounds,
            method=solver_method,
        )
    except Exception as exc:  # noqa: BLE001 - UI feedback
        st.error(f"Solver failed: {exc}")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Objective", f"{out['objective']:.10g}")
        c2.metric("Max |constraint residual|", f"{out['max_constraint_violation']:.3e}")
        c3.metric("Iterations", str(out["nit"]))

        st.success(f"Success={out['success']} | {out['message']}")

        st.markdown("### Optimal Variables")
        st.dataframe(
            {
                "index": [f"x{i+1}" for i in range(n_dim)],
                "x*": np.asarray(out["x"], dtype=float),
            },
            use_container_width=False,
            hide_index=True,
        )

        if not maximize:
            # For this demo problem with one equality and positive quadratic weights:
            # x_i ∝ 1 / w_i gives analytic reference.
            inv_w = 1.0 / np.maximum(weights_arr, 1e-12)
            analytic = target_sum * inv_w / np.sum(inv_w)
            rel = np.linalg.norm(np.asarray(out["x"]) - analytic) / max(np.linalg.norm(analytic), 1e-12)
            st.markdown("### Analytic Reference (for this quadratic demo)")
            st.write("For minimization with positive weights, optimal `x_i` is proportional to `1 / w_i`.")
            st.write(f"Relative difference vs analytic reference: `{rel:.3e}`")
