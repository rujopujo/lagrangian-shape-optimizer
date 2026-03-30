# 📐 Geometric Shape Optimizer using Lagrangian Method (NLPP)

This project demonstrates **nonlinear programming with one equality constraint** (NLPP) on classic geometry problems. The main path uses the **Lagrangian** and **Karush–Kuhn–Tucker (KKT)** conditions in **closed form**. A separate module, **`numeric_shape_solver.py`**, solves the **same** models with **SciPy**’s **SLSQP** (a general numeric solver) so you can compare analytic vs black-box results.

---

## 📌 Problem Statement

Each problem has **one equality constraint** (perimeter/circumference or volume). We form $\mathcal{L} = f - \lambda g$ and use **KKT** / symmetry for a **closed-form** optimum.

**2D (maximize area for fixed perimeter, unless noted)**  
- **Triangle** — equilateral triangle maximizes area.  
- **Circle** — maximize $\pi r^2$ with $2\pi r = P$.  
- **Rectangle** — maximize $lw$ with $2l+2w=P$ (optimum: square).  
- **Parallelogram** — maximize $ab\sin\theta$ with $2a+2b=P$ (optimum: rectangle / square).  
- **Rhombus** — maximize $s^2\sin\theta$ with $4s=P$ (optimum: square).

**3D (minimize surface area for fixed volume)**  
- **Cylinder** — includes top and bottom; optimum satisfies $h=2r$.  
- **Rectangular box** — optimum is a **cube**.  
- **Sphere** — minimize $4\pi r^2$ with $\frac{4}{3}\pi r^3 = V$.

General quadrilaterals (e.g. arbitrary trapezoids) need more parameters or constraints than a single $g=0$; this app focuses on clean one-constraint NLPPs with analytic solutions.

---

## ⚙️ How this project works (code)

At a high level the **Streamlit** UI calls the **closed-form** solver only; it does not invoke SciPy at runtime.

1. **`shape_data.py`** — Central metadata for each supported shape: whether the constraint is **perimeter / circumference** or **volume**, human-readable dimension names, validation rules, LaTeX snippets for the UI, and a **feasible “competitor”** geometry used only to illustrate how much better the KKT optimum is.

2. **`lagrangian_solver.py`** — A `LagrangianShapeSolver` is constructed with a shape name and a single constraint value ($P$ or $V$). Its `solve()` method **branches** to the analytic KKT / symmetry solution for that shape (same formulas as in the README), computes the optimal objective (area or surface area), optional step-by-step text for the UI, and constraint residuals as a sanity check.

3. **`app.py`** — Reads the user’s shape and input, calls `validate_input`, runs the solver, and displays metrics, **optimal vs competitor** comparison, and plots.

4. **`visualizer.py`** — Builds **Matplotlib** figures: 2D schematics where relevant, objective-vs-parameter curves, Lagrangian contour sketches, and a comparison bar chart. Figures are returned as `Figure` objects for `st.pyplot`.

5. **`mathematical_formulation.md`** — Shipped as reference text; the Mathematics expander in the app renders this file so users can read the full eight-shape write-up in one place.

6. **`numeric_shape_solver.py`** — Optional **general numeric** path: **`NumericShapeSolver`** encodes the same objective and equality constraint $g(\mathbf{x})=0$ per shape and uses **`scipy.optimize.minimize` … `method="SLSQP"`** (sequential least squares programming) with variable bounds. **`compare_analytic_vs_numeric(shape, value)`** runs the closed-form solver and the numeric solver and reports relative error on the objective—useful to show that KKT solutions match what a black-box constrained optimizer finds. Run `python numeric_shape_solver.py` for a batch comparison.

Together, this is an **end-to-end demo**: from NLPP definition → KKT solution → visualization, with everything reproducible from the command line (`python lagrangian_solver.py`) or the browser, plus an optional numeric cross-check (`python numeric_shape_solver.py`).

---

## 🌍 Applications & where this kind of optimization shows up

**Everyday intuition**

- **Fence / land use:** “What rectangle gives the largest garden for a fixed amount of fencing?” is exactly the rectangle problem here (optimum: a square). Similar questions appear in agriculture, landscaping, and zoning when area must be maximized under a boundary limit.
- **Packaging and material cost:** For a **fixed volume**, shapes with **smaller surface area** use less material (metal, cardboard, plastic). The cylinder and box results (e.g. can-like $h=2r$, cube-like box) mirror classical **operations research** and **packaging design** trade-offs—minimize material or coating subject to volume.

**Engineering and design**

- **Structural and mechanical design** often minimizes weight or cost subject to stress or volume constraints; the mathematics scales to more variables, but the **Lagrangian and KKT conditions** remain the theoretical backbone.
- **CAD / simulation** workflows sometimes expose optimization (parameter sweeps, shape optimization). Large industrial problems use **numerical** solvers, but the **same KKT theory** explains what “optimal” means at a constraint.

**Software and tools that use constrained optimization (related ideas)**

- **Numerical libraries:** **SciPy** (`scipy.optimize`), **NLopt**, **IPOPT**, and **CVXPY** solve constrained problems when closed forms are unavailable; many implement or approximate **KKT-based** methods.
- **Machine learning:** Constrained training (e.g. penalties, projected gradients, **Lagrangians** in constrained objectives) uses related calculus-of-variations ideas at scale.
- **Simulation suites:** **MATLAB Optimization Toolbox**, **ANSYS**, **COMSOL**, and similar tools include constrained optimization for geometry and physics models—again, often numerical, but grounded in the same NLPP / KKT framework illustrated here.

This repository stays intentionally small: it **implements the math in closed form** so the Lagrangian method is transparent. Real production software typically chains **models + numerical optimization**; understanding these toy geometries makes that stack easier to reason about.

---

## 🧮 Mathematical Formulation

### 1. Rectangle (maximize area)

**Objective:** $f(l,w) = lw$  

**Constraint:** $g(l,w) = 2l + 2w - P = 0$

**Lagrangian:** $\mathcal{L}(l,w,\lambda) = lw - \lambda(2l + 2w - P)$

**KKT:** $w - 2\lambda = 0$, $l - 2\lambda = 0$, $2l + 2w = P$  

**Solution:** $l^\ast = w^\ast = P/4$ (a **square** maximizes area).

---

### 2. Cylinder (minimize surface area)

**Objective:** $f(r,h) = 2\pi r^2 + 2\pi r h$

**Constraint:** $g(r,h) = \pi r^2 h - V = 0$

**Lagrangian:** $\mathcal{L} = 2\pi r^2 + 2\pi r h - \lambda(\pi r^2 h - V)$

**KKT (algebra):** From $\partial\mathcal{L}/\partial h = 0$ get $\lambda = 2/r$; substituting into $\partial\mathcal{L}/\partial r = 0$ yields **$h^\ast = 2r^\ast$** (height equals **diameter**). Then $2\pi r^3 = V$.

---

### 3. Box (minimize surface area)

**Objective:** $f(l,w,h) = 2(lw + lh + wh)$

**Constraint:** $g = lwh - V = 0$

**Lagrangian:** $\mathcal{L} = 2(lw+lh+wh) - \lambda(lwh - V)$

**KKT:** symmetry gives **$l^\ast = w^\ast = h^\ast = V^{1/3}$** (a **cube** minimizes surface area for fixed volume).

---

### 4. Triangle (maximize area, equilateral)

**Objective:** $f(a,b,c) = A(a,b,c) = \sqrt{s(s-a)(s-b)(s-c)}$ with $s = (a+b+c)/2$ (Heron).

**Constraint:** $g(a,b,c) = a+b+c - P = 0$

**Lagrangian:** $\mathcal{L}(a,b,c,\lambda) = A(a,b,c) - \lambda(a+b+c-P)$

**KKT / symmetry:** Stationarity together with symmetry under permuting $(a,b,c)$ implies **$a^\ast=b^\ast=c^\ast=P/3$** (equilateral triangle).

**Solution:** **$a^\ast=b^\ast=c^\ast=P/3$**, **$A^\ast = \dfrac{\sqrt{3}}{36}P^2$**.

---

### 5. Circle (maximize area)

**Objective:** $f(r) = \pi r^2$

**Constraint:** $g(r) = 2\pi r - P = 0$ (circumference)

**Lagrangian:** $\mathcal{L}(r,\lambda) = \pi r^2 - \lambda(2\pi r - P)$

**KKT:** $\dfrac{d\mathcal{L}}{dr} = 2\pi r - 2\pi\lambda = 0$, and $2\pi r = P$

**Solution:** **$r^\ast = P/(2\pi)$**, **$A^\ast = P^2/(4\pi)$**.

---

### 6. Parallelogram (maximize area)

**Objective:** $f(a,b,\theta) = ab\sin\theta$

**Constraint:** $g = 2a + 2b - P = 0$

**Lagrangian:** $\mathcal{L}(a,b,\theta,\lambda) = ab\sin\theta - \lambda(2a+2b-P)$

**KKT:** $\partial\mathcal{L}/\partial\theta = ab\cos\theta = 0 \Rightarrow \theta^\ast=\pi/2$ (**rectangle**). Then $\partial\mathcal{L}/\partial a = b\sin\theta - 2\lambda = 0$ and $\partial\mathcal{L}/\partial b = a\sin\theta - 2\lambda = 0$ with $2a+2b=P$ give $a=b$.

**Solution:** **$a^\ast=b^\ast=P/4$**, $\theta^\ast=\pi/2$, **$A^\ast=(P/4)^2$** (a **square**).

---

### 7. Rhombus (maximize area)

**Objective:** $f(s,\theta) = s^2\sin\theta$

**Constraint:** $g = 4s - P = 0$ (fixing $s = P/4$)

**Lagrangian:** $\mathcal{L}(s,\theta,\lambda) = s^2\sin\theta - \lambda(4s-P)$

**KKT:** $\partial\mathcal{L}/\partial\theta = s^2\cos\theta = 0 \Rightarrow \theta^\ast=\pi/2$; with $s=P/4$ from the constraint.

**Solution:** **$A^\ast=(P/4)^2$** (a **square**).

---

### 8. Sphere (minimize surface area)

**Objective:** $f(r) = 4\pi r^2$

**Constraint:** $g(r) = \frac{4}{3}\pi r^3 - V = 0$

**Lagrangian:** $\mathcal{L}(r,\lambda) = 4\pi r^2 - \lambda\left(\frac{4}{3}\pi r^3 - V\right)$

**KKT:** $\dfrac{d\mathcal{L}}{dr} = 8\pi r - 4\pi\lambda r^2 = 0 \Rightarrow \lambda = 2/r$, and $\frac{4}{3}\pi r^3 = V$

**Solution:** **$r^\ast = \left(\dfrac{3V}{4\pi}\right)^{1/3}$**, **$S^\ast = 4\pi r^{\ast 2}$**.

---

## 📁 Project Structure

```text
lagrangian-shape-optimizer/
├── app.py
├── mathematical_formulation.md
├── lagrangian_solver.py
├── numeric_shape_solver.py
├── shape_data.py
├── visualizer.py
├── requirements.txt
├── run_local.ps1
├── .streamlit/
│   └── config.toml
└── README.md
```

---

## 🚀 How to Run Locally

Requires **Python 3.10+** (see **Technologies Used** for the recommended version).

```bash
pip install -r requirements.txt
streamlit run app.py
```

Optional CLI checks:

```bash
python lagrangian_solver.py
python numeric_shape_solver.py
```

---

## 🌐 Live Demo

**Streamlit Community Cloud:** [lagrangian-shape-optimizer.streamlit.app](https://lagrangian-shape-optimizer.streamlit.app/)

---

## 📊 Features

- Closed-form **Lagrangian / KKT** solutions for **eight** one-constraint geometry NLPPs (2D area max, 3D surface min).
- **Streamlit** UI with LaTeX, metrics, and plots.
- **Comparison** of optimal vs a simple non-optimal feasible competitor.
- **Matplotlib** figures (returned as `Figure` objects for `st.pyplot`).

---

## 🛠️ Technologies Used

- Python 3.11+
- NumPy
- Matplotlib
- Streamlit

---

## 👥 Contributors

| Name | GitHub |
|------|--------|
|Ruhaan Joshi|[rujopujo](https://github.com/rujopujo)|
|Om Thakur|[omthakur9819](https://github.com/omthakur9819)|
|Sarthak Bongane|[sarthak03bongane](https://github.com/sarthak03bongane)|
|Rudra Jain|[Rudrajain756](https://github.com/Rudrajain756)|
|Shubham Singh|[shubhamsinnn](https://github.com/shubhamsinnn)|
|Shamita Chavan|[sha11082006-gif](https://github.com/sha11082006-gif)|

---

## 📄 License & Use

This repository is maintained for an **academic mini-project** (Engineering Mathematics — NLPP / optimization demonstration). Reuse or redistribution beyond course requirements is at your discretion; cite the project if you build on it.

---

## 📚 References

1. Bazaraa, M. S., Sherali, H. D., & Shetty, C. M. *Nonlinear Programming: Theory and Algorithms* (Wiley).
2. Boyd, S., & Vandenberghe, L. *Convex Optimization* (Cambridge University Press).
3. Courant, R., & Hilbert, D. *Methods of Mathematical Physics* (Wiley) — classical variational problems.
