# 📐 Geometric Shape Optimizer using Lagrangian Method (NLPP)

This project demonstrates **nonlinear programming with one equality constraint** (NLPP) on classic geometry problems. Each problem is solved with the **Lagrangian** and **Karush–Kuhn–Tucker (KKT)** conditions in **closed form**—no black-box numerical optimizers.

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

## 📁 Project Structure

```text
lagrangian-shape-optimizer/
├── app.py
├── lagrangian_solver.py
├── shape_data.py
├── visualizer.py
├── requirements.txt
├── render.yaml
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

Optional CLI check:

```bash
python lagrangian_solver.py
```

---

## 🌐 Live Demo

**Placeholder:** `https://lagrangian-shape-optimizer.onrender.com`  
(Replace with your Render URL after deployment.)

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

| Name | GitHub | Role |
|------|--------|------|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

---

## 📚 References

1. Bazaraa, M. S., Sherali, H. D., & Shetty, C. M. *Nonlinear Programming: Theory and Algorithms* (Wiley).
2. Boyd, S., & Vandenberghe, L. *Convex Optimization* (Cambridge University Press).
3. Courant, R., & Hilbert, D. *Methods of Mathematical Physics* (Wiley) — classical variational problems.
