# Mathematical formulation (all eight shapes)

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
