# Cylindrical Tank Cost Optimization Algorithm

### Minimum-Cost Cylindrical Tank: Gradient Method with a Penalty Function

- Computational assignment for **(IPRJ, UERJ) Computational Modeling**. The program finds the radius `r` and height `h` of a cylindrical tank that **minimize construction cost** while holding a **minimum volume**.

### The problem

Material costs: base R$ 150/m², lateral wall R$ 100/m², top R$ 200/m². The total cost to minimize is:

$$f(r, h) = 150(\pi r^2) + 100(2\pi r h) + 200(\pi r^2) = 350\pi r^2 + 200\pi r h$$

Without a restriction, `r` and `h` would shrink to zero. So the tank must hold at least 500 m³:

$$g(r, h) = 500 - \pi r^2 h \le 0$$

### The approach

1. **Penalty function.** The restriction is turned into a penalty, giving an unconstrained function to minimize:
   $$P(r, h) = f(r, h) + \mu \cdot [\max(0, g(r, h))]^2$$
   If the volume is ≥ 500, the penalty is zero. If it is below 500, a large `μ` makes the cost explode, pushing the algorithm away.

2. **Finite differences.** The partial derivatives of `P` are computed numerically with a small step `Δ`:
   $$\frac{\partial P}{\partial r} \approx \frac{P(r+\Delta, h) - P(r, h)}{\Delta} \qquad \frac{\partial P}{\partial h} \approx \frac{P(r, h+\Delta) - P(r, h)}{\Delta}$$

3. **Gradient method.** Starting from random values, each iteration walks against the gradient, with step size `α`:
   $$r^{(k+1)} = r^{(k)} - \alpha \frac{\partial P}{\partial r} \qquad h^{(k+1)} = h^{(k)} - \alpha \frac{\partial P}{\partial h}$$

The loop stops when `r` and `h` change by less than `tol`. Every 1000 iterations, `α` is multiplied by 0.95, and `μ` is doubled if the restriction is still violated.

Current parameters: `mu = 10`, `alpha = 1e-6`, `delta = 1e-6`, `tol = 1e-7`, at most 1,000,000 iterations per run.

## Running

```terminal
pip install numpy matplotlib
python main.py
```
