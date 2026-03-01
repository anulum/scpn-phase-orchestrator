# UPDE Numerics

## Equation

```
dtheta_i/dt = omega_i
            + sum_j K_ij sin(theta_j - theta_i - alpha_ij)
            + zeta sin(Psi - theta_i)
```

## Integration Methods

### Euler (default)

```
theta(t+dt) = theta(t) + dt * f(theta(t))
```

First-order. Sufficient when `dt` satisfies the stability condition.

### RK4

```
k1 = f(theta)
k2 = f(theta + dt/2 * k1)
k3 = f(theta + dt/2 * k2)
k4 = f(theta + dt * k3)
theta(t+dt) = theta(t) + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
```

Fourth-order. Use for stiff systems or when accuracy matters more than speed. 4x the derivative evaluations per step.

## Stability Condition

CFL-like bound:

```
dt < 1 / (max(omega) + N * max(K) + zeta)
```

Where `N` is the number of oscillators. The coupling term contributes up to `N * max(K)` to the effective frequency. Exceeding this bound causes phase jumps that break the wrapping invariant.

The binding spec `sample_period_s` sets `dt`. Validate at initialisation.

## Phase Wrapping

After every step: `theta = theta % (2*pi)`. This is the ONLY place wrapping occurs. Intermediate computations (derivative, scratch arrays) operate on unwrapped differences.

## Scratch Array Pre-Allocation

`UPDEEngine.__init__` allocates:

| Array | Shape | Purpose |
|-------|-------|---------|
| `_phase_diff` | `(N, N)` | `theta_j - theta_i - alpha_ij` |
| `_sin_diff` | `(N, N)` | `sin(phase_diff)` |
| `_scratch_dtheta` | `(N,)` | derivative accumulator |

All operations use `out=` parameter to avoid allocation during stepping. For RK4, `k1`-`k3` are copied since the scratch arrays are reused.

## Numerical Considerations

- `sin(theta_j - theta_i)` handles wrap-around implicitly: `sin(5.9 - 0.1) = sin(5.8) ≈ sin(-0.48)`.
- Double precision (float64) throughout. No single-precision paths.
- Order parameter `R = |mean(exp(i*theta))|` computed via complex arithmetic, not via trigonometric identities.

## References

- **[kuramoto1975]** Y. Kuramoto (1975). Self-entrainment of a population of coupled non-linear oscillators. *Lecture Notes in Physics* 39, 420–422. — UPDE equation origin.
- **[hairer1993]** E. Hairer, S. P. Nørsett & G. Wanner (1993). *Solving Ordinary Differential Equations I*. 2nd ed., Springer. — Euler and RK4 integrator theory.
- **[courant1928]** R. Courant, K. Friedrichs & H. Lewy (1928). Über die partiellen Differenzengleichungen der mathematischen Physik. *Math. Annalen* 100, 32–74. — CFL stability condition.
