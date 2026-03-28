# Kuramoto Theory — Mathematical Foundations

## The Kuramoto Model

Yoshiki Kuramoto (1975) proposed the simplest model of synchronization
in a population of coupled oscillators:

```
dθ_i/dt = ω_i + (K/N) Σ_{j=1}^N sin(θ_j - θ_i)
```

where:
- θ_i ∈ [0, 2π): phase of oscillator i
- ω_i: natural frequency (drawn from distribution g(ω))
- K: global coupling strength
- N: number of oscillators

## Order Parameter

The complex order parameter z = R·exp(iΨ) summarizes the collective state:

```
z = (1/N) Σ_{j=1}^N exp(iθ_j)
```

- R = |z| ∈ [0, 1]: coherence (0 = incoherent, 1 = perfect sync)
- Ψ = arg(z): mean phase

The Kuramoto model can be rewritten using z:

```
dθ_i/dt = ω_i + K·R·sin(Ψ - θ_i)
```

Each oscillator is pulled toward the mean phase Ψ with force proportional
to both K and R. This creates a positive feedback loop: partial sync
(R > 0) creates a stronger pull, which increases sync further.

## Critical Coupling

For a symmetric unimodal frequency distribution g(ω), there exists a
critical coupling K_c below which the incoherent state (R = 0) is stable:

```
K_c = 2 / (π · g(0))
```

For Lorentzian g(ω) with half-width Δ centered at ω₀:

```
K_c = 2Δ
```

Above K_c, the order parameter grows as R ~ √(K - K_c).

## Ott-Antonsen Reduction

For Lorentzian frequency distributions, the infinite-N Kuramoto model
reduces exactly to a single complex ODE (Ott & Antonsen, 2008):

```
dz/dt = -(Δ + iω₀)z + (K/2)(z - |z|²z)
```

Steady-state: R_ss = √(1 - 2Δ/K) for K > K_c.

This is implemented in `scpn_phase_orchestrator.upde.reduction` and used
by the MPC supervisor for O(1) prediction.

## Generalized Coupling

SPO extends the basic model with:

### Sakaguchi-Kuramoto (Phase Lags)
```
dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j - θ_i - α_ij)
```
The phase lag α_ij models transport delays or asymmetric coupling.

### Stuart-Landau (Amplitude Dynamics)
```
dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j - θ_i)
dr_i/dt = (μ_i - r_i²)r_i + ε Σ_j K^r_ij r_j cos(θ_j - θ_i)
```
Adds amplitude r_i with Hopf bifurcation: μ > 0 → r → √μ (active),
μ < 0 → r → 0 (quiescent).

### Simplicial (Higher-Order)
```
dθ_i/dt = ω_i + (σ₁/N) Σ_j K_ij sin(θ_j - θ_i)
                + (σ₂/N²) Σ_{j,k} sin(θ_j + θ_k - 2θ_i)
```
3-body interactions produce explosive (first-order) synchronization
transitions (Gambuzza et al. 2023, Nature Physics).

### Inertial (Second-Order)
```
m_i θ̈_i + d_i θ̇_i = P_i + Σ_j K_ij sin(θ_j - θ_i)
```
The swing equation for power grid stability. Inertia m_i represents
rotating mass of generators.

## Spectral Alignment Function

The Synchrony Alignment Function (Skardal & Taylor, 2016) provides a
closed-form approximation of R from the Laplacian eigenstructure of K:

```
R ≈ 1 - (1/2N) Σ_{j=2}^N λ_j⁻² ⟨v^j, ω⟩²
```

where λ_j are eigenvalues and v^j eigenvectors of the graph Laplacian
L = D - K. This enables coupling topology optimization without ODE
integration (10x faster).

## Inverse Problem

Given observed phases θ_i(t), infer K and ω by minimising:

```
L(K, ω) = Σ_t (1 - cos(θ_predicted(t) - θ_observed(t))) + λ·‖K‖₁
```

The L1 penalty on K promotes sparsity (topology discovery). Gradients
computed via JAX autodiff through the ODE solver.

## References

- Kuramoto Y (1975). Self-entrainment of a population of coupled
  non-linear oscillators. Lecture Notes in Physics 39:420-422.
- Ott E, Antonsen TM (2008). Low dimensional behavior of large systems
  of globally coupled oscillators. Chaos 18:037113.
- Skardal PS, Taylor D (2016). Optimal synchronization of directed
  complex networks. Chaos 26:094807.
- Gambuzza LV et al. (2023). Stability of synchronization in simplicial
  complexes. Nature Physics 19:1427-1434.
- Filatrella G et al. (2008). Analysis of a power grid using a
  Kuramoto-like model. European Physical Journal B 61:485-491.
- O'Keeffe KP et al. (2017). Oscillators that sync and swarm.
  Nature Communications 8:1504.
