# Advanced Dynamics

Beyond basic Kuramoto, SPO provides several unique dynamical modules
that no other oscillator library offers.

## Variational Free Energy Predictor

Implementation of Friston's Free Energy Principle mapped to Kuramoto dynamics.
The system minimizes its own prediction error as a coupling objective.

- Precision-weighted prediction error (maps to coupling K_ij)
- KL complexity term on precision
- Online precision estimation from error variance
- Forward prediction with error injection into the UPDE right-hand side

```python
from scpn_phase_orchestrator.upde.prediction import VariationalPredictor

predictor = VariationalPredictor(n=16, dt=0.01)
# prediction_error = predictor.step(phases, predicted_phases)
# error_coupling() returns ε_gain·ε_i for injection into UPDE
```

::: scpn_phase_orchestrator.upde.prediction

## Hodge Decomposition of Coupling

Decomposes the coupling matrix K into three orthogonal components
via Hodge theory (Jiang et al. 2011):

- **Gradient**: conservative phase-locking flow (symmetric part)
- **Curl**: rotational circulation flow (antisymmetric part)
- **Harmonic**: topological residual (invariant under dynamics)

Answers: "Is this synchronization conservative or rotational?"

```python
from scpn_phase_orchestrator.coupling.hodge import hodge_decompose

gradient, curl, harmonic = hodge_decompose(K)
```

::: scpn_phase_orchestrator.coupling.hodge

## Simplicial (3-Body) Coupling

Higher-order interactions beyond pairwise: the 3-body term induces
explosive (first-order) synchronization transitions.

```
dθ_i/dt = ω_i + (σ₁/N) Σ_j K_ij sin(θ_j - θ_i)
                + (σ₂/N²) Σ_{j,k} sin(θ_j + θ_k - 2θ_i)
```

Gambuzza et al. 2023, Nature Physics; Tang et al. 2025.

::: scpn_phase_orchestrator.upde.simplicial

## Time-Delayed Coupling

Circular buffer supports arbitrary time delays with automatic fallback
to instantaneous coupling. Time delays generate "effective higher-order
interactions for free" (Ciszak et al. 2025).

::: scpn_phase_orchestrator.upde.delay

## Stochastic Resonance with Optimal Noise

Euler-Maruyama integration with automatic optimal noise tuning.
Counter-intuitive: noise at D* INCREASES synchronization.

Optimal noise: D* ≈ K·R_det/2 (Tselios et al. 2025). Self-consistency
via modified Bessel transcendental equation (Acebrón et al. 2005).

::: scpn_phase_orchestrator.upde.stochastic

## Geometric (Torus-Preserving) Integrator

Symplectic Euler on T^N via SO(2) exponential map. Works in unit complex
representation z_i = exp(iθ_i), avoiding mod 2π discontinuity errors
that cause subtle numerical drift in standard integrators.

Essential for long timescale simulations where standard Euler accumulates
phase wrapping errors.

::: scpn_phase_orchestrator.upde.geometric

## Ott-Antonsen Mean-Field Reduction

Exact analytical reduction for Lorentzian frequency distributions:

```
dz/dt = -(Δ + iω₀)z + (K/2)(z - |z|²z)
```

Critical coupling K_c = 2Δ. Steady-state: R_ss = √(1 - 2Δ/K).
Used by the PredictiveSupervisor as a fast forward model for MPC
(O(1) vs O(N) for full simulation).

::: scpn_phase_orchestrator.upde.reduction

## Second-Order Inertial Kuramoto (Power Grids)

The swing equation models power grid transient stability:

```
m_i θ̈_i + d_i θ̇_i = P_i + Σ_j K_ij sin(θ_j - θ_i)
```

where m_i is rotor inertia, d_i is damping, P_i is power injection
(positive = generator, negative = load), K_ij is line susceptance.

Desynchronization = cascading blackout (Iberian Peninsula, April 2025).

```python
from scpn_phase_orchestrator.upde.inertial import InertialKuramotoEngine

engine = InertialKuramotoEngine(n=100, dt=0.01)
theta, omega, theta_traj, omega_traj = engine.run(
    theta0, omega0, power, knm, inertia, damping, n_steps=10000
)
freq_dev = engine.frequency_deviation(omega)  # Hz deviation
R = engine.coherence(theta)  # phase coherence
```

::: scpn_phase_orchestrator.upde.inertial
