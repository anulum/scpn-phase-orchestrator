<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# Koopman EDMD — Data-Driven Linear Predictor with Control

`monitor.koopman_edmd` fits a **linear** predictive model of a nonlinear
controlled system directly from data. It is the model layer of the
grid-forming (dVOC) oscillation pack: the matrices it produces are consumed
unchanged by the convex Koopman-MPC controller (`actuation.koopman_mpc`).

## 1. Mathematical formalism

Koopman operator theory lifts a nonlinear controlled system
`x_{k+1} = f(x_k, u_k)` into a higher-dimensional space of observables
`ψ : ℝⁿ → ℝᴺ` where the evolution is approximately linear. Extended Dynamic
Mode Decomposition with control (Korda & Mezić 2018) fits that lifted linear
system from snapshot triples `(x_i, u_i, y_i)` with `y_i = f(x_i, u_i)`:

    z_{k+1} = A z_k + B u_k,        x̂_k = C z_k,        z_0 = ψ(x_0),

with `A ∈ ℝᴺˣᴺ`, `B ∈ ℝᴺˣᵐ`, `C ∈ ℝⁿˣᴺ`.

### 1.1 The two least-squares problems

The matrices minimise the lifted one-step residual and the reconstruction
residual (Korda eq. 17 and eq. 20):

    [A, B] = argmin Σ_i ‖ψ(y_i) − A ψ(x_i) − B u_i‖²
    C      = argmin Σ_i ‖x_i − C ψ(x_i)‖²

Both are solved in closed form through Tikhonov-regularised normal equations
(row-major snapshot convention, `X_lift` is `K×N`):

    (ΦᵀΦ + ρI) [Aᵀ; Bᵀ] = Φᵀ Y_lift,     Φ = [X_lift | U]
    (X_liftᵀ X_lift + ρI) Cᵀ = X_liftᵀ X

The ridge `ρ ≥ 0` keeps the Gram matrices well-posed. When the dictionary
contains the state coordinates, `C` reduces to the selection `[I, 0]` and the
recovery is exact on the training data.

### 1.2 Observable dictionaries

| `kind` | `ψ(x)` (before the optional constant) | Use |
|---|---|---|
| `identity` | `x` | linear systems; exact closure |
| `polynomial` | monomials up to `degree` | polynomial vector fields, Koopman-invariant subspaces |
| `rbf` | `x` plus Gaussian radial bases at `centres` | smooth general nonlinearities |
| `phase` | `x`, `cos θ_i`, `sin θ_i`, and the Kuramoto order-parameter components `R cos Ψ`, `R sin Ψ` | phase oscillators / Sakaguchi–Kuramoto |

The `phase` dictionary is the SPO-specific value-add: the first-harmonic Fourier
features render the phase vector field close to linear in the lifted space.

## 2. Python API

```python
from scpn_phase_orchestrator.monitor.koopman_edmd import (
    KoopmanDictionary, fit_koopman_predictor,
)

dictionary = KoopmanDictionary(kind="polynomial", state_dim=2, degree=2)
predictor = fit_koopman_predictor(states, next_states, inputs, dictionary=dictionary)
trajectory = predictor.predict(initial_state, input_sequence)   # (T+1, n)
```

`fit_koopman_predictor(states, next_states, inputs, *, dictionary, regularisation=1e-8)`
returns a frozen `KoopmanPredictor` carrying `(A, B, C)`, the dictionary, and the
RMS one-step lift residual. `KoopmanPredictor.predict` rolls the linear model
forward; an empty input sequence yields the single-row reconstruction.

## 3. Multi-backend fallback chain

The heavy step is the least-squares solve; it runs on the standard chain
**Rust → Mojo → Julia → Go → Python**, selected fastest-first. The dictionary
lift and predictor roll-out are control flow over that kernel and stay
Python-side.

### 3.1 Parity budget

Every backend reproduces the NumPy reference solve to within `1e-9` (absolute
and relative) on the same lifted snapshots; the parity gate
(`tests/test_koopman_edmd_backends.py`) fails the build otherwise. Rust, Julia
and Go carry the matrices numerically; Mojo round-trips them through a text
subprocess protocol, which is why the shared budget is `1e-9` rather than the
`1e-12` of the in-process backends.

### 3.2 Building the backends

| Backend | Build |
|---|---|
| Rust | `maturin develop --release -m spo-kernel/crates/spo-ffi/Cargo.toml` |
| Go | `cd go && go build -buildmode=c-shared -o libkoopman_edmd.so koopman_edmd.go` |
| Mojo | `mojo build mojo/koopman_edmd.mojo -o mojo/koopman_edmd_mojo -Xlinker -lm` |
| Julia | interpreted; loaded through `juliacall` at first use |

## 4. Tested invariants

- **Exact linear recovery** — the `identity` dictionary recovers `(A, B, C)` of a
  known linear controlled system to `1e-9`.
- **Koopman-invariant subspace** — on a Brunton–Tu slow-manifold system whose
  `[x₁, x₂, x₁²]` span is Koopman-invariant, the `polynomial` dictionary predicts
  the original states over a long horizon to `<1e-6`, while `identity` cannot.
- **One-step lift gain** — the `rbf` dictionary lowers the one-step
  state-prediction error below the linear dictionary.
- **Backend parity** — all backends match the reference to `1e-9`.

## 5. Pipeline position

`oscillation_modes` / `modal_participation` (monitor) → **`koopman_edmd`
(model)** → `koopman_mpc` (actuation, review-only) → `prc_oscillation`
(assurance). The predictor never actuates; it supplies the linear model the
controller's convex programme is built from.

## 6. References

- Korda & Mezić 2018, *Automatica* 93, 149–160 (arXiv:1611.03537) — linear
  predictors for nonlinear dynamical systems: Koopman operator meets MPC.
- Williams, Kevrekidis & Rowley 2015, *J. Nonlinear Sci.* 25, 1307 — a
  data-driven approximation of the Koopman operator (EDMD).

## 7. API reference

::: scpn_phase_orchestrator.monitor.koopman_edmd

## 8. Learned phase-autoencoder observables

The analytic dictionaries above are fixed feature maps. `monitor.phase_koopman`
instead uses a trained phase autoencoder (`nn.phase_autoencoder`, frozen to the
pure-NumPy `oscillators.phase_reduction` evaluator) as the observable map: the
learned latent is the coordinate in which a nonlinear oscillator's dynamics are
close to linear, so a predictor fitted in it captures dynamics the analytic
dictionaries miss. `LearnedKoopmanDictionary` satisfies the same
`KoopmanObservables` protocol the fit and predictor consume, and the lift is
state-inclusive (`ψ(x) = [x, g(x)]`) so the output map reconstructs the state
exactly while the learned block sharpens the linear evolution. There is no JAX on
the control path.

::: scpn_phase_orchestrator.monitor.phase_koopman
