<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Bayesian UPDE API -->

# UPDE — Bayesian Uncertainty

## Why Bayesian uncertainty exists in this project

Deterministic UPDE runs provide a single trajectory. Bayesian uncertainty
surfaces how sensitive that trajectory is to uncertainty in estimated
frequencies and couplings. For production planning and safety review, that
difference is material.

The module exposes uncertainty in a structured way so teams can make explicit
risk decisions rather than relying only on point estimates.

`scpn_phase_orchestrator.upde.bayesian` propagates uncertainty in natural
frequencies `omega` and coupling matrices `K_nm` through the existing UPDE
integrator. It reports posterior-predictive order parameter summaries as
`R ± sigma` plus a configurable credible interval.

The production backend is deterministic NumPy Monte Carlo over explicit array
distributions. `numpyro` and `blackjax` are reserved backend names and raise
`NotImplementedError` until their samplers are implemented, benchmarked, and
validated against the NumPy propagation baseline.

`fit_gaussian_upde_posterior()` provides the deterministic production baseline
for posterior fitting from observed Kuramoto phase trajectories. It uses a
finite-difference regression against the same UPDE coupling surface, enforces
non-negative zero-diagonal coupling, emits JSON-safe diagnostics, and feeds the
resulting Gaussian distributions directly into `bayesian_upde_run()`.

`audit_bayesian_backend_status()` probes backend names through the same
execution path. The NumPy backend must execute, while reserved sampler names
such as `numpyro` and `blackjax` must fail closed with audit records until they
have validated implementations and benchmark evidence.

## Minimal Example

```python
import numpy as np

from scpn_phase_orchestrator.upde import (
    BayesianUPDEConfig,
    GaussianArrayDistribution,
    bayesian_upde_run,
)

phases = np.array([0.0, 0.4, 1.1, 1.9])
omega_mean = np.array([0.9, 1.0, 1.08, 1.16])
knm_mean = np.full((4, 4), 0.18)
np.fill_diagonal(knm_mean, 0.0)
alpha = np.zeros((4, 4))

result = bayesian_upde_run(
    phases,
    omega=GaussianArrayDistribution(omega_mean, np.full(4, 0.015)),
    knm=GaussianArrayDistribution(
        knm_mean,
        np.full((4, 4), 0.01),
        non_negative=True,
        zero_diagonal=True,
    ),
    alpha=alpha,
    zeta=0.02,
    psi=0.1,
    config=BayesianUPDEConfig(n_samples=256, seed=7, n_steps=25),
)

r_mean, r_sigma = result.r_plus_minus
```

## Semantics

- `omega` and `knm` may be deterministic arrays or distribution objects.
- `GaussianArrayDistribution` samples independent normal uncertainty per array
  entry and can enforce non-negative coupling and zero self-coupling.
- `fit_gaussian_upde_posterior()` estimates Gaussian `omega` and `K_nm`
  distributions from finite phase trajectories with explicit ridge and
  uncertainty floors.
- `audit_bayesian_backend_status()` records executable and fail-closed backend
  state for release and safety-review evidence.
- The existing UPDE kernel performs every rollout, so deterministic engine
  validation, phase wrapping, and backend dispatch semantics are preserved.
- `BayesianUPDEResult.to_audit_record()` emits JSON-safe uncertainty
  diagnostics suitable for safety review and replay logs.

## How this is used operationally

Use Bayesian runs when estimates are sparse, noisy, or manually inferred.
In that setting the output carries both central tendency and uncertainty so
decision logic can enforce explicit safety thresholds before proposing changes.

## Production interpretation

- Use Bayesian UPDE when uncertainty is itself a policy input, not an afterthought.
- The fail-closed backend status check is a compliance boundary: unavailable
  advanced samplers must not silently become “best effort” paths.
- `r_plus_minus` is intended for risk-aware controller policy: a narrower sigma
  can justify higher coupling, while a wider sigma should force conservative
  action bounds.

## Practical overview

Bayesian UPDE is the uncertainty surface for domains where a single trajectory is
not enough to support operational action.

The module keeps one strict boundary: uncertainty must remain explicit in the
audit record. `r_plus_minus` is not a display-only value; it is intended to feed
risk-aware decision logic and conservative policy envelopes.

That is why backend-gated execution is important here. The code path refuses to
promote incomplete uncertainty backends into a production claim while preserving
the deterministic NumPy baseline as an auditable anchor.

### How teams usually use this surface

- Start from deterministic fitting (`fit_gaussian_upde_posterior`) on observed
  trajectories.
- Pass distributions through `bayesian_upde_run`.
- Compare posterior spread against control limits before accepting aggressive knob
  proposals.
::: scpn_phase_orchestrator.upde.bayesian
