<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Phase Orchestrator — Bayesian UPDE API
-->

# UPDE — Bayesian Uncertainty

`scpn_phase_orchestrator.upde.bayesian` propagates uncertainty in natural
frequencies `omega` and coupling matrices `K_nm` through the existing UPDE
integrator. It reports posterior-predictive order parameter summaries as
`R ± sigma` plus a configurable credible interval.

The production backend is deterministic NumPy Monte Carlo over explicit array
distributions. `numpyro` and `blackjax` are reserved backend names and raise
`NotImplementedError` until their samplers are implemented, benchmarked, and
validated against the NumPy propagation baseline.

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
- The existing UPDE kernel performs every rollout, so deterministic engine
  validation, phase wrapping, and backend dispatch semantics are preserved.
- `BayesianUPDEResult.to_audit_record()` emits JSON-safe uncertainty
  diagnostics suitable for safety review and replay logs.

::: scpn_phase_orchestrator.upde.bayesian
