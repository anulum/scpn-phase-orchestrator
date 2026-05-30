<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Phase Orchestrator — causal coupling inference API
-->

# Coupling — Causal Inference

`scpn_phase_orchestrator.coupling.infer` packages transfer-entropy-based
coupling discovery as `auto-coupling-estimation`.

The estimator accepts phase time series with shape `(oscillators, timesteps)`
and returns a directed source-to-target matrix: `knm[i, j]` is the inferred
influence from oscillator `i` to oscillator `j`. Standard UPDE matrices use
target-by-source orientation, so `CouplingInferenceResult.to_upde_knm()`
returns the transposed matrix for direct engine input.

The current production backend is transfer entropy over binned phase states.
`granger` and `notears` are reserved method names and raise
`NotImplementedError` until those estimators have benchmarked implementations.
The boundary rejects boolean aliases, complex values, non-finite samples, and
non-2-D phase series before inference. Backend transfer-entropy matrices must
be finite, non-negative, correctly shaped, and zero on the diagonal; invalid
backend scores fail closed instead of being silently normalised.

## CLI

```bash
spo auto-coupling-estimation phases.csv \
  --orientation time-by-oscillator \
  --n-bins 8 \
  --threshold-quantile 0.75 \
  --json-out
```

CSV and `.npy` sources are supported. Use `--orientation oscillator-by-time`
when rows are oscillators and columns are timesteps.

::: scpn_phase_orchestrator.coupling.infer
