# Knm Calibration

## Overview

The coupling matrix K_nm encodes interaction strengths between oscillators. Calibration determines K_nm values from domain data, prior knowledge, or optimisation targets.

## Construction Pipeline

```
CouplingBuilder.build(n_osc, base_strength, decay_alpha)
    → distance-decay initialisation
    → template overlay (optional)
    → geometry projection (optional)
    → imprint modulation (runtime)
```

### Distance-Decay Initialisation

Default construction uses exponential decay with oscillator index distance:

```
K_nm = base_strength * exp(-decay_alpha * |n - m|)
diag(K) = 0
```

Parameters `base_strength` and `decay_alpha` are specified in `binding_spec.coupling`.

### Template Overlay

Coupling templates (named K_nm patterns) can be defined in `binding_spec.coupling.templates` and switched at runtime by the supervisor via `CouplingState.active_template`. Templates allow domain-specific coupling topologies (e.g., nearest-neighbour, small-world, hierarchical).

## Calibration Methods

### Manual

Set `base_strength` and `decay_alpha` from domain knowledge. Appropriate for small oscillator counts or well-characterised systems.

### Empirical

Estimate K_nm from observed phase-locking:

1. Record phase trajectories from domain data.
2. Compute pairwise PLV (phase-locking value) matrix.
3. Threshold or regress PLV → K_nm.

Not yet automated in the CLI; available as a workflow via `compute_plv` in `upde/order_params.py`.

### Bayesian

Fit K_nm posteriors given observed R trajectories and known omega_n. Requires an outer optimisation loop (e.g., scipy.optimize or MCMC). Described in the SCPN-MASTER-REPO notebook `04_bayesian_knm_calibration`.

## Anchoring

For specific oscillator pairs with known coupling strengths (e.g., from literature), override individual K_nm entries after distance-decay initialisation. Document anchors in the domainpack README.

## Validation

After calibration:

1. Run `spo run` with the calibrated binding_spec.
2. Check that R_good converges and R_bad stays low.
3. Compare simulated PLV matrix against empirical PLV (if available).
4. Verify `eval_protocol.md` metrics meet thresholds.

## References

- `src/scpn_phase_orchestrator/coupling/knm.py` — `CouplingBuilder`, `CouplingState`.
- `docs/specs/knm_semantics.md` — semantic interpretation of K_nm entries.
- `docs/specs/eval_protocol.md` — evaluation metrics for calibration quality.
