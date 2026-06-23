# Subsystem: `coupling` — K_nm construction, adaptation, analysis

Builds and adapts the coupling matrix that drives the integrator. 27 files,
~5.4k LOC.

## Inputs

Domain coupling spec (base strength, decay, templates); phases / time series for
estimators and adapters; the current `knm` for plasticity and transfer-entropy
adaptation; a 16-layer SCPN timescale table for the physics builder.

## Outputs

- `CouplingState(knm, alpha, active_template, knm_r)` — `knm` `(N, N)`,
  non-negative, symmetric, zero diagonal.
- Hodge decomposition (`f_grad`, `f_curl`, `f_harm`, node potential, Betti
  number); spectral gap / Fiedler vector; adapted `knm` from plasticity or
  transfer entropy.

## Processing model

`CouplingBuilder` (exponential decay or 16-layer SCPN physics, handshakes,
amplitude). Analysis and adaptation: Hodge decomposition on the simplicial
complex (Jiang–Lim–Yao–Ye 2011), three-factor Hebbian plasticity (Friston 2005),
transfer-entropy adaptation (Lizier 2012), spectral connectivity, Bayesian
priors, HCP connectome loading, named topology templates, and non-negativity /
symmetry constraints. `infer.py` implements coupling inference for the
transfer-entropy method only (others raise `NotImplementedError`).

## Backends

Rust paths exist for Hodge, transfer-entropy adaptation, spectral, prior, and
connectome kernels (try/except import, validated within tolerance). Two are
**Python-only**: `coupling_est` (no Rust dispatch) and `plasticity` (a Rust
`plasticity.rs` exists in the kernel but is not bound to Python). Multi-language
forwarders route to `experimental/accelerators/coupling/`.

## Wiring

`knm` is supplied to `upde_run` every step. The output of plasticity and
transfer-entropy adaptation is fed back as the next step's coupling.

## Scope boundaries

`coupling_est` is pure Python (small-N only). SINDy and inference outputs are for
inspection, not runtime actuation. See also [ssgf.md](ssgf.md) for the gauge-field
coupling closure and [autotune.md](autotune.md) for offline estimation.
