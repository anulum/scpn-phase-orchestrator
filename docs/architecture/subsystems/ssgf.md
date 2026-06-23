# Subsystem: `ssgf` — self-stabilising gauge-field closure

An optional closed-loop coupling synthesiser: a coupling matrix is decoded from a
latent vector and driven by a cost. 8 files, ~1.6k LOC. Not part of the default
step loop — opt-in.

## Inputs

`phases` `(N,)` and a coupling matrix `W` `(N, N)` for the cost terms; a latent
vector `z` (default dimension 8) for the decoder; cost weights.

## Outputs

- `SSGFState(z, W, cost, grad_norm, step)` — the decoded coupling and its cost.
- `SSGFCosts(c1_sync, c2_spectral_gap, c3_sparsity, c4_symmetry, u_total)`.
- Observer snapshots from PGBO and TCBO (diagnostic only).

## Processing model

`GeometryCarrier` decodes `z → W` and updates `z` by finite-difference gradient
descent on a four-term cost: `c1 = 1 − R` (synchrony deficit), `c2 = −λ₂(L(W))`
(maximise algebraic connectivity), `c3 = ‖W‖₁/N²` (sparsity), `c4 = ‖W − Wᵀ‖_F/N`
(symmetry). `CyberneticClosure` runs the loop phases → costs → gradient → decode
→ feedback, tracking a non-increasing total cost. PGBO (phase-geometry observer)
and TCBO (a persistent-homology consciousness-boundary observer, `ripser`/PLV)
produce diagnostics; the TCBO gate feeds the coupling plasticity update.

## Backends

`compute_ssgf_costs` and the carrier decoder have Rust paths with NumPy fallback.

## Wiring

The closure returns a coupling matrix the caller may feed back to `upde`. It is
**not** invoked in the default simulation; integration is user-driven. The TCBO
`is_conscious` gate is consumed by `coupling/plasticity`.

## Scope boundaries

`EthicalCost` is an early-stage module, exported but not part of the active cost
objective. PGBO/TCBO are observation-only — they do not fail a step or block
actuation.
