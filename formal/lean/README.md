<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Lean proof lane -->

# Lean Formal Proof Lane

This directory contains the Lean 4 proof lane for small, safety-critical
integer/fixed-point contracts that mirror the Rust supervisor boundary.

## Current proof boundary

- `SPOFormal.Projector`: actuator clamp, bounded projection, fixed-point slew
  step, final bounded-projection slew preservation, and adaptive rate-limit
  range contracts. The proof boundary includes in-bounds clamp identity,
  signed fixed-point projector bounds, zero-slew immobility, nominal-risk
  adaptive-rate exactness, and saturated-risk clamping.
- `SPOFormal.Regime`: finite-input, fixed-point regime-classification contracts
  for hard violations, subcritical coherence, degraded-band behaviour,
  degraded hysteresis hold, critical-to-recovery transition boundaries, and
  recovery-to-nominal release. The proof boundary includes exact
  `R_CRITICAL`/`R_DEGRADED` threshold behaviour and hysteresis release
  behaviour for degraded and recovery states. It also mirrors the finite
  transition guard for same-state no-ops, critical cooldown bypass,
  soft-downward hold blocking, and non-critical cooldown blocking.
- `SPOFormal.Kinematic`: finite-horizon, fixed-point kinematic safety
  templates for PHA-C moving-frame and merge-window pipelines. The proof
  boundary includes a discrete Gronwall-style relative-distance budget,
  zero-gain linear horizon certification, and a Boolean phase-plus-spatial
  merge-window mirror. It also includes `PhaseBudgetBounds`, which keeps
  accepted replay phase dispersion separate from configured predictive
  phase-drift slack before proving the reviewed phase-lock certificate, plus
  `AcceptanceKinematicReplayBounds`, which keeps final-position,
  maximum-velocity, and path-length equation replay provenance explicit before
  a combined PHA-C acceptance certificate can discharge.
- `SPOFormal.Continuous`: fixed-point continuous-envelope templates for PHA-C
  handoffs. The proof boundary records per-second relative-velocity and
  residual rate bounds, samples those rates over the reviewed horizon time,
  proves the continuous horizon certificate, and bridges the same assumptions
  back into the sampled discrete kinematic budget.

The Lean lane is intentionally dependency-light: it uses Lean core plus `Std`,
not Mathlib. It is an independent specification mirror, not a replacement for
Kani harnesses over the Rust implementation.

## Out of scope

These proofs do not claim nonlinear Lyapunov stability under arbitrary
topology, hardware deadline guarantees, or site-specific actuator safety. The
continuous lane proves dependency-light fixed-point envelope arithmetic rather
than importing Mathlib real analysis; callers must still justify the runtime
conversion from physical units and numerical trajectories into the fixed-point
`ContinuousEnvelopeBounds` and `KinematicBounds` assumptions.

## Build

```bash
./tools/check_lean_proofs.sh
```

The script runs the CI-equivalent proof gate:

- reject Lean proof files containing `sorry`, `admit`, `axiom`, or `unsafe`;
- reject proof-source linter suppression via `set_option linter.* false`;
- treat Lean warnings as proof-gate failures on direct module checks;
- directly check `SPOFormal/Projector.lean`;
- directly check `SPOFormal/Regime.lean`;
- directly check `SPOFormal/Kinematic.lean`;
- directly check `SPOFormal/Continuous.lean`;
- build the `SPOFormal` Lake library;
- check the package entry point;
- directly check `test/KinematicTest.lean`.
