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
  range contracts.
- `SPOFormal.Regime`: finite-input, fixed-point regime-classification contracts
  for hard violations, subcritical coherence, degraded-band behaviour,
  degraded hysteresis hold, critical-to-recovery transition boundaries, and
  recovery-to-nominal release.

The Lean lane is intentionally dependency-light: it uses Lean core plus `Std`,
not Mathlib. It is an independent specification mirror, not a replacement for
Kani harnesses over the Rust implementation.

## Out of scope

These proofs do not claim continuous-time Kuramoto stability, nonlinear
Lyapunov stability under arbitrary topology, hardware deadline guarantees, or
site-specific actuator safety. Those remain separate research and deployment
validation tasks.

## Build

```bash
cd formal/lean
lake build
```

For CI parity, also run the direct module checks:

```bash
cd formal/lean
lake env lean SPOFormal/Projector.lean
lake env lean SPOFormal/Regime.lean
lake env lean SPOFormal.lean
```

CI rejects Lean proof files containing proof placeholders or unsafe proof
surface constructs: `sorry`, `admit`, `axiom`, or `unsafe`.
