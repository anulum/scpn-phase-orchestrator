<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Multi-language backend release gate -->

# Multi-Language Backend Review Gate

This gate is required before each minor release (`vX.Y.0`) to review auxiliary
language backends (Go, Julia, Mojo, and future additions) and classify them as:

- `keep`
- `demote` (experimental)
- `removal-candidate` (not removed by default)

Rust FFI and JAX remain primary paths.

## Non-destructive default

The default outcome is conservative:

- Prefer `keep` or `demote`.
- Use `removal-candidate` when burden is high and value is low.
- Actual removal requires explicit project-lead sign-off in the target release.

## Required Inputs

1. Parity evidence against Python reference (and Rust where relevant).
2. CI stability and recurrent failure patterns.
3. Benchmark value vs maintenance cost.
4. Operational/toolchain burden across supported platforms.
5. Maintained workload dependency and ownership.

## Decision Criteria

### Keep

- Stable parity.
- Acceptable CI reliability.
- Measurable value not already covered by Rust/JAX at lower cost.
- Clear maintainer ownership.

### Demote

- Works, but marginal value or elevated maintenance burden.
- Kept available for research/compatibility, not promoted for production.

### Removal-candidate

- No maintained workload depends on it, or parity/CI burden is persistently poor.
- Documented for follow-up; removal is optional and separately approved.

## Release Checklist

- [ ] Collect backend parity summaries.
- [ ] Review CI pass/flake data per backend.
- [ ] Update benchmark deltas for backend-owned kernels.
- [ ] Classify each auxiliary backend (`keep`/`demote`/`removal-candidate`).
- [ ] Update [Backend Fallback Chain](backend_fallbacks.md) if status changed.
- [ ] Capture decisions in release notes/changelog.

## Related

- [Backend Fallback Chain](backend_fallbacks.md)
- [Governance](../../GOVERNANCE.md)
