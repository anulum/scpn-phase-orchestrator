<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Multi-language backend release gate -->

# Multi-Language Backend Review Gate

## Why this gate exists

The repo keeps multiple language backends for experimentation and reach, but not all
of them can hold the same reliability bar at the same time.

This gate exists to keep production support boundaries explicit. It documents
which implementations are production-safe versus exploratory, and it gives release
leaders a stable artifact for decisions that affect operator expectations, support
cost, and runtime complexity.

The expected outcome is a short, defensible statement that can be read during a
release review without opening CI logs first.

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

In practice this is a capability governance gate: each backend is compared against
the same decision framework used for production components, including observability,
failure recovery profile, and maintenance burden.

## Required Inputs

The checklist is tied to concrete product risk. For each backend, collect:

- where it is used in examples, demos, or customer pilots,
- any user-visible assumptions it introduces (e.g., runtime installation constraints),
- what parity or reproducibility evidence is available for the most relevant
  workloads.

1. Parity evidence against Python reference (and Rust where relevant).
2. CI stability and recurrent failure patterns.
3. Benchmark value vs maintenance cost.
4. Operational/toolchain burden across supported platforms.
5. Maintained workload dependency and ownership.

When any of these inputs is missing or stale, the safe classification is at least
`demote` until evidence is updated in the same release cycle.

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

Use this checklist as a release evidence log. Keeping it complete lets downstream
users understand what is guaranteed today versus what is retained for research
continuity.

## Related

- [Backend Fallback Chain](backend_fallbacks.md)
- Governance: `GOVERNANCE.md` in the repository root
