<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# Architecture decisions

This page records the major architectural decisions behind SCPN Phase
Orchestrator and the reasoning for each, so a new contributor can understand not
just *what* the code does but *why* it is shaped this way. Each decision lists
its context, the decision, the rationale, and the consequences. The product-tier
boundaries are detailed separately in
[Product boundaries](architecture_product_boundaries.md).

## 1. Four product tiers, enforced as an import contract

**Context.** The codebase mixes a stable control core, a serving runtime,
external-system connectors, and exploratory research code with very different
stability expectations.

**Decision.** Every module is classified into one of four tiers — **core**
(`actuation`, `binding`, `coupling`, `imprint`, `monitor`, `oscillators`,
`ssgf`, `supervisor`, `upde`), **runtime** (`api`, `apps`, `assurance`, `audit`,
`reporting`, `runtime`, `scaffold`, `studio`, …), **integrations** (`adapters`,
`drivers`), and **experimental** (`nn`, `experimental`, the language shims).
`tools/check_product_boundaries.py` fails CI if core imports runtime, integration,
or experimental code, or if integrations import runtime or experimental code.

**Rationale.** The control core must be auditable and dependency-light; letting it
acquire a dependency on the serving runtime or on optional ML code would make it
impossible to reason about or to certify. A static import check is cheaper and
more reliable than directory discipline alone.

**Consequences.** A pipeline that needs a runtime-tier service (for example the
NERC PRC screener) lives in the runtime tier, not the core; the dependency-light
pieces it composes stay in the core.

## 2. Review-only actuation

**Context.** SPO is decision-support for safety-relevant cyclic systems (grids,
neuro, fusion-adjacent). The actuator is owned by the operator, not by SPO.

**Decision.** SPO **proposes** control and **never** actuates. Proposals are
frozen, content-hashed dataclasses (a sorted/compact JSON SHA-256, rounded for
reproducibility); the supervisor, governor, and barrier filters admit, constrain,
or reject but emit a record rather than driving hardware. Live-actuation,
hardware, QPU, and neuro paths require explicit operator approval.

**Rationale.** It keeps SPO inside the value-capture ceiling it can defend
(decision-support and compliance evidence), it is the only honest posture for a
system the operator must remain accountable for, and it makes every proposal an
auditable artifact rather than an irreversible action.

## 3. Multi-language acceleration, fastest-first and parity-gated

**Context.** Hot numerical kernels (EDMD solves, coupling, monitors) benefit from
native code, but a single native dependency would compromise portability.

**Decision.** Each accelerated kernel ships a five-backend chain — **Rust → Mojo
→ Julia → Go → Python** — dispatched fastest-available-first. Every non-Python
backend is parity-gated against the Python reference to a tight tolerance, and a
backend loader verifies its built artifact at load time so an unbuilt backend
never enters the available set. Python is always the final, dependency-free floor.

**Rationale.** Portability is preserved (the pure-Python floor always works);
performance is captured where the toolchain is present; and parity gating plus
load-time verification keep the backends honest — a number is never silently
backend-divergent. The studio evidence schema records the active backend and the
parity tolerance for exactly this reason.

## 4. Optional heavy dependencies as extras with deterministic fallbacks

**Context.** Some capabilities need JAX/equinox (differentiable `nn`), `osqp`
(an MPC QP backend), or FMI tooling, which are large or platform-specific.

**Decision.** These live behind extras (`[nn]`, `[mpc]`, …) and lazy import
guards (`HAS_JAX` / `require_jax`); the package imports and the base capabilities
run without them. Where an optional backend computes something, a deterministic
in-house implementation is the supported floor — the ADMM QP solver under the
optional `osqp`, the pure-NumPy `oscillators.phase_reduction` evaluator under the
JAX-trained phase autoencoder, the FMI model that needs no FMI runtime.

**Rationale.** A base install stays small and reproducible; the heavy paths are
opt-in; and the control path never *requires* JAX, so a trained model is consumed
through frozen NumPy weights rather than dragging a GPU framework into production.

## 5. Koopman model → MPC → assurance for the dVOC pack

**Context.** Grid-forming (dVOC) oscillation damping needs a controllable model
of a nonlinear oscillator and a convex online controller.

**Decision.** The pack is a chain of cited methods: an EDMD-with-control Koopman
predictor (Korda & Mezić 2018, *Automatica* 93; EDMD per Williams, Kevrekidis &
Rowley 2015) lifts the state to observables where the dynamics are linear; a
condensed Koopman MPC (Korda eq. 24) builds the QP directly from `(A, B, C)` with
an online cost independent of the lift dimension; the QP is solved by an in-house
ADMM solver (operator splitting per Stellato et al. 2020, OSQP) with an optional
`osqp` backend; observables may be the analytic dictionaries or a learned
phase-autoencoder coordinate (Yawata, Fukami, Taira & Nakao 2024, *Chaos* 34);
and the closed-loop result is re-screened into hash-sealed NERC PRC evidence.

**Rationale.** Each layer matches a publication exactly (no simplified proxy), the
condensed MPC keeps the online cost small, and ending in screened evidence makes
the damping improvement auditable end to end — the value SPO can sell.

## 6. Evidence is the product: hash-sealed, content-addressed, replayable

**Context.** SPO's differentiator is honest, auditable evidence rather than raw
control.

**Decision.** Runs emit a SHA-256-chained, optionally HMAC-signed audit JSONL that
replays deterministically; review records are content-addressed frozen
dataclasses; formal safety properties are machine-checked Lean certificates
(PHA-C kinematic lemmas); and the capability manifest is content-addressed and
deterministic. Evidence carries its numeric provenance (backend, tolerance) and
safety tier, and a formal-proof certificate is a distinct *modality* from an
empirically-graded measurement, not a higher grade of it.

**Rationale.** A reviewer must be able to recompute every on-screen number and to
distinguish a measured claim from a proven one; this is the honesty-as-product
surface and the basis of the cross-studio evidence standard.

## 7. Two safety tiers gate execution

**Context.** A declarative binding may describe a research experiment or a
production controller; the same runtime must not execute both the same way.

**Decision.** A binding declares `safety_tier`. The local runtime executes
**research**-tier specs; a **production**-tier spec is refused by the local
runtime and must go through the formal-export and certified-controller pipeline.
`spo validate --security --hard` additionally scans an untrusted domainpack's
files for dangerous patterns before it is trusted.

**Rationale.** It makes the line between "explore this locally" and "this would
drive a real plant" explicit and enforceable, rather than a matter of operator
care.

## 8. Public API is explicit and machine-checked

**Context.** A large surface area invites accidental coupling to internals.

**Decision.** Public symbols are listed in each package's `__all__`; internals are
`_`-prefixed; the capability manifest enumerates the public surface from
`git ls-files`; and governance tests assert that public modules have autodoc
pages, that public APIs carry NumPy-contract docstrings, and that nothing drifts
from the committed manifest.

**Rationale.** An explicit, tested public boundary lets internals refactor freely
and gives downstream studios a stable contract to depend on. (Because the manifest
enumerates tracked files, new modules must be `git add`-ed before the manifest is
regenerated.)

## 9. Two build backends; one abi3 wheel

**Context.** The project is a pure-Python package plus a separate PyO3/Rust
extension (`spo_kernel`).

**Decision.** The Python package builds with setuptools; the Rust extension builds
with maturin as the standalone `spo-kernel` package, with the `pyo3/abi3-py310`
feature so it ships a single `cp310-abi3` wheel per platform that loads on CPython
3.10+. The abi3 flag is a maturin feature, not a Cargo default, so `cargo test
--no-default-features` is unaffected.

**Rationale.** abi3 collapses the wheel matrix from one-per-Python-version to
one-per-platform, shrinking the release and download surface, while the test build
is untouched.

## 10. Testing: 100% target with an environment-aware gate

**Context.** A single CI lane cannot reach full coverage because optional backends
and extras are absent.

**Decision.** New code targets 100% line+branch with multi-angle tests. The
pyproject `fail_under` is a local soft floor; the CI test and ffi lanes run with
`--cov-fail-under=0`; the authoritative gate is the `coverage-guard` job, which
combines the test and ffi coverage and enforces the per-module thresholds in
`tools/coverage_guard_thresholds.json`. JAX/equinox-trained behaviour is asserted
on platform-robust invariants (a learned frequency is checked by magnitude, not
sign, because of the phase autoencoder's reflection symmetry).

**Rationale.** Per-lane gating would fail spuriously whenever a backend is missing;
combining coverage and gating per module captures the real picture, and asserting
robust invariants keeps ML tests from flaking across jaxlib builds.
