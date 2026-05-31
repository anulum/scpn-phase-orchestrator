# Codex session — dimension exact backend contracts

Timestamp: 2026-05-31T135310
Project: SCPN-PHASE-ORCHESTRATOR
Task: Continue U1 physics/math boundary hardening from the open roadmap row.

## Scope

Hardened the `monitor.dimension` public and direct backend boundary for exact mathematical preservation:

- Added exact Grassberger-Procaccia correlation-integral reference checks for direct Go, Julia, and Mojo calls.
- Added exact Kaplan-Yorke reference checks for direct Go, Julia, and Mojo calls.
- Added public dispatcher fallback when accelerated non-Rust backends emit plausible but non-reference outputs.
- Kept Rust subsampling API compatibility: full-pairs Rust is exact-checked; Rust-owned subsampling remains shape/range validated because its RNG is intentionally kernel-owned.
- Updated module-specific tests in `tests/test_dimension_backends.py` only.
- Updated dimension API docs, dimension benchmark boundary metadata, and U1 roadmap evidence.

## Verification

Per the current user workflow, no pytest/ruff/mypy execution was run in this task. Staged compliance gates are run before commit.

## Commit

Pending at time of log creation.
