# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **[P0]** `verify_determinism` compared global R against mean-of-layer-R (different quantities); now compares against `stability_proxy`
- **[P0]** `UPDEEngine.step()` accepted shape-mismatched arrays silently; now validates all input shapes
- **[P0]** Rust `UPDEStepper` validated `n_substeps` but ignored it (always 1); now loops `n_substeps` iterations at `sub_dt = dt / n_substeps`
- **[P0]** Rust `LagModel` propagated NaN distances into alpha matrix; now rejects NaN/Inf distances with `IntegrationDiverged`
- **[P0]** `RegimeManager.transition()` took redundant `current` param that could diverge from `self._current`; removed for Python↔Rust parity
- **[P1]** `InformationalExtractor` theta always cancelled to ~0 (`2π * f * dt` where `f = 1/dt`); use median freq × total time
- **[P1]** `ImprintModel.modulate_lag` added row-wise offset destroying antisymmetry; use `m[i] - m[j]` (antisymmetric)
- **[P1]** CLI scaffold generated `version: '0.1'` (fails validator); now `'0.1.0'`
- **[P1]** CLI `run` did not compute `cross_layer_alignment`; now uses `compute_plv` between layer pairs
- **[P1]** CLI zeta had no TTL expiry; now decrements TTL counter and resets to 0 on expiry
- **[P1]** Rust UPDE stepper did not check omegas/knm for NaN/Inf; now rejects with `IntegrationDiverged`
- **[P0]** Merge duplicate `validate_binding_spec` — `loader.py` and `validator.py` had divergent implementations; canonical version now in `validator.py` with all checks merged
- **[P1]** `PhysicalExtractor._snr_estimate` always returned ~1.0 due to `Re(hilbert(x)) == x` identity; replaced with envelope coefficient-of-variation metric
- **[P1]** `UPDEEngine.compute_order_parameter` reimplemented inline, bypassing Rust-accelerated `order_params`; now delegates to canonical implementation
- **[P1]** `AuditLogger` file writes never flushed; crash could lose audit records; switched to line-buffered I/O
- **[P1]** CLI `run` command ignored spec drivers, boundaries, and actuators; hardcoded `omegas=1` and `zeta=0`; rewritten to wire supervisor, boundary observer, and spec-derived frequencies

### Changed

- `BoundaryState.soft_warnings` renamed to `soft_violations` for Rust parity
- 7 source files import `TWO_PI`/`HAS_RUST` from `_compat` instead of redefining
- `RegimeManager.transition()` takes only `proposed` param (breaking: callers updated)
- Rust `event_phase` uses median freq × total time (matches Python fix)
- ROADMAP domainpack names match actual directory names
- CONTRIBUTING import path corrected: `extractors/` → `oscillators/`
- `spo-types`: `serde_json` moved to `[dev-dependencies]`
- CHANGELOG heading `Improved` → `Changed` per Keep a Changelog spec
- CLI import path uses canonical `from scpn_phase_orchestrator.binding import validate_binding_spec`
- `oscillators/__init__.py` exports `PhysicalExtractor`, `InformationalExtractor`, `SymbolicExtractor`, `PhaseQualityScorer`
- `adapters/__init__.py` exports `SCPNControlBridge`
- Rust: `Debug` impl for `UPDEStepper` (manual, omits scratch buffers), `#[derive(Debug)]` for `ImprintModel`, `LagModel`
- Rust: `///` doc comments on `UPDEStepper`, `ImprintModel`, `CouplingState`, `LagModel`, `RegimeManager`, `CoherenceMonitor`

### Added

- `src/scpn_phase_orchestrator/_compat.py` — shared `TWO_PI`, `HAS_RUST` constants (single source of truth)
- `src/scpn_phase_orchestrator/py.typed` — PEP 561 marker for downstream type checkers
- `tools/check_version_sync.py` — asserts pyproject.toml, CITATION.cff, Cargo.toml versions match
- `.dockerignore` — excludes .git, target, caches, site
- CI lint job runs `check_version_sync.py`
- Publish preflight runs Rust clippy + tests
- PyPI and docs badges in README
- `repository` field in Cargo workspace metadata
- 4 new validator tests: `control_period_s` positive/ordering, actuator limits ordering, empty objectives
- 5 new hypothesis property tests: phase wrapping, R unit interval, `project_knm` symmetry, imprint saturation bound, regime FSM skip guard
- 4 new `test_coupling_lags` tests: negative lag direction, large lag, constant signal, alpha diagonal
- 4 new `test_coupling_templates` tests: duplicate overwrite, empty set, frozen dataclass, error message content
- 2 new `test_oscillator_physical` tests: clean sinusoid quality, clean-vs-noisy discrimination

## [0.1.1] - 2026-03-02

### Changed

- Migrate remaining `try/except ImportError` probe-imports to `importlib.util.find_spec` with lazy imports (engine, order_params, physical, knm)
- Remove dead `_HAS_RUST` assignments and unused imports from regimes.py, coherence.py
- Add `#[must_use]` to all pure public Rust functions; enable `must_use_candidate = "warn"` workspace lint
- Add crate-level `//!` doc comments to spo-types, spo-engine, spo-oscillators, spo-supervisor
- Refactor physical.rs Pass 1 from index loop to `iter_mut().zip()` iterators
- Make `LockSignature`, `LayerState`, `UPDEState` frozen dataclasses

### Fixed

- 5 hardening sprints of bug fixes, CI correctness, supply chain security, and lint cleanup since v0.1.0

### Changed

- CI: add pip + cargo caching, replace manual `cargo-audit` install with `rustsec/audit-check` action
- Add `Documentation` and `Changelog` URLs to `project.urls` (PyPI sidebar)

## [0.1.0] - 2026-03-01

### Added

- UPDE engine with Euler/RK4 integration and pre-allocated scratch arrays
- 3-channel oscillator model: Physical, Informational, Symbolic (P/I/S)
- Coupling matrix (Knm) management with exponential decay and template switching
- Supervisor with RegimeManager (NOMINAL/DEGRADED/CRITICAL/RECOVERY) and policy actions
- Actuation mapper and ActionProjector for domain-agnostic output binding
- Memory imprint model with exponential decay and coupling/lag modulation
- Boundary observer (soft/hard violations) feeding regime decisions
- CLI entry point (`spo`) with init, run, replay, status commands
- 4 domainpacks: minimal_domain, queuewaves, geometry_walk, bio_stub
- Binding spec JSON Schema for domainpack configuration and validation
- PhaseExtractor base class for domain-specific signal intake
- Assumption registry (`docs/ASSUMPTIONS.md`) documenting all empirical thresholds
- Bibliography (`docs/references.bib`) with 10 canonical references (Kuramoto, Acebrón, Sakaguchi, Strogatz, Dörfler, Lachaux, Gabor, Pikovsky, Hairer, Courant)
- Citation metadata (`CITATION.cff`) for Zenodo and academic databases
- Coverage regression guard (`tools/coverage_guard.py`) enforcing 90% line coverage
- Module linkage guard (`tools/check_test_module_linkage.py`) requiring test files for all source modules
- Rust kernel (`spo-kernel/`) with PyO3 bindings for UPDEEngine, RegimeManager, CoherenceMonitor

[Unreleased]: https://github.com/anulum/scpn-phase-orchestrator/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/anulum/scpn-phase-orchestrator/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/anulum/scpn-phase-orchestrator/releases/tag/v0.1.0
