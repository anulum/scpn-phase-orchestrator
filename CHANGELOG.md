# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Improved

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
