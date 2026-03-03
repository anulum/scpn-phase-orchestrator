# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-03-04

### Added

- **Compound policy DSL** — `CompoundCondition` with AND/OR logic over multiple `PolicyCondition` triggers
- **Action chains** — `PolicyRule.actions` accepts a list of `PolicyAction` items fired on a single trigger
- **Rule rate-limiting** — per-rule `cooldown_s` and `max_fires` fields
- **`stability_proxy` metric** in policy conditions (global mean R)
- **OpenTelemetry export** — `OTelExporter` with span instrumentation, gauge metrics (`spo.r_global`, `spo.stability_proxy`), step counter; no-op fallback when `opentelemetry-api` is absent
- `otel` optional dependency group (`opentelemetry-api>=1.20`, `opentelemetry-sdk>=1.20`)
- Pre-commit hook for version consistency check across pyproject.toml, CITATION.cff, Cargo.toml
- **QueueWaves** — real-time microservice cascade failure detector (`apps/queuewaves/`)
  - PrometheusCollector with persistent async httpx client and ring buffers
  - PhaseComputePipeline wrapping UPDE engine for Kuramoto phase analysis
  - AnomalyDetector: retry-storm, cascade-propagation, chronic-degradation
  - WebhookAlerter with deduplication and Slack/generic webhook formats
  - FastAPI server with REST API, WebSocket streaming, Prometheus exposition
  - Single-file HTML dashboard (R timeline, phase wheel, alert table)
  - CLI subcommands: `spo queuewaves serve`, `spo queuewaves check`
  - Graceful shutdown with task cancellation and resource cleanup
  - 60 tests, coverage >90%
- 12 new domainpacks: cardiac_rhythm, circadian_biology, chemical_reactor, epidemic_sir, firefly_swarm, laser_array, manufacturing_spc (upgraded), neuroscience_eeg, pll_clock, power_grid, rotating_machinery, swarm_robotics (total: 21)
- 3 adapter bridges: FusionCoreBridge, PlasmaControlBridge, QuantumControlBridge
- RK45 adaptive integration with configurable tolerance and max-step limits
- PolicyEngine: declarative YAML rules with regime/metric triggers
- ActionProjector wiring for supervisor → actuation pipeline
- BindingLoadError exception and validator guards for malformed specs
- Phase-synchronization control theory docs (scope-of-competence, hardware pipeline)
- Synchronization manifold header image
- Queuewaves retry-storm demo notebook
- **Deterministic replay** from audit.jsonl with chained phase-vector verification
  - `AuditLogger.log_header()` writes engine config (n, dt, method, seed)
  - `AuditLogger.log_step()` now records full UPDE inputs (phases, omegas, knm, alpha, zeta, psi)
  - `ReplayEngine.verify_determinism_chained()` replays logged steps and compares output phases
  - `ReplayEngine.build_engine()` reconstructs UPDEEngine from header record
  - CLI `spo replay --verify` validates bit-for-bit reproducibility
  - CLI `spo run --seed` makes initial RNG seed configurable and logged

### Fixed

- **[P0]** Rust `ImprintModel.modulate_lag` added row-wise `m[i]` offset; now uses `m[i] - m[j]` preserving antisymmetry
- **[P0]** CLI `run` silently dropped `K` and `Psi` supervisor actions; now applies coupling scaling and target phase
- **[P0]** CLI `stability_proxy` used only first layer R; now uses mean R across all layers
- **[P1]** Rust `PhysicalExtractor` quality hardcoded to 1.0; now computes envelope coefficient-of-variation
- **[P1]** `compute_plv` silently truncated mismatched arrays; now raises `ValueError` / `SpoError::InvalidDimension`
- All domainpack binding specs use semver (`0.1.0` not `0.1`)
- 9 mypy type errors in bridges and CLI resolved
- CI: queuewaves optional deps installed for test coverage
- Ruff format violations in audit/logger.py and tests/test_audit_replay.py

### Changed

- `PolicyRule` now uses `actions: list[PolicyAction]` instead of top-level knob/scope/value/ttl_s fields
- `PolicyRule.condition` accepts both `PolicyCondition` and `CompoundCondition`
- `PolicyEngine` tracks per-rule fire counts and cooldown timestamps
- `OTelAdapter` stub replaced by production `OTelExporter` class
- FFI `PyUPDEStepper` accepts `n_substeps` parameter
- FFI `PyCoherenceMonitor` exposes `detect_phase_lock` with full CLA matrix
- `ImprintState` and `CouplingState` are frozen dataclasses
- CI: PRs to `develop` branch trigger CI; FFI test job runs full test suite
- CI: install `.[dev,queuewaves]` in all jobs for full coverage
- Previous sprints 1–9 entries moved to [0.1.1] section below
- Repo hygiene: PEP 639 classifier fix, pre-commit pins, stub exports

## [0.1.1] - 2026-03-02

### Fixed

- **[P0]** `verify_determinism` compared global R against mean-of-layer-R (different quantities); now compares against `stability_proxy`
- **[P0]** `UPDEEngine.step()` accepted shape-mismatched arrays silently; now validates all input shapes
- **[P0]** Rust `UPDEStepper` validated `n_substeps` but ignored it (always 1); now loops `n_substeps` iterations at `sub_dt = dt / n_substeps`
- **[P0]** Rust `LagModel` propagated NaN distances into alpha matrix; now rejects NaN/Inf distances with `IntegrationDiverged`
- **[P0]** `RegimeManager.transition()` took redundant `current` param that could diverge from `self._current`; removed for Python↔Rust parity
- **[P0]** Merge duplicate `validate_binding_spec` — canonical version now in `validator.py` with all checks merged
- **[P1]** `InformationalExtractor` theta always cancelled to ~0; use median freq × total time
- **[P1]** Python `ImprintModel.modulate_lag` added row-wise offset; use `m[i] - m[j]` (antisymmetric)
- **[P1]** CLI scaffold generated `version: '0.1'` (fails validator); now `'0.1.0'`
- **[P1]** CLI `run` did not compute `cross_layer_alignment`; now uses `compute_plv` between layer pairs
- **[P1]** CLI zeta had no TTL expiry; now decrements TTL counter and resets to 0 on expiry
- **[P1]** Rust UPDE stepper did not check omegas/knm for NaN/Inf; now rejects with `IntegrationDiverged`
- **[P1]** `PhysicalExtractor._snr_estimate` always returned ~1.0; replaced with envelope-CV metric
- **[P1]** `UPDEEngine.compute_order_parameter` reimplemented inline; now delegates to canonical implementation
- **[P1]** `AuditLogger` file writes never flushed; switched to line-buffered I/O
- **[P1]** CLI `run` command ignored spec drivers, boundaries, and actuators; rewritten to wire supervisor

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
- Rust: `Debug` impl for `UPDEStepper`, `#[derive(Debug)]` for `ImprintModel`, `LagModel`
- Rust: doc comments on all public types
- Migrate remaining probe-imports to `importlib.util.find_spec` with lazy imports
- Remove dead `_HAS_RUST` assignments from regimes.py, coherence.py
- Add `#[must_use]` to all pure public Rust functions
- Add crate-level `//!` doc comments to all crates
- Refactor physical.rs Pass 1 to `iter_mut().zip()` iterators
- Make `LockSignature`, `LayerState`, `UPDEState` frozen dataclasses
- CI: add pip + cargo caching, replace manual `cargo-audit` install with `rustsec/audit-check` action
- Add `Documentation` and `Changelog` URLs to `project.urls` (PyPI sidebar)

### Added

- `src/scpn_phase_orchestrator/_compat.py` — shared `TWO_PI`, `HAS_RUST` constants
- `src/scpn_phase_orchestrator/py.typed` — PEP 561 marker
- `tools/check_version_sync.py` — version sync check across pyproject.toml, CITATION.cff, Cargo.toml
- `.dockerignore` — excludes .git, target, caches, site
- CI lint job runs `check_version_sync.py`
- Publish preflight runs Rust clippy + tests
- PyPI and docs badges in README
- `repository` field in Cargo workspace metadata
- 4 new validator tests, 5 hypothesis property tests, 4 coupling lags tests, 4 coupling templates tests, 2 physical extractor tests

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

[Unreleased]: https://github.com/anulum/scpn-phase-orchestrator/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/anulum/scpn-phase-orchestrator/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/anulum/scpn-phase-orchestrator/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/anulum/scpn-phase-orchestrator/releases/tag/v0.1.0
