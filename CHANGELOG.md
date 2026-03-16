# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- `build_nengo_network()` replaced with pure-NumPy `build_numpy_network()` (old name kept as alias)
- `nengo` optional extra is now empty (nengo 4.x incompatible with NumPy 2.x)

### Added

- `bench/bench_stuart_landau.py` — Stuart-Landau engine benchmark harness
- `bench/baseline.json` — UPDE + SL baseline timing data (Python + Rust)
- `tests/test_geometry_walk.py` — geometry_walk domainpack spec/run/policy tests
- `tests/test_queuewaves_pipeline.py` — QueueWaves ConfigCompiler + PhaseComputePipeline tests
- `ARCHITECTURE.md` — system overview, pipeline diagram, module map
- `SUPPORT.md` — help channels, security advisories link
- `GOVERNANCE.md` — decision process, project lead
- `CONTRIBUTORS.md` — contributor attribution
- `NOTICE.md` — AGPL compliance boundary table, third-party attribution
- `REUSE.toml` — REUSE 3.0 license compliance
- `Makefile` — 20 convenience targets (test, lint, fmt, docs, bench, bridge, etc.)
- `requirements.txt` / `requirements-dev.txt` — pinned dependency files
- `.gitattributes` — LF normalization, linguist overrides
- `_typos.toml` — domain-specific allowlist for typos checker
- `.github/FUNDING.yml` — sponsorship links
- `.github/workflows/pre-commit.yml` — pre-commit hook enforcement in CI
- `.github/workflows/codeql.yml` — CodeQL semantic security analysis (weekly + PR)
- `.github/workflows/scorecard.yml` — OpenSSF Scorecard (weekly + push)
- `.github/workflows/stale.yml` — auto-close stale issues/PRs (60+14 day lifecycle)
- `.github/workflows/release.yml` — automated GitHub Release with changelog extraction and SBOM
- `src/scpn_phase_orchestrator/exceptions.py` — centralized exception hierarchy (`SPOError`, `BindingError`, `ValidationError`, `ExtractorError`, `EngineError`, `PolicyError`, `AuditError`)
- `resolve_extractor_type()` in `binding/types.py` — maps aliases (`physical`/`informational`/`symbolic`) to algorithm names (`hilbert`/`event`/`ring`)
- `Discussions` link in `[project.urls]`
- `spo report` command: coherence summary from audit log (text + `--json-out`)
- `spo replay --verify` for Stuart-Landau logs (chained phase-amplitude replay)
- `ReplayEngine.verify_determinism_sl_chained()` for SL audit log verification
- `linux-aarch64` wheel target in `publish.yml`

### Fixed

- **Contract drift**: Loader now resolves extractor_type aliases to algorithm names at parse time. Both `physical` and `hilbert` are valid input; internally normalized to algorithm names.
- **Audit hash chain**: `_write_record` strips `_hash` key before digest to prevent user-data collision
- **Stuart-Landau coupling**: clamp `r >= 0` in `_derivative` to prevent sign-flip in RK intermediate stages
- **CLI audit logger**: simulation loop wrapped in `try/finally` to guarantee `close()` on exception
- **Hilbert guard**: `PhysicalExtractor.extract()` validates signal shape before transform
- **Lag model**: `build_alpha_matrix` uses `carrier_freq_hz` for correct seconds→radians conversion
- **Exception swallowing**: `except Exception` → `except ImportError` in stuart_landau.py and pac.py
- **Coupling diagonal**: layer-scoped K update zeros `knm[idx,idx]` after row/column scaling
- **Regime state machine**: CRITICAL must pass through RECOVERY before NOMINAL
- **Geometry projection**: `project_knm` zeros diagonal after constraint projection (Rust parity)
- **Zeta lower bound**: `cli.py` zeta accumulation clamped to `max(0.0, ...)` to prevent negative drive
- **Control period**: `control_period_s` from binding spec now gates supervisor/policy evaluation interval
- **Gallery doc**: corrected "all notebooks validated in CI" to reflect actual 5/24 coverage

### Changed

- `SECURITY.md` — updated supported versions, added Security Advisories link
- `__init__.py` public API expanded: +`AuditLogger`, `BoundaryObserver`, `CouplingBuilder`, `RegimeManager`, `SPOError`, `SupervisorPolicy`
- FFI: `PyStuartLandauStepper.run()` exposed via PyO3
- Coverage guard: reporting threshold raised from 40% to 90%
- `system_overview.md`: added RK45 to methods list, 5 missing data structures to key structures table
- `CONTRIBUTING.md`: added `boundaries:`, `actuators:`, and `policy.yaml` examples
- `ROADMAP.md`: corrected v0.4.1 test counts (1011 Python, 180 Rust)
- `ASSUMPTIONS.md`: corrected CFL and max_dt line references
- `upde_numerics.md`: CFL formula corrected from `1` to `pi` (matching code and ASSUMPTIONS.md)
- `policy_dsl.md`: domainpack count corrected (12/17 → 24/24), added 5 amplitude metrics to fields table
- `index.md`: Policy DSL label changed from "planned v0.2" to "v0.2+"
- `01_new_domain_checklist.md`: policy.yaml example updated to current schema
- `README.md`: 3 missing domainpacks added (autonomous_vehicles, network_security, satellite_constellation)
- `VALIDATION.md`: test counts and domain count updated (1011/180/24)
- `CITATION.cff`: added `abstract` field
- `bio_stub/README.md`: corrected boundary severity (lower: soft, not hard)
- `safety_tier` runtime warning for non-research tiers
- `pyproject.toml`: `scpn-all` extra now requires `spo-kernel>=0.2.0` (was stale `>=0.1.1`)
- `docs/index.md`: added 6 missing nav entries to match `mkdocs.yml` (concepts, specs, gallery)
- `mkdocs.yml`: added `references.bib` to Reference nav section
- `README.md`: quickstart now includes `spo scaffold` example
- `ruff.lint.ignore`: added `S603` (subprocess in tests is expected)
- `CHANGELOG.md`: corrected QueueWaves path from `apps/queuewaves/` to full module path

## [0.4.1] - 2026-03-04

### Added

- **Rust `StuartLandauStepper`** — phase-amplitude ODE integrator in `spo-engine/src/stuart_landau.rs` with Euler/RK4/RK45, zero-alloc scratch, 12 inline tests
- **Rust PAC** — `modulation_index` and `pac_matrix` in `spo-engine/src/pac.rs` (Tort et al. 2010), 5 inline tests
- **FFI `PyStuartLandauStepper`** — PyO3 wrapper delegating to Rust Stuart-Landau stepper
- **FFI `pac_modulation_index` / `pac_matrix_compute`** — PAC functions exposed to Python
- Python `StuartLandauEngine` auto-delegates to Rust when `spo_kernel` available
- Python `modulation_index()` auto-delegates to Rust when `spo_kernel` available
- PAC-driven policy rules: `pac_max`, `mean_amplitude`, `subcritical_fraction`, `amplitude_spread`, `mean_amplitude_layer` metrics in `_extract_metric`
- Amplitude configs (`amplitude:` YAML block) for 6 domainpacks: neuroscience_eeg, cardiac_rhythm, plasma_control, firefly_swarm, rotating_machinery, power_grid
- `pac_gating_alert` and `subcritical_recovery` policy rules for all 6 amplitude domainpacks
- `CoherencePlot` matplotlib implementations: `plot_r_timeline`, `plot_regime_timeline`, `plot_action_audit`, `plot_amplitude_timeline`, `plot_pac_heatmap`
- 4 Rust benchmarks: `sl_euler_step_n64`, `sl_rk4_step_n64`, `sl_1000steps_n64`, `pac_mi_n1000`
- ~30 new Python tests across 5 test files (total ~895)

## [0.4.0] - 2026-03-04

### Added

- **Stuart-Landau amplitude engine** — `StuartLandauEngine` integrates coupled phase-amplitude ODEs (Acebrón et al. 2005, Rev. Mod. Phys.): `dr_i/dt = (μ - r²)r + ε Σ K^r_ij r_j cos(θ_j - θ_i)`. Euler/RK4/RK45 methods, pre-allocated scratch arrays, amplitude clamping, weighted order parameter.
- **Phase-amplitude coupling (PAC)** — `modulation_index()` (Tort et al. 2010), `pac_matrix()`, `pac_gate()` in `upde/pac.py`
- **Modulation envelopes** — `extract_envelope()` (sliding-window RMS), `envelope_modulation_depth()`, `EnvelopeState` in `upde/envelope.py`
- `AmplitudeSpec` dataclass in binding types; `amplitude:` YAML block activates Stuart-Landau mode
- `CouplingState.knm_r` — amplitude coupling matrix alongside phase coupling
- `CouplingBuilder.build_with_amplitude()` for joint phase + amplitude coupling
- `ImprintModel.modulate_mu()` — imprint-dependent bifurcation parameter modulation
- `LayerState.mean_amplitude`, `LayerState.amplitude_spread` (backward-compatible defaults)
- `UPDEState.mean_amplitude`, `UPDEState.pac_max`, `UPDEState.subcritical_fraction`
- CLI `run` command branches on amplitude mode: builds `StuartLandauEngine`, computes PAC, tracks envelope metrics
- Audit logger records `amplitude_mode` in header; replay reconstructs correct engine type
- ~80 new tests across 7 new test files (total ~860)

### Changed

- `AuditLogger.log_header()` accepts `amplitude_mode` parameter
- `ReplayEngine.build_engine()` returns `StuartLandauEngine` when header has `amplitude_mode=True`
- Binding validator rejects `amplitude.epsilon < 0` and non-finite `amplitude.mu`

## [0.3.0] - 2026-03-04

### Added

- **Petri net regime FSM** — `PetriNet`, `Place`, `Arc`, `Transition`, `Marking`, `Guard` for multi-phase protocol sequencing
- **`PetriNetAdapter`** — maps Petri net markings to `Regime` values with highest-severity-wins priority
- **`ProtocolNetSpec`** — binding spec `protocol_net:` key for declarative protocol sequencing in YAML
- **Event-driven transitions** — `EventBus` + `RegimeEvent` pub/sub system with bounded history
- **`RegimeManager.force_transition()`** — bypasses cooldown and hysteresis hold
- **`RegimeManager.transition_history`** — deque of (step, prev, new) tuples (maxlen=100)
- **`hysteresis_hold_steps`** — consecutive-step requirement for soft downward transitions
- **`BoundaryObserver` event wiring** — posts `boundary_breach` events to EventBus
- **SNN controller bridge** (`SNNControllerBridge`) — pure-numpy LIF rate model + Nengo/Lava optional backends
- `nengo` and `lava` optional dependency groups
- Event kinds: `boundary_breach`, `r_threshold`, `regime_transition`, `manual`, `petri_transition`
- CLI wires EventBus, BoundaryObserver events, and Petri net when binding spec declares `protocol_net:`
- Rust `RegimeManager.force_transition()` and `transition_log` for FFI contract parity
- ~90 new tests across 5 new test files

### Changed

- `SupervisorPolicy` accepts optional `petri_adapter` argument; when present, `decide()` delegates regime to Petri net
- `BoundaryObserver.observe()` accepts optional `step` kwarg for event attribution
- `RegimeManager` constructor accepts `event_bus` and `hysteresis_hold_steps` params
- `adapters/__init__.py` exports `SNNControllerBridge`

## [0.2.0] - 2026-03-04

### Added

- **Compound policy DSL** — `CompoundCondition` with AND/OR logic over multiple `PolicyCondition` triggers
- **Action chains** — `PolicyRule.actions` accepts a list of `PolicyAction` items fired on a single trigger
- **Rule rate-limiting** — per-rule `cooldown_s` and `max_fires` fields
- **`stability_proxy` metric** in policy conditions (global mean R)
- **OpenTelemetry export** — `OTelExporter` with span instrumentation, gauge metrics (`spo.r_global`, `spo.stability_proxy`), step counter; no-op fallback when `opentelemetry-api` is absent
- `otel` optional dependency group (`opentelemetry-api>=1.20`, `opentelemetry-sdk>=1.20`)
- Pre-commit hook for version consistency check across pyproject.toml, CITATION.cff, Cargo.toml
- **QueueWaves** — real-time microservice cascade failure detector (`src/scpn_phase_orchestrator/apps/queuewaves/`)
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

[Unreleased]: https://github.com/anulum/scpn-phase-orchestrator/compare/v0.4.1...HEAD
[0.4.1]: https://github.com/anulum/scpn-phase-orchestrator/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/anulum/scpn-phase-orchestrator/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/anulum/scpn-phase-orchestrator/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/anulum/scpn-phase-orchestrator/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/anulum/scpn-phase-orchestrator/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/anulum/scpn-phase-orchestrator/releases/tag/v0.1.0
