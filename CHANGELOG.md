# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-03-01

### Added

- UPDE engine with RK4 integration and pre-allocated scratch arrays
- 3-channel oscillator model (phase, instantaneous frequency, coherence)
- Coupling matrix management with decay and cross-hierarchy boosts
- Supervisor with order-parameter monitoring and threshold actions
- Actuation mapper for domain-agnostic output binding
- CLI entry point (`spo`) with init, run, status commands
- 4 domainpacks: audio, neuro, fusion, generic
- Binding spec schema for domainpack configuration
- PhaseExtractor base class for domain-specific signal intake

[Unreleased]: https://github.com/anulum/scpn-phase-orchestrator/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/anulum/scpn-phase-orchestrator/releases/tag/v0.1.0
