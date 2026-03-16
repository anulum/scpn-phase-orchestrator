# SCPN Phase Orchestrator

![Synchronization Manifold](assets/synchronization_manifold.png)

Domain-agnostic Kuramoto phase-dynamics engine for synchronising coupled-cycle systems.

## Documentation

### Concepts

- [System Overview](concepts/system_overview.md) -- pipeline, dual-objective control
- [Oscillators: P/I/S Channels](concepts/oscillators_PIS.md) -- three extraction channels
- [Control Knobs: K, alpha, zeta, Psi](concepts/knobs_K_alpha_zeta_Psi.md) -- equations, supervisor use
- [Memory Imprint Model](concepts/memory_imprint.md) -- exposure accumulation, decay, attribution
- [Phase Control Theory](concepts/phase_control_theory.md) -- control-theoretic framing
- [Domain Utilisation Schemas](concepts/domain_utilisation_schemas.md) -- per-domain mapping patterns

### Specifications

- [Binding Spec Schema](specs/binding_spec.schema.json) -- JSON Schema for domain bindings
- [Phase Contract](specs/phase_contract.md) -- theta, omega, quality, channel contract
- [UPDE Numerics](specs/upde_numerics.md) -- integrator details, stability, wrapping
- [Lock Metrics](specs/lock_metrics.md) -- PLV, layer R, cross-layer alignment
- [Knm Semantics](specs/knm_semantics.md) -- coupling matrix contract
- [Regime Manager](specs/regime_manager.md) -- NOMINAL / DEGRADED / CRITICAL / RECOVERY
- [Boundary Contract](specs/boundary_contract.md) -- soft/hard boundaries, observer
- [Action Compose](specs/action_compose.md) -- ControlAction, mapper, projector
- [Eval Protocol](specs/eval_protocol.md) -- R_good/R_bad convergence, replay
- [Plugin API](specs/plugin_api.md) -- custom extractors, constraints, drivers
- [Policy DSL](specs/policy_dsl.md) -- YAML rule engine (v0.2+)
- [Imprint Dynamics](specs/imprint_dynamics.md) -- exposure model, decay, modulation
- [Geometry Constraints](specs/geometry_constraints.md) -- Knm projection constraints
- [Knm Calibration](specs/knm_calibration.md) -- coupling matrix tuning procedure
- [Audit Trace](specs/audit_trace.md) -- JSONL format, deterministic replay

### Tutorials

- [01 -- New Domain Checklist](tutorials/01_new_domain_checklist.md)
- [02 -- Oscillator Hunt Sheet](tutorials/02_oscillator_hunt_sheet.md)
- [03 -- Build Knm Templates](tutorials/03_build_knm_templates.md)

### Gallery

- [Domainpack Gallery](galleries/domainpack_gallery.md) -- all 24 domainpacks

### Reference

- [Assumptions & Empirical Constants](ASSUMPTIONS.md) -- provenance of every threshold
- [references.bib](references.bib) -- BibTeX bibliography

## License

GNU AGPL v3. Commercial licensing available.
