# Autotune

The autotune subsystem provides tools for identifying unknown system
parameters and discovering governing dynamics from raw data.

## Why this subsystem is review-oriented

Autotune is designed as an evidence generator before controller changes, not as a
direct production control channel. It turns observed traces into candidate hypotheses
that humans can review against domain constraints and safety policy.

In practical usage, teams typically use it to:

- discover coupling hypotheses in previously unmapped domains,
- validate model-form assumptions before full policy activation,
- and generate bounded transfer proposals that can be replayed and compared.

The review-only boundary is intentional: it preserves explainability and prevents
autonomous, opaque changes from entering live control surfaces without policy
inspection.

## Phase-SINDy Symbolic Discovery

The `PhaseSINDy` module implements **Sparse Identification of Nonlinear
Dynamics** tailored for phase oscillator networks. It allows the
orchestrator to act as an "Autonomous Physicist," reverse-engineering
the differential equations of a system from observed time-series data.

### Theoretical Basis

SINDy assumes that the dynamics $\dot{\theta}$ can be represented as
a sparse linear combination of terms from a library $\Theta$:

$$ \dot{\theta} = \Theta(\theta) \Xi $$

For SPO, the library $\Theta$ includes:
1. **Constant terms:** Representing natural frequencies $\omega_i$.
2. **Coupling terms:** $\sin(\theta_j - \theta_i)$ representing Kuramoto-style interactions.

The model uses **Sequentially Thresholded Least Squares (STLSQ)** to
discover the sparsest set of coefficients that explain the data,
effectively filtering out noise and revealing the underlying topology.

### Use Cases
- **System Identification:** Discovering the coupling strength $K_{nm}$
  in a biological network where the wiring is unknown.
- **Topological Verification:** Verifying that a physical system actually
  follows the assumed Kuramoto model before engageing control logic.
- **Anomaly Detection:** Detecting shifts in the governing equations
  (e.g., a component failure that changes the interaction physics).

::: scpn_phase_orchestrator.autotune.sindy

## Frequency Identification

Identifies natural frequencies $\omega_i$ from phase time-series.

The dedicated frequency-identification reference page owns the full
mkdocstrings inventory for `scpn_phase_orchestrator.autotune.freq_id`. This
aggregate page links to that surface instead of declaring a second primary
mkdocstrings target for the same dataclasses.

See [Frequency Identification](autotune_freq_id.md).

## Coupling Estimation

Estimates the coupling matrix $K_{nm}$ assuming a fixed interaction model.

::: scpn_phase_orchestrator.autotune.coupling_est

## End-to-End Pipeline

The pipeline module composes phase extraction, frequency identification,
SINDy-style discovery, and coupling estimation into reviewable auto-binding
candidate records.

::: scpn_phase_orchestrator.autotune.pipeline

## Reviewable Binding Proposals

The binding-proposal module converts time-series CSV, event-log JSON, and graph
JSON payloads into `StudioProjectState` records containing reviewable
`binding_spec.yaml` text, confidence factors, provenance, and binding-validator
diagnostics.

::: scpn_phase_orchestrator.autotune.binding_proposal

## Time-Series Discovery Evidence

The discovery module extracts deterministic review evidence from raw
time-series tables: sparse derivative regressions, phase-aware Kuramoto SINDy
fits for phase-like columns, residual-scored SINDy library selection,
correlation graph edges, lagged directed graph inference,
connected-component clusters, and regular time-column sample-rate inference.
Non-phase data carries an explicit phase-SINDy skipped status. The reports are
JSON-ready provenance for binding review and do not promote actuation.

::: scpn_phase_orchestrator.autotune.discovery

## Replay-Only Learners

The learner module exposes PPO-like, SAC-like, and hybrid-physics proposal
generators behind the existing replay gates. These helpers emit audit records
and keep `actuation_permitted` false.

::: scpn_phase_orchestrator.autotune.learners

## Operator use model

Autotune in this system is intended as a discovery and review surface first.
Its outputs should be understood as candidate proposals with evidence, not as
immediate production actions.

That separation is reflected by the `actuation_permitted=false` audit flag and
the existing replay-only flow: operators can inspect candidate dynamics, compare
against domain constraints, and explicitly promote a policy only through normal
supervision gates.

In practical terms, autotune is most valuable in three moments:
- preflight analysis on unknown domains,
- topological recovery after a major drift event,
- and proposal generation for domain-specific handoff when new systems are onboarded.

The same evidence record model used here is what allows these candidate policies to
be replayed and compared across time windows and boundary profiles.
