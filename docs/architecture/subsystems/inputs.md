# Subsystem: `binding` / `oscillators` / `drivers` / `imprint` — the front end

Turns a domainpack into phase. `binding` 26 files (~7.1k LOC, incl. `digital_twin/`
and `semantic/` subpackages), `oscillators` 8, `drivers` 4, `imprint` 3.

## Inputs

- A domainpack YAML/JSON file. Root keys (fail-closed): `layers`,
  `oscillator_families`, `coupling`, `drivers`, `objectives`, `boundaries`,
  `actuators`, plus `name`, `version`, `safety_tier`, `sample_period_s`,
  `control_period_s`; optional `channels`, `cross_channel_couplings`,
  `imprint_model`, `geometry_prior`, `amplitude`, `value_alignment`.
- Raw signals into extractors: a `(T,)` waveform (physical), sorted timestamps
  (informational), or a state-index sequence (symbolic), plus `sample_rate`.

## Outputs

- `BindingSpec` — the typed, validated hierarchy contract.
- `list[PhaseState]` — `theta`, `omega`, `amplitude`, `quality`, `channel`,
  `node_id` per oscillator. The canonical phase hand-off record.
- Driver reference phases `Ψ(t)`; imprint modulations of `K`, `α`, `μ`.

## Processing model

- `load_binding_spec` (`loader.py`) — fail-closed parse; `validate_binding_spec`
  cross-field checks; `resolve_extractor_type` maps aliases to algorithms.
- **Physical** extractors — Hilbert analytic signal, wavelet ridge, and
  zero-crossing algorithms; instantaneous phase/frequency with
  algorithm-specific quality.
- **Informational** extractor — inter-event-interval cadence → phase;
  inverse-coefficient-of-variation quality.
- **Symbolic** extractor — ring (position on the circle) or graph-walk cumulative
  phase; transition-regularity quality.
- **Drivers** — sinusoidal (physical), cadence-wrapped (informational), cyclic
  lookup (symbolic).
- **Imprint** — exponential-decay + saturation memory state modulating coupling.

## Backends

Extractors carry optional Rust paths (`spo_kernel.physical_extract`,
`event_phase`, `ring_phases_rust`, …) with NumPy fallback. The phase-extraction
path is otherwise Python/NumPy by design (LAPACK/FFT-favoured).

## Wiring

`load_binding_spec` is the single entry for every domainpack. `extract_initial_phases`
wires families to extractors for UPDE start-up; MQTT and OPC-UA runtime tag
bridges also dispatch waveform tags through their declared physical extractor
type. The resolved `BindingSpec` flows to `coupling` and `upde`.

## Scope boundaries

- Runtime waveform dispatch is implemented for the generic MQTT and OPC-UA input
  bridges. Domain-specific app pipelines that expose their own service schemas
  may still define narrower extraction policy.
- Inert fields parsed but unused: `PhaseState.node_id` downstream,
  `ChannelSpec.metric_semantics`, `ImprintState.attribution`.
