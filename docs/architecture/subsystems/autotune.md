# Subsystem: `autotune` — offline binding-spec inference

Infers a binding spec from raw time series so a domain can be onboarded from data
rather than hand-authored. 14 files, ~5.9k LOC. Offline; not in the step loop.

## Inputs

Multichannel signals `(n_channels, n_samples)` + sampling rate; optionally a
graph or an event log.

## Outputs

`AutoTuneResult(omegas, knm, alpha, n_layers, dominant_freqs, K_c_estimate)`;
discovered symbolic equations (SINDy); `BindingSpec` proposals ready for the CLI
or server.

## Processing model

`identify_binding_spec` chains: phase extraction → frequency identification by
exact Dynamic Mode Decomposition → least-squares coupling estimation → critical-
coupling prior. `PhaseSINDy` performs sparse symbolic regression over a
trigonometric library. RL-style policy search, learner generators
(PPO/SAC/hybrid), knob attribution, and a candidate-safety certificate round out
the lane.

## Backends

SINDy has a Rust path; coupling estimation is Python-only (least-squares).

## Wiring

`propose_binding_from_*` produces a `BindingSpec` consumed by `load_binding_spec`
and then the normal pipeline. SINDy output is for inspection.

## Scope boundaries

The candidate-safety certificate generates a report but does **not** block a
proposal (no enforcement gate). Policy-search and learner outputs are proposals,
not live actuation.
