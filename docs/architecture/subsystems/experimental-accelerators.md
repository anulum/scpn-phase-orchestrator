# Subsystem: `experimental/accelerators` — polyglot acceleration backends

Despite the name, this is **not aspirational research**: it is the load-bearing
polyglot acceleration layer that the production `coupling`, `monitor`, and `upde`
subsystems dispatch to. 170 files, ~18.9k LOC, organised solely as
`experimental/accelerators/{coupling, monitor, upde}/`.

## Structure

Each accelerated kernel has, per theme, a set of backend modules
`<theme>_{go, julia, mojo[, rust, webgpu], validation}.py`. Go is loaded by
`ctypes`, Julia by `juliacall`, Mojo and WebGPU by their bridges; a Python
`_validation` module guards types, shapes, and finiteness for each theme.

- `coupling/` (17 files): Hodge, spectral, attention residuals, spatial modulator.
- `monitor/` (70 files): Koopman-EDMD, Lyapunov, transfer entropy, ITPC,
  recurrence, chimera, Poincaré, winding, dimension, embedding, twin-confidence,
  entropy production, NPE, optimal entropy, PID, merge-window, …
- `upde/` (79 files): the engine (incl. WebGPU), order parameters, basin
  stability, delay, Doppler, envelope, geometric, hypergraph, inertial, market,
  moving-frame, PAC, reduction, simplicial, splitting, swarmalator, PHA-C
  acceptance/handoff/timeline.

## Inputs / outputs

Backends take flat `float64` arrays (phases, omegas, flattened coupling) plus
scalar parameters and return `float64` arrays / tuples / scalars; the producer
modules in `coupling`/`monitor`/`upde` wrap them for type-checking and audit.
Validation runs before dtype coercion: Python booleans, NumPy boolean scalars,
boolean dtypes, and raw containers carrying boolean aliases are rejected rather
than widened to `0.0`/`1.0` backend payloads.
Direct UPDE delay, Doppler, and moving-frame Go/Julia/Mojo adapters also replay
the owning output validators after backend execution: delayed-Kuramoto outputs
must be finite phase vectors in `[0, 2*pi)`, Doppler outputs must be finite
principal-branch phases, and moving-frame outputs must be finite phase/position
vectors whose final positions match the submitted ballistic velocity schedule.
The public market dispatcher replays the same market output validators for
optional backend returns before publication: `R(t)` must be finite and bounded
in `[0, 1]`, and rolling PLV payloads must keep the expected cardinality, unit
diagonals, symmetry, and `[0, 1]` bounds.
The public geometric dispatcher applies the direct torus output validator to
optional backend returns before publication: phase vectors must keep oscillator
cardinality, contain finite values, and remain in `[0, 2*pi)`.
The public simplicial dispatcher applies the shared direct torus output
validator to optional backend returns before publication: higher-order Kuramoto
phase vectors must keep oscillator cardinality, contain finite values, and stay
inside `[0, 2*pi)`.
The public hypergraph dispatcher and Rust wrapper apply the shared hypergraph
output validator to optional backend returns before publication: mixed-order
phase vectors must keep oscillator cardinality, contain finite values, and stay
inside `[0, 2*pi)`.
The public inertial dispatcher and Rust wrapper apply the shared inertial
output validator to optional backend returns before publication: swing-equation
`theta` and `omega_dot` vectors must keep oscillator cardinality and finite
values, with returned phases remaining inside `[0, 2*pi)`.
The public swarmalator dispatcher and Rust wrapper apply the shared direct
swarmalator output validator to optional backend returns before publication:
positions must keep `(N, D)` shape or `N*D` flattened cardinality, phases must
keep oscillator cardinality, values must be finite real numbers, phases must
stay inside `[0, 2*pi)`, and boolean aliases are rejected before float
coercion.
The public spatial-modulator dispatcher and Rust wrapper, plus the direct Julia
bridge, apply the shared direct spatial-modulator output validator before
publication: outputs must keep `N*N` cardinality or matrix shape, contain finite
real non-boolean values, and preserve the zero self-coupling diagonal before the
public dispatcher reshapes them for callers.
The public Hodge dispatcher and Rust wrapper, plus the direct Go, Julia, and
Mojo bridges, apply the shared Hodge output validator before publication or
parity fallback: gradient, curl, and harmonic payloads must keep `N*N`
cardinality or `(N, N)` matrix shape, contain finite real non-boolean values,
and remain antisymmetric. Malformed backend outputs raise immediately, while
validated numerical parity mismatches still fall back to the NumPy reference.
The public PAC dispatcher applies the direct phase-amplitude-coupling output
validators to optional backend returns before publication: modulation-index
scalars must be finite values in `[0, 1]`, and PAC-matrix payloads must keep
`N*N` cardinality with every entry inside `[0, 1]`.
The public basin-stability dispatcher applies the shared direct output
validator to optional backend returns before publication: steady-state order
parameters must be finite non-boolean scalars inside `[0, 1]`, preserving
fallback only for loader/runtime unavailability and not for malformed backend
physics evidence.
The public order-parameter dispatcher applies the direct scalar output
validators to optional backend returns before publication: order-parameter
magnitudes, PLV, and layer coherence must be finite real scalars inside
`[0, 1]`, and boolean aliases are rejected instead of widened to `0.0` or
`1.0`.
The public Ott-Antonsen reduction dispatcher applies the shared OA output
validator to optional backend returns before publication: returned scalar
records must be finite non-boolean values, `z` must stay inside the OA unit
disk, `R` must match `|z|`, and `psi` must match `atan2(Im(z), Re(z))` for
non-zero radius.
The public envelope dispatcher now applies its direct accelerator output
validators to optional backend returns before publication: RMS-envelope vectors
must keep input cardinality, finite values, and non-negative values, while
modulation depth must be a finite scalar in `[0, 1]`.

## Wiring

The per-language forwarder modules in the production subsystems
(`upde/_engine_mojo.py`, `monitor/_lyapunov_julia.py`, …) re-export from here.
Selection is the per-lane fastest-first chain (see [backends.md](../backends.md)).
Production UPDE, monitor, and coupling Julia forwarders, plus the direct
accelerator Julia `_ensure()` loaders, require `juliacall.Main`, not just an
importable `juliacall` package; partial Julia initialisation is treated as an
unavailable optional backend and the dispatcher falls through instead of
advertising a bridge that will fail after selection. Direct Go loaders demote
host dynamic-loader failures for present but unloadable `c-shared` artefacts to
`ImportError`, preserving the same optional-backend unavailability contract as a
missing shared library. Direct Mojo executable probes now reject present but
non-executable compiled backend artefacts as optional-backend unavailability,
and direct Mojo subprocess launch failures such as host file-format rejection
are demoted to the same `ImportError` contract, while each bridge keeps its
module-specific missing-file build command.
Nothing here is re-exported in the public API — access is always indirect
through a production subsystem's dispatcher.

## Scope boundaries

- The naming is misleading; treat this as the acceleration layer, not a research
  sandbox.
- The polyglot backends are environment-gated — Go/Julia/Mojo require their
  toolchains; absent or unloadable Go shared libraries, partially initialised
  Julia runtimes, non-executable Mojo backend artefacts, and host-rejected Mojo
  executable formats fail before backend execution and fall through to Python
  through the owning dispatcher.
- `monitor/psychedelic` is a heuristic with no cited reference; the PHA-C
  acceptance lane is a deterministic evidence-binding chain (distinct from the
  conformal twin-confidence gate, which lives in `monitor/twin_conformal_gate.py`).
