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
Direct UPDE Doppler and moving-frame Go/Julia/Mojo adapters also replay the
owning public output validators after backend execution: Doppler outputs must be
finite principal-branch phases, and moving-frame outputs must be finite
phase/position vectors whose final positions match the submitted ballistic
velocity schedule.

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
