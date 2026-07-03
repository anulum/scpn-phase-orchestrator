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
The public spectral dispatcher, plus the direct Go, Julia, and Mojo bridges,
apply the shared spectral output validator before publication: eigensystem
payloads must contain finite real non-boolean eigenvalue and Fiedler vectors of
length `N`, eigenvalues must be non-negative and sorted ascending, and
non-trivial Fiedler vectors must be non-zero. Malformed spectral payloads raise
immediately, preserving fallback only for loader or runtime unavailability.
The public AttnRes dispatcher, plus the direct Go, Julia, and Mojo bridges,
apply the shared AttnRes output validator before publication: modulated
coupling payloads may be flat `N*N` vectors or `(N, N)` matrices, but must
contain finite real non-boolean values, remain symmetric, keep a zero diagonal,
and preserve the input `K_nm` zero-edge topology. Malformed AttnRes physics
payloads raise immediately, preserving fallback only for loader or runtime
unavailability.
Merge-window Rust, Go, Julia, and Mojo source-contract adapters route
`MergeReport` evidence through the shared validator before parity publication:
numeric fields must be finite real non-boolean scalars, lock fields must be
plain booleans, consecutive counts must be non-negative integers, and every
signed-margin field must replay the Python reference within tolerance.
The public PAC dispatcher applies the direct phase-amplitude-coupling output
validators to optional backend returns before publication: modulation-index
scalars must be finite values in `[0, 1]`, and PAC-matrix payloads must keep
`N*N` cardinality with every entry inside `[0, 1]`.
The public basin-stability dispatcher applies the shared direct output
validator to optional backend returns before publication: steady-state order
parameters must be finite non-boolean, non-numeric-string scalars inside
`[0, 1]`. Public and direct phase, frequency, flattened coupling, phase-lag,
scalar-control, and count inputs also reject numeric-string aliases before
float coercion, preserving fallback only for loader/runtime unavailability and
not for malformed backend physics evidence.
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
Direct Koopman-EDMD Go, Julia, and Mojo bridges apply the shared EDMD backend
validator before and after execution: snapshot matrices and returned `(A, B, C)`
payloads must be finite real non-boolean numeric matrices, reject complex and
numeric-string aliases before float coercion, and preserve the contracted
`(N, N)`, `(N, m)`, and `(n, N)` shapes.
Direct Lyapunov Go, Julia, and Mojo bridges apply the shared Lyapunov backend
validator before optional runtime loading and after backend execution:
phase/frequency vectors and coupling/lag matrices reject boolean aliases,
complex/object-complex aliases, numeric-string aliases, non-finite values,
shape mismatches, and non-zero coupling diagonals before float coercion.
Returned spectra must keep `N` cardinality, contain finite real non-boolean
non-string exponents, and remain sorted in descending Lyapunov order.
The public NPE dispatcher and direct Go, Julia, and Mojo bridges apply the
shared NPE backend validators before optional runtime loading and after backend
execution: phase vectors and phase-distance matrices reject boolean aliases,
complex/object-complex aliases, numeric-string aliases, non-finite values,
wrong cardinality, asymmetric matrices, non-zero diagonals, and out-of-range
distances before float coercion or publication.
The public ITPC dispatcher and direct Go, Julia, and Mojo bridges apply shared
ITPC validators before optional runtime loading and after backend execution:
trial phase buffers, backend ITPC vectors, persistence scalars, and
exact-reference payloads reject boolean aliases, complex/object-complex
aliases, numeric-string aliases, non-finite values, wrong cardinality,
unit-interval violations, and exact-estimator divergence before float coercion
or publication.
The public OPT-entropy dispatcher and direct Go, Julia, and Mojo bridges apply
the shared ordinal-transition entropy validators before optional runtime loading
and after backend execution: scalar series and ordinal-code vectors reject
boolean aliases, complex/object-complex aliases, numeric-string aliases,
non-finite values, wrong cardinality, non-integer code aliases, and out-of-range
ordinal codes before float coercion or publication.
The public fractal-dimension dispatcher and direct Go, Julia, and Mojo bridges
apply the shared dimension validators before optional runtime loading and after
backend execution: trajectory, epsilon, pair-index, Lyapunov-spectrum,
correlation-integral output, and Kaplan-Yorke scalar payloads reject boolean
aliases, complex/object-complex aliases, numeric-string aliases, non-finite
values, shape drift, and out-of-domain physics values before float coercion or
publication.
The public recurrence dispatcher and direct Go, Julia, and Mojo bridges apply
the shared recurrence validators before optional runtime loading and after
backend execution: trajectory payloads, backend recurrence matrices, and exact
reference-output payloads reject boolean aliases, complex/object-complex
aliases, numeric-string aliases, non-finite values, wrong cardinality,
non-binary cells, diagonal/symmetry drift, and exact-threshold divergence before
float or `uint8` coercion or publication.
The public chimera dispatcher and direct Go, Julia, and Mojo bridges apply the
shared chimera validators before optional runtime loading and after backend
execution: phase vectors, flattened coupling matrices, and backend local-order
outputs reject boolean aliases, complex/object-complex aliases, numeric-string
aliases, non-finite values, wrong cardinality, non-zero coupling diagonals, and
unit-interval drift before float coercion or publication.
The public delay-embedding dispatcher and direct Go, Julia, and Mojo bridges
apply shared embedding validators before optional runtime loading and after
backend execution: scalar signals, embedded trajectories, mutual-information
scalars, and nearest-neighbor backend outputs reject boolean aliases,
complex/object-complex aliases, numeric-string aliases, non-finite values, wrong
cardinality, delay-index drift, non-integral neighbor indices, and self-neighbor
violations before float coercion or publication.
The public transfer-entropy dispatcher and direct Go, Julia, and Mojo bridges
apply shared TE validators before optional runtime loading and after backend
execution: source/target phase vectors, flattened phase-series payloads,
backend scalar or matrix outputs, and exact-reference payloads reject boolean
aliases, numeric-string aliases, complex values, non-finite values, wrong
cardinality, entropy-bound violations, non-zero matrix diagonals, and
exact-histogram-estimator divergence before float coercion or publication.
The public entropy-production dispatcher and direct Go, Julia, and Mojo bridges
apply shared entropy-production validators before optional runtime loading and
after backend execution: phase vectors, frequency vectors, coupling matrices,
scalar controls, and backend entropy-rate outputs reject boolean aliases,
numeric-string aliases, complex values, non-finite values, shape mismatches,
negative timesteps, and negative rates before float coercion or publication.
The public Poincare dispatcher and direct Go, Julia, and Mojo bridges apply
shared Poincare validators before optional runtime loading and after backend
execution: trajectory buffers, phase buffers, normal vectors, scalar controls,
result records, and backend crossing/time outputs reject boolean aliases,
numeric-string aliases, complex values, non-finite values, malformed
cardinality, out-of-range crossing counts, and non-increasing crossing times
before float coercion or publication.
The public winding dispatcher and direct Go, Julia, and Mojo bridges apply
shared winding validators before optional runtime loading and after backend
execution: phase buffers, integer scalar controls, and backend winding vectors
reject boolean aliases, numeric-string aliases, complex values, non-finite
values, malformed cardinality, fractional integer evidence, out-of-bound
winding counts, and exact-reference divergence before float or integer
coercion or publication.
The public Strang-splitting dispatcher and direct Go, Julia, and Mojo bridges
apply shared splitting validators before optional runtime loading and after
backend execution: phase vectors, frequency vectors, flattened coupling and
phase-lag matrices, and backend phase outputs reject numeric-string aliases,
complex values, non-finite values, malformed cardinality, non-zero
self-coupling, and out-of-domain torus phases before float coercion or
publication. The direct adapters also reject boolean aliases before optional
runtime loading. Mojo stdout remains a text transport with exact phase-line
cardinality checks before the same torus validator runs.

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
- `monitor/psychedelic` Go, Julia, and Mojo entropy adapters share one direct
  validator that rejects boolean, numeric-string, complex, non-finite, and
  invalid-bin-count aliases before optional runtime loading. Backend entropy
  outputs are rechecked as finite real scalars inside `[0, log(n_bins)]` before
  public monitor publication.
- `monitor/pid` Go, Julia, and Mojo partial-information-decomposition adapters
  share one direct validator that rejects boolean, numeric-string, complex,
  non-finite, non-integral, out-of-range, and invalid-bin-count aliases before
  optional runtime loading. Backend redundancy/synergy outputs are rechecked as
  finite non-negative real scalars before public monitor publication.
- `monitor/entropy_prod` Go, Julia, and Mojo adapters share one direct validator
  that rejects boolean, numeric-string, complex, non-finite, shape-mismatched,
  and negative-timestep aliases before optional runtime loading. Backend
  entropy-rate outputs are rechecked as finite non-negative real scalars before
  public monitor publication.
- `monitor/poincare` Go, Julia, and Mojo adapters share one direct validator
  that rejects boolean, numeric-string, complex, non-finite, cardinality, and
  crossing-count aliases before optional runtime loading. Backend crossing and
  time buffers are rechecked as finite real vectors with valid sampled-interval
  bounds before public monitor publication; Mojo stdout remains a text transport
  with explicit line-cardinality checks.
- `monitor/winding` Go, Julia, and Mojo adapters share one direct validator
  that rejects boolean, numeric-string, complex, non-finite, cardinality, and
  scalar-control aliases before optional runtime loading. Backend winding vectors
  are rechecked as finite integer vectors with valid wrapped-increment bounds
  and exact NumPy reference equality before public monitor publication.
- `monitor/twin_confidence` Go, Julia, and Mojo adapters share one direct
  validator that rejects boolean, numeric-string, complex, non-finite,
  cardinality, and order-parameter range aliases before optional runtime
  loading. Backend divergence pairs are rechecked as finite real
  `(Jensen-Shannon, Wasserstein-1)` outputs inside `[0, ln 2] x [0, 1]` before
  public monitor, Prometheus, Studio, or conformal-gate publication.
