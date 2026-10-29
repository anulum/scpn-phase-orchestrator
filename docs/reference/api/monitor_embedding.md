# Delay Embedding — Phase-Space Reconstruction

## Why this module is exposed

Delay embedding is how SPO can recover state-space structure when only scalar
observations are available. It lets monitoring and anomaly workflows operate from
raw traces without requiring a separate external embedding stack.

The module is intentionally strict at boundaries so a downstream controller uses
phase-space features with deterministic semantics.

The `monitor.embedding` module reconstructs phase-space trajectories from
scalar oscillator traces. It provides delay-coordinate embedding,
Fraser-Swinney average mutual information, nearest-neighbor distances,
and Python-side wrappers for optimal delay and embedding dimension.

## Operational use

Use this module when a monitor needs geometry and short-term predictability from
non-vector observations (for example, single-channel physiology traces mapped into
state-space structure).

A practical sequence is:

1. Build a validated embedded matrix (`delay_embed` or `auto_embed`);
2. Select delay/dimension from information-theoretic and neighborhood diagnostics;
3. Feed the reconstructed state to downstream monitor logic;
4. Validate resulting geometrical signals against replay baselines before using them
   in policy experiments.

Keep this boundary explicit: embedding is a feature-construction layer, not a
replacement for raw trace quality checks.

## API

```python
from scpn_phase_orchestrator.monitor.embedding import (
    delay_embed,
    embedding_backend,
    mutual_information,
    nearest_neighbor_distances,
    optimal_delay,
    optimal_dimension,
    auto_embed,
)
```

`delay_embed(signal, delay, dimension)` returns the standard
delay-coordinate matrix:

```text
v(t) = [x(t), x(t + tau), x(t + 2 tau), ...]
```

`mutual_information(signal, lag, n_bins)` estimates average mutual
information for delay selection. `nearest_neighbor_distances(embedded)`
supports false-nearest-neighbor dimension selection.

## Owner selection

Every function accepts `backend="auto"` or one owner: `rust`, `mojo`, `julia`,
`go` or `python`.

- `"auto"` uses the fastest available owner that provides the operation. Rust
  owns delay embedding and the two native searches. Rust has no standalone MI
  or kNN kernel, so those operations go to the next available owner.
- An explicit owner must be available, must provide the operation and must
  succeed. Otherwise the call raises `RuntimeError`; it never falls back.
  For example, `mutual_information(..., backend="rust")` is refused.
- `optimal_delay` and `optimal_dimension` with a non-Rust owner compose that
  owner's MI, embedding and kNN primitives.
- `embedding_backend(slot, backend="auto")` reports the owner that serves
  `"de"`, `"mi"`, `"nn"`, `"optimal_delay"` or `"optimal_dimension"`. For the
  searches, `python` means the composed search.

## Shared numerical law

All owners compute the same quantities:

- **Mutual information.** Each window (`x(t)` and `x(t + lag)`) is binned over
  its own range with the edges of NumPy `histogram2d`: edge `k` is
  `k * step + min` and the last edge is `max`. A value on an interior edge
  opens the next bin; the last edge closes the last bin. Probabilities divide
  counts by the number of pairs. Native results are checked against the
  NumPy reference with a relative tolerance of `1e-12`.
- **Nearest neighbours.** The neighbour of a row is the first minimum of the
  squared distance accumulated over coordinates in order; the distance is its
  square root. A native result is accepted when each distance equals the
  distance to its own neighbour within `1e-14` relative and that neighbour is
  minimal within `1e-12` relative. The Mojo compiler may fuse the squared sum
  into multiply-adds, which changes the last bit.

## Direct backend boundary

Go, Julia, and Mojo direct bridge calls share the same typed pre-dispatch
contract before optional runtime loading. Signal payloads must be finite
real one-dimensional samples. Real numeric object arrays are supported;
Boolean, text, complex and temporal aliases, including object-stored durations,
are rejected before float conversion. Delay, dimension, lag, bin-count, row-count, and
embedding-dimension controls must be integer values in the public API domain.
Embedded nearest-neighbor payloads must be finite real flat `float64` arrays
whose length matches `T*m`, with numeric-string aliases rejected at the same
pre-dispatch boundary.

Boolean aliases, object-dtype complex aliases, complex samples, non-finite
samples, numeric-string aliases, non-vector signal payloads, malformed
flattened embedding lengths, invalid delay/dimension requests, invalid lags,
and invalid bin counts are rejected before shared-library, Julia, or
subprocess execution.

Direct backend return payloads are validated before they are handed back to the
public monitor boundary. The public dispatcher repeats the same physics-facing
checks after backend fallback resolution so a shape-correct optional backend
cannot silently return the wrong phase-space reconstruction. Delay-embedding
outputs must have the exact `(T_effective, dimension)` shape and match the
mathematical indexing `x[t + k*tau]`; numeric-string aliases and object-dtype
complex aliases are rejected before float coercion; mutual-information outputs
must be finite non-negative real scalars; nearest-neighbor outputs must contain
finite non-negative distances and integral in-range neighbor indices, with
self-neighbors rejected for non-trivial embeddings. Malformed Mojo text output
is normalised to deterministic `ValueError` failures rather than leaking parser
exceptions.

The installed Rust entry points retain their contiguous one-dimensional float64
array ABI. They check original non-Boolean integer metadata, finite samples and
finite nonnegative FNN tolerances. Embedding-window, output-size and histogram-size
products are checked before allocation. A valid delay exceeding the series length
returns dimension 1 from the native FNN selector without signed truncation.

The [real-runtime parity record](../data/embedding_real_runtime_2026-10-29.json)
holds the canonical parity gate on actual Python, Rust, Go, Julia and Mojo
backends at 64 and 160 samples, with source, extension and native-artefact
hashes. Rust has no standalone mutual-information or nearest-neighbor FFI; its
supported contracts are recorded separately. Three calls per backend on a shared
host are local regression evidence, not a controlled speedup comparison. The
earlier [source-type benchmark](../data/embedding_measurement_types_benchmark_2026-09-26.json)
predates the shared binning law.

## Invariants

`delay_embed` is exact indexing and should match across backends without
tolerance. Mutual information is non-negative. Nearest-neighbor distances
are finite non-negative values with integer neighbor indices in range and
no self-neighbor for non-trivial inputs. Backend outputs that violate those
invariants, or that disagree with the reference, raise `ValueError`; they are
never returned. In automatic selection, an owner whose native call fails with a
runtime or loading error is replaced by the Python reference; an explicit owner
raises `RuntimeError` instead.

## Practical usage profile

Teams typically use this module during inspection and replay pipelines:

- validate embedding settings with `optimal_delay`/`optimal_dimension`,
- extract trajectory geometry with delay coordinates,
- use downstream monitors on the reconstructed state space instead of direct raw
  samples.

That pattern keeps signal reconstruction and control logic in one audited path.

::: scpn_phase_orchestrator.monitor.embedding
