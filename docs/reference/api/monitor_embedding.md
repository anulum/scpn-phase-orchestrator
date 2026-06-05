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

## Direct backend boundary

Go, Julia, and Mojo direct bridge calls share the same typed pre-dispatch
contract before optional runtime loading. Signal payloads must be finite
real one-dimensional `float64` arrays. Delay, dimension, lag, bin-count,
row-count, and embedding-dimension controls must be integer values in the
public API domain. Embedded nearest-neighbor payloads must be finite real
flat `float64` arrays whose length matches `T*m`.

Boolean aliases, object-dtype complex aliases, complex samples, non-finite
samples, non-vector signal payloads, malformed flattened embedding lengths,
invalid delay/dimension requests, invalid lags, and invalid bin counts are
rejected before shared-library, Julia, or subprocess execution.

Direct backend return payloads are validated before they are handed back to the
public monitor boundary. The public dispatcher repeats the same physics-facing
checks after backend fallback resolution so a shape-correct optional backend
cannot silently return the wrong phase-space reconstruction. Delay-embedding
outputs must have the exact `(T_effective, dimension)` shape and match the
mathematical indexing `x[t + k*tau]`; object-dtype complex aliases are rejected
before float coercion; mutual-information outputs must be finite non-negative
real scalars; nearest-neighbor outputs must contain finite non-negative
distances and integral in-range neighbor indices, with self-neighbors rejected
for non-trivial embeddings. Malformed Mojo text output is normalised to
deterministic `ValueError` failures rather than leaking parser exceptions.

## Invariants

`delay_embed` is exact indexing and should match across backends without
tolerance. Mutual information is non-negative. Nearest-neighbor distances
are finite non-negative values with integer neighbor indices in range and
no self-neighbor for non-trivial inputs. Optional backend outputs that violate
those invariants are rejected and the dispatcher falls back to the next
available backend rather than returning a corrupted embedding or a truncated
neighbor index.

## Practical usage profile

Teams typically use this module during inspection and replay pipelines:

- validate embedding settings with `optimal_delay`/`optimal_dimension`,
- extract trajectory geometry with delay coordinates,
- use downstream monitors on the reconstructed state space instead of direct raw
  samples.

That pattern keeps signal reconstruction and control logic in one audited path.

::: scpn_phase_orchestrator.monitor.embedding
