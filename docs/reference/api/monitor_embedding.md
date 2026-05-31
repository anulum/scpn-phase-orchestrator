# Delay Embedding — Phase-Space Reconstruction

The `monitor.embedding` module reconstructs phase-space trajectories from
scalar oscillator traces. It provides delay-coordinate embedding,
Fraser-Swinney average mutual information, nearest-neighbor distances,
and Python-side wrappers for optimal delay and embedding dimension.

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

Boolean aliases, complex samples, non-finite samples, non-vector signal
payloads, malformed flattened embedding lengths, invalid delay/dimension
requests, invalid lags, and invalid bin counts are rejected before
shared-library, Julia, or subprocess execution.

Direct backend return payloads are validated before they are handed back to the
public monitor boundary. The public dispatcher repeats the same physics-facing
checks after backend fallback resolution so a shape-correct optional backend
cannot silently return the wrong phase-space reconstruction. Delay-embedding
outputs must have the exact `(T_effective, dimension)` shape and match the
mathematical indexing `x[t + k*tau]`; mutual-information outputs must be finite
non-negative real scalars; nearest-neighbor outputs must contain finite
non-negative distances and integral in-range neighbor indices, with
self-neighbors rejected for non-trivial embeddings. Malformed Mojo text output
is normalised to deterministic `ValueError` failures rather than leaking parser
exceptions.

## Invariants

`delay_embed` is exact indexing and should match across backends without
tolerance. Mutual information is non-negative. Nearest-neighbor distances
are finite non-negative values with integer neighbor indices in range and
no self-neighbor for non-trivial inputs. Optional backend outputs that violate
those invariants are rejected and the dispatcher falls back to the next
available backend rather than returning a corrupted embedding or a truncated
neighbor index.

::: scpn_phase_orchestrator.monitor.embedding
