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

## Invariants

`delay_embed` is exact indexing and should match across backends without
tolerance. Mutual information is non-negative. Nearest-neighbor distances
are finite non-negative values with integer neighbor indices in range and
no self-neighbor for non-trivial inputs.

::: scpn_phase_orchestrator.monitor.embedding
