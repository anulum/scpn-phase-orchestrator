# Geometry Constraints

Geometry constraints project candidate coupling matrices before downstream
integration. `project_knm` validates finite real square inputs, applies the
supplied constraint sequence and zeros the diagonal. It returns independent
storage without modifying the input.

## Built-in constraints

| Constraint | Mathematical result | Diagonal |
|------------|---------------------|----------|
| `SymmetryConstraint` | `(K + K.T) / 2` | Preserved |
| `NonNegativeConstraint` | `max(K, 0)` | Clipped as any other entry |

Symmetry uses addition before halving whenever the sum is finite. Only
an overflowing sum uses `K/2 + K.T/2`. Thus two maximum finite `float64`
coefficients produce their finite mean, and two least positive subnormals
retain that value. Blanket half-before-addition would erase the latter.
Both methods validate original source types before numeric conversion.
Boolean, complex, text and temporal aliases are refused; real numeric object
arrays remain compatible. The computation is a float64 structural projection,
not evidence of stable physical integration for arbitrarily large weights.

## Public Python and Rust entry points

```python
import numpy as np
from scpn_phase_orchestrator.coupling import (
    NonNegativeConstraint, SymmetryConstraint, project_knm, validate_knm,
)

raw = np.array([[2.0, 0.8], [-0.4, 3.0]])
projected = project_knm(raw, [SymmetryConstraint(), NonNegativeConstraint()])
np.testing.assert_allclose(projected, [[0.0, 0.2], [0.2, 0.0]])
validate_knm(projected)
```

Python projection executes NumPy. The explicit native method
`spo_kernel.PyCouplingBuilder.project(raw.ravel(), 2)` calls Rust
`spo_engine::coupling::project_knm` and returns a new row-major flat list.
It has the same symmetry-then-non-negativity-then-zero-diagonal result.
Native shape admission requires exact `n*n` cardinality and a representable
size product. Invalid original source values or cardinality refuse before
mutation. Negative native count extraction raises `OverflowError`; other
invalid count types, overflowing products and wrong cardinality raise
`ValueError`. Both implementations accept an empty matrix with zero dimensions;
`validate_knm` accepts its vacuous structural invariants.

## Constraint order and custom extensions

Constraint order changes the result for signed pairs: `[1, -1]` averages to
zero when symmetrised first, but clips to `[1, 0]` and averages to `0.5` when
non-negativity is applied first. `project_knm` applies exactly the supplied
order. Symmetry and non-negativity are not implicit when their constraints
are absent. Only diagonal zeroing is unconditional at the end.

Custom constraints must subclass `GeometryConstraint` and implement
`project(knm)`. Each returned matrix is checked for original real source
types, square shape, unchanged dimensions and finite values before the next
constraint runs. A rejected result does not change the original input.
Arbitrary custom stacks do not guarantee symmetry or non-negativity;
call `validate_knm` when the downstream profile requires those invariants.

## Runtime integration and verification

The runtime constructs this constraint list from `binding_spec.geometry_prior`:
`symmetric` selects symmetry, and `non_negative` or `nonneg` selects clipping.
Projection is applied to effective coupling after imprint modulation and
before the UPDE step. Tests exercise real projected matrices in the public RK4
engine and the CLI's geometry-prior path, including original-input preservation
and refusal followed by valid recovery.

`tests/test_geometry_projection_finite.py` covers finite extremes,
subnormal rounding, cancellation, empty systems and public consumption.
`native-tests/test_geometry_projection.py` exercises the installed native
entry point and compares both paths with exact rational pair means.
Run public tests in an actual installed-native and an actually kernel-absent
interpreter; only the native lane includes the direct-native selection.

The [runtime snapshot](../reference/data/geometry_projection_runtime_benchmark_2026-10-02.json)
records repeated Python/direct-PyO3 samples, input/source hashes, actual runtime
versions, native binary provenance and shared-host load. Reproduce with
`python -m benchmarks.geometry_projection_benchmark`. Rust core measurements
use `cargo bench -p spo-engine --bench coupling_projection_bench`. These
non-isolated workstation measurements support local regression and parity
checks, not production latency or causal speed-up claims.
