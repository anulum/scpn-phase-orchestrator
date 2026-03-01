# Plugin API

Extension points for domain-specific behaviour without modifying engine code.

## Custom PhaseExtractor

Subclass `PhaseExtractor` and implement `extract()` and `quality_score()`:

```python
from scpn_phase_orchestrator.oscillators.base import PhaseExtractor, PhaseState

class MyExtractor(PhaseExtractor):
    def extract(self, signal, sample_rate):
        # signal: NDArray, sample_rate: float
        # Return: list[PhaseState]
        ...

    def quality_score(self, phase_states):
        # Return: float in [0, 1]
        ...
```

Register in the binding spec:

```yaml
oscillator_families:
  my_sensor:
    channel: P
    extractor_type: my_module.MyExtractor
    config:
      param1: value1
```

The loader resolves `extractor_type` by dotted import path when it is not one of the built-in types (`physical`, `informational`, `symbolic`).

## Custom GeometryConstraint

Implement a callable that constrains the Knm matrix:

```python
def my_constraint(knm, params):
    # Enforce sparsity, band structure, etc.
    # Return: modified knm (NDArray)
    ...
```

Register in the binding spec:

```yaml
geometry_prior:
  constraint_type: my_module.my_constraint
  params:
    bandwidth: 3
```

Called after `CouplingBuilder.build()` and after every template switch.

## Custom DriverSpec

The `drivers` section of the binding spec configures per-channel external drive parameters:

```yaml
drivers:
  physical:
    zeta: 0.5
    psi: 0.0
  informational:
    zeta: 0.0
  symbolic:
    zeta: 0.1
    psi: 3.14
```

Custom driver logic can override the default `zeta * sin(Psi - theta)` by providing a driver class:

```python
class MyDriver:
    def drive(self, phases, t):
        # Return: NDArray of drive contributions per oscillator
        ...
```

## Registration

All custom classes are resolved at binding-spec load time via Python's `importlib`. The module must be importable from the Python path. No global registry -- resolution is per-spec.
