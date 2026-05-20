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

For one-off local experiments, custom classes may still be resolved at
binding-spec load time via Python's `importlib`. The module must be importable
from the Python path.

For reusable extensions, publish a plugin manifest through the
`scpn_phase_orchestrator.plugins` Python entry-point group. The manifest
declares versioned capabilities before runtime code is imported:

```python
from scpn_phase_orchestrator.plugins import PluginCapability, PluginManifest

def spo_plugin_manifest():
    return PluginManifest(
        name="my_domain_pack",
        version="0.1.0",
        package="my_domain_pack",
        capabilities=(
            PluginCapability(
                kind="extractor",
                name="my_sensor",
                target="my_domain_pack.extractors:MyExtractor",
                channels=("P",),
            ),
        ),
    )
```

`validate_plugin_manifest()` and `compatibility_report()` provide the stable
CI gate for domainpack, extractor, actuator, and bridge extensions.

Runtime loading is available only through the explicit Python-owned loader. It
is disabled by default and must be enabled with a policy at the deployment
boundary that owns the risk decision:

```python
from scpn_phase_orchestrator.plugins import (
    PluginRuntimeLoadPolicy,
    load_plugin_capability,
)

loaded = load_plugin_capability(
    manifest,
    "extractor",
    "my_sensor",
    policy=PluginRuntimeLoadPolicy(loading_permitted=True),
)

extractor_cls = loaded.target_object
audit_record = loaded.audit_record
```

The loader validates manifest compatibility before import, resolves only the
declared capability, keeps targets inside the manifest package by default,
requires callable runtime targets, and records `scpn_plugin_runtime_load_v1`
audit metadata. Domainpack metadata entries remain manifest/catalogue records
rather than directly runtime-loadable callables unless an explicit future policy
expands that boundary.

## References

Phase extraction contracts are defined in [phase_contract.md](phase_contract.md). Binding spec validation uses [binding_spec.schema.json](binding_spec.schema.json). Custom geometry constraints must preserve the Knm invariants documented in [knm_semantics.md](knm_semantics.md).
