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
    PluginRuntimeExecutionPolicy,
    PluginRuntimeLoadPolicy,
    execute_plugin_capability,
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

Calling a loaded target requires a second opt-in policy:

```python
executed = execute_plugin_capability(
    manifest,
    "extractor",
    "my_sensor",
    args=(signal,),
    kwargs={"sample_rate": 100.0},
    policy=PluginRuntimeExecutionPolicy(
        loading_permitted=True,
        execution_permitted=True,
    ),
)
```

The execution audit record stores the load hash, target hash, argument count,
keyword names, result type, and deterministic execution hash. It deliberately
does not store argument values, because runtime payloads may contain proprietary
measurements or credentials.

For production deployments, bind execution to reviewed target hashes through a request stage:

```python
policy = PluginRuntimeExecutionPolicy(
    loading_permitted=True,
    execution_permitted=True,
    approved_target_hashes=(reviewed_target_hash,),
    require_target_hash_approval=True,
)
```

This approval check happens before the implementation module is imported. It is
the intended guard for promoting reviewed plugin metadata into a deployment
request stage, not an execution gate by itself.

Approval artefacts alone do not execute plugins.

`build_plugin_execution_request()` consumes both the reviewed plan and operator
approval artefacts together. It verifies `plan_hash`, `target_hash`, `plugin`,
`kind`, `name`, and operator binding before returning a request envelope.

The request envelope remains import-free and non-invoking; it sets policy such
that `loading_permitted` and `execution_permitted` are true only when
`require_target_hash_approval=True` and `approved_target_hashes` contains the
reviewed `target_hash`.

For traceability, keep operator ownership in a deployment artefact alongside
that approval check. The deployment record should include the reviewer/operator
identity or reference, the `plan_hash`, and the `target_hash`. Missing or altered
identity/hash pairing should keep execution in fail-closed mode.

## Review-only runtime execution planning

Runtime planning is a separate non-executing boundary.
`build_plugin_execution_plan()` builds deterministic metadata from manifest
compatibility, invocation shape, and target-hash policy without importing any
plugin module or invoking any target.

The planning primitive records immutable audit-relevant shape rather than runtime
state:

- required capability identity and kind
- compatible/incompatible status and rejection reasons
- target hashes precomputed from manifest and policy
- reviewed `plan_hash` for immutable operator sign-off
- positional argument count and keyword argument names for each candidate call
- policy flags that gate loading and execution

Reviewers can approve execution using policy outputs and reviewed target hashes
before any `load_plugin_capability`/`execute_plugin_capability` path is enabled.
If an approved hash is missing or invalid, the surface must fail closed.

Execution-path fail-closed conditions are enforced before import:

- execution disabled by policy
- missing capability declaration
- unsupported capability kind
- target outside plugin package when package boundary is required
- missing approval hash when `require_target_hash_approval=True`

No payload values are stored in the plan artifact; only argument count and keyword
names remain.

No argument values are part of this plan output. Argument payloads are only
handled at execution time and remain outside deterministic plan artifacts. The
operator-facing command is
`spo plugins plan-execution <plugin> <kind> <capability>` with optional
`--approved-target-hash` and `--require-target-hash-approval` flags.

The operator request artifact is produced with:

```bash
spo plugins request-execution PLAN_JSON APPROVAL_JSON
```

Runtime code consumes that request through `execute_plugin_execution_request()`.
The helper rebuilds the plan for the supplied argument shape, verifies the
request `plan_hash` and `target_hash`, and then invokes the declared capability
through the explicit runtime execution policy. Mismatched argument counts,
keyword names, manifests, or target hashes fail before plugin import.

Deployment storage must run `validate_plugin_execution_request()` before use.
That validator recomputes the request hash from the stored envelope and rejects
tampered audit records or request hashes present in a deployment-owned revocation
set. Approval rotation therefore happens by revoking old request hashes before a
new request envelope is consumed.

Storage custody is represented by
`build_plugin_execution_request_storage_manifest()`. The manifest records the
request hash, approval hash, target hash, storage URI, backend, retention
policy, manifest creator, revoked request hashes, and deterministic manifest
hash. `validate_plugin_execution_request_storage_manifest()` rejects stale or
tampered storage records before the request is handed to runtime code.

The local-file persistence adapter is
`write_plugin_execution_request_storage_bundle()`. It first builds and validates
`scpn_plugin_execution_request_storage_bundle_v1`, then writes the JSON bundle
atomically. Existing bundles are not overwritten unless `overwrite=True`, so a
deployment cannot silently replace an approved request without making that
rotation explicit.

Non-local deployment persistence is represented by
`build_plugin_execution_request_storage_adapter_manifest()`. It validates
backend/URI compatibility for `s3_object`, `gcs_object`, `azure_blob`,
`oci_object`, and `https_api`, rejects URI credentials, binds the storage
manifest hash plus bundle hash, and records
`adapter_mode=deployment_owned_external_write`. SPO emits this handoff manifest
only; deployment-specific writers must consume it and perform object-store or
HTTPS writes under their own credentials and audit controls.

The CLI surface for that adapter is:

```bash
spo plugins persist-execution-request REQUEST_JSON OUTPUT_JSON \
  --storage-uri file:///var/lib/spo/plugin-requests/request.json \
  --created-by deployment_gate
```

The CLI surface for external handoff manifests is:

```bash
spo plugins storage-adapter-manifest REQUEST_JSON \
  --storage-uri s3://spo-prod/plugin-requests/request.json \
  --storage-backend s3_object \
  --created-by deployment_gate
```

Revocation is represented by
`build_plugin_execution_request_revocation()` and exposed as:

```bash
spo plugins revoke-execution-request REQUEST_JSON \
  --revoked-by deployment_gate \
  --revocation-reference REV-2026-05-20-01 \
  --revocation-reason operator_rotation
```

The revocation artefact is immutable metadata. It does not delete stored bundles;
the deployment store must feed the emitted `request_hash` into its revoked-hash
set before persisting or consuming replacement requests.

For stores that maintain multiple lifecycle decisions,
`build_plugin_execution_request_revocation_list()` aggregates revocation
artefacts into a deterministic list, rejects duplicate request hashes, and
exposes `as_revoked_request_hashes()` for direct use with request validation.

```bash
spo plugins revocation-list REVOCATION_JSON --created-by deployment_gate
```

`spo plugins persist-execution-request` accepts `--revocation-list` to bind the
validated aggregate revoked-hash set into the storage manifest. If the request
being persisted is already present in that list, persistence fails before any
bundle is written.

## References

Phase extraction contracts are defined in [phase_contract.md](phase_contract.md). Binding spec validation uses [binding_spec.schema.json](binding_spec.schema.json). Custom geometry constraints must preserve the Knm invariants documented in [knm_semantics.md](knm_semantics.md).
