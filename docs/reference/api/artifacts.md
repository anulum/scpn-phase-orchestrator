# Artifacts

The artifacts API owns deterministic review payloads that cross the boundary
between SCPN Phase Orchestrator and downstream execution or review systems. The
current public surface is the QPU data artifact used to hand a validated
oscillator network to SCPN Quantum Control without enabling live QPU execution.

This page is intentionally explicit because artifact files can outlive the
Python process that produced them. A reviewer must be able to inspect the JSON,
recompute its digests, understand its provenance, and decide whether it is safe
for publication, replay, or downstream compilation.

::: scpn_phase_orchestrator.artifacts

## Public surface

| Symbol | Kind | Purpose |
|---|---|---|
| `QPUDataArtifact` | dataclass | Canonical validated oscillator artifact object. |
| `emit_qpu_data_artifact` | function | Build a validated JSON-compatible payload from arrays. |
| `compile_domain_to_qpu_artifact` | function | Compile a domainpack or binding spec into a QPU artifact. |
| `validate_qpu_data_artifact` | function | Revalidate a payload or dataclass and optionally enforce publication safety. |
| `read_qpu_data_artifact` | function | Load a JSON artifact file and validate it. |
| `write_qpu_data_artifact` | function | Write canonical JSON plus trailing newline. |
| `SCHEMA_VERSION` | constant | Current schema identifier. |
| `REAL_SOURCE_MODES` | constant | Source modes allowed for real or curated evidence. |
| `SYNTHETIC_SOURCE_MODES` | constant | Source modes treated as synthetic or replay-only evidence. |
| `ALL_SOURCE_MODES` | constant | Complete accepted source-mode set. |

## Design contract

The artifact module is not a simulator, not a QPU runner, and not a persistence
service. Its responsibility is narrower: convert an already-reviewed oscillator
network into a deterministic data envelope that another system can validate.

The boundary has five non-negotiable contracts:

1. Numeric arrays must be finite real `float64` arrays after coercion.
2. The oscillator network must contain at least one oscillator.
3. The coupling graph must be a symmetric non-negative matrix with a zero diagonal.
4. The manifest must carry provenance sufficient for review or replay.
5. The JSON digest must change whenever the payload changes.

The module therefore rejects malformed inputs early and returns plain Python
mappings that serialise through the standard JSON encoder with `allow_nan=False`.
No optional backend import is required for basic validation.

## Schema version

The current schema string is:

```text
scpn-quantum-control.qpu-data-artifact.v1
```

The schema name encodes the consumer family and the artifact type. It does not
claim that hardware execution is enabled. It only states that the payload shape
matches the QPU data artifact contract used by SCPN Quantum Control review and
compilation tooling.

Schema version handling is strict:

| Condition | Result |
|---|---|
| Exact current schema | Accepted after all other checks pass. |
| Older schema string | Rejected as unsupported. |
| Missing schema string | Rejected as missing a required field. |
| Non-string schema value | Coerced to string only through the dataclass path, then compared. |
| Unknown future schema | Rejected until the producer and consumer are deliberately upgraded. |

Strict schema comparison prevents accidental cross-loading of incompatible
payloads. It also avoids silent drift when partner tooling changes its JSON
shape.

## Required fields

A valid payload contains every field below.

| Field | Type after validation | Required meaning |
|---|---|---|
| `schema_version` | `str` | Exact current schema identifier. |
| `domain` | `str` | Non-empty domain or domainpack name. |
| `source_name` | `str` | Non-empty source label selected by the caller. |
| `source_mode` | `str` | One accepted source mode. |
| `K_nm` | `list[list[float]]` | Symmetric non-negative coupling matrix. |
| `omega` | `list[float]` | Natural-frequency vector. |
| `theta0` | `list[float]` or `null` | Optional initial phase vector. |
| `layer_assignments` | `list[str]` | Optional layer name per oscillator. |
| `normalization` | `str` | Non-empty code-level normalisation description. |
| `extraction_method` | `str` | Non-empty extraction or compilation method label. |
| `source_timestamp` | `str` or `null` | Review timestamp for recorded sources. |
| `replay_id` | `str` or `null` | Replay identifier for reproducible datasets. |
| `metadata` | `dict` | JSON-compatible metadata object. |
| `hashes` | `dict[str, str]` | Array digests verified against the arrays. |
| `artifact_sha256` | `str` | Digest of the canonical payload excluding itself. |

`theta0` is explicitly present even when absent because consumers should not
infer whether initial phase was forgotten or intentionally unavailable.

## Source modes

Source mode is a review classification, not a physics parameter.

| Source mode | Set | Publication-safe by default? | Typical use |
|---|---|---:|---|
| `recorded` | `REAL_SOURCE_MODES` | Yes, with timestamp or replay id | Sensor-derived network. |
| `replay` | `REAL_SOURCE_MODES` | Yes, with replay id | Reproduced audit or replay corpus. |
| `curated` | `REAL_SOURCE_MODES` | Yes, with timestamp or replay id | Hand-reviewed domainpack fixture. |
| `derived` | `REAL_SOURCE_MODES` | Yes, with timestamp or replay id | Derived from accepted upstream record. |
| `synthetic` | `SYNTHETIC_SOURCE_MODES` | No | Synthetic development data. |
| `simulation` | `SYNTHETIC_SOURCE_MODES` | No | Simulator-only output. |
| `fixture` | `SYNTHETIC_SOURCE_MODES` | No | Test fixture or example-only payload. |

Publication safety is enforced by `validate_qpu_data_artifact` unless the caller
passes `require_publication_safe=False`.

## Publication-safe rule

A payload is publication-safe only when both conditions are true:

1. `source_mode` is not synthetic, simulation, or fixture.
2. Either `source_timestamp` or `replay_id` is present.

This rule prevents a convenient development payload from being mistaken for a
reviewable scientific or operational record. It also ensures that real data can
be traced back to a timestamped acquisition or replay corpus.

The publication-safe rule does not prove that a dataset is scientifically
adequate. It only enforces the minimum provenance gate at the artifact boundary.
Domain-level validation and downstream target-readiness checks remain separate.

## Numeric contract

All numeric arrays are coerced to `np.float64` and then checked for finite
values. Inputs containing `NaN`, positive infinity, or negative infinity are
rejected before hashes are computed.

| Array | Required rank | Shape relation |
|---|---:|---|
| `K_nm` | 2 | Non-empty square matrix `(N, N)` where `N >= 1`. |
| `omega` | 1 | Vector `(N,)`. |
| `theta0` | 1 when present | Same shape as `omega`. |

The artifact boundary does not accept object arrays, ragged arrays, complex
values, or non-finite floating values. The JSON reader also rejects non-finite
JSON constants through a custom `parse_constant` hook.

## Coupling matrix invariants

`K_nm` encodes the coupling graph supplied to the QPU review layer.

The accepted matrix must satisfy all invariants below:

| Invariant | Enforcement |
|---|---|
| Square shape | `K_nm.shape[0] == K_nm.shape[1]`. |
| Non-empty network | `K_nm.shape[0] >= 1`. |
| Zero diagonal | `np.allclose(np.diag(K_nm), 0.0, atol=1e-12)`. |
| Non-negative weights | No entry may be less than `-1e-12`. |
| Symmetry | `np.allclose(K_nm, K_nm.T, atol=1e-12)`. |
| Finite values | Every entry must be finite. |

The zero diagonal preserves the no-self-coupling convention used across the
Kuramoto and UPDE surfaces. Symmetry is required because the current QPU handoff
maps the matrix to Kuramoto-XY style terms where pair coupling is undirected.
Negative coupling is rejected at this boundary; lagged or signed interpretations
belong in metadata or in a future explicit schema.

## Frequency vector contract

`omega` stores natural frequencies for the same oscillator order as `K_nm`.
The vector must contain exactly one finite real value per oscillator.

The artifact boundary does not require `omega` values to be strictly positive.
That choice is intentional: rotating-frame, centred, or signed frequency
representations may be valid at the domain level. The contract here is shape,
finite real-valuedness, and stable ordering.

Downstream target compilers may impose stricter constraints when a target cannot
represent a signed or centred frequency term.

## Initial phase contract

`theta0` is optional. When provided, it must have the same length as `omega` and
must contain only finite real values.

`theta0` is not wrapped by the artifact boundary. The caller is responsible for
choosing whether phase coordinates are represented as raw radians, wrapped
radians, or another reviewed coordinate convention. The chosen convention should
be described through `normalization` and metadata.

If the initial phase is not known, `theta0` should be `null`, not an empty list.
An empty list would fail the shape contract for non-empty networks.

## Layer assignments

`layer_assignments` is optional, but when present its length must match the
oscillator count. Every item is converted to a string.

Layer labels are metadata for review, grouping, and downstream visualisation.
They do not alter `K_nm`, `omega`, or `theta0`. The deterministic compiler path
from a binding spec builds this list by repeating each layer name for the
oscillators assigned to that layer.

A consumer should treat list order as significant. The first layer label belongs
to row and column `0` of `K_nm`, index `0` of `omega`, and index `0` of `theta0`
when present.

## Metadata contract

`metadata` must be a mapping. The module copies the mapping into the dataclass
and includes it in the canonical payload.

When loading legacy or hand-authored JSON, `metadata: null` and `hashes: null`
are accepted and normalised to empty mappings before the canonical digest is
verified. Emitted payloads always contain concrete objects for both fields.

Metadata values must be JSON serialisable and finite. Non-finite numeric values
fail when the payload digest is computed because the encoder uses
`allow_nan=False`.

The domainpack compiler currently emits metadata with these keys:

| Metadata key | Meaning |
|---|---|
| `source_project` | Producer project identifier. |
| `binding_spec` | Binding-spec file name. |
| `binding_version` | Binding-spec version. |
| `safety_tier` | Domainpack safety tier. |
| `sample_period_s` | Domain sample period. |
| `control_period_s` | Domain control period. |
| `n_layers` | Number of binding layers. |
| `n_oscillators` | Total oscillator count. |
| `coupling` | Base-strength and decay-alpha summary. |

Metadata is included in `artifact_sha256`. Changing metadata changes the final
payload digest.

## Hash model

The module computes two levels of digest.

| Digest | Input | Purpose |
|---|---|---|
| `K_nm_sha256` | Contiguous `float64` bytes of `K_nm`. | Detect coupling matrix drift. |
| `omega_sha256` | Contiguous `float64` bytes of `omega`. | Detect frequency vector drift. |
| `theta0_sha256` | Contiguous `float64` bytes of `theta0`. | Detect initial phase drift. |
| `artifact_sha256` | Canonical JSON payload before adding `artifact_sha256`. | Detect manifest drift. |

Array digests are checked if the caller supplies pre-existing values. A mismatch
raises a precise error such as `K_nm_sha256 does not match artifact data`.

The final artifact digest is recomputed after loading. A mismatch raises
`artifact_sha256 does not match artifact payload`.

## Canonical JSON

`QPUDataArtifact.to_json()` serialises with:

```python
json.dumps(payload, allow_nan=False, indent=2, sort_keys=True)
```

The canonical digest uses compact sorted JSON separators to avoid differences
from indentation. This gives stable hashes while keeping the file output readable.

`write_qpu_data_artifact()` appends a trailing newline. This keeps the file
compatible with ordinary POSIX tooling and avoids noisy diffs when files are
concatenated or inspected from shells.

## `QPUDataArtifact`

`QPUDataArtifact` is an immutable dataclass. Validation runs in `__post_init__`.
After validation, the object stores normalised strings, `np.float64` arrays,
string layer assignments, copied metadata, and verified array hashes.

Constructor responsibilities:

| Field group | Constructor behaviour |
|---|---|
| Provenance strings | Trimmed and checked for non-empty values. |
| Source mode | Checked against `ALL_SOURCE_MODES`. |
| Numeric arrays | Coerced to finite `float64` arrays. |
| Coupling graph | Checked for square, diagonal, sign, and symmetry constraints. |
| Hashes | Copied and completed with verified array digests. |
| Metadata | Copied to isolate the artifact object from caller mutation. |

The dataclass does not write files, access networks, invoke QPU libraries, or
mutate the source domainpack. It is a pure validation and serialisation object.

## `n_oscillators`

`n_oscillators` returns `int(self.K_nm.shape[0])`.

Use this property when consumers need a stable count after validation. It is
safer than recomputing the count from metadata because metadata may contain a
summary from another producer while the validated array shape is authoritative.

## `is_synthetic`

`is_synthetic` returns true when `source_mode` belongs to
`SYNTHETIC_SOURCE_MODES`.

This property is a review convenience. It is not a substitute for calling
`require_publication_safe()` when producing publication or deployment artifacts.

## `require_publication_safe`

`require_publication_safe()` raises when a payload is synthetic or lacks both
`source_timestamp` and `replay_id`.

Call it at boundaries where an artifact may leave development-only workflows.
The default public validator already calls it, so most callers should use
`validate_qpu_data_artifact(payload)` rather than invoking the method directly.

## `to_dict`

`to_dict()` returns the canonical JSON-compatible mapping. The method always
recomputes `artifact_sha256` from the current validated object.

The returned mapping is safe to pass to `json.dumps(..., allow_nan=False)`. It
contains Python lists instead of NumPy arrays, a plain metadata mapping, a plain
hash mapping, and explicit `null` for absent `theta0`, `source_timestamp`, or
`replay_id`.

## `from_dict`

`from_dict()` reconstructs a validated artifact from a mapping.

Load-time checks include:

| Check | Error class |
|---|---|
| Input is not a mapping | `ValueError`. |
| Required field missing | `ValueError`. |
| Schema version unsupported | `ValueError`. |
| Metadata is not a mapping | `ValueError`. |
| Hashes are not a mapping | `ValueError`. |
| Array digest mismatch | `ValueError`. |
| Final digest mismatch | `ValueError`. |

Consumers should prefer `QPUDataArtifact.from_dict(payload)` over manual field
access when loading data from any untrusted or stale source.

## `from_json`

`from_json()` decodes JSON and then delegates to `from_dict()`.

The JSON decoder rejects non-finite constants through `parse_constant`. Malformed
JSON syntax raises `json.JSONDecodeError`; non-finite constants are converted to
`ValueError` with a finite JSON message.

This split is deliberate. Syntax errors are useful for file-format debugging,
while non-finite constants are a domain boundary violation.

## `emit_qpu_data_artifact`

`emit_qpu_data_artifact()` is the direct array-to-payload builder.

Use it when the caller already has reviewed arrays and provenance:

```python
from scpn_phase_orchestrator.artifacts.qpu_data import emit_qpu_data_artifact

payload = emit_qpu_data_artifact(
    domain="minimal_domain",
    source_name="domainpack:minimal_domain",
    source_mode="curated",
    K_nm=[[0.0, 0.4], [0.4, 0.0]],
    omega=[1.0, 1.2],
    theta0=[0.0, 0.5],
    layer_assignments=["lower", "upper"],
    normalization="reviewed radians and Hz-equivalent frequencies",
    extraction_method="manual-review",
    replay_id="domainpack:minimal_domain:0.1.0",
    metadata={"review": "unit example"},
)
```

The function returns a dictionary, not a dataclass. This is convenient for CLI
and review surfaces that immediately write or display JSON.

## `compile_domain_to_qpu_artifact`

`compile_domain_to_qpu_artifact()` is the domainpack path.

It accepts either:

- a domainpack directory containing `binding_spec.yaml`; or
- a direct path to a binding spec file.

The function loads the binding spec, counts oscillators from layers, builds a
coupling matrix using `CouplingBuilder().build(...)`, reads natural frequencies
from the binding spec, derives layer assignments, attaches domain metadata, and
emits a validated QPU payload.

The compiler path uses this normalisation label:

```text
binding_spec CouplingBuilder.build exponential K_nm
```

The extraction method is:

```text
scpn_phase_orchestrator.binding_spec.v1
```

These labels are deliberately explicit so downstream reviewers can distinguish
a domainpack-derived artifact from a direct sensor-derived or manually curated
payload.

## Domainpack compile flow

The compile flow is deterministic for a fixed binding spec and fixed optional
`theta0`.

1. Resolve the binding path.
2. Load `BindingSpec` through `load_binding_spec`.
3. Count oscillators from layer assignments.
4. Build coupling through `CouplingBuilder`.
5. Read frequencies through `BindingSpec.get_omegas()`.
6. Expand layer names to one label per oscillator.
7. Build domain metadata.
8. Emit the QPU data artifact.
9. Optionally enforce publication safety.

No downstream QPU runtime is imported or executed by this function.

## `validate_qpu_data_artifact`

`validate_qpu_data_artifact()` accepts either a `QPUDataArtifact` instance or a
mapping. It returns a validated `QPUDataArtifact`.

The default call enforces publication safety:

```python
artifact = validate_qpu_data_artifact(payload)
```

For development-only fixtures, use an explicit opt-out:

```python
artifact = validate_qpu_data_artifact(
    payload,
    require_publication_safe=False,
)
```

The opt-out should stay close to the test or fixture that requires it. Do not
hide it in shared helper code, because it changes the safety posture of the
artifact boundary.

## `read_qpu_data_artifact`

`read_qpu_data_artifact(path)` reads UTF-8 JSON and validates it through
`QPUDataArtifact.from_json()`.

It rejects:

- malformed JSON syntax;
- non-mapping JSON payloads;
- non-finite JSON constants;
- missing required fields;
- unsupported schema versions;
- malformed arrays;
- digest mismatches.

The function returns a dataclass, not a raw dictionary, so consumers receive a
normalised object with NumPy arrays and verified hashes.

## `write_qpu_data_artifact`

`write_qpu_data_artifact(path, artifact)` writes validated canonical JSON with a
trailing newline.

The function expects a `QPUDataArtifact`, not an arbitrary mapping. Validate or
construct the artifact first, then write it. This prevents accidental file output
from bypassing the dataclass validation path.

## Error semantics

The module uses `ValueError` for domain-contract failures. JSON syntax failures
from `json.loads` remain `json.JSONDecodeError`.

Typical error messages are stable enough for operator diagnosis:

| Message fragment | Meaning |
|---|---|
| `domain must be non-empty` | Missing domain provenance. |
| `source_name must be non-empty` | Missing source label. |
| `source_mode must be one of` | Unknown source classification. |
| `K_nm must be square` | Coupling matrix rank or shape is invalid. |
| `artifact must contain at least one oscillator` | Empty oscillator network supplied. |
| `K_nm diagonal must be zero` | Self-coupling detected. |
| `K_nm must be non-negative` | Negative coupling weight detected. |
| `K_nm must be symmetric` | Directed coupling supplied to an undirected schema. |
| `omega shape must be` | Frequency count does not match oscillator count. |
| `theta0 shape must match omega shape` | Initial phase count mismatch. |
| `layer_assignments length must match` | Layer metadata count mismatch. |
| `artifact_sha256 does not match` | Payload was changed after digesting. |
| `synthetic artifacts are not publication-safe` | Synthetic source rejected by publication gate. |

Do not parse these messages as the primary machine interface. Use schema and
exception class at program boundaries; preserve messages for operator context.

## Handoff posture

The QPU artifact is a data handoff, not an execution permit.

A valid payload means:

- oscillator arrays are finite and shape-consistent;
- the coupling graph matches current Kuramoto-XY handoff assumptions;
- provenance fields are present at the artifact layer;
- hashes verify against payload content;
- the JSON can be reviewed and replayed by another process.

A valid payload does not mean:

- a QPU target has been selected;
- circuit compilation has been performed;
- simulator parity has been checked;
- hardware execution is authorised;
- scientific claims are proven by the artifact alone.

Target-readiness and execution approval belong to bridge, compiler, and operator
surfaces outside this module.

## Interoperability with SCPN Quantum Control

SCPN Phase Orchestrator owns the domain-to-oscillator compilation step. SCPN
Quantum Control owns quantum-control compilation and validation after receiving
the reviewed oscillator artifact.

The boundary is intentionally JSON and hash-based. This keeps the handoff
portable across Python versions, CLI tools, review notebooks, and future service
interfaces.

The consumer should re-run `validate_qpu_data_artifact` or equivalent schema
checks before using the payload. It should not trust a file merely because it has
the expected extension or file name.

## Security and robustness notes

Artifact files are ordinary JSON and may come from old workspaces or external
review systems. Treat them as untrusted until validated.

Robust loading rules:

- do not use raw `json.load` without validation;
- do not accept non-finite JSON constants;
- do not ignore digest mismatches;
- do not infer missing provenance;
- do not auto-enable downstream execution from a valid artifact;
- do not rewrite source mode to bypass publication-safety checks.

These rules keep offline review artifacts from becoming implicit actuation or
hardware-control triggers.

## Determinism notes

Determinism depends on three implementation choices:

1. Arrays are converted to contiguous `float64` before array hashes.
2. JSON used for the final digest is sorted and compact.
3. Metadata order does not affect the final digest because JSON keys are sorted.

If a future schema changes numeric precision or adds fields, the schema version
must change. Otherwise identical-looking artifacts could produce incompatible
backend semantics.

## Review checklist

Before accepting a QPU data artifact for downstream compilation, reviewers should
check:

- `schema_version` matches the expected version;
- `source_mode` is appropriate for the intended use;
- `source_timestamp` or `replay_id` is present when publication safety matters;
- `domain` and `source_name` identify the source unambiguously;
- `K_nm` encodes at least one oscillator;
- `K_nm` is square, symmetric, non-negative, and zero diagonal;
- `omega` length equals the oscillator count;
- `theta0` is absent or shape-compatible;
- `layer_assignments` length is absent or exact;
- `metadata` explains the source and coupling construction;
- `hashes` match the arrays;
- `artifact_sha256` matches the complete payload;
- no downstream execution flag is inferred from this data file.

## Minimal valid payload shape

A minimal publication-safe payload still needs real provenance:

```json
{
  "schema_version": "scpn-quantum-control.qpu-data-artifact.v1",
  "domain": "minimal_domain",
  "source_name": "domainpack:minimal_domain",
  "source_mode": "curated",
  "K_nm": [[0.0, 0.4], [0.4, 0.0]],
  "omega": [1.0, 1.2],
  "theta0": null,
  "layer_assignments": ["lower", "upper"],
  "normalization": "binding_spec CouplingBuilder.build exponential K_nm",
  "extraction_method": "scpn_phase_orchestrator.binding_spec.v1",
  "source_timestamp": null,
  "replay_id": "domainpack:minimal_domain:0.1.0",
  "metadata": {"source_project": "scpn-phase-orchestrator"},
  "hashes": {
    "K_nm_sha256": "computed by the emitter",
    "omega_sha256": "computed by the emitter"
  },
  "artifact_sha256": "computed by the emitter"
}
```

Do not write these placeholder digest strings in real files. Use the emitter so
hashes are computed from the actual arrays.

## Failure examples

A matrix with self-coupling is rejected:

```python
emit_qpu_data_artifact(
    domain="unit",
    source_name="unit",
    source_mode="curated",
    K_nm=[[1.0, 0.4], [0.4, 0.0]],
    omega=[1.0, 1.2],
    normalization="unit",
    extraction_method="unit",
    replay_id="unit:replay:1",
)
```

A synthetic payload is rejected by the default publication gate:

```python
payload = emit_qpu_data_artifact(
    domain="unit",
    source_name="unit",
    source_mode="synthetic",
    K_nm=[[0.0, 0.4], [0.4, 0.0]],
    omega=[1.0, 1.2],
    normalization="unit",
    extraction_method="unit",
    replay_id="unit:replay:1",
)
validate_qpu_data_artifact(payload)
```

A tampered file is rejected when `artifact_sha256` no longer matches the payload.
The correct fix is to regenerate the artifact from reviewed inputs, not to edit
the digest by hand.

## Testing evidence

The dedicated module test surface is `tests/test_qpu_data_artifact.py`.

It covers:

- curated domainpack compilation;
- malformed field rejection;
- empty oscillator-network rejection;
- publication-safety enforcement;
- stable and verified array hashes;
- final artifact digest verification;
- metadata normalisation;
- domainpack metadata shape;
- optional provenance handling;
- missing required fields;
- unsupported schema versions;
- non-mapping metadata and hashes;
- JSON roundtrip stability;
- file write/read behaviour;
- malformed JSON rejection;
- non-finite JSON constant rejection;
- non-finite metadata rejection;
- non-serialisable metadata rejection.

The API reference depth guard for this page is
`tests/test_reference_api_artifacts.py`.

## Operational boundaries

Use this API when you need a durable, reviewable data product. Do not use it as
a shortcut around domain validation or target readiness.

Good uses:

- exporting a reviewed domainpack oscillator network;
- attaching a replay identifier to a QPU compilation review;
- preserving hashes for audit trails;
- validating a received QPU data file before compilation;
- comparing artifacts across builds.

Bad uses:

- storing arbitrary runtime state;
- bypassing binding-spec validation;
- marking simulator data as recorded data;
- treating a valid JSON file as hardware approval;
- editing hashes manually to match changed content.

## Relationship to other APIs

| Related API | Relationship |
|---|---|
| Binding loader | Supplies validated domainpack specifications. |
| Coupling builder | Produces the domainpack-derived `K_nm` matrix. |
| Quantum control bridge | Imports and exports quantum phase artifacts at the adapter level. |
| Hybrid co-compiler | Builds review packages that may include quantum and neuromorphic components. |
| Runtime CLI | Emits JSON review artifacts for operator workflows. |

The artifacts module is deliberately small. It should stay focused on canonical
payload construction and validation rather than absorbing downstream compiler or
operator workflow logic.

## Extension rules

Future artifact schemas should follow these rules:

1. Add a new schema version when payload semantics change.
2. Keep finite JSON enforcement at the loader boundary.
3. Include digests for high-value numeric arrays.
4. Validate publication or deployment provenance explicitly.
5. Keep execution disabled unless a separate operator workflow enables it.
6. Add dedicated module tests for every new artifact family.
7. Update this reference page and any consumer-facing examples.

These rules preserve the core design: artifacts are portable evidence, not
implicit execution instructions.

## QPU data artifact API

::: scpn_phase_orchestrator.artifacts.qpu_data
