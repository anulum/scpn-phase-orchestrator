# Visualization API reference

`scpn_phase_orchestrator.visualization` is the public presentation facade for
SPO phase-state and coupling-state renderers.
It provides deterministic JSON encoders for D3-style network graphs,
coupling heatmaps, phase wheels, torus coordinate views, and an optional
outbound WebSocket streamer for live visual clients.

The visualization package is deliberately presentation-only.
It validates numeric payloads, converts them into renderer-friendly JSON, and
leaves solver state unchanged.
It does not infer physics.
It does not update coupling matrices.
It does not execute actuation.
It does not accept runtime commands from clients.

## Primary use cases

- Render a finite `K_nm` coupling matrix as a D3 force graph.
- Render the same finite `K_nm` matrix as a coupling heatmap.
- Render oscillator phases on a unit phase wheel.
- Render oscillator phases as deterministic Three.js torus coordinates.
- Broadcast already-computed telemetry to WebXR or Three.js clients.
- Provide review artefacts for notebooks, Studio panels, demos, and run reports.
- Verify that presentation output remains deterministic across repeated calls.

## Design boundary

The visualization subsystem is not a scientific estimator.
It is a serialization and presentation boundary.

The package accepts validated solver outputs and emits JSON strings or outbound
WebSocket messages.
It never becomes the owner of the solver state that produced those values.
A caller remains responsible for:

- choosing the solver or replay source,
- deciding whether the matrix or phase vector is scientifically meaningful,
- preserving layer labels outside the numeric arrays,
- deciding how a browser or dashboard renders the JSON,
- deciding whether a live streamer should be started,
- protecting any deployment network boundary outside this module.

The package is responsible for:

- rejecting malformed visualization inputs,
- preserving deterministic rounding semantics,
- preserving documented graph orientation,
- preserving finite JSON payload boundaries,
- preventing unsupported Python objects from entering outbound broadcasts,
- keeping stream broadcasts outbound and visualization-only.

## Exported symbols

| Symbol | Kind | Purpose |
| --- | --- | --- |
| `VisualizerStreamer` | class | Optional outbound WebSocket broadcaster for visualization clients. |
| `network_graph_json` | function | Convert a finite square coupling matrix into D3 node/link JSON. |
| `coupling_heatmap_json` | function | Convert a finite square coupling matrix into heatmap JSON. |
| `torus_points_json` | function | Convert a finite phase vector into Three.js torus coordinate JSON. |
| `phase_wheel_json` | function | Convert a finite phase vector into unit-circle phase-wheel JSON. |

The package facade re-exports all five symbols from
`scpn_phase_orchestrator.visualization`.
Callers should prefer the facade import unless they need module-local helpers
for implementation work.

## Module layout

| Module | Public surface | Contract |
| --- | --- | --- |
| `visualization.network` | `network_graph_json`, `coupling_heatmap_json` | Coupling matrix encoders for graph and heatmap renderers. |
| `visualization.torus` | `torus_points_json`, `phase_wheel_json` | Phase vector encoders for torus and phase-wheel renderers. |
| `visualization.streamer` | `VisualizerStreamer` | Optional WebSocket broadcast adapter for outbound visual telemetry. |
| `visualization.__init__` | facade exports | Stable import surface for application code and docs examples. |

## Numeric validation policy

The JSON encoders reject values that would make visualization output ambiguous
or unsafe to serialize.

Coupling matrices must be:

- numeric,
- two-dimensional,
- square,
- finite,
- non-boolean,
- non-complex.

Phase vectors must be:

- numeric,
- one-dimensional,
- finite,
- non-boolean,
- non-complex.

Metric vectors such as `R_values` must be:

- numeric,
- one-dimensional,
- finite,
- non-boolean,
- non-complex,
- length-matched to the node or phase count.

Layer names must be:

- supplied as a list,
- length-matched to the node or phase count,
- non-empty strings after whitespace checks.

Radius and threshold parameters reject boolean aliases even though Python treats
`bool` as an integer subclass.
This keeps UI knobs from silently accepting `True` or `False` as physical
numeric values.

## Deterministic JSON policy

The encoders return JSON strings, not Python dictionaries.
This makes the public functions directly usable by browser-facing panels,
notebooks, and CLI-style report writers.

The functions use deterministic structural fields and fixed rounding to four
decimal places for presentation values.
Identical inputs produce identical payload strings for the tested surfaces.

The JSON strings are presentation records.
They are not replay state.
If a caller needs replay-grade evidence, persist the solver state or audit
record that produced the visualization in addition to the visualization JSON.

## network_graph_json

`network_graph_json(knm, layer_names=None, R_values=None, threshold=0.01)`
creates a D3 force-graph payload from a finite square coupling matrix.

Input contracts:

- `knm` is a finite square numeric matrix.
- `layer_names`, when supplied, is a list of non-empty strings with one name per
  node.
- `R_values`, when supplied, is a finite numeric vector with one value per node.
- `threshold` is finite and non-negative.

Output contract:

- the JSON object has `nodes` and `links`,
- every node has `id`, `name`, and `R`,
- every link has `source`, `target`, and `weight`,
- default names are `L0`, `L1`, and so on,
- default `R` values are zero,
- values are rounded for stable presentation.

### Graph edge rule

`network_graph_json` iterates over upper-triangular matrix pairs.
For each pair `(i, j)` with `j > i`, it reads `knm[i, j]`.
A link is emitted only when `abs(knm[i, j]) > threshold`.
The threshold is exclusive.
A weight exactly equal to the threshold is omitted.

This graph is a compact undirected presentation view over the upper triangle.
It is not a full directed coupling audit.
Use domain-specific coupling reports when every directed edge must be preserved.

### Graph example

```python
import json
import numpy as np

from scpn_phase_orchestrator.visualization import network_graph_json

knm = np.array(
    [
        [0.0, 0.2, 0.0],
        [0.2, 0.0, 0.5],
        [0.0, 0.5, 0.0],
    ],
    dtype=np.float64,
)

data = json.loads(
    network_graph_json(
        knm,
        layer_names=["P", "I", "S"],
        R_values=[0.91, 0.72, 0.64],
        threshold=0.01,
    )
)

assert len(data["nodes"]) == 3
assert len(data["links"]) == 2
```

## coupling_heatmap_json

`coupling_heatmap_json(knm, layer_names=None)` creates a heatmap payload from a
finite square coupling matrix.

Input contracts:

- `knm` is a finite square numeric matrix,
- `layer_names`, when supplied, is a list of non-empty strings with one label
  per row and column.

Output contract:

- the JSON object has `labels`, `matrix`, `min`, and `max`,
- default labels are `L0`, `L1`, and so on,
- every matrix entry is rounded to four decimal places,
- `min` and `max` summarize the displayed matrix values.

A zero-size square matrix is accepted by the shared matrix validator but cannot
produce `min` or `max`; NumPy raises a zero-size reduction error.
The dedicated tests document that boundary.

### Heatmap example

```python
import json
import numpy as np

from scpn_phase_orchestrator.visualization import coupling_heatmap_json

knm = np.array([[0.0, 1.0], [0.5, 0.0]], dtype=np.float64)
heatmap = json.loads(coupling_heatmap_json(knm, layer_names=["Driver", "Target"]))

assert heatmap["labels"] == ["Driver", "Target"]
assert heatmap["min"] == 0.0
assert heatmap["max"] == 1.0
```

## torus_points_json

`torus_points_json(phases, R_values=None, major_radius=2.0, minor_radius=0.5)`
maps a finite one-dimensional phase vector into deterministic 3D torus
coordinates.

Input contracts:

- `phases` is a finite one-dimensional numeric vector,
- `R_values`, when supplied, is a finite numeric vector with one value per
  phase,
- `major_radius` is finite and positive,
- `minor_radius` is finite and positive.

Output contract:

- the JSON object has `points`,
- every point has `x`, `y`, `z`, `phase`, `R`, and `id`,
- phases are preserved in radians after rounding,
- `R` carries the supplied metric value or defaults to one,
- coordinates are rounded to four decimal places.

### Torus coordinate rule

For oscillator index `i`, the function sets a minor-circle angle
`phi_i = 2*pi*i/N`.
It then maps phase `theta_i` to:

```text
x = (major_radius + minor_radius * cos(phi_i)) * cos(theta_i)
y = (major_radius + minor_radius * cos(phi_i)) * sin(theta_i)
z = minor_radius * sin(phi_i)
```

This is a visual embedding.
It does not change the physics state.
It does not imply that the oscillator dynamics live on a physical torus unless
the caller's model gives that interpretation.

### Torus example

```python
import json
import numpy as np

from scpn_phase_orchestrator.visualization import torus_points_json

phases = np.array([0.0, np.pi / 2, np.pi], dtype=np.float64)
points = json.loads(torus_points_json(phases, R_values=[1.0, 0.8, 0.6]))

assert len(points["points"]) == 3
assert points["points"][0]["id"] == 0
```

## phase_wheel_json

`phase_wheel_json(phases, layer_names=None)` maps a finite one-dimensional phase
vector to unit-circle coordinates.

Input contracts:

- `phases` is a finite one-dimensional numeric vector,
- `layer_names`, when supplied, is a list of non-empty strings with one name per
  phase.

Output contract:

- the JSON object has `oscillators`,
- every oscillator has `name`, `phase`, `x`, and `y`,
- `x = cos(phase)`,
- `y = sin(phase)`,
- default names are `L0`, `L1`, and so on,
- values are rounded to four decimal places.

### Phase-wheel example

```python
import json
import numpy as np

from scpn_phase_orchestrator.visualization import phase_wheel_json

phases = np.array([0.0, np.pi / 2], dtype=np.float64)
wheel = json.loads(phase_wheel_json(phases, layer_names=["A", "B"]))

assert wheel["oscillators"][0]["x"] == 1.0
assert wheel["oscillators"][1]["name"] == "B"
```

## VisualizerStreamer

`VisualizerStreamer(host="127.0.0.1", port=8765)` owns an optional WebSocket
server for outbound visualization broadcasts.

Constructor contracts:

- `host` must be a non-empty string,
- `host` must not contain control characters,
- `port` must be an integer in `[1, 65535]`,
- boolean aliases are rejected for `port`.

Runtime contracts:

- `start()` starts a background thread when the optional `websockets` dependency
  is available,
- `start()` raises `RuntimeError` when the optional dependency is unavailable,
- repeated `start()` calls do not start duplicate live threads,
- `stop()` asks the owned event loop to stop when a loop exists,
- `_handler()` registers a client until it closes,
- `broadcast()` serializes JSON-safe data and schedules sends to current
  clients,
- send failures discard failed clients,
- invalid broadcast payloads are ignored rather than emitted.

## Streamer JSON-safe conversion

`VisualizerStreamer.broadcast()` accepts a dictionary and internally converts
payload values through `_json_safe`.

Supported payload values are:

- dictionaries,
- lists,
- tuples,
- NumPy arrays with finite floating values,
- NumPy scalar values,
- finite Python floats,
- strings,
- integers,
- booleans,
- `None`.

Unsupported payload values raise internally during conversion and cause the
broadcast to be skipped.
Non-finite float values are rejected.
Non-finite floating NumPy arrays are rejected.

The conversion is recursive.
Nested arrays become nested lists.
Tuples become lists because JSON has array values but no tuple type.

## Streamer safety model

The streamer is outbound-only from the perspective of SPO state.
Connected clients wait for broadcasts and are not used as a command channel.
The `_handler()` method waits for client closure; it does not parse client
messages into solver actions.

Deployment security still belongs outside this module.
If a deployment exposes the stream beyond localhost, the operator must provide
network isolation, authentication, TLS termination, and access policy at the
service boundary.
The visualization module only guarantees that its own public streamer surface
is a presentation adapter and not a control API.

## Pipeline integration

The intended pipeline is:

1. a solver or replay produces phases and coupling matrices,
2. monitors produce review metrics such as order parameters,
3. visualization encoders validate the finite presentation payload,
4. JSON output is handed to notebooks, Studio, dashboards, or browser views,
5. solver state remains owned by the runtime that produced it.

A dedicated test wires `UPDEEngine` output through `network_graph_json` and
`phase_wheel_json`.
That test guards the public presentation pipeline without turning visualization
into a solver module.

## Error semantics

The visualization functions use `ValueError` for invalid public input.
Typical invalid inputs are:

- boolean coupling matrices,
- complex coupling matrices,
- non-numeric coupling matrices,
- non-finite coupling matrices,
- non-square coupling matrices,
- boolean phase vectors,
- complex phase vectors,
- non-numeric phase vectors,
- non-finite phase vectors,
- multi-dimensional phase arrays,
- malformed `R_values`,
- mismatched `R_values` length,
- malformed layer names,
- mismatched layer-name length,
- invalid thresholds,
- invalid radii,
- invalid streamer host,
- invalid streamer port.

The streamer intentionally drops malformed broadcast payloads instead of sending
partial JSON to connected clients.
This keeps the visual stream from emitting invalid renderer state when a caller
accidentally includes non-serializable objects.

## Type-hint contract

Public array inputs use parameterized `NDArray[np.float64]` aliases.
The dedicated test surface checks that:

- `network_graph_json(knm=...)` uses the precise array alias,
- `coupling_heatmap_json(knm=...)` uses the precise array alias,
- `torus_points_json(phases=...)` uses the precise array alias,
- `phase_wheel_json(phases=...)` uses the precise array alias.

This supports the broader typed NumPy signature work without changing runtime
behaviour in this documentation slice.

## Polyglot and benchmark notes

This documentation update does not change production visualization behaviour,
Rust kernels, Go wrappers, Julia wrappers, Mojo wrappers, or benchmark outputs.
The visualization package is currently a Python presentation surface.
No polyglot counterpart is changed by this slice.
No benchmark update is required because no runtime algorithm changed.

If a future renderer bridge or polyglot encoder is added, it should preserve:

- finite numeric validation,
- boolean alias rejection,
- complex alias rejection,
- layer-name validation,
- deterministic rounding semantics,
- graph threshold exclusivity,
- torus coordinate equations,
- phase-wheel unit-circle equations,
- outbound-only streamer semantics.

## Behavioural tests guarding this reference

The dedicated module test surface is `tests/test_visualization.py`.
It checks:

- valid network graph JSON,
- node counts,
- link threshold exclusivity,
- invalid threshold rejection,
- custom layer names,
- layer-name length rejection,
- invalid layer-name rejection,
- whitespace layer-name rejection,
- empty graph behaviour,
- deterministic graph payloads,
- `R_values` presentation,
- invalid `R_values` rejection,
- `R_values` length rejection,
- invalid coupling matrix rejection,
- non-square coupling matrix rejection,
- heatmap labels and matrix shape,
- heatmap minimum and maximum values,
- zero-size heatmap reduction boundary,
- torus point fields,
- custom torus radii,
- invalid radius rejection,
- invalid phase rejection,
- multi-dimensional phase rejection,
- phase-wheel unit-circle coordinates,
- visualization pipeline wiring from `UPDEEngine`,
- public typed-array annotation contracts.

The reference depth guard is `tests/test_reference_api_visualization.py`.
It prevents this page from regressing to a shallow overview and checks that the
public visualization contracts remain documented.

## Operational checklist

- Confirm coupling matrices are finite before rendering.
- Confirm coupling matrices are square before rendering.
- Confirm coupling matrices do not use boolean aliases.
- Confirm coupling matrices do not use complex aliases.
- Confirm phase vectors are one-dimensional before rendering.
- Confirm phase vectors are finite before rendering.
- Confirm phase vectors do not use boolean aliases.
- Confirm phase vectors do not use complex aliases.
- Confirm R_values length matches node or phase count.
- Confirm R_values are finite presentation metrics.
- Confirm layer_names length matches node or phase count.
- Confirm layer_names are non-empty strings.
- Use network_graph_json for compact node-link views.
- Use coupling_heatmap_json when matrix values must remain visible.
- Use torus_points_json for 3D coordinate views.
- Use phase_wheel_json for unit-circle phase views.
- Use VisualizerStreamer only when a live visual client is required.
- Keep VisualizerStreamer outbound-only.
- Do not parse browser messages as solver commands through this module.
- Do not mutate solver state from visualization callbacks.
- Do not treat visualization JSON as replay state.
- Persist solver audit records separately from visualization payloads.
- Keep graph threshold values finite and non-negative.
- Remember graph threshold comparison is exclusive.
- Use heatmap output when directed matrix asymmetry matters visually.
- Do not use network graph output as a full directed-edge audit.
- Review torus coordinate interpretation with domain context.
- Do not claim physical torus dynamics from visual embedding alone.
- Review phase-wheel output near phase wraparound boundaries.
- Keep major_radius finite and positive.
- Keep minor_radius finite and positive.
- Reject malformed streamer hosts before startup.
- Reject malformed streamer ports before startup.
- Handle missing optional websockets dependency as a runtime deployment issue.
- Keep invalid broadcast payloads out of outbound client messages.
- Keep NumPy arrays finite before broadcast.
- Keep Python floats finite before broadcast.
- Keep unsupported objects out of broadcast payloads.
- Close failed send coroutines when submission fails.
- Discard failed clients after send errors.
- Avoid starting duplicate streamer threads.
- Use localhost defaults unless deployment isolation is provided.
- Provide deployment security outside this module.
- Keep browser rendering code separate from this Python encoder contract.
- Keep Studio panels aligned with these JSON field names.
- Keep notebooks aligned with facade imports.
- Keep CLI or report writers aligned with deterministic JSON strings.
- Keep tests module-specific for visualization behaviour.
- Avoid broad mixed-module visualization bucket tests.
- Avoid mock-only streamer tests that bypass public behaviour.
- Validate graph nodes rather than only checking that JSON exists.
- Validate graph links rather than only checking that JSON exists.
- Validate heatmap min and max when matrix ranges matter.
- Validate torus coordinates for known special cases.
- Validate phase wheel unit-circle coordinates for known phases.
- Validate deterministic payloads for identical inputs.
- Validate error paths for malformed layer names.
- Validate error paths for malformed R_values.
- Validate error paths for malformed radii.
- Validate error paths for malformed thresholds.
- Validate type hints when public array contracts change.
- Document new renderer fields before exposing them publicly.
- Document new streamer semantics before changing client behaviour.
- Document any future polyglot renderer compatibility rules.
- Document any future benchmark impact only after measurement.
- Keep this page updated when visualization exports change.
- Keep changelog entries factual when this page changes.
- Keep roadmap entries tied to documented evidence.
- Keep closure evidence module-specific.
- Review visualization output with representative domain data.
- Review visual labels for operator clarity.
- Review heatmap color scales outside this encoder.
- Review graph layout forces outside this encoder.
- Review WebXR camera controls outside this encoder.
- Review browser performance outside this encoder.
- Review streamer lifecycle in deployment runbooks.
- Review streamer shutdown behaviour in service wrappers.
- Review optional dependency installation in deployment docs.
- Review dashboard access policy outside this module.
- Review whether visualization payloads include only intended fields.
- Review whether visualization payloads omit private operator notes.
- Review whether external dashboards preserve numeric precision expectations.
- Review whether rounded values are sufficient for the intended view.
- Review whether raw numerical arrays should be stored separately.
- Review whether matrix directionality is represented correctly in the UI.
- Review whether graph compression hides important directed asymmetry.
- Review whether heatmap labels match binding layer names.
- Review whether phase-wheel labels match oscillator identities.
- Review whether torus point IDs match oscillator indices.
- Review whether R_values represent the intended coherence metric.
- Review whether missing R_values should default to zero or one by view.
- Review whether future functions need their own docs guard phrases.
- Review whether future functions need dedicated behaviour tests.
- Review whether future functions require typed-array annotations.
- Review whether future functions preserve JSON-safe output.
- Review whether future functions reject non-finite numeric data.
- Review whether future functions reject boolean aliases.
- Review whether future functions reject complex aliases.
- Review whether future functions preserve deterministic rounding.
- Review whether future streamer changes preserve outbound-only semantics.
- Review whether future streamer changes need security documentation.
- Review whether future streamer changes need deployment documentation.
- Review whether future streamer changes need lifecycle tests.
- Review whether future streamer changes need failure-path tests.
- Review whether future renderer changes affect existing dashboards.
- Review whether future renderer changes affect notebooks.
- Review whether future renderer changes affect Studio panels.
- Review whether future renderer changes affect public examples.
- Review whether future renderer changes affect API docs.
- Review whether future renderer changes affect roadmap status.
- Review whether future renderer changes affect release notes.
- Review whether future renderer changes affect benchmark notes.
- Review whether future renderer changes affect polyglot notes.
- Review whether future renderer changes affect pipeline wiring tests.
- Review whether future renderer changes affect typed NumPy signature tests.
- Review whether future renderer changes affect docs reference guards.
- Review whether future renderer changes affect examples in this page.
- Review whether future renderer changes affect operator checklists.
- Review whether future renderer changes affect presentation-only boundaries.
- Review whether future renderer changes remain non-actuating.
- Review whether future renderer changes remain non-inferential.
- Review whether future renderer changes preserve solver ownership outside visualization.
- Review whether future renderer changes preserve finite JSON boundaries.
- Review whether future renderer changes preserve deterministic field names.
- Review whether future renderer changes preserve public facade exports.
- Review whether future renderer changes preserve module-level public imports.
- Review whether future renderer changes preserve documented error semantics.
- Review whether future renderer changes preserve documented use cases.
- Review whether future renderer changes preserve documented safety boundaries.

## Release-note summary

The visualization API provides deterministic presentation encoders and an
optional outbound live-stream adapter for already-computed SPO state.
It validates finite numeric matrices, phase vectors, metrics, labels, radii, and
stream payloads before producing renderer-facing JSON or outbound messages.
The API remains presentation-only: it does not mutate solvers, infer coupling,
or create a control path.

::: scpn_phase_orchestrator.visualization.streamer

::: scpn_phase_orchestrator.visualization.network

::: scpn_phase_orchestrator.visualization.torus
